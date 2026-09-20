import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import multiprocessing
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from nonseparable_sensor_model_20260921 import SensorProblem
from nonseparable_sensor_sampler_20260921 import annealed_smc, diffusion_sample, timed_sample
from run_sensor_study_20260921 import asset_name, load_reference, metrics, prepare_one
from run_solid_20260920 import sha256, write_json
from tail_bridge_smc_20260921 import sample


PARTICLES = 16384
SOURCE_FILES = ["run_tail_bridge_study_20260921.py", "tail_bridge_smc_20260921.py",
                "nonseparable_sensor_model_20260921.py", "nonseparable_sensor_sampler_20260921.py",
                "run_sensor_study_20260921.py", "run_solid_20260920.py", "run_anchored_confirmation_20260921.py",
                "docs/CONTINUED_ITERATION_PROTOCOL_20260921.md", "docs/TAIL_BRIDGE_STUDY_PROTOCOL_20260921.md"]


def development_settings():
    rows = []
    for reference, components in [("gaussian", 1), ("mixture", 2), ("mixture", 4)]:
        for covariance_scale, scale in itertools.product([1., 4.], [.5, 1.]):
            rows.append(dict(setting_id=f"bridge_{reference}{components}_cov{covariance_scale:g}_s{scale:g}",
                kind="bridge", reference=reference, components=components, covariance_scale=covariance_scale,
                proposal_scale=scale, ess_target=.8, moves=3, mode_steps=32, particles=PARTICLES, role="candidate"))
    for scale in [.15, .35, .65]:
        rows.append(dict(setting_id=f"adaptive_prior_s{scale:g}", kind="bridge", reference="prior", components=1,
            covariance_scale=1., proposal_scale=scale, ess_target=.8, moves=3, mode_steps=32,
            particles=PARTICLES, role="baseline"))
    for stages, scale in itertools.product([32, 64, 128], [.15, .35, .65]):
        rows.append(dict(setting_id=f"smc_k{stages}_s{scale:g}", kind="smc", stages=stages,
            proposal_scale=scale, moves=3, particles=PARTICLES, role="baseline"))
    rows += [dict(setting_id="full", kind="diffusion", method="full", particles=PARTICLES,
                  steps=2048, batch=4, role="diagnostic"),
             dict(setting_id="tail_anchored", kind="diffusion", method="tail_anchored", particles=PARTICLES,
                  steps=2048, batch=8, role="diagnostic")]
    return rows


def problem_configs(phase):
    seeds = range(1400, 1404) if phase == "development" else range(1500, 1520)
    return [dict(dimension=d, regime=r, dataset_seed=s, groups=12)
            for d, r, s in itertools.product([2, 8], ["ambiguous", "regular"], seeds)]


def gpu_processes():
    result = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True)
    return [line for line in result.splitlines() if line.strip()]


def execute(problem, setting, seed, device):
    if setting["kind"] == "bridge":
        x, w, logz, timing, counts, metadata = sample(problem, setting["particles"], seed,
            reference=setting["reference"], components=setting["components"], covariance_scale=setting["covariance_scale"],
            ess_target=setting["ess_target"], proposal_scale=setting["proposal_scale"], moves=setting["moves"],
            mode_steps=setting["mode_steps"], max_stages=256, device=device)
        return x, w, logz, dict(seconds=timing["total_seconds"], timing=timing, counts=counts, metadata=metadata)
    if setting["kind"] == "smc":
        (x, w, logz, records), seconds = timed_sample(annealed_smc, problem, setting["particles"], seed,
            stages=setting["stages"], proposal_scale=setting["proposal_scale"], moves=setting["moves"], device=device)
        calls = setting["particles"] * problem.groups + sum(record["factor_calls"] for record in records)
    else:
        grid = np.linspace(np.sqrt(20.), 0., setting["steps"] + 1) ** 2
        (x, w, logz, records, _), seconds = timed_sample(diffusion_sample, problem, grid, setting["particles"],
            seed, setting["method"], batch=setting["batch"], device=device)
        calls = sum(record["factor_calls"] for record in records)
    return x, w, logz, dict(seconds=seconds, timing=dict(total_seconds=seconds), counts=dict(factor_calls=calls),
                           metadata=dict(records=records, stages=len(records)))


def finish_cell(item):
    root, cell, seed, logz, diagnostics = item
    asset = root / "assets" / asset_name(cell)
    output = root / cell["phase"] / "cells" / f"cell_{cell['cell_id']:04d}"
    problem = SensorProblem.from_dict(json.loads((asset / "problem.json").read_text()))
    with np.load(output / "samples.npz") as saved:
        x, w = saved["samples"], saved["weights"]
    row = dict(cell, seed=seed, log_normalizer=logz, status="completed", seconds=diagnostics["seconds"],
               factor_calls=diagnostics["counts"]["factor_calls"], stages=diagnostics["metadata"]["stages"],
               ess_fraction=float(1 / np.square(w).sum() / len(w)), **metrics(x, w, load_reference(asset), problem))
    if cell["kind"] != "diffusion":
        row["log_evidence_error"] = logz - problem.log_evidence()
    write_json(output / "summary.json", row)
    write_json(output / "receipt.json", dict(files={path.name: sha256(path) for path in output.iterdir()}))
    (output / "done").write_text("completed\n")
    return row


def select(rows, configurations):
    lookup = {row["setting_id"]: row for row in configurations}
    expected = {(problem["dimension"], problem["regime"], problem["dataset_seed"], name)
                for problem in problem_configs("development") for name in lookup}
    observed = {(row["dimension"], row["regime"], row["dataset_seed"], row["setting_id"]) for row in rows}
    if len(lookup) != len(configurations) or observed != expected or len(rows) != len(expected):
        raise RuntimeError("selection requires the complete frozen development cohort")
    if any(not np.isfinite(row[key]) for row in rows for key in ["sliced_w1_32", "seconds", "factor_calls"]):
        raise FloatingPointError("selection cannot use nonfinite results")
    means = {name: {key: float(np.mean([row[key] for row in rows if row["setting_id"] == name]))
                    for key in ["sliced_w1_32", "seconds", "factor_calls"]} for name in lookup}
    baselines = [name for name, c in lookup.items() if c["role"] == "baseline"]
    best_error = min(means[name]["sliced_w1_32"] for name in baselines)
    accurate = [name for name in baselines if means[name]["sliced_w1_32"] <= best_error + .001]
    baseline = min(accurate, key=lambda name: (means[name]["seconds"], name))
    fixed = "smc_k64_s0.15"
    def eligible(name):
        for other in {baseline, fixed}:
            if means[name]["sliced_w1_32"] > means[other]["sliced_w1_32"] + .002:
                return False
            if means[name]["seconds"] >= .8 * means[other]["seconds"]:
                return False
            for dimension, regime in itertools.product([2, 8], ["ambiguous", "regular"]):
                group = [row for row in rows if (row["dimension"], row["regime"]) == (dimension, regime)]
                candidate = np.mean([row["sliced_w1_32"] for row in group if row["setting_id"] == name])
                comparison = np.mean([row["sliced_w1_32"] for row in group if row["setting_id"] == other])
                if candidate > comparison + .004:
                    return False
        return True
    candidates = [name for name, c in lookup.items() if c["role"] == "candidate" and eligible(name)]
    chosen = min(candidates, key=lambda name: (means[name]["seconds"], name)) if candidates else None
    return dict(status="development_completed", baseline=baseline, fixed_baseline=fixed, candidate=chosen,
                eligible=candidates, means=means, settings=configurations, expand_confirmation=bool(chosen),
                development_only=True, audit_status="independent audit required")


def confirmation_settings(selection):
    if not selection["expand_confirmation"]:
        raise RuntimeError("no candidate passed the frozen expansion gate")
    lookup = {row["setting_id"]: row for row in selection["settings"]}
    rows = [lookup[selection[key]].copy() for key in ["candidate", "baseline", "fixed_baseline"]]
    candidate = lookup[selection["candidate"]]
    for reference in ["prior", "gaussian"]:
        row = dict(candidate, reference=reference, components=1, role="ablation",
                   covariance_scale=1. if reference == "prior" else candidate["covariance_scale"],
                   setting_id=f"matched_reference_{reference}")
        rows.append(row)
    rows.extend([lookup["full"].copy(), lookup["tail_anchored"].copy()])
    unique = {}
    for row in rows:
        unique.setdefault(row["setting_id"], row)
    return list(unique.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--phase", choices=["development", "confirmation"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.device == "cuda" and (not torch.cuda.is_available() or gpu_processes()):
        raise RuntimeError("an initially idle CUDA device is required")
    phase_root = args.root / args.phase
    phase_root.mkdir(parents=True, exist_ok=False)
    configurations = development_settings() if args.phase == "development" else confirmation_settings(json.loads((args.root / "selection.json").read_text()))
    problems = problem_configs(args.phase)
    repetitions = range(1) if args.phase == "development" else range(2)
    cells = [dict(problem, **setting, phase=args.phase, repeat=repeat, cell_id=i)
             for i, (problem, setting, repeat) in enumerate(itertools.product(problems, configurations, repetitions))]
    order = np.random.default_rng(20261010 if args.phase == "development" else 20261011).permutation(len(cells))
    sources = {name: sha256(name) for name in SOURCE_FILES}
    write_json(phase_root / "manifest.json", dict(cells=cells, settings=configurations, order=order.tolist(),
        source_commit=args.source_commit, sources=sources, particles=PARTICLES, phase=args.phase,
        paired_sampler_repeats=len(repetitions), independent_data_seeds=len({c["dataset_seed"] for c in problems})))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        list(pool.map(prepare_one, [(args.root, problem) for problem in problems]))
    first = SensorProblem.from_dict(json.loads((args.root / "assets" / asset_name(problems[0]) / "problem.json").read_text()))
    warm = dict(configurations[0], particles=256)
    execute(first, warm, 191, args.device)
    annealed_smc(first, 256, 191, stages=2, device=args.device)
    diffusion_sample(first, [20., 19.99], 256, 191, "full", device=args.device)
    started, futures = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        for index in order:
            if time.perf_counter() - started > 7200:
                raise TimeoutError("phase exceeded two-hour wall-clock budget")
            cell = cells[index]
            before = gpu_processes() if args.device == "cuda" else []
            if args.device == "cuda" and len(before) != 1:
                raise RuntimeError("external GPU work invalidates isolated timing")
            asset = args.root / "assets" / asset_name(cell)
            problem = SensorProblem.from_dict(json.loads((asset / "problem.json").read_text()))
            seed = 8100000 + 1000 * cell["dimension"] + 10 * cell["dataset_seed"] + cell["repeat"]
            x, w, logz, diagnostics = execute(problem, cell, seed, args.device)
            after = gpu_processes() if args.device == "cuda" else []
            if args.device == "cuda" and len(after) != 1:
                raise RuntimeError("external GPU work invalidates isolated timing")
            output = phase_root / "cells" / f"cell_{cell['cell_id']:04d}"
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "config.json", cell)
            np.savez_compressed(output / "samples.npz", samples=x, weights=w)
            write_json(output / "diagnostics.json", diagnostics)
            write_json(output / "gpu_processes.json", dict(before=before, after=after))
            futures.append(pool.submit(finish_cell, (args.root, cell, seed, logz, diagnostics)))
            print(json.dumps(dict(cell_id=cell["cell_id"], setting_id=cell["setting_id"], seconds=diagnostics["seconds"],
                                  stages=diagnostics["metadata"]["stages"], sampled=True)), flush=True)
        rows = [future.result() for future in futures]
    if len(rows) != len(cells):
        raise RuntimeError("incomplete cohort")
    write_json(phase_root / "rows.json", rows)
    if args.phase == "development":
        write_json(args.root / "selection.json", select(rows, configurations))
    write_json(phase_root / "state.json", dict(status="completed", cells=len(cells), seconds=time.perf_counter() - started,
        sampler_seconds=sum(row["seconds"] for row in rows), workers=args.workers, device=args.device))
    (phase_root / "done").write_text("completed\n")


if __name__ == "__main__":
    main()
