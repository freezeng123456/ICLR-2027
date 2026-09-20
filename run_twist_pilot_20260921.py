import argparse
import itertools
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from nonseparable_sensor_model_20260921 import SensorProblem
from nonseparable_sensor_sampler_20260921 import diffusion_sample, annealed_smc, timed_sample
from run_sensor_study_20260921 import asset_name, load_reference, metrics, prepare_one
from run_solid_20260920 import sha256, write_json
from twisted_sensor_sampler_20260921 import twisted_sensor_sample


def settings():
    return [dict(setting_id=f"twist_{ref}", kind="twist", reference=ref, method="full", particles=8192, steps=1024) for ref in ["mean", "anchor", "mirror"]] + [
        dict(setting_id="twist_mirror_fine", kind="twist", reference="mirror", method="full", particles=16384, steps=2048),
        dict(setting_id="twist_mirror_batch8", kind="twist", reference="mirror", method="tail_anchored", particles=8192, steps=1024, batch=8),
        dict(setting_id="full", kind="full", method="full", particles=16384, steps=2048),
        dict(setting_id="smc", kind="smc", method="annealed_smc", particles=16384, stages=64, proposal_scale=.15)]


def gpu_processes():
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True)
    return [line.strip() for line in output.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.root.mkdir(parents=True, exist_ok=False)
    if args.device == "cuda" and (not torch.cuda.is_available() or gpu_processes()):
        raise RuntimeError("CUDA must be available and initially idle")
    configurations = [dict(dimension=d, regime=r, dataset_seed=s, groups=12) for d, r, s in itertools.product([2, 8], ["ambiguous", "regular"], [1300, 1301])]
    for config in configurations:
        prepare_one((args.root, config))
    cells = [dict(c, **s, cell_id=i, phase="development", line="twist") for i, (c, s) in enumerate(itertools.product(configurations, settings()))]
    sources = {name: sha256(name) for name in ["run_twist_pilot_20260921.py", "twisted_sensor_sampler_20260921.py", "gaussian_twisted_path_20260921.py", "nonseparable_sensor_model_20260921.py", "nonseparable_sensor_sampler_20260921.py", "run_sensor_study_20260921.py", "run_anchored_confirmation_20260921.py", "run_solid_20260920.py", "docs/TWIST_PILOT_PROTOCOL_20260921.md"]}
    order = np.random.default_rng(20261002).permutation(len(cells))
    phase_root = args.root / "development"
    phase_root.mkdir()
    write_json(phase_root / "manifest.json", dict(cells=cells, order=order.tolist(), sources=sources, commit=args.source_commit, device=args.device))
    warm = SensorProblem.from_dict(json.loads((args.root / "assets" / asset_name(configurations[0]) / "problem.json").read_text()))
    for kind in ["full", "twist", "smc"]:
        if kind == "twist":
            twisted_sensor_sample(warm, [20., 19.99], 128, 17, device=args.device)
        elif kind == "full":
            diffusion_sample(warm, [20., 19.99], 128, 17, "full", device=args.device)
        else:
            annealed_smc(warm, 128, 17, stages=2, device=args.device)
    start, rows = time.perf_counter(), []
    for index in order:
        if time.perf_counter() - start > 1800:
            raise TimeoutError("pilot exceeded thirty-minute sampling budget")
        c = cells[index]
        before = gpu_processes() if args.device == "cuda" else []
        if args.device == "cuda" and len(before) != 1:
            raise RuntimeError("external GPU work detected; stopping this study")
        asset = args.root / "assets" / asset_name(c)
        problem = SensorProblem.from_dict(json.loads((asset / "problem.json").read_text()))
        reference = load_reference(asset)
        grid = np.linspace(np.sqrt(20.), 0., c.get("steps", 1) + 1) ** 2
        seed = 6100000 + 100 * c["dimension"] + c["dataset_seed"]
        if c["kind"] == "twist":
            result, seconds = timed_sample(twisted_sensor_sample, problem, grid, c["particles"], seed,
                method=c["method"], batch=c.get("batch", 4), reference=c["reference"], device=args.device)
            x, w, logz = result["samples"], result["weights"], result["log_normalizer"]
        elif c["kind"] == "full":
            (x, w, logz, _, _), seconds = timed_sample(diffusion_sample, problem, grid, c["particles"], seed, "full", device=args.device)
            result = {}
        else:
            (x, w, logz, _), seconds = timed_sample(annealed_smc, problem, c["particles"], seed,
                stages=c["stages"], proposal_scale=c["proposal_scale"], device=args.device)
            result = {}
        after = gpu_processes() if args.device == "cuda" else []
        output = phase_root / "cells" / f"cell_{c['cell_id']:04d}"
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "config.json", c)
        arrays = dict(samples=x, weights=w)
        if "log_weights" in result:
            arrays["log_weights"] = result["log_weights"]
        np.savez_compressed(output / "samples.npz", **arrays)
        row = dict(c, seed=seed, device=args.device, seconds=seconds, log_normalizer=logz, status="completed",
                   ess_fraction=float(1 / np.square(w).sum() / len(w)), **metrics(x, w, reference, problem))
        write_json(output / "summary.json", row)
        write_json(output / "gpu_processes.json", dict(before=before, after=after))
        write_json(output / "receipt.json", dict(files={p.name: sha256(p) for p in output.iterdir()}))
        (output / "done").write_text("completed\n")
        if args.device == "cuda" and len(after) != 1:
            raise RuntimeError("external GPU work detected; timing is invalid")
        rows.append(row)
        print(json.dumps({k: row[k] for k in ["cell_id", "setting_id", "sliced_w1_32", "ess_fraction", "seconds"]}), flush=True)
    means = {s["setting_id"]: {key: float(np.mean([r[key] for r in rows if r["setting_id"] == s["setting_id"]])) for key in ["sliced_w1_32", "seconds", "ess_fraction"]} for s in settings()}
    eligible = [s["setting_id"] for s in settings() if s["kind"] == "twist" and s["method"] == "full" and means[s["setting_id"]]["sliced_w1_32"] <= means["full"]["sliced_w1_32"] + .005 and means[s["setting_id"]]["seconds"] < .8 * means["full"]["seconds"]]
    write_json(args.root / "selection.json", dict(means=means, eligible=eligible, expand_confirmation=bool(eligible), evidence_status="development only; independent numerical audit required"))
    (phase_root / "done").write_text("completed\n")
    write_json(args.root / "state.json", dict(status="completed", seconds=time.perf_counter() - start, cells=len(rows)))


if __name__ == "__main__":
    main()
