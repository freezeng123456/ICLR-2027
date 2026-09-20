import argparse
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from certified_composition_torch import FactorParameters, sample
from composition_benchmark import measure, runtime
from composition_extension import ExtensionModel, certificate
from factorized_baseline_20260921 import time_factorized
from optimized_anchored_tail_20260921 import OptimizedAnchoredTailModel
from run_anchored_confirmation_20260921 import anchored_sample
from run_dual_20260921 import asset_name, joint_metrics, load_npz, paired_seed, prepare_asset
from run_solid_20260920 import sha256, write_json


def settings():
    rows = [dict(method=m, particles=32768, steps=2048, batch=4, setting_id=m)
            for m in ["full", "factorized_full"]]
    for n, k in [(8192, 1024), (32768, 2048)]:
        for batch in [4, 8, 16]:
            rows.append(dict(method="tail_anchored", particles=n, steps=k, batch=batch,
                             setting_id=f"anchor_n{n}_k{k}_m{batch}"))
    rows.append(dict(method="tail_anchored", particles=32768, steps=4096, batch=4,
                     setting_id="anchor_n32768_k4096_m4"))
    return rows


def problems(phase):
    seeds = (range(2), range(800, 804)) if phase == "development" else (range(5), range(900, 920))
    return [dict(family="learned", training_seed=t, dataset_seed=d, repeat=0)
            for t, d in itertools.product(*seeds)]


def select(rows):
    means = {}
    for setting in settings():
        subset = [r for r in rows if r["setting_id"] == setting["setting_id"]]
        if len(subset) != 8:
            raise ValueError("selection requires all eight development problems")
        means[setting["setting_id"]] = dict(w1=float(np.mean([r["w1_mean"] for r in subset])),
                                             seconds=float(np.mean([r["seconds"] for r in subset])))
    candidates = [s for s in settings() if s["method"] == "tail_anchored"]
    eligible = [s for s in candidates if means[s["setting_id"]]["w1"] <= means["full"]["w1"] + .002]
    chosen = min(eligible, key=lambda s: means[s["setting_id"]]["seconds"]) if eligible else min(
        candidates, key=lambda s: means[s["setting_id"]]["w1"])
    return dict(selected=chosen, development_eligible=bool(eligible), setting_means=means)


def run_cell(config, asset, output, device):
    output.mkdir(parents=True, exist_ok=False)
    params, reference = load_npz(asset / "parameters.npz"), load_npz(asset / "reference.npz")
    grid = np.linspace(math.sqrt(20), 0, config["steps"] + 1) ** 2
    cert_model = ExtensionModel(64, 8, "learned", "cpu", parameters=params)
    population = certificate(cert_model, dict(config, method="full"))
    if not population["finite_normalizer"]:
        raise RuntimeError("full/tail-control certificate failed")
    write_json(output / "config.json", dict(config, grid=grid.tolist()))
    torch.cuda.synchronize()
    start = time.perf_counter()
    if config["method"] == "tail_anchored":
        model = OptimizedAnchoredTailModel(64, 8, "learned", device, "without_replacement", params, grid[:-1])
        parameters = FactorParameters(**{key: torch.as_tensor(value, device=device) for key, value in params.items()})
        x, w, logz, records = anchored_sample(parameters, model, grid, config["particles"], config["seed"], config["batch"], .5, device)
        x, w = x.cpu().numpy(), w.cpu().numpy()
    elif config["method"] == "factorized_full":
        (x, w, logz, records), _ = time_factorized(params, grid, config["particles"], config["seed"], device)
    else:
        x, w, logz, records = sample(params, grid, config["particles"], config["seed"], method="full", device=device)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    if not np.isfinite(x).all() or not np.isfinite(w).all() or (w < 0).any():
        raise FloatingPointError("invalid sampled arrays")
    np.testing.assert_allclose(w.sum(), 1, atol=1e-10)
    np.savez_compressed(output / "samples.npz", samples=x, weights=w)
    report = dict(config, seconds=seconds, seconds_including_preparation=seconds, log_normalizer=float(logz),
                  **measure(x, w, reference), **joint_metrics(x, w, reference), status="completed")
    report.update({"true_" + key: value for key, value in measure(x, w, load_npz(asset / "true_reference.npz")).items()})
    write_json(output / "steps.json", records)
    write_json(output / "certificate.json", population)
    write_json(output / "summary.json", report)
    write_json(output / "receipt.json", dict(status="completed", files={p.name: sha256(p) for p in sorted(output.iterdir())}))
    (output / "done").write_text("completed\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--hours", type=float, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not 0 < args.hours <= 4:
        raise ValueError("CUDA and bounded runtime are required")
    torch.set_num_threads(1)
    args.root.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    write_json(args.root / "runtime.json", runtime())
    source_files = ["run_paper_anchor_20260921.py", "optimized_anchored_tail_20260921.py", "anchored_tail_20260921.py",
                    "run_anchored_confirmation_20260921.py", "certified_composition_torch.py", "factorized_baseline_20260921.py",
                    "composition_benchmark.py", "composition_extension.py", "gaussian_integrability.py", "learned_sbi.py",
                    "run_dual_20260921.py", "run_solid_20260920.py", "docs/PAPER_LEVEL_PROTOCOL_20260921.md"]
    source_hashes = {name: sha256(name) for name in source_files}
    models = {}
    selection = None
    for phase in ["development", "confirmation"]:
        phase_root = args.root / phase
        phase_root.mkdir()
        configs = settings()
        if phase == "confirmation":
            wanted = {"full", "factorized_full", "anchor_n32768_k2048_m4", selection["selected"]["setting_id"]}
            configs = [s for s in configs if s["setting_id"] in wanted]
        cells = [dict(problem, **config, cell_id=i, groups=64, dimension=8, u_max=20., diffusion=1.,
                      ess_fraction=.5, line="paper_anchor", phase=phase, seed=paired_seed(problem))
                 for i, (problem, config) in enumerate(itertools.product(problems(phase), configs))]
        order = np.random.default_rng(20260926 if phase == "development" else 20260927).permutation(len(cells)).tolist()
        write_json(phase_root / "manifest.json", dict(commit=args.commit, cells=cells, order=order, sources=source_hashes,
                   checkpoints={str(t): sha256(args.training_root / f"training_{t}" / "final.pt") for t in range(5)}))
        for problem in problems(phase):
            prepare_asset(args.root, problem, args.training_root, models)
        if phase == "development":
            params = load_npz(args.root / "assets" / asset_name(cells[0]) / "parameters.npz")
            sample(params, [20., 19.99], 128, 1, method="full", device="cuda")
            time_factorized(params, [20., 19.99], 128, 1, "cuda")
            warm_grid = np.array([20., 19.99])
            warm_model = OptimizedAnchoredTailModel(64, 8, "learned", "cuda", "without_replacement", params, warm_grid[:-1])
            warm_parameters = FactorParameters(**{key: torch.as_tensor(value, device="cuda") for key, value in params.items()})
            anchored_sample(warm_parameters, warm_model, warm_grid, 128, 1, 4, .5, "cuda")
        rows = []
        for index, cell_id in enumerate(order):
            if time.perf_counter() - start > args.hours * 3600:
                write_json(args.root / "state.json", dict(status="time_limit", phase=phase, completed=index, expected=len(cells)))
                raise SystemExit(2)
            config = cells[cell_id]
            report = run_cell(config, args.root / "assets" / asset_name(config), phase_root / "cells" / f"cell_{cell_id:04d}", "cuda")
            rows.append(report)
            write_json(args.root / "state.json", dict(status="running", phase=phase, completed=index + 1, expected=len(cells)))
            print(json.dumps({key: report[key] for key in ["phase", "cell_id", "setting_id", "w1_mean", "seconds"]}), flush=True)
        (phase_root / "done").write_text("completed\n")
        if phase == "development":
            selection = select(rows)
            write_json(args.root / "selection.json", selection)
    write_json(args.root / "state.json", dict(status="completed", seconds=time.perf_counter() - start))


if __name__ == "__main__":
    main()
