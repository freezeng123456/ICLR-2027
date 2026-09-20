import argparse
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from certified_composition_torch import sample
from composition_benchmark import measure, runtime
from composition_extension import ExtensionModel, certificate
from factorized_baseline_20260921 import time_factorized
from run_dual_20260921 import asset_name, joint_metrics, load_npz, paired_seed, prepare_asset
from run_solid_20260920 import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.root.mkdir(parents=True, exist_ok=False)
    phase = args.root / "development"
    phase.mkdir()
    problem_list = [dict(family=f, training_seed=-1, dataset_seed=-1, repeat=s)
                    for f, s in itertools.product(["gaussian", "mixture", "weak_mixture"], range(60, 65))]
    problem_list += [dict(family="learned", training_seed=0, dataset_seed=d, repeat=0) for d in range(500, 505)]
    configs = [dict(problem, method=method, setting_id=method, particles=8192, steps=512,
                    groups=64, dimension=8, diffusion=1.0, u_max=20.0, batch=4, ess_threshold=0.5,
                    line="factorization", phase="development", seed=paired_seed(problem))
               for problem, method in itertools.product(problem_list, ["full", "factorized_full"])]
    for i, config in enumerate(configs):
        config["cell_id"] = i
    source_files = ["run_factorization_diagnostic_20260921.py", "factorized_baseline_20260921.py", "certified_composition_torch.py",
                    "run_dual_20260921.py", "docs/FACTORIZATION_DIAGNOSTIC_20260921.md"]
    order = np.random.default_rng(20260922).permutation(len(configs)).tolist()
    manifest = dict(commit=args.commit, cells=configs, sources={name: sha256(name) for name in source_files}, order=order,
                    checkpoints={str(t): sha256(args.training_root / f"training_{t}" / "final.pt") for t in range(5)})
    write_json(phase / "manifest.json", manifest)
    models = {}
    for problem in problem_list:
        prepare_asset(args.root, problem, args.training_root, models)
    write_json(phase / "runtime.json", runtime())
    grid = np.linspace(math.sqrt(20), 0, 513) ** 2
    for cell_id in order:
        config = configs[cell_id]
        asset = args.root / "assets" / asset_name(config)
        params, reference = load_npz(asset / "parameters.npz"), load_npz(asset / "reference.npz")
        output = phase / "cells" / f"cell_{cell_id:04d}"
        output.mkdir(parents=True)
        write_json(output / "config.json", config)
        method = config["method"]
        if method == "full":
            sample(params, [20, 19.99], 128, 18, method="full", device=args.device)
            if args.device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            values = sample(params, grid, config["particles"], config["seed"], method="full", device=args.device)
            if args.device == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - started
        else:
            time_factorized(params, [20, 19.99], 128, 18, args.device)
            values, seconds = time_factorized(params, grid, config["particles"], config["seed"], args.device)
        samples, weights, logz, records = values
        report = dict(config, **measure(samples, weights, reference), **joint_metrics(samples, weights, reference),
                      seconds=seconds, seconds_including_preparation=seconds, log_normalizer=logz, status="completed",
                      resampling_count=sum(row.get("resampled", row.get("resampled_coordinates", 0)) for row in records))
        if (asset / "true_reference.npz").exists():
            report.update({"true_" + key: value for key, value in measure(samples, weights, load_npz(asset / "true_reference.npz")).items()})
        model = ExtensionModel(64, 8, config["family"], "cpu", parameters=params)
        population = certificate(model, dict(config, method="full"))
        assert population["finite_normalizer"]
        write_json(output / "certificate.json", population)
        write_json(output / "summary.json", report)
        write_json(output / "steps.json", records)
        np.savez_compressed(output / "samples.npz", samples=samples, weights=weights)
        write_json(output / "receipt.json", dict(status="completed", files={p.name: sha256(p) for p in sorted(output.iterdir())}))
        (output / "done").write_text("completed\n")
        write_json(args.root / "state.json", dict(status="running", completed=len(list((phase / "cells").glob("*/done"))), expected=40))
        print(json.dumps({key: report[key] for key in ["cell_id", "family", "method", "w1_mean", "seconds"]}), flush=True)
    assert len(list((phase / "cells").glob("*/done"))) == 40
    (phase / "done").write_text("completed\n")
    write_json(args.root / "state.json", dict(status="completed", completed=40, expected=40))


if __name__ == "__main__":
    main()
