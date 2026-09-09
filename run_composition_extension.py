import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import trapezoid
import torch

from composition_benchmark import measure, run_cell, runtime, self_check
from composition_extension import ExtensionModel, certificate, verify_extension
from learned_sbi import dataset, exact_parameters, load_model, predict, verify_posterior


METHODS = [("full", "with_replacement"), ("unbiased", "with_replacement"), ("unbiased", "without_replacement"), ("tail", "with_replacement"), ("tail", "without_replacement")]


def matrix(kind):
    base = {"batch": 4, "particles": 8192, "u_max": 20.0, "ess_threshold": 0.5, "diffusion": 1.0}
    configs = []
    if kind == "oracle":
        choices = [("full", "with_replacement", 4)] + [(method, sampling, batch) for method, sampling in METHODS[1:] for batch in [2, 4, 8]]
        for family, dimension, steps, seed, (method, sampling, batch) in itertools.product(
            ["gaussian", "mixture", "weak_mixture"], [1, 8], [512, 2048], range(5), choices,
        ):
            configs.append(dict(base, kind=kind, family=family, groups=64, dimension=dimension, steps=steps, seed=seed, method=method, sampling=sampling, batch=batch))
        for family, seed, (method, sampling), (particles, u_max) in itertools.product(
            ["gaussian", "mixture", "weak_mixture"], range(5), METHODS, [(32768, 20), (131072, 20), (8192, 10), (8192, 30)],
        ):
            configs.append(dict(base, kind=kind, family=family, groups=64, dimension=8, steps=512, seed=seed, method=method, sampling=sampling, particles=particles, u_max=float(u_max)))
        assert len(configs) == 1080
    elif kind == "learned":
        for training_seed, dataset_seed, groups, dimension, steps, (method, sampling) in itertools.product(
            range(5), range(5), [16, 64], [1, 8], [512, 2048], METHODS,
        ):
            configs.append(dict(base, kind=kind, family="learned_mixture", groups=groups, dimension=dimension, steps=steps,
                                seed=100000 + 1000 * training_seed + dataset_seed, training_seed=training_seed, dataset_seed=dataset_seed,
                                method=method, sampling=sampling, particles=32768))
        assert len(configs) == 1000
    else:
        raise ValueError(kind)
    assert len({json.dumps(config, sort_keys=True) for config in configs}) == len(configs)
    return configs


def source_hashes():
    names = ["composition_benchmark.py", "composition_extension.py", "gaussian_integrability.py", "learned_sbi.py", "run_composition_extension.py", "run_extension_scnet.sh", "launch_extension.py", "docs/EXTENSION_PROTOCOL.md"]
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}


def run_extension_cell(config, output, training_root):
    parameters = None
    learned_model = None
    if config["kind"] == "learned":
        checkpoint = Path(training_root) / f"training_{config['training_seed']}" / "final.pt"
        config["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        setup_started = time.perf_counter()
        learned_model = load_model(checkpoint, config["device"])
        theta, context = dataset(config["groups"], config["dimension"], config["dataset_seed"])
        parameters = predict(learned_model, context)
        config["parameter_prediction_seconds"] = time.perf_counter() - setup_started
    model = ExtensionModel(config["groups"], config["dimension"], config["family"], config["device"], config["sampling"], parameters)
    population = certificate(model, config)
    summary = run_cell(config, output, model)
    output = Path(output)
    (output / "certificate.json").write_text(json.dumps(population, indent=2))
    if learned_model is not None:
        np.savez_compressed(output / "learned_parameters.npz", **parameters, theta=theta, context=context)
        true_model = ExtensionModel(config["groups"], config["dimension"], "true_mixture", "cpu", parameters=exact_parameters(context))
        true_reference = true_model.reference()
        np.savez_compressed(output / "true_reference.npz", **true_reference)
        with np.load(output / "samples.npz") as samples:
            true_metrics = measure(samples["samples"], samples["weights"], true_reference)
        with np.load(output / "reference.npz") as learned_reference:
            grid = true_reference["grid"]
            true_density = true_reference["density"]
            learned_density = learned_reference["density"]
            kl = trapezoid(true_density * (np.log(np.maximum(true_density, 1e-300)) - np.log(np.maximum(learned_density, 1e-300))), grid, axis=1)
            learned_error = {"composed_true_to_learned_kl_mean": float(kl.mean()), "composed_true_to_learned_w1_mean": float(trapezoid(abs(true_reference["cdf"] - learned_reference["cdf"]), grid, axis=1).mean())}
        (output / "true_metrics.json").write_text(json.dumps(dict(true_metrics, **learned_error), indent=2))
    (output / "extension_done").write_text("completed\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--kind", choices=["oracle", "learned"], default="oracle")
    parser.add_argument("--training-root")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)
    configs = matrix(args.kind)
    if args.manifest_only:
        manifest = {"commit": args.commit, "source_hashes": source_hashes(), "expected_cells": len(configs), "tasks": args.tasks, "concurrency": 40, "cells": configs}
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps({key: value for key, value in manifest.items() if key != "cells"}))
        return
    if args.check:
        self_check(root / "baseline_check.json")
        checks = {"extension": verify_extension(), "posterior": verify_posterior()}
        (root / "extension_check.json").write_text(json.dumps(checks, indent=2))
        print(json.dumps(checks))
        return
    if args.device == "cuda":
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    task_root = root / f"task_{args.task:03d}"
    task_root.mkdir(exist_ok=False)
    (task_root / "runtime.json").write_text(json.dumps(dict(runtime(), source_hashes=source_hashes()), indent=2))
    if args.smoke:
        selected = [(0, dict(configs[0], particles=1024, steps=128)), (1, dict(configs[1], particles=1024, steps=128, sampling="without_replacement"))]
    else:
        selected = [(i, config) for i, config in enumerate(configs) if i % args.tasks == args.task]
    started = time.perf_counter()
    for i, config in selected:
        config = dict(config, device=args.device, commit=args.commit, runtime=runtime(), cell_id=i)
        run_extension_cell(config, task_root / f"cell_{i:04d}", args.training_root)
    summary = {"status": "completed", "expected_cells": len(selected), "completed_cells": len(list(task_root.glob("cell_*/extension_done"))), "seconds": time.perf_counter() - started}
    assert summary["expected_cells"] == summary["completed_cells"]
    (task_root / "summary.json").write_text(json.dumps(summary, indent=2))
    (task_root / "done").write_text("completed\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
