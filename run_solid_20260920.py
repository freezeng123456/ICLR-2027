import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from composition_benchmark import measure, run_cell, runtime, self_check
from composition_extension import ExtensionModel, certificate
from gaussian_integrability import audit_path
from learned_sbi import dataset, exact_parameters, load_model, predict


FAMILIES = ["gaussian", "mixture", "weak_mixture"]
DELTAS = [-0.25, -0.1, -0.05, -0.025, -0.01, 0.0, 0.01, 0.025, 0.05, 0.1, 0.25]
SOURCE_FILES = ["run_solid_20260920.py", "composition_benchmark.py", "composition_extension.py",
                "gaussian_integrability.py", "learned_sbi.py", "docs/SOLID_PROTOCOL_20260920.md",
                "run_solid_20260920.sh", "tests/test_solid_20260920.py"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def critical_batch(family, steps):
    return ({512: 18, 2048: 17} if family == "weak_mixture" else {512: 60, 2048: 59})[steps]


def matrix():
    base = dict(groups=64, dimension=8, u_max=20.0, ess_threshold=0.5, diffusion=1.0, batch=4)
    cells = []
    for training_seed, dataset_seed in itertools.product(range(5), range(100, 120)):
        for method, sampling in [("full", "with_replacement"), ("unbiased", "without_replacement"), ("tail", "without_replacement")]:
            cells.append(dict(base, suite="heldout", family="learned_mixture", steps=2048, particles=32768,
                              training_seed=training_seed, dataset_seed=dataset_seed,
                              seed=9000000 + 1000 * training_seed + dataset_seed, method=method, sampling=sampling,
                              condition=method))
    for family, steps, seed in itertools.product(FAMILIES, [512, 2048], range(20, 30)):
        threshold = critical_batch(family, steps)
        conditions = [("full", "full", "with_replacement", 4),
                      ("below", "unbiased", "without_replacement", threshold - 1),
                      ("at", "unbiased", "without_replacement", threshold),
                      ("tail", "tail", "without_replacement", 4)]
        for condition, method, sampling, batch in conditions:
            cells.append(dict(base, suite="boundary", family=family, steps=steps, particles=8192, seed=seed,
                              method=method, sampling=sampling, batch=batch, condition=condition, critical_batch=threshold))
    for family, seed, delta in itertools.product(FAMILIES, range(20, 30), [-0.1, 0.0, 0.1]):
        cells.append(dict(base, suite="tail_error", family=family, steps=512, particles=8192, seed=seed,
                          method="cv", sampling="with_replacement", condition=f"delta_{delta:+.2f}", control_delta=delta))
    for cell_id, cell in enumerate(cells):
        cell["cell_id"] = cell_id
    assert len(cells) == 630
    assert len({json.dumps(cell, sort_keys=True) for cell in cells}) == len(cells)
    return cells


def controlled_certificate(model, config):
    variances = model.variance.cpu().numpy()
    controls = model.control_variance.cpu().numpy()
    rows = [audit_path(1 / variances[:, d] - 1, config["steps"], config["batch"], "cv",
                       diffusion=config["diffusion"], maximum_u=config["u_max"], control_variance=controls[:, d])
            for d in range(model.dimension)]
    assert all(row["finite_normalizer"] or row["minimum_denominator"] < -1e-10 for row in rows)
    return dict(finite_normalizer=all(row["finite_normalizer"] for row in rows), coordinates=rows,
                certificate_method="arbitrary_affine_control", scope="Gaussian exact; mixture strict-domain certificate")


def diagnostics(root):
    rows = []
    for family, steps in itertools.product(FAMILIES, [512, 2048]):
        model = ExtensionModel(64, 8, family, "cpu", "without_replacement")
        threshold = critical_batch(family, steps)
        for batch, expected in [(threshold - 1, False), (threshold, True)]:
            config = dict(steps=steps, batch=batch, diffusion=1.0, u_max=20.0, method="unbiased")
            result = certificate(model, config)
            assert result["finite_normalizer"] == expected, (family, steps, batch, result)
        for delta in DELTAS:
            model.control_variance = model.variance * (1 + delta)
            config = dict(steps=steps, batch=4, diffusion=1.0, u_max=20.0)
            result = controlled_certificate(model, config)
            rows.append(dict(family=family, steps=steps, delta=delta, **result))
    assert len(rows) == 66
    write_json(root / "tail_sensitivity.json", rows)
    write_json(root / "boundary_validation.json", {"status": "passed", "families": FAMILIES,
                                                 "steps": [512, 2048], "boundary_comparisons": 12})


class CachedModel(ExtensionModel):
    def __init__(self, *args, reference_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_path = reference_path

    def reference(self, points=65537, bound=12.0):
        if self.reference_path.exists():
            with np.load(self.reference_path) as archive:
                return {key: archive[key] for key in archive.files}
        reference = super().reference(points, bound)
        self.reference_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.reference_path, **reference)
        return reference


def run_one(config, output, root, training_root, models):
    started = time.perf_counter()
    parameters = None
    if config["suite"] == "heldout":
        training_seed = config["training_seed"]
        checkpoint = training_root / f"training_{training_seed}" / "final.pt"
        assert sha256(checkpoint) == config["checkpoint_sha256"]
        if training_seed not in models:
            models[training_seed] = load_model(checkpoint, config["device"])
        theta, context = dataset(config["groups"], config["dimension"], config["dataset_seed"])
        parameters = predict(models[training_seed], context)
        asset_name = f"learned_{training_seed}_{config['dataset_seed']}"
    else:
        asset_name = config["family"]
    asset_root = root / "assets" / asset_name
    model = CachedModel(config["groups"], config["dimension"], config["family"], config["device"],
                        config["sampling"], parameters, reference_path=asset_root / "reference.npz")
    if config["suite"] == "tail_error":
        model.control_variance = model.variance * (1 + config["control_delta"])
        population = controlled_certificate(model, config)
    else:
        population = certificate(model, config)
    summary = run_cell(config, output, model)
    write_json(output / "certificate.json", population)
    if parameters is not None:
        np.savez_compressed(output / "learned_parameters.npz", **parameters, theta=theta, context=context)
        true_model = CachedModel(config["groups"], config["dimension"], "true_mixture", "cpu",
                                 parameters=exact_parameters(context), reference_path=asset_root / "true_reference.npz")
        true_reference = true_model.reference()
        np.savez_compressed(output / "true_reference.npz", **true_reference)
        with np.load(output / "samples.npz") as archive:
            truth_metrics = measure(archive["samples"], archive["weights"], true_reference)
        write_json(output / "true_metrics.json", truth_metrics)
    summary["cell_wall_seconds"] = time.perf_counter() - started
    write_json(output / "summary.json", summary)
    required = ["config.json", "summary.json", "samples.npz", "reference.npz", "certificate.json", "metrics.csv", "run.log", "done"]
    if parameters is not None:
        required += ["learned_parameters.npz", "true_reference.npz", "true_metrics.json"]
    write_json(output / "receipt.json", {"status": "completed", "files": {name: sha256(output / name) for name in required}})
    (output / "solid_done").write_text("completed\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["prepare", "diagnostics", "smoke", "run"], required=True)
    parser.add_argument("--hours", type=float, default=6)
    args = parser.parse_args()
    torch.set_num_threads(1)
    hashes = {name: sha256(name) for name in SOURCE_FILES}
    if args.mode == "prepare":
        args.root.mkdir(parents=True, exist_ok=False)
        checkpoints = {str(seed): sha256(args.training_root / f"training_{seed}" / "final.pt") for seed in range(5)}
        for seed in range(5):
            original = json.loads((args.training_root / f"training_{seed}" / "summary.json").read_text())
            assert original["status"] == "completed" and original["updates"] == 40000
            assert original["final_sha256"] == checkpoints[str(seed)]
        cells = matrix()
        for cell in cells:
            if cell["suite"] == "heldout":
                cell["checkpoint_sha256"] = checkpoints[str(cell["training_seed"])]
        order = np.random.default_rng(20260920).permutation(len(cells)).tolist()
        write_json(args.root / "manifest.json", dict(commit=args.commit, sources=hashes, checkpoints=checkpoints,
                                                      expected_cells=len(cells), cells=cells, order=order))
        write_json(args.root / "state.json", dict(status="prepared", completed=0, expected=len(cells)))
        return
    manifest = json.loads((args.root / "manifest.json").read_text())
    assert manifest["commit"] == args.commit and manifest["sources"] == hashes
    if args.mode == "diagnostics":
        diagnostics(args.root)
        return
    assert args.device != "cuda" or (torch.cuda.is_available() and torch.cuda.device_count() == 1)
    write_json(args.root / f"runtime_{args.mode}.json", dict(runtime(), pid=os.getpid(), source_hashes=hashes))
    models = {}
    if args.mode == "smoke":
        self_check(args.root / "baseline_check.json")
        selected = [manifest["cells"][0], manifest["cells"][301], manifest["cells"][-1]]
        for i, original in enumerate(selected):
            config = dict(original, steps=128, particles=1024, device=args.device, commit=args.commit, runtime=runtime())
            run_one(config, args.root / "smoke" / f"cell_{i:03d}", args.root / "smoke", args.training_root, models)
        write_json(args.root / "smoke_receipt.json", dict(status="passed", cells=3))
        return
    assert json.loads((args.root / "smoke_receipt.json").read_text())["status"] == "passed"
    started = time.perf_counter()
    for cell_id in manifest["order"]:
        output = args.root / "cells" / f"cell_{cell_id:04d}"
        if (output / "solid_done").exists():
            receipt = json.loads((output / "receipt.json").read_text())
            assert receipt["status"] == "completed"
            assert all(sha256(output / name) == value for name, value in receipt["files"].items())
            continue
        if time.perf_counter() - started > args.hours * 3600:
            write_json(args.root / "state.json", dict(status="time_limit", next_cell=cell_id))
            raise SystemExit(2)
        config = dict(manifest["cells"][cell_id], device=args.device, commit=args.commit, runtime=runtime())
        write_json(args.root / "state.json", dict(status="running", current_cell=cell_id,
                                                  completed=len(list((args.root / "cells").glob("*/solid_done"))), expected=630))
        run_one(config, output, args.root, args.training_root, models)
    completed = len(list((args.root / "cells").glob("*/solid_done")))
    assert completed == manifest["expected_cells"]
    write_json(args.root / "state.json", dict(status="completed", completed=completed, expected=630,
                                              seconds=time.perf_counter() - started))


if __name__ == "__main__":
    main()
