import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

BOOTSTRAPS = 10000
METRICS = ("mse", "rmse", "coverage90", "coverage95", "width90", "width95")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def empirical_quantile(values, weights, probabilities):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.ndim != 1 or values.size != weights.size:
        raise ValueError("invalid empirical shapes")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("nonfinite or negative empirical input")
    probabilities = np.asarray(probabilities, dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("quantile probabilities must be in [0, 1]")
    keep = weights > 0
    if not keep.any() or weights[keep].sum() <= 0:
        raise ValueError("nonpositive total weight")
    values, weights = values[keep], weights[keep]
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cdf = np.cumsum(weights) / weights.sum()
    cdf[-1] = 1.0
    return values[np.searchsorted(cdf, probabilities, side="left")]


def reference_decisions(path):
    data = np.load(path, allow_pickle=False)
    grid = np.asarray(data["grid"], dtype=float)
    density = np.asarray(data["density"], dtype=float)
    cdf = np.asarray(data["cdf"], dtype=float)
    if density.shape != cdf.shape or density.shape[1] != grid.size:
        raise ValueError("reference shape mismatch")
    if not np.isfinite(grid).all() or not np.isfinite(density).all() or not np.isfinite(cdf).all():
        raise ValueError("nonfinite reference")
    if not (np.diff(grid) > 0).all() or (density < 0).any() or (np.diff(cdf, axis=1) < -1e-12).any():
        raise ValueError("invalid reference grid, density or CDF")
    if not np.allclose(trapezoid(density, grid, axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("reference density is not normalized")
    if not np.allclose(cdf[:, 0], 0, atol=1e-10, rtol=0) or not np.allclose(cdf[:, -1], 1, atol=1e-10, rtol=0):
        raise ValueError("reference CDF endpoints are invalid")
    mean = trapezoid(density * grid[None, :], grid, axis=1)
    interval90 = np.array([np.interp([0.05, 0.95], cdf[d], grid) for d in range(density.shape[0])])
    interval95 = np.array([np.interp([0.025, 0.975], cdf[d], grid) for d in range(density.shape[0])])
    return mean, interval90, interval95


def metric_row(method, theta, mean, interval90, interval95, metadata):
    theta, mean = np.asarray(theta, dtype=float), np.asarray(mean, dtype=float)
    interval90, interval95 = np.asarray(interval90, dtype=float), np.asarray(interval95, dtype=float)
    if theta.ndim != 1 or mean.shape != theta.shape or interval90.shape != (theta.size, 2) or interval95.shape != (theta.size, 2):
        raise ValueError("invalid metric shapes")
    if not np.isfinite(theta).all() or not np.isfinite(mean).all() or not np.isfinite(interval90).all() or not np.isfinite(interval95).all():
        raise ValueError("nonfinite metric input")
    if (interval90[:, 1] < interval90[:, 0]).any() or (interval95[:, 1] < interval95[:, 0]).any():
        raise ValueError("invalid interval bounds")
    error = mean - theta
    return {**metadata, "method": method, "mse": float(np.mean(error * error)),
            "rmse": float(np.sqrt(np.mean(error * error))),
            "coverage90": float(np.mean((theta >= interval90[:, 0]) & (theta <= interval90[:, 1]))),
            "coverage95": float(np.mean((theta >= interval95[:, 0]) & (theta <= interval95[:, 1]))),
            "width90": float(np.mean(interval90[:, 1] - interval90[:, 0])),
            "width95": float(np.mean(interval95[:, 1] - interval95[:, 0]))}


def sample_decisions(samples, weights):
    if samples.ndim != 2 or not np.isfinite(samples).all():
        raise ValueError("invalid samples")
    if weights.ndim != 1 or len(weights) != len(samples):
        raise ValueError("sample and weight dimensions differ")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("invalid weights")
    if not np.isclose(weights.sum(), 1.0, atol=2e-12, rtol=0):
        raise ValueError("saved weights do not sum to one")
    normalized = weights / weights.sum()
    mean = np.sum(samples * normalized[:, None], axis=0)
    interval90 = np.array([empirical_quantile(samples[:, d], weights, [0.05, 0.95]) for d in range(samples.shape[1])])
    interval95 = np.array([empirical_quantile(samples[:, d], weights, [0.025, 0.975]) for d in range(samples.shape[1])])
    return mean, interval90, interval95


def stable_seed(*parts):
    return 20260921 + sum((index + 1) * ord(char) for index, char in enumerate(":".join(parts)))


def self_checks():
    values = np.array([0.0, 1.0, 2.0])
    weights = np.array([0.25, 0.5, 0.25])
    assert np.isclose(np.sum(values * weights), 1.0)
    assert np.array_equal(empirical_quantile(values, weights, [0.0, 0.5, 1.0]), [0.0, 1.0, 2.0])
    assert np.array_equal(empirical_quantile(values, weights, [0.75]), [1.0])


def crossed_bootstrap(matrix, model_count, data_count, seed):
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAPS)
    for index in range(BOOTSTRAPS):
        models = rng.integers(0, model_count, model_count)
        datasets = rng.integers(0, data_count, data_count)
        values[index] = matrix[np.ix_(models, datasets)].mean()
    return {"mean": float(matrix.mean()), "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)), "replicates": BOOTSTRAPS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    self_checks()
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    run, phase_dir = args.run, args.run / args.phase
    root_state = json.loads((run / "state.json").read_text())
    manifest = json.loads((phase_dir / "manifest.json").read_text())
    if root_state.get("status") != "completed" or not manifest.get("cells"):
        raise RuntimeError("completed run and nonempty manifest are required")
    phase_done = phase_dir / "done"
    if not phase_done.is_file() or phase_done.read_text() != "completed\n":
        raise RuntimeError("phase completion marker is missing or invalid")
    cells = manifest["cells"]
    methods = sorted({cell["method"] for cell in cells})
    model_seeds = sorted({int(cell["training_seed"]) for cell in cells})
    data_seeds = sorted({int(cell["dataset_seed"]) for cell in cells})
    keys = [(int(cell["training_seed"]), int(cell["dataset_seed"]), cell["method"]) for cell in cells]
    expected_keys = {(model, dataset, method) for model in model_seeds for dataset in data_seeds for method in methods}
    if len(keys) != len(set(keys)) or set(keys) != expected_keys or any(cell.get("family") != "learned" for cell in cells):
        raise RuntimeError("manifest cells are not a rectangular unique learned cohort")
    if len(cells) != len(model_seeds) * len(data_seeds) * len(methods):
        raise RuntimeError("manifest rectangle count mismatch")
    rows, receipt_failures = [], []
    references, asset_cache = {}, {}
    for training_seed in model_seeds:
        for dataset_seed in data_seeds:
            asset = run / "assets" / f"learned_{training_seed}_{dataset_seed}"
            asset_receipt = json.loads((asset / "receipt.json").read_text())
            for name, expected in asset_receipt.get("files", {}).items():
                if sha256(asset / name) != expected:
                    receipt_failures.append(f"asset_{training_seed}_{dataset_seed}:{name}")
            data = np.load(asset / "data.npz", allow_pickle=False)
            theta = np.asarray(data["theta"], dtype=float)
            asset_cache[(training_seed, dataset_seed)] = (theta, reference_decisions(asset / "reference.npz"), reference_decisions(asset / "true_reference.npz"))
    for cell_meta in cells:
        cell = phase_dir / "cells" / f"cell_{int(cell_meta['cell_id']):04d}"
        receipt = json.loads((cell / "receipt.json").read_text())
        if receipt.get("status") != "completed" or (cell / "done").read_text() != "completed\n":
            raise RuntimeError(f"cell is not completed: {cell.name}")
        for name, expected in receipt.get("files", {}).items():
            if sha256(cell / name) != expected:
                receipt_failures.append(f"{cell.name}:{name}")
        training_seed, dataset_seed = int(cell_meta["training_seed"]), int(cell_meta["dataset_seed"])
        theta, learned_reference, true_reference = asset_cache[(training_seed, dataset_seed)]
        key = (training_seed, dataset_seed)
        if key in references:
            old = references[key]
            if not np.allclose(learned_reference[0], old[1][0], atol=1e-10, rtol=0):
                raise RuntimeError(f"learned reference mismatch: {key}")
            if not np.allclose(true_reference[0], old[2][0], atol=1e-10, rtol=0):
                raise RuntimeError(f"true reference mismatch: {key}")
        references[key] = (theta, learned_reference, true_reference)
        samples_file = np.load(cell / "samples.npz", allow_pickle=False)
        samples, weights = np.asarray(samples_file["samples"], float), np.asarray(samples_file["weights"], float)
        if samples.ndim != 2 or samples.shape[1] != theta.size:
            raise ValueError(f"sample dimension mismatch: {cell.name}")
        sample_mean, sample90, sample95 = sample_decisions(samples, weights)
        metadata = {"cell_id": int(cell_meta["cell_id"]), "training_seed": training_seed,
                    "dataset_seed": dataset_seed, "setting_id": cell_meta["setting_id"],
                    "particles": int(cell_meta["particles"])}
        rows.append(metric_row(cell_meta["method"], theta, sample_mean, sample90, sample95, metadata))
    if receipt_failures:
        raise RuntimeError("receipt failures: " + ", ".join(receipt_failures))
    for (training_seed, dataset_seed), (theta, learned_reference, true_reference) in sorted(references.items()):
        metadata = {"cell_id": -1, "training_seed": training_seed, "dataset_seed": dataset_seed, "setting_id": "reference", "particles": 0}
        rows.append(metric_row("learned_reference", theta, *learned_reference, metadata))
        rows.append(metric_row("true_reference", theta, *true_reference, metadata))
    for method in ("true_reference",):
        for dataset_seed in data_seeds:
            values = [row["mse"] for row in rows if row["method"] == method and row["dataset_seed"] == dataset_seed]
            if max(values) - min(values) > 1e-10:
                raise RuntimeError(f"oracle duplicate check failed: {method}, dataset {dataset_seed}")
    if len(rows) != len(cells) + 2 * len(model_seeds) * len(data_seeds):
        raise RuntimeError("cell CSV must contain all sampler cells and two reference rows per problem")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output / "cell_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"protocol_status": "exploratory/posthoc; does not alter predeclared W1/timegate or select methods",
               "run": str(run), "phase": args.phase, "manifest_cell_count": len(cells), "methods": methods,
               "training_seeds": model_seeds, "dataset_seeds": data_seeds, "sampler_cells": len(cells),
               "receipt_validation": "passed", "asset_receipt_validation": "passed", "manifest_sha256": sha256(phase_dir / "manifest.json"),
               "phase_done_sha256": sha256(phase_done), "metric_definition": "per-problem mean over saved theta coordinates; no coordinate is an independent dataset",
               "reference_labels": ["learned_reference", "true_reference"], "method_results": {},
               "analysis_source_sha256": sha256(__file__)}
    all_methods = methods + ["learned_reference", "true_reference"]
    for method in all_methods:
        selected = [row for row in rows if row["method"] == method]
        result = {"overall": {metric: float(np.mean([row[metric] for row in selected])) for metric in METRICS},
                  "per_training_seed": {}, "per_dataset_seed": {}, "crossed_bootstrap": {}}
        for seed in model_seeds:
            result["per_training_seed"][str(seed)] = {metric: float(np.mean([row[metric] for row in selected if row["training_seed"] == seed])) for metric in METRICS}
        for seed in data_seeds:
            result["per_dataset_seed"][str(seed)] = {metric: float(np.mean([row[metric] for row in selected if row["dataset_seed"] == seed])) for metric in METRICS}
        for metric in METRICS:
            matrix = np.array([[next(row[metric] for row in selected if row["training_seed"] == model and row["dataset_seed"] == dataset) for dataset in data_seeds] for model in model_seeds])
            result["crossed_bootstrap"][metric] = crossed_bootstrap(matrix, len(model_seeds), len(data_seeds), stable_seed("summary", metric))
        summary["method_results"][method] = result
    baseline = "full"
    summary["paired_vs_full"] = {}
    for method in methods:
        if method == baseline:
            continue
        summary["paired_vs_full"][method] = {}
        for metric in METRICS:
            method_matrix = np.array([[next(row[metric] for row in rows if row["method"] == method and row["training_seed"] == model and row["dataset_seed"] == dataset) for dataset in data_seeds] for model in model_seeds])
            baseline_matrix = np.array([[next(row[metric] for row in rows if row["method"] == baseline and row["training_seed"] == model and row["dataset_seed"] == dataset) for dataset in data_seeds] for model in model_seeds])
            summary["paired_vs_full"][method][metric] = {"mean_difference": float((method_matrix - baseline_matrix).mean()),
                "same_crossed_bootstrap": crossed_bootstrap(method_matrix - baseline_matrix, len(model_seeds), len(data_seeds), stable_seed("difference", metric))}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
