import hashlib
import json
import math
import os
import time
import argparse
from pathlib import Path

import numpy as np
import torch

from anchored_tail_20260921 import AnchoredTailModel
from composition_extension import ExtensionModel
from learned_sbi import dataset, load_model, predict


TRAINING_ROOT = Path("/Users/zenghang/Documents/Codex/2026-09-09/https-github-com-freezeng123456-iclr-2027/work/extension-recovered/run/training")
OUTPUT_ROOT = Path("work/anchored-conditional")
GRID = np.array([4.0, 1.0, 0.2, 0.0], dtype=np.float64)
GROUPS = 64
DIMENSION = 8
REPLICATES = 4096
BATCH = 4
RNG_SEED = 717


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hashes():
    paths = [Path("docs/ANCHORED_CONDITIONAL_PROTOCOL_20260921.md"), Path("anchored_tail_20260921.py"), Path("composition_extension.py"), Path("composition_benchmark.py"), Path("learned_sbi.py")]
    return {str(path): sha256(path) for path in paths}


def load_strict(training_seed):
    root = TRAINING_ROOT / f"training_{training_seed}"
    checkpoint = root / "final.pt"
    done = root / "done"
    config = root / "config.json"
    summary = root / "summary.json"
    if not all(path.is_file() for path in [checkpoint, done, config, summary]):
        raise FileNotFoundError(root)
    if done.read_text() != "completed\n":
        raise RuntimeError(f"Training receipt is not completed: {done}")
    model = load_model(checkpoint, device="cpu")
    return model, {"root": str(root), "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "config_sha256": sha256(config), "summary_sha256": sha256(summary)}


def fixed_directions():
    rng = np.random.default_rng(RNG_SEED)
    directions = rng.choice(np.array([-1.0, 1.0]), size=(8, DIMENSION))
    return directions


def metrics(estimates, exact, theoretical_variance):
    estimates = np.asarray(estimates, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    centered = estimates - exact
    variance = estimates.var(axis=0, ddof=1)
    standard_error = np.sqrt(variance / estimates.shape[0])
    return {"mean": np.asarray(estimates.mean(axis=0)).tolist(), "variance": np.asarray(variance).tolist(), "standard_error": np.asarray(standard_error).tolist(), "error": np.asarray(estimates.mean(axis=0) - exact).tolist(), "theoretical_drift_variance": None if theoretical_variance is None else np.asarray(theoretical_variance).tolist()}


def run_pair(training_seed, dataset_seed, directions):
    model, receipt = load_strict(training_seed)
    theta, context = dataset(GROUPS, DIMENSION, dataset_seed)
    parameters = predict(model, context)
    identity = {"theta_sha256": json_hash(np.asarray(theta, dtype=np.float64).tolist()), "context_sha256": json_hash(np.asarray(context, dtype=np.float64).tolist()), "groups": GROUPS, "dimension": DIMENSION, "training_seed": training_seed, "dataset_seed": dataset_seed}
    original = ExtensionModel(GROUPS, DIMENSION, "learned", "cpu", "without_replacement", parameters)
    original.control_variance = original.variance
    anchored = AnchoredTailModel(GROUPS, DIMENSION, "learned", "cpu", "without_replacement", parameters, GRID)
    anchor_points = []
    for u in GRID:
        anchor_record = anchored.anchor_parameters(float(u))
        width = 1 / np.sqrt(1 - anchor_record["a"].sum(0))
        anchor = anchor_record["anchor"]
        anchor_points.append(np.vstack((anchor[None, :], anchor[None, :] + directions * width[None, :])))
    points = np.stack(anchor_points)
    raw_original_drift = np.empty((len(GRID), len(points[0]), REPLICATES, DIMENSION), dtype=np.float64)
    raw_anchored_drift = np.empty_like(raw_original_drift)
    raw_original_potential = np.empty((len(GRID), len(points[0]), REPLICATES), dtype=np.float64)
    raw_anchored_potential = np.empty_like(raw_original_potential)
    exact_drift = np.empty((len(GRID), len(points[0]), DIMENSION), dtype=np.float64)
    exact_potential = np.empty((len(GRID), len(points[0])), dtype=np.float64)
    theoretical_original = np.empty((len(GRID), len(points[0]), DIMENSION), dtype=np.float64)
    theoretical_anchored = np.empty_like(theoretical_original)
    rng = np.random.default_rng(RNG_SEED + 10000 * training_seed + dataset_seed)
    start = time.perf_counter()
    for time_index, u in enumerate(GRID):
        for point_index, point in enumerate(points[time_index]):
            x = torch.as_tensor(point[None, :], dtype=torch.float64)
            exact_drift_t, exact_potential_t = original.exact(x, float(u))
            exact_drift[time_index, point_index] = exact_drift_t[0].numpy()
            exact_potential[time_index, point_index] = exact_potential_t[0].item()
            keys = rng.random((REPLICATES, GROUPS))
            indices = torch.as_tensor(np.argpartition(keys, BATCH - 1, axis=1)[:, :BATCH], dtype=torch.long)
            repeated = x.repeat(REPLICATES, 1)
            original_drift, original_potential = original.estimate(repeated, float(u), BATCH, torch.Generator().manual_seed(0), control=True, indices=indices)
            anchored_drift, anchored_potential = anchored.estimate(repeated, float(u), BATCH, torch.Generator().manual_seed(0), control=True, indices=indices)
            raw_original_drift[time_index, point_index] = original_drift.numpy()
            raw_anchored_drift[time_index, point_index] = anchored_drift.numpy()
            raw_original_potential[time_index, point_index] = original_potential.numpy()
            raw_anchored_potential[time_index, point_index] = anchored_potential.numpy()
            for target, destination in [(original, theoretical_original), (anchored, theoretical_anchored)]:
                if target is anchored:
                    anchor_entry = anchored.anchor_parameters(float(u))
                    a = anchor_entry["a"]
                    b = anchor_entry["b_anchored"]
                else:
                    coefficients = target.coefficients(float(u))
                    a = coefficients[2].numpy()
                    b = coefficients[3].numpy()
                residuals, _ = target.residuals(repeated[:1], float(u), torch.arange(GROUPS)[None])
                residual = residuals[0].numpy() - (a * point[None, :] + b)
                residual_centered = residual - residual.mean(0, keepdims=True)
                destination[time_index, point_index] = (GROUPS**2 / (4 * BATCH) * (1 - BATCH / GROUPS) * (residual_centered**2).sum(0) / (GROUPS - 1))
    elapsed = time.perf_counter() - start
    def summaries(drift, potential, theoretical):
        records = []
        for time_index, u in enumerate(GRID):
            for point_index in range(len(points[0])):
                drift_metrics = metrics(drift[time_index, point_index], exact_drift[time_index, point_index], theoretical[time_index, point_index])
                potential_metrics = metrics(potential[time_index, point_index], np.array(exact_potential[time_index, point_index]), None)
                records.append({"u": float(u), "point_index": point_index, "anchor": point_index == 0, "drift": drift_metrics, "potential": potential_metrics})
        return records
    pair_id = f"training{training_seed}_dataset{dataset_seed}"
    output = OUTPUT_ROOT / pair_id
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "inputs.npz", theta=np.asarray(theta), context=np.asarray(context), variance=np.asarray(parameters["variance"]), means=np.asarray(parameters["means"]), weights=np.asarray(parameters["weights"]))
    np.savez_compressed(output / "raw.npz", points=points, exact_drift=exact_drift, exact_potential=exact_potential, original_drift=raw_original_drift, anchored_drift=raw_anchored_drift, original_potential=raw_original_potential, anchored_potential=raw_anchored_potential, theoretical_original=theoretical_original, theoretical_anchored=theoretical_anchored)
    receipt = {"pair_id": pair_id, "protocol": {"grid": GRID.tolist(), "groups": GROUPS, "dimension": DIMENSION, "batch": BATCH, "replicates": REPLICATES, "directions_seed": RNG_SEED, "direction_count": 8, "same_wor_subsets": True, "device": "cpu", "threads": 1}, "training": receipt, "dataset": identity, "parameter_hash": json_hash({key: np.asarray(value).tolist() for key, value in parameters.items()}), "source_hashes": source_hashes(), "artifact_sha256": {"raw.npz": sha256(output / "raw.npz"), "inputs.npz": sha256(output / "inputs.npz")}, "anchored_cost": anchored.cost_report(), "runtime_seconds": elapsed, "records": {"original": summaries(raw_original_drift, raw_original_potential, theoretical_original), "anchored": summaries(raw_anchored_drift, raw_anchored_potential, theoretical_anchored)}}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False))
    (output / "completed").write_text("completed\n")
    (output / "receipt.sha256").write_text(f"{sha256(output / 'receipt.json')}  receipt.json\n")
    (output / "completed.sha256").write_text(f"{sha256(output / 'completed')}  completed\n")
    return receipt


def summarize(receipts):
    values = []
    for receipt in receipts:
        for method in ["original", "anchored"]:
            for record in receipt["records"][method]:
                values.append({"pair_id": receipt["pair_id"], "method": method, **record})
    anchor = [row for row in values if row["anchor"]]
    nonanchor = [row for row in values if not row["anchor"]]
    def ratios(rows):
        drift = {method: float(np.mean([np.mean(row["drift"]["variance"]) for row in rows if row["method"] == method])) for method in ["original", "anchored"]}
        potential = {method: float(np.mean([np.mean(row["potential"]["variance"]) for row in rows if row["method"] == method])) for method in ["original", "anchored"]}
        return {"count": len(rows) // 2, "drift_variance_mean_by_method": drift, "potential_variance_mean_by_method": potential, "anchored_over_original_drift_variance_ratio": drift["anchored"] / drift["original"], "anchored_over_original_potential_variance_ratio": potential["anchored"] / potential["original"]}
    by_u = {}
    for u in GRID:
        selected = [row for row in values if row["u"] == float(u)]
        by_u[str(float(u))] = {"anchor": ratios([row for row in selected if row["anchor"]]), "nonanchor": ratios([row for row in selected if not row["anchor"]])}
    by_model = {}
    for method in ["original", "anchored"]:
        selected = [row for row in values if row["method"] == method]
        by_model[method] = {"anchor": {"count": len([row for row in selected if row["anchor"]]), "drift_variance_mean": float(np.mean([np.mean(row["drift"]["variance"]) for row in selected if row["anchor"]])), "potential_variance_mean": float(np.mean([np.mean(row["potential"]["variance"]) for row in selected if row["anchor"]]))}, "nonanchor": {"count": len([row for row in selected if not row["anchor"]]), "drift_variance_mean": float(np.mean([np.mean(row["drift"]["variance"]) for row in selected if not row["anchor"]])), "potential_variance_mean": float(np.mean([np.mean(row["potential"]["variance"]) for row in selected if not row["anchor"]]))}}
    return {"overall": {"anchor": ratios(anchor), "nonanchor": ratios(nonanchor)}, "per_u": by_u, "per_model": by_model, "scope": "mechanism diagnostic; no end-to-end accuracy or speed claim"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    global OUTPUT_ROOT
    OUTPUT_ROOT = args.output
    torch.set_num_threads(1)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    tmpdir = OUTPUT_ROOT / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=False)
    os.environ["TMPDIR"] = str(tmpdir)
    directions = fixed_directions()
    np.save(OUTPUT_ROOT / "directions.npy", directions)
    receipts = [run_pair(training_seed, dataset_seed, directions) for training_seed in [0, 1] for dataset_seed in range(600, 604)]
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summarize(receipts), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
