import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.special import expit


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def array_hash(value):
    return json_hash(np.asarray(value).tolist())


def close_error(actual, expected):
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    return float(np.max(np.abs(a - b)))


def relative_error(actual, expected, floor=1e-12):
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), floor)))


def finite_array(value):
    return bool(np.isfinite(np.asarray(value, dtype=np.float64)).all())


def reconstruct_dataset(groups, dimension, seed):
    rng = np.random.default_rng(1000 + seed)
    theta = rng.normal(size=dimension)
    phase = 2 * math.pi * (np.arange(groups)[:, None] + 0.37 * np.arange(dimension)[None]) / groups
    sigma = 0.45 + 0.75 * (0.5 + 0.5 * np.sin(phase))
    offset = 0.3 + 0.5 + 0.5 * np.cos(2 * phase)
    sign = 2 * rng.integers(0, 2, size=(groups, dimension)) - 1
    y = theta[None] + sign * offset + sigma * rng.normal(size=(groups, dimension))
    context = np.stack((y, sigma, offset), axis=-1)
    return theta, context


def coefficient_arrays(parameters, u):
    variance = np.asarray(parameters["variance"], dtype=np.float64)
    means = np.asarray(parameters["means"], dtype=np.float64)
    weights = np.asarray(parameters["weights"], dtype=np.float64)
    alpha = math.exp(-float(u) / 2)
    alpha2 = alpha * alpha
    v = 1 - alpha2 + alpha2 * variance
    a = 1 - 1 / v
    mean = (weights * means).sum(-1)
    b = alpha * mean / v
    delta = means[..., 1] - means[..., 0]
    slope = alpha * delta / v
    intercept = np.log(weights[..., 1] / weights[..., 0]) - alpha2 * (means[..., 1] ** 2 - means[..., 0] ** 2) / (2 * v)
    e0 = alpha * (means[..., 0] - mean) / v
    edelta = alpha * delta / v
    return {"a": a, "b": b, "e0": e0, "edelta": edelta, "slope": slope, "intercept": intercept}


def residuals_at(points, coeffs):
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("points must have shape (points, dimension)")
    probability = expit(x[:, None, :] * coeffs["slope"][None] + coeffs["intercept"][None])
    residual = x[:, None, :] * coeffs["a"][None] + coeffs["b"][None]
    residual = residual + coeffs["e0"][None] + coeffs["edelta"][None] * probability
    return residual


def exact_from_residual(points, residual):
    total = residual.sum(1)
    drift = 0.5 * total
    potential = 0.5 * ((total * total).sum(-1) - (residual * residual).sum((1, 2)))
    return drift, potential


def anchored_parameters(coeffs, directions):
    a = coeffs["a"]
    b = coeffs["b"]
    total_a = a.sum(0)
    anchor = b.sum(0) / (1 - total_a)
    for _ in range(6):
        anchor_residual = residuals_at(anchor[None], coeffs)[0] - anchor[None] * a - b
        total = (anchor[None] * a + b + anchor_residual).sum(0)
        probability = expit(anchor[None] * coeffs["slope"] + coeffs["intercept"])
        derivative = (a + coeffs["edelta"] * coeffs["slope"] * probability * (1 - probability)).sum(0)
        denominator = np.maximum(1 - derivative, 0.2)
        step = np.clip((total - anchor) / denominator, -1.0, 1.0)
        anchor = anchor + step
    anchored_b = b + residuals_at(anchor[None], coeffs)[0] - anchor[None] * a - b
    width = 1 / np.sqrt(1 - a.sum(0))
    return {"anchor": anchor, "width": width, "a": a, "b": anchored_b}


def wor_drift_variance(points, coeffs, control_b, groups, batch):
    residual = residuals_at(points, coeffs)
    affine = points[:, None, :] * coeffs["a"][None] + control_b[None]
    centered = (residual - affine) - (residual - affine).mean(1, keepdims=True)
    return groups**2 / (4 * batch) * (1 - batch / groups) * (centered * centered).sum(1) / (groups - 1)


def verify_training_record(record, verification_context=None, verification_parameters=None):
    failures = []
    training = record["training"]
    root = Path(training["root"])
    checkpoint = root / "final.pt"
    config = root / "config.json"
    summary = root / "summary.json"
    done = root / "done"
    expected = {
        "checkpoint_sha256": checkpoint,
        "config_sha256": config,
        "summary_sha256": summary,
    }
    actual = {}
    for name, path in expected.items():
        if not path.is_file():
            failures.append(f"missing strict training artifact: {path}")
            actual[name] = None
        else:
            actual[name] = sha256(path)
            if actual[name] != training[name]:
                failures.append(f"training hash mismatch: {path}")
    if not done.is_file():
        failures.append(f"missing strict completion marker: {done}")
    else:
        if done.read_text() != "completed\n":
            failures.append(f"invalid strict completion marker: {done}")
    prediction = {}
    if checkpoint.is_file():
        try:
            import sys
            import torch
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from learned_sbi import PosteriorMDN
            model = PosteriorMDN()
            state = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(state, strict=True)
            model.eval()
            context = np.asarray(verification_context, dtype=np.float32)
            expected = verification_parameters
            if context.size and expected:
                with torch.no_grad():
                    variance, means, log_weights = model(torch.as_tensor(context))
                predicted = {"variance": variance.numpy(), "means": means.numpy(), "weights": log_weights.double().softmax(-1).numpy()}
                prediction = {name: {"max_abs": close_error(predicted[name], np.asarray(expected[name])), "max_relative": relative_error(predicted[name], np.asarray(expected[name]))} for name in predicted}
                if any(item["max_abs"] > 2e-5 for item in prediction.values()):
                    failures.append("strict checkpoint prediction mismatch")
        except Exception as exc:
            failures.append(f"strict checkpoint reload/prediction failed: {exc}")
    return {"passed": not failures, "failures": failures, "actual_hashes": actual, "checkpoint_prediction": prediction}


def audit_pair(pair, directions, source_paths):
    failures = []
    receipt_path = pair / "receipt.json"
    raw_path = pair / "raw.npz"
    inputs_path = pair / "inputs.npz"
    completed_path = pair / "completed"
    receipt_sha_path = pair / "receipt.sha256"
    completed_sha_path = pair / "completed.sha256"
    receipt = json.loads(receipt_path.read_text())
    inputs = np.load(inputs_path, allow_pickle=False)
    raw = np.load(raw_path, allow_pickle=False)
    groups = int(receipt["protocol"]["groups"])
    dimension = int(receipt["protocol"]["dimension"])
    batch = int(receipt["protocol"]["batch"])
    replicates = int(receipt["protocol"]["replicates"])
    grid = np.asarray(receipt["protocol"]["grid"], dtype=np.float64)
    parameters = {key: np.asarray(inputs[key], dtype=np.float64) for key in ["variance", "means", "weights"]}
    checks = {"pair_id": receipt["pair_id"], "failures": []}
    if receipt["pair_id"] != pair.name:
        failures.append("pair_id mismatch")
    for path in [receipt_path, raw_path, inputs_path, completed_path, receipt_sha_path, completed_sha_path]:
        if not path.is_file():
            failures.append(f"missing pair artifact: {path.name}")
    if completed_path.is_file() and completed_path.read_text() != "completed\n":
        failures.append("completed marker content mismatch")
    if receipt_sha_path.is_file():
        expected = receipt_sha_path.read_text().strip().split()[0]
        if expected != sha256(receipt_path):
            failures.append("receipt.sha256 mismatch")
    if completed_sha_path.is_file():
        expected = completed_sha_path.read_text().strip().split()[0]
        if expected != sha256(completed_path):
            failures.append("completed.sha256 mismatch")
    if receipt["artifact_sha256"].get("raw.npz") != sha256(raw_path):
        failures.append("raw.npz hash mismatch")
    if receipt["artifact_sha256"].get("inputs.npz") != sha256(inputs_path):
        failures.append("inputs.npz hash mismatch")
    source_hash_check = {}
    for name, expected in receipt["source_hashes"].items():
        path = source_paths.get(name, Path(name))
        actual = sha256(path) if path.is_file() else None
        source_hash_check[name] = {"expected": expected, "actual": actual, "passed": actual == expected}
        if actual != expected:
            failures.append(f"source hash mismatch or missing: {name}")
    strict_training = verify_training_record(receipt, inputs["context"], parameters)
    if not strict_training["passed"]:
        failures.extend(strict_training["failures"])
    theta_expected, context_expected = reconstruct_dataset(groups, dimension, int(receipt["dataset"]["dataset_seed"]))
    context_error = close_error(inputs["context"], context_expected)
    theta_error = close_error(inputs["theta"], theta_expected)
    if array_hash(inputs["theta"]) != receipt["dataset"]["theta_sha256"]:
        failures.append("theta receipt hash mismatch")
    if array_hash(inputs["context"]) != receipt["dataset"]["context_sha256"]:
        failures.append("context receipt hash mismatch")
    parameter_hash = json_hash({key: parameters[key].tolist() for key in ["variance", "means", "weights"]})
    if parameter_hash != receipt["parameter_hash"]:
        failures.append("parameter hash mismatch")
    if inputs["theta"].shape != theta_expected.shape or inputs["context"].shape != context_expected.shape:
        failures.append("context reconstruction shape mismatch")
    if context_error != 0 or theta_error != 0:
        failures.append("context reconstruction numerical mismatch")
    if directions.shape != (int(receipt["protocol"]["direction_count"]), dimension):
        failures.append("direction shape mismatch")
    expected_directions = np.random.default_rng(int(receipt["protocol"]["directions_seed"])).choice(np.array([-1.0, 1.0]), size=directions.shape)
    direction_error = close_error(directions, expected_directions)
    if direction_error > 1e-12:
        failures.append("directions reconstruction mismatch")
    shapes = {key: list(raw[key].shape) for key in raw.files}
    expected_shapes = {
        "points": [len(grid), 1 + directions.shape[0], dimension],
        "exact_drift": [len(grid), 1 + directions.shape[0], dimension],
        "exact_potential": [len(grid), 1 + directions.shape[0]],
        "original_drift": [len(grid), 1 + directions.shape[0], replicates, dimension],
        "anchored_drift": [len(grid), 1 + directions.shape[0], replicates, dimension],
        "original_potential": [len(grid), 1 + directions.shape[0], replicates],
        "anchored_potential": [len(grid), 1 + directions.shape[0], replicates],
        "theoretical_original": [len(grid), 1 + directions.shape[0], dimension],
        "theoretical_anchored": [len(grid), 1 + directions.shape[0], dimension],
    }
    if shapes != expected_shapes:
        failures.append("raw array shape mismatch")
    discrepancy = {"context_max_abs": context_error, "theta_max_abs": theta_error, "direction_max_abs": direction_error, "max": max(context_error, theta_error, direction_error)}
    recomputed = {"exact_drift": [], "exact_potential": [], "theoretical_original": [], "theoretical_anchored": []}
    raw_receipt_discrepancies = {"original_drift_mean": [], "anchored_drift_mean": [], "original_drift_variance": [], "anchored_drift_variance": [], "original_potential_mean": [], "anchored_potential_mean": [], "original_potential_variance": [], "anchored_potential_variance": [], "theoretical_original": [], "theoretical_anchored": []}
    raw_receipt_relative = {key: [] for key in raw_receipt_discrepancies}
    for time_index, u in enumerate(grid):
        coeffs = coefficient_arrays(parameters, float(u))
        anchor_info = anchored_parameters(coeffs, directions)
        anchor = anchor_info["anchor"]
        width = anchor_info["width"]
        expected_points = np.vstack((anchor, anchor[None] + directions * width[None]))
        point_error = close_error(raw["points"][time_index], expected_points)
        discrepancy.setdefault("points", []).append(point_error)
        if point_error > 1e-12:
            failures.append(f"point reconstruction mismatch at u={u}")
        original_residual = residuals_at(raw["points"][time_index], coeffs)
        original_exact_drift, original_exact_potential = exact_from_residual(raw["points"][time_index], original_residual)
        anchored_residual = original_residual
        anchored_exact_drift, anchored_exact_potential = exact_from_residual(raw["points"][time_index], anchored_residual)
        exact_drift_error = close_error(raw["exact_drift"][time_index], original_exact_drift)
        exact_potential_error = close_error(raw["exact_potential"][time_index], original_exact_potential)
        discrepancy.setdefault("exact_drift", []).append(exact_drift_error)
        discrepancy.setdefault("exact_potential", []).append(exact_potential_error)
        if exact_drift_error > 1e-11 or exact_potential_error > 1e-10:
            failures.append(f"exact formula mismatch at u={u}")
        theoretical_original = wor_drift_variance(raw["points"][time_index], coeffs, coeffs["b"], groups, batch)
        theoretical_anchored = wor_drift_variance(raw["points"][time_index], coeffs, anchor_info["b"], groups, batch)
        discrepancy.setdefault("theoretical_original", []).append(close_error(raw["theoretical_original"][time_index], theoretical_original))
        discrepancy.setdefault("theoretical_anchored", []).append(close_error(raw["theoretical_anchored"][time_index], theoretical_anchored))
        for method, drift_key, potential_key, exact_drift, exact_potential, theory in [
            ("original", "original_drift", "original_potential", original_exact_drift, original_exact_potential, theoretical_original),
            ("anchored", "anchored_drift", "anchored_potential", anchored_exact_drift, anchored_exact_potential, theoretical_anchored),
        ]:
            drift = raw[drift_key][time_index]
            potential = raw[potential_key][time_index]
            saved = receipt["records"][method][time_index * (1 + directions.shape[0]):(time_index + 1) * (1 + directions.shape[0])]
            for point_index, record in enumerate(saved):
                raw_mean = drift[point_index].mean(0)
                raw_variance = drift[point_index].var(0, ddof=1)
                raw_potential_mean = potential[point_index].mean()
                raw_potential_variance = potential[point_index].var(ddof=1)
                expected_error = raw_mean - exact_drift[point_index]
                for key, actual, expected in [
                    (f"{method}_drift_mean", raw_mean, record["drift"]["mean"]),
                    (f"{method}_drift_variance", raw_variance, record["drift"]["variance"]),
                    (f"{method}_potential_mean", raw_potential_mean, record["potential"]["mean"]),
                    (f"{method}_potential_variance", raw_potential_variance, record["potential"]["variance"]),
                ]:
                    raw_receipt_discrepancies[key].append(close_error(actual, expected))
                    raw_receipt_relative[key].append(relative_error(actual, expected, floor=1e-6))
                raw_receipt_discrepancies[f"{method}_drift_mean"].append(close_error(expected_error, record["drift"]["error"]))
                raw_receipt_relative[f"{method}_drift_mean"].append(relative_error(expected_error, record["drift"]["error"], floor=1e-6))
                raw_receipt_discrepancies["theoretical_original" if method == "original" else "theoretical_anchored"].append(close_error(theory[point_index], record["drift"]["theoretical_drift_variance"]))
                raw_receipt_relative["theoretical_original" if method == "original" else "theoretical_anchored"].append(relative_error(theory[point_index], record["drift"]["theoretical_drift_variance"], floor=1e-6))
                if not np.isfinite(drift[point_index]).all() or not np.isfinite(potential[point_index]).all():
                    failures.append(f"nonfinite raw values at {method} u={u} point={point_index}")
    discrepancy["max"] = max([value for value in discrepancy.values() if isinstance(value, (int, float))] + [max(value) for value in discrepancy.values() if isinstance(value, list) and value] + [0.0])
    discrepancy["raw_vs_receipt_max"] = {key: max(value) if value else 0.0 for key, value in raw_receipt_discrepancies.items()}
    discrepancy["raw_vs_receipt_relative_max"] = {key: max(value) if value else 0.0 for key, value in raw_receipt_relative.items()}
    discrepancy["raw_vs_receipt_relative_error_floor"] = 1e-6
    discrepancy["raw_vs_receipt_maximum"] = max(discrepancy["raw_vs_receipt_max"].values())
    discrepancy["raw_vs_receipt_relative_maximum"] = max(discrepancy["raw_vs_receipt_relative_max"].values())
    if discrepancy["raw_vs_receipt_maximum"] > 1e-8 or discrepancy["raw_vs_receipt_relative_maximum"] > 1e-8:
        failures.append("raw receipt recomputation exceeds absolute/relative tolerance")
    ratio_data = {}
    for method in ["original", "anchored"]:
        for time_index, u in enumerate(grid):
            key = f"{method}:{float(u)}"
            drifts = raw[f"{method}_drift"][time_index, 1:]
            potentials = raw[f"{method}_potential"][time_index, 1:]
            ratio_data[key] = {"point_count": int(drifts.shape[0]), "drift_variance_mean": float(np.mean(drifts.var(axis=1, ddof=1))), "potential_variance_mean": float(np.mean(potentials.var(axis=1, ddof=1)))}
    checks.update({"passed": not failures, "failures": failures, "source_hashes": source_hash_check, "strict_training": strict_training, "context_reconstruction": {"theta_max_abs": theta_error, "context_max_abs": context_error}, "discrepancies": discrepancy, "ratio_data": ratio_data, "finite_raw": all(finite_array(raw[key]) for key in raw.files)})
    checks["failure_count"] = len(failures)
    return checks


def summarize_ratios(pair_audits):
    rows = []
    for audit in pair_audits:
        pair_id = audit["pair_id"]
        training_seed = int(pair_id.split("_")[0].replace("training", ""))
        dataset_seed = int(pair_id.split("_")[1].replace("dataset", ""))
        for key, value in audit["ratio_data"].items():
            method, u = key.split(":")
            rows.append({"pair_id": pair_id, "training_seed": training_seed, "dataset_seed": dataset_seed, "method": method, "u": float(u), **value})
    def ratios(selected):
        original = [row for row in selected if row["method"] == "original"]
        anchored = [row for row in selected if row["method"] == "anchored"]
        original_d = float(np.mean([row["drift_variance_mean"] for row in original]))
        anchored_d = float(np.mean([row["drift_variance_mean"] for row in anchored]))
        original_p = float(np.mean([row["potential_variance_mean"] for row in original]))
        anchored_p = float(np.mean([row["potential_variance_mean"] for row in anchored]))
        return {"pair_count": len(set(row["pair_id"] for row in selected)), "record_count_per_method": int(sum(row["point_count"] for row in original)), "method_record_count": int(sum(row["point_count"] for row in selected)), "original_drift_variance_mean": original_d, "anchored_drift_variance_mean": anchored_d, "anchored_over_original_drift_variance_ratio": anchored_d / original_d, "original_potential_variance_mean": original_p, "anchored_potential_variance_mean": anchored_p, "anchored_over_original_potential_variance_ratio": anchored_p / original_p}
    overall = ratios(rows)
    per_u = {str(u): ratios([row for row in rows if row["u"] == u]) for u in sorted(set(row["u"] for row in rows), reverse=True)}
    per_training_seed = {str(seed): ratios([row for row in rows if row["training_seed"] == seed]) for seed in sorted(set(row["training_seed"] for row in rows))}
    if overall["record_count_per_method"] != 256 or any(item["record_count_per_method"] != 64 for item in per_u.values()) or any(item["record_count_per_method"] != 128 for item in per_training_seed.values()):
        raise ValueError("non-anchor aggregation count mismatch")
    return {"nonanchor_scope": "variance ratios use point indices 1:9 at each of 4 saved times, including u=0; exactly 8 pairs x 4 times x 8 non-anchor points = 256 records per method", "overall_256_nonanchor_points": overall, "per_u": per_u, "per_training_seed": per_training_seed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("work/anchored-conditional-run-20260921-final-v3"))
    parser.add_argument("--output", type=Path, default=Path("work/anchored-conditional-audit.json"))
    args = parser.parse_args()
    os.environ["TMPDIR"] = str(args.root)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    np.seterr(all="raise")
    root = args.root
    summary = json.loads((root / "summary.json").read_text())
    directions = np.load(root / "directions.npy", allow_pickle=False)
    source_paths = {name: Path(name) for name in ["docs/ANCHORED_CONDITIONAL_PROTOCOL_20260921.md", "anchored_tail_20260921.py", "composition_extension.py", "composition_benchmark.py", "learned_sbi.py"]}
    pair_dirs = sorted(path for path in root.glob("training*_dataset*") if path.is_dir())
    pair_audits = [audit_pair(path, directions, source_paths) for path in pair_dirs]
    expected_pair_ids = {f"training{training}_dataset{dataset}" for training in [0, 1] for dataset in range(600, 604)}
    actual_pair_ids = {audit["pair_id"] for audit in pair_audits}
    failures = []
    if actual_pair_ids != expected_pair_ids:
        failures.append("pair directory set mismatch")
    if len(pair_audits) != 8:
        failures.append("expected 8 pair directories")
    summary_counts = {"anchor": summary["overall"]["anchor"]["count"], "nonanchor": summary["overall"]["nonanchor"]["count"]}
    if summary_counts != {"anchor": 32, "nonanchor": 256}:
        failures.append("summary count mismatch")
    for audit in pair_audits:
        failures.extend([f"{audit['pair_id']}: {message}" for message in audit["failures"]])
    recomputed_ratios = summarize_ratios(pair_audits)
    audited_overall = recomputed_ratios["overall_256_nonanchor_points"]
    expected_overall = summary["overall"]["nonanchor"]
    summary_fields = [
        "original_drift_variance_mean", "anchored_drift_variance_mean",
        "anchored_over_original_drift_variance_ratio", "original_potential_variance_mean",
        "anchored_potential_variance_mean", "anchored_over_original_potential_variance_ratio",
    ]
    summary_crossvalidation = {}
    summary_lookup = {
        "original_drift_variance_mean": expected_overall["drift_variance_mean_by_method"]["original"],
        "anchored_drift_variance_mean": expected_overall["drift_variance_mean_by_method"]["anchored"],
        "original_potential_variance_mean": expected_overall["potential_variance_mean_by_method"]["original"],
        "anchored_potential_variance_mean": expected_overall["potential_variance_mean_by_method"]["anchored"],
        "anchored_over_original_drift_variance_ratio": expected_overall["anchored_over_original_drift_variance_ratio"],
        "anchored_over_original_potential_variance_ratio": expected_overall["anchored_over_original_potential_variance_ratio"],
    }
    for field in summary_fields:
        audit_value = audited_overall[field]
        summary_value = summary_lookup[field]
        absolute = abs(audit_value - summary_value)
        relative = absolute / max(abs(summary_value), 1e-300)
        summary_crossvalidation[field] = {"audit": audit_value, "summary": summary_value, "absolute_error": absolute, "relative_error": relative}
        if absolute > 1e-8 and relative > 1e-10:
            failures.append(f"summary cross-validation mismatch: {field}")
    posthoc_source = Path(__file__)
    output = {"status": "passed" if not failures else "failed", "scope": "independent local audit of completed anchored conditional run; no rerun of experiment", "input_root": str(root), "pair_count": len(pair_audits), "expected_pair_count": 8, "summary_original_per_model_not_used_for_training_breakdown": True, "global_failures": failures, "source_hash_provenance": {"audit_runner": str(posthoc_source), "audit_runner_sha256": sha256(posthoc_source), "classification": "posthoc audit-runner provenance; not part of the original pre-run source hash set"}, "strict_requirements": {"cpu_threads": 1, "tmpdir": str(root), "gpu_used": False, "server_used": False, "git_used": False}, "pair_audits": pair_audits, "recomputed_variance_ratios": recomputed_ratios, "summary_crossvalidation": summary_crossvalidation, "mc_vs_analytic_relative_variance": "descriptive only; raw empirical variance compared with saved analytic WOR drift variance, without inferential claims", "original_summary": {"scope": summary.get("scope"), "per_model_present": "per_model" in summary}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
