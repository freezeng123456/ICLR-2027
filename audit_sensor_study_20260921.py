import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp
from scipy.stats import norm, wasserstein_distance


def _read_json(path):
    return json.loads(path.read_text())


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sign_matrix(groups):
    values = np.arange(1 << groups, dtype=np.uint64)[:, None]
    shifts = np.arange(groups, dtype=np.uint64)[None, :]
    return ((values >> shifts) & 1).astype(np.int8)


def _problem_arrays(payload):
    directions = np.asarray(payload["directions"], dtype=np.float64)
    noise_std = np.asarray(payload["noise_std"], dtype=np.float64)
    positive_probability = np.asarray(payload["positive_probability"], dtype=np.float64)
    observations = np.asarray(payload["observations"], dtype=np.float64)
    truth = np.asarray(payload["truth"], dtype=np.float64)
    return directions, noise_std, positive_probability, observations, truth


def _independent_exact(payload):
    directions, noise_std, positive_probability, observations, _ = _problem_arrays(payload)
    groups, dimension = directions.shape
    identity = np.eye(dimension, dtype=np.float64)
    precision = np.empty((groups, dimension, dimension), dtype=np.float64)
    means = np.empty((groups, 2, dimension), dtype=np.float64)
    logdet_factor = np.empty(groups, dtype=np.float64)
    for group in range(groups):
        denominator = noise_std[group] ** 2 + directions[group] @ directions[group]
        covariance = identity - np.outer(directions[group], directions[group]) / denominator
        precision[group] = np.linalg.inv(covariance)
        logdet_factor[group] = np.linalg.slogdet(covariance)[1]
        center = observations[group] * directions[group] / denominator
        means[group, 0] = -center
        means[group, 1] = center
    total_precision = precision.sum(axis=0) - (groups - 1) * identity
    factor, lower = cho_factor(total_precision, lower=True, check_finite=True)
    covariance = cho_solve((factor, lower), identity, check_finite=True)
    signs = _sign_matrix(groups)
    selected_means = means[np.arange(groups)[None, :], signs]
    selected_precision_means = np.einsum("gij,kgj->kgi", precision, selected_means)
    b_values = selected_precision_means.sum(axis=1)
    component_means = np.einsum("ij,kj->ki", covariance, b_values)
    selected_quadratic = np.einsum("kgi,kgi->kg", selected_means, selected_precision_means)
    selected_log_weights = np.log(np.where(signs, positive_probability, 1.0 - positive_probability))
    component_logs = (selected_log_weights.sum(axis=1)
                      - 0.5 * selected_quadratic.sum(axis=1)
                      - 0.5 * logdet_factor.sum()
                      + 0.5 * np.einsum("ki,ij,kj->k", b_values, covariance, b_values))
    probabilities = np.exp(component_logs - logsumexp(component_logs))
    return probabilities, component_means, covariance, precision


def _direct_sensor_log_evidence(payload):
    directions, noise_std, positive_probability, observations, _ = _problem_arrays(payload)
    groups = directions.shape[0]
    signs = _sign_matrix(groups)
    noise_covariance = np.diag(noise_std ** 2)
    terms = np.empty(signs.shape[0], dtype=np.float64)
    for row, bits in enumerate(signs):
        signed = directions * (2.0 * bits[:, None] - 1.0)
        covariance = noise_covariance + signed @ signed.T
        factor, lower = cho_factor(covariance, lower=True, check_finite=True)
        solved = cho_solve((factor, lower), observations, check_finite=True)
        logdet = 2.0 * np.log(np.diag(factor)).sum()
        terms[row] = (np.log(np.where(bits, positive_probability, 1.0 - positive_probability)).sum()
                      - 0.5 * (groups * np.log(2.0 * np.pi) + logdet + observations @ solved))
    return float(logsumexp(terms))


def _composition_normalizer(payload, sensor_log_evidence):
    _, noise_std, _, observations, _ = _problem_arrays(payload)
    directions = np.asarray(payload["directions"], dtype=np.float64)
    variances = noise_std ** 2 + np.einsum("gi,gi->g", directions, directions)
    marginal_sum = np.sum(-0.5 * (np.log(2.0 * np.pi * variances) + observations ** 2 / variances))
    return float(sensor_log_evidence - marginal_sum)


def _direct_log_density(payload, points, log_evidence):
    directions, noise_std, positive_probability, observations, _ = _problem_arrays(payload)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points[None, :]
    prior = norm.logpdf(points, loc=0.0, scale=1.0).sum(axis=1)
    sensor = np.zeros(points.shape[0], dtype=np.float64)
    for group in range(directions.shape[0]):
        projection = points @ directions[group]
        terms = np.column_stack((
            np.log1p(-positive_probability[group])
            + norm.logpdf(observations[group], loc=-projection, scale=noise_std[group]),
            np.log(positive_probability[group])
            + norm.logpdf(observations[group], loc=projection, scale=noise_std[group]),
        ))
        sensor += logsumexp(terms, axis=1)
    return prior + sensor - log_evidence


def _mixture_log_density(points, probabilities, means, covariance):
    points = np.asarray(points, dtype=np.float64)
    factor, lower = cho_factor(covariance, lower=True, check_finite=True)
    delta = points[:, None, :] - means[None, :, :]
    solved = cho_solve((factor, lower), delta.reshape(-1, delta.shape[-1]).T,
                       check_finite=True).T.reshape(delta.shape)
    quadratic = np.einsum("nki,nki->nk", delta, solved)
    logdet = 2.0 * np.log(np.diag(factor)).sum()
    values = -0.5 * (quadratic + logdet + points.shape[1] * np.log(2.0 * np.pi))
    return logsumexp(values + np.log(probabilities)[None, :], axis=1)


def _factor_precision(payload, time):
    directions, noise_std, _, _, _ = _problem_arrays(payload)
    dimension = directions.shape[1]
    identity = np.eye(dimension, dtype=np.float64)
    rho = np.exp(-0.5 * float(time))
    result = []
    for direction, sigma in zip(directions, noise_std):
        denominator = sigma * sigma + direction @ direction
        covariance = identity - np.outer(direction, direction) / denominator
        noised = rho * rho * covariance + (1.0 - rho * rho) * identity
        result.append(np.linalg.inv(noised))
    return np.asarray(result)


def _matrix_certificate(asset_payload, steps):
    if steps < 1:
        return {"status": "invalid_steps"}
    dimension = len(asset_payload["truth"])
    grid = np.linspace(np.sqrt(20.0), 0.0, steps + 1) ** 2
    covariance = np.eye(dimension, dtype=np.float64)
    minimum = math.inf
    statuses = []
    for index, (u, next_u) in enumerate(zip(grid[:-1], grid[1:])):
        h = float(u - next_u)
        A = _factor_precision(asset_payload, u) - np.eye(dimension)[None, :, :]
        total = A.sum(axis=0)
        C = 0.5 * (total.T @ total - np.einsum("gji,gjk->ik", A, A))
        L = np.eye(dimension) - h * (2.0 * total + np.eye(dimension)) / 2.0
        precision = np.linalg.inv(covariance) - 2.0 * h * C
        eigenvalues = np.linalg.eigvalsh(precision)
        minimum = min(minimum, float(eigenvalues[0]))
        if eigenvalues[0] <= 1e-11 * max(1.0, np.linalg.norm(precision, ord=2)):
            statuses.append("nonfinite")
            break
        factor, lower = cho_factor(precision, lower=True, check_finite=True)
        inverse = cho_solve((factor, lower), np.eye(dimension), check_finite=True)
        covariance = L @ inverse @ L.T + h * np.eye(dimension)
    return {"status": "finite" if len(statuses) == 0 else statuses[0],
            "steps": steps, "minimum_precision_eigenvalue": minimum,
            "final_tail_covariance": covariance.tolist()}


def _receipt_check(directory):
    receipt_path = directory / "receipt.json"
    if not receipt_path.exists():
        return {"status": "missing", "checked": 0, "mismatches": []}
    receipt = _read_json(receipt_path)
    if not isinstance(receipt.get("files"), dict) or not receipt["files"]:
        return {"status": "empty", "checked": 0, "mismatches": []}
    mismatches = []
    for name, expected in receipt.get("files", {}).items():
        path = directory / name
        if not path.exists() or _sha256(path) != expected:
            mismatches.append(name)
    return {"status": "ok" if not mismatches else "mismatch", "checked": len(receipt.get("files", {})),
            "mismatches": mismatches}


def _audit_asset(asset):
    problem = _read_json(asset / "problem.json")
    probabilities, means, covariance, precision = _independent_exact(problem)
    with np.load(asset / "exact.npz") as saved:
        saved_probabilities = saved["probabilities"]
        saved_means = saved["means"]
        saved_covariance = saved["covariance"]
    exact_checks = {
        "probabilities": bool(np.allclose(saved_probabilities, probabilities, atol=2e-11, rtol=2e-11)),
        "means": bool(np.allclose(saved_means, means, atol=2e-11, rtol=2e-11)),
        "covariance": bool(np.allclose(saved_covariance, covariance, atol=2e-11, rtol=2e-11)),
        "weights_sum": bool(np.isclose(np.sum(saved_probabilities), 1.0, atol=2e-11)),
    }
    directions, _, _, _, truth = _problem_arrays(problem)
    rng = np.random.default_rng(20260921 + directions.shape[0] + directions.shape[1])
    points = rng.normal(size=(8, directions.shape[1]))
    sensor_log_evidence = _direct_sensor_log_evidence(problem)
    composition_logz = _composition_normalizer(problem, sensor_log_evidence)
    density = _direct_log_density(problem, points, sensor_log_evidence)
    mixture_density = _mixture_log_density(points, saved_probabilities, saved_means, saved_covariance)
    density_check = bool(np.all(np.isfinite(density)) and
                         np.allclose(density, mixture_density, atol=2e-9, rtol=2e-9))
    reference_checks = _receipt_check(asset)
    reference_path = asset / "reference.npz"
    reference_summary = {}
    if reference_path.exists():
        with np.load(reference_path) as reference:
            ref_samples = reference["samples"]
            ref_mean = reference["mean"]
            ref_covariance = reference["covariance"]
            sample_mean = ref_samples.mean(axis=0)
            sample_covariance = np.cov(ref_samples, rowvar=False)
            reference_summary = {
                "sample_count": int(ref_samples.shape[0]),
                "mean_error": float(np.linalg.norm(sample_mean - ref_mean)),
                "covariance_error": float(np.linalg.norm(sample_covariance - ref_covariance)),
                "truth_quantile_coverage": [
                    {"lower95": bool(np.quantile(ref_samples[:, index], 0.025) <= truth[index]),
                     "upper95": bool(truth[index] <= np.quantile(ref_samples[:, index], 0.975))}
                    for index in range(truth.size)
                ],
                "mean_matches_exact": bool(np.allclose(ref_mean, saved_probabilities @ saved_means,
                                                        atol=2e-9, rtol=2e-9)),
                "covariance_matches_exact": bool(np.allclose(
                    ref_covariance,
                    saved_covariance + (saved_means - saved_probabilities @ saved_means).T
                    @ ((saved_means - saved_probabilities @ saved_means) * saved_probabilities[:, None]),
                    atol=2e-9, rtol=2e-9)),
                "directions_normalized": bool(np.allclose(
                    np.linalg.norm(reference["directions"], axis=1), 1.0, atol=2e-12, rtol=0.0)),
            }
            if ref_samples.shape[1] == 2:
                component_std = np.sqrt(np.diag(saved_covariance)).max()
                grid_bound = float(np.max(np.abs(saved_means)) + 8.0 * component_std)
                grid_bound = max(grid_bound, 4.0)
                grid = np.linspace(-grid_bound, grid_bound, 281)
                xx, yy = np.meshgrid(grid, grid, indexing="ij")
                grid_points = np.column_stack((xx.ravel(), yy.ravel()))
                grid_density = np.exp(_direct_log_density(problem, grid_points, sensor_log_evidence)).reshape(xx.shape)
                mass = np.trapezoid(np.trapezoid(grid_density, grid, axis=1), grid)
                grid_mean = np.array([(grid_density * xx).sum(), (grid_density * yy).sum()]) * (grid[1] - grid[0]) ** 2
                reference_summary["grid_mass"] = float(mass)
                reference_summary["grid_mean_error"] = float(np.linalg.norm(
                    grid_mean - saved_probabilities @ saved_means))
                reference_summary["grid_bound"] = grid_bound
                reference_summary["grid_normalization_pass"] = bool(abs(mass - 1.0) <= 5e-3)
            regenerated = []
            dataset_seed = int(asset.name.rsplit("_", 1)[1])
            rng = np.random.default_rng(921000 + dataset_seed)
            for _ in range(2):
                indices = rng.choice(saved_probabilities.size, size=ref_samples.shape[0], p=saved_probabilities)
                regenerated.append(saved_means[indices] + rng.multivariate_normal(
                    np.zeros(saved_means.shape[1]), saved_covariance, size=ref_samples.shape[0]))
            reference_summary["regenerated_reference_available"] = bool("second_samples" in reference.files)
            reference_summary["regenerated_first_matches"] = bool(np.allclose(regenerated[0], ref_samples))
            if "second_samples" in reference.files:
                reference_summary["regenerated_second_matches"] = bool(
                    np.allclose(regenerated[1], reference["second_samples"]))
            expected_directions = np.random.default_rng(921032).normal(size=(32, directions.shape[1]))
            expected_directions /= np.linalg.norm(expected_directions, axis=1, keepdims=True)
            reference_summary["directions_matches"] = bool(reference["directions"].shape == expected_directions.shape
                and np.allclose(reference["directions"], expected_directions, atol=1e-14, rtol=0))
    asset_pass = (reference_checks["status"] == "ok"
                  and all(exact_checks.values())
                  and density_check
                  and bool(reference_summary)
                  and reference_summary.get("regenerated_reference_available", False)
                  and reference_summary.get("sample_count") == 65536
                  and all(value for key, value in reference_summary.items()
                                                    if key.endswith("_matches_exact")
                                                    or key.endswith("_normalization_pass")
                                                    or key.endswith("_matches")))
    return {
        "asset": asset.name,
        "groups": int(directions.shape[0]),
        "dimension": int(directions.shape[1]),
        "receipt": reference_checks,
        "exact": exact_checks,
        "density_finite": density_check,
        "density_matches_saved_mixture": density_check,
        "pass": bool(asset_pass),
        "log_sensor_evidence": sensor_log_evidence,
        "log_composition_normalizer": composition_logz,
        "reference": reference_summary,
    }


def _independent_metrics(samples, weights, reference, truth):
    dimension = samples.shape[1]
    projected = [wasserstein_distance(samples @ direction, reference["samples"] @ direction, u_weights=weights)
                 for direction in reference["directions"]]
    marginal = [wasserstein_distance(samples[:, d], reference["samples"][:, d], u_weights=weights) for d in range(dimension)]
    mean = np.sum(samples * weights[:, None], axis=0)
    centered = samples - mean
    covariance = np.einsum("n,ni,nj->ij", weights, centered, centered)
    quantiles = []
    for d in range(dimension):
        order = np.argsort(samples[:, d], kind="stable")
        cdf = weights[order].cumsum()
        cdf[-1] = 1.
        quantiles.append(samples[order[np.searchsorted(cdf, [.025, .05, .95, .975])], d])
    quantiles = np.array(quantiles)
    return dict(sliced_w1_32=float(np.mean(projected)), w1_mean=float(np.mean(marginal)),
        mean_error=float(np.linalg.norm(mean - reference["mean"]) / np.sqrt(dimension)),
        covariance_error=float(np.linalg.norm(covariance - reference["covariance"]) / dimension),
        state_mse=float(np.mean((mean - truth) ** 2)),
        coverage90=float(np.mean((quantiles[:, 1] <= truth) & (truth <= quantiles[:, 2]))),
        coverage95=float(np.mean((quantiles[:, 0] <= truth) & (truth <= quantiles[:, 3]))),
        width90=float(np.mean(quantiles[:, 2] - quantiles[:, 1])), width95=float(np.mean(quantiles[:, 3] - quantiles[:, 0])))


def _audit_cell(cell):
    receipt = _receipt_check(cell)
    result = {"cell": str(cell), "receipt": receipt}
    samples_path = cell / "samples.npz"
    summary_path = cell / "summary.json"
    if not samples_path.exists() or not summary_path.exists():
        result["status"] = "missing_outputs"
        result["pass"] = False
        return result
    summary = _read_json(summary_path)
    config_path = cell / "config.json"
    with np.load(samples_path) as data:
        samples = data["samples"]
        weights = data["weights"]
    metric_check = {}
    structural_pass = False
    if config_path.exists():
        config = _read_json(config_path)
        asset_name = f"d{config['dimension']}_{config['regime']}_{config['dataset_seed']}"
        reference_path = cell.parents[2] / "assets" / asset_name / "reference.npz"
        structural_pass = (samples.shape == (config["particles"], config["dimension"])
            and weights.shape == (config["particles"],) and np.isfinite(samples).all()
            and np.isfinite(weights).all() and (weights >= 0).all()
            and np.isclose(weights.sum(), 1., atol=1e-10, rtol=0)
            and all(summary.get(key) == value for key, value in config.items())
            and (cell / "done").is_file() and (cell / "done").read_text().strip() == "completed")
        if reference_path.exists() and structural_pass:
            with np.load(reference_path) as reference:
                payload = _read_json(reference_path.parent / "problem.json")
                values = _independent_metrics(samples, weights, reference, np.array(payload["truth"]))
            metric_check = {key: dict(value=value, discrepancy=abs(value - summary.get(key, np.inf)),
                passed=bool(np.isclose(value, summary.get(key, np.nan), atol=2e-10, rtol=2e-10))) for key, value in values.items()}
    result.update({"status": summary.get("status", "unknown"), "setting_id": summary.get("setting_id"),
                   "sample_count": int(samples.shape[0]), "weight_sum": float(weights.sum()),
                   "weight_min": float(weights.min()), "weight_max": float(weights.max()),
                   "finite": bool(np.isfinite(samples).all() and np.isfinite(weights).all()),
                   "effective_sample_size": float(1.0 / np.sum(weights ** 2))})
    result["independent_metrics"] = metric_check
    metrics_pass = bool(metric_check) and all(item["passed"] for item in metric_check.values())
    result["pass"] = bool(receipt["status"] == "ok"
                           and result["status"] == "completed"
                           and result["finite"]
                           and np.isclose(result["weight_sum"], 1.0, atol=2e-10)
                           and np.all(weights >= 0.0)
                           and structural_pass and metrics_pass)
    return result


def audit(root, output, assets_only=False, phase=None, sources=None, workers=1):
    root = Path(root)
    output = Path(output)
    assets = sorted(path for path in (root / "assets").glob("*") if (path / "problem.json").exists())
    if not 1 <= workers <= 4:
        raise ValueError("workers must be in [1,4]")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        asset_reports = list(pool.map(_audit_asset, assets))
    cell_reports = []
    if not assets_only:
        phases = [phase] if phase else ["development", "confirmation"]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            cell_reports = list(pool.map(_audit_cell, [cell for name in phases for cell in sorted((root / name).glob("cells/cell_*"))]))
    source_report = {}
    if sources:
        for source in sources:
            path = Path(source)
            if path.is_dir():
                candidates = sorted(item for item in path.rglob("*") if item.is_file())
                for item in candidates:
                    key = str(item.relative_to(path))
                    source_report[key] = {"exists": True, "sha256": _sha256(item)}
                    source_report[item.name] = {"exists": True, "sha256": _sha256(item)}
            else:
                source_report[path.name] = {"exists": path.exists(),
                                            "sha256": _sha256(path) if path.exists() else None}
    manifest_reports = []
    for name in ([phase] if phase else ["development", "confirmation"]):
        manifest_path = root / name / "manifest.json"
        if not manifest_path.exists():
            manifest_reports.append(dict(phase=name, complete=False, reason="missing_manifest"))
            continue
        manifest = _read_json(manifest_path)
        actual = sorted((root / name / "cells").glob("cell_*"))
        done = (root / name / "done").is_file() and (root / name / "done").read_text().strip() == "completed"
        source_matches = False
        if sources:
            source_matches = bool(manifest.get("sources")) and all(
                (source in source_report or Path(source).name in source_report)
                and source_report.get(source, source_report.get(Path(source).name, {})).get("exists")
                and source_report.get(source, source_report.get(Path(source).name, {})).get("sha256") == digest
                for source, digest in manifest.get("sources", {}).items()
            )
        configurations = manifest.get("cells", [])
        observed_ids = {p.name for p in actual}
        expected_ids = {f"cell_{c['cell_id']:04d}" for c in configurations}
        configuration_match = bool(configurations) and len(expected_ids) == len(configurations) and observed_ids == expected_ids
        if configuration_match:
            configuration_match = all(_read_json(root / name / "cells" / f"cell_{c['cell_id']:04d}" / "config.json") == c for c in configurations)
        if configurations and configurations[0].get("line") == "sensor":
            seeds = range(1100, 1102) if name == "development" else range(1200, 1220)
            setting_ids = {c["setting_id"] for c in configurations}
            expected_keys = set(itertools.product([2, 8], ["ambiguous", "regular"], seeds, setting_ids))
            observed_keys = {(c["dimension"], c["regime"], c["dataset_seed"], c["setting_id"]) for c in configurations}
            configuration_match = configuration_match and expected_keys == observed_keys and len(observed_keys) == len(configurations)
            configuration_match = configuration_match and len(configurations) == (96 if name == "development" else 480)
            configuration_match = configuration_match and all(c["groups"] == 12 for c in configurations)
        else:
            configuration_match = False
        manifest_reports.append({"phase": name, "manifest_cells": len(configurations),
                                "actual_cell_directories": len(actual),
                                "done_marker": done,
                                "source_matches": source_matches,
                                "configuration_match": configuration_match,
                                "complete": bool(done and len(actual) == len(manifest.get("cells", []))
                                               and source_matches and configuration_match)})
    matrix_reports = []
    matrix_cache = {}
    if not assets_only:
        for cell in [Path(item["cell"]) for item in cell_reports if "cell" in item]:
            config_path = cell / "config.json"
            if not config_path.exists():
                continue
            config = _read_json(config_path)
            if "steps" not in config or "dimension" not in config:
                continue
            if config["method"] == "unbiased":
                saved = _read_json(cell / "certificate.json")
                if saved["status"] != "not_certified":
                    raise ValueError("raw batch certificate must remain unclassified")
                continue
            asset_name = f"d{config['dimension']}_{config['regime']}_{config['dataset_seed']}"
            key = (asset_name, int(config["steps"]))
            if key not in matrix_cache:
                payload = _read_json(root / "assets" / asset_name / "problem.json")
                matrix_cache[key] = _matrix_certificate(payload, key[1])
            saved = _read_json(cell / "certificate.json")
            computed = matrix_cache[key]
            passed = saved["status"] == computed["status"] and np.isclose(saved["minimum_precision_eigenvalue"], computed["minimum_precision_eigenvalue"], atol=1e-9, rtol=1e-8)
            if saved["status"] == "finite":
                passed = passed and np.allclose(saved["final_tail_covariance"], computed["final_tail_covariance"], atol=1e-9, rtol=1e-8)
            matrix_reports.append({"asset": asset_name, "steps": key[1], "certificate": computed, "passed": bool(passed)})
    receipt_failures = sum(item["receipt"]["status"] != "ok" for item in asset_reports)
    receipt_failures += sum(item["receipt"]["status"] != "ok" for item in cell_reports)
    numerical_failures = sum(not item.get("pass", False) for item in asset_reports)
    numerical_failures += sum(not item.get("pass", False) for item in cell_reports)
    full_requested = not assets_only
    numerical_failures += sum(not item["passed"] for item in matrix_reports)
    status = "partial" if assets_only else ("failed" if not assets or not cell_reports or receipt_failures or numerical_failures or any(
        not item["complete"] for item in manifest_reports) else "completed")
    report = {
        "status": status,
        "auditor_sha256": _sha256(Path(__file__)),
        "runtime": dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
        "root": str(root),
        "mode": "assets_only" if assets_only else "full",
        "counts": {"assets": len(asset_reports), "cells": len(cell_reports),
                    "assets_exact_pass": sum(all(item["exact"][key] for key in
                                                ("probabilities", "means", "covariance", "weights_sum"))
                                             for item in asset_reports),
                    "receipts_checked": sum(item["receipt"]["checked"] for item in asset_reports),
                    "numerical_pass": len(asset_reports) + len(cell_reports) - numerical_failures,
                    "numerical_failures": numerical_failures},
        "assets": asset_reports,
        "cells": cell_reports,
        "manifests": manifest_reports,
        "sources": source_report,
        "matrix_certificates": matrix_reports,
        "audit_scope": {"full_requested": full_requested, "phase": phase},
        "timing_status": "uncontrolled_external_concurrency_exclude_speed_gate",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets-only", action="store_true")
    parser.add_argument("--phase", choices=["development", "confirmation"])
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    report = audit(args.root, args.output, assets_only=args.assets_only, phase=args.phase, sources=args.source, workers=args.workers)
    print(json.dumps(dict(status=report["status"], counts=report["counts"])), flush=True)
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
