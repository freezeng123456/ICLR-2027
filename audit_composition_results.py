import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.integrate import simpson
from scipy.stats import wasserstein_distance

from gaussian_integrability import audit_path


def sha256(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def independent_metrics(cell):
    with np.load(cell / "samples.npz") as archive:
        samples, weights = archive["samples"], archive["weights"]
    with np.load(cell / "reference.npz") as archive:
        grid, density, cdf = archive["grid"], archive["density"], archive["cdf"]
    assert np.isfinite(samples).all() and np.isfinite(weights).all()
    assert np.all(weights >= 0) and abs(weights.sum() - 1) < 1e-12
    assert np.max(np.abs(samples)) < grid[-1]
    rows = []
    for coordinate in range(samples.shape[1]):
        x = samples[:, coordinate]
        p = density[coordinate]
        target_mean = simpson(grid * p, x=grid)
        target_variance = simpson((grid - target_mean) ** 2 * p, x=grid)
        target_weights = p.copy()
        target_weights[[0, -1]] *= 0.5
        target_weights /= target_weights.sum()
        # SciPy 独立计算离散参考分布与粒子分布之间的 Wasserstein 距离。
        w1 = wasserstein_distance(x, grid, u_weights=weights, v_weights=target_weights)
        mean = np.average(x, weights=weights)
        variance = np.average((x - mean) ** 2, weights=weights)
        unique, inverse = np.unique(x, return_inverse=True)
        mass = np.bincount(inverse, weights=weights)
        right = np.cumsum(mass)
        truth = np.interp(unique, grid, cdf[coordinate])
        ks = max(np.max(np.abs(right - truth)), np.max(np.abs(right - mass - truth)))
        sign_error = abs(np.average(x > 0, weights=weights) - simpson(p[grid >= 0], x=grid[grid >= 0]))
        rows.append([w1, ks, abs(mean - target_mean), abs(variance / target_variance - 1), sign_error])
    values = np.mean(rows, axis=0)
    names = ["w1_mean", "ks_mean", "mean_absolute_error", "variance_relative_error", "positive_mass_error"]
    dimension = samples.shape[1]
    sign_code = (samples > 0).astype(np.int64) @ (2 ** np.arange(dimension))
    empirical_sign_mass = np.bincount(sign_code, weights=weights, minlength=2 ** dimension)
    positive_probability = np.array([1 - np.interp(0, grid, item) for item in cdf])
    bits = ((np.arange(2 ** dimension)[:, None] >> np.arange(dimension)) & 1).astype(bool)
    reference_sign_mass = np.where(bits, positive_probability, 1 - positive_probability).prod(1)
    joint_sign_tv = np.abs(empirical_sign_mass - reference_sign_mass).sum() / 2
    centered = samples - np.average(samples, axis=0, weights=weights)
    empirical_variance = np.average(centered ** 2, axis=0, weights=weights)
    assert np.all(empirical_variance > 0)
    standardized = centered / np.sqrt(empirical_variance)
    correlation = standardized.T @ (weights[:, None] * standardized)
    off_diagonal = correlation[~np.eye(dimension, dtype=bool)]
    joint = {"joint_sign_total_variation": float(joint_sign_tv), "off_diagonal_correlation_rms": float(np.sqrt(np.mean(off_diagonal ** 2))) if dimension > 1 else 0.0}
    return dict(zip(names, values)), float(np.max(np.diff(grid))), joint


def certificate(config):
    groups, dimension = config["groups"], config["dimension"]
    phase = 2 * np.pi * (np.arange(groups)[:, None] + 0.37 * np.arange(dimension)[None, :]) / groups
    if config["family"] == "weak_mixture":
        strength = (4 + 2 * np.sin(phase)) / groups
    else:
        strength = 1 / (0.5 + 0.15 * np.sin(phase)) - 1
    control_variance = 1 / (1 + strength)
    if config["family"] != "gaussian":
        positive_weight = 0.5 + 0.1 * np.sin(phase + 0.7)
        separation = (0.6 + 0.1 * np.cos(phase)) / np.sqrt(groups) if config["family"] == "weak_mixture" else 0.7 + 0.2 * np.cos(2 * phase)
        control_variance += 4 * positive_weight * (1 - positive_weight) * separation ** 2
    results = {}
    for method in ("full", "unbiased", "cv"):
        coordinates = [audit_path(strength[:, d], config["steps"], config["batch"], method, diffusion=config["diffusion"], maximum_u=config["u_max"], control_variance=control_variance[:, d] if method == "cv" else None) for d in range(dimension)]
        assert all(row["finite_normalizer"] or row["minimum_denominator"] < -1e-10 for row in coordinates)
        results[method] = {"strictly_finite": all(row["finite_normalizer"] for row in coordinates), "coordinates": coordinates}
    return results


def one_step_diagnostics(cell, config, summary):
    with np.load(cell / "diagnostics.npz") as archive:
        assert all(np.isfinite(archive[name]).all() for name in archive.files)
        edges, mass = archive["histogram_edges"], archive["histogram_mass"]
        x, weights = archive["largest_weight_x"], archive["largest_weights"]
    assert abs(mass.sum() - 1) < 1e-10 and np.min(mass) >= -1e-14
    midpoint = (edges[:-1] + edges[1:]) / 2
    radius = np.diff(edges) / 2
    assert abs(np.sum(mass * midpoint) - summary["weighted_mean"]) <= np.max(radius) + 1e-10
    second_moment = summary["weighted_variance"] + summary["weighted_mean"] ** 2
    tolerance = np.sum(mass * (2 * np.abs(midpoint) * radius + radius ** 2))
    assert abs(np.sum(mass * midpoint ** 2) - second_moment) <= tolerance + 1e-10
    assert abs(weights.max() - summary["largest_normalized_weight"]) < 1e-14
    if config["method"] == "full":
        c = 346 / 693
        reconstructed = np.exp(0.5 * c * x ** 2 - summary["log_normalizer"]) / config["particles"]
        np.testing.assert_allclose(weights, reconstructed, rtol=2e-12, atol=1e-15)
    else:
        a = np.array([1 / 21, 1 / 11, 1 / 2, 2 / 3])
        batch = config["batch"]
        values = a[np.array(list(itertools.combinations_with_replacement(range(4), batch)))]
        summed, squared = values.sum(1), (values * values).sum(1)
        possible_c = np.sort(0.5 * (16 * (summed ** 2 - squared) / (batch * (batch - 1)) - 4 * squared / batch))
        selected = np.abs(x) > 1e-3
        inferred = 2 * (np.log(config["particles"] * weights[selected]) + summary["log_normalizer"]) / x[selected] ** 2
        upper = np.clip(np.searchsorted(possible_c, inferred), 0, len(possible_c) - 1)
        lower = np.maximum(upper - 1, 0)
        errors = np.minimum(abs(possible_c[upper] - inferred), abs(possible_c[lower] - inferred))
        assert errors.max() < 1e-7


def audit(root, output):
    manifest = json.loads((root / "manifest.json").read_text())
    recovery = json.loads((root / "recovery_manifest.json").read_text())
    assert manifest["commit"] == recovery["source_commit"]
    benchmark_source = subprocess.run(["git", "show", f"{manifest['commit']}:composition_benchmark.py"], check=True, capture_output=True).stdout
    source_digest = hashlib.sha256(benchmark_source).hexdigest()
    for record in recovery["files"]:
        path = root / record["path"]
        assert path.is_file() and path.stat().st_size == record["bytes"]
        assert sha256(path) == record["sha256"], path
    schedule = list(csv.DictReader((root / "slurm.tsv").open(), delimiter="|"))
    assert len(schedule) == manifest["tasks"], len(schedule)
    assert all(row["State"] == "COMPLETED" and row["ExitCode"] == "0:0" for row in schedule)
    rows, certificates = [], {}
    maxima = {}
    for cell in sorted(root.glob("task_*/cell_*")):
        config = json.loads((cell / "config.json").read_text())
        expected = manifest["cells"][config["cell_id"]]
        assert all(config[key] == value for key, value in expected.items())
        assert config["commit"] == manifest["commit"] and (cell / "done").is_file()
        summary = json.loads((cell / "summary.json").read_text())
        assert summary["status"] == "completed"
        assert len(config["runtime"]["cuda_visible_devices"].split(",")) == 1
        assert config["runtime"]["source_sha256"] == source_digest
        if config.get("kind") == "one_step":
            one_step_diagnostics(cell, config, summary)
            summary["independent_metric_scope"] = "histogram moment bounds and largest-weight reconstruction; raw full particle sample not retained"
        else:
            independent, dx, joint = independent_metrics(cell)
            summary.update(joint)
            for name, value in independent.items():
                delta = abs(value - summary[name])
                maxima[name] = max(maxima.get(name, 0), delta)
                tolerance = dx if name == "w1_mean" else 2e-6
                assert delta <= tolerance, (cell, name, delta, tolerance)
                summary[f"independent_{name}"] = float(value)
            key = f"{config['family']}_G{config['groups']}_d{config['dimension']}_K{config['steps']}"
            if key not in certificates:
                certificates[key] = certificate(config)
            method = config["method"]
            certificate_method = {"full": "full", "tail": "full", "tail_cumulant": "full", "naive": "unbiased", "unbiased": "unbiased", "cumulant": "unbiased", "cv": "cv", "cv_cumulant": "cv"}
            if method == "unweighted":
                summary["integrability_certificate"] = "finite"
            else:
                summary["integrability_certificate"] = "finite" if certificates[key][certificate_method[method]]["strictly_finite"] else "infinite"
        rows.append({key: value for key, value in summary.items() if key != "runtime"})
    assert sorted(row["cell_id"] for row in rows) == list(range(len(manifest["cells"])))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "audited_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(row.keys() for row in rows))))
        writer.writeheader()
        writer.writerows(rows)
    (output / "integrability_certificates.json").write_text(json.dumps(certificates, indent=2))
    report = {"status": "passed", "source_commit": manifest["commit"], "verified_files": len(recovery["files"]), "verified_cells": len(rows), "verified_slurm_tasks": len(schedule), "independent_metric_maximum_discrepancy": maxima, "w1_tolerance": "one reference-grid interval; independent SciPy discrete-measure W1", "other_metric_tolerance": 2e-6, "passed": ["all file hashes", "all configs and completion markers", "Slurm completed and exit zero", "one GPU visible per cell", "raw particle metrics via independent numerical routines"], "failed": [], "blocked": [], "not_run": ["learned neural score benchmarks", "general non-diagonal covariance integrability", "full raw one-step metric recomputation when only diagnostics retained"]}
    (output / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    audit(arguments.root, arguments.output)
