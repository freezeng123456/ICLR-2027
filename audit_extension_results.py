import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson, trapezoid
from scipy.special import logsumexp
from scipy.stats import norm, wasserstein_distance

from audit_composition_references import parameters as oracle_parameters


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_arrays(variance, means, weights):
    grid = np.linspace(-16, 16, 131073)
    dimension = variance.shape[1]
    densities, cdfs = [], []
    for coordinate in range(dimension):
        blocks = []
        for start in range(0, len(grid), 2048):
            x = grid[start:start + 2048]
            components = norm.logpdf(x[:, None, None], means[None, :, coordinate, :], np.sqrt(variance[None, :, coordinate, None])) + np.log(weights[None, :, coordinate, :])
            blocks.append(logsumexp(components, axis=-1).sum(-1) - (len(variance) - 1) * norm.logpdf(x))
        log_density = np.concatenate(blocks)
        density = np.exp(log_density - log_density.max())
        density /= simpson(density, x=grid)
        assert max(density[0], density[-1]) < 1e-14
        cdf = cumulative_trapezoid(density, grid, initial=0)
        cdf /= cdf[-1]
        densities.append(density)
        cdfs.append(cdf)
    return dict(grid=grid, density=np.array(densities), cdf=np.array(cdfs))


def compare_reference(path, independent):
    discrepancies = []
    with np.load(path) as archive:
        grid, density, cdf = archive["grid"], archive["density"], archive["cdf"]
        for d in range(len(density)):
            cdf_error = np.max(abs(cdf[d] - np.interp(grid, independent["grid"], independent["cdf"][d])))
            moments = [abs(simpson(grid ** power * density[d], x=grid) - simpson(independent["grid"] ** power * independent["density"][d], x=independent["grid"])) for power in [1, 2, 4]]
            assert cdf_error < 3e-6 and max(moments) < 1e-7, (path, d, cdf_error, moments)
            discrepancies.append({"coordinate": d, "cdf_discrepancy": float(cdf_error), "moment_discrepancy": max(moments)})
    return discrepancies


def true_parameters(context):
    y, sigma, offset = [context[..., i] for i in range(3)]
    variance = 1 / (1 + sigma ** -2)
    means = variance[..., None] * np.stack((y - offset, y + offset), axis=-1) / sigma[..., None] ** 2
    evidence = norm.logpdf(y[..., None], loc=np.stack((offset, -offset), axis=-1), scale=np.sqrt(1 + sigma[..., None] ** 2))
    weights = np.exp(evidence - logsumexp(evidence, axis=-1, keepdims=True))
    return variance, means, weights


def independent_certificate(variance, config):
    levels = np.linspace(math.sqrt(config["u_max"]), 0, config["steps"] + 1) ** 2
    full = config["method"] in ["full", "tail"]
    groups, batch = config["groups"], config["batch"]
    rows = []
    for coordinate in range(config["dimension"]):
        current_variance = 1.0
        minimum = 1.0
        result = None
        for step, u in enumerate(levels[:-1]):
            h = u - levels[step + 1]
            alpha2 = math.exp(-u)
            a = np.sort((1 - alpha2 + alpha2 * variance[:, coordinate]) ** -1 - 1)
            if full:
                candidates = [(a.sum(), np.tril(np.outer(a, a), k=-1).sum())]
            elif config["sampling"] == "with_replacement":
                candidates = [(groups * value, groups * (groups - 1) / 2 * value ** 2) for value in [a[0], a[-1]]]
            else:
                candidates = []
                for k in range(batch + 1):
                    subset = np.concatenate((a[:k], a[groups - batch + k:]))
                    pair = sum(subset[i] * subset[j] for i in range(batch) for j in range(i))
                    candidates.append((groups / batch * subset.sum(), groups * (groups - 1) / (batch * (batch - 1)) * pair))
            denominators = np.array([1 - 2 * h * coefficient * current_variance for _, coefficient in candidates])
            minimum = min(minimum, float(denominators.min()))
            if minimum <= 0:
                assert minimum < -1e-10, (config, minimum)
                result = {"finite_normalizer": False, "failure_step": step + 1, "minimum_denominator": minimum}
                break
            multiplier = np.array([1 - h * ((1 + config["diffusion"]) * total + config["diffusion"]) / 2 for total, _ in candidates])
            current_variance = max(multiplier ** 2 * current_variance / denominators + config["diffusion"] * h)
        if result is None:
            result = {"finite_normalizer": True, "minimum_denominator": minimum, "maximum_terminal_component_variance": current_variance}
        rows.append(result)
    return {"finite_normalizer": all(row["finite_normalizer"] for row in rows), "coordinates": rows}


def independent_metrics(samples, weights, reference):
    assert np.isfinite(samples).all() and np.isfinite(weights).all()
    assert np.min(weights) >= 0 and abs(weights.sum() - 1) < 1e-11
    grid, density, cdf = [reference[key] for key in ["grid", "density", "cdf"]]
    metrics = []
    for d in range(samples.shape[1]):
        target_weights = density[d].copy()
        target_weights[[0, -1]] *= 0.5
        target_weights /= target_weights.sum()
        target_mean = simpson(grid * density[d], x=grid)
        target_variance = simpson((grid - target_mean) ** 2 * density[d], x=grid)
        mean = np.average(samples[:, d], weights=weights)
        variance = np.average((samples[:, d] - mean) ** 2, weights=weights)
        unique, inverse = np.unique(samples[:, d], return_inverse=True)
        mass = np.bincount(inverse, weights=weights)
        right = np.cumsum(mass)
        truth = np.interp(unique, grid, cdf[d])
        ks = max(np.max(abs(right - truth)), np.max(abs(right - mass - truth)))
        metrics.append([wasserstein_distance(samples[:, d], grid, u_weights=weights, v_weights=target_weights), ks, abs(mean - target_mean), abs(variance / target_variance - 1), abs(np.average(samples[:, d] > 0, weights=weights) - (1 - np.interp(0, grid, cdf[d])))])
    values = np.mean(metrics, axis=0)
    names = ["w1_mean", "ks_mean", "mean_absolute_error", "variance_relative_error", "positive_mass_error"]
    return dict(zip(names, map(float, values)))


def audit(root, output, kind):
    manifest = json.loads((root / kind / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    references, certificates, true_references = {}, {}, {}
    rows, reference_report, discrepancies = [], {}, {}
    for cell in sorted((root / kind).glob("task_*/cell_*")):
        config = json.loads((cell / "config.json").read_text())
        summary = json.loads((cell / "summary.json").read_text())
        expected = manifest["cells"][config["cell_id"]]
        assert all(config[key] == value for key, value in expected.items())
        assert config["commit"] == manifest["commit"] and summary["status"] == "completed"
        assert (cell / "done").is_file() and (cell / "extension_done").is_file()
        assert config["runtime"]["gpu"] == "NVIDIA GeForce RTX 3080"
        assert len(config["runtime"]["cuda_visible_devices"].split(",")) == 1
        assert config["runtime"]["slurm_cpus_per_task"] == "4"
        assert config["runtime"]["source_sha256"] == manifest["source_hashes"]["composition_benchmark.py"]
        reference_key = (kind, config["family"], config["groups"], config["dimension"], config.get("training_seed"), config.get("dataset_seed"))
        if kind == "oracle":
            variance, means, weights = oracle_parameters(config["groups"], config["dimension"], config["family"])
        else:
            checkpoint = root / "training" / f"training_{config['training_seed']}" / "final.pt"
            assert sha256(checkpoint) == config["checkpoint_sha256"]
            with np.load(cell / "learned_parameters.npz") as archive:
                variance, means, weights, context = [archive[key] for key in ["variance", "means", "weights", "context"]]
            assert np.all(variance > 0) and np.all(variance < 1)
            assert np.all(weights > 0)
            np.testing.assert_allclose(weights.sum(-1), 1, atol=1e-12)
        if reference_key not in references:
            independent = reference_arrays(variance, means, weights)
            reference_report[str(reference_key)] = compare_reference(cell / "reference.npz", independent)
            references[reference_key] = {"learned": independent}
            if kind == "learned":
                true_key = (config["groups"], config["dimension"], config["dataset_seed"])
                if true_key not in true_references:
                    true_references[true_key] = reference_arrays(*true_parameters(context))
                independent_true = true_references[true_key]
                reference_report[str(reference_key) + "_true"] = compare_reference(cell / "true_reference.npz", independent_true)
                references[reference_key]["true"] = independent_true
        reference = references[reference_key]["learned"]
        full_certificate = config["method"] in ["full", "tail"]
        cert_key = reference_key + (config["steps"], "full" if full_certificate else config["sampling"], 0 if full_certificate else config["batch"], config["u_max"])
        if cert_key not in certificates:
            certificates[cert_key] = independent_certificate(variance, config)
        certificate = certificates[cert_key]
        stored_certificate = json.loads((cell / "certificate.json").read_text())
        assert certificate["finite_normalizer"] == stored_certificate["finite_normalizer"]
        for actual, stored in zip(certificate["coordinates"], stored_certificate["coordinates"]):
            assert actual["finite_normalizer"] == stored["finite_normalizer"]
            if not actual["finite_normalizer"]:
                assert actual["failure_step"] == stored["failure_step"]
        with np.load(cell / "samples.npz") as archive:
            samples, particle_weights = archive["samples"], archive["weights"]
        assert samples.shape == (config["particles"], config["dimension"])
        metrics = independent_metrics(samples, particle_weights, reference)
        for key, value in metrics.items():
            difference = abs(value - summary[key])
            discrepancies[key] = max(discrepancies.get(key, 0), difference)
            tolerance = 0.0007 if key == "w1_mean" else 1e-5
            assert difference < tolerance, (cell, key, difference)
        result = {key: value for key, value in summary.items() if key != "runtime"}
        result.update({"integrability_certificate": "finite" if certificate["finite_normalizer"] else "infinite", "relative_path": str(cell.relative_to(root))})
        if kind == "learned":
            true_metrics = independent_metrics(samples, particle_weights, references[reference_key]["true"])
            stored_true = json.loads((cell / "true_metrics.json").read_text())
            for key, value in true_metrics.items():
                assert abs(value - stored_true[key]) < (0.0007 if key == "w1_mean" else 1e-5)
                result["true_" + key] = value
            independent_true = references[reference_key]["true"]
            independent_kl = float(simpson(independent_true["density"] * (np.log(np.maximum(independent_true["density"], 1e-300)) - np.log(np.maximum(reference["density"], 1e-300))), x=reference["grid"], axis=1).mean())
            independent_w1 = float(trapezoid(abs(independent_true["cdf"] - reference["cdf"]), reference["grid"], axis=1).mean())
            for key, value in [("composed_true_to_learned_kl_mean", independent_kl), ("composed_true_to_learned_w1_mean", independent_w1)]:
                assert abs(value - stored_true[key]) < 1e-5
                result[key] = value
        rows.append(result)
        if len(rows) % 25 == 0:
            print(json.dumps({"kind": kind, "audited_cells": len(rows)}), flush=True)
    assert sorted(row["cell_id"] for row in rows) == list(range(len(manifest["cells"])))
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (output / f"{kind}_audited.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / f"{kind}_reference_audit.json").write_text(json.dumps(reference_report, indent=2))
    report = {"status": "passed", "kind": kind, "verified_cells": len(rows), "unique_references": len(references), "metric_maximum_discrepancy": discrepancies,
              "scope": "Independent SciPy metrics, doubled/widened quadrature, independent Gaussian variance recursion, configs and raw sample checks; scheduler and recovery hashes audited separately"}
    (output / f"{kind}_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=["oracle", "learned"], required=True)
    args = parser.parse_args()
    audit(args.root, args.output, args.kind)
