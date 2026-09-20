import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import re

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from scipy.integrate import simpson
from scipy.special import logsumexp, ndtr
from scipy.stats import multivariate_normal, norm, wasserstein_distance


FIELDS = ("directions", "variance", "means", "weights")
SOURCES = ("learned_factor_bridge_20260921.py", "learned_factor_reference_20260921.py",
           "run_learned_factor_study_20260921.py", "tail_bridge_smc_20260921.py", "learned_sbi.py",
           "docs/LEARNED_FACTOR_STUDY_PROTOCOL_20260921.md")
BANK_HASHES = (
    "ce9344dce5f56b4cda176bcdb36371f48911618aa600da1465617419b4995b69",
    "642c9ac8b37cc879a46c5dc332c9d99eb4ae9d0bd2c4b38a1ce3bbe1af22951a",
    "4ad56f1f2de8ac7e99a997dc798fcc9a82d5e41bcaa01483fa8161ba24fa2737",
    "40d2f60f7b01b721909df18cdd26cc0bda1e28ef62dd47bb3bc9b7acfaa1fbc4",
    "4f8f7d024dee2ba1d205fcf59f39c4f32d87524ab703a9051d4ad4e7ca38f59f")
ASSET_FILES = {"parameters.npz", "true_parameters.npz", "observations.npz", "learned_reference.npz",
               "true_reference.npz", "projections.npz", "summary.json"}
CELL_FILES = {"config.json", "samples.npz", "diagnostics.json", "gpu_processes.json", "summary.json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    require(all(not np.iscomplexobj(v) and np.isfinite(v).all() for v in result.values()),
            f"nonfinite or complex array: {path}")
    return result


def close(actual, expected, label, atol=1e-10, rtol=1e-10):
    require(np.shape(actual) == np.shape(expected) and np.isfinite(actual).all()
            and np.isfinite(expected).all() and np.allclose(actual, expected, atol=atol, rtol=rtol),
            f"numerical mismatch: {label}")
    return float(np.max(np.abs(np.asarray(actual) - expected)))


def verify_receipt(folder, expected):
    folder = Path(folder)
    receipt = read_json(folder / "receipt.json")
    require(set(receipt) == {"files"} and set(receipt["files"]) == set(expected), f"receipt file set: {folder}")
    require({p.name for p in folder.iterdir()} == set(expected) | {"receipt.json"}
            | ({"done"} if "config.json" in expected else set()), f"unexpected files: {folder}")
    for name, digest in receipt["files"].items():
        require(not (folder / name).is_symlink() and sha256(folder / name) == digest,
                f"receipt hash mismatch: {folder / name}")
    return sha256(folder / "receipt.json")


def configurations():
    rows = [dict(setting_id=f"prior_s{s:g}", reference="prior", proposal_scale=s,
                 global_probability=0., direct_is=False, role="baseline") for s in (.15, .35, .65)]
    for name, reference, refresh, direct in (("gaussian", "gaussian", 0., False),
            ("mixture", "mixture", 0., False), ("mixture_direct_is", "mixture", 0., True),
            ("mixture_global", "mixture", .1, False), ("exact_enumeration", "exact", 0., False)):
        rows.append(dict(setting_id=name, reference=reference, proposal_scale=.5,
                         global_probability=refresh, direct_is=direct,
                         role="candidate" if name == "mixture" else "diagnostic"))
    return rows


def expected_cells(phase, baseline=None):
    require(phase in ("development", "confirmation"), "invalid phase")
    settings = configurations()
    if phase == "confirmation":
        require(baseline in {row["setting_id"] for row in settings if row["role"] == "baseline"}, "invalid baseline")
        settings = [row for row in settings if row["role"] != "baseline" or row["setting_id"] == baseline]
    seeds = range(1600, 1604) if phase == "development" else range(1700, 1712)
    cells = [dict(setting, dataset_seed=seed, repeat=repeat, cell_id=index, particles=4096, phase=phase)
             for index, (seed, setting, repeat) in enumerate(itertools.product(seeds, settings, range(3)))]
    return settings, cells


def validate_manifest(root, source, phase, baseline=None):
    folder = Path(root) / phase
    manifest = read_json(folder / "manifest.json")
    settings, cells = expected_cells(phase, baseline)
    require(manifest["settings"] == settings and manifest["cells"] == cells, "frozen rectangular manifest mismatch")
    order = np.random.default_rng(20261022 if phase == "development" else 20261023).permutation(len(cells)).tolist()
    require(manifest["order"] == order, "execution order mismatch")
    require(re.fullmatch(r"[0-9a-f]{40}", manifest["source_commit"]) is not None, "source commit format")
    require(set(manifest["sources"]) == set(SOURCES), "source file set")
    for name, digest in manifest["sources"].items():
        require(sha256(Path(source) / name) == digest, f"source hash mismatch: {name}")
    bank = manifest["factor_bank"]
    require(bank["checkpoints"] == [dict(training_seed=i, sha256=digest, strict_reload=True)
                                   for i, digest in enumerate(BANK_HASHES)], "frozen checkpoint receipt mismatch")
    require(np.isfinite(bank["load_seconds"]) and bank["load_seconds"] > 0, "bank loading time")
    require((folder / "done").read_text().strip() == "completed", "phase done marker")
    state = read_json(folder / "state.json")
    require(state["status"] == "completed" and state["cells"] == len(cells), "phase completion state")
    require({p.name for p in (folder / "cells").iterdir()} == {f"cell_{c['cell_id']:04d}" for c in cells},
            "cell directory set")
    rows = read_json(folder / "rows.json")
    require(len(rows) == len(cells) and len({r["cell_id"] for r in rows}) == len(cells), "aggregate row uniqueness")
    by_id = {row["cell_id"]: row for row in rows}
    for cell in cells:
        cell_folder = folder / "cells" / f"cell_{cell['cell_id']:04d}"
        verify_receipt(cell_folder, CELL_FILES)
        require((cell_folder / "done").read_text().strip() == "completed", "cell done marker")
        require(by_id[cell["cell_id"]] == read_json(cell_folder / "summary.json"), "rows versus cell summary")
        require(all(by_id[cell["cell_id"]][key] == value for key, value in cell.items()), "summary configuration mismatch")
        require(by_id[cell["cell_id"]]["status"] == "completed", "cell status")
    close(state["sampler_seconds"], sum(row["sampling_seconds"] for row in rows), "phase total sampling time", atol=1e-8)
    require(np.isfinite(state["seconds"]) and state["seconds"] > 0, "phase time")
    return manifest, rows


def development_selection(rows):
    _, cells = expected_cells("development")
    require(len(rows) == 96 and {(r["dataset_seed"], r["setting_id"], r["repeat"]) for r in rows}
            == {(r["dataset_seed"], r["setting_id"], r["repeat"]) for r in cells}, "selection cohort mismatch")
    means = {s["setting_id"]: {key: float(np.mean([r[key] for r in rows if r["setting_id"] == s["setting_id"]]))
             for key in ("sliced_w1_32", "seconds")} for s in configurations()}
    require(all(np.isfinite(v) and v >= 0 for row in means.values() for v in row.values()), "selection metrics")
    priors = [s["setting_id"] for s in configurations() if s["role"] == "baseline"]
    best = min(means[name]["sliced_w1_32"] for name in priors)
    baseline = min((name for name in priors if means[name]["sliced_w1_32"] <= best + .001),
                   key=lambda name: (means[name]["seconds"], name))
    passed = (means["mixture"]["sliced_w1_32"] <= means[baseline]["sliced_w1_32"] + .002
              and means["mixture"]["seconds"] < .8 * means[baseline]["seconds"])
    return dict(baseline=baseline, candidate="mixture", means=means, expand_confirmation=passed,
                development_only=True, comparison_scope="against prior-SMC; exact enumeration also reported")


def validate_parameters(parameters):
    require(set(parameters) == set(FIELDS), "sampler input must contain only four learned parameter arrays")
    a, v, m, w = [np.asarray(parameters[key], dtype=float) for key in FIELDS]
    require(a.shape == (5, 2) and v.shape == (5,) and m.shape == w.shape == (5, 2), "factor parameter shapes")
    require(all(np.isfinite(value).all() for value in (a, v, m, w)), "nonfinite parameters")
    close(np.linalg.norm(a, axis=1), np.ones(5), "factor directions", atol=1e-12, rtol=0)
    require(np.all((v > 0) & (v < 1)) and np.all(w > 0), "factor variance or weight domain")
    close(w.sum(1), np.ones(5), "factor weight sums", atol=1e-12, rtol=0)


def input_hash(parameters):
    validate_parameters(parameters)
    digest = hashlib.sha256()
    for key in FIELDS:
        array = np.asarray(parameters[key])
        digest.update(key.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.astype("<f8").tobytes())
    return digest.hexdigest()


def direct_log_product(parameters, points):
    points = np.asarray(points)
    result = norm.logpdf(points).sum(axis=1)
    for direction, variance, means, weights in zip(*(parameters[key] for key in FIELDS)):
        projection = points @ direction
        result += logsumexp(norm.logpdf(projection[:, None], means, np.sqrt(variance))
                            + np.log(weights), axis=1) - norm.logpdf(projection)
    return result


def validate_reference(parameters, exact):
    validate_parameters(parameters)
    require(set(exact) == {"probabilities", "means", "covariance", "log_normalizer"}, "reference fields")
    probability, means, covariance = (exact[key] for key in ("probabilities", "means", "covariance"))
    require(probability.shape == (32,) and means.shape == (32, 2) and covariance.shape == (2, 2), "reference shapes")
    require(all(np.isfinite(v).all() for v in exact.values()) and np.all(probability >= 0), "reference finite values")
    close(probability.sum(), 1., "reference probabilities", atol=1e-12, rtol=0)
    close(covariance, covariance.T, "reference symmetry", atol=1e-12, rtol=0)
    np.linalg.cholesky(covariance)
    require(np.shape(exact["log_normalizer"]) == (), "reference normalizer scalar")
    sigma = np.sqrt(np.diag(covariance))
    lower, upper = means.min(0) - 10 * sigma, means.max(0) + 10 * sigma
    axes = [np.linspace(lower[i], upper[i], 513) for i in range(2)]
    xx, yy = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    logdensity = np.concatenate([direct_log_product(parameters, points[start:start + 8192])
                                 for start in range(0, len(points), 8192)])
    density = np.exp(logdensity - float(exact["log_normalizer"])).reshape(513, 513)
    integral = float(simpson(simpson(density, x=axes[1], axis=1), x=axes[0]))
    require(np.isfinite(integral) and abs(integral - 1) <= 1e-8, "2D grid normalization error exceeds 1e-8")
    probes = np.vstack((means, np.zeros((1, 2)), points[::521]))
    positive = probability > 0
    mixture_log = logsumexp(np.column_stack([np.log(weight) + multivariate_normal.logpdf(probes, mean, covariance)
                            for weight, mean in zip(probability[positive], means[positive])]), axis=1)
    discrepancy = close(mixture_log + exact["log_normalizer"], direct_log_product(parameters, probes),
                        "direct prior-corrected density", atol=2e-10, rtol=0)
    outside = float(probability @ (norm.cdf(lower, means, sigma) + norm.sf(upper, means, sigma)).sum(1))
    return dict(normalization_error=abs(integral - 1), numerical_log_normalizer=float(exact["log_normalizer"]) + np.log(integral),
                pointwise_log_density_max_error=discrepancy, grid_shape=[513, 513], lower=lower.tolist(), upper=upper.tolist(),
                gaussian_tail_mass_union_bound=outside,
                qualification="Numerical quadrature check; no rigorous floating-point integration certificate")


def projection_grids(exact, directions):
    require(directions.shape == (32, 2), "32 projection directions required")
    angle = np.arange(32) * np.pi / 32
    close(directions, np.column_stack((np.cos(angle), np.sin(angle))), "evaluation projections", atol=1e-12, rtol=0)
    grids = []
    for direction in directions:
        means = exact["means"] @ direction
        sigma = float(np.sqrt(direction @ exact["covariance"] @ direction))
        nodes = np.linspace(means.min() - 10 * sigma, means.max() + 10 * sigma, 32769)
        midpoint = (nodes[:-1] + nodes[1:]) / 2
        probability = exact["probabilities"]
        cdf = ndtr((midpoint[:, None] - means) / sigma) @ probability
        masses = np.diff(np.concatenate(([0.], cdf, [1.])))
        require(np.min(masses) >= -2e-15, "nonmonotone Gaussian CDF")
        masses = np.maximum(masses, 0.)
        masses /= masses.sum()
        left = (nodes[0] - means) / sigma
        right = (means - nodes[-1]) / sigma
        tails = float(probability @ (sigma * (norm.pdf(left) + left * ndtr(left))
                                    + sigma * (norm.pdf(right) + right * ndtr(right))))
        spacing = float(nodes[1] - nodes[0])
        bound = .5 * spacing + tails + 1e-9
        grids.append((nodes, masses, bound, tails, spacing))
    return grids


def validate_samples(samples, weights, particles):
    require(samples.shape == (particles, 2) and weights.shape == (particles,), "sample/weight shape mismatch")
    require(np.isfinite(samples).all() and np.isfinite(weights).all() and np.all(weights >= 0), "invalid samples/weights")
    close(weights.sum(), 1., "sample weight sum", atol=1e-12, rtol=0)


def independent_metrics(samples, weights, exact, directions, grids):
    validate_samples(samples, weights, len(samples))
    target_mean = np.sum(exact["probabilities"][:, None] * exact["means"], axis=0)
    target_delta = exact["means"] - target_mean
    target_covariance = exact["covariance"] + np.einsum("i,ij,ik->jk", exact["probabilities"], target_delta, target_delta)
    mean = np.sum(weights[:, None] * samples, axis=0)
    delta = samples - mean
    covariance = np.einsum("i,ij,ik->jk", weights, delta, delta)
    values = [float(wasserstein_distance(samples @ direction, grid[0], weights, grid[1]))
              for direction, grid in zip(directions, grids)]
    return dict(mean_error=float(np.linalg.norm(mean - target_mean) / np.sqrt(2)),
                covariance_error=float(np.linalg.norm(covariance - target_covariance) / 2),
                grid_sliced_w1_32=float(np.mean(values)), projected_grid_w1=values,
                w1_absolute_discrepancy_bound=float(np.mean([grid[2] for grid in grids])),
                per_projection_discrepancy_bounds=[grid[2] for grid in grids],
                ess_fraction=float(1 / (weights @ weights) / len(weights)))


def prepare_asset(folder, seed):
    receipt_hash = verify_receipt(folder, ASSET_FILES)
    parameters = load_npz(folder / "parameters.npz")
    true_parameters = load_npz(folder / "true_parameters.npz")
    observed = load_npz(folder / "observations.npz")
    require(set(observed) == {"truth", "context", "generated_sign"}, "observation fields")
    angle = (np.arange(5) + .25) * np.pi / 5
    directions = np.column_stack((np.cos(angle), np.sin(angle)))
    sigma, offset = .55 + .1 * np.arange(5), .4 + .15 * np.arange(5)
    rng = np.random.default_rng(920000 + seed)
    truth = rng.normal(size=2)
    sign = 2 * rng.integers(0, 2, size=5) - 1
    observed_y = directions @ truth + sign * offset + sigma * rng.normal(size=5)
    context = np.column_stack((observed_y, sigma, offset))
    for key, expected in (("truth", truth), ("context", context), ("generated_sign", sign)):
        close(observed[key], expected, f"dataset reconstruction {key}", atol=1e-12, rtol=0)
    for payload in (parameters, true_parameters):
        close(payload["directions"], directions, "dataset factor directions", atol=1e-12, rtol=0)
    true_means = np.column_stack((observed_y - offset, observed_y + offset)) / (1 + sigma ** 2)[:, None]
    true_weights = norm.pdf(np.column_stack((observed_y - offset, observed_y + offset)), scale=np.sqrt(1 + sigma ** 2)[:, None])
    true_weights /= true_weights.sum(1, keepdims=True)
    close(true_parameters["means"], true_means, "true factor means")
    close(true_parameters["variance"], sigma ** 2 / (1 + sigma ** 2), "true factor variance")
    close(true_parameters["weights"], true_weights, "true factor probabilities")
    exact = load_npz(folder / "learned_reference.npz")
    true = load_npz(folder / "true_reference.npz")
    references = dict(learned=validate_reference(parameters, exact), true=validate_reference(true_parameters, true))
    projections = load_npz(folder / "projections.npz")
    require(set(projections) == {"directions"}, "projection fields")
    grids = projection_grids(exact, projections["directions"])
    summary = read_json(folder / "summary.json")
    require(summary["seed"] == seed and summary["network_calls"] == 5 and summary["exact_components"] == 32, "asset metadata")
    require(np.isfinite(summary["prediction_seconds"]) and summary["prediction_seconds"] > 0, "prediction time")
    require(np.isfinite(summary["learned_to_true_projected_w1"]) and summary["learned_to_true_projected_w1"] >= 0, "model error value")
    report = dict(seed=seed, receipt_sha256=receipt_hash, parameter_sha256=input_hash(parameters), references=references,
                  model_error=dict(saved=summary["learned_to_true_projected_w1"], coverage="receipt checked; quadrature not recomputed"),
                  maximum_projection_spacing=max(g[4] for g in grids), maximum_tail_expectation=max(g[3] for g in grids))
    return dict(parameters=parameters, exact=exact, directions=projections["directions"], grids=grids, summary=summary, report=report)


def gpu_pid(record, required):
    before, after = record["before"], record["after"]
    require(set(record) == {"before", "after"}, "GPU record fields")
    if not before and not after:
        require(not required, "missing GPU process evidence")
        return None
    require(len(before) == len(after) == 1, "GPU must have exactly one process before/after")
    parsed = [next(csv.reader([line])) for line in (before[0], after[0])]
    require(all(len(row) == 3 and row[0].strip().isdigit() and int(row[0]) > 0 and row[1].strip() for row in parsed), "GPU process CSV")
    require(int(parsed[0][0]) == int(parsed[1][0]) and parsed[0][1].strip() == parsed[1][1].strip(), "GPU PID changed")
    return int(parsed[0][0])


def audit_cell(folder, cell, asset, require_gpu=True):
    folder = Path(folder)
    verify_receipt(folder, CELL_FILES)
    require(folder.name == f"cell_{cell['cell_id']:04d}", "cell path identity")
    require((folder / "done").read_text().strip() == "completed", "cell done marker")
    seed = 9300000 + 10 * cell["dataset_seed"] + cell["repeat"]
    require(read_json(folder / "config.json") == dict(cell, sampler_seed=seed), "cell config/paired sampler seed")
    row, diagnostic = read_json(folder / "summary.json"), read_json(folder / "diagnostics.json")
    require(all(row[key] == value for key, value in cell.items()) and row["status"] == "completed", "cell summary configuration/status")
    for key in ("mean_error", "covariance_error", "sliced_w1_32", "quadrature_error_bound", "sampling_seconds",
                "prediction_seconds", "seconds", "log_normalizer", "log_normalizer_error", "ess_fraction", "factor_calls", "stages"):
        require(np.isfinite(row[key]), f"nonfinite summary {key}")
    require(all(row[key] >= 0 for key in ("mean_error", "covariance_error", "sliced_w1_32", "quadrature_error_bound", "factor_calls", "stages")), "negative metrics")
    require(row["sampling_seconds"] > 0 and row["prediction_seconds"] > 0, "invalid timing")
    close(row["sampling_seconds"], diagnostic["seconds"], "sampling time")
    close(row["prediction_seconds"], asset["summary"]["prediction_seconds"], "common prediction time")
    close(row["seconds"], row["sampling_seconds"] + row["prediction_seconds"], "total time")
    require(row["factor_calls"] == diagnostic["counts"]["factor_calls"] and row["stages"] == diagnostic["stages"], "diagnostic counters")
    pid = gpu_pid(read_json(folder / "gpu_processes.json"), require_gpu)
    if cell["reference"] == "exact":
        require(diagnostic["device"] == "cpu" and diagnostic["exact_components"] == 32 and diagnostic["stages"] == 0, "exact baseline diagnostics")
        close(row["log_normalizer"], float(asset["exact"]["log_normalizer"]), "enumeration normalizer")
        hash_coverage = "not recorded by frozen enumeration runner; payload route inspected in source"
    else:
        require(not require_gpu or diagnostic["device"].startswith("cuda"), "non-GPU production sampler")
        require(diagnostic["parameter_sha256"] == input_hash(asset["parameters"]), "sampler input hash mismatch")
        require(diagnostic["target"] == "prior_corrected_product_of_frozen_learned_projection_posteriors"
                and diagnostic["logZ_semantics"] == "composition_normalizer", "target semantics")
        for key in ("reference", "proposal_scale", "global_probability", "direct_is"):
            require(diagnostic[key] == cell[key], f"sampler setting {key}")
        records = diagnostic["records"]
        require(1 <= len(records) <= 128 and len(records) == row["stages"], "stage records")
        beta, logz = 0., 0.
        for index, record in enumerate(records):
            require(record["stage"] == index and record["beta_previous"] == beta and beta < record["beta"] <= 1, "temperature history")
            beta = record["beta"]
            require(record["resampled"] == (beta < 1) and 0 < record["ess_fraction"] <= 1 + 1e-10, "stage resampling/ESS")
            logz += record["log_normalizer_increment"]
            close(record["log_normalizer"], logz, "cumulative normalizer")
        require(beta == 1, "incomplete bridge")
        close(row["log_normalizer"], logz, "saved normalizer versus increments")
        hash_coverage = "independently matched four-array learned payload"
    saved = load_npz(folder / "samples.npz")
    require(set(saved) == {"samples", "weights"}, "sample fields")
    validate_samples(saved["samples"], saved["weights"], cell["particles"])
    measured = independent_metrics(saved["samples"], saved["weights"], asset["exact"], asset["directions"], asset["grids"])
    errors = {key: close(row[key], measured[key], key, atol=1e-10, rtol=1e-10)
              for key in ("mean_error", "covariance_error", "ess_fraction")}
    errors["log_normalizer_error"] = close(row["log_normalizer_error"], row["log_normalizer"]
                                            - float(asset["exact"]["log_normalizer"]), "logZ error")
    discrepancy = abs(row["sliced_w1_32"] - measured["grid_sliced_w1_32"])
    require(discrepancy <= measured["w1_absolute_discrepancy_bound"], "projected W1 exceeds independent grid bound")
    return dict(cell_id=cell["cell_id"], setting_id=cell["setting_id"], dataset_seed=cell["dataset_seed"], repeat=cell["repeat"],
                sampler_seed=seed, gpu_pid=pid, input_hash_coverage=hash_coverage, metric_absolute_errors=errors,
                saved_sliced_w1_32=row["sliced_w1_32"], w1_absolute_discrepancy=discrepancy, **measured)


def audit_asset_job(arguments):
    root, phase, seed, cells = arguments
    root = Path(root)
    asset = prepare_asset(root / "assets" / f"seed_{seed}", seed)
    results = [audit_cell(root / phase / "cells" / f"cell_{cell['cell_id']:04d}", cell, asset) for cell in cells]
    pairs = []
    for repeat in sorted({cell["repeat"] for cell in cells}):
        matched = {cell["setting_id"]: cell for cell in cells if cell["repeat"] == repeat
                   and cell["setting_id"] in ("gaussian", "mixture")}
        if len(matched) == 2:
            paths = [root / phase / "cells" / f"cell_{matched[name]['cell_id']:04d}" for name in ("gaussian", "mixture")]
            samples = [load_npz(path / "samples.npz") for path in paths]
            diagnostics = [read_json(path / "diagnostics.json") for path in paths]
            pairs.append(dict(repeat=repeat,
                samples_bitwise_equal=samples[0]["samples"].tobytes() == samples[1]["samples"].tobytes(),
                weights_bitwise_equal=samples[0]["weights"].tobytes() == samples[1]["weights"].tobytes(),
                gaussian_components=len(diagnostics[0]["reference_parameters"]["weights"]),
                mixture_components=len(diagnostics[1]["reference_parameters"]["weights"])))
    asset["report"]["gaussian_mixture_paired_observations"] = pairs
    asset["report"]["prediction_seconds"] = asset["summary"]["prediction_seconds"]
    return dict(asset=asset["report"], cells=results)


def inspect_sampler_source(source):
    tree = ast.parse((Path(source) / "learned_factor_bridge_20260921.py").read_text())
    fields = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "FIELDS" for t in node.targets)]
    require(fields == [FIELDS], "source sampler input fields")
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    require(not any(name and ("reference_20260921" in name or "learned_sbi" in name) for name in imports), "sampler oracle import")
    forbidden = {"open", "load", "read_text", "read_bytes", "load_npz", "exact_mixture", "metrics", "model_error"}
    calls = [node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
             for node in ast.walk(tree) if isinstance(node, ast.Call)]
    require(not forbidden.intersection(calls), "sampler oracle/file access in frozen source")
    runner = ast.parse((Path(source) / "run_learned_factor_study_20260921.py").read_text())
    execute = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
    exact_calls = [node for node in ast.walk(execute) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == "exact_mixture"]
    require(len(exact_calls) == 1 and isinstance(exact_calls[0].args[0], ast.Name)
            and exact_calls[0].args[0].id == "payload", "enumeration must construct reference from payload")
    return "static sampler-interface check plus per-cell payload hash; not an OS-level information-flow proof"


def audit_run(root, source, phase, workers=4):
    require(1 <= workers <= 4, "workers must be in 1..4")
    root, source = Path(root), Path(source)
    dev_manifest, dev_rows = validate_manifest(root, source, "development")
    selection = development_selection(dev_rows)
    saved_selection = read_json(root / "selection.json")
    require(set(saved_selection) == set(selection) and all(saved_selection[key] == selection[key]
            for key in selection if key != "means"), "development selection mismatch")
    require(set(saved_selection["means"]) == set(selection["means"]), "selection methods")
    for name, values in selection["means"].items():
        require(set(saved_selection["means"][name]) == set(values), "selection metrics")
        for key, value in values.items():
            close(saved_selection["means"][name][key], value, f"selection {name}/{key}", atol=1e-12, rtol=1e-12)
    if phase == "confirmation":
        require(selection["expand_confirmation"], "confirmation without passing development gate")
        manifest, rows = validate_manifest(root, source, phase, selection["baseline"])
        require(manifest["sources"] == dev_manifest["sources"] and manifest["source_commit"] == dev_manifest["source_commit"], "two-phase source mismatch")
    else:
        require(phase == "development", "invalid phase")
        manifest, rows = dev_manifest, dev_rows
    source_coverage = inspect_sampler_source(source)
    seeds = sorted({cell["dataset_seed"] for cell in manifest["cells"]})
    jobs = [(str(root), phase, seed, [cell for cell in manifest["cells"] if cell["dataset_seed"] == seed]) for seed in seeds]
    if workers == 1:
        results = [audit_asset_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(audit_asset_job, jobs))
    cells = sorted([cell for result in results for cell in result["cells"]], key=lambda row: row["cell_id"])
    pids = {cell["gpu_pid"] for cell in cells}
    require(None not in pids and len(pids) == 1, "production GPU PID must remain identical across phase")
    return dict(status="passed", phase=phase, source_commit=manifest["source_commit"], sources=manifest["sources"],
                manifest_sha256=sha256(root / phase / "manifest.json"), auditor_sha256=sha256(Path(__file__)),
                selection=selection, checked_cells=len(cells), assets=[r["asset"] for r in results], cells=cells,
                maximum_w1_absolute_discrepancy=max(c["w1_absolute_discrepancy"] for c in cells),
                maximum_w1_discrepancy_bound=max(c["w1_absolute_discrepancy_bound"] for c in cells),
                maximum_metric_absolute_errors={key: max(c["metric_absolute_errors"][key] for c in cells)
                                                for key in cells[0]["metric_absolute_errors"]},
                descriptive_timing=dict(prediction_seconds_by_seed={str(r["asset"]["seed"]): r["asset"]["prediction_seconds"] for r in results},
                    qualification="First-call/cold-start interpretation is descriptive; recorded total times and original gate remain unchanged"),
                coverage=dict(no_oracle=source_coverage, checkpoint_bank="frozen manifest hashes and strict-reload receipts checked; checkpoint loading not repeated",
                    model_error="asset receipt/value checked; model-error quadrature not independently recomputed",
                    w1="32 independent 32769-node Gaussian CDF quantizations; mean half-spacing plus Gaussian stoploss tails plus 1e-9",
                    w1_roundoff="1e-9 numerical allowance; no certified floating-point bound",
                    per_direction="saved summary supplies mean W1 only; all 32 grid distances and bounds retained",
                    gpu="single identical PID before/after each cell; monitoring between snapshots unavailable",
                    source_commit="manifest identifier retained; source bytes checked without git",
                    development="receipt-verified selection; numerical cell checks apply to requested phase only",
                    performance="integrity/numerical audit only; confirmation bootstrap performance gate not evaluated"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.is_relative_to(args.root.resolve()) and output != args.source.resolve(),
            "audit output must be separate from original run and source directory")
    output.mkdir(parents=True, exist_ok=False)
    # 独占输出目录；失败证据保留，异常继续传播以产生非零退出状态。
    try:
        report = audit_run(args.root, args.source, args.phase, args.workers)
    except Exception as error:
        report = dict(status="failed", error_type=type(error).__name__, error=str(error), phase=args.phase,
                      root=str(args.root.resolve()), source=str(args.source.resolve()), auditor_sha256=sha256(Path(__file__)))
        with (output / "audit.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        raise
    with (output / "audit.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report["status"], checked_cells=report["checked_cells"], report=str(output / "audit.json"))))


if __name__ == "__main__":
    main()
