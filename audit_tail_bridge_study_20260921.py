import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np
import scipy
from scipy.special import logsumexp

from audit_sensor_study_20260921 import (
    _audit_asset as _sensor_asset_audit,
    _direct_log_density,
    _independent_metrics,
    _sha256,
)


SOURCE_FILES = (
    "run_tail_bridge_study_20260921.py", "tail_bridge_smc_20260921.py",
    "nonseparable_sensor_model_20260921.py", "nonseparable_sensor_sampler_20260921.py",
    "run_sensor_study_20260921.py", "run_solid_20260920.py",
    "run_anchored_confirmation_20260921.py", "docs/CONTINUED_ITERATION_PROTOCOL_20260921.md",
    "docs/TAIL_BRIDGE_STUDY_PROTOCOL_20260921.md",
)
PARAMETERS = ("directions", "noise_std", "positive_probability", "observations")
METRICS = ("sliced_w1_32", "w1_mean", "mean_error", "covariance_error", "state_mse",
           "coverage90", "coverage95", "width90", "width95")
ASSET_FILES = {"problem.json", "exact.npz", "reference.npz", "reference_summary.json"}
CELL_FILES = {"config.json", "samples.npz", "diagnostics.json", "gpu_processes.json", "summary.json"}


class AuditViolation(ValueError):
    pass


def _require(condition, label):
    if not bool(condition):
        raise AuditViolation(label)


def _finite_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)
    elif isinstance(value, float):
        _require(math.isfinite(value), "nonfinite JSON number")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(path):
    value = json.loads(Path(path).read_text(), object_pairs_hook=_unique_pairs)
    _finite_tree(value)
    return value


def _same(left, right, label):
    _require(json.dumps(left, sort_keys=True, allow_nan=False)
             == json.dumps(right, sort_keys=True, allow_nan=False), label)


def _near(left, right, label, atol=2e-10, rtol=2e-10):
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    _require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
             and np.allclose(a, b, atol=atol, rtol=rtol), label)


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum, label)
    return value


def _done(directory):
    path = directory / "done"
    _require(path.is_file() and path.read_text().strip() == "completed", f"missing/invalid done: {directory}")


def _receipt(directory, expected):
    receipt = _json(directory / "receipt.json")
    files = receipt.get("files")
    _require(isinstance(files, dict) and set(files) == expected, f"receipt coverage: {directory}")
    for name, digest in files.items():
        _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), f"invalid SHA256: {name}")
        path = directory / name
        _require(path.is_file() and path.resolve().parent == directory.resolve(), f"invalid receipt path: {name}")
        _require(_sha256(path) == digest, f"SHA256 mismatch: {path}")
    return {"passed": True, "files_checked": len(files), "receipt_sha256": _sha256(directory / "receipt.json")}


def frozen_settings():
    rows = []
    for reference, components in (("gaussian", 1), ("mixture", 2), ("mixture", 4)):
        for covariance, proposal in itertools.product((1.0, 4.0), (0.5, 1.0)):
            rows.append(dict(setting_id=f"bridge_{reference}{components}_cov{covariance:g}_s{proposal:g}",
                kind="bridge", reference=reference, components=components, covariance_scale=covariance,
                proposal_scale=proposal, ess_target=0.8, moves=3, mode_steps=32, particles=16384, role="candidate"))
    for proposal in (0.15, 0.35, 0.65):
        rows.append(dict(setting_id=f"adaptive_prior_s{proposal:g}", kind="bridge", reference="prior",
            components=1, covariance_scale=1.0, proposal_scale=proposal, ess_target=0.8, moves=3,
            mode_steps=32, particles=16384, role="baseline"))
    for stages, proposal in itertools.product((32, 64, 128), (0.15, 0.35, 0.65)):
        rows.append(dict(setting_id=f"smc_k{stages}_s{proposal:g}", kind="smc", stages=stages,
                         proposal_scale=proposal, moves=3, particles=16384, role="baseline"))
    for method, batch in (("full", 4), ("tail_anchored", 8)):
        rows.append(dict(setting_id=method, kind="diffusion", method=method, particles=16384,
                         steps=2048, batch=batch, role="diagnostic"))
    return rows


def _problems(phase):
    _require(phase in ("development", "confirmation"), "unsupported phase")
    seeds = range(1400, 1404) if phase == "development" else range(1500, 1520)
    return [dict(dimension=d, regime=r, dataset_seed=s, groups=12)
            for d, r, s in itertools.product((2, 8), ("ambiguous", "regular"), seeds)]


def _asset_name(config):
    return f"d{config['dimension']}_{config['regime']}_{config['dataset_seed']}"


def expected_cells(phase, settings):
    repeats = range(1 if phase == "development" else 2)
    return [dict(problem, **setting, phase=phase, repeat=repeat, cell_id=i)
            for i, (problem, setting, repeat) in enumerate(itertools.product(_problems(phase), settings, repeats))]


def _rows_valid(rows, cells):
    _require(isinstance(rows, list) and len(rows) == len(cells), "rows count")
    lookup = {}
    for row in rows:
        index = _integer(row.get("cell_id"), "row cell_id")
        _require(index < len(cells) and index not in lookup, "duplicate/out-of-range row cell_id")
        for key, value in cells[index].items():
            _same(row.get(key), value, f"row config field {index}/{key}")
        _require(row.get("status") == "completed", f"row not completed: {index}")
        for key in METRICS + ("seconds", "factor_calls", "log_normalizer", "ess_fraction"):
            _require(type(row.get(key)) in (float, int) and math.isfinite(row[key]), f"invalid row {index}/{key}")
        _require(row["seconds"] > 0 and row["factor_calls"] > 0, "nonpositive costs")
        lookup[index] = row
    return lookup


def independent_selection(rows):
    settings = frozen_settings()
    _rows_valid(rows, expected_cells("development", settings))
    means = {}
    for setting in settings:
        items = [r for r in rows if r["setting_id"] == setting["setting_id"]]
        means[setting["setting_id"]] = {key: math.fsum(r[key] for r in items) / len(items)
                                       for key in ("sliced_w1_32", "seconds", "factor_calls")}
    baselines = [s["setting_id"] for s in settings if s["role"] == "baseline"]
    minimum_error = min(means[s]["sliced_w1_32"] for s in baselines)
    baseline = min((s for s in baselines if means[s]["sliced_w1_32"] <= minimum_error + 0.001),
                   key=lambda s: (means[s]["seconds"], s))
    fixed = "smc_k64_s0.15"
    eligible = []
    for setting in settings:
        name = setting["setting_id"]
        if setting["role"] != "candidate":
            continue
        valid = True
        for other in {baseline, fixed}:
            valid &= means[name]["sliced_w1_32"] <= means[other]["sliced_w1_32"] + 0.002
            valid &= means[name]["seconds"] < 0.8 * means[other]["seconds"]
            for dimension, regime in itertools.product((2, 8), ("ambiguous", "regular")):
                values = {}
                for setting_id in (name, other):
                    group = [r["sliced_w1_32"] for r in rows if r["setting_id"] == setting_id
                             and r["dimension"] == dimension and r["regime"] == regime]
                    values[setting_id] = math.fsum(group) / len(group)
                valid &= values[name] <= values[other] + 0.004
        if valid:
            eligible.append(name)
    candidate = min(eligible, key=lambda s: (means[s]["seconds"], s)) if eligible else None
    return dict(baseline=baseline, fixed_baseline=fixed, candidate=candidate, eligible=eligible,
                means=means, settings=settings, expand_confirmation=candidate is not None)


def _check_selection(saved, computed):
    for key in ("baseline", "fixed_baseline", "candidate", "eligible", "settings", "expand_confirmation"):
        _same(saved.get(key), computed[key], f"selection {key}")
    _require(saved.get("status") == "development_completed" and saved.get("development_only") is True,
             "selection provenance flags")
    _require(set(saved.get("means", {})) == set(computed["means"]), "selection means keys")
    for name, values in computed["means"].items():
        for key, value in values.items():
            _near(saved["means"][name].get(key), value, f"selection mean {name}/{key}")


def confirmation_settings(selection):
    _require(selection["expand_confirmation"] and selection["candidate"] is not None,
             "confirmation opened without eligible candidate")
    lookup = {s["setting_id"]: s for s in frozen_settings()}
    result = [lookup[selection[key]].copy() for key in ("candidate", "baseline", "fixed_baseline")]
    candidate = lookup[selection["candidate"]]
    for kind in ("prior", "gaussian"):
        result.append(dict(candidate, reference=kind, components=1, role="ablation",
                           covariance_scale=1.0 if kind == "prior" else candidate["covariance_scale"],
                           setting_id=f"matched_reference_{kind}"))
    result.extend([lookup["full"], lookup["tail_anchored"]])
    unique = {}
    for setting in result:
        unique.setdefault(setting["setting_id"], setting)
    return list(unique.values())


def _sources(manifest, source):
    _require(source.is_dir(), "--source must be the frozen source directory")
    sources = manifest.get("sources")
    _require(isinstance(sources, dict) and set(sources) == set(SOURCE_FILES), "manifest source coverage")
    report = {}
    for name in SOURCE_FILES:
        path = source / name
        _require(path.is_file() and path.resolve().is_relative_to(source.resolve()), f"missing source {name}")
        digest = _sha256(path)
        _require(digest == sources[name], f"source hash mismatch: {name}")
        report[name] = digest
    _require(isinstance(manifest.get("source_commit"), str)
             and re.fullmatch(r"[0-9a-f]{7,40}", manifest["source_commit"]), "source_commit syntax")
    return report


def _manifest_check(root, phase, source, settings):
    directory = root / phase
    _done(directory)
    manifest = _json(directory / "manifest.json")
    expected = expected_cells(phase, settings)
    for key, value in dict(settings=settings, cells=expected, phase=phase, particles=16384,
                           paired_sampler_repeats=1 if phase == "development" else 2,
                           independent_data_seeds=4 if phase == "development" else 20).items():
        _same(manifest.get(key), value, f"frozen manifest {key}")
    order = np.random.default_rng(20261010 if phase == "development" else 20261011).permutation(len(expected)).tolist()
    _same(manifest.get("order"), order, "frozen random cell order")
    actual = {p.name for p in (directory / "cells").iterdir()}
    _require(actual == {f"cell_{c['cell_id']:04d}" for c in expected}, "missing/unexpected cell directories")
    sources = _sources(manifest, source)
    state = _json(directory / "state.json")
    _require(state.get("status") == "completed" and state.get("cells") == len(expected), "phase state completeness")
    _require(state.get("device") in ("cpu", "cuda"), "phase device")
    _require(type(state.get("workers")) is int and 1 <= state["workers"] <= 4, "phase worker bound")
    _require(state.get("seconds", 0) > 0 and state.get("sampler_seconds", 0) > 0, "phase costs")
    rows = _json(directory / "rows.json")
    lookup = _rows_valid(rows, expected)
    _same([r["cell_id"] for r in rows], order, "rows random order")
    _near(state["sampler_seconds"], math.fsum(row["seconds"] for row in rows), "phase sampler time sum", atol=1e-8)
    return expected, lookup, state, {"passed": True, "cells": len(expected), "sources": sources,
                                    "manifest_sha256": _sha256(directory / "manifest.json"),
                                    "rows_sha256": _sha256(directory / "rows.json"),
                                    "source_commit_claim": manifest["source_commit"]}


def _npz(path, keys):
    with np.load(path, allow_pickle=False) as saved:
        _require(set(saved.files) == set(keys), f"NPZ fields: {path}")
        result = {key: saved[key] for key in keys}
    for key, value in result.items():
        _require(value.dtype == np.dtype("float64") and np.isfinite(value).all(), f"NPZ float64/finite: {path}/{key}")
    return result


def audit_asset(asset, config):
    receipt = _receipt(asset, ASSET_FILES)
    payload = _json(asset / "problem.json")
    groups, dimension = config["groups"], config["dimension"]
    _require(set(payload) == set(PARAMETERS) | {"truth"}, "problem fields")
    _require(np.shape(payload["directions"]) == (groups, dimension) and np.shape(payload["truth"]) == (dimension,),
             "problem shape")
    for key in PARAMETERS[1:]:
        _require(np.shape(payload[key]) == (groups,), f"problem {key} shape")
    _require(np.min(payload["noise_std"]) > 0, "problem noise")
    probability = np.asarray(payload["positive_probability"])
    _require(((probability > 0) & (probability < 1)).all(), "problem sign prior")
    _check_problem_seed(payload, config)
    exact = _npz(asset / "exact.npz", ("probabilities", "means", "covariance"))
    _require(exact["probabilities"].shape == (2 ** groups,) and exact["means"].shape == (2 ** groups, dimension)
             and exact["covariance"].shape == (dimension, dimension), "exact shapes")
    _require((exact["probabilities"] >= 0).all(), "negative exact probability")
    reference = _npz(asset / "reference.npz", ("samples", "second_samples", "directions", "mean", "covariance"))
    _require(reference["samples"].shape == reference["second_samples"].shape == (65536, dimension), "reference sample shapes")
    _require(reference["mean"].shape == (dimension,) and reference["covariance"].shape == (dimension, dimension),
             "reference moment shapes")
    _require(reference["directions"].shape == (32, dimension), "32 projection directions required")
    helper = _sensor_asset_audit(asset)
    _require(helper["pass"] and helper["reference"]["directions_normalized"], "independent sensor asset audit")
    # 同平台固定 RNG 重演采用明确的绝对阈值，并逐一核对两组数组。
    rng = np.random.default_rng(921000 + config["dataset_seed"])
    rng_errors = {}
    for key in ("samples", "second_samples"):
        indices = rng.choice(len(exact["probabilities"]), size=65536, p=exact["probabilities"])
        regenerated = exact["means"][indices] + rng.multivariate_normal(np.zeros(dimension), exact["covariance"], size=65536)
        _near(regenerated, reference[key], f"reference RNG {key}", atol=2e-12, rtol=0)
        rng_errors[key] = float(np.max(np.abs(regenerated - reference[key])))
    summary = _json(asset / "reference_summary.json")
    pair_metrics = _independent_metrics(reference["second_samples"], np.full(65536, 1 / 65536),
                                        reference, np.asarray(payload["truth"]))
    for key in METRICS:
        _near(pair_metrics[key], summary["reference_pair_metrics"].get(key), f"reference metric {key}")
    helper.update(receipt_strict=receipt, strict_rng_max_errors=rng_errors,
                  reference_pair_metrics=pair_metrics, passed=True)
    return helper, payload, reference


def _check_problem_seed(payload, config):
    # 独立复现冻结的参数生成顺序；真值仅用于离线审计，不传给采样器。
    groups, dimension = config["groups"], config["dimension"]
    _require(config["regime"] in ("regular", "ambiguous"), "unsupported problem regime")
    rng = np.random.default_rng(config["dataset_seed"])
    raw = rng.normal(size=(groups, dimension))
    if config["regime"] == "regular" and dimension >= 2:
        angle = np.linspace(0.1, 2.7, groups) + 0.03 * rng.normal(size=groups)
        raw[:, 0], raw[:, 1] = np.cos(angle), np.sin(angle)
    directions = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    truth = rng.normal(size=dimension)
    low, span = (0.8, 0.15) if config["regime"] == "ambiguous" else (0.35, 0.08)
    noise = low + span * rng.random(groups)
    probability = np.where(np.arange(groups) % 2 == 0, 0.65, 0.8)
    signs = np.where(rng.random(groups) < probability, 1.0, -1.0)
    observed = signs * (directions @ truth) + noise * rng.normal(size=groups)
    expected = dict(directions=directions, truth=truth, noise_std=noise,
                    positive_probability=probability, observations=observed)
    for name, values in expected.items():
        _near(payload[name], values, f"seed-generated problem {name}", atol=2e-13, rtol=0)


def _gpu_snapshots(path, device):
    snapshots = _json(path)
    _require(set(snapshots) == {"before", "after"}, "GPU snapshot fields")
    if device == "cpu":
        _require(snapshots == {"before": [], "after": []}, "CPU snapshot must contain empty arrays")
        return {"passed": True, "gpu_verified": False, "pid": None, "scope": "CPU run"}
    pids = []
    for key in ("before", "after"):
        lines = snapshots[key]
        _require(isinstance(lines, list) and len(lines) == 1 and isinstance(lines[0], str), f"GPU singleton {key}")
        fields = next(csv.reader(lines, skipinitialspace=True))
        _require(len(fields) == 3 and fields[0].strip().isdigit() and int(fields[0]) > 0
                 and fields[1].strip() and re.fullmatch(r"\d+\s+MiB", fields[2].strip()), f"GPU row {key}")
        pids.append(int(fields[0]))
    _require(pids[0] == pids[1], "GPU PID changed within cell")
    return {"passed": True, "gpu_verified": True, "pid": pids[0], "scope": "saved before/after snapshots"}


def _parameter_hash(payload):
    digest = hashlib.sha256()
    for name in PARAMETERS:
        value = np.asarray(payload[name], dtype=np.float64)
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def _no_oracle_fields(value):
    forbidden = {"truth", "true_theta", "oracle", "exact_posterior", "exact_samples", "reference_samples", "confirmation_metrics"}
    if isinstance(value, dict):
        _require(not forbidden.intersection(value), "oracle fields in sampling metadata")
        for item in value.values():
            _no_oracle_fields(item)
    elif isinstance(value, list):
        for item in value:
            _no_oracle_fields(item)


def _reference_check(meta, config, payload, x, weights, log_evidence):
    c, dimension = meta["components_actual"], config["dimension"]
    _integer(c, "actual component count", 1)
    _require(c <= (config["components"] if config["reference"] == "mixture" else 1), "component count bound")
    ref = meta["reference_parameters"]
    mixture_weights, means, covariance, precision = [np.asarray(ref[key], dtype=float)
                                                   for key in ("weights", "means", "covariance", "tail_precision")]
    _require(mixture_weights.shape == (c,) and means.shape == (c, dimension)
             and covariance.shape == precision.shape == (dimension, dimension), "reference shapes")
    _require((mixture_weights > 0).all(), "reference positive weights")
    _near(mixture_weights.sum(), 1.0, "reference weights sum", atol=1e-12, rtol=0)
    _near(covariance, covariance.T, "reference symmetry", atol=1e-12, rtol=0)
    _require(np.linalg.eigvalsh(covariance).min() > 0, "reference covariance SPD")
    a = np.asarray(payload["directions"])
    scaled = a / np.asarray(payload["noise_std"])[:, None]
    target_precision = np.eye(dimension) + scaled.T @ scaled
    search = meta["mode_search"]
    if config["reference"] == "prior":
        _near(means, np.zeros((1, dimension)), "prior center", atol=0, rtol=0)
        _near(covariance, np.eye(dimension), "prior covariance", atol=0, rtol=0)
        _near(precision, np.eye(dimension), "prior precision", atol=0, rtol=0)
        _same(search, dict(algorithm="prior_identity", mode_steps=0, starts=0,
                           requested_components=1, actual_components=1), "prior skips mode search")
    else:
        _near(precision, target_precision, "tail precision", atol=2e-11, rtol=0)
        _near(covariance @ target_precision, config["covariance_scale"] * np.eye(dimension), "scaled tail covariance", atol=2e-10, rtol=0)
        _require(search["algorithm"] == "fixed_tail_precision_latent_sign_em"
                 and search["mode_steps"] == config["mode_steps"] and search["starts"] == 5 + 2 * config["groups"]
                 and search["actual_components"] == c and search["requested_components"] == config["components"], "mode search configuration")
        values = _direct_log_density(payload, means, log_evidence) + log_evidence
        _near(values, search["selected_log_joint"], "reference selected target heights", atol=1e-8, rtol=1e-10)
        _near(mixture_weights, np.exp(values - logsumexp(values)), "reference peak weights", atol=2e-10, rtol=0)
    delta = x[:, None] - means
    solved = np.linalg.solve(covariance, delta.reshape(-1, dimension).T).T.reshape(delta.shape)
    logs = np.log(mixture_weights) - 0.5 * (np.einsum("nci,nci->nc", delta, solved)
            + np.linalg.slogdet(covariance)[1] + dimension * math.log(2 * math.pi))
    responsibilities = np.exp(logs - logsumexp(logs, axis=1, keepdims=True))
    _near(weights @ responsibilities, meta["weighted_reference_responsibility_mass"], "weighted component responsibilities")
    return {"passed": True, "components": c, "minimum_covariance_eigenvalue": float(np.linalg.eigvalsh(covariance).min()),
            "input_parameter_sha256": _parameter_hash(payload)}


def _bridge_diagnostics(config, diagnostics, summary, payload, x, weights, evidence, device):
    meta, counts, timing = (diagnostics[key] for key in ("metadata", "counts", "timing"))
    n, groups = config["particles"], config["groups"]
    _no_oracle_fields(meta)
    _require(meta.get("status") == "completed" and meta.get("method") == "tail_informed_annealed_smc", "bridge status")
    for key in ("reference", "particles", "groups", "dimension", "ess_target", "moves", "proposal_scale", "mode_steps", "covariance_scale"):
        _same(meta.get(key), config[key], f"bridge config {key}")
    _same(meta.get("components_requested"), config["components"], "requested components")
    _require(meta.get("seed") == summary["seed"] and meta.get("sign_flip") is True
             and meta.get("temperature_strategy") == "adaptive_ess" and meta.get("max_stages") == 256
             and meta.get("return_numpy") is True and meta.get("dtype") == "float64", "bridge execution flags")
    _require(meta.get("device") == "cpu" if device == "cpu" else bool(re.fullmatch(r"cuda:\d+", meta.get("device", ""))), "bridge device")
    _require(meta.get("parameter_sha256") == _parameter_hash(payload), "bridge parameter hash")
    _require(meta.get("target") == "normalized_prior_times_normalized_sensor_likelihoods"
             and meta.get("logZ_semantics") == "estimated_log_joint_observed_data_evidence"
             and meta.get("normalizer_unbiasedness_claim") is False, "bridge target/evidence semantics")
    _near(meta.get("covariance_scale_applied"), 1.0 if config["reference"] == "prior" else config["covariance_scale"], "applied covariance scale")
    seed = summary["seed"]
    _same(meta.get("random_streams"), dict(motion_seed=seed, auxiliary_seed=(seed + 1000003) % (2 ** 63),
                                           resampling_seed=(seed + 2000003) % (2 ** 63)), "RNG streams")
    records = meta["records"]
    stages = _integer(meta["stages"], "bridge stages", 1)
    _require(stages == len(records) == summary["stages"] and stages <= 256, "bridge stage lengths")
    _same(meta["temperatures"], [0.0] + [row["beta"] for row in records], "temperature history")
    previous, log_z = 0.0, 0.0
    accepted, sign_accepted = 0, 0
    for index, row in enumerate(records):
        _require(row["stage"] == index and previous < row["beta"] <= 1, "strict beta progression")
        _near(row["beta_previous"], previous, "previous beta", atol=0, rtol=0)
        _near(row["delta_beta"], row["beta"] - previous, "delta beta", atol=1e-14, rtol=0)
        _require(config["ess_target"] - 1e-10 <= row["ess_fraction"] <= 1.0 + 1e-10, "adaptive ESS threshold")
        if index < stages - 1:
            _near(row["ess_fraction"], config["ess_target"], "maximal adaptive beta", atol=1e-9, rtol=0)
        _same(row["resampled"], index < stages - 1, "resampling rule")
        for name, proposals in (("mh", n * config["moves"]), ("sign", n)):
            _require(row[name + "_proposals"] == proposals, f"{name} proposals")
            value = _integer(row[name + "_accepted"], f"{name} accepted")
            _require(value <= proposals, f"{name} accepted bound")
            _near(row["acceptance" if name == "mh" else "sign_acceptance"], value / proposals, f"{name} acceptance")
        accepted += row["mh_accepted"]
        sign_accepted += row["sign_accepted"]
        _require(row["factor_calls"] == n * groups * (config["moves"] + 1)
                 and row["sign_factor_calls"] == n * groups, "stage factor counts")
        _require(math.isfinite(row["weighted_residual_mean_before_move"])
                 and row["weighted_residual_variance_before_move"] >= 0, "residual moments")
        log_z += row["log_normalizer_increment"]
        _near(row["log_normalizer"], log_z, "cumulative logZ", atol=1e-9, rtol=0)
        previous = row["beta"]
    _require(previous == 1.0, "terminal beta must equal one")
    _near(log_z, summary["log_normalizer"], "increment sum logZ", atol=1e-9, rtol=0)
    _near(records[-1]["ess_fraction"], summary["ess_fraction"], "terminal weights ESS")
    mode_points = 0 if config["reference"] == "prior" else (5 + 2 * groups) * (config["mode_steps"] + 1)
    target_points = n * (1 + stages * (config["moves"] + 1))
    expected_counts = dict(factor_preparation_calls=groups, mode_factor_calls=mode_points * groups,
        particle_factor_calls=target_points * groups, mode_point_evaluations=mode_points,
        target_point_evaluations=target_points, reference_samples=n, mh_proposals=stages * n * config["moves"],
        mh_accepted=accepted, sign_proposals=stages * n, sign_accepted=sign_accepted,
        sign_factor_calls=stages * n * groups, resampling_calls=stages - 1,
        reference_component_evaluations=n * meta["components_actual"] * (2 + stages * (2 * config["moves"] + 2)),
        factor_calls=groups + mode_points * groups + target_points * groups)
    _same(counts, expected_counts, "complete bridge counts")
    _near(summary["factor_calls"], counts["factor_calls"], "summary factor counts", atol=0, rtol=0)
    parts = [timing[key] for key in ("preparation_seconds", "initialization_seconds", "annealing_seconds", "output_seconds")]
    _require(all(value > 0 for value in parts) and timing.get("includes_cpu_output_and_statistics") is True, "preparation/output timing inclusion")
    _near(sum(parts), timing["total_seconds"], "timing parts", atol=1e-9, rtol=0)
    statistics = meta["statistics"]
    mean = weights @ x
    centered = x - mean
    _near(statistics["weighted_mean"], mean, "saved weighted mean")
    _near(statistics["weighted_covariance"], (centered * weights[:, None]).T @ centered, "saved weighted covariance")
    _same(statistics["quantile_levels"], [0.025, 0.05, 0.95, 0.975], "quantile levels")
    quantiles = []
    for column in x.T:
        order = np.argsort(column, kind="stable")
        cumulative = np.cumsum(weights[order])
        cumulative[-1] = 1.0
        quantiles.append(column[order[np.searchsorted(cumulative, statistics["quantile_levels"])]])
    _near(statistics["weighted_quantiles"], quantiles, "weighted quantiles")
    for key, value in dict(ess=1 / np.dot(weights, weights), weight_min=weights.min(), weight_max=weights.max()).items():
        _near(statistics[key], value, f"statistics {key}")
    reference = _reference_check(meta, config, payload, x, weights, evidence)
    return {"passed": True, "stages": stages, "reference": reference,
            "increment_sum": log_z, "factor_calls": counts["factor_calls"],
            "no_oracle_scope": "four-field parameter hash, metadata schema and frozen source receipt"}


def _legacy_diagnostics(config, diagnostics, summary, weights):
    records = diagnostics["metadata"]["records"]
    n, groups = config["particles"], config["groups"]
    stages = config["stages"] if config["kind"] == "smc" else config["steps"]
    _require(len(records) == stages == diagnostics["metadata"]["stages"] == summary["stages"], "legacy stages")
    calls = n * groups if config["kind"] == "smc" else 0
    for i, row in enumerate(records):
        _require(row["step"] == i and 0 < row["ess_fraction"] <= 1 + 1e-10, "legacy step/ESS")
        if config["kind"] == "smc":
            _near(row["beta"], (i + 1) / stages, "fixed SMC beta", atol=1e-14, rtol=0)
            _require(0 <= row["acceptance"] <= 1 and 0 <= row["sign_acceptance"] <= 1, "legacy acceptance")
            expected = n * groups * (config["moves"] + 1)
        else:
            _same(row["resampled"], row["ess_fraction"] < 0.5 and i < stages - 1, "diffusion resampling rule")
            expected = n * (groups if config["method"] == "full" else config["batch"])
        _require(row["factor_calls"] == expected, "legacy factor calls")
        calls += expected
    _require(diagnostics["counts"] == {"factor_calls": calls} and summary["factor_calls"] == calls, "legacy total costs")
    if config["kind"] == "smc":
        _near(weights, np.full(n, 1 / n), "SMC final resampled weights", atol=2e-15, rtol=0)
    else:
        _near(records[-1]["ess_fraction"], summary["ess_fraction"], "diffusion final ESS")
    return {"passed": True, "stages": stages, "factor_calls": calls,
            "logZ_increment_check": "not_saved_by_frozen_legacy_sampler",
            "cost_scope": "frozen runner factor-call convention"}


def audit_cell(cell, config, payload, reference, evidence, device, expected_row=None):
    report = {"cell_id": config["cell_id"], "setting_id": config["setting_id"], "passed": False}
    try:
        _done(cell)
        report["receipt"] = _receipt(cell, CELL_FILES)
        _same(_json(cell / "config.json"), config, "cell config differs from frozen matrix")
        summary = _json(cell / "summary.json")
        if expected_row is not None:
            _same(summary, expected_row, "summary differs from phase rows")
        for key, value in config.items():
            _same(summary.get(key), value, f"summary config {key}")
        seed = 8100000 + 1000 * config["dimension"] + 10 * config["dataset_seed"] + config["repeat"]
        _require(summary.get("seed") == seed and summary.get("status") == "completed", "paired seed/status")
        arrays = _npz(cell / "samples.npz", ("samples", "weights"))
        x, w = arrays["samples"], arrays["weights"]
        _require(x.shape == (config["particles"], config["dimension"]) and w.shape == (config["particles"],), "particle shapes")
        _require((w >= 0).all(), "negative particle weights")
        _near(w.sum(), 1.0, "particle weight normalization", atol=1e-10, rtol=0)
        _near(summary["ess_fraction"], 1.0 / np.dot(w, w) / len(w), "summary ESS")
        measured = _independent_metrics(x, w, reference, np.asarray(payload["truth"]))
        report["metrics"] = {}
        for key in METRICS:
            _near(measured[key], summary.get(key), f"independent metric {key}")
            report["metrics"][key] = {"value": measured[key], "saved": summary[key],
                                      "absolute_error": abs(measured[key] - summary[key])}
        diagnostics = _json(cell / "diagnostics.json")
        _require(summary["seconds"] > 0 and summary["factor_calls"] > 0, "cell positive costs")
        _near(summary["seconds"], diagnostics["seconds"], "diagnostic seconds", atol=0, rtol=0)
        _near(summary["seconds"], diagnostics["timing"]["total_seconds"], "total seconds", atol=0, rtol=0)
        report["gpu"] = _gpu_snapshots(cell / "gpu_processes.json", device)
        if config["kind"] == "diffusion":
            _require("log_evidence_error" not in summary, "FK logZ must not be labeled sensor evidence")
            _require(math.isfinite(summary["log_normalizer"]), "finite FK logZ")
            report["normalizer"] = {"scope": "FK diagnostic, sensor evidence comparison not applicable"}
        else:
            error = summary["log_normalizer"] - evidence
            _near(summary.get("log_evidence_error"), error, "independent observation-space evidence error", atol=1e-9, rtol=1e-10)
            report["normalizer"] = {"scope": "sensor joint evidence", "independent_log_evidence": evidence,
                                     "estimate": summary["log_normalizer"], "error": error,
                                     "accuracy_gate_applied": False}
        if config["kind"] == "bridge":
            report["diagnostics"] = _bridge_diagnostics(config, diagnostics, summary, payload, x, w, evidence, device)
        elif config["kind"] in ("smc", "diffusion"):
            report["diagnostics"] = _legacy_diagnostics(config, diagnostics, summary, w)
        else:
            raise AuditViolation("unsupported kind")
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    return report


def _asset_job(job):
    root, phase, config, cells, rows, device = job
    asset = Path(root) / "assets" / _asset_name(config)
    result = {"asset": asset.name, "passed": False, "cells": []}
    try:
        result["asset_checks"], payload, reference = audit_asset(asset, config)
        evidence = result["asset_checks"]["log_sensor_evidence"]
        for cell in cells:
            path = Path(root) / phase / "cells" / f"cell_{cell['cell_id']:04d}"
            result["cells"].append(audit_cell(path, cell, payload, reference, evidence, device, rows[cell["cell_id"]]))
        result["passed"] = bool(result["cells"]) and all(item["passed"] for item in result["cells"])
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def audit(root, output, *, phase, source, workers=4):
    root, output, source = Path(root), Path(output), Path(source)
    _require(type(workers) is int and 1 <= workers <= 4, "workers must be in [1,4]")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"status": "failed", "passed": False, "phase": phase, "root": str(root.resolve()),
              "source": str(source.resolve()), "workers": workers, "assets": [], "errors": [],
              "auditor_sha256": _sha256(Path(__file__)),
              "helper_sha256": _sha256(Path(__file__).with_name("audit_sensor_study_20260921.py")),
              "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__},
              "counts": {"assets_expected": 16 if phase == "development" else 80,
                         "assets_checked": 0, "assets_passed": 0,
                         "cells_expected": 416 if phase == "development" else None,
                         "cells_checked": 0, "cells_passed": 0, "metrics_per_cell": len(METRICS)},
              "scope": "selected completed phase; numerical integrity, not a performance confidence gate"}
    # 独占创建输出；失败报告保留，输入产物始终只读。
    with output.open("x") as stream:
        try:
            _require(root.is_dir(), "missing run root")
            _require(phase in ("development", "confirmation"), "invalid phase")
            selection = None
            if phase == "confirmation":
                _, dev_rows, _, dev_manifest = _manifest_check(root, "development", source, frozen_settings())
                selection = independent_selection(list(dev_rows.values()))
                _check_selection(_json(root / "selection.json"), selection)
                report["selection_basis"] = {"manifest": dev_manifest,
                    "scope": "independent gate recomputation from frozen development rows; numerical development audit is separate"}
            settings = frozen_settings() if phase == "development" else confirmation_settings(selection)
            cells, rows, state, manifest = _manifest_check(root, phase, source, settings)
            report["manifest"] = manifest
            report["state"] = state
            if phase == "development":
                selection = independent_selection(list(rows.values()))
                _check_selection(_json(root / "selection.json"), selection)
            report["selection"] = selection
            problems = _problems(phase)
            jobs = [(str(root), phase, problem,
                     [c for c in cells if _asset_name(c) == _asset_name(problem)], rows, state["device"])
                    for problem in problems]
            if workers == 1:
                report["assets"] = [_asset_job(job) for job in jobs]
            else:
                names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
                previous = {name: os.environ.get(name) for name in names}
                try:
                    for name in names:
                        os.environ[name] = "1"
                    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                        report["assets"] = list(pool.map(_asset_job, jobs))
                finally:
                    for name, value in previous.items():
                        if value is None:
                            os.environ.pop(name, None)
                        else:
                            os.environ[name] = value
            checked_cells = [cell for asset in report["assets"] for cell in asset["cells"]]
            report["counts"] = {"assets_expected": len(problems), "assets_checked": len(report["assets"]),
                "assets_passed": sum(a.get("asset_checks", {}).get("passed", False) for a in report["assets"]),
                "cells_expected": len(cells), "cells_checked": len(checked_cells),
                "cells_passed": sum(c["passed"] for c in checked_cells), "metrics_per_cell": len(METRICS)}
            _require(len(checked_cells) == len(cells) and all(a["passed"] for a in report["assets"]), "asset/cell validation failed")
            if state["device"] == "cuda":
                pids = {c["gpu"]["pid"] for c in checked_cells}
                _require(len(pids) == 1 and None not in pids, "GPU PID changed across phase")
                report["gpu_isolation"] = {"passed": True, "pid": next(iter(pids)),
                                            "scope": "before/after snapshots for every cell, not continuous monitoring"}
            else:
                report["gpu_isolation"] = {"passed": None, "scope": "CPU phase; GPU isolation not verified"}
            report["passed"] = True
            report["status"] = "passed"
        except Exception as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
            report["traceback"] = traceback.format_exc()
        report["seconds"] = time.perf_counter() - started
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args(argv)
    report = audit(args.root, args.output, phase=args.phase, source=args.source, workers=args.workers)
    print(json.dumps({"status": report["status"], "counts": report.get("counts"), "errors": report["errors"]}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
