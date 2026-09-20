import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import wasserstein_distance
import torch

from composition_benchmark import runtime
from matrix_tail_certificate_20260921 import matrix_certificate
from nonseparable_sensor_model_20260921 import make_problem, SensorProblem
from nonseparable_sensor_sampler_20260921 import annealed_smc, diffusion_sample, timed_sample
from run_solid_20260920 import sha256, write_json


def settings():
    rows = []
    for n, k in [(8192, 1024), (16384, 2048)]:
        for method, batch in [("full", 4), ("tail_anchored", 4), ("tail_anchored", 8)]:
            rows.append(dict(method=method, particles=n, steps=k, batch=batch,
                             setting_id=f"{method}_n{n}_k{k}_m{batch}"))
    rows += [dict(method=m, particles=16384, steps=2048, batch=4, setting_id=m)
             for m in ["tail_fixed", "unbiased"]]
    rows += [dict(method="annealed_smc", particles=16384, stages=k, proposal_scale=s,
                  setting_id=f"smc_k{k}_scale{s}") for k, s in [(64, .15), (64, .35), (64, .65), (128, .35)]]
    return rows


FULL_ID = "full_n16384_k2048_m4"


def problems(phase):
    return [dict(dimension=d, regime=r, dataset_seed=s, groups=12)
            for d, r, s in itertools.product([2, 8], ["ambiguous", "regular"],
                                               range(1100, 1102) if phase == "development" else range(1200, 1220))]


def asset_name(config):
    return f"d{config['dimension']}_{config['regime']}_{config['dataset_seed']}"


def metrics(x, w, reference, problem):
    if not np.isfinite(x).all() or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("invalid metric input")
    np.testing.assert_allclose(w.sum(), 1, atol=1e-10)
    reference_samples, directions = reference["samples"], reference["directions"]
    projections = [wasserstein_distance(x @ direction, reference_samples @ direction, u_weights=w) for direction in directions]
    marginals = [wasserstein_distance(x[:, d], reference_samples[:, d], u_weights=w) for d in range(x.shape[1])]
    mean = w @ x
    covariance = (x - mean).T @ ((x - mean) * w[:, None])
    quantiles = np.empty((x.shape[1], 4))
    for d in range(x.shape[1]):
        order = np.argsort(x[:, d], kind="stable")
        cumulative = np.cumsum(w[order])
        cumulative[-1] = 1.
        quantiles[d] = x[order[np.searchsorted(cumulative, [.025, .05, .95, .975], side="left")], d]
    return dict(sliced_w1_32=float(np.mean(projections)), w1_mean=float(np.mean(marginals)),
                mean_error=float(np.linalg.norm(mean - reference["mean"]) / math.sqrt(x.shape[1])),
                covariance_error=float(np.linalg.norm(covariance - reference["covariance"]) / x.shape[1]),
                state_mse=float(np.mean((mean - problem.truth) ** 2)),
                coverage90=float(np.mean((quantiles[:, 1] <= problem.truth) & (problem.truth <= quantiles[:, 2]))),
                coverage95=float(np.mean((quantiles[:, 0] <= problem.truth) & (problem.truth <= quantiles[:, 3]))),
                width90=float(np.mean(quantiles[:, 2] - quantiles[:, 1])), width95=float(np.mean(quantiles[:, 3] - quantiles[:, 0])))


def prepare_one(item):
    root, config = item
    asset = root / "assets" / asset_name(config)
    if (asset / "receipt.json").exists():
        receipt = json.loads((asset / "receipt.json").read_text())
        if not all(sha256(asset / name) == digest for name, digest in receipt["files"].items()):
            raise RuntimeError("asset checksum mismatch")
        return asset.name
    asset.mkdir(parents=True, exist_ok=False)
    problem = make_problem(config["dataset_seed"], groups=12, dimension=config["dimension"], regime=config["regime"])
    write_json(asset / "problem.json", problem.to_dict())
    start = time.perf_counter()
    p, means, covariance = problem.exact_posterior()
    enumeration_seconds = time.perf_counter() - start
    mean = p @ means
    total_covariance = covariance + (means - mean).T @ ((means - mean) * p[:, None])
    np.savez_compressed(asset / "exact.npz", probabilities=p, means=means, covariance=covariance)
    rng = np.random.default_rng(921000 + config["dataset_seed"])
    samples = []
    for _ in range(2):
        indices = rng.choice(len(p), size=65536, p=p)
        samples.append(means[indices] + rng.multivariate_normal(np.zeros(problem.dimension), covariance, size=65536))
    directions = np.random.default_rng(921032).normal(size=(32, problem.dimension))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    reference = dict(samples=samples[0], directions=directions, mean=mean, covariance=total_covariance)
    np.savez_compressed(asset / "reference.npz", **reference, second_samples=samples[1])
    commutators = [np.linalg.norm(a @ b - b @ a) for a, b in itertools.combinations(problem.factor_parameters()[0], 2)]
    write_json(asset / "reference_summary.json", dict(enumeration_seconds=enumeration_seconds,
               maximum_precision_commutator=float(max(commutators)),
               reference_pair_metrics=metrics(samples[1], np.full(65536, 1 / 65536), reference, problem)))
    write_json(asset / "receipt.json", dict(files={p.name: sha256(p) for p in sorted(asset.iterdir())}))
    return asset.name


def load_reference(asset):
    with np.load(asset / "reference.npz") as data:
        return {name: data[name] for name in data.files}


def certificate(A, grid):
    V = np.eye(A.shape[-1])
    minimum = math.inf
    for step, h in enumerate(-np.diff(grid)):
        result = matrix_certificate(A[step], V, float(h), 1.)
        minimum = min(minimum, float(result["H_eigenvalues"][0]))
        if result["status"] != "finite":
            return dict(status=result["status"], step=step, minimum_precision_eigenvalue=minimum)
        V = result["V_next"]
    return dict(status="finite", steps=len(grid) - 1, minimum_precision_eigenvalue=minimum, final_tail_covariance=V.tolist())


def choose(rows):
    means = {}
    for s in settings():
        observed = [r for r in rows if r["setting_id"] == s["setting_id"]]
        if len(observed) != 8:
            raise ValueError("complete development observations are required")
        means[s["setting_id"]] = dict(error=float(np.mean([r["sliced_w1_32"] for r in observed])),
                                      seconds=float(np.mean([r["seconds"] for r in observed])))
    choices = {}
    for method in ["tail_anchored", "annealed_smc"]:
        candidates = [s for s in settings() if s["method"] == method]
        eligible = [s for s in candidates if means[s["setting_id"]]["error"] <= means[FULL_ID]["error"] + .005]
        selected = min(eligible, key=lambda s: means[s["setting_id"]]["seconds"]) if eligible else min(candidates, key=lambda s: means[s["setting_id"]]["error"])
        choices[method] = dict(selected=selected, development_eligible=bool(eligible))
    return dict(choices=choices, means=means)


def run_one(config, asset, output):
    problem = SensorProblem.from_dict(json.loads((asset / "problem.json").read_text()))
    reference = load_reference(asset)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    method = config["method"]
    seed = 5100000 + 100 * config["dimension"] + config["dataset_seed"]
    population = dict(status="not_certified", scope="random batch full-history matrix certificate not computed")
    if method == "exact":
        start = time.perf_counter()
        x = problem.sample(np.random.default_rng(seed), config["particles"])
        seconds = time.perf_counter() - start
        w, logz, records = np.full(len(x), 1 / len(x)), None, []
        population = dict(status="exact_target", reference_solver="finite sign enumeration")
    elif method == "annealed_smc":
        (x, w, logz, records), seconds = timed_sample(annealed_smc, problem, config["particles"], seed,
                     stages=config["stages"], proposal_scale=config["proposal_scale"], device="cuda")
        population = dict(status="finite", scope="bounded likelihood and invariant pCN/Metropolis kernels")
    else:
        grid = np.linspace(math.sqrt(20.), 0., config["steps"] + 1) ** 2
        (x, w, logz, records, A), seconds = timed_sample(diffusion_sample, problem, grid, config["particles"], seed,
                                    method, batch=config["batch"], device="cuda")
        if method != "unbiased":
            population = certificate(A, grid)
    np.savez_compressed(output / "samples.npz", samples=x, weights=w)
    write_json(output / "steps.json", records)
    write_json(output / "certificate.json", population)
    report = dict(config, seed=seed, seconds=seconds, seconds_including_preparation=seconds, log_normalizer=logz,
                  status="completed", **metrics(x, w, reference, problem))
    write_json(output / "summary.json", report)
    write_json(output / "receipt.json", dict(files={p.name: sha256(p) for p in sorted(output.iterdir())}))
    (output / "done").write_text("completed\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["prepare", "run"], required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--hours", type=float, default=2)
    args = parser.parse_args()
    if not 0 < args.hours <= 2:
        raise ValueError("maximum runtime is two hours")
    torch.set_num_threads(1)
    args.root.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=4) as pool:
            for name in pool.map(prepare_one, [(args.root, c) for phase in ["development", "confirmation"] for c in problems(phase)]):
                print(name, flush=True)
        write_json(args.root / "preparation.json", dict(status="completed", seconds=time.perf_counter() - start, commit=args.commit, assets=88))
        return
    if not torch.cuda.is_available() or not (args.root / "preparation.json").exists():
        raise RuntimeError("prepared assets and CUDA are required")
    start = time.perf_counter()
    sources = {p.name: sha256(p) for p in Path(".").glob("*sensor*20260921.py")}
    sources.update({p: sha256(p) for p in ["matrix_tail_certificate_20260921.py", "run_solid_20260920.py", "run_anchored_confirmation_20260921.py", "docs/SENSOR_PROTOCOL_20260921.md"]})
    write_json(args.root / "runtime.json", runtime())
    warm = make_problem(999, groups=4, dimension=2)
    for method in ["full", "tail_fixed", "tail_anchored", "unbiased"]:
        diffusion_sample(warm, [20., 19.99], 128, 1, method, device="cuda")
    annealed_smc(warm, 128, 1, stages=2, device="cuda")
    selection = None
    for phase in ["development", "confirmation"]:
        phase_root = args.root / phase
        phase_root.mkdir(exist_ok=False)
        configs = settings()
        if phase == "confirmation":
            ids = {FULL_ID, "tail_fixed", "unbiased"} | {choice["selected"]["setting_id"] for choice in selection["choices"].values()}
            configs = [s for s in configs if s["setting_id"] in ids] + [dict(method="exact", particles=16384, setting_id="exact")]
        cells = [dict(c, **s, phase=phase, line="sensor", cell_id=i) for i, (c, s) in enumerate(itertools.product(problems(phase), configs))]
        order = np.random.default_rng(20260928 if phase == "development" else 20260929).permutation(len(cells)).tolist()
        write_json(phase_root / "manifest.json", dict(commit=args.commit, sources=sources, cells=cells, order=order))
        rows = []
        for index, cell_id in enumerate(order):
            if time.perf_counter() - start > args.hours * 3600:
                write_json(args.root / "state.json", dict(status="time_limit", phase=phase, completed=index, expected=len(cells)))
                raise SystemExit(2)
            c = cells[cell_id]
            asset = args.root / "assets" / asset_name(c)
            prepare_one((args.root, c))
            report = run_one(c, asset, phase_root / "cells" / f"cell_{cell_id:04d}")
            rows.append(report)
            write_json(args.root / "state.json", dict(status="running", phase=phase, completed=index + 1, expected=len(cells)))
            print(json.dumps({key: report[key] for key in ["phase", "cell_id", "setting_id", "sliced_w1_32", "seconds"]}), flush=True)
        (phase_root / "done").write_text("completed\n")
        if phase == "development":
            selection = choose(rows)
            write_json(args.root / "selection.json", selection)
    write_json(args.root / "state.json", dict(status="completed", seconds=time.perf_counter() - start))


if __name__ == "__main__":
    main()
