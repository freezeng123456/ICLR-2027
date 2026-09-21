import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.optimize import brentq

from confseq.betting import betting_mart
from confseq.predmix import predmix_empbern_twosided_cs

from run_decision_evidence_20260922 import rare_problems


def official_interval(values, method, alpha=0.025):
    values = np.asarray(values)
    if method == "empbern":
        lower, upper = predmix_empbern_twosided_cs(values, alpha=alpha, running_intersection=True)
        return float(lower[-1]), float(upper[-1])
    if method != "betting":
        raise ValueError("Unknown official CS method")
    tail_alpha = alpha / 2
    threshold = 1 / tail_alpha

    def excess(mean, orientation):
        with np.errstate(over="ignore"):
            wealth = betting_mart(values, mean, alpha=tail_alpha, theta=orientation, trunc_scale=0.5)
        return float(wealth[-1] - threshold)

    lower = 0.0 if excess(0, 1) <= 0 else brentq(lambda mean: excess(mean, 1), 0, 1, xtol=1e-10)
    upper = 1.0 if excess(1, 0) <= 0 else brentq(lambda mean: excess(mean, 0), 0, 1, xtol=1e-10)
    # 向外舍入，避免数值求根缩窄置信集合。
    return max(0.0, lower - 1e-9), min(1.0, upper + 1e-9)


def run_control(problem, method, seed, budget=4096):
    rng = np.random.default_rng(seed)
    positive = list(problem.sample(rng, True, 2))
    negative = list(problem.sample(rng, False, 2))
    counts = [2, 2]
    lower = [0.0, 0.0]
    upper = [1.0, 1.0]
    scales = [problem.positive_scale, problem.negative_scale]
    trace = []
    observations = [(0, value) for value in positive] + [(1, value) for value in negative]
    decision = 0
    active = [0, 1]
    while True:
        for index in active:
            data = positive if index == 0 else negative
            current_lower, current_upper = official_interval(data, method)
            lower[index] = max(lower[index], current_lower)
            upper[index] = min(upper[index], current_upper)
        difference_lower = scales[0] * lower[0] - scales[1] * upper[1]
        difference_upper = scales[0] * upper[0] - scales[1] * lower[1]
        calls = sum(counts)
        trace.append((calls, *counts, *lower, *upper, difference_lower, difference_upper))
        if difference_lower > 0:
            decision = 1
            break
        if difference_upper < 0:
            decision = -1
            break
        if calls >= budget:
            break
        widths = [scale * (hi - lo) for scale, lo, hi in zip(scales, lower, upper)]
        index = int(np.argmax(widths))
        count = min(16, budget - calls)
        new_values = list(problem.sample(rng, index == 0, count))
        observations.extend((index, value) for value in new_values)
        if index == 0:
            positive.extend(new_values)
        else:
            negative.extend(new_values)
        counts[index] += count
        active = [index]
    truth = int(np.sign(problem.integral))
    return {
        "problem": problem.name, "method": method, "seed": seed, "calls": sum(counts),
        "decision": decision, "truth": truth, "certified": decision != 0,
        "wrong": decision != 0 and (truth == 0 or decision != truth),
        "positive_queries": counts[0], "negative_queries": counts[1],
        "trace": trace, "observations": observations,
    }


def run_one(arguments):
    problem, method, seed, root = arguments
    directory = Path(root) / problem.name / method / f"seed_{seed:05d}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(json.dumps({"problem": asdict(problem), "method": method, "seed": seed, "budget": 4096}, indent=2) + "\n")
    started = time.perf_counter()
    result = run_control(problem, method, seed)
    result["runtime_seconds"] = time.perf_counter() - started
    trace = np.asarray(result.pop("trace"), dtype=float)
    observations = np.asarray(result.pop("observations"), dtype=float)
    np.savez_compressed(directory / "trace.npz", trace=trace, observations=observations)
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (directory / "done").write_text("complete\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--method", choices=["betting", "empbern"], default="betting")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    selected = rare_problems()
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    confseq_commit = subprocess.check_output(["git", "-C", "work/confseq", "rev-parse", "HEAD"], text=True).strip()
    if confseq_commit != "5ffe733ca2447a2e28c2c91f3b00086173f2ab2c":
        raise ValueError("Official baseline revision differs from protocol")
    jobs = [(problem, args.method, seed, str(root)) for problem in selected for seed in range(2000, 2000 + args.repetitions)]
    provenance = {
        "source_commit": source_commit, "confseq_commit": confseq_commit,
        "confseq_repository": "https://github.com/gostevehoward/confseq",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "expected_cells": len(jobs), "alpha_total": 0.05, "alpha_per_stratum": 0.025,
        "allocation": "maximum scaled CS width", "check_batch_size": 16,
        "initial_observations": "included in official confidence sequences and total cost",
        "method": args.method,
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        summaries = list(executor.map(run_one, jobs, chunksize=1))
    (root / "summaries.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (root / "done").write_text("complete\n")
    for problem in selected:
        rows = [row for row in summaries if row["problem"] == problem.name]
        print(json.dumps({"problem": problem.name, "method": args.method, "calls": float(np.mean([row["calls"] for row in rows])), "certified": sum(row["certified"] for row in rows), "wrong": sum(row["wrong"] for row in rows)}))
