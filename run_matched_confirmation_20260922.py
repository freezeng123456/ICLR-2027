import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import scipy

from decision_evidence_20260922 import choose_design, bounded_empirical_bet
from official_cs_control_20260922 import official_interval
from run_decision_evidence_20260922 import rare_problems


METHODS = ("growth", "uniform", "variance", "paired", "betting")
CONFSEQ_COMMIT = "5ffe733ca2447a2e28c2c91f3b00086173f2ab2c"


def trial(problem, method, seed, budget=4096):
    if method not in METHODS or budget < 20 or budget % 2:
        raise ValueError("Unknown method or invalid even query budget")
    rng = np.random.default_rng(seed)
    positive = list(problem.sample(rng, True, 2))
    negative = list(problem.sample(rng, False, 2))
    observations = [(0, float(value)) for value in positive] + [(1, float(value)) for value in negative]
    a, b = problem.positive_scale, problem.negative_scale
    differences = list(a * np.asarray(positive) - b * np.asarray(negative))
    lower, upper = [0.0, 0.0], [1.0, 1.0]
    log_positive = log_negative = 0.0
    trace, checkpoints = [], []
    calls, updates, decision = 4, 0, 0
    boundary = np.log(40)
    while calls < budget and decision == 0:
        count = min(16, budget - calls)
        updates += 1
        if method == "betting":
            index = int(np.argmax([a * (upper[0] - lower[0]), b * (upper[1] - lower[1])]))
            values = problem.sample(rng, index == 0, count)
            observations.extend((index, float(value)) for value in values)
            (positive if index == 0 else negative).extend(values)
            calls += count
            post_pilot = (positive if index == 0 else negative)[2:]
            lo, hi = official_interval(post_pilot, "betting")
            lower[index], upper[index] = max(lower[index], lo), min(upper[index], hi)
            lo_gap, hi_gap = a * lower[0] - b * upper[1], a * upper[0] - b * lower[1]
            trace.append((calls, len(positive), len(negative), *lower, *upper, lo_gap, hi_gap))
            decision = 1 if lo_gap > 0 else (-1 if hi_gap < 0 else 0)
        else:
            if method == "paired":
                mass = 0.5
                plus, _ = bounded_empirical_bet(differences, -b)
                minus, _ = bounded_empirical_bet([-value for value in differences], -a)
                selections = range(count // 2)
            else:
                mass, plus, minus = choose_design(method, positive, negative, a, b)
                selections = rng.random(count) < mass
            for selected in selections:
                if method == "paired":
                    p = float(problem.sample(rng, True, 1)[0])
                    n = float(problem.sample(rng, False, 1)[0])
                    observations.extend([(0, p), (1, n)])
                    positive.append(p)
                    negative.append(n)
                    weighted = a * p - b * n
                    differences.append(weighted)
                    calls += 2
                else:
                    raw = float(problem.sample(rng, bool(selected), 1)[0])
                    observations.append((0 if selected else 1, raw))
                    (positive if selected else negative).append(raw)
                    weighted = a * raw / mass if selected else -b * raw / (1 - mass)
                    calls += 1
                if min(plus * weighted, -minus * weighted) <= -1:
                    raise ValueError("Unsafe evidence increment")
                log_positive += float(np.log1p(plus * weighted))
                log_negative += float(np.log1p(-minus * weighted))
                trace.append((calls, mass, plus, minus, weighted, log_positive, log_negative))
            decision = 1 if log_positive >= boundary else (-1 if log_negative >= boundary else 0)
        checkpoints.append(calls)
    truth = int(np.sign(problem.integral))
    return {
        "problem": problem.name, "method": method, "seed": seed, "calls": calls,
        "decision": decision, "truth": truth, "wrong": decision != 0 and (truth == 0 or decision != truth),
        "certified": decision != 0, "design_updates": updates, "certificate_checks": len(checkpoints),
        "positive_queries": len(positive), "negative_queries": len(negative),
        "trace": trace, "observations": observations, "checkpoints": checkpoints,
    }


def run_one(arguments):
    problem, method, seed, budget, root = arguments
    directory = Path(root) / problem.name / method / f"seed_{seed:05d}"
    directory.mkdir(parents=True, exist_ok=False)
    config = {"problem": asdict(problem), "method": method, "seed": seed, "budget": budget, "check_every_queries": 16, "pilot_queries": 4, "alpha": 0.05}
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    started = time.perf_counter()
    result = trial(problem, method, seed, budget)
    result["runtime_seconds"] = time.perf_counter() - started
    arrays = {key: np.asarray(result.pop(key), dtype=float) for key in ["trace", "observations", "checkpoints"]}
    np.savez_compressed(directory / "trace.npz", **arrays)
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (directory / "done").write_text("complete\n")
    return result


def source_provenance():
    root = Path(__file__).resolve().parent
    names = ["run_matched_confirmation_20260922.py", "decision_evidence_20260922.py", "official_cs_control_20260922.py", "run_decision_evidence_20260922.py"]
    git = ["git"]
    if "SOURCE_GIT_DIR" in os.environ:
        git.append("--git-dir=" + os.environ["SOURCE_GIT_DIR"])
    revision = os.environ.get("SOURCE_COMMIT", "HEAD")
    commit = subprocess.check_output(git + ["rev-parse", revision + "^{commit}"], cwd=root, text=True).strip()
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    for name, digest in hashes.items():
        committed = subprocess.check_output(git + ["show", f"{commit}:{name}"], cwd=root)
        if hashlib.sha256(committed).hexdigest() != digest:
            raise ValueError(f"Uncommitted experiment source: {name}")
    dependency = Path(os.environ.get("CONFSEQ_ROOT", str(root / "work/confseq"))).resolve()
    official = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dependency, text=True).strip()
    changes = subprocess.check_output(["git", "status", "--porcelain"], cwd=dependency, text=True)
    if official != CONFSEQ_COMMIT or changes.strip():
        raise ValueError("Official baseline revision is not clean and pinned")
    return {"source_commit": commit, "source_hashes": hashes, "confseq_commit": official, "source_archive_sha256": os.environ.get("SOURCE_ARCHIVE_SHA256")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=9000)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--problem", action="append")
    parser.add_argument("--method", choices=METHODS, action="append")
    args = parser.parse_args()
    if min(args.repetitions, args.workers) < 1:
        raise ValueError("Positive repetitions and workers required")
    provenance = source_provenance()
    selected = [problem for problem in rare_problems() if not args.problem or problem.name in args.problem]
    if not selected or (args.problem and len(selected) != len(set(args.problem))):
        raise ValueError("Invalid problem selection")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    jobs = [(problem, method, seed, args.budget, str(root)) for problem in selected for seed in range(args.seed_start, args.seed_start + args.repetitions) for method in (args.method or METHODS)]
    provenance.update({
        "expected_cells": len(jobs), "arguments": {**vars(args), "output": str(root)},
        "python": platform.python_version(), "python_executable": os.path.realpath(os.sys.executable),
        "numpy": np.__version__, "scipy": scipy.__version__, "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "scope": "fresh seeds for previously selected synthetic problem family; matched checkpoint protocol",
    })
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        summaries = []
        for summary in pool.map(run_one, jobs, chunksize=1):
            summaries.append(summary)
            if len(summaries) % 100 == 0:
                print(json.dumps({"completed": len(summaries), "expected": len(jobs)}), flush=True)
    (root / "summaries.json").write_text(json.dumps(summaries, indent=2) + "\n")
    completion = {"completed_cells": len(summaries), "expected_cells": len(jobs), "runtime_seconds": time.perf_counter() - started}
    (root / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    (root / "done").write_text("complete\n")
    print(json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
