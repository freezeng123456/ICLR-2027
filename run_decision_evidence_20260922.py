import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import scipy

from decision_evidence_20260922 import (
    BoundedStratumProblem,
    growth_optimal_mass,
    numerical_oracle_mass,
    oracle_growth,
    run_bounded_trial,
    run_stratified_trial,
)


def problems():
    items = []
    for ratio in [1.0, 1.2, 2.0, 9.0, 100.0]:
        items.append(BoundedStratumProblem(f"concentrated_r{ratio:g}", ratio, 1, (8, 2), (8, 2)))
    for ratio in [2.0, 9.0]:
        items.append(BoundedStratumProblem(f"variable_r{ratio:g}", ratio, 1, (0.4, 0.1), (8, 2)))
    originals = list(items)
    for item in originals:
        if item.integral != 0:
            items.append(
                BoundedStratumProblem(
                    item.name + "_mirror", item.negative_scale, item.positive_scale,
                    item.negative_beta, item.positive_beta,
                )
            )
    return items


def rare_problems():
    return [
        BoundedStratumProblem(f"rare_p{probability:g}_b{contribution:g}", 1, contribution / probability, (8, 2), (probability, 1 - probability))
        for probability in [0.002, 0.02]
        for contribution in [0.2, 0.4, 0.8, 1.2]
    ]


def run_one(arguments):
    problem, method, seed, budget, alpha, root = arguments
    directory = Path(root) / problem.name / method / f"seed_{seed:05d}"
    directory.mkdir(parents=True, exist_ok=False)
    config = {"problem": asdict(problem), "method": method, "seed": seed, "max_samples": budget, "alpha": alpha}
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    started = time.perf_counter()
    runner = run_stratified_trial if method in {"paired", "upper_bound"} else run_bounded_trial
    result = runner(problem, method, seed, max_samples=budget, alpha=alpha)
    result["runtime_seconds"] = time.perf_counter() - started
    trace = np.asarray(result.pop("trace"), dtype=np.float64)
    np.savez_compressed(directory / "trace.npz", trace=trace)
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (directory / "done").write_text("complete\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--problem", action="append")
    parser.add_argument("--suite", choices=["ordinary", "rare"], default="ordinary")
    parser.add_argument("--method", action="append", choices=["uniform", "variance", "growth", "paired", "upper_bound"])
    args = parser.parse_args()
    if args.repetitions < 1 or args.workers < 1:
        raise ValueError("Positive repetitions and workers required")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    available = rare_problems() if args.suite == "rare" else problems()
    selected = [item for item in available if not args.problem or item.name in args.problem]
    if not selected or (args.problem and len(selected) != len(set(args.problem))):
        raise ValueError("Unknown or empty problem selection")
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_files = [Path(__file__), Path(__file__).with_name("decision_evidence_20260922.py")]
    source_hashes = {item.name: hashlib.sha256(item.read_bytes()).hexdigest() for item in source_files}
    provenance = {
        "source_commit": source_sha,
        "source_hashes": source_hashes,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "expected_cells": len(selected) * len(args.method or ["uniform", "variance", "growth", "paired", "upper_bound"]) * args.repetitions,
        "scope": "synthetic bounded-stratum development; not an application benchmark",
        "arguments": {**vars(args), "output": str(root)},
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    oracle = []
    for ratio in [1.001, 1.2, 2, 9, 100, 10000]:
        optimum = float(growth_optimal_mass(ratio, 1))
        variance = ratio / (ratio + 1)
        oracle.append({
            "ratio": ratio, "growth_mass": optimum, "numerical_mass": numerical_oracle_mass(ratio, 1),
            "variance_mass": variance,
            "growth_optimum": oracle_growth(ratio, 1, optimum),
            "growth_variance": oracle_growth(ratio, 1, variance),
        })
    (root / "oracle_mechanism.json").write_text(json.dumps(oracle, indent=2) + "\n")
    jobs = [
        (problem, method, seed, args.max_samples, 0.05, str(root))
        for problem in selected
        for seed in range(args.seed_start, args.seed_start + args.repetitions)
        for method in (args.method or ["uniform", "variance", "growth", "paired", "upper_bound"])
    ]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        summaries = []
        for result in executor.map(run_one, jobs, chunksize=1):
            summaries.append(result)
            if len(summaries) % 25 == 0:
                print(json.dumps({"completed": len(summaries), "expected": len(jobs)}), flush=True)
    (root / "summaries.json").write_text(json.dumps(summaries, indent=2) + "\n")
    completion = {"completed_cells": len(summaries), "expected_cells": len(jobs), "runtime_seconds": time.perf_counter() - started}
    (root / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    (root / "done").write_text("complete\n")
    print(json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
