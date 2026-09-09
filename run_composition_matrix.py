import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

import torch

from composition_benchmark import run_cell, runtime, self_check


def matrix():
    configs = []
    for family, groups, dimension, steps, seed, method in itertools.product(
        ["gaussian", "mixture", "weak_mixture"], [16, 64], [1, 8], [128, 512, 2048], range(5),
        ["full", "unweighted", "naive", "unbiased", "cumulant", "cv", "cv_cumulant"],
    ):
        configs.append({"family": family, "groups": groups, "dimension": dimension, "steps": steps, "seed": seed, "method": method, "batch": 4, "particles": 8192, "u_max": 20.0, "ess_threshold": 0.5, "diffusion": 1.0})
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=28)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    configs = matrix()
    if args.manifest_only:
        manifest = {"commit": args.commit, "expected_cells": len(configs), "tasks": args.tasks, "concurrency": 4, "task_hours_limit": 0.5, "maximum_allocated_gpu_hours": 14, "protocol": "docs/COMPOSITION_PROTOCOL.md", "cells": configs}
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps({k: v for k, v in manifest.items() if k != "cells"}))
        return
    if args.device == "cuda":
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() == 1, "Slurm must allocate one visible GPU per task"
    task_root = root / f"task_{args.task:03d}"
    task_root.mkdir(exist_ok=False)
    (task_root / "runtime.json").write_text(json.dumps(runtime(), indent=2))
    if args.smoke:
        self_check(task_root / "self_check.json")
        configs = [dict(configs[0], groups=16, dimension=1, particles=1024, steps=128, method=method) for method in ["full", "unbiased", "cv"]]
        selected = list(enumerate(configs))
    else:
        selected = [(i, config) for i, config in enumerate(configs) if i % args.tasks == args.task]
    start = time.perf_counter()
    for index, config in selected:
        config = dict(config, device=args.device, runtime=runtime(), commit=args.commit, cell_id=index)
        run_cell(config, task_root / f"cell_{index:04d}")
    summary = {"expected_cells": len(selected), "completed_cells": len(list(task_root.glob("cell_*/done"))), "seconds": time.perf_counter() - start, "status": "completed"}
    assert summary["expected_cells"] == summary["completed_cells"]
    (task_root / "summary.json").write_text(json.dumps(summary, indent=2))
    (task_root / "done").write_text("completed\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
