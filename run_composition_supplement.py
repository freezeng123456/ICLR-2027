import argparse
import itertools
import json
from pathlib import Path
import time

import torch

from composition_benchmark import run_cell, runtime, self_check
from one_step_experiment import run_one_step


def matrix():
    configs = []
    for family, groups, dimension, steps, seed, method in itertools.product(
        ["mixture", "weak_mixture"], [16, 64], [1, 8], [128, 512, 2048], range(5), ["tail", "tail_cumulant"],
    ):
        configs.append({"kind": "tail", "family": family, "groups": groups, "dimension": dimension, "steps": steps, "seed": seed, "method": method, "batch": 4, "particles": 8192, "u_max": 20.0, "ess_threshold": 0.5, "diffusion": 1.0})
    for particles, seed in itertools.product([1024, 16384, 262144, 1048576], range(20)):
        configs.append({"kind": "one_step", "particles": particles, "seed": seed, "batch": 2, "method": "full"})
        for batch in [2, 4, 8, 16, 32]:
            configs.append({"kind": "one_step", "particles": particles, "seed": seed, "batch": batch, "method": "unbiased"})
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)
    configs = matrix()
    if args.manifest_only:
        manifest = {"commit": args.commit, "expected_cells": len(configs), "tasks": args.tasks, "concurrency": 4, "task_hours_limit": 1 / 3, "maximum_allocated_gpu_hours": 4, "cells": configs}
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps({k: v for k, v in manifest.items() if k != "cells"}))
        return
    if args.device == "cuda":
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    task_root = root / f"task_{args.task:03d}"
    task_root.mkdir(exist_ok=False)
    (task_root / "runtime.json").write_text(json.dumps(runtime(), indent=2))
    if args.smoke:
        self_check(task_root / "self_check.json")
        selected = [(0, dict(configs[0], particles=1024, steps=128)), (1, {"kind": "one_step", "particles": 16384, "seed": 0, "batch": 2, "method": "unbiased"})]
    else:
        selected = [(i, config) for i, config in enumerate(configs) if i % args.tasks == args.task]
    started = time.perf_counter()
    for i, config in selected:
        config = dict(config, device=args.device, commit=args.commit, runtime=runtime(), cell_id=i)
        if config["kind"] == "tail":
            run_cell(config, task_root / f"cell_{i:04d}")
        else:
            run_one_step(config, task_root / f"cell_{i:04d}")
    summary = {"status": "completed", "expected_cells": len(selected), "completed_cells": len(list(task_root.glob("cell_*/done"))), "seconds": time.perf_counter() - started}
    assert summary["expected_cells"] == summary["completed_cells"]
    (task_root / "summary.json").write_text(json.dumps(summary, indent=2))
    (task_root / "done").write_text("completed\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
