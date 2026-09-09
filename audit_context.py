import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from pilot_context import evaluate, load_checkpoint
from train_repro import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cells = sorted(p.parent for p in args.root.glob("*/done"))
    assert len(cells) == 12
    checkpoints = {"gp64": "pfn_cond_w64.pt", "gp128": "pfn_cond_40k.pt", "jump64": "pfn_jump_w64.pt", "jump128": "pfn_jump_40k.pt"}
    results = []
    for cell in cells:
        config = json.loads((cell / "config.json").read_text())
        assert config["tasks"] == 512 and config["device"] == "cuda"
        assert config["gpu"] == "NVIDIA GeForce RTX 3080"
        for name, expected in config["source_sha256"].items():
            content = subprocess.check_output(["git", "show", f"{config['commit']}:{name}"])
            assert hashlib.sha256(content).hexdigest() == expected
        for name, expected in config["checkpoint_sha256"].items():
            assert sha256(checkpoints[name]) == expected
        record = np.load(cell / "records.npz")
        for key in record.files:
            if record[key].dtype.kind != "U":
                assert np.isfinite(record[key]).all()
        assert record["xc"].shape == (512, config["n_context"])
        assert record["xq"].shape == (512, 16)
        entry = {"cell": cell.name, "hashes_and_shapes": "passed"}
        if config["seed"] == 20260910:
            data = {key: record[key][:8] for key in ["xc", "yc", "xq"]}
            checks = {}
            for name, path in checkpoints.items():
                if f"{name}_mean" not in record:
                    continue
                prediction = evaluate(load_checkpoint(path), data, "cpu", batch_size=8)
                checks[name] = {}
                for key, value in prediction.items():
                    saved = record[f"{name}_{key}"][:8]
                    np.testing.assert_allclose(value, saved, atol=2e-4, rtol=2e-4)
                    checks[name][key] = float(np.max(np.abs(value - saved)))
            entry["cross_runtime_max_abs_error"] = checks
        results.append(entry)
    with (args.root / "slurm_accounting.tsv").open() as stream:
        jobs = [row for row in csv.DictReader(stream, delimiter="|") if "." not in row["JobID"]]
    assert len(jobs) == 12
    gpu_seconds = 0
    for job in jobs:
        assert job["State"] == "COMPLETED" and job["ExitCode"] == "0:0"
        assert "gres/gpu=1" in job["ReqTRES"] and "gres/gpu=1" in job["AllocTRES"]
        h, m, s = map(int, job["Elapsed"].split(":"))
        gpu_seconds += h * 3600 + m * 60 + s
    report = {"status": "passed", "cells": results, "completed_jobs": len(jobs), "gpu_seconds": gpu_seconds,
              "source_commit": config["commit"], "source_archive_sha256": "157a5ec23ed0153868a1f4d730340c62d5994ffe2aec4012e47e03422f7e68ce",
              "download_sha256": "9dd790a61454372d9e0f664b097cdb935a2a08626725362ab5ac1f5c49eb9281",
              "cross_runtime_scope": "first 8 tasks in each prior/context condition at seed 20260910; 3 checkpoints per condition"}
    (args.root / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "cells"}, indent=2))


if __name__ == "__main__":
    main()
