import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd
import torch

from audit_composition_results import independent_metrics, one_step_diagnostics
from run_composition_matrix import matrix as main_matrix
from run_composition_supplement import matrix as supplement_matrix


def execute(command, directory, log):
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with log.open("w") as stream:
        subprocess.run(command, cwd=directory, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    assert not subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    for name, factory in [("main", main_matrix), ("supplement", supplement_matrix)]:
        manifest = json.loads((root / "results" / f"composition_{name}_20260909" / "run_manifest.json").read_text())
        configs = factory()
        assert len(configs) == len(manifest["cells"])
        assert all(all(saved[k] == v for k, v in expected.items()) for expected, saved in zip(configs, manifest["cells"]))

    # 在完整历史源码的干净解压目录中重跑正式参数，仅将执行设备改为 CPU。
    sources = {}
    for name, historical_commit in [("main", "185e953d57699e678314593985abe20c62d7d129"), ("supplement", "a66b5e3c37c8a33201d515c128157bb8028554e0")]:
        archive = output / f"{name}-source.tar"
        subprocess.run(["git", "archive", "--format=tar", "--output", str(archive), historical_commit], cwd=root, check=True)
        directory = output / f"{name}-source"
        directory.mkdir()
        with tarfile.open(archive) as stream:
            stream.extractall(directory, filter="data")
        sources[name] = directory

    benchmark = []
    for name, family, method in [("main", "gaussian", "full"), ("supplement", "weak_mixture", "tail")]:
        cell = output / f"rerun-{family}-{method}"
        command = [sys.executable, "composition_benchmark.py", "--output", str(cell), "--device", "cpu", "--family", family, "--method", method, "--groups", "16", "--dimension", "1", "--particles", "8192", "--steps", "512", "--batch", "4", "--seed", "0"]
        execute(command, sources[name], output / f"{family}.log")
        summary = json.loads((cell / "summary.json").read_text())
        independent, dx, joint = independent_metrics(cell)
        for key, value in independent.items():
            assert abs(value - summary[key]) <= (dx if key == "w1_mean" else 2e-6)
        assert (cell / "done").is_file()
        benchmark.append({"family": family, "method": method, "w1_mean": summary["w1_mean"], "independent_metrics": independent, "joint": joint})

    one_step = []
    for method, batch in [("full", 2), ("unbiased", 2), ("unbiased", 32)]:
        cell = output / f"one-step-{method}-{batch}"
        command = [sys.executable, "one_step_experiment.py", "--output", str(cell), "--device", "cpu", "--method", method, "--particles", "1048576", "--batch", str(batch), "--seed", "0"]
        execute(command, sources["supplement"], output / f"one-step-{method}-{batch}.log")
        config = json.loads((cell / "config.json").read_text())
        summary = json.loads((cell / "summary.json").read_text())
        one_step_diagnostics(cell, config, summary)
        if method == "full":
            assert abs(summary["log_normalizer"] - summary["full_exact_log_normalizer"]) < 0.005
            assert abs(summary["weighted_variance"] - summary["full_exact_variance"]) < 0.01
        one_step.append({key: summary[key] for key in ["method", "batch", "particles", "log_normalizer", "weighted_variance", "ess_fraction", "potential_mse", "nonintegrable_batch_count"]})

    checks = [
        ["composition_benchmark.py", "--self-check", str(output / "estimator.json")],
        ["gaussian_integrability.py", "--output", str(output / "integrability.json")],
        ["operator_audit.py", "--output", str(output / "operator")],
        ["plot_composition_results.py", "--main", str(root / "results/composition_main_20260909"), "--supplement", str(root / "results/composition_supplement_20260909"), "--output", str(output / "figures")],
    ]
    for index, command in enumerate(checks):
        execute([sys.executable, *command], root, output / f"check-{index}.log")
    for name in ["one_step", "accuracy_time", "integrability_grid"]:
        assert (output / "figures" / f"{name}.png").read_bytes() == (root / "manuscript/figures" / f"{name}.png").read_bytes()
    for name in ["benchmark_aggregate", "matched_comparison", "one_step_aggregate"]:
        pd.testing.assert_frame_equal(pd.read_csv(output / "figures" / f"{name}.csv"), pd.read_csv(root / "manuscript/figures" / f"{name}.csv"))
    report = {"status": "passed", "analysis_commit": commit, "python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "device": "cpu", "benchmark_reruns": benchmark, "one_step_reruns": one_step, "passed": ["clean Git checkout", "both fixed manifests reproduced", "historical-source representative CPU reruns", "independent rerun metrics", "exact estimator and integrability checks", "local operator checks", "all three figure PNGs byte-identical", "all three aggregate CSVs equal"], "failed": [], "blocked": [], "not_run": ["bitwise CPU versus SCNet GPU equality", "full repeat of 1980 SCNet cells"], "scope": "Reproducible code and analysis from a clean checkout, with representative historical-source CPU reruns; raw SCNet archives are independently verified by audit_composition_results.py."}
    (output / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
