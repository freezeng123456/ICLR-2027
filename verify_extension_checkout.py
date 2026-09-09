import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd

from audit_extension_results import independent_metrics
from run_composition_extension import matrix


def execute(command, directory, log):
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with log.open("w") as stream:
        subprocess.run(command, cwd=directory, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)


def verify(raw, output):
    root = Path(__file__).resolve().parent
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((raw / "provenance.json").read_text())["commit"]
    for kind in ["oracle", "learned"]:
        manifest = json.loads((raw / kind / "manifest.json").read_text())
        assert manifest["cells"] == matrix(kind) and manifest["commit"] == frozen
    archive = output / "source.tar"
    subprocess.run(["git", "archive", "--format=tar", "--output", str(archive), frozen], cwd=root, check=True)
    source = output / "source"
    source.mkdir()
    with tarfile.open(archive) as stream:
        stream.extractall(source, filter="data")
    execute([sys.executable, "run_composition_extension.py", "--root", str(output / "checks"), "--commit", frozen, "--check", "--device", "cpu"], source, output / "checks.log")
    execute([sys.executable, "learned_sbi.py", "--root", str(output / "training-smoke"), "--seed", "0", "--device", "cpu", "--updates", "20"], source, output / "training-smoke.log")
    smoke = json.loads((output / "training-smoke/training_0/summary.json").read_text())
    assert smoke["updates"] == 20 and smoke["parameter_delta_l2"] > 0 and smoke["finite_parameters"]
    reruns = []
    for kind in ["oracle", "learned"]:
        for method in ["unbiased", "tail"]:
            candidates = matrix(kind)
            index, config = next((i, c) for i, c in enumerate(candidates) if c["method"] == method and c["sampling"] == "without_replacement" and c["dimension"] == 1 and c["steps"] == 512 and c["batch"] == 4 and (kind == "learned" or c["family"] == "weak_mixture"))
            cell = output / f"{kind}-{method}"
            configuration = output / f"{kind}-{method}.json"
            configuration.write_text(json.dumps(dict(config, cell_id=index, device="cpu", commit=frozen)))
            code = "import json,sys,torch; from pathlib import Path; from run_composition_extension import run_extension_cell; from composition_benchmark import runtime; torch.set_num_threads(1); c=json.loads(Path(sys.argv[1]).read_text()); c['runtime']=runtime(); run_extension_cell(c,Path(sys.argv[2]),sys.argv[3])"
            execute([sys.executable, "-c", code, str(configuration), str(cell), str(raw / "training")], source, output / f"{kind}-{method}.log")
            with np.load(cell / "reference.npz") as data:
                reference = {key: data[key] for key in ["grid", "density", "cdf"]}
            with np.load(cell / "samples.npz") as data:
                metrics = independent_metrics(data["samples"], data["weights"], reference)
            summary = json.loads((cell / "summary.json").read_text())
            for key, value in metrics.items():
                assert abs(value - summary[key]) < (0.0007 if key == "w1_mean" else 1e-5)
            assert (cell / "extension_done").is_file()
            reruns.append({"kind": kind, "method": method, "formal_configuration_index": index, "metrics": metrics})
    execute([sys.executable, "scan_without_replacement.py", "--output", str(output / "scan")], root, output / "scan.log")
    results = root / "results/composition_extension_20260910"
    pd.testing.assert_frame_equal(pd.read_csv(output / "scan/without_replacement_batch_scan.csv"), pd.read_csv(results / "without_replacement_batch_scan.csv"))
    analysis = output / "analysis"
    shutil.copytree(results, analysis)
    execute([sys.executable, "plot_extension_results.py", "--analysis", str(analysis), "--figures", str(output / "figures")], root, output / "figures.log")
    for name in ["learned_results", "learned_results_coarse", "extension_sensitivity", "extension_sampling"]:
        assert (output / f"figures/{name}.png").read_bytes() == (root / f"manuscript/figures/{name}.png").read_bytes()
    for name in ["oracle_grouped", "learned_grouped", "learned_composition_errors", "training_audited"]:
        pd.testing.assert_frame_equal(pd.read_csv(analysis / f"{name}.csv"), pd.read_csv(results / f"{name}.csv"))
    report = {"status": "passed", "analysis_commit": commit, "experiment_commit": frozen, "device": "cpu", "representative_formal_configuration_reruns": reruns, "training_smoke_updates": 20,
              "passed": ["clean checkout", "both complete fixed manifests", "frozen source extraction", "estimator and exact Bayes checks", "fresh training with finite changed weights", "four representative CPU trajectories with independent metrics", "936 certificate configurations reproduced", "all four extension figure PNGs byte-identical"],
              "failed": [], "blocked": [], "not_run": ["repeat of all 2080 GPU sampling cells", "repeat of five complete GPU training runs", "bitwise CPU/GPU equality"]}
    (output / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.raw_root.resolve(), args.output.resolve())
