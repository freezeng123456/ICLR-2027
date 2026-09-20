import argparse
import json
from pathlib import Path

import numpy as np

from run_solid_20260920 import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_manifest = args.original / "confirmation" / "manifest.json"
    manifest = json.loads((args.replay / "confirmation" / "manifest.json").read_text())
    original_manifest = json.loads(source_manifest.read_text())
    assert manifest["original_manifest_sha256"] == sha256(source_manifest)
    assert manifest["original_commit"] == original_manifest["commit"]
    assert len(manifest["cells"]) == len(original_manifest["cells"]) == 150
    tolerances = dict(sample_max_abs=1e-8, weight_max_abs=1e-10, logz_abs=1e-8)
    maxima = {key: 0. for key in tolerances}
    counts = {}
    for config, original_config in zip(manifest["cells"], original_manifest["cells"]):
        assert all(original_config[key] == value for key, value in config.items() if key != "line")
        paths = [root / "confirmation" / "cells" / f"cell_{config['cell_id']:04d}" for root in [args.original, args.replay]]
        assert all((path / "done").read_text() == "completed\n" for path in paths)
        arrays = [np.load(path / "samples.npz", allow_pickle=False) for path in paths]
        summaries = [json.loads((path / "summary.json").read_text()) for path in paths]
        deltas = dict(sample_max_abs=float(np.max(np.abs(arrays[0]["samples"] - arrays[1]["samples"]))),
                      weight_max_abs=float(np.max(np.abs(arrays[0]["weights"] - arrays[1]["weights"]))),
                      logz_abs=abs(summaries[0]["log_normalizer"] - summaries[1]["log_normalizer"]))
        saved = json.loads((paths[1] / "equivalence.json").read_text())
        assert saved["passed"] and saved["cell_id"] == config["cell_id"] and saved["method"] == config["method"]
        for key, delta in deltas.items():
            assert np.isfinite(delta) and delta <= tolerances[key]
            assert delta == saved[key]
            maxima[key] = max(maxima[key], delta)
        counts[config["method"]] = counts.get(config["method"], 0) + 1
    assert counts == dict(full=50, tail_fixed=50, tail_anchored=50)
    global_report = json.loads((args.replay / "equivalence.json").read_text())
    assert global_report["status"] == "passed" and len(global_report["cells"]) == 150
    assert global_report["maxima"] == maxima
    result = dict(status="passed", cells=150, method_counts=counts, maxima=maxima, tolerances=tolerances,
                  source_sha256=sha256(__file__),
                  scope="independent local comparison of every recovered particle, weight and log normalizer; same cohort, no additional independent data")
    write_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
