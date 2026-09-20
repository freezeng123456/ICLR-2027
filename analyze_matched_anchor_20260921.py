import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from analyze_dual_20260921 import independent_joint, validate_inputs
from analyze_paper_anchor_20260921 import aggregate, source_checks, validate_summary_rows, write_cells
from audit_extension_results import independent_metrics
from run_dual_20260921 import asset_name, load_npz
from run_solid_20260920 import sha256, write_json


def check_target(job):
    root, base, independent, cells, original_path = job
    root, base, independent = Path(root), Path(base), Path(independent)
    name = asset_name(cells[0])
    reference = load_npz(independent / "references" / f"{name}.npz")
    # 同一目标的参考积分已在父审计中独立验证，复用其固定输入。
    true = load_npz(base / "assets" / name / "true_reference.npz")
    original_root = Path(original_path)
    original_certificate = json.loads((original_root / "certificate.json").read_text())
    maximum = 0.
    for config in cells:
        cell = root / "confirmation" / "cells" / f"cell_{config['cell_id']:04d}"
        arrays, summary = load_npz(cell / "samples.npz"), json.loads((cell / "summary.json").read_text())
        values = independent_metrics(arrays["samples"], arrays["weights"], reference)
        values.update(independent_joint(arrays["samples"], arrays["weights"], reference))
        values.update({"true_" + key: value for key, value in independent_metrics(arrays["samples"], arrays["weights"], true).items()})
        for key, value in values.items():
            delta = abs(value - summary[key])
            maximum = max(maximum, delta)
            if delta > (2e-5 if "w1" in key else 4e-6):
                raise RuntimeError(f"metric discrepancy {name} {key} {delta}")
        if json.loads((cell / "certificate.json").read_text()) != original_certificate:
            raise RuntimeError("same-grid certificate differs from independently audited original")
        if config["method"] == "tail_anchored":
            saved = load_npz(original_root / "samples.npz")
            for key in arrays:
                np.testing.assert_allclose(arrays[key], saved[key], atol=1e-12, rtol=1e-12)
    return dict(asset=name, cells=len(cells), maximum_metric_discrepancy=maximum)


def main():
    parser = argparse.ArgumentParser()
    for name in ["root", "base", "base-audit", "source", "training-root", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    prior_audit = json.loads((args.base_audit / "audit.json").read_text())
    if prior_audit["status"] != "passed":
        raise RuntimeError("original independent audit must pass first")
    manifest = json.loads((args.root / "confirmation" / "manifest.json").read_text())
    if manifest["parent_manifest_sha256"] != sha256(args.base / "confirmation" / "manifest.json"):
        raise RuntimeError("parent manifest mismatch")
    sources = source_checks(args.source, manifest)
    inputs = validate_inputs(args.root, "confirmation", manifest, args.training_root)
    rows = [json.loads(p.read_text()) for p in sorted((args.root / "confirmation" / "cells").glob("*/summary.json"))]
    validate_summary_rows(rows, manifest, args.root, "confirmation")
    if len(rows) != 300 or {(r["training_seed"], r["dataset_seed"], r["method"]) for r in rows} != {
        (t, d, m) for t in range(5) for d in range(900, 920) for m in ["full", "factorized_full", "tail_anchored"]}:
        raise RuntimeError("matched-budget cohort mismatch")
    if any(r["particles"] != 8192 or r["steps"] != 1024 for r in rows):
        raise RuntimeError("budgets differ")
    for row in rows:
        processes = json.loads((args.root / f"gpu_{row['cell_id']:04d}.json").read_text())
        if len(processes["before"]) != 1 or len(processes["after"]) != 1:
            raise RuntimeError("uncontrolled timing")
    args.output.mkdir(parents=True, exist_ok=True)
    statistics = aggregate(rows, "confirmation", manifest, canonical=False)
    statistics["evidence_status"] = "post-hoc matched-budget ablation on the same 100 targets"
    write_json(args.output / "statistics.json", statistics)
    write_cells(rows, args.output)
    original_manifest = json.loads((args.base / "confirmation" / "manifest.json").read_text())
    original_paths = {asset_name(c): args.base / "confirmation" / "cells" / f"cell_{c['cell_id']:04d}"
                      for c in original_manifest["cells"] if c["setting_id"] == "anchor_n8192_k1024_m8"}
    jobs = [(str(args.root), str(args.base), str(args.base_audit), [c for c in manifest["cells"] if asset_name(c) == name], str(original_paths[name]))
            for name in sorted({asset_name(c) for c in manifest["cells"]})]
    with ProcessPoolExecutor(max_workers=4) as pool:
        checks = list(pool.map(check_target, jobs))
    result = dict(status="passed", source_commit=manifest["commit"], sources=sources, inputs=inputs, checks=checks,
                  parent_audit_sha256=sha256(args.base_audit / "audit.json"), auditor_sha256=sha256(__file__),
                  scope="all recovered particle metrics and exact anchor replay; same-target reference integration and same-grid population certificate reused from passed parent audit")
    write_json(args.output / "audit.json", result)
    print(json.dumps(dict(status="passed", cells=300)), flush=True)


if __name__ == "__main__":
    main()
