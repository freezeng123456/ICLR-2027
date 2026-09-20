import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from analyze_dual_20260921 import independent_joint, validate_inputs
from audit_extension_results import independent_metrics
from run_dual_20260921 import asset_name, load_npz
from run_solid_20260920 import sha256, write_json


def verify_cell(job):
    run, audit, phase, config = job
    run, audit = Path(run), Path(audit)
    cell = run / phase / "cells" / f"cell_{config['cell_id']:04d}"
    summary = json.loads((cell / "summary.json").read_text())
    arrays = load_npz(cell / "samples.npz")
    reference = load_npz(audit / "references" / f"{asset_name(config)}.npz")
    assert np.isfinite(arrays["samples"]).all() and np.isfinite(arrays["weights"]).all()
    assert (arrays["weights"] >= 0).all()
    np.testing.assert_allclose(arrays["weights"].sum(), 1, atol=1e-12, rtol=1e-12)
    actual = independent_metrics(arrays["samples"], arrays["weights"], reference)
    actual.update(independent_joint(arrays["samples"], arrays["weights"], reference))
    deltas = {key: abs(value - summary[key]) for key, value in actual.items()}
    for key, delta in deltas.items():
        assert delta < (2e-5 if "w1" in key else 4e-6), (config["cell_id"], key, delta)
    return deltas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--phase", choices=["development", "confirmation"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert 1 <= args.workers <= 4
    assert (args.run / args.phase / "done").read_text().strip() == "completed"
    manifest = json.loads((args.run / args.phase / "manifest.json").read_text())
    source_checks = {name: sha256(args.source / name) == digest for name, digest in manifest["sources"].items()}
    assert source_checks and all(source_checks.values())
    audit = json.loads((args.audit / "audit.json").read_text())
    assert audit["status"] == "passed" and audit["source_commit"] == manifest["commit"]
    inputs = validate_inputs(args.run, args.phase, manifest, args.training_root)
    assert inputs == audit["inputs"]
    jobs = [(str(args.run), str(args.audit), args.phase, cell) for cell in manifest["cells"]]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        errors = list(executor.map(verify_cell, jobs))
    maxima = {key: max(row[key] for row in errors) for key in errors[0]}
    report = dict(status="passed", source_commit=manifest["commit"], source_checks=source_checks,
                  inputs=inputs, maximum_metric_discrepancies=maxima, phase=args.phase,
                  scope="local recovered raw particles, weights, receipts, checkpoint reload, saved independent reference and joint metrics; true-reference integration and population checks performed by server audit",
                  verification_source_sha256=sha256(__file__))
    write_json(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
