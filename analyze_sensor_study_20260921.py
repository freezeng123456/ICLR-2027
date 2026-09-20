import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_solid_20260920 import sha256, write_json


METRICS = ["sliced_w1_32", "w1_mean", "mean_error", "covariance_error", "state_mse", "coverage90", "coverage95", "width90", "width95", "seconds"]


def summarize(rows, manifest, phase):
    configurations = manifest["cells"]
    expected = {(c["dimension"], c["regime"], c["dataset_seed"], c["setting_id"]) for c in configurations}
    observed = {(r["dimension"], r["regime"], r["dataset_seed"], r["setting_id"]) for r in rows}
    if len(observed) != len(rows) or observed != expected:
        raise RuntimeError("observed keys must exactly equal the manifest")
    settings = sorted({c["setting_id"] for c in configurations})
    seeds = sorted({c["dataset_seed"] for c in configurations})
    strata = sorted({(c["dimension"], c["regime"]) for c in configurations})
    lookup = {(r["dimension"], r["regime"], r["dataset_seed"], r["setting_id"]): r for r in rows}
    rng = np.random.default_rng(20261001)
    indices = rng.integers(0, len(seeds), size=(10000, len(seeds)))
    result = dict(phase=phase, settings=settings, seeds=seeds, strata=strata, bootstrap_replicates=10000,
                  bootstrap_seed=20261001, bootstrap_unit="shared data seed, paired across all four settings", groups={})
    for group_name, group_strata in [("overall", strata)] + [(f"d{d}_{regime}", [(d, regime)]) for d, regime in strata]:
        group = dict(by_setting={}, comparisons={})
        values, boot = {}, {}
        for setting in settings:
            group["by_setting"][setting] = {}
            for metric in METRICS:
                matrix = np.array([[lookup[d, regime, seed, setting][metric] for seed in seeds] for d, regime in group_strata])
                if not np.isfinite(matrix).all():
                    raise FloatingPointError("nonfinite summary metrics")
                samples = matrix[:, indices].mean((0, 2))
                values[setting, metric], boot[setting, metric] = matrix, samples
                group["by_setting"][setting][metric] = dict(mean=float(matrix.mean()),
                    ci95=np.quantile(samples, [.025, .975]).tolist(), per_seed=matrix.mean(0).tolist())
        baselines = [s for s in settings if s == "full_n16384_k2048_m4" or s.startswith("smc_") or s == "exact"]
        for setting in settings:
            group["comparisons"][setting] = {}
            for baseline in baselines:
                if setting == baseline:
                    continue
                delta = boot[setting, "sliced_w1_32"] - boot[baseline, "sliced_w1_32"]
                ratio = boot[setting, "seconds"] / boot[baseline, "seconds"]
                group["comparisons"][setting][baseline] = dict(
                    error_difference=float((values[setting, "sliced_w1_32"] - values[baseline, "sliced_w1_32"]).mean()),
                    error_difference_ci95=np.quantile(delta, [.025, .975]).tolist(), error_difference_upper95=float(np.quantile(delta, .95)),
                    time_ratio=float(values[setting, "seconds"].mean() / values[baseline, "seconds"].mean()),
                    time_ratio_ci95=np.quantile(ratio, [.025, .975]).tolist(), time_ratio_upper95=float(np.quantile(ratio, .95)),
                    accuracy_gate=bool(np.quantile(delta, .95) <= .005),
                    accuracy_cost_gate=None,
                    timing_status="not controlled: another training task overlapped the final confirmation interval")
        result["groups"][group_name] = group
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["development", "confirmation"], required=True)
    args = parser.parse_args()
    phase_root = args.root / args.phase
    if (phase_root / "done").read_text().strip() != "completed":
        raise RuntimeError("phase has not completed")
    manifest = json.loads((phase_root / "manifest.json").read_text())
    rows = []
    certificates = {}
    references = {}
    for c in manifest["cells"]:
        cell = phase_root / "cells" / f"cell_{c['cell_id']:04d}"
        if (cell / "done").read_text().strip() != "completed":
            raise RuntimeError("cell has not completed")
        receipt = json.loads((cell / "receipt.json").read_text())
        if not receipt["files"] or not all(sha256(cell / name) == value for name, value in receipt["files"].items()):
            raise RuntimeError("cell receipt mismatch")
        row = json.loads((cell / "summary.json").read_text())
        if not all(row[key] == value for key, value in c.items()):
            raise RuntimeError("summary configuration mismatch")
        rows.append(row)
        certificate = json.loads((cell / "certificate.json").read_text())
        key = f"{c['method']}:{certificate['status']}"
        certificates[key] = certificates.get(key, 0) + 1
        name = f"d{c['dimension']}_{c['regime']}_{c['dataset_seed']}"
        references[name] = json.loads((args.root / "assets" / name / "reference_summary.json").read_text())
    if len(list((phase_root / "cells").glob("*/done"))) != len(rows):
        raise RuntimeError("unexpected extra cells")
    result = summarize(rows, manifest, args.phase)
    result.update(cells=len(rows), targets=len(references), certificate_statuses=certificates,
                  source_commit=manifest["commit"], manifest_sha256=sha256(phase_root / "manifest.json"),
                  reference_pair_sliced_w1=float(np.mean([r["reference_pair_metrics"]["sliced_w1_32"] for r in references.values()])),
                  audit_status="separate independent numerical audit required",
                  timing_status="observational only; other GPU training started 03:11 while confirmation ended 03:15 local time")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "statistics.json", result)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (args.output / "cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(cells=len(rows), targets=len(references), certificate_statuses=certificates)), flush=True)


if __name__ == "__main__":
    main()
