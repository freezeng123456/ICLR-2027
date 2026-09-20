import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from run_solid_20260920 import write_json


def sensor(root, output):
    rows = json.loads((root / "development" / "rows.json").read_text())
    selected = json.loads((root / "selection.json").read_text())
    if len(rows) != 416 or not (root / "development" / "done").exists():
        raise RuntimeError("complete sensor development required")
    baseline = selected["baseline"]
    candidates = {c["setting_id"] for c in selected["settings"] if c["role"] == "candidate"}
    best = min(candidates, key=lambda name: selected["means"][name]["sliced_w1_32"])
    means = selected["means"]
    diagnostics = []
    references = {}
    for row in rows:
        if row["setting_id"] not in candidates:
            continue
        path = root / "development" / "cells" / f"cell_{row['cell_id']:04d}"
        diagnostic = json.loads((path / "diagnostics.json").read_text())
        metadata = diagnostic["metadata"]
        reference = metadata["reference_parameters"]
        asset = f"d{row['dimension']}_{row['regime']}_{row['dataset_seed']}"
        if asset not in references:
            with np.load(root / "assets" / asset / "reference.npz") as saved:
                references[asset] = saved["samples"]
        points = references[asset]
        delta = points[:, None] - np.asarray(reference["means"])
        distance = np.einsum("nki,ij,nkj->nk", delta, np.linalg.inv(reference["covariance"]), delta)
        outside = float(np.mean(distance.min(1) > chi2.ppf(.99, row["dimension"])))
        diagnostics.append(dict(cell_id=row["cell_id"], setting_id=row["setting_id"], asset=asset,
            reference_outside_99pct_ellipsoid_union=outside, reference_sample_count=len(points),
            q_outside_union_upper_bound=.01, sliced_w1_32=row["sliced_w1_32"], seconds=row["seconds"],
            preparation_seconds=diagnostic["timing"]["preparation_seconds"],
            annealing_seconds=diagnostic["timing"]["annealing_seconds"],
            output_seconds=diagnostic["timing"]["output_seconds"],
            actual_components=metadata["components_actual"],
            maximum_selected_gradient_norm=max(metadata["mode_search"]["selected_score_norms"])))
    strata = []
    for dimension in [2, 8]:
        for regime in ["ambiguous", "regular"]:
            for name in [best, baseline, "smc_k64_s0.15", "full", "tail_anchored"]:
                group = [r for r in rows if (r["dimension"], r["regime"], r["setting_id"]) == (dimension, regime, name)]
                strata.append(dict(dimension=dimension, regime=regime, setting_id=name,
                    **{key: float(np.mean([r[key] for r in group])) for key in ["sliced_w1_32", "seconds", "factor_calls"]}))
    result = dict(status="development_negative", cells=len(rows), independent_data_seeds=4,
        target_instances=16, candidate_confirmation_opened=False, selected_baseline=baseline,
        smallest_error_candidate=best, error_difference=means[best]["sliced_w1_32"] - means[baseline]["sliced_w1_32"],
        time_ratio=means[best]["seconds"] / means[baseline]["seconds"],
        factor_call_ratio=means[best]["factor_calls"] / means[baseline]["factor_calls"],
        means=means, strata=strata, diagnostics=diagnostics,
        scope="development-only descriptive findings; source/metric audit reported separately")
    write_json(output / "sensor-development-analysis.json", result)
    with (output / "sensor-strata.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(strata[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(strata)
    print(json.dumps({k: result[k] for k in ["status", "cells", "smallest_error_candidate", "selected_baseline",
                                              "error_difference", "time_ratio", "factor_call_ratio"]}, indent=2))


def learned(root, output):
    selected = json.loads((root / "selection.json").read_text())
    result = dict(selection=selected, scope="development-only" if not selected["expand_confirmation"] else "confirmation required")
    if (root / "confirmation" / "done").exists():
        rows = json.loads((root / "confirmation" / "rows.json").read_text())
        matrix = []
        for seed in range(1700, 1712):
            pairs = []
            for name in ["mixture", selected["baseline"]]:
                group = [r for r in rows if r["dataset_seed"] == seed and r["setting_id"] == name]
                if len(group) != 3:
                    raise RuntimeError("three repeats per paired confirmation problem required")
                pairs.append([np.mean([r[key] for r in group]) for key in ["sliced_w1_32", "seconds"]])
            matrix.append(pairs)
        matrix = np.asarray(matrix)
        index = np.random.default_rng(20261024).integers(0, 12, size=(10000, 12))
        means = matrix[index].mean(1)
        errors = means[:, 0, 0] - means[:, 1, 0]
        times = means[:, 0, 1] / means[:, 1, 1]
        error_upper, time_upper = np.quantile(errors, .95), np.quantile(times, .95)
        result["confirmation"] = dict(data_seed_clusters=12, paired_repeats=3,
            error_difference=float((matrix[:, 0, 0] - matrix[:, 1, 0]).mean()),
            time_ratio=float(matrix[:, 0, 1].mean() / matrix[:, 1, 1].mean()),
            error_upper95=float(error_upper), time_ratio_upper95=float(time_upper),
            joint_gate_pass=bool(error_upper <= .002 and time_upper < .8))
        result["scope"] = "fixed-bank two-dimensional learned composition only"
    write_json(output / "learned-factor-analysis.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", choices=["sensor", "learned"], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (sensor if args.study == "sensor" else learned)(args.root, args.output)


if __name__ == "__main__":
    main()
