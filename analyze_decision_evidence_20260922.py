import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, t


def audit_cell(directory):
    config = json.loads((directory / "config.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    with np.load(directory / "trace.npz", allow_pickle=False) as data:
        trace = data["trace"]
    if not (directory / "done").is_file() or trace.ndim != 2 or trace.shape[1] != 7 or not np.isfinite(trace).all():
        raise ValueError(f"Invalid trace or completion marker: {directory}")
    method = config["method"]
    problem = config["problem"]
    a, b = problem["positive_scale"], problem["negative_scale"]
    costs, mass, positive_bet, negative_bet, value, lp, ln = trace.T
    expected_cost = 2 if method == "paired" else 1
    if not np.array_equal(costs, 4 + expected_cost * np.arange(1, len(trace) + 1)):
        raise ValueError(f"Query accounting mismatch: {directory}")
    if summary["calls"] != costs[-1] or costs[-1] > config["max_samples"]:
        raise ValueError(f"Budget mismatch: {directory}")
    if np.any(positive_bet < 0) or np.any(negative_bet < 0):
        raise ValueError(f"Negative bet: {directory}")
    if method in {"uniform", "variance", "growth"}:
        if np.any(mass <= 0) or np.any(mass >= 1):
            raise ValueError(f"Missing proposal support: {directory}")
        lower, upper = -b / (1 - mass), a / mass
    else:
        lower, upper = -b, a
    if np.any(positive_bet * lower <= -1) or np.any(-negative_bet * upper <= -1):
        raise ValueError(f"Global wealth bound violated: {directory}")
    if np.any(value < lower - 1e-10) or np.any(value > upper + 1e-10):
        raise ValueError(f"Observation bound violated: {directory}")
    replay_lp = np.cumsum(np.log1p(positive_bet * value))
    replay_ln = np.cumsum(np.log1p(-negative_bet * value))
    if not np.allclose(lp, replay_lp, rtol=1e-12, atol=1e-10) or not np.allclose(ln, replay_ln, rtol=1e-12, atol=1e-10):
        raise ValueError(f"Evidence replay mismatch: {directory}")
    boundary = np.log(2 / config["alpha"])
    crossings = (lp >= boundary) | (ln >= boundary)
    if crossings[:-1].any():
        raise ValueError(f"Simulation continued after first crossing: {directory}")
    decision = 1 if lp[-1] >= boundary else (-1 if ln[-1] >= boundary else 0)
    pa, pb = problem["positive_beta"]
    na, nb = problem["negative_beta"]
    truth = int(np.sign(a * pa / (pa + pb) - b * na / (na + nb)))
    wrong = decision != 0 and (truth == 0 or truth != decision)
    expected = {"decision": decision, "truth": truth, "wrong": wrong, "certified": decision != 0}
    if any(summary[key] != val for key, val in expected.items()):
        raise ValueError(f"Decision label mismatch: {directory}")
    if not np.allclose([summary["log_positive"], summary["log_negative"]], [lp[-1], ln[-1]], atol=1e-10):
        raise ValueError(f"Summary evidence mismatch: {directory}")
    return summary


def binomial_interval(successes, total):
    return [
        0.0 if successes == 0 else float(beta.ppf(0.025, successes, total - successes + 1)),
        1.0 if successes == total else float(beta.ppf(0.975, successes + 1, total - successes)),
    ]


def analyze(root):
    provenance = json.loads((root / "provenance.json").read_text())
    completion = json.loads((root / "completion.json").read_text())
    if not (root / "done").is_file():
        raise ValueError("Run is incomplete")
    directories = sorted(path.parent for path in root.glob("*/*/seed_*/summary.json"))
    if len(directories) != provenance["expected_cells"] or len(directories) != completion["completed_cells"]:
        raise ValueError("Cell count differs from manifest")
    cells = [audit_cell(directory) for directory in directories]
    consolidated = json.loads((root / "summaries.json").read_text())
    by_key = lambda rows: {(row["problem"], row["method"], row["seed"]): row for row in rows}
    if len(by_key(cells)) != len(cells) or by_key(cells) != by_key(consolidated):
        raise ValueError("Consolidated results are inconsistent")
    grouped = defaultdict(list)
    for row in cells:
        grouped[(row["problem"], row["method"])].append(row)
    statistics = []
    for (problem, method), rows in sorted(grouped.items()):
        count = len(rows)
        wrong = sum(row["wrong"] for row in rows)
        statistics.append({
            "problem": problem, "method": method, "repetitions": count,
            "mean_calls": float(np.mean([row["calls"] for row in rows])),
            "median_calls": float(np.median([row["calls"] for row in rows])),
            "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in rows])),
            "certified_fraction": sum(row["certified"] for row in rows) / count,
            "wrong": wrong, "wrong_fraction": wrong / count,
            "wrong_probability_interval_95": binomial_interval(wrong, count),
        })
    comparisons = []
    for problem in sorted({row["problem"] for row in cells}):
        candidate = {row["seed"]: row for row in grouped[(problem, "growth")]}
        for method in ["uniform", "variance", "paired", "upper_bound"]:
            if (problem, method) not in grouped:
                continue
            control = {row["seed"]: row for row in grouped[(problem, method)]}
            if candidate.keys() != control.keys():
                raise ValueError("Unpaired seeds")
            difference = np.array([candidate[key]["calls"] - control[key]["calls"] for key in candidate])
            center = float(difference.mean())
            radius = float(t.ppf(0.975, len(difference) - 1) * difference.std(ddof=1) / np.sqrt(len(difference))) if len(difference) > 1 else None
            control_mean = float(np.mean([row["calls"] for row in control.values()]))
            comparisons.append({
                "problem": problem, "control": method,
                "mean_candidate_minus_control_calls": center,
                "paired_interval_95": [center - radius, center + radius] if radius is not None else None,
                "relative_call_reduction": -center / control_mean,
                "interval_scope": "descriptive development comparison; no multiplicity correction",
            })
    report = {
        "audit_passed": True, "audited_cells": len(cells),
        "total_recorded_queries": sum(row["calls"] for row in cells),
        "statistics": statistics, "comparisons": comparisons,
        "scope": "bounded synthetic strata; development only",
    }
    (root / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Signed-integral evidence development", "", f"Audited cells: {len(cells)}.", "", "Development data; bounded synthetic strata; no application or GPU speed claim.", "", "| Problem | Method | Mean queries | Certified | Wrong | Mean seconds |", "|---|---|---:|---:|---:|---:|"]
    for row in statistics:
        lines.append(f"| {row['problem']} | {row['method']} | {row['mean_calls']:.2f} | {row['certified_fraction']:.2%} | {row['wrong']} | {row['mean_runtime_seconds']:.6f} |")
    (root / "analysis.md").write_text("\n".join(lines) + "\n")
    manifest = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file() and path.name != "SHA256.json"
    }
    (root / "SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"audited_cells": len(cells), "total_recorded_queries": report["total_recorded_queries"], "files": len(manifest)}))
    for problem in ["concentrated_r1.2", "concentrated_r2", "concentrated_r9", "concentrated_r100", "variable_r9"]:
        rows = [row for row in statistics if row["problem"] == problem]
        print(json.dumps({"problem": problem, "queries": {row["method"]: row["mean_calls"] for row in rows}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    analyze(parser.parse_args().root.resolve())
