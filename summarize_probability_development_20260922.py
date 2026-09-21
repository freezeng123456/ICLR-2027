from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def main():
    base = Path("results")
    root = base / "probability_pivot_development_review_20260922"
    root.mkdir(exist_ok=False)
    runs = [
        "decision_evidence_development_20260922", "rare_counterevidence_development_20260922",
        "official_betting_control_20260922", "official_empbern_control_20260922",
    ]
    data, receipts = {}, []
    for run in runs:
        directory = base / run
        audit_name = "audit.json" if run.startswith("official") else "analysis.json"
        audit = json.loads((directory / audit_name).read_text())
        if not (audit.get("passed") or audit.get("audit_passed")):
            raise ValueError(f"Unaudited development run: {run}")
        rows = json.loads((directory / "summaries.json").read_text())
        provenance = json.loads((directory / "provenance.json").read_text())
        if len(rows) != audit["audited_cells"] or len(rows) != provenance["expected_cells"]:
            raise ValueError("Development ledger mismatch")
        data[run] = rows
        receipts.append({"run": run, "cells": len(rows), "queries": sum(row["calls"] for row in rows), "audit": audit_name})
    grouped = defaultdict(list)
    for run in runs[1:]:
        for row in data[run]:
            grouped[(row["problem"], row["method"])].append(row)
    comparisons = []
    lines = ["# Rare-counterevidence development comparisons", "", "All eight development problems; 100 seeds each; 4096-query cap. Query totals include unfinished runs at the cap. These data selected the confirmation hypothesis.", "", "Candidate methods stopped after each observation; official CS controls checked every16 queries. Official controls also included the four pilot observations in their evidence. The matched confirmation uses common16-query checkpoints and pilot-only learning for all methods.", "", "| Problem | Growth | Uniform | Variance | Paired | Envelope | Official betting | Official empirical Bernstein |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for problem in sorted({key[0] for key in grouped}):
        methods = ["growth", "uniform", "variance", "paired", "upper_bound", "betting", "empbern"]
        means = {method: float(np.mean([row["calls"] for row in grouped[(problem, method)]])) for method in methods}
        lines.append("| " + problem + " | " + " | ".join(f"{means[method]:.2f}" for method in methods) + " |")
        candidate = {row["seed"]: row for row in grouped[(problem, "growth")]}
        for method in methods[1:]:
            control = {row["seed"]: row for row in grouped[(problem, method)]}
            if candidate.keys() != control.keys():
                raise ValueError("Development seed sets differ")
            differences = np.array([candidate[seed]["calls"] - control[seed]["calls"] for seed in sorted(candidate)])
            radius = t.ppf(0.975, len(differences) - 1) * differences.std(ddof=1) / np.sqrt(len(differences))
            comparisons.append({
                "problem": problem, "control": method, "mean_query_difference": float(differences.mean()),
                "descriptive_paired_interval_95": [float(differences.mean() - radius), float(differences.mean() + radius)],
                "relative_query_reduction": 1 - means["growth"] / means[method],
                "growth_certified": sum(row["certified"] for row in candidate.values()),
                "control_certified": sum(row["certified"] for row in control.values()),
                "growth_wrong": sum(row["wrong"] for row in candidate.values()),
                "control_wrong": sum(row["wrong"] for row in control.values()),
            })
    result = {"scope": "development only, unadjusted descriptive intervals", "runs": receipts, "cells": sum(row["cells"] for row in receipts), "queries": sum(row["queries"] for row in receipts), "comparisons": comparisons}
    (root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "analysis.md").write_text("\n".join(lines) + "\n")
    manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.iterdir()) if path.is_file()}
    (root / "SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ["cells", "queries"]}))
    for row in comparisons:
        if row["problem"] == "rare_p0.002_b0.2" and row["control"] in {"paired", "uniform", "betting"}:
            print(json.dumps(row))


if __name__ == "__main__":
    main()
