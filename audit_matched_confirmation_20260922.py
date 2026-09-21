import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import beta, t
from confseq.betting import betting_mart


def audit_cell(directory, replay_official=False):
    config = json.loads((directory / "config.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if not (directory / "done").is_file():
        raise ValueError("Missing completion marker")
    with np.load(directory / "trace.npz", allow_pickle=False) as archive:
        trace, observations, checkpoints = archive["trace"], archive["observations"], archive["checkpoints"]
    for values in [trace, observations, checkpoints]:
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite record")
    if observations.ndim != 2 or observations.shape[1] != 2 or len(observations) != summary["calls"]:
        raise ValueError("Raw query accounting mismatch")
    if not np.isin(observations[:, 0], [0, 1]).all() or np.any(observations[:, 1] < 0) or np.any(observations[:, 1] > 1):
        raise ValueError("Observation outside global bounds")
    if not np.array_equal(observations[:4, 0], [0, 0, 1, 1]):
        raise ValueError("Pilot allocation differs from protocol")
    expected_checkpoints = np.minimum(np.arange(20, config["budget"] + 16, 16), config["budget"])
    if not np.array_equal(checkpoints, expected_checkpoints[:len(checkpoints)]) or checkpoints[-1] != len(observations):
        raise ValueError("Unmatched checkpoint schedule")
    if summary["certificate_checks"] != len(checkpoints) or summary["design_updates"] != len(checkpoints):
        raise ValueError("Update/check count mismatch")
    problem, method = config["problem"], config["method"]
    a, b = problem["positive_scale"], problem["negative_scale"]
    if summary["method"] != method or summary["seed"] != config["seed"] or summary["problem"] != problem["name"]:
        raise ValueError("Cell identity mismatch")
    if directory.name != f"seed_{config['seed']:05d}" or directory.parent.name != method or directory.parent.parent.name != problem["name"]:
        raise ValueError("Directory identity mismatch")
    if config["alpha"] != 0.05 or config["check_every_queries"] != 16 or config["pilot_queries"] != 4:
        raise ValueError("Protocol constants changed")
    if summary["positive_queries"] != np.sum(observations[:, 0] == 0) or summary["negative_queries"] != np.sum(observations[:, 0] == 1):
        raise ValueError("Stratum count mismatch")
    official_replays = 0
    if method == "betting":
        if trace.shape != (len(checkpoints), 9) or not np.array_equal(trace[:, 0], checkpoints):
            raise ValueError("Invalid official checkpoint trace")
        previous_calls = 4
        lower, upper = np.zeros(2), np.ones(2)
        decisions = []
        for row in trace:
            calls = int(row[0])
            selected = int(np.argmax([a * (upper[0] - lower[0]), b * (upper[1] - lower[1])]))
            if not np.all(observations[previous_calls:calls, 0] == selected):
                raise ValueError("Official allocation differs from previous interval widths")
            if not np.array_equal(row[1:3], [np.sum(observations[:calls, 0] == 0), np.sum(observations[:calls, 0] == 1)]):
                raise ValueError("Official checkpoint query counts disagree")
            if np.any(row[3:5] < lower) or np.any(row[5:7] > upper) or np.any(row[3:7] < 0) or np.any(row[3:7] > 1):
                raise ValueError("Invalid running intersection")
            other = 1 - selected
            if row[3 + other] != lower[other] or row[5 + other] != upper[other]:
                raise ValueError("An unqueried stratum interval changed")
            if not np.allclose(row[7:9], [a * row[3] - b * row[6], a * row[5] - b * row[4]], rtol=1e-12, atol=1e-10):
                raise ValueError("Risk interval arithmetic differs")
            if replay_official:
                post_pilot = observations[4:calls]
                values = post_pilot[post_pilot[:, 0] == selected, 1]
                for endpoint, old, orientation, edge in [(row[3 + selected], lower[selected], 1, 0), (row[5 + selected], upper[selected], 0, 1)]:
                    if endpoint != old and endpoint != edge:
                        wealth = betting_mart(values, endpoint, alpha=0.0125, theta=orientation, trunc_scale=0.99)[-1]
                        if not np.isfinite(wealth) or wealth < 80 * (1 - 1e-7) or abs(np.log(wealth / 80)) > 1e-3:
                            raise ValueError("Official endpoint does not match an outward-rounded wealth root")
                        grid = np.linspace(max(0, endpoint - 1e-6), min(1, endpoint + 1e-6), 5)
                        wealth_grid = np.array([betting_mart(values, mean, alpha=0.0125, theta=orientation, trunc_scale=0.99)[-1] for mean in grid])
                        expected_sign = -1 if orientation == 1 else 1
                        if np.any(expected_sign * np.diff(wealth_grid) < -1e-7):
                            raise ValueError("Official local root monotonicity failed")
                official_replays += 1
            lower, upper = row[3:5], row[5:7]
            decisions.append(1 if row[7] > 0 else (-1 if row[8] < 0 else 0))
            previous_calls = calls
    else:
        cost = 2 if method == "paired" else 1
        if trace.ndim != 2 or trace.shape[1] != 7 or not np.array_equal(trace[:, 0], 4 + cost * np.arange(1, len(trace) + 1)):
            raise ValueError("Evidence cost trace differs")
        if trace[-1, 0] != len(observations):
            raise ValueError("Unrecorded paid observations")
        mass, plus, minus, weighted = trace[:, 1], trace[:, 2], trace[:, 3], trace[:, 4]
        raw = observations[4:]
        if method == "paired":
            if not np.array_equal(raw[:, 0], np.tile([0, 1], len(trace))):
                raise ValueError("Paired observation order differs")
            expected_weighted = a * raw[::2, 1] - b * raw[1::2, 1]
            lower_bound, upper_bound = -b, a
        else:
            if np.any(mass < 0.02 - 1e-12) or np.any(mass > 0.98 + 1e-12):
                raise ValueError("Proposal mass outside protocol")
            expected_weighted = np.where(raw[:, 0] == 0, a * raw[:, 1] / mass, -b * raw[:, 1] / (1 - mass))
            lower_bound, upper_bound = -b / (1 - mass), a / mass
        if not np.allclose(weighted, expected_weighted, rtol=1e-12, atol=1e-10):
            raise ValueError("Weighted observation differs from raw query")
        if np.any(plus < 0) or np.any(minus < 0) or np.any(plus * lower_bound <= -1) or np.any(-minus * upper_bound <= -1):
            raise ValueError("Global safe betting bound violated")
        for start in range(0, len(trace), 16 // cost):
            block = trace[start:start + 16 // cost, 1:4]
            if not np.all(block == block[0]):
                raise ValueError("Proposal or bet changed inside a block")
        lp = np.cumsum(np.log1p(plus * weighted))
        ln = np.cumsum(np.log1p(-minus * weighted))
        if not np.allclose(trace[:, 5], lp, rtol=1e-12, atol=1e-10) or not np.allclose(trace[:, 6], ln, rtol=1e-12, atol=1e-10):
            raise ValueError("Independent wealth replay mismatch")
        indices = ((checkpoints - 4) / cost - 1).astype(int)
        decisions = np.where(lp[indices] >= np.log(40), 1, np.where(ln[indices] >= np.log(40), -1, 0)).tolist()
    if any(decisions[:-1]) or (decisions[-1] == 0 and summary["calls"] != config["budget"]):
        raise ValueError("Stopping does not match the first certified checkpoint or budget")
    pa, pb = problem["positive_beta"]
    na, nb = problem["negative_beta"]
    truth = int(np.sign(a * pa / (pa + pb) - b * na / (na + nb)))
    decision = decisions[-1]
    expected = {"truth": truth, "decision": decision, "certified": decision != 0, "wrong": decision != 0 and (truth == 0 or decision != truth)}
    if any(summary[key] != value for key, value in expected.items()):
        raise ValueError("Decision summary mismatch")
    return summary, observations[:4], official_replays


def analyze(root):
    provenance = json.loads((root / "provenance.json").read_text())
    for name, digest in provenance["source_hashes"].items():
        blob = subprocess.check_output(["git", "show", provenance["source_commit"] + ":" + name])
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError("Source commit and recorded file digest disagree")
    if provenance["confseq_commit"] != "5ffe733ca2447a2e28c2c91f3b00086173f2ab2c":
        raise ValueError("Unregistered official library revision")
    summaries = json.loads((root / "summaries.json").read_text())
    directories = sorted(path.parent for path in root.glob("*/*/seed_*/summary.json"))
    completion = json.loads((root / "completion.json").read_text())
    if not (root / "done").is_file() or len(directories) != provenance["expected_cells"] or completion["completed_cells"] != len(directories):
        raise ValueError("Confirmation is incomplete")
    cells, pilots, official_replays = [], {}, 0
    grouped = defaultdict(list)
    for directory in directories:
        seed = int(directory.name.split("_")[1])
        row, pilot, count = audit_cell(directory, replay_official=seed in {8000, 9000, 9249, 9499})
        key = (row["problem"], seed)
        if key in pilots and not np.array_equal(pilots[key], pilot):
            raise ValueError("Methods did not receive identical pilot observations")
        pilots[key] = pilot
        official_replays += count
        cells.append(row)
        grouped[(row["problem"], row["method"])].append(row)
    key = lambda row: (row["problem"], row["method"], row["seed"])
    if sorted(cells, key=key) != sorted(summaries, key=key) or len({key(row) for row in cells}) != len(cells):
        raise ValueError("Consolidated confirmation differs from cells")
    statistics = []
    for (problem, method), rows in sorted(grouped.items()):
        n, wrong = len(rows), sum(row["wrong"] for row in rows)
        statistics.append({"problem": problem, "method": method, "repetitions": n, "mean_calls": float(np.mean([row["calls"] for row in rows])), "median_calls": float(np.median([row["calls"] for row in rows])), "certified": sum(row["certified"] for row in rows), "wrong": wrong, "wrong_interval_95": [0 if wrong == 0 else float(beta.ppf(0.025, wrong, n - wrong + 1)), 1 if wrong == n else float(beta.ppf(0.975, wrong + 1, n - wrong))], "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in rows]))})
    primary = "rare_p0.002_b0.2"
    comparisons, rng = [], np.random.default_rng(424242)
    candidate = {row["seed"]: row for row in grouped[(primary, "growth")]}
    if len(candidate) >= 2:
        for method in ["uniform", "paired", "betting"]:
            control = {row["seed"]: row for row in grouped[(primary, method)]}
            if candidate.keys() != control.keys():
                raise ValueError("Primary comparison seeds differ")
            seeds = sorted(candidate)
            difference = np.array([candidate[seed]["calls"] - control[seed]["calls"] for seed in seeds])
            rate_loss = np.array([int(control[seed]["certified"]) - int(candidate[seed]["certified"]) for seed in seeds])
            bootstrap_indices = rng.integers(0, len(seeds), size=(20000, len(seeds)))
            upper = float(difference.mean() + t.ppf(1 - 0.05 / 3, len(seeds) - 1) * difference.std(ddof=1) / np.sqrt(len(seeds)))
            bootstrap_upper = float(np.quantile(difference[bootstrap_indices].mean(axis=1), 1 - 0.05 / 3))
            reduction = -float(difference.mean()) / float(np.mean([control[seed]["calls"] for seed in seeds]))
            comparisons.append({"control": method, "relative_query_reduction": reduction, "paired_mean_difference": float(difference.mean()), "bonferroni_one_sided_t_upper": upper, "bonferroni_bootstrap_upper": bootstrap_upper, "observed_certification_rate_loss": float(rate_loss.mean()), "certification_loss_bootstrap_upper_95": float(np.quantile(rate_loss[bootstrap_indices].mean(axis=1), 0.95)), "protocol_comparison_pass": bool(reduction >= 0.1 and upper < 0 and rate_loss.mean() <= 0.05), "bootstrap_supports_reduction": bootstrap_upper < 0})
    problem_names = {f"rare_p{p:g}_b{b:g}" for p in [0.002, 0.02] for b in [0.2, 0.4, 0.8, 1.2]}
    expected_keys = {(problem, method, seed) for problem in problem_names for method in ["growth", "uniform", "variance", "paired", "betting"] for seed in range(9000, 9500)}
    formal = {key(row) for row in cells} == expected_keys and provenance["arguments"]["budget"] == 4096
    report = {"audited_cells": len(cells), "audit_passed": True, "total_queries": sum(row["calls"] for row in cells), "official_replay_checkpoints": official_replays, "official_replay_scope": "seeds 9000, 9249, 9499 per problem; seed8000 for smoke", "statistics": statistics, "primary_comparisons": comparisons, "formal_matrix": formal, "application_exploration_gate": bool(formal and len(comparisons) == 3 and all(row["protocol_comparison_pass"] and row["bootstrap_supports_reduction"] for row in comparisons))}
    (root / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.rglob("*")) if path.is_file() and path.name != "SHA256.json"}
    (root / "SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["audited_cells", "audit_passed", "formal_matrix", "application_exploration_gate"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    analyze(parser.parse_args().root.resolve())
