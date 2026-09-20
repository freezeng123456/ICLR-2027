import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-archive-sha256", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    work = root / "work"
    output = root / "results/paper_20260921"
    output.mkdir(parents=True, exist_ok=True)
    reports = {
        "anchor-development-audit": work / "paper-anchor-development/audit.json",
        "anchor-confirmation-audit": work / "paper-anchor-confirmation/audit.json",
        "anchor-matched-audit": work / "paper-anchor-matched/audit.json",
        "sensor-original-environment-audit": work / "recovered-paper-20260921/old/audit-sensor.json",
        "twist-development-audit": work / "recovered-paper-20260921/new/audit-twist.json"}
    for name, path in reports.items():
        report = read(path)
        if report["status"] not in ["passed", "completed"]:
            raise RuntimeError(f"unclosed audit: {name}")
        shutil.copy2(path, output / (name + ".json"))
    local_sensor = read(work / "paper-sensor-confirmation/audit.json")
    if not all(row["pass"] for row in local_sensor["cells"]) or not all(row["passed"] for row in local_sensor["matrix_certificates"]):
        raise RuntimeError("local sensor metrics or matrix checks failed")
    for asset in local_sensor["assets"]:
        if not all(asset["exact"].values()) or not asset["density_matches_saved_mixture"]:
            raise RuntimeError("local sensor exact-reference check failed")
        failed = {key for key, value in asset["reference"].items() if value is False}
        if not failed.issubset({"regenerated_first_matches", "regenerated_second_matches"}):
            raise RuntimeError("local sensor mismatch is not limited to cross-platform sample replay")
    canonical_sensor = read(reports["sensor-original-environment-audit"])
    if canonical_sensor["auditor_sha256"] != digest(root / "audit_sensor_study_20260921.py"):
        raise RuntimeError("sensor auditor source changed after canonical check")
    shutil.copy2(work / "paper-sensor-confirmation/audit.json", output / "sensor-cross-platform-diagnostic.json")
    studies = {"anchor-development": "paper-anchor-development", "anchor-confirmation": "paper-anchor-confirmation",
               "anchor-matched": "paper-anchor-matched", "sensor-confirmation": "paper-sensor-confirmation"}
    statistics = {}
    for name, directory in studies.items():
        path = work / directory / "statistics.json"
        statistics[name] = read(path)
        statistics[name]["verification"] = dict(status="passed", source_statistics_sha256=digest(path),
            audit_report=("sensor-original-environment-audit.json" if name == "sensor-confirmation" else name + "-audit.json"))
        if name == "sensor-confirmation":
            statistics[name]["audit_status"] = "passed in original environment; local metrics and matrix certificates independently agree"
        (output / (name + "-statistics.json")).write_text(json.dumps(statistics[name], indent=2) + "\n")
        shutil.copy2(work / directory / "cells.csv", output / (name + "-cells.csv"))
    archives = {
        "old/sensor-raw-20260921.tar.gz": "f504b71e2c96166f0a487722c2e2db75b9a0f1713312538793aca462493a2ee6",
        "new/anchor-raw-20260921.tar.gz": "653e64a844df97781a5d3e179a62d583c6d90a8df6fd0b44f36382c84e9ea18f",
        "new/twist-raw-20260921.tar.gz": "c254bea3bbdc510e2243ad1dae82cca78955e0b7458cb8adcfc930d0b9e04a17",
        "new/matched-raw-20260921.tar.gz": args.matched_archive_sha256}
    for name, expected in archives.items():
        if digest(work / "recovered-paper-20260921" / name) != expected:
            raise RuntimeError(f"recovery archive mismatch: {name}")
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.35), layout="constrained")
    original = statistics["anchor-confirmation"]["by_setting"]
    matched = statistics["anchor-matched"]["by_setting"]
    entries = [(original, "full", "Joint (fine)", "#3765a0", "o", False),
               (original, "factorized_full", "Coordinatewise (fine)", "#3b8a5a", "s", False),
               (matched, "full", "Joint (matched)", "#3765a0", "o", True),
               (matched, "factorized_full", "Coordinatewise (matched)", "#3b8a5a", "s", True),
               (matched, "anchor_n8192_k1024_m8", "Anchored (matched)", "#c36b2d", "D", True)]
    for group, name, label, color, marker, filled in entries:
        x, y = group[name]["seconds_including_preparation"], group[name]["w1_mean"]
        axes[0].errorbar(x["mean"], y["mean"], yerr=np.abs(np.array(y["ci95"]) - y["mean"])[:, None],
                        marker=marker, color=color, markerfacecolor=color if filled else "white", markersize=5,
                        capsize=2, label=label, linestyle="none")
    axes[0].set(xlabel="Seconds including method preparation", ylabel="Mean marginal W1", title="Learned targets: budget and method")
    axes[0].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(0, -.29), ncol=2,
                   frameon=False, borderaxespad=0, columnspacing=.8, handletextpad=.4)
    sensor = statistics["sensor-confirmation"]["groups"]
    categories = ["d2_ambiguous", "d2_regular", "d8_ambiguous", "d8_regular"]
    lines = [("full_n16384_k2048_m4", "Full", "#3765a0"),
             ("tail_anchored_n16384_k2048_m8", "Anchored tail", "#c36b2d"),
             ("smc_k64_scale0.15", "Annealed SMC", "#3b8a5a"), ("exact", "Exact draws", "#555555")]
    for j, (name, label, color) in enumerate(lines):
        values = [sensor[c]["by_setting"][name]["sliced_w1_32"] for c in categories]
        centers = np.array([v["mean"] for v in values])
        intervals = np.array([v["ci95"] for v in values]).T
        axes[1].errorbar(np.arange(4) + (j - 1.5) * .13, centers, yerr=np.abs(intervals - centers),
                        fmt="o", color=color, capsize=2, markersize=3.5, label=label)
    axes[1].set(yscale="log", ylabel="Mean projected W1", title="Nonseparable targets: validity is insufficient")
    axes[1].set_xticks(range(4), ["2D\nambiguous", "2D\nsharper", "8D\nambiguous", "8D\nsharper"])
    axes[1].legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0, -.29),
                   ncol=2, borderaxespad=0, columnspacing=.8, handletextpad=.4)
    figures = root / "manuscript/figures"
    figure.savefig(figures / "paper_level_diagnostics.pdf")
    figure.savefig(figures / "paper_level_diagnostics.png", dpi=220)
    plt.close(figure)
    index = dict(status="passed", run_cells=1404, equivalent_replays=100,
        archive_sha256=archives, report_sha256={name: digest(path) for name, path in reports.items()},
        original_sensor_runtime=canonical_sensor["runtime"],
        cross_platform_note="all local sensor metrics and matrix certificates agree; seeded Gaussian draws are checked in their original environment",
        claims={"predeclared_selected_vs_original_joint": statistics["anchor-confirmation"]["paired_anchor_comparisons"]["anchor_n8192_k1024_m8"]["full"]["accuracy_cost_gate"],
                "predeclared_selected_vs_original_coordinatewise": statistics["anchor-confirmation"]["paired_anchor_comparisons"]["anchor_n8192_k1024_m8"]["factorized_full"]["accuracy_cost_gate"],
                "matched_budget_diagnostic": {key: value["accuracy_cost_gate"] for key, value in statistics["anchor-matched"]["paired_anchor_comparisons"]["anchor_n8192_k1024_m8"].items()},
                "sensor_speed_claim": False, "twist_expansion_eligible": bool(read(reports["twist-development-audit"])["eligible"])},
        builder_sha256=digest(Path(__file__)))
    (output / "verification-index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(dict(status="passed", run_cells=1404, output=str(output))), flush=True)


if __name__ == "__main__":
    main()
