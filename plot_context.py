import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads((args.root / "aggregate.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5), gridspec_kw={"width_ratios": [1.1, 1]})
    colors = {64: "#4285b4", 128: "#b3613e"}
    ax = axes[0]
    for i, (prior, n) in enumerate([("gp", 8), ("gp", 24), ("jump", 8), ("jump", 24)]):
        for width in [64, 128]:
            values = [100 * row["improvement"] for row in data["stability_rows"] if row["prior"] == prior and row["n_context"] == n and row["width"] == width]
            y = i + (-0.13 if width == 64 else 0.13)
            label = "width64 / 20k" if width == 64 else "width128 / 40k"
            ax.scatter(values, [y] * 3, color=colors[width], s=35, label=label if i == 0 else None, zorder=3)
            ax.plot([min(values), max(values)], [y, y], color=colors[width], alpha=0.5)
    ax.axvline(0, color="#555555", lw=1)
    ax.set_yticks(range(4), ["GP, 8 context", "GP, 24 context", "Jump, 8 context", "Jump, 24 context"])
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Risk reduction after adding stability (%)")
    ax.set_title("Context stability: no consistent incremental gain", loc="left", fontsize=11, pad=16)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.grid(axis="x", alpha=0.15)
    ax.text(0, -0.2, "Each dot: an independent task seed.\nPredeclared continuation threshold: +15% (outside plot).", transform=ax.transAxes, fontsize=9, color="#555555")
    ax = axes[1]
    for i, n in enumerate([8, 24]):
        for j, seed in enumerate([20260910, 20260911, 20260912]):
            row = data["routing"][f"n{n}_s{seed}"]
            value = 100 * row["improvement_vs_best_fixed"]
            ci = 100 * np.array(row["ci95"])
            ax.errorbar(value, i + (j - 1) * 0.15, xerr=np.array([[value - ci[0]], [ci[1] - value]]), fmt="o", color="#43866d", capsize=3)
    ax.axvline(0, color="#555555", lw=1)
    ax.axvline(10, color="#ad5151", lw=1, ls="--")
    ax.text(9.75, 0.46, "10% criterion", rotation=90, va="center", ha="right", color="#ad5151", fontsize=9)
    ax.set_yticks([0, 1], ["8 context", "24 context"])
    ax.invert_yaxis()
    ax.set_xlim(-1, 11)
    ax.set_ylim(1.4, -0.4)
    ax.set_title("Task-family routing: positive but modest gain", loc="left", fontsize=11, pad=16)
    ax.set_xlabel("Risk reduction vs. best fixed specialist (%)")
    ax.grid(axis="x", alpha=0.15)
    ax.text(0, -0.2, "Equal GP / Jump mixture; three task seeds.\nBars: 95% task-bootstrap intervals (exploratory).", transform=ax.transAxes, fontsize=9, color="#555555")
    fig.suptitle("SCNet pilot | 12 completed cells | 3,072 independent test tasks", fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.24, top=0.84, wspace=0.35)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")


if __name__ == "__main__":
    main()
