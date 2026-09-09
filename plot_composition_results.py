import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd


METHODS = ["full", "unweighted", "naive", "unbiased", "cumulant", "cv", "cv_cumulant", "tail", "tail_cumulant"]
LABELS = ["Full", "Unweighted", "Plug-in", "Unbiased", "Unbiased + pair", "Moment CV", "Moment CV + pair", "Tail CV", "Tail CV + pair"]
COLORS = ["#252525", "#999999", "#d55e00", "#e69f00", "#cc79a7", "#0072b2", "#56b4e9", "#009e73", "#447744"]
FAMILIES = ["gaussian", "mixture", "weak_mixture"]
FAMILY_LABELS = ["Gaussian", "Mixture", "Weak mixture"]


def save(fig, output, name):
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def one_step_figure(frame, output):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    palette = ["#d55e00", "#e69f00", "#cc79a7", "#0072b2", "#009e73"]
    for batch, color in zip([2, 4, 8, 16, 32], palette):
        selected = frame[(frame.method == "unbiased") & (frame.batch == batch)]
        for ax, metric in zip(axes[:2], ["log_normalizer", "ess_fraction"]):
            grouped = selected.groupby("particles")[metric]
            median, lower, upper = grouped.median(), grouped.quantile(0.25), grouped.quantile(0.75)
            x = median.index.to_numpy()
            ax.plot(x, median.to_numpy(), color=color, marker="o", markersize=3, label=f"M={batch}")
            ax.fill_between(x, lower.to_numpy(), upper.to_numpy(), color=color, alpha=0.13, linewidth=0)
    full = frame[frame.method == "full"]
    exact = full.full_exact_log_normalizer.iloc[0]
    axes[0].axhline(exact, color="#252525", linestyle="--", linewidth=1, label="Full reference")
    full_ess = full.groupby("particles").ess_fraction.median()
    axes[1].plot(full_ess.index, full_ess, color="#252525", linestyle="--", linewidth=1)
    largest = frame[(frame.method == "unbiased") & (frame.particles == 1048576)]
    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_xlabel("Particles P")
        ax.set_xticks([1024, 16384, 262144, 1048576], ["1k", "16k", "262k", "1,049k"])
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Estimated log normalizer")
    axes[0].set_title("(a) Weight normalization")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("ESS / P")
    axes[1].set_title("(b) Effective sample size")
    grouped = largest.groupby("batch").potential_mse
    means, std = grouped.mean(), grouped.std()
    axes[2].errorbar(means.index, means, yerr=std, marker="o", markersize=3, color="#0072b2", capsize=2)
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log")
    axes[2].set_xticks([2, 4, 8, 16, 32], ["2", "4", "8", "16", "32"])
    axes[2].set_xlabel("Batch size M")
    axes[2].set_ylabel("Potential MSE")
    axes[2].set_title("(c) Local error at P=1,048,576")
    axes[0].legend(fontsize=6, ncol=2, frameon=False, loc="upper left")
    for ax in axes:
        ax.grid(alpha=0.15)
    save(fig, output, "one_step")


def certificate_figure(frame, output):
    rows, labels = [], []
    for family, family_label in zip(FAMILIES, FAMILY_LABELS):
        for groups in [16, 64]:
            for steps in [128, 512, 2048]:
                row = []
                for method in METHODS:
                    selected = frame[(frame.family == family) & (frame.groups == groups) & (frame.steps == steps) & (frame.method == method)]
                    if selected.empty:
                        row.append(2)
                    else:
                        statuses = selected.integrability_certificate.unique()
                        assert len(statuses) == 1
                        row.append(0 if statuses[0] == "finite" else 1)
                rows.append(row)
                labels.append(f"{family_label}, G={groups}, K={steps}")
    fig, ax = plt.subplots(figsize=(7.2, 4.25), constrained_layout=True)
    array = np.array(rows)
    ax.imshow(array, cmap=ListedColormap(["#d4eee3", "#f2cdc5", "#ededed"]), vmin=0, vmax=2, aspect="auto", interpolation="nearest")
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            ax.text(j, i, ["F", "I", "-"][array[i, j]], ha="center", va="center", fontsize=8, color="#234239" if array[i, j] == 0 else "#693328")
    ax.set_xticks(range(len(METHODS)), LABELS, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0, labelsize=7)
    ax.set_title("Population certificate: F = finite, I = infinite, - = not run", fontsize=9)
    for y in [5.5, 11.5]:
        ax.axhline(y, color="white", linewidth=2)
    save(fig, output, "integrability_grid")


def benchmark_figure(frame, output):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
    for ax, family, title in zip(axes, FAMILIES, FAMILY_LABELS):
        selected = frame[(frame.family == family) & (frame.groups == 64) & (frame.dimension == 8)]
        for method, label, color in zip(METHODS, LABELS, COLORS):
            subset = selected[selected.method == method]
            if subset.empty:
                continue
            grouped = subset.groupby("steps")
            x, y = grouped.seconds.median(), grouped.w1_mean.median()
            ax.plot(x, y, color=color, linewidth=1, alpha=0.9, label=label)
            for steps in x.index:
                finite = subset[subset.steps == steps].integrability_certificate.iloc[0] == "finite"
                ax.scatter(x.loc[steps], y.loc[steps], s=19, marker="o" if finite else "x", color=color, linewidths=1.1, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([0.2, 0.5, 1, 2, 5, 10], ["0.2", "0.5", "1", "2", "5", "10"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        yticks = {"gaussian": [0.01, 0.02, 0.05, 0.1], "mixture": [0.3, 0.5, 1, 2], "weak_mixture": [0.02, 0.03, 0.05, 0.08]}[family]
        ax.set_yticks(yticks, [f"{value:g}" for value in yticks])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Sampler seconds")
        ax.set_title(title)
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Marginal W1 (median of 5 seeds)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=6.5, frameon=False)
    save(fig, output, "accuracy_time")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.main, args.supplement):
        assert json.loads((path / "verification.json").read_text())["status"] == "passed"
    first = pd.read_csv(args.main / "audited_summary.csv")
    second = pd.read_csv(args.supplement / "audited_summary.csv")
    benchmark = pd.concat([first, second[second.kind == "tail"]], ignore_index=True)
    assert len(benchmark) == 1500
    counts = benchmark.groupby(["family", "groups", "dimension", "steps", "method"]).size()
    assert np.all(counts.to_numpy() == 5)
    one_step = second[second.kind == "one_step"]
    assert len(one_step) == 480
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    one_step_figure(one_step, args.output)
    certificate_figure(benchmark, args.output)
    benchmark_figure(benchmark, args.output)
    aggregated = benchmark.groupby(["family", "groups", "dimension", "steps", "method", "integrability_certificate"]).agg(w1_mean=("w1_mean", "mean"), w1_std=("w1_mean", "std"), seconds_median=("seconds", "median"), score_evaluations=("score_evaluations", "first"), ess_median=("minimum_ess_fraction", "median"), joint_sign_tv_mean=("joint_sign_total_variation", "mean"), joint_sign_tv_std=("joint_sign_total_variation", "std"), correlation_rms_mean=("off_diagonal_correlation_rms", "mean"), cells=("cell_id", "count")).reset_index()
    aggregated.to_csv(args.output / "benchmark_aggregate.csv", index=False)
    table = aggregated[(aggregated.groups == 64) & (aggregated.dimension == 8) & (aggregated.steps == 512)]
    table.to_csv(args.output / "matched_comparison.csv", index=False)
    one_step.groupby(["particles", "method", "batch"]).agg(logz_median=("log_normalizer", "median"), logz_mean=("log_normalizer", "mean"), logz_std=("log_normalizer", "std"), ess_median=("ess_fraction", "median"), mse_mean=("potential_mse", "mean"), mse_std=("potential_mse", "std"), nonintegrable_batches=("nonintegrable_batch_count", "sum"), all_strongest_batches=("all_strongest_batch_count", "sum"), runs=("cell_id", "count")).to_csv(args.output / "one_step_aggregate.csv")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
