import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["Full", "U-WR", "U-WOR", "Tail-WR", "Tail-WOR"]
COLORS = ["#222222", "#d55e00", "#cc79a7", "#0072b2", "#009e73"]


def label(row):
    if row["method"] == "full":
        return "Full"
    prefix = "U" if row["method"] == "unbiased" else "Tail"
    return prefix + ("-WR" if row["sampling"] == "with_replacement" else "-WOR")


def summarize(frame, keys, output):
    metrics = [name for name in ["w1_mean", "true_w1_mean", "ks_mean", "seconds", "parameter_prediction_seconds", "composed_true_to_learned_kl_mean", "composed_true_to_learned_w1_mean"] if name in frame]
    rows = []
    for values, subset in frame.groupby(keys, dropna=False):
        row = dict(zip(keys, values))
        row.update(cells=len(subset), finite=int((subset.integrability_certificate == "finite").sum()))
        for metric in metrics:
            row.update({metric + "_mean": subset[metric].mean(), metric + "_std": subset[metric].std(), metric + "_median": subset[metric].median()})
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    return result


def learned_figure(frame, steps, output):
    fig, axes = plt.subplots(2, 4, figsize=(6.0, 3.6), sharey=True)
    for column, (groups, dimension) in enumerate([(16, 1), (16, 8), (64, 1), (64, 8)]):
        subset = frame[(frame.groups == groups) & (frame.dimension == dimension) & (frame.steps == steps)]
        for row, metric in enumerate(["w1_mean", "true_w1_mean"]):
            ax = axes[row, column]
            for x, (method, color) in enumerate(zip(METHODS, COLORS)):
                data = subset[subset.display_method == method].sort_values(["training_seed", "dataset_seed"])
                assert len(data) == 25
                jitter = np.linspace(-0.23, 0.23, len(data))
                for finite, marker in [(True, "o"), (False, "x")]:
                    mask = (data.integrability_certificate == "finite").to_numpy() == finite
                    ax.scatter(x + jitter[mask], data[metric].to_numpy()[mask], s=9, color=color, marker=marker, alpha=0.6, linewidths=0.6)
                ax.plot([x - 0.28, x + 0.28], [data[metric].mean()] * 2, color=color, lw=1.6)
            ax.set_yscale("log")
            ax.set_xticks(range(5), METHODS if row else [""] * 5, rotation=45, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.18)
            if row == 0:
                ax.set_title(f"G={groups}, d={dimension}", fontsize=10)
            if column == 0:
                ax.set_ylabel("W1 vs learned" if row == 0 else "W1 vs true")
    fig.tight_layout(pad=0.6)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(fig)


def sensitivity_figure(frame, output):
    fig, axes = plt.subplots(3, 2, figsize=(6.5, 6.0), sharey="row")
    for row, family in enumerate(["gaussian", "mixture", "weak_mixture"]):
        base = frame[(frame.family == family) & (frame.groups == 64) & (frame.dimension == 8) & (frame.steps == 512) & (frame.batch == 4)]
        for column, (variable, mask) in enumerate([("particles", base.u_max == 20), ("u_max", base.particles == 8192)]):
            ax = axes[row, column]
            for method, color in zip(METHODS, COLORS):
                data = base[mask & (base.display_method == method)]
                grouped = data.groupby(variable).w1_mean
                center, low, high = grouped.median(), grouped.quantile(0.25), grouped.quantile(0.75)
                assert len(center) == 3, (family, method, variable)
                ax.plot(center.index, center, color=color, lw=1, marker="o", ms=3, label=method)
                ax.fill_between(center.index, low, high, color=color, alpha=0.12)
            ax.set_yscale("log")
            if variable == "particles":
                ax.set_xscale("log", base=2)
                ax.set_xticks([8192, 32768, 131072], ["8,192", "32,768", "131,072"])
            else:
                ax.set_xticks([10, 20, 30])
            ax.set_xlabel("Particles (U=20)" if column == 0 else "U (P=8,192; K fixed)")
            ax.grid(alpha=0.18)
            if column == 0:
                ax.set_ylabel(f"{family.replace('_', ' ').capitalize()}\nW1")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(fig)


def sampling_figure(frame, output):
    fig, axes = plt.subplots(4, 3, figsize=(6.5, 7.3), sharex=True, sharey="row")
    for row, (dimension, steps) in enumerate([(1, 512), (1, 2048), (8, 512), (8, 2048)]):
        for column, family in enumerate(["gaussian", "mixture", "weak_mixture"]):
            subset = frame[(frame.family == family) & (frame.dimension == dimension) & (frame.steps == steps) & (frame.particles == 8192) & (frame.u_max == 20)]
            ax = axes[row, column]
            full = subset[subset.display_method == "Full"].w1_mean
            assert len(full) == 5
            ax.axhline(full.median(), color=COLORS[0], lw=1, label=METHODS[0])
            ax.fill_between([2, 8], full.quantile(0.25), full.quantile(0.75), color=COLORS[0], alpha=0.08)
            for method, color in zip(METHODS[1:], COLORS[1:]):
                data = subset[subset.display_method == method]
                grouped = data.groupby("batch").w1_mean
                center, low, high = grouped.median(), grouped.quantile(0.25), grouped.quantile(0.75)
                assert len(data) == 15
                ax.plot(center.index, center, color=color, lw=1, label=method)
                ax.fill_between(center.index, low, high, color=color, alpha=0.12)
                for batch, value in center.items():
                    states = data[data.batch == batch].integrability_certificate.unique()
                    assert len(states) == 1
                    ax.scatter(batch, value, color=color, marker="o" if states[0] == "finite" else "x", s=15, linewidths=0.8)
            ax.set_yscale("log")
            ax.set_xticks([2, 4, 8])
            ax.grid(alpha=0.15)
            if row == 0:
                ax.set_title(family.replace("_", " ").capitalize(), fontsize=9)
            if column == 0:
                ax.set_ylabel(f"d={dimension}, K={steps}\nW1", fontsize=8)
            if row == 3:
                ax.set_xlabel("Batch size M")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(fig)


def learned_tables(frame, output):
    tables = []
    for steps in [512, 2048]:
        lines = [r"\begin{table}[htbp]", r" \centering",
                 r" \caption{Complete learned-posterior results at $K=" + str(steps) + r"$. W1 values are means $\pm$ sample standard deviations across 25 crossed checkpoint/dataset pairs. Time is mean recorded sampler seconds. F and I denote finite and infinite population normalizers.}",
                 r" \label{tab:learned-" + str(steps) + "}", r" \small",
                 r" \begin{tabular}{rrlcllr}", r" \toprule",
                 r" $G$ & $d$ & Method & Domain & W1 vs learned & W1 vs true & Seconds\\",
                 r" \midrule", ""]
        for groups, dimension in [(16, 1), (16, 8), (64, 1), (64, 8)]:
            for method in METHODS:
                data = frame[(frame.steps == steps) & (frame.groups == groups) &
                             (frame.dimension == dimension) & (frame.display_method == method)]
                assert len(data) == 25
                assert set(zip(data.training_seed, data.dataset_seed)) == {(a, b) for a in range(5) for b in range(5)}
                states = data.integrability_certificate.unique()
                assert len(states) == 1 and states[0] in ["finite", "infinite"]
                domain = "F" if states[0] == "finite" else "I"
                learned = f"${data.w1_mean.mean():.4f}\\pm{data.w1_mean.std():.4f}$"
                true = f"${data.true_w1_mean.mean():.4f}\\pm{data.true_w1_mean.std():.4f}$"
                lines.append(f"{groups} & {dimension} & {method} & {domain} & {learned} & {true} & {data.seconds.mean():.2f}" + r"\\")
            if (groups, dimension) != (64, 8):
                lines.append(r"\midrule")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        tables.append("\n".join(lines))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(tables) + "\n")


def main(analysis, figures, tables=None):
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    frames = {}
    for kind, expected in [("oracle", 1080), ("learned", 1000)]:
        assert json.loads((analysis / f"{kind}_audit.json").read_text())["status"] == "passed"
        frame = pd.read_csv(analysis / f"{kind}_audited.csv")
        assert len(frame) == expected
        frame["display_method"] = frame.apply(label, axis=1)
        frames[kind] = frame
        summarize(frame, ["family", "groups", "dimension", "steps", "display_method", "batch", "particles", "u_max"], analysis / f"{kind}_grouped.csv")
    learned_figure(frames["learned"], 2048, figures / "learned_results.pdf")
    learned_figure(frames["learned"], 512, figures / "learned_results_coarse.pdf")
    sensitivity_figure(frames["oracle"], figures / "extension_sensitivity.pdf")
    sampling_figure(frames["oracle"], figures / "extension_sampling.pdf")
    unique = frames["learned"].drop_duplicates(["groups", "dimension", "training_seed", "dataset_seed"])
    unique[["groups", "dimension", "training_seed", "dataset_seed", "composed_true_to_learned_kl_mean", "composed_true_to_learned_w1_mean"]].to_csv(analysis / "learned_composition_errors.csv", index=False)
    training = json.loads((analysis / "training_audit.json").read_text())
    assert training["status"] == "passed"
    pd.DataFrame(training["rows"]).to_csv(analysis / "training_audited.csv", index=False)
    if tables is not None:
        learned_tables(frames["learned"], tables / "learned_tables.tex")
    print(json.dumps({"oracle_cells": len(frames["oracle"]), "learned_cells": len(frames["learned"]), "learned_factor_sets": len(unique), "status": "figures and complete grouped tables written"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--tables", type=Path, help="Output directory for the two complete learned-posterior LaTeX tables")
    args = parser.parse_args()
    main(args.analysis, args.figures, args.tables)
