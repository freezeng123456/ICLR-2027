import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--factorization", type=Path, required=True)
    parser.add_argument("--anchored", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stats = json.loads(args.confirmation.read_text())
    methods = ["full", "tail_fixed", "surrogate_only", "surrogate_full", "certified"]
    labels = ["Full", "Tail", "Interpolation\nonly", "Exact\ncorrection", "Poisson\ncorrection"]
    colors = ["#36454F", "#D68046", "#6C9E8E", "#7F87B2", "#2873AD"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), layout="constrained")
    for ax, key, title in zip(axes, ["w1_mean", "seconds_including_preparation", "sign_joint_tv"],
                              ["Coordinate W1", "Sampling time (seconds)", "Joint sign total variation"]):
        means = np.array([stats["intervals"][method][key]["mean"] for method in methods])
        ci = np.array([stats["intervals"][method][key]["ci95"] for method in methods])
        ax.bar(np.arange(5), means, color=colors, width=0.68)
        ax.errorbar(np.arange(5), means, yerr=np.array([means-ci[:, 0], ci[:, 1]-means]), fmt="none", color="#222222", capsize=3)
        ax.set_xticks(np.arange(5), labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle("Held-out confirmation: 5 trained models × 10 datasets\n32,768 particles; 2,048 steps; preparation included; 95% crossed-bootstrap intervals", fontsize=12)
    fig.savefig(args.output / "heldout-confirmation.png", dpi=220)
    fig.savefig(args.output / "heldout-confirmation.svg")
    plt.close(fig)
    factorization = json.loads(args.factorization.read_text())["settings"]
    families = ["gaussian", "mixture", "weak_mixture", "learned"]
    labels = ["Gaussian", "Mixture", "Weak mixture", "Learned"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for ax, key, title in zip(axes, ["w1_mean", "sign_joint_tv"], ["Coordinate W1", "Joint sign total variation"]):
        for i, (method, name, color) in enumerate([("full", "Joint resampling", "#36454F"),
                                                  ("factorized_full", "Coordinate resampling", "#2873AD")]):
            values = [factorization[family][method][key] for family in families]
            ax.bar(np.arange(4)+(i-0.5)*0.34, values, width=0.32, color=color, label=name)
        ax.set_xticks(np.arange(4), labels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Separable-target diagnostic: identical full Euler population target\n5 problems per family; 8,192 particles; 512 steps", fontsize=12)
    fig.savefig(args.output / "factorization-diagnostic.png", dpi=220)
    fig.savefig(args.output / "factorization-diagnostic.svg")
    plt.close(fig)
    if args.anchored is not None:
        anchored = json.loads(args.anchored.read_text())
        methods = ["full", "tail_fixed", "tail_anchored"]
        labels = ["Full", "Original tail", "Anchored tail"]
        colors = ["#36454F", "#D68046", "#2873AD"]
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), layout="constrained")
        for ax, key, title in zip(axes, ["w1_mean", "seconds_including_preparation", "sign_joint_tv"],
                                  ["Coordinate W1", "Sampling time (seconds)", "Joint sign total variation"]):
            means = np.array([anchored["intervals"][method][key]["mean"] for method in methods])
            ci = np.array([anchored["intervals"][method][key]["ci95"] for method in methods])
            ax.bar(np.arange(3), means, color=colors, width=0.65)
            ax.errorbar(np.arange(3), means, yerr=np.array([means-ci[:, 0], ci[:, 1]-means]), fmt="none", color="#222222", capsize=3)
            ax.set_xticks(np.arange(3), labels, rotation=20, ha="right")
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.18)
            ax.set_axisbelow(True)
        fig.suptitle("Anchored-tail confirmation: 5 trained models × 10 new datasets\n32,768 particles; 2,048 steps; preparation included; 95% crossed-bootstrap intervals", fontsize=12)
        fig.savefig(args.output / "anchored-confirmation.png", dpi=220)
        fig.savefig(args.output / "anchored-confirmation.svg")
        plt.close(fig)


if __name__ == "__main__":
    main()
