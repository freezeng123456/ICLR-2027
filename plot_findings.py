import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 图上一律用英文标签：这两张图直接进九页稿，而且中文字体在容器里不一定装了。

SETTINGS = {"w64 @20k": ("results/conditioning_pfn_cond_w64.json", 0.30 * 20000),
            "w64 @40k": ("results/conditioning_pfn_cond_w64_40k.json", 0.30 * 40000),
            "w128 @20k": ("results/conditioning_pfn_cond.json", 1.19 * 20000),
            "w128 @40k": ("results/conditioning_pfn_cond_40k.json", 1.19 * 40000)}
OFF_CURVE = {"reweighted ctx sizes": ("results/conditioning_pfn_cond_rw.json", 1.19 * 20000),
             "exact-target distill": ("results/conditioning_pfn_cond_distill.json", 1.19 * 20000)}


def rows(path):
    return json.loads(Path(path).read_text())["rows"]


def panel_trend(ax, path, key, fmt, title, ylabel=True):
    """上下文点数越多、偏离越大。按结构那一维的隐变量上色，均匀设计、sigma = 0.05。"""
    r = [x for x in rows(path) if x["design"] == "uniform" and x["sigma"] == 0.05]
    vals = sorted({x[key] for x in r})
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(vals)))
    n_up = 0
    for v, c in zip(vals, cmap):
        sub = sorted([x for x in r if x[key] == v], key=lambda x: x["n_ctx"])
        ns = [x["n_ctx"] for x in sub]
        g = [x["gap"] for x in sub]
        n_up += int(g[-1] > g[0])
        ax.errorbar(ns, g, yerr=[x["gap_se"] for x in sub], marker="o", ms=4, lw=1.6,
                    color=c, label=fmt.format(v))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([8, 16, 24]); ax.set_xticklabels([8, 16, 24])
    ax.set_xticks([], minor=True)
    ax.set_xlim(7.2, 27)
    ax.set_ylim(0.008, 0.55)
    ax.set_xlabel("context points")
    if ylabel:
        ax.set_ylabel("excess KL over exact Bayes (nats)")
    ax.set_title(f"{title}\n{n_up}/{len(vals)} curves rise", fontsize=10)
    ax.legend(fontsize=6.5, ncol=2, loc="upper left", frameon=False)
    ax.grid(alpha=0.25, which="both")


JUMP_SETTINGS = {"w64 @20k": ("results/jump_pfn_jump_w64.json", 0.30 * 20000),
                 "w64 @40k": ("results/jump_pfn_jump_w64_40k.json", 0.30 * 40000),
                 "w128 @20k": ("results/jump_pfn_jump.json", 1.19 * 20000),
                 "w128 @40k": ("results/jump_pfn_jump_40k.json", 1.19 * 40000)}


def panel_scaling(ax):
    """两个先验上超额 KL 随算力的标度。指数差 2.5 倍，最差格子那一条更悬殊。"""
    base = min(c for _, c in SETTINGS.values())
    series = [(SETTINGS, "GP prior", "tab:blue", "o", "-"),
              (JUMP_SETTINGS, "jump prior", "tab:orange", "^", "--")]
    for cfg, name, color, mk, ls in series:
        pts = sorted((c, np.mean([x["gap"] for x in rows(p)]),
                      max(x["gap"] for x in rows(p))) for p, c in cfg.values())
        comp = np.array([p[0] for p in pts]) / base
        for idx, kind, alpha in ((1, "mean", 1.0), (2, "worst cell", 0.45)):
            vals = np.array([p[idx] for p in pts])
            coef = np.polyfit(np.log(comp), np.log(vals), 1)
            xx = np.linspace(comp.min() * 0.85, comp.max() * 1.6, 50)
            ax.plot(xx, np.exp(np.polyval(coef, np.log(xx))), color=color, lw=1.2,
                    ls=ls, alpha=0.55 * alpha)
            ax.plot(comp, vals, mk, color=color, ms=7, alpha=alpha,
                    label=f"{name}, {kind}: $\\propto$ compute$^{{{coef[0]:.3f}}}$")

    for (name, (p, c)), dy in zip(OFF_CURVE.items(), (7, -13)):
        v = np.mean([x["gap"] for x in rows(p)])
        ax.plot(c / base, v, "x", color="k", ms=9, mew=2)
        ax.annotate(f"{name} (off curve)", (c / base, v),
                    textcoords="offset points", xytext=(10, dy), fontsize=7)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("relative compute (params $\\times$ steps)")
    ax.set_ylabel("excess KL over exact Bayes (nats)")
    ax.set_title("The price is prior-dependent\n0.371 on GP, 0.146 on jump", fontsize=10)
    ax.legend(fontsize=6.5, loc="lower left", frameon=False)
    ax.grid(alpha=0.25, which="both")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))
    panel_trend(axes[0], SETTINGS["w128 @20k"][0], "ell", "$\\ell$ = {:.3f}",
                "GP prior (unimodal predictive)")
    panel_trend(axes[1], "results/jump_pfn_jump.json", "rate", "rate = {:.2f}",
                "Jump prior (multimodal predictive)", ylabel=False)
    panel_scaling(axes[2])
    plt.tight_layout()
    fig.savefig("findings.png", dpi=170)
    print("已保存 findings.png")


if __name__ == "__main__":
    main()
