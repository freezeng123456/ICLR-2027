import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import spearmanr

from exp_conditioning import (CKPT, N_QUERY, X_HALF, draw_design, draw_latent, load_pfn,
                              sample_gp, sweep_cells)
from eval_conditioning import quad_grids
from identifiability import gauss_kl, mixture_posterior

# 已测出的偏离是系统性的，而且**逐区域**不同：均值被压向先验、方差被推高，
# 两者都随上下文点数与函数结构变重。在先验平均里这些偏离相互抵消，
# 所以修正必须依区域而变，用上下文自身的统计量把区域认出来。
#
# 拟合只用网络自己的预训练先验采样，最小化留出 NLL：
# 不需要精确后验、不需要真实数据、不重训网络。

N_FIT_TASKS = 800
N_EVAL_TASKS = 40
SHORT_LAG = 0.15
OUT_PATH = Path("results/correction.json")


def pfn_predict(model, xc, yc, xq):
    x = torch.tensor(np.concatenate([xc, xq])[None], dtype=torch.float32)
    y = torch.tensor(np.concatenate([yc, np.zeros_like(xq)])[None], dtype=torch.float32)
    with torch.no_grad():
        mu, logv = model(x, y, len(xc))
    return mu[0].numpy().astype(np.float64), logv[0].numpy().astype(np.float64)


def context_features(xc, yc):
    """认出区域用的两个统计量。

    经验方差给出信号加噪声的总量；短滞后的变差函数给出「隔得很近的两点差多少」，
    也就是把长度尺度与噪声分开的那一维。两者都只需要上下文本身。
    """
    lag = np.abs(xc[:, None] - xc[None, :])
    iu = np.triu_indices(len(xc), 1)
    lag, sq = lag[iu], 0.5 * (yc[:, None] - yc[None, :])[iu] ** 2
    near = lag < SHORT_LAG
    short = sq[near].mean() if near.any() else sq.min()
    return np.log(np.var(yc) + 1e-8), np.log(short + 1e-8)


def design_matrix(logv, n_ctx, feats, n_terms):
    """修正的特征列。前两列是常数与上下文点数，后面依次加入区域统计量。"""
    ones = np.ones_like(logv)
    cols = [ones, ones * np.log(n_ctx), logv, ones * feats[0], ones * feats[1]]
    return np.stack(cols[:n_terms], axis=1)


def apply_correction(theta, mu, logv, n_ctx, feats, n_terms):
    """均值乘一个系数把被压向先验的部分推回去，对数方差减去多余的弥散。"""
    F = design_matrix(logv, n_ctx, feats, n_terms)
    a, b = F @ theta[:n_terms], F @ theta[n_terms:]
    return mu * np.exp(a), logv + b


def sample_fit_data(model, rng, n_tasks):
    data = []
    for _ in range(n_tasks):
        n_ctx = int(rng.integers(8, 25))
        design = "uniform" if rng.random() < 0.5 else "paired"
        xc = draw_design(rng, n_ctx, design)
        xq = rng.uniform(-X_HALF, X_HALF, N_QUERY)
        ell, sigma = draw_latent(rng)
        allx = np.concatenate([xc, xq])
        y = sample_gp(rng, allx, ell) + sigma * rng.standard_normal(len(allx))
        mu, logv = pfn_predict(model, xc, y[:n_ctx], xq)
        data.append((mu, logv, y[n_ctx:], n_ctx, context_features(xc, y[:n_ctx])))
    return data


def fit_correction(data, n_terms):
    """最小化期望 NLL 的解就是贝叶斯最优预测，所以这一步是朝贝叶斯走。"""
    def nll(theta):
        total = 0.0
        for mu, logv, yq, n_ctx, feats in data:
            m, lv = apply_correction(theta, mu, logv, n_ctx, feats, n_terms)
            total += np.mean(0.5 * (lv + np.log(2 * np.pi) + (yq - m) ** 2 / np.exp(lv)))
        return total / len(data)

    res = minimize(nll, np.zeros(2 * n_terms), method="L-BFGS-B")
    return res.x, float(nll(np.zeros(2 * n_terms))), float(res.fun)


def run_cell(model, ell, sigma, n_ctx, design, grids, thetas, n_tasks=N_EVAL_TASKS, seed=0):
    rng = np.random.default_rng(seed)
    names = ["raw"] + [f"correct{k}" for k in sorted(thetas)]
    acc = {n: {"kl": [], "nll": []} for n in names}
    for _ in range(n_tasks):
        xc = draw_design(rng, n_ctx, design)
        xq = rng.uniform(-X_HALF, X_HALF, N_QUERY)
        allx = np.concatenate([xc, xq])
        y = sample_gp(rng, allx, ell) + sigma * rng.standard_normal(len(allx))
        yc, yq = y[:n_ctx], y[n_ctx:]
        mu_e, var_e, _ = mixture_posterior(xc, yc, xq, *grids)
        mu_r, logv_r = pfn_predict(model, xc, yc, xq)
        feats = context_features(xc, yc)

        outs = {"raw": (mu_r, logv_r)}
        for k, th in thetas.items():
            outs[f"correct{k}"] = apply_correction(th, mu_r, logv_r, n_ctx, feats, k)
        for name, (m, lv) in outs.items():
            v = np.exp(lv)
            acc[name]["kl"].append(float(np.mean(gauss_kl(mu_e, var_e, m, v))))
            acc[name]["nll"].append(float(np.mean(
                0.5 * (lv + np.log(2 * np.pi) + (yq - m) ** 2 / v))))

    row = {"ell": ell, "sigma": sigma, "n_ctx": n_ctx, "design": design}
    for name in names:
        row[name] = {"kl": float(np.mean(acc[name]["kl"])),
                     "kl_se": float(np.std(acc[name]["kl"]) / np.sqrt(n_tasks)),
                     "nll": float(np.mean(acc[name]["nll"]))}
    return row


def report(rows, thetas, fits):
    names = ["raw"] + [f"correct{k}" for k in sorted(thetas)]
    print(f"\n    {'修正项数':<12}{'拟合集 NLL':>13}{'留出集 NLL':>13}")
    for k in sorted(thetas):
        print(f"    {k:<12}{fits[k]['fit']:>13.4f}{fits[k]['held']:>13.4f}")
    print(f"    {'不修正':<12}{fits[min(thetas)]['fit0']:>13.4f}"
          f"{fits[min(thetas)]['held0']:>13.4f}")

    print(f"\n    {'方法':<14}{'超额 KL 均值':>14}{'相对 raw':>10}{'最差格子':>11}"
          f"{'相对 raw':>10}{'留出 NLL 均值':>15}{'变好的格子':>12}")
    stats = {}
    raw_kl = np.array([r["raw"]["kl"] for r in rows])
    for name in names:
        kl = np.array([r[name]["kl"] for r in rows])
        nll = np.array([r[name]["nll"] for r in rows])
        rn = np.array([r["raw"]["nll"] for r in rows])
        better = int((kl < raw_kl).sum())
        stats[name] = {"kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
                       "kl_ratio": float(kl.mean() / raw_kl.mean()),
                       "kl_max_ratio": float(kl.max() / raw_kl.max()),
                       "nll_mean": float(nll.mean()), "nll_delta": float((nll - rn).mean()),
                       "n_better": better}
        print(f"    {name:<14}{kl.mean():>14.4f}{kl.mean() / raw_kl.mean():>10.3f}"
              f"{kl.max():>11.4f}{kl.max() / raw_kl.max():>10.3f}"
              f"{nll.mean():>15.4f}{f'{better}/{len(rows)}':>12}")

    print(f"\n    按上下文点数分组的超额 KL 均值（括号里是相对 raw 的比值）")
    ns = sorted({r["n_ctx"] for r in rows})
    print(f"    {'方法':<14}" + "".join(f"{f'n={n}':>20}" for n in ns))
    for name in names:
        cells = []
        for n in ns:
            sub = [r for r in rows if r["n_ctx"] == n]
            v = np.mean([r[name]["kl"] for r in sub])
            base = np.mean([r["raw"]["kl"] for r in sub])
            cells.append(f"{v:.4f} ({v / base:.3f})")
        stats[name]["by_n"] = cells
        print(f"    {name:<14}" + "".join(f"{c:>20}" for c in cells))
    return stats


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else CKPT
    term_list = [2, 5]

    model, d_model = load_pfn(ckpt)
    print(f"    检查点 {ckpt}，宽度 {d_model}", flush=True)

    fit_data = sample_fit_data(model, np.random.default_rng(12345), N_FIT_TASKS)
    held_data = sample_fit_data(model, np.random.default_rng(999), N_FIT_TASKS // 2)
    thetas, fits = {}, {}

    def held_nll(theta, n_terms):
        return float(np.mean([np.mean(0.5 * (lv + np.log(2 * np.pi) + (yq - m) ** 2 / np.exp(lv)))
                              for mu, logv, yq, n, f in held_data
                              for m, lv in [apply_correction(theta, mu, logv, n, f, n_terms)]]))

    for k in term_list:
        th, nll0, nll1 = fit_correction(fit_data, k)
        thetas[k] = th
        fits[k] = {"theta": th.tolist(), "fit": nll1, "fit0": nll0,
                   "held": held_nll(th, k), "held0": held_nll(np.zeros(2 * k), k)}
        print(f"    {k} 项修正：拟合集 NLL {nll0:.4f} -> {nll1:.4f}，"
              f"留出集 {fits[k]['held0']:.4f} -> {fits[k]['held']:.4f}", flush=True)

    grids = quad_grids()
    rows = []
    for i, (ell, sigma, n_ctx, design) in enumerate(sweep_cells()):
        rows.append(run_cell(model, ell, sigma, n_ctx, design, grids, thetas))
        print(f"    格子 {i + 1}/96 完成", flush=True)

    OUT_PATH.write_text(json.dumps({"rows": rows, "fits": fits}, ensure_ascii=False, indent=1))
    stats = report(rows, thetas, fits)
    OUT_PATH.write_text(json.dumps({"rows": rows, "fits": fits, "stats": stats},
                                   ensure_ascii=False, indent=1))
    print(f"\n    结果写到 {OUT_PATH}")
