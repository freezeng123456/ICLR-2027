import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import spearmanr

from exp_conditioning import load_pfn
from exp_jump import (CKPT, N_QUERY, RATE_HI, RATE_LO, SIG_HI, SIG_LO, X_HALF,
                      draw_design, sweep_cells)
from identifiability import gauss_kl
from prior_jump import LEVELS, mixture_posterior, predict_single, sample_task

# 与高斯过程那一套完全相同的测量，换到跳变过程先验上。
# 差距的定义不变：网络只输出均值与方差、训练目标是高斯 NLL，
# 所以它能达到的最优就是精确混合后验的矩匹配高斯。
# 这里的精确后验是多峰的高斯混合，所以矩匹配这一步真的有损——
# 「方差膨胀是对自身均值误差的对冲」那条机制在这里受到更强的检验。

N_TASKS = 40
QUAD_RATE, QUAD_SIG = 16, 10
Z_PRIOR = np.array([0.5 * (np.log(RATE_LO) + np.log(RATE_HI)),
                    0.5 * (np.log(SIG_LO) + np.log(SIG_HI))])


def quad_grids(n_rate=QUAD_RATE, n_sig=QUAD_SIG):
    return (np.exp(np.linspace(np.log(RATE_LO), np.log(RATE_HI), n_rate)),
            np.exp(np.linspace(np.log(SIG_LO), np.log(SIG_HI), n_sig)))


def pfn_predict(model, xc, yc, xq):
    x = torch.tensor(np.concatenate([xc, xq])[None], dtype=torch.float32)
    y = torch.tensor(np.concatenate([yc, np.zeros_like(xq)])[None], dtype=torch.float32)
    with torch.no_grad():
        mu, logv = model(x, y, len(xc))
    return mu[0].numpy().astype(np.float64), np.exp(logv[0].numpy().astype(np.float64))


def single_moments(xc, yc, xq, rate, sigma):
    """单个 (rate, sigma) 下预测分布的矩匹配高斯。"""
    g, _ = predict_single(xc, yc, xq, rate, sigma)
    mu = g @ LEVELS
    var = g @ (LEVELS ** 2) + sigma ** 2 - mu ** 2
    return mu, np.maximum(var, 1e-8)


def implied_latent(xc, yc, xq, mu_t, var_t):
    """把一个预测分布投影回单个 (rate, sigma) 的后验族，返回对数坐标下的隐变量。

    精确混合后验与网络输出都投到同一个族上，两者之差就是纯粹的摊销误差。
    """
    def obj(z):
        rate, sigma = np.exp(z)
        m, v = single_moments(xc, yc, xq, rate, sigma)
        return float(np.mean(gauss_kl(mu_t, var_t, m, v)))

    bounds = [(np.log(RATE_LO), np.log(RATE_HI)), (np.log(SIG_LO), np.log(SIG_HI))]
    best, best_val = None, np.inf
    for z0 in ([np.log(3.0), np.log(0.1)], [np.log(20.0), np.log(0.05)],
               [np.log(1.0), np.log(0.3)]):
        r = minimize(obj, z0, method="L-BFGS-B", bounds=bounds)
        if r.fun < best_val:
            best, best_val = r.x, r.fun
    edge = any(min(abs(best[i] - lo), abs(best[i] - hi)) < 1e-3
               for i, (lo, hi) in enumerate(bounds))
    return best, edge, best_val


def quadrature_check():
    """求积网格必须比要测的差距精细得多，否则测的是求积误差。"""
    rng = np.random.default_rng(123)
    coarse, fine = quad_grids(), quad_grids(26, 16)
    diffs = []
    for _ in range(12):
        xc = rng.uniform(-X_HALF, X_HALF, 16)
        xq = rng.uniform(-X_HALF, X_HALF, N_QUERY)
        y = sample_task(rng, np.concatenate([xc, xq]), 6.0, 0.15)
        a = mixture_posterior(xc, y[:16], xq, *coarse)
        b = mixture_posterior(xc, y[:16], xq, *fine)
        diffs.append(float(np.mean(gauss_kl(b[0], b[1], a[0], a[1]))))
    err = float(np.mean(diffs))
    print(f"    求积网格 {QUAD_RATE}x{QUAD_SIG} 相对 26x16 的 KL：{err:.2e}")
    return err


def run_cell(model, rate, sigma, n_ctx, design, grids, n_tasks=N_TASKS, seed=0):
    rng = np.random.default_rng(seed)
    gaps, bayes_nll, resid = [], [], []
    z_hats, z_stars, second_w = [], [], []
    mu_num, mu_den, dlogv, excess_var, mean_err2 = 0.0, 0.0, [], [], []
    n_edge = 0
    for _ in range(n_tasks):
        xc = draw_design(rng, n_ctx, design)
        xq = rng.uniform(-X_HALF, X_HALF, N_QUERY)
        allx = np.concatenate([xc, xq])
        y = sample_task(rng, allx, rate, sigma)
        yc, yq = y[:n_ctx], y[n_ctx:]

        mu_e, var_e, _ = mixture_posterior(xc, yc, xq, *grids)
        mu_p, var_p = pfn_predict(model, xc, yc, xq)
        gaps.append(float(np.mean(gauss_kl(mu_e, var_e, mu_p, var_p))))
        bayes_nll.append(float(np.mean(0.5 * (np.log(2 * np.pi * var_e)
                                              + (yq - mu_e) ** 2 / var_e))))
        mu_num += float(mu_e @ mu_p)
        mu_den += float(mu_e @ mu_e)
        dlogv.append(float(np.mean(np.log(var_p) - np.log(var_e))))
        excess_var.append(float(np.mean(var_p - var_e)))
        mean_err2.append(float(np.mean((mu_p - mu_e) ** 2)))

        # 精确后验的多峰程度，用来看矩匹配这一步损失多少
        g, _ = predict_single(xc, yc, xq, rate, sigma)
        second_w.append(float(np.mean(np.sort(g, axis=1)[:, -2])))

        z_hat, edge_hat, res_hat = implied_latent(xc, yc, xq, mu_p, var_p)
        z_star, edge_star, res_star = implied_latent(xc, yc, xq, mu_e, var_e)
        resid.append((res_hat, res_star))
        if edge_hat and edge_star:
            n_edge += 1
        else:
            z_hats.append(z_hat)
            z_stars.append(z_star)

    zh = np.array(z_hats) - Z_PRIOR
    zs = np.array(z_stars) - Z_PRIOR
    return {"rate": rate, "sigma": sigma, "n_ctx": n_ctx, "design": design,
            "gap": float(np.mean(gaps)), "gap_se": float(np.std(gaps) / np.sqrt(n_tasks)),
            "bayes_nll": float(np.mean(bayes_nll)),
            "mean_slope": mu_num / mu_den, "dlogvar": float(np.mean(dlogv)),
            "excess_var": float(np.mean(excess_var)), "mean_err2": float(np.mean(mean_err2)),
            "second_weight": float(np.mean(second_w)), "n_edge": n_edge,
            "shift_net": float(np.mean(np.linalg.norm(zh, axis=1))),
            "shift_exact": float(np.mean(np.linalg.norm(zs, axis=1))),
            "z_net": zh.tolist(), "z_exact": zs.tolist(),
            "fit_resid_net": float(np.mean([a for a, _ in resid])),
            "fit_resid_exact": float(np.mean([b for _, b in resid]))}


def report(rows, quad_err):
    gap = np.array([r["gap"] for r in rows])
    print(f"\n    {'design':>9}{'n_ctx':>6}{'rate':>8}{'sigma':>7}{'gap':>10}{'±se':>9}"
          f"{'次峰权重':>10}{'边界':>6}")
    last = None
    for r in rows:
        if last is not None and (r["n_ctx"], r["design"]) != last:
            print()
        last = (r["n_ctx"], r["design"])
        print(f"    {r['design']:>9}{r['n_ctx']:>6}{r['rate']:>8.2f}{r['sigma']:>7.2f}"
              f"{r['gap']:>10.4f}{r['gap_se']:>9.4f}{r['second_weight']:>10.3f}"
              f"{r['n_edge']:>6}")
    print(f"\n    差距的动态范围 {gap.min():.4f} – {gap.max():.4f}"
          f"（求积误差 {quad_err:.2e}，相差 {gap.min() / max(quad_err, 1e-12):.0f} 倍以上）")

    # 走向：差距随上下文点数是升还是降
    by = {}
    for r in rows:
        by.setdefault((r["design"], r["rate"], r["sigma"]), {})[r["n_ctx"]] = r["gap"]
    ns = sorted({r["n_ctx"] for r in rows})
    up, ratios, bounds = 0, [], {}
    tot = 0
    for (design, rate, _), d in by.items():
        if len(d) < len(ns):
            continue
        tot += 1
        g = [d[n] for n in ns]
        rising = g[-1] > g[0]
        up += int(rising)
        ratios.append(g[-1] / g[0])
        b = bounds.setdefault(design, {"rise_min": np.inf, "fall_max": 0.0})
        if rising:
            b["rise_min"] = min(b["rise_min"], rate)
        else:
            b["fall_max"] = max(b["fall_max"], rate)
    print(f"\n    走向：{tot} 个组合里 {up} 个随上下文点数上升，最大增幅 {max(ratios):.1f} 倍")
    for design, b in bounds.items():
        print(f"    {design} 设计的分界落在跳变速率 {b['fall_max']:.2f} 与 {b['rise_min']:.2f} 之间")

    # 形状：两个坐标分别按真值落在先验均值哪一侧分组
    print(f"\n    收缩系数（按该坐标自己那一维分组）")
    mids = {"rate": np.exp(Z_PRIOR[0]), "sigma": np.exp(Z_PRIOR[1])}
    shape = {}
    for coord, key in ((0, "rate"), (1, "sigma")):
        for side, sel in (("下方", lambda r: r[key] < mids[key]),
                          ("上方", lambda r: r[key] > mids[key])):
            sub = [r for r in rows if sel(r) and r["z_net"]]
            if not sub:
                print(f"      log {key} 真值在先验均值{side}：该侧没有格子")
                continue
            zn = np.vstack([np.array(r["z_net"]) for r in sub])[:, coord]
            ze = np.vstack([np.array(r["z_exact"]) for r in sub])[:, coord]
            b = float((ze @ zn) / (ze @ ze))
            shape[f"log {key} {side}"] = b
            print(f"      log {key} 真值在先验均值{side}：{b:.3f}（{len(sub)} 个格子）")

    zn = np.vstack([np.array(r["z_net"]) for r in rows if r["z_net"]])
    ze = np.vstack([np.array(r["z_exact"]) for r in rows if r["z_net"]])
    print(f"\n    {'坐标':<10}{'乘性 beta':>11}{'乘性残差':>11}{'加性偏移':>11}{'加性残差':>11}")
    for coord, name in ((0, "log rate"), (1, "log sigma")):
        a, b_ = ze[:, coord], zn[:, coord]
        beta = float((a @ b_) / (a @ a))
        delta = float(np.mean(b_ - a))
        print(f"    {name:<10}{beta:>11.3f}{np.std(b_ - beta * a):>11.4f}"
              f"{delta:>+11.3f}{np.std(b_ - a - delta):>11.4f}")
        shape[f"{name} beta"] = beta
        shape[f"{name} delta"] = delta

    ms = np.array([r["mean_slope"] for r in rows])
    dv = np.array([r["dlogvar"] for r in rows])
    ex = np.array([r["excess_var"] for r in rows])
    me = np.array([r["mean_err2"] for r in rows])
    ratio = ex / me
    print(f"\n    预测均值的回归斜率：中位数 {np.median(ms):.3f}，"
          f"{int((ms < 1).sum())}/{len(ms)} 个格子小于 1")
    print(f"    预测方差的对数之差：中位数 {np.median(dv):+.3f}，"
          f"{int((dv > 0).sum())}/{len(dv)} 个格子为正")
    print(f"    方差膨胀与自身均值误差的比：中位数 {np.median(ratio):.3f}，"
          f"四分位区间 {np.percentile(ratio, 25):.3f} – {np.percentile(ratio, 75):.3f}，"
          f"膨胀为正的格子 {int((ex > 0).sum())}/{len(ex)}")
    print(f"    方差对数之差与均值斜率的 Spearman：{spearmanr(dv, ms).statistic:+.3f}")

    rn = np.array([r["fit_resid_net"] for r in rows])
    re_ = np.array([r["fit_resid_exact"] for r in rows])
    print(f"\n    单隐变量族的投影残差：网络 {np.median(rn):.4f}，精确后验 {np.median(re_):.4f}，"
          f"相对差距的中位数比值 {np.median(rn / gap):.3f}")
    sw = np.array([r["second_weight"] for r in rows])
    print(f"    精确后验的次峰权重中位数 {np.median(sw):.3f}，"
          f"与差距的 Spearman {spearmanr(sw, gap).statistic:+.3f}")

    return {"shape": shape, "n_rising": up, "n_total": tot, "max_ratio": float(max(ratios)),
            "mean_slope_median": float(np.median(ms)), "n_slope_below": int((ms < 1).sum()),
            "dlogvar_median": float(np.median(dv)), "n_dlogvar_pos": int((dv > 0).sum()),
            "hedge_ratio_median": float(np.median(ratio)),
            "hedge_iqr": [float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))],
            "gap_mean": float(gap.mean()), "gap_max": float(gap.max()),
            "fit_resid_ratio": float(np.median(rn / gap))}


if __name__ == "__main__":
    n_tasks = int(sys.argv[1]) if len(sys.argv) > 1 else N_TASKS
    ckpt = sys.argv[2] if len(sys.argv) > 2 else CKPT
    cells = sweep_cells()
    suffix = ""
    if len(sys.argv) > 3:
        cells = cells[:int(sys.argv[3])]
        suffix = f"_first{len(cells)}"
    out_path = Path(f"results/jump_{Path(ckpt).stem}{suffix}.json")

    model, d_model = load_pfn(ckpt)
    print(f"    检查点 {ckpt}，宽度 {d_model}", flush=True)
    quad_err = quadrature_check()
    grids = quad_grids()
    rows = []
    for i, (rate, sigma, n_ctx, design) in enumerate(cells):
        rows.append(run_cell(model, rate, sigma, n_ctx, design, grids, n_tasks))
        print(f"    格子 {i + 1}/{len(cells)} 完成", flush=True)
    out_path.write_text(json.dumps({"rows": rows, "quad_err": quad_err}, ensure_ascii=False,
                                   indent=1))
    stats = report(rows, quad_err)
    out_path.write_text(json.dumps({"rows": rows, "stats": stats, "quad_err": quad_err},
                                   ensure_ascii=False, indent=1))
    print(f"\n    结果写到 {out_path}")
