import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import spearmanr

from exp_conditioning import (CKPT, ELL_HI, ELL_LO, N_QUERY, PRIOR_PREC, SIG_HI, SIG_LO,
                              X_HALF, draw_design, load_pfn, sample_gp, sweep_cells)
from identifiability import conditioning, gauss_kl, gp_posterior, mixture_posterior

N_TASKS = 40
QUAD_ELL, QUAD_SIG = 24, 14
OUT_PATH = Path("results/conditioning.json")


def quad_grids(n_ell=QUAD_ELL, n_sig=QUAD_SIG):
    return (np.exp(np.linspace(np.log(ELL_LO), np.log(ELL_HI), n_ell)),
            np.exp(np.linspace(np.log(SIG_LO), np.log(SIG_HI), n_sig)))


def pfn_predict(model, xc, yc, xq):
    x = torch.tensor(np.concatenate([xc, xq])[None], dtype=torch.float32)
    y = torch.tensor(np.concatenate([yc, np.zeros_like(xq)])[None], dtype=torch.float32)
    with torch.no_grad():
        mu, logv = model(x, y, len(xc))
    return mu[0].numpy().astype(np.float64), np.exp(logv[0].numpy().astype(np.float64))


def implied_latent(xc, yc, xq, mu_t, var_t):
    """把一个预测分布投影回单个 (ell, sigma) 的 GP 后验族，返回对数坐标下的隐变量。

    精确混合后验与网络输出都投到同一个族上，两者之差就是纯粹的摊销误差，
    与「混合后验本身不是单个 z 的后验」这件事无关。
    """
    def obj(z):
        ell, sigma = np.exp(z)
        m, v = gp_posterior(xc, yc, xq, ell, sigma)
        return float(np.mean(gauss_kl(mu_t, var_t, m, v)))

    bounds = [(np.log(ELL_LO), np.log(ELL_HI)), (np.log(SIG_LO), np.log(SIG_HI))]
    best, best_val = None, np.inf
    for z0 in ([np.log(0.1), np.log(0.1)], [np.log(1.0), np.log(0.05)],
               [np.log(0.03), np.log(0.3)]):
        r = minimize(obj, z0, method="L-BFGS-B", bounds=bounds)
        if r.fun < best_val:
            best, best_val = r.x, r.fun
    edge = any(min(abs(best[i] - lo), abs(best[i] - hi)) < 1e-3
               for i, (lo, hi) in enumerate(bounds))
    return best, edge, best_val


Z_PRIOR = np.array([0.5 * (np.log(ELL_LO) + np.log(ELL_HI)),
                    0.5 * (np.log(SIG_LO) + np.log(SIG_HI))])


def run_cell(model, ell, sigma, n_ctx, design, ell_grid, sig_grid, n_tasks=N_TASKS, seed=0):
    rng = np.random.default_rng(seed)
    gaps, bayes_nll, frac_v1, geo = [], [], [], []
    z_hats, z_stars, Gs, resid = [], [], [], []
    mu_num, mu_den, dlogv = 0.0, 0.0, []
    excess_var, mean_err2 = [], []
    n_edge = 0
    for _ in range(n_tasks):
        xc = draw_design(rng, n_ctx, design)
        xq = rng.uniform(-X_HALF, X_HALF, N_QUERY)
        allx = np.concatenate([xc, xq])
        f = sample_gp(rng, allx, ell)
        y = f + sigma * rng.standard_normal(len(allx))
        yc, yq = y[:n_ctx], y[n_ctx:]

        mu_e, var_e, _ = mixture_posterior(xc, yc, xq, ell_grid, sig_grid)
        mu_p, var_p = pfn_predict(model, xc, yc, xq)
        gaps.append(float(np.mean(gauss_kl(mu_e, var_e, mu_p, var_p))))
        bayes_nll.append(float(np.mean(0.5 * (np.log(2 * np.pi * var_e)
                                              + (yq - mu_e) ** 2 / var_e))))
        # 不依赖任何投影的收缩测量：先验的预测均值为零，所以更新不足会让斜率小于 1
        mu_num += float(mu_e @ mu_p)
        mu_den += float(mu_e @ mu_e)
        dlogv.append(float(np.mean(np.log(var_p) - np.log(var_e))))
        # 用高斯 NLL 训练的网络，其最优方差要覆盖自身均值误差：
        # 若 var_p - var_e 恰好等于 (mu_p - mu_e)^2，方差膨胀就只是均值误差的后果
        excess_var.append(float(np.mean(var_p - var_e)))
        mean_err2.append(float(np.mean((mu_p - mu_e) ** 2)))

        # 误差方向：投影到广义特征基，看误差落在高 lambda 方向的比例
        vals, vecs, F, G = conditioning(xc, xq, ell, sigma, PRIOR_PREC)
        A = F + np.diag(PRIOR_PREC)
        V = vecs / np.sqrt(np.einsum("ia,ij,ja->a", vecs, A, vecs))  # 归一到 A 范数为 1
        z_hat, edge_hat, res_hat = implied_latent(xc, yc, xq, mu_p, var_p)
        z_star, edge_star, res_star = implied_latent(xc, yc, xq, mu_e, var_e)
        resid.append((res_hat, res_star))
        # 两个拟合都顶在先验盒子边界上时，两者之差不再携带方向信息
        if edge_hat and edge_star:
            n_edge += 1
        else:
            c = np.linalg.solve(V, z_hat - z_star)
            if c @ c > 1e-12:
                frac_v1.append(float(c[0] ** 2 / (c @ c)))
            z_hats.append(z_hat)
            z_stars.append(z_star)
            Gs.append(G)
        geo.append((vals[0], vals.sum(), np.trace(F), np.trace(G)))

    geo = np.mean(geo, axis=0)
    zh = np.array(z_hats) - Z_PRIOR
    zs = np.array(z_stars) - Z_PRIOR
    return {"ell": ell, "sigma": sigma, "n_ctx": n_ctx, "design": design,
            "gap": float(np.mean(gaps)), "gap_se": float(np.std(gaps) / np.sqrt(len(gaps))),
            "bayes_nll": float(np.mean(bayes_nll)),
            "frac_v1": float(np.mean(frac_v1)) if frac_v1 else float("nan"),
            "n_dir": len(frac_v1), "n_edge": n_edge,
            "kappa": float(geo[0]), "trace_ratio": float(geo[1]),
            "trF": float(geo[2]), "trG": float(geo[3]),
            # 逐任务的隐变量位移，用来判定失败模型
            "shift_net": float(np.mean(np.linalg.norm(zh, axis=1))),
            "shift_exact": float(np.mean(np.linalg.norm(zs, axis=1))),
            "z_net": zh.tolist(), "z_exact": zs.tolist(),
            "G_mean": np.mean(Gs, axis=0).tolist(),
            # 投影残差：网络输出有多少不落在单个隐变量的后验族里
            "fit_resid_net": float(np.mean([a for a, _ in resid])),
            "fit_resid_exact": float(np.mean([b for _, b in resid])),
            "mean_slope": mu_num / mu_den,
            "dlogvar": float(np.mean(dlogv)),
            "excess_var": float(np.mean(excess_var)),
            "mean_err2": float(np.mean(mean_err2))}


def quadrature_check(model):
    """求积网格必须比要测的差距精细得多，否则测的是求积误差。"""
    rng = np.random.default_rng(123)
    coarse, fine = quad_grids(), quad_grids(40, 24)
    diffs = []
    for _ in range(12):
        n_ctx = 16
        xc = rng.uniform(-X_HALF, X_HALF, n_ctx)
        xq = rng.uniform(-X_HALF, X_HALF, N_QUERY)
        allx = np.concatenate([xc, xq])
        y = sample_gp(rng, allx, 0.1) + 0.1 * rng.standard_normal(len(allx))
        a = mixture_posterior(xc, y[:n_ctx], xq, *coarse)
        b = mixture_posterior(xc, y[:n_ctx], xq, *fine)
        diffs.append(float(np.mean(gauss_kl(b[0], b[1], a[0], a[1]))))
    print(f"    求积网格 {QUAD_ELL}x{QUAD_SIG} 相对 40x24 的 KL：{np.mean(diffs):.2e}")
    return float(np.mean(diffs))


def report(rows, quad_err):
    keys = ["kappa", "trace_ratio", "trF", "trG", "bayes_nll"]
    gap = np.array([r["gap"] for r in rows])
    print(f"\n    {'design':>9}{'n_ctx':>6}{'ell':>8}{'sigma':>7}{'gap':>10}{'±se':>9}"
          f"{'kappa':>9}{'tr A^-1G':>10}{'tr F':>9}{'tr G':>8}{'v1 占比':>10}{'边界':>6}")
    last = None
    for r in rows:
        if last is not None and (r["n_ctx"], r["design"]) != last:
            print()
        last = (r["n_ctx"], r["design"])
        print(f"    {r['design']:>9}{r['n_ctx']:>6}{r['ell']:>8.3f}{r['sigma']:>7.2f}"
              f"{r['gap']:>10.4f}{r['gap_se']:>9.4f}{r['kappa']:>9.3f}"
              f"{r['trace_ratio']:>10.3f}{r['trF']:>9.2f}{r['trG']:>8.3f}"
              f"{r['frac_v1']:>10.3f}{r['n_edge']:>6}")

    print(f"\n    差距的动态范围 {gap.min():.4f} – {gap.max():.4f}"
          f"（求积误差 {quad_err:.2e}，相差 {gap.min() / max(quad_err, 1e-12):.0f} 倍以上）")

    print(f"\n    {'预测量':<14}{'Spearman':>10}{'log-log 斜率':>14}{'log-log R^2':>13}")
    stats = {}
    for k in keys:
        v = np.array([r[k] for r in rows])
        rho = spearmanr(v, gap).statistic
        if np.all(v > 0):
            A = np.vstack([np.log(v), np.ones(len(v))]).T
            coef, *_ = np.linalg.lstsq(A, np.log(gap), rcond=None)
            r2 = 1 - ((np.log(gap) - A @ coef) ** 2).sum() \
                / ((np.log(gap) - np.log(gap).mean()) ** 2).sum()
            slope = float(coef[0])
        else:
            slope, r2 = float("nan"), float("nan")  # 贝叶斯 NLL 会取负值，不能取对数
        stats[k] = {"spearman": float(rho), "slope": slope, "r2": float(r2)}
        print(f"    {k:<14}{rho:>10.3f}{slope:>14.3f}{r2:>13.3f}")

    kap = np.array([r["kappa"] for r in rows])
    slope = float((kap @ gap) / (kap @ kap))
    resid = gap - slope * kap
    r2_lin = 1 - (resid ** 2).sum() / ((gap - gap.mean()) ** 2).sum()
    stats["linear_through_origin"] = {"a": slope, "r2": r2_lin, "eps": 2 * slope}
    print(f"\n    过原点线性拟合 gap = a * kappa：a = {slope:.4f}，R^2 = {r2_lin:.3f}"
          f"，隐含信息缺口 eps = 2a = {2 * slope:.4f} nat")

    fr = np.array([r["frac_v1"] for r in rows])
    ok = ~np.isnan(fr)
    print(f"    误差落在 v1 方向的占比：均值 {fr[ok].mean():.3f}"
          f"（各方向等概率的零假设为 0.500），{ok.sum()} 个可用格子里"
          f" {int((fr[ok] > 0.5).sum())} 个超过 0.5")
    stats["frac_v1"] = {"mean": float(fr[ok].mean()), "n_cells": int(ok.sum()),
                        "n_above_half": int((fr[ok] > 0.5).sum())}

    # 设计交叉：同样的点数、同样的先验，只改上下文点的位置
    print(f"\n    设计交叉（正号表示成对设计更差）\n"
          f"    {'n_ctx':>6}{'ell':>8}{'sigma':>7}{'kappa 之差':>13}{'gap 之差':>12}"
          f"{'±se':>9}{'tr F 之差':>12}{'符号一致':>10}")
    agree = 0
    pairs = {}
    for r in rows:
        pairs.setdefault((r["n_ctx"], r["ell"], r["sigma"]), {})[r["design"]] = r
    cross = []
    for key, d in pairs.items():
        if len(d) < 2:
            continue
        u, p = d["uniform"], d["paired"]
        dk, dg = p["kappa"] - u["kappa"], p["gap"] - u["gap"]
        se = np.hypot(p["gap_se"], u["gap_se"])
        df = p["trF"] - u["trF"]
        same = np.sign(dk) == np.sign(dg)
        agree += int(same)
        cross.append({"n_ctx": key[0], "ell": key[1], "sigma": key[2],
                      "d_kappa": dk, "d_gap": dg, "d_gap_se": float(se), "d_trF": df,
                      "sign_agree": bool(same)})
        print(f"    {key[0]:>6}{key[1]:>8.3f}{key[2]:>7.2f}{dk:>+13.3f}{dg:>+12.4f}"
              f"{se:>9.4f}{df:>+12.2f}{'是' if same else '否':>10}")
    print(f"\n    kappa 与实测差距的符号一致：{agree}/{len(cross)} 格")
    stats["crossover"] = {"rows": cross, "n_agree": agree, "n_total": len(cross)}
    stats.update(update_deficit(rows))
    return stats


def update_deficit(rows):
    """判定失败模型：网络的隐含隐变量是否系统性地停在先验与精确后验之间。

    z_net 与 z_exact 都是同一批数据的确定性函数，没有测量误差，
    所以回归斜率不受衰减偏差影响。
    """
    zn = np.vstack([np.array(r["z_net"]) for r in rows])
    ze = np.vstack([np.array(r["z_exact"]) for r in rows])
    B, *_ = np.linalg.lstsq(ze, zn, rcond=None)  # z_net ≈ z_exact @ B
    beta_iso = float((ze * zn).sum() / (ze * ze).sum())
    print(f"\n    收缩矩阵 B（z_net ≈ B (z_exact - 先验均值)，全部 {len(zn)} 个任务合并拟合）")
    print(f"      [[{B[0, 0]:+.3f} {B[1, 0]:+.3f}]\n       [{B[0, 1]:+.3f} {B[1, 1]:+.3f}]]")
    print(f"    各向同性收缩系数 beta = {beta_iso:.3f}"
          f"（1.000 表示与精确贝叶斯一致，小于 1 表示更新不足）")

    n_less = sum(int(r["shift_net"] < r["shift_exact"]) for r in rows)
    print(f"    隐变量位移的幅度：{n_less}/{len(rows)} 个格子里网络小于精确贝叶斯")

    # 不依赖投影的收缩测量
    ms = np.array([r["mean_slope"] for r in rows])
    dv = np.array([r["dlogvar"] for r in rows])
    print(f"\n    预测均值的回归斜率（网络对精确后验）：中位数 {np.median(ms):.3f}，"
          f"范围 {ms.min():.3f} – {ms.max():.3f}，{int((ms < 1).sum())}/{len(ms)} 个格子小于 1")
    print(f"    预测方差的对数之差（网络减精确）：中位数 {np.median(dv):+.3f}，"
          f"{int((dv > 0).sum())}/{len(dv)} 个格子为正（网络更宽，更接近先验）")

    ex = np.array([r["excess_var"] for r in rows])
    me = np.array([r["mean_err2"] for r in rows])
    print(f"    方差膨胀与自身均值误差的比：中位数 {np.median(ex / me):.3f}"
          f"（1.000 表示膨胀恰好覆盖均值误差，此时膨胀只是均值误差的后果），"
          f"Spearman {spearmanr(ex, me).statistic:+.3f}")

    # 投影残差：这套隐变量语言能覆盖网络输出的多少
    rn = np.array([r["fit_resid_net"] for r in rows])
    re_ = np.array([r["fit_resid_exact"] for r in rows])
    gap_all = np.array([r["gap"] for r in rows])
    print(f"\n    单隐变量族的投影残差：网络 {np.median(rn):.4f}，精确后验 {np.median(re_):.4f}，"
          f"相对差距的中位数比值 {np.median(rn / gap_all):.3f}")

    # 逐格子的收缩系数，看它是不是稳定的常数
    betas = np.array([float((np.array(r["z_exact"]) * np.array(r["z_net"])).sum()
                            / (np.array(r["z_exact"]) ** 2).sum()) for r in rows])
    print(f"    逐格子的 beta：中位数 {np.median(betas):.3f}，"
          f"四分位区间 {np.percentile(betas, 25):.3f} – {np.percentile(betas, 75):.3f}，"
          f"与上下文点数的 Spearman {spearmanr([r['n_ctx'] for r in rows], betas).statistic:+.3f}")

    # 用全局拟合的 B 预测每个格子的差距：gap = 0.5 * E[(z_net - z_exact)^T G (z_net - z_exact)]
    gap = np.array([r["gap"] for r in rows])
    pred_B, pred_iso = [], []
    for r in rows:
        ze_c, G = np.array(r["z_exact"]), np.array(r["G_mean"])
        for out, mat in ((pred_B, B.T - np.eye(2)), (pred_iso, (beta_iso - 1) * np.eye(2))):
            d = ze_c @ mat.T
            out.append(0.5 * np.mean(np.einsum("ti,ij,tj->t", d, G, d)))
    print(f"\n    {'失败模型':<28}{'Spearman':>10}{'比值中位数':>12}{'log-log R^2':>13}")
    out = {"B": B.tolist(), "beta_iso": beta_iso, "n_shift_less": n_less,
           "mean_slope_median": float(np.median(ms)), "n_slope_below_one": int((ms < 1).sum()),
           "dlogvar_median": float(np.median(dv)), "n_dlogvar_pos": int((dv > 0).sum()),
           "excess_var_over_mean_err2": float(np.median(ex / me)),
           "fit_resid_ratio_median": float(np.median(rn / gap_all)),
           "beta_per_cell_median": float(np.median(betas)),
           "beta_per_cell_iqr": [float(np.percentile(betas, 25)),
                                 float(np.percentile(betas, 75))]}
    for name, p in (("更新不足（B 为 2x2）", np.array(pred_B)),
                    ("更新不足（各向同性 beta）", np.array(pred_iso)),
                    ("固定隐变量分辨率（tr G）", np.array([r["trG"] for r in rows])),
                    ("信息缺口（kappa）", np.array([r["kappa"] for r in rows]))):
        rho = spearmanr(p, gap).statistic
        A = np.vstack([np.log(p), np.ones(len(p))]).T
        coef, *_ = np.linalg.lstsq(A, np.log(gap), rcond=None)
        r2 = 1 - ((np.log(gap) - A @ coef) ** 2).sum() \
            / ((np.log(gap) - np.log(gap).mean()) ** 2).sum()
        out[name] = {"spearman": float(rho), "median_ratio": float(np.median(p / gap)),
                     "r2": float(r2)}
        print(f"    {name:<28}{rho:>10.3f}{np.median(p / gap):>12.3f}{r2:>13.3f}")
    return {"failure_model": out}


if __name__ == "__main__":
    n_tasks = int(sys.argv[1]) if len(sys.argv) > 1 else N_TASKS
    ckpt = sys.argv[2] if len(sys.argv) > 2 else CKPT
    OUT_PATH = Path(f"results/conditioning_{Path(ckpt).stem}.json")

    model, d_model = load_pfn(ckpt)
    print(f"    检查点 {ckpt}，宽度 {d_model}", flush=True)

    cells = sweep_cells()
    if len(sys.argv) > 3:
        cells = cells[:int(sys.argv[3])]

    quad_err = quadrature_check(model)
    ell_grid, sig_grid = quad_grids()
    rows = []
    for i, (ell, sigma, n_ctx, design) in enumerate(cells):
        rows.append(run_cell(model, ell, sigma, n_ctx, design, ell_grid, sig_grid, n_tasks))
        print(f"    格子 {i + 1}/{len(cells)} 完成", flush=True)
    OUT_PATH.write_text(json.dumps({"rows": rows, "quad_err": quad_err, "n_tasks": n_tasks},
                                   ensure_ascii=False, indent=1))
    stats = report(rows, quad_err)
    OUT_PATH.write_text(json.dumps({"rows": rows, "stats": stats,
                                    "quad_err": quad_err, "n_tasks": n_tasks},
                                   ensure_ascii=False, indent=1))
    print(f"\n    结果写到 {OUT_PATH}")
