import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import spearmanr

from exp_conditioning import (CKPT, ELL_HI, ELL_LO, N_QUERY, PFN, PRIOR_PREC, SIG_HI, SIG_LO,
                              X_HALF, cell_geometry, sample_gp, sweep_cells)
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
    return best


def run_cell(model, ell, sigma, n_ctx, ell_grid, sig_grid, n_tasks=N_TASKS, seed=0):
    rng = np.random.default_rng(seed)
    gaps, bayes_nll, frac_v1, geo = [], [], [], []
    for _ in range(n_tasks):
        xc = rng.uniform(-X_HALF, X_HALF, n_ctx)
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

        # 误差方向：投影到广义特征基，看误差落在高 lambda 方向的比例
        vals, vecs, F, G = conditioning(xc, xq, ell, sigma, PRIOR_PREC)
        A = F + np.diag(PRIOR_PREC)
        V = vecs / np.sqrt(np.einsum("ia,ij,ja->a", vecs, A, vecs))  # 归一到 A 范数为 1
        z_hat = implied_latent(xc, yc, xq, mu_p, var_p)
        z_star = implied_latent(xc, yc, xq, mu_e, var_e)
        c = np.linalg.solve(V, z_hat - z_star)
        if c @ c > 1e-12:
            frac_v1.append(float(c[0] ** 2 / (c @ c)))
        geo.append((vals[0], vals.sum(), np.trace(F), np.trace(G)))

    geo = np.mean(geo, axis=0)
    return {"ell": ell, "sigma": sigma, "n_ctx": n_ctx,
            "gap": float(np.mean(gaps)), "gap_se": float(np.std(gaps) / np.sqrt(len(gaps))),
            "bayes_nll": float(np.mean(bayes_nll)),
            "frac_v1": float(np.mean(frac_v1)), "n_dir": len(frac_v1),
            "kappa": float(geo[0]), "trace_ratio": float(geo[1]),
            "trF": float(geo[2]), "trG": float(geo[3])}


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
    print(f"\n    {'n_ctx':>6}{'ell':>8}{'sigma':>7}{'gap':>10}{'±se':>9}"
          f"{'kappa':>9}{'tr A^-1G':>10}{'tr F':>9}{'tr G':>8}{'误差在 v1 的占比':>18}")
    last_n = None
    for r in rows:
        if last_n is not None and r["n_ctx"] != last_n:
            print()
        last_n = r["n_ctx"]
        print(f"    {r['n_ctx']:>6}{r['ell']:>8.3f}{r['sigma']:>7.2f}{r['gap']:>10.4f}"
              f"{r['gap_se']:>9.4f}{r['kappa']:>9.3f}{r['trace_ratio']:>10.3f}"
              f"{r['trF']:>9.2f}{r['trG']:>8.3f}{r['frac_v1']:>18.3f}")

    print(f"\n    差距的动态范围 {gap.min():.4f} – {gap.max():.4f}"
          f"（求积误差 {quad_err:.2e}，相差 {gap.min() / max(quad_err, 1e-12):.0f} 倍以上）")
    print(f"\n    {'预测量':<14}{'Spearman':>10}{'log-log 斜率':>14}{'log-log R^2':>13}")
    stats = {}
    for k in keys:
        v = np.array([r[k] for r in rows])
        rho = spearmanr(v, gap).statistic
        A = np.vstack([np.log(v), np.ones(len(v))]).T
        coef, *_ = np.linalg.lstsq(A, np.log(gap), rcond=None)
        pred = A @ coef
        r2 = 1 - ((np.log(gap) - pred) ** 2).sum() / ((np.log(gap) - np.log(gap).mean()) ** 2).sum()
        stats[k] = {"spearman": float(rho), "slope": float(coef[0]), "r2": float(r2)}
        print(f"    {k:<14}{rho:>10.3f}{coef[0]:>14.3f}{r2:>13.3f}")

    kap = np.array([r["kappa"] for r in rows])
    slope = float((kap @ gap) / (kap @ kap))
    resid = gap - slope * kap
    r2_lin = 1 - (resid ** 2).sum() / ((gap - gap.mean()) ** 2).sum()
    print(f"\n    过原点线性拟合 gap = a * kappa：a = {slope:.4f}，R^2 = {r2_lin:.3f}"
          f"，隐含信息缺口 eps = 2a = {2 * slope:.4f} nat")

    fr = np.array([r["frac_v1"] for r in rows])
    print(f"    误差落在 v1 方向的占比：均值 {fr.mean():.3f}"
          f"（各方向等概率的零假设为 0.500），{len(fr)} 个格子里 {int((fr > 0.5).sum())} 个超过 0.5")
    return stats


if __name__ == "__main__":
    model = PFN()
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    n_tasks = int(sys.argv[1]) if len(sys.argv) > 1 else N_TASKS

    cells = sweep_cells()
    if len(sys.argv) > 2:
        cells = cells[:int(sys.argv[2])]

    quad_err = quadrature_check(model)
    ell_grid, sig_grid = quad_grids()
    rows = []
    for i, (ell, sigma, n_ctx) in enumerate(cells):
        rows.append(run_cell(model, ell, sigma, n_ctx, ell_grid, sig_grid, n_tasks))
        print(f"    格子 {i + 1}/{len(cells)} 完成", flush=True)
    stats = report(rows, quad_err)
    OUT_PATH.write_text(json.dumps({"rows": rows, "stats": stats,
                                    "quad_err": quad_err, "n_tasks": n_tasks},
                                   ensure_ascii=False, indent=1))
    print(f"\n    结果写到 {OUT_PATH}")
