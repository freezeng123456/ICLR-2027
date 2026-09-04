import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from exp_conditioning import (CKPT, N_QUERY, PFN, PRIOR_PREC, X_HALF, draw_design, sweep_cells)
from identifiability import conditioning

# 真实 PFN 的先验没有闭式，F 与 G 必须只靠模拟器采样估出来。
# 这里在解析答案已知的 GP 上验证这套估计量。

N_RFF = 1024
N_SIM = 256
N_DESIGNS = 4
DELTA = 0.08  # 对数坐标下的中心差分步长
LAG_EDGES = np.array([0.0, 0.04, 0.15, 0.35, 0.70, 1.20, 2.01])
OUT_PATH = Path("results/geometry_estimator.json")


def rff_basis(rng):
    """固定一组随机 Fourier 特征。共用随机数时，样本对 log ell 是光滑的，
    中心差分才有意义；用 K 的分解采样会因特征向量排序跳变而不可微。"""
    return rng.standard_normal(N_RFF), rng.uniform(0, 2 * np.pi, N_RFF)


def rff_sample(x, ell, v, b, theta):
    w = v / ell
    phi = np.sqrt(2.0 / N_RFF) * np.cos(x[:, None] * w[None, :] + b[None, :])
    return phi @ theta


def summaries(x, y, mask):
    """上下文的充分统计代理：总方差与各滞后区间上的变差函数，都取对数。"""
    lag = np.abs(x[:, None] - x[None, :])
    iu = np.triu_indices(len(x), 1)
    lag, sq = lag[iu], 0.5 * (y[:, None] - y[None, :])[iu] ** 2
    out = [np.log(np.var(y) + 1e-12)]
    for k in np.flatnonzero(mask):
        sel = (lag >= LAG_EDGES[k]) & (lag < LAG_EDGES[k + 1])
        out.append(np.log(sq[sel].mean() + 1e-12))
    return np.array(out)


def lag_mask(x):
    lag = np.abs(x[:, None] - x[None, :])[np.triu_indices(len(x), 1)]
    return np.array([((lag >= LAG_EDGES[k]) & (lag < LAG_EDGES[k + 1])).sum() >= 2
                     for k in range(len(LAG_EDGES) - 1)])


def simulate(x, ell, sigma, v, b, thetas, eps):
    f = rff_sample(x, ell, v, b, thetas.T)  # (n_points, n_sim)
    return f + sigma * eps.T


def estimate_F(x, ell, sigma, rng):
    """只用模拟器采样估证据信息量：F = J^T Sigma^-1 J，J 是统计量均值对 z 的雅可比。

    这是真实 F 的下界（统计量不充分时信息会丢），所以要检验的是它能否复现排序。
    """
    v, b = rff_basis(rng)
    thetas = rng.standard_normal((N_SIM, N_RFF))
    eps = rng.standard_normal((N_SIM, len(x)))
    mask = lag_mask(x)

    def stat_mean(e, s):
        Y = simulate(x, e, s, v, b, thetas, eps)
        S = np.array([summaries(x, Y[:, i], mask) for i in range(N_SIM)])
        return S.mean(0), S

    s0, S0 = stat_mean(ell, sigma)
    Sigma = np.cov(S0.T) + 1e-8 * np.eye(S0.shape[1])
    J = np.empty((S0.shape[1], 2))
    for a, (de, ds) in enumerate([(DELTA, 0.0), (0.0, DELTA)]):
        hi, _ = stat_mean(ell * np.exp(de), sigma * np.exp(ds))
        lo, _ = stat_mean(ell * np.exp(-de), sigma * np.exp(-ds))
        J[:, a] = (hi - lo) / (2 * DELTA)
    return J.T @ np.linalg.solve(Sigma, J)


def net_predict_batch(model, xc, xq, Y):
    n_ctx, n_sim = len(xc), Y.shape[1]
    allx = np.concatenate([xc, xq])
    x = torch.tensor(np.tile(allx, (n_sim, 1)), dtype=torch.float32)
    y = torch.zeros((n_sim, len(allx)), dtype=torch.float32)
    y[:, :n_ctx] = torch.tensor(Y[:n_ctx].T, dtype=torch.float32)
    with torch.no_grad():
        mu, logv = model(x, y, n_ctx)
    return mu.numpy().astype(np.float64), logv.numpy().astype(np.float64)


def estimate_G(model, xc, xq, ell, sigma, rng):
    """用训练好的网络自身的输出对 z 做中心差分，估预测敏感度。

    解析的 G 是训练前的量；这个版本是训练后的诊断。
    两者在 GP 上一致，才有资格把它用在没有闭式的先验上。
    """
    v, b = rff_basis(rng)
    thetas = rng.standard_normal((N_SIM, N_RFF))
    eps = rng.standard_normal((N_SIM, len(xc) + len(xq)))
    allx = np.concatenate([xc, xq])

    def out(e, s):
        Y = simulate(allx, e, s, v, b, thetas, eps)
        return net_predict_batch(model, xc, xq, Y)

    m0, lv0 = out(ell, sigma)
    dm, dlv = [], []
    for de, ds in [(DELTA, 0.0), (0.0, DELTA)]:
        mh, lh = out(ell * np.exp(de), sigma * np.exp(ds))
        ml, ll = out(ell * np.exp(-de), sigma * np.exp(-ds))
        dm.append((mh - ml) / (2 * DELTA))
        dlv.append((lh - ll) / (2 * DELTA))
    s0 = np.exp(lv0)
    G = np.empty((2, 2))
    for a in range(2):
        for b_ in range(2):
            G[a, b_] = np.mean(dm[a] * dm[b_] / s0 + 0.5 * dlv[a] * dlv[b_])
    return G


def run_cell(model, ell, sigma, n_ctx, design, seed=0):
    rng = np.random.default_rng(seed)
    est, ana = [], []
    for _ in range(N_DESIGNS):
        xc = draw_design(rng, n_ctx, design)
        xq = rng.uniform(-X_HALF, X_HALF, N_QUERY)
        Fh = estimate_F(xc, ell, sigma, rng)
        Gh = estimate_G(model, xc, xq, ell, sigma, rng)
        Ah = Fh + np.diag(PRIOR_PREC)
        La = np.linalg.cholesky(Ah)
        W = np.linalg.solve(La, np.linalg.solve(La, Gh).T).T
        est.append((np.linalg.eigvalsh(0.5 * (W + W.T)).max(), np.trace(Fh), np.trace(Gh)))
        vals, _, F, G = conditioning(xc, xq, ell, sigma, PRIOR_PREC)
        ana.append((vals[0], np.trace(F), np.trace(G)))
    est, ana = np.mean(est, axis=0), np.mean(ana, axis=0)
    return {"ell": ell, "sigma": sigma, "n_ctx": n_ctx, "design": design,
            "kappa_hat": float(est[0]), "trF_hat": float(est[1]), "trG_hat": float(est[2]),
            "kappa": float(ana[0]), "trF": float(ana[1]), "trG": float(ana[2])}


if __name__ == "__main__":
    model = PFN()
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    cells = sweep_cells()
    if len(sys.argv) > 1:
        cells = cells[:int(sys.argv[1])]

    rows = []
    for i, (ell, sigma, n_ctx, design) in enumerate(cells):
        rows.append(run_cell(model, ell, sigma, n_ctx, design))
        print(f"    格子 {i + 1}/{len(cells)} 完成", flush=True)

    print(f"\n    {'design':>9}{'n_ctx':>6}{'ell':>8}{'sigma':>7}"
          f"{'kappa 解析':>12}{'kappa 估计':>12}{'tr F 解析':>11}{'tr F 估计':>11}")
    for r in rows:
        print(f"    {r['design']:>9}{r['n_ctx']:>6}{r['ell']:>8.3f}{r['sigma']:>7.2f}"
              f"{r['kappa']:>12.3f}{r['kappa_hat']:>12.3f}{r['trF']:>11.2f}{r['trF_hat']:>11.2f}")

    stats = {}
    for k in ("kappa", "trF", "trG"):
        a = np.array([r[k] for r in rows])
        h = np.array([r[k + "_hat"] for r in rows])
        rho = spearmanr(a, h).statistic
        ratio = float(np.median(h / a))
        stats[k] = {"spearman": float(rho), "median_ratio": ratio}
        print(f"\n    {k}：估计与解析的 Spearman {rho:.3f}，比值中位数 {ratio:.3f}")

    pairs = {}
    for r in rows:
        pairs.setdefault((r["n_ctx"], r["ell"], r["sigma"]), {})[r["design"]] = r
    agree = tot = 0
    for d in pairs.values():
        if len(d) < 2:
            continue
        tot += 1
        agree += int(np.sign(d["paired"]["kappa_hat"] - d["uniform"]["kappa_hat"])
                     == np.sign(d["paired"]["kappa"] - d["uniform"]["kappa"]))
    print(f"\n    设计交叉的符号：估计与解析一致 {agree}/{tot} 格")
    stats["crossover_sign"] = {"n_agree": agree, "n_total": tot}
    OUT_PATH.write_text(json.dumps({"rows": rows, "stats": stats}, ensure_ascii=False, indent=1))
    print(f"\n    结果写到 {OUT_PATH}")
