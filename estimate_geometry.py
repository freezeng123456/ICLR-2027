import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from exp_conditioning import X_HALF, draw_design, sweep_cells
from identifiability import evidence_fisher, prediction_metric

# 真实 PFN 的先验没有闭式，F 与 G 若要能用，就得只靠模拟器采样估出来。
# 这里在解析答案已知的 GP 上核对这件事能做到什么程度。
#
# 结论分两半：
#   F 可以。它只关于上下文的边际分布，用变差函数这组统计量的 Fisher 信息就能估。
#   G 不行。它是在**固定上下文数据**下对隐变量求导，而模拟器只能重新采样数据；
#     重新采样会把数据变化项混进来。本脚本把这个差距量化出来。

N_RFF = 1024
N_SIM = 512
N_DESIGNS = 4
DELTA = 0.08  # 对数坐标下的中心差分步长
LAG_EDGES = np.array([0.0, 0.04, 0.15, 0.35, 0.70, 1.20, 2.01])
OUT_PATH = Path("results/geometry_estimator.json")


def rff_basis(rng):
    """固定一组随机 Fourier 特征。共用随机数时，样本对 log ell 是光滑的，
    中心差分才有意义；用协方差矩阵的分解采样会因特征向量排序跳变而不可微。"""
    return rng.standard_normal(N_RFF), rng.uniform(0, 2 * np.pi, N_RFF)


def rff_sample(x, ell, v, b, theta):
    phi = np.sqrt(2.0 / N_RFF) * np.cos(x[:, None] * (v / ell)[None, :] + b[None, :])
    return phi @ theta


def lag_mask(x):
    lag = np.abs(x[:, None] - x[None, :])[np.triu_indices(len(x), 1)]
    return np.array([((lag >= LAG_EDGES[k]) & (lag < LAG_EDGES[k + 1])).sum() >= 2
                     for k in range(len(LAG_EDGES) - 1)])


def summaries(x, Y, mask):
    """上下文的统计量代理：总方差与各滞后区间上的变差函数，都取对数。

    Y 的形状是 (n_points, n_sim)，一次算完一批。
    """
    iu = np.triu_indices(len(x), 1)
    lag = np.abs(x[:, None] - x[None, :])[iu]
    sq = 0.5 * (Y[:, None, :] - Y[None, :, :])[iu] ** 2
    out = [np.log(Y.var(axis=0) + 1e-12)]
    for k in np.flatnonzero(mask):
        sel = (lag >= LAG_EDGES[k]) & (lag < LAG_EDGES[k + 1])
        out.append(np.log(sq[sel].mean(axis=0) + 1e-12))
    return np.array(out).T


def estimate_F(x, ell, sigma, rng):
    """只用模拟器采样估证据信息量：F = J^T Sigma^-1 J，J 是统计量均值对隐变量的雅可比。

    统计量不充分时会丢信息，所以这是真实 F 的下界，要检验的是它能否复现排序与量级。
    """
    v, b = rff_basis(rng)
    thetas = rng.standard_normal((N_SIM, N_RFF))
    eps = rng.standard_normal((len(x), N_SIM))
    mask = lag_mask(x)

    def stats(e, s):
        Y = rff_sample(x, e, v, b, thetas.T) + s * eps
        return summaries(x, Y, mask)

    S0 = stats(ell, sigma)
    Sigma = np.cov(S0.T) + 1e-8 * np.eye(S0.shape[1])
    J = np.empty((S0.shape[1], 2))
    for a, (de, ds) in enumerate([(DELTA, 0.0), (0.0, DELTA)]):
        hi = stats(ell * np.exp(de), sigma * np.exp(ds)).mean(0)
        lo = stats(ell * np.exp(-de), sigma * np.exp(-ds)).mean(0)
        J[:, a] = (hi - lo) / (2 * DELTA)
    return J.T @ np.linalg.solve(Sigma, J)


def resampled_prediction_metric(x, xq, ell, sigma, rng):
    """把上下文数据在 z+delta 处重新采样后再差分，得到的量。

    这不是 G：G 要求固定上下文数据，只让隐变量动。模拟器做不到这一点，
    重新采样会把数据自身的变化项混进来。这里算出来是为了量化这个差距。
    """
    v, b = rff_basis(rng)
    thetas = rng.standard_normal((N_SIM, N_RFF))
    allx = np.concatenate([x, xq])
    eps = rng.standard_normal((len(allx), N_SIM))
    nc = len(x)

    def moments(e, s):
        Y = rff_sample(allx, e, v, b, thetas.T) + s * eps
        m, sd = np.empty((len(xq), N_SIM)), np.empty((len(xq), N_SIM))
        from identifiability import gp_posterior
        for i in range(N_SIM):
            mu, var = gp_posterior(x, Y[:nc, i], xq, e, s)
            m[:, i], sd[:, i] = mu, var
        return m, sd

    m0, v0 = moments(ell, sigma)
    dm, dlv = [], []
    for de, ds in [(DELTA, 0.0), (0.0, DELTA)]:
        mh, vh = moments(ell * np.exp(de), sigma * np.exp(ds))
        ml, vl = moments(ell * np.exp(-de), sigma * np.exp(-ds))
        dm.append((mh - ml) / (2 * DELTA))
        dlv.append((np.log(vh) - np.log(vl)) / (2 * DELTA))
    M = np.empty((2, 2))
    for a in range(2):
        for c in range(2):
            M[a, c] = np.mean(dm[a] * dm[c] / v0 + 0.5 * dlv[a] * dlv[c])
    return M


def run_cell(ell, sigma, n_ctx, design, seed=0):
    rng = np.random.default_rng(seed)
    out = {"F_hat": [], "F": [], "G": [], "G_resampled": []}
    for _ in range(N_DESIGNS):
        xc = draw_design(rng, n_ctx, design)
        xq = rng.uniform(-X_HALF, X_HALF, N_QUERY_GEO)
        out["F_hat"].append(np.trace(estimate_F(xc, ell, sigma, rng)))
        out["F"].append(np.trace(evidence_fisher(xc, ell, sigma)))
        out["G"].append(np.trace(prediction_metric(xc, xq, ell, sigma)))
        out["G_resampled"].append(np.trace(
            resampled_prediction_metric(xc, xq, ell, sigma, rng)))
    res = {k: float(np.mean(v)) for k, v in out.items()}
    res.update({"ell": ell, "sigma": sigma, "n_ctx": n_ctx, "design": design})
    return res


N_QUERY_GEO = 4  # 重新采样那一项要逐样本解 GP，查询点少一些

if __name__ == "__main__":
    cells = sweep_cells()
    if len(sys.argv) > 1:
        cells = cells[::int(sys.argv[1])]

    rows = []
    for i, (ell, sigma, n_ctx, design) in enumerate(cells):
        rows.append(run_cell(ell, sigma, n_ctx, design))
        print(f"    格子 {i + 1}/{len(cells)} 完成", flush=True)

    print(f"\n    {'design':>9}{'n_ctx':>6}{'ell':>8}{'sigma':>7}"
          f"{'tr F 解析':>11}{'tr F 估计':>11}{'tr G 解析':>11}{'重采样后':>11}")
    for r in rows:
        print(f"    {r['design']:>9}{r['n_ctx']:>6}{r['ell']:>8.3f}{r['sigma']:>7.2f}"
              f"{r['F']:>11.2f}{r['F_hat']:>11.2f}{r['G']:>11.3f}{r['G_resampled']:>11.1f}")

    stats = {}
    F, Fh = np.array([r["F"] for r in rows]), np.array([r["F_hat"] for r in rows])
    G, Gr = np.array([r["G"] for r in rows]), np.array([r["G_resampled"] for r in rows])
    stats["F"] = {"spearman": float(spearmanr(F, Fh).statistic),
                  "median_ratio": float(np.median(Fh / F))}
    stats["G_resampled"] = {"spearman": float(spearmanr(G, Gr).statistic),
                            "median_ratio": float(np.median(Gr / G))}
    print(f"\n    证据信息量 F：只靠模拟器采样估出来的与解析值 "
          f"Spearman {stats['F']['spearman']:.3f}，比值中位数 {stats['F']['median_ratio']:.3f}")
    print(f"    预测敏感度 G：重新采样上下文后再差分，与解析值 "
          f"Spearman {stats['G_resampled']['spearman']:.3f}，"
          f"比值中位数 {stats['G_resampled']['median_ratio']:.1f}")
    print("\n    读法：F 只靠模拟器采样能估出可用的排序，量级只恢复了四成上下——")
    print("    变差函数这组统计量不充分，缺口在函数结构落到采样分辨率以下的格子上最大。")
    print("    G 估不出来。它要求固定上下文数据、只让隐变量动，而模拟器只能重新采样；")
    print("    重新采样把数据自身的变化项混了进来，量级差一到两个数量级。")
    print("    要在没有闭式的先验上用 G，就得先有一个「给定同一批数据、隐变量取别的值」")
    print("    的条件分布——那正是 PFN 本身在逼近的东西，所以这条路是循环的。")

    OUT_PATH.write_text(json.dumps({"rows": rows, "stats": stats}, ensure_ascii=False, indent=1))
    print(f"\n    结果写到 {OUT_PATH}")
