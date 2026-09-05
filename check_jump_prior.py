import numpy as np

from prior_jump import K_LEVELS, LEVELS, mixture_posterior, predict_single, sample_task, transition

# 精确推断必须先被校验，后面所有测量都压在它上面。
# 校验办法是穷举：把问题缩小到 8 个点，枚举全部 5^8 条水平序列，逐条算先验与似然。
# 这给出的是精确答案，不是估计——按先验重要性采样在这个模型上会退化，
# 因为先验采样几乎从不命中观测。

N_CTX, N_QUERY = 6, 2
RATE_GRID = np.exp(np.linspace(np.log(0.5), np.log(50.0), 4))
SIG_GRID = np.exp(np.linspace(np.log(0.02), np.log(0.5), 3))


def enumerate_posterior(xs, y_sorted, rate, sigma):
    """枚举全部水平序列，返回每个位置上水平的精确后验与对数边际似然。"""
    n = len(xs)
    seqs = np.array(np.unravel_index(np.arange(K_LEVELS ** n), (K_LEVELS,) * n)).T
    logp = np.full(len(seqs), -np.log(K_LEVELS))
    for i in range(n - 1):
        logT = np.log(transition(xs[i + 1] - xs[i], rate))
        logp += logT[seqs[:, i], seqs[:, i + 1]]
    obs = ~np.isnan(y_sorted)
    for i in np.flatnonzero(obs):
        d = y_sorted[i] - LEVELS[seqs[:, i]]
        logp += -0.5 * (d / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
    m = logp.max()
    w = np.exp(logp - m)
    logZ = m + np.log(w.sum())
    w /= w.sum()
    gamma = np.zeros((n, K_LEVELS))
    for k in range(K_LEVELS):
        gamma[:, k] = (w[:, None] * (seqs == k)).sum(0)
    return gamma, logZ


def sorted_view(xc, yc, xq):
    x_all = np.concatenate([xc, xq])
    y_all = np.concatenate([yc, np.full(len(xq), np.nan)])
    order = np.argsort(x_all)
    return x_all[order], y_all[order], order


def main():
    rng = np.random.default_rng(0)
    xc = rng.uniform(-1, 1, N_CTX)
    xq = rng.uniform(-1, 1, N_QUERY)
    rate_true, sig_true = 6.0, 0.15
    y = sample_task(rng, np.concatenate([xc, xq]), rate_true, sig_true)
    yc = y[:N_CTX]
    xs, ys, order = sorted_view(xc, yc, xq)
    q_positions = np.flatnonzero(order >= N_CTX)

    print(f"  枚举 {K_LEVELS}^{len(xs)} = {K_LEVELS ** len(xs)} 条水平序列\n")

    print("  校验一：固定 (rate, sigma) 下查询点的水平后验与对数边际似然")
    g_fb, logZ_fb = predict_single(xc, yc, xq, rate_true, sig_true)
    g_en, logZ_en = enumerate_posterior(xs, ys, rate_true, sig_true)
    g_en_q = g_en[q_positions][np.argsort(order[q_positions] - N_CTX)]
    err_g = np.abs(g_fb - g_en_q).max()
    err_z = abs(logZ_fb - logZ_en)
    print(f"    水平后验的最大绝对差 {err_g:.2e}，对数边际似然的绝对差 {err_z:.2e}")
    for j in range(N_QUERY):
        print(f"    查询点 {j}：前向后向 {np.array2string(g_fb[j], precision=4)}")
        print(f"              穷举     {np.array2string(g_en_q[j], precision=4)}")
    assert err_g < 1e-9 and err_z < 1e-9, "前向后向与穷举不一致"

    print("\n  校验二：网格求积后的预测均值与方差")
    mu, var, _ = mixture_posterior(xc, yc, xq, RATE_GRID, SIG_GRID)
    logZs, gammas, sigs = [], [], []
    for r in RATE_GRID:
        for s in SIG_GRID:
            g, lz = enumerate_posterior(xs, ys, r, s)
            gq = g[q_positions][np.argsort(order[q_positions] - N_CTX)]
            gammas.append(gq); logZs.append(lz); sigs.append(s)
    logZs = np.array(logZs)
    w = np.exp(logZs - logZs.max()); w /= w.sum()
    gammas, sig2 = np.array(gammas), np.array(sigs)[:, None] ** 2
    mu_en = np.einsum("g,gqk,k->q", w, gammas, LEVELS)
    var_en = np.einsum("g,gqk,gk->q", w, gammas, sig2 + LEVELS[None, :] ** 2) - mu_en ** 2
    print(f"    {'查询点':>6}{'求积均值':>12}{'穷举均值':>12}{'求积方差':>12}{'穷举方差':>12}")
    for j in range(N_QUERY):
        print(f"    {j:>6}{mu[j]:>12.6f}{mu_en[j]:>12.6f}{var[j]:>12.6f}{var_en[j]:>12.6f}")
    print(f"    均值最大绝对差 {np.abs(mu - mu_en).max():.2e}，"
          f"方差最大绝对差 {np.abs(var - var_en).max():.2e}")
    assert np.abs(mu - mu_en).max() < 1e-9, "求积与穷举不一致"

    print("\n  校验三：小噪声下发射概率不下溢")
    # 隐变量投影会在真实噪声 0.2 的数据上试探 sigma = 0.02，
    # 此时观测可能离最近水平 1.0 以上，直接取指数会全部变成零
    rng3 = np.random.default_rng(11)
    xc3 = rng3.uniform(-1, 1, 24)
    xq3 = rng3.uniform(-1, 1, 8)
    y3 = sample_task(rng3, np.concatenate([xc3, xq3]), 6.0, 0.4)
    for s in (0.02, 0.05, 0.2):
        g3, lz3 = predict_single(xc3, y3[:24], xq3, 6.0, s)
        worst = np.abs(y3[:24, None] - LEVELS[None, :]).min(axis=1).max()
        assert np.isfinite(g3).all() and np.isfinite(lz3), f"sigma={s} 出现 NaN"
        print(f"    sigma = {s:.2f}：观测离最近水平最远 {worst:.2f}，"
              f"权重和 {g3.sum(axis=1).min():.6f}，log Z = {lz3:.1f}")

    print("\n  预测分布的多峰性（与高斯过程的关键区别）")
    rng2 = np.random.default_rng(7)
    heavy = 0
    for _ in range(200):
        xc2 = rng2.uniform(-1, 1, 24)
        xq2 = rng2.uniform(-1, 1, 8)
        y2 = sample_task(rng2, np.concatenate([xc2, xq2]), 6.0, 0.15)
        g, _ = predict_single(xc2, y2[:24], xq2, 6.0, 0.15)
        second = np.sort(g, axis=1)[:, -2]
        heavy += int((second > 0.15).sum())
    print(f"    随机 200 个任务 x 8 个查询点里，第二大分量权重超过 0.15 的有 {heavy} 个"
          f"（{heavy / 16:.1f}%）")


if __name__ == "__main__":
    main()
