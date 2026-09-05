import numpy as np

# 非高斯过程的先验：函数在 K 个离散水平之间按泊松速率跳变，观测加高斯噪声。
#
# 两个连续隐变量与高斯过程那一套一一对应：
#   rate  跳变速率，结构那一维（速率越高函数变化越快，对应长度尺度的倒数）
#   sigma 观测噪声，噪声那一维
#
# 与高斯过程的关键区别：给定数据，查询点的预测分布是 K 个分量的高斯混合，
# 多峰、非高斯。所以「网络只能输出均值与方差」这件事在这里是真的有损，
# 方差膨胀那条机制会受到更强的检验。
#
# 精确后验用前向后向算法算，再在 (rate, sigma) 网格上求积。

K_LEVELS = 5
LEVELS = np.linspace(-1.6, 1.6, K_LEVELS)


def transition(gap, rate):
    """跨过一段间距后水平的转移矩阵。

    间距内没有跳变的概率是 exp(-rate*gap)，此时水平不变；
    一旦发生跳变，水平就从 K 个里重新均匀抽一个（可能抽回原值）。
    """
    a = np.exp(-rate * gap)
    b = (1.0 - a) / K_LEVELS
    T = np.full((K_LEVELS, K_LEVELS), b)
    T[np.diag_indices(K_LEVELS)] += a
    return T


def sample_task(rng, x, rate, sigma):
    """按排序后的位置逐段生成水平，返回带噪观测。"""
    order = np.argsort(x)
    xs = x[order]
    z = np.empty(len(xs), dtype=int)
    z[0] = rng.integers(K_LEVELS)
    for i in range(1, len(xs)):
        if rng.random() < np.exp(-rate * (xs[i] - xs[i - 1])):
            z[i] = z[i - 1]
        else:
            z[i] = rng.integers(K_LEVELS)
    f = np.empty(len(x))
    f[order] = LEVELS[z]
    return f + sigma * rng.standard_normal(len(x))


def _emissions(y_obs, sigma):
    """发射概率，逐位置把最大值因子提出来。

    直接取指数会在小噪声下全部下溢到零：sigma = 0.02 而观测离最近水平 1.0 时，
    exp(-1250) 就是 0，归一化随即产生 NaN。提出因子后每个位置至少有一项等于 1，
    因子的对数累加进 log Z，结果仍然精确。

    查询位置没有观测，发射概率取 1。
    """
    n = len(y_obs)
    e = np.ones((n, K_LEVELS))
    obs = ~np.isnan(y_obs)
    d = y_obs[obs, None] - LEVELS[None, :]
    loge = -0.5 * (d / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
    m = loge.max(axis=1, keepdims=True)
    e[obs] = np.exp(loge - m)
    return e, float(m.sum())


def forward_backward(xs, y_obs, rate, sigma):
    """返回每个位置上水平的后验，以及数据的对数边际似然。

    xs 必须已排序。y_obs 在查询位置取 nan。
    每步做归一化，把归一化常数累加成 log Z，避免下溢。
    """
    n = len(xs)
    e, log_scale = _emissions(y_obs, sigma)
    Ts = [transition(xs[i + 1] - xs[i], rate) for i in range(n - 1)]

    alpha = np.empty((n, K_LEVELS))
    a = e[0] / K_LEVELS
    logZ = log_scale + np.log(a.sum())
    alpha[0] = a / a.sum()
    for i in range(1, n):
        a = (alpha[i - 1] @ Ts[i - 1]) * e[i]
        s = a.sum()
        logZ += np.log(s)
        alpha[i] = a / s

    beta = np.ones((n, K_LEVELS))
    for i in range(n - 2, -1, -1):
        b = Ts[i] @ (e[i + 1] * beta[i + 1])
        beta[i] = b / b.sum()

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma, logZ


def predict_single(xc, yc, xq, rate, sigma):
    """给定一组 (rate, sigma) 的精确预测分布，返回混合权重与分量参数。

    分量是 K 个 N(LEVELS[k], sigma^2)，权重是查询位置上水平的后验。
    """
    x_all = np.concatenate([xc, xq])
    y_all = np.concatenate([yc, np.full(len(xq), np.nan)])
    order = np.argsort(x_all)
    gamma, logZ = forward_backward(x_all[order], y_all[order], rate, sigma)
    # 把排序后的位置映射回查询点原来的顺序
    pos = np.empty(len(x_all), dtype=int)
    pos[order] = np.arange(len(x_all))
    return gamma[pos[len(xc):]], logZ


def mixture_posterior(xc, yc, xq, rate_grid, sig_grid):
    """先验下的精确最优预测：在 (rate, sigma) 网格上求积，返回矩匹配的均值与方差。

    网络只输出均值与方差，训练目标是高斯 NLL，所以它能达到的最优就是这里的矩匹配高斯。
    """
    grid = [(r, s) for r in rate_grid for s in sig_grid]
    gammas, logZs, sigs = [], [], []
    for r, s in grid:
        g, lz = predict_single(xc, yc, xq, r, s)
        gammas.append(g)
        logZs.append(lz)
        sigs.append(s)
    logZs = np.array(logZs)
    w = np.exp(logZs - logZs.max())
    w /= w.sum()

    gammas = np.array(gammas)                      # (格点, 查询点, K)
    sig2 = np.array(sigs)[:, None] ** 2            # (格点, 1)
    mu = np.einsum("g,gqk,k->q", w, gammas, LEVELS)
    second = np.einsum("g,gqk,gk->q", w, gammas,
                       sig2 + LEVELS[None, :] ** 2)
    return mu, np.maximum(second - mu ** 2, 1e-8), w
