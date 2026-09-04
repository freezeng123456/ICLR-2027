import numpy as np

# 隐变量取对数坐标 z = (log ell, log sigma)，两个方向都是无量纲的，才能放进同一个广义特征问题。

SIGNAL_VAR = 1.0


def rbf(a, b, ell):
    return np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell ** 2)


def d_rbf_dlogell(a, b, ell):
    r2 = (a[:, None] - b[None, :]) ** 2
    return rbf(a, b, ell) * r2 / ell ** 2


def kernel_and_grads(xc, ell, sigma):
    """返回 K 与它对 (log ell, log sigma) 的一阶导数。"""
    K = SIGNAL_VAR * rbf(xc, xc, ell) + sigma ** 2 * np.eye(len(xc))
    dK_dlogell = SIGNAL_VAR * d_rbf_dlogell(xc, xc, ell)
    dK_dlogsigma = 2.0 * sigma ** 2 * np.eye(len(xc))
    return K, [dK_dlogell, dK_dlogsigma]


def evidence_fisher(xc, ell, sigma):
    """上下文对隐变量携带的证据信息量。

    y ~ N(0, K_z) 的 Fisher 信息矩阵 F_ab = 0.5 * tr(K^-1 dK_a K^-1 dK_b)。
    单位是「每单位 z 的平方所对应的证据 nat 数」。
    """
    K, dKs = kernel_and_grads(xc, ell, sigma)
    Kinv = np.linalg.inv(K)
    M = [Kinv @ dK for dK in dKs]
    F = np.empty((2, 2))
    for a in range(2):
        for b in range(2):
            F[a, b] = 0.5 * np.trace(M[a] @ M[b])
    return F


def prediction_metric(xc, xq, ell, sigma):
    """查询点预测分布对隐变量的 Fisher-Rao 度量，对一批查询点取平均。

    单个查询点的预测分布是 N(m_z, s_z)。把 z 挪动 dz 会让它移动
    KL = 0.5 * dz^T G dz，其中对上下文标签 y ~ N(0, K_z) 取了期望。
    对查询点取平均，是为了和「在同一批查询点上平均的预测差距」量纲一致。
    单位是「每单位 z 的平方所对应的预测 nat 数」。
    """
    xq = np.atleast_1d(xq)
    K, dKs = kernel_and_grads(xc, ell, sigma)
    Kinv = np.linalg.inv(K)
    Cq = SIGNAL_VAR * rbf(xc, xq, ell)
    dCq = SIGNAL_VAR * d_rbf_dlogell(xc, xq, ell)

    G = np.zeros((2, 2))
    for j in range(len(xq)):
        c, Ac = Cq[:, j], Kinv @ Cq[:, j]
        dc = [dCq[:, j], np.zeros(len(xc))]
        s = SIGNAL_VAR + sigma ** 2 - c @ Ac
        # m = c^T K^-1 y 对 z 的导数写成 u_a^T y，于是 E[d_a m d_b m] = u_a^T K u_b
        u = [Kinv @ dc[a] - Kinv @ (dKs[a] @ Ac) for a in range(2)]
        ds = [-2.0 * dc[a] @ Ac + Ac @ (dKs[a] @ Ac) for a in range(2)]
        ds[1] += 2.0 * sigma ** 2  # 预测方差里含观测噪声项
        for a in range(2):
            for b in range(2):
                G[a, b] += u[a] @ K @ u[b] / s + 0.5 * ds[a] * ds[b] / s ** 2
    return G / len(xq)


def log_uniform_precision(lo, hi):
    """log-uniform 先验在对数坐标下的精度，取其方差的倒数。"""
    return 12.0 / (np.log(hi) - np.log(lo)) ** 2


def conditioning(xc, xq, ell, sigma, prior_precision):
    """摊销条件数：预测最敏感、后验最不约束的那个方向上的比值。

    网络要表示的是后验，所以分母是后验精度 A = F + Lambda 而不是单独的 F。
    数据不约束的方向由先验免费约束住，那里精确贝叶斯与网络都退回先验，不产生误差。

    解广义特征问题 G v = lambda A v。lambda 的单位是「预测 nat / 后验 nat」：
    每一个没被解析出来的后验信息 nat 要在预测上付出多少代价。
    """
    F = evidence_fisher(xc, ell, sigma)
    G = prediction_metric(xc, xq, ell, sigma)
    A = F + np.diag(prior_precision)
    La = np.linalg.cholesky(A)
    W = np.linalg.solve(La, np.linalg.solve(La, G).T).T  # A^-1 G 的对称化形式
    vals, vecs = np.linalg.eigh(0.5 * (W + W.T))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    # 特征向量要从白化坐标转回 z 坐标
    vecs = np.linalg.solve(La.T, vecs[:, order])
    vecs /= np.linalg.norm(vecs, axis=0, keepdims=True)
    return vals, vecs, F, G


def log_marginal(xc, yc, ell, sigma):
    K = SIGNAL_VAR * rbf(xc, xc, ell) + sigma ** 2 * np.eye(len(xc))
    L = np.linalg.cholesky(K)
    a = np.linalg.solve(L, yc)
    return -0.5 * a @ a - np.log(np.diag(L)).sum() - 0.5 * len(xc) * np.log(2 * np.pi)


def gp_posterior(xc, yc, xq, ell, sigma):
    K = SIGNAL_VAR * rbf(xc, xc, ell) + sigma ** 2 * np.eye(len(xc))
    Ks = SIGNAL_VAR * rbf(xc, xq, ell)
    mean = Ks.T @ np.linalg.solve(K, yc)
    var = SIGNAL_VAR + sigma ** 2 - np.sum(Ks * np.linalg.solve(K, Ks), axis=0)
    return mean, np.maximum(var, 1e-8)


def mixture_posterior(xc, yc, xq, ell_grid, sigma_grid):
    """先验下的精确最优预测：在 (ell, sigma) 网格上做求积。

    网格是连续 log-uniform 先验的求积近似，密度由调用方通过网格大小控制。
    """
    grid = [(e, s) for e in ell_grid for s in sigma_grid]
    lml = np.array([log_marginal(xc, yc, e, s) for e, s in grid])
    w = np.exp(lml - lml.max())
    w /= w.sum()
    ms, vs = zip(*[gp_posterior(xc, yc, xq, e, s) for e, s in grid])
    ms, vs = np.array(ms), np.array(vs)
    mu = w @ ms
    var = w @ (vs + ms ** 2) - mu ** 2
    return mu, np.maximum(var, 1e-8), w


def gauss_kl(mu_p, var_p, mu_q, var_q):
    """KL(P || Q)，两个都是对角高斯。P 取精确后验，Q 取网络输出。"""
    return 0.5 * (np.log(var_q / var_p) + (var_p + (mu_p - mu_q) ** 2) / var_q - 1.0)
