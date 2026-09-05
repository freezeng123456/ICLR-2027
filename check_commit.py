import sys

import numpy as np

from exp_conditioning import load_pfn
from exp_jump import X_HALF, draw_design
from eval_jump import pfn_predict, quad_grids
from prior_jump import LEVELS, mixture_posterior, sample_task

# 算力买不到东西的格子集中在「长平台加干净观测」这一角，而且网络在那里比精确后验更自信。
# 猜测的机制是：精确后验要保留「查询点与最近观测之间是否发生过跳变」这份残余怀疑，
# 而网络直接把预测吸附到某一个水平上、把怀疑丢掉。
#
# 判据：预测均值到最近水平的距离。吸附意味着这个距离更小。

N_TASKS = 60


def dist_to_level(mu):
    return np.abs(mu[:, None] - LEVELS[None, :]).min(axis=1)


def run(model, rate, sigma, n_ctx, design, grids, seed=0):
    rng = np.random.default_rng(seed)
    d_net, d_exact, dlogv, kl_mean_part, kl_var_part = [], [], [], [], []
    for _ in range(N_TASKS):
        xc = draw_design(rng, n_ctx, design)
        xq = rng.uniform(-X_HALF, X_HALF, 16)
        y = sample_task(rng, np.concatenate([xc, xq]), rate, sigma)
        yc = y[:n_ctx]
        mu_e, var_e, _ = mixture_posterior(xc, yc, xq, *grids)
        mu_p, var_p = pfn_predict(model, xc, yc, xq)
        d_net.append(dist_to_level(mu_p).mean())
        d_exact.append(dist_to_level(mu_e).mean())
        dlogv.append(np.mean(np.log(var_p) - np.log(var_e)))
        # KL 拆成两项：方差不匹配那一项，与均值误差那一项
        kl_var_part.append(np.mean(0.5 * (np.log(var_p / var_e) + var_e / var_p - 1)))
        kl_mean_part.append(np.mean(0.5 * (mu_e - mu_p) ** 2 / var_p))
    return {"d_net": float(np.mean(d_net)), "d_exact": float(np.mean(d_exact)),
            "dlogvar": float(np.mean(dlogv)),
            "kl_var": float(np.mean(kl_var_part)), "kl_mean": float(np.mean(kl_mean_part))}


if __name__ == "__main__":
    ckpts = sys.argv[1:] or ["pfn_jump_w64.pt", "pfn_jump_40k.pt"]
    grids = quad_grids()
    # 前两格是算力买不到东西的那一角，后两格是正常改善的一角
    cells = [(0.50, 0.05, 16, "uniform"), (0.97, 0.05, 24, "uniform"),
             (25.90, 0.20, 16, "uniform"), (50.00, 0.20, 24, "uniform")]
    print(f"    水平间距 {LEVELS[1] - LEVELS[0]:.2f}\n")
    for ckpt in ckpts:
        model, d_model = load_pfn(ckpt)
        print(f"    {ckpt}（宽度 {d_model}）")
        print(f"    {'rate':>7}{'sigma':>7}{'n':>4}{'均值到最近水平的距离':>22}"
              f"{'方差对数之差':>14}{'KL 方差项':>12}{'KL 均值项':>12}")
        print(f"    {'':>7}{'':>7}{'':>4}{'网络 / 精确后验':>22}")
        for rate, sigma, n_ctx, design in cells:
            r = run(model, rate, sigma, n_ctx, design, grids)
            dists = f"{r['d_net']:.3f} / {r['d_exact']:.3f}"
            print(f"    {rate:>7.2f}{sigma:>7.2f}{n_ctx:>4}{dists:>22}"
                  f"{r['dlogvar']:>+14.3f}{r['kl_var']:>12.4f}{r['kl_mean']:>12.4f}")
        print()
