"""把「anchor 掰动先验」画成一张图：同一批本地上下文、同一批查询点，
只改变远处 anchor 的内容，看 PFN 在查询区的预测曲线如何整体移动。

左半边是 anchor 区（离查询区很远，函数相关性实际为零），右半边是查询区。
"""

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False

from exp_gp_prior_tilt import ELL_GRID_P, NOISE, PFN, mixture_posterior, sample_gp

Q_LO, Q_HI, A_LO, A_HI = 1.2, 3.0, -3.0, -1.2
N_LOCAL, N_ANCHOR, ELL_TRUE = 5, 20, 0.35
SEED = 7


def pfn_pred(model, xc, yc, xq):
    x = torch.tensor(np.concatenate([xc, xq])[None], dtype=torch.float32)
    y = torch.tensor(np.concatenate([yc, np.zeros_like(xq)])[None], dtype=torch.float32)
    with torch.no_grad():
        mu, _ = model(x, y, len(xc))
    return mu[0].numpy()


def main():
    model = PFN()
    model.load_state_dict(torch.load("pfn_gp.pt", map_location="cpu"))
    model.eval()
    rng = np.random.default_rng(SEED)

    xq = np.linspace(Q_LO, Q_HI, 60)
    xl = np.sort(rng.uniform(Q_LO, Q_HI, N_LOCAL))
    xa = np.sort(rng.uniform(A_LO, A_HI, N_ANCHOR))
    allx = np.concatenate([xl, xq, xa])
    f = sample_gp(rng, allx, ELL_TRUE) + NOISE * rng.standard_normal(len(allx))
    yl, ya_true = f[:N_LOCAL], f[N_LOCAL + len(xq):]
    ya_fake = sample_gp(rng, xa, ELL_GRID_P[-1])  # 人造平滑假点

    settings = [
        ("不加 anchor", np.array([]), np.array([]), "tab:gray", "-"),
        ("加真实观测 anchor（波动）", xa, ya_true, "tab:red", "-"),
        ("加人造平滑 anchor（假点）", xa, ya_fake, "tab:blue", "-"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), height_ratios=[1, 1.25])

    ax = axes[0]
    ax.scatter(xa, ya_true, s=26, c="tab:red", marker="o", label="真实观测 anchor（波动剧烈）")
    ax.scatter(xa, ya_fake, s=26, c="tab:blue", marker="s", label="人造平滑 anchor（构造的假点）")
    ax.scatter(xl, yl, s=90, c="k", marker="*", zorder=5, label="本地上下文（唯一与查询区相关的数据）")
    ax.axvspan(Q_LO, Q_HI, color="k", alpha=0.05)
    ax.text((Q_LO + Q_HI) / 2, ax.get_ylim()[1] * 0.8, "查询区", ha="center", fontsize=9)
    ax.set_xlim(-3.2, 3.2)
    ax.set_title("上：两套 anchor 都放在远处，对查询区的函数值不含任何信息", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    for label, xa_i, ya_i, color, ls in settings:
        xc = np.concatenate([xl, xa_i])
        yc = np.concatenate([yl, ya_i])
        p = pfn_pred(model, xc, yc, xq)
        a, _, w = mixture_posterior(xc, yc, xq, ELL_GRID_P)
        ell = np.exp(w @ np.log(ELL_GRID_P))
        ax.plot(xq, p, color=color, ls=ls, lw=2,
                label=f"PFN：{label}（先验隐含尺度 {ell:.2f}）")
        ax.plot(xq, a, color=color, ls=":", lw=1.5, alpha=0.75)
    ax.scatter(xl, yl, s=110, c="k", marker="*", zorder=5, label="本地上下文（三种情况完全相同）")
    ax.set_xlim(Q_LO, Q_HI)
    ax.set_title("下：查询区的预测。实线 = PFN，虚线 = 解析解。"
                 "本地数据一模一样，只因远处 anchor 不同，曲线整体移动", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("x")

    for a_ in axes:
        a_.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("prior_tilt.png", dpi=140)
    print("已保存 prior_tilt.png")


if __name__ == "__main__":
    main()
