"""GP 侧的决定性对照：anchor 掰动 vs 一个标量的方差重标定。

表格侧只能比最终指标，这里有解析真值，可以问一个更尖锐的问题：

    anchor 造成的改变，能不能被「把所有预测的方差乘同一个常数」复现？

温度缩放（以及概率收缩）本质上就是一个**全局、与查询点无关**的单调变换：
所有点被同等地拉宽或收紧，点与点之间的相对关系一点不变，argmax 也不变。

所以只要回答两件事，就能判定 anchor 到底有没有独立价值：

  Q1  anchor 引起的方差变化，是不是各点乘同一个常数？
      拟合 sigma_with = c * sigma_without，看 R^2。
      R^2 ≈ 1 -> 就是均匀缩放，温度缩放能完全替代，anchor 没有独立价值。
      R^2 < 1 -> 有逐点结构，温度缩放做不到。

  Q2  anchor 有没有改变预测**均值**？
      温度缩放在数学上不可能改变均值（分类里则是不可能改变 argmax）。
      任何均值上的变化，都是后处理校准原理上做不到的事。

两个模型都测：PFN 本身，以及它自己先验下的解析后验（机制的「理想版本」）。
"""

import numpy as np
import torch

from eval_why_axis import (A_HI, A_LO, ELL_MID, N_ANCHOR, NOISE_HIGH, NOISE_LOW, build,
                           pfn_out)
from exp_why_axis import PFN, PRIORS, mixture_posterior, sample_gp

np.set_printoptions(precision=4, suppress=True, linewidth=110)
torch.set_num_threads(2)

N_TRIALS = 300


def collect(model, ells, noises, rng):
    """返回加 anchor 前后的 (均值, 标准差)，PFN 与解析解各一份。"""
    out = {k: {"mu0": [], "sd0": [], "mu1": [], "sd1": []} for k in ("PFN", "解析后验")}
    for _ in range(N_TRIALS):
        xl, yl, xq, xa = build(rng)
        base = sample_gp(rng, xa, ELL_MID)
        ya_clean = base + NOISE_LOW * rng.standard_normal(N_ANCHOR)
        ya_noisy = base + NOISE_HIGH * rng.standard_normal(N_ANCHOR)

        for tag, ya in (("0", ya_clean), ("1", ya_noisy)):
            xc = np.concatenate([xl, xa])[None]
            yc = np.concatenate([yl, ya])[None]
            mu, sd = pfn_out(model, xc, yc, xq[None])
            out["PFN"][f"mu{tag}"].append(mu[0])
            out["PFN"][f"sd{tag}"].append(sd[0])
            m, v = mixture_posterior(xc[0], yc[0], xq, ells, noises)
            out["解析后验"][f"mu{tag}"].append(m)
            out["解析后验"][f"sd{tag}"].append(np.sqrt(v))
    return {k: {kk: np.concatenate(vv) for kk, vv in d.items()} for k, d in out.items()}


def analyse(name, d):
    mu0, sd0, mu1, sd1 = d["mu0"], d["sd0"], d["mu1"], d["sd1"]

    # Q1：如果只是均匀缩放，逐点的 sigma_with / sigma_without 应当是同一个常数。
    #     所以直接看这个比值的离散程度，比拟合 R^2 更直白（也避开过原点拟合的口径问题）。
    ratio = sd1 / np.maximum(sd0, 1e-9)
    cv = float(ratio.std() / ratio.mean())                   # 变异系数，0 = 完全均匀
    q10, q90 = np.percentile(ratio, [10, 90])
    corr_sd = float(np.corrcoef(sd0, sd1)[0, 1])             # 均匀缩放时应为 1.0
    c = float(np.sum(sd0 * sd1) / np.sum(sd0 ** 2))
    rel_struct = float(np.abs(sd1 - c * sd0).mean() / np.abs(sd1 - sd0).mean())

    # Q2：均值变了多少（相对于它自身的尺度）
    dmu = mu1 - mu0
    mu_shift = float(np.abs(dmu).mean())
    mu_scale = float(np.abs(mu0).mean())

    print(f"\n  ── {name} ──")
    print(f"    加 anchor 前 平均标准差 {sd0.mean():.4f}   加噪声 anchor 后 {sd1.mean():.4f}"
          f"   （平均放大 {sd1.mean()/sd0.mean():.2f} 倍）")
    print(f"\n    Q1 这个变化能被单个常数复现吗？（均匀缩放时逐点比值应处处相同）")
    print(f"       逐点比值 sigma_with/sigma_without：中位数 {np.median(ratio):.2f}，"
          f"10–90 分位 [{q10:.2f}, {q90:.2f}]")
    print(f"       比值的变异系数 CV = {cv:.3f}      （0 = 完全均匀）")
    print(f"       sigma_without 与 sigma_with 的相关 = {corr_sd:+.3f}   （1.0 = 完全均匀）")
    print(f"       扣掉最优均匀缩放后，残留的逐点结构占总变化的 {rel_struct*100:.0f}%")
    verdict1 = ("基本就是均匀缩放，温度缩放可以完全替代" if cv < 0.15 and corr_sd > 0.95
                else "**不是均匀缩放**，存在温度缩放原理上无法复现的逐点结构")
    print(f"       -> {verdict1}")
    print(f"\n    Q2 均值动了吗？（温度缩放在数学上不可能改变均值）")
    print(f"       平均 |Δmu| = {mu_shift:.4f}，相当于 |mu| 本身的 {mu_shift/mu_scale*100:.0f}%")
    verdict2 = "**均值有实质变化**，这是后处理校准原理上做不到的" if mu_shift / mu_scale > 0.05 \
        else "均值几乎不动，anchor 的作用基本只在不确定度上"
    print(f"       -> {verdict2}")
    return dict(cv=cv, corr=corr_sd, rel_struct=rel_struct,
                mu_shift=mu_shift, mu_scale=mu_scale)


def main():
    print("\n" + "=" * 86)
    print("GP 侧：anchor 掰动 vs 单标量方差重标定")
    print("=" * 86)
    print(f"""
  {N_TRIALS} 次重复。布局与 2x2 掰动实验相同：查询点与稀疏本地上下文在 x∈[1.2,3.0]，
  {N_ANCHOR} 个 anchor 全部远置在 x∈[{A_LO},{A_HI}]，两种 anchor 底层函数完全相同，
  只有观测噪声不同（{NOISE_LOW} vs {NOISE_HIGH}）。""")

    for mname in ("B(只有噪声变)", "C(都变)"):
        ells, noises = PRIORS[mname]
        model = PFN()
        ckpt = f"pfn_{mname[0]}.pt"
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
        print(f"\n{'='*86}\n模型 {mname}（{ckpt}）\n{'='*86}")
        d = collect(model, ells, noises, np.random.default_rng(2024))
        for k in ("PFN", "解析后验"):
            analyse(k, d[k])

    print(f"""

{'='*86}
怎么读这两组结果
{'='*86}

  「解析后验」那一行是机制的**理想版本**——精确贝叶斯在同样的 anchor 下会怎么动。
  「PFN」那一行是模型实际做到的。

  如果连理想版本都只是均匀缩放，那 anchor 这条路在原理上就替代不了温度缩放，
  论文的方法半边只能靠「不需要保留集」这一个卖点，很薄。

  如果理想版本有明显的逐点结构、而 PFN 也跟上了，那 anchor 做的是温度缩放
  原理上做不到的事，方法半边就立得住——此时该把论文的主张从「改善校准」
  收紧成「做出与查询点相关的、结构化的不确定度调整」。
""")


if __name__ == "__main__":
    main()
