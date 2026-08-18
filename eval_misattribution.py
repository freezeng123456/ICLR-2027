"""直接检验「错误归因」：同一个操纵，在先验不同的模型上应留下不同的签名。

2(操纵) x 2(读数) x 3(模型) 设计：

  操纵：尺度（anchor 波动 vs 平滑）、噪声（anchor 干净 vs 吵）
  读数：查询点预测均值的变化、查询点预测标准差的变化

如果模型只是「对缺失的轴没有反应」，那么：
  A（无噪声隐变量）在噪声操纵下两个读数都该接近不动
如果模型是「把操纵错误归因到自己仅有的隐变量」，那么：
  A 在噪声操纵下会把它当成尺度变化 -> 均值也被明显带动
  B 在尺度操纵下会把它当成噪声变化 -> 标准差也被明显带动

每个格子都同时给出「该模型自己先验下精确贝叶斯的相应变化」。
若 PFN 在所有格子上都跟住了自己的解析参照，就说明它始终是自己先验下忠实的
贝叶斯推断者——错的不是推断，是世界观。
"""

import numpy as np
import torch

from eval_why_axis import (A_HI, A_LO, ELL_MID, ELL_SMOOTH, ELL_WIGGLY, N_ANCHOR,
                           NOISE_HIGH, NOISE_LOW, build, pfn_out)
from exp_why_axis import PFN, PRIORS, mixture_posterior, sample_gp

N_TRIALS = 200


def both_readouts(model, ells, noises, manip, rng):
    """返回 {读数: (相关, 斜率, PFN 幅度, 解析幅度)}。"""
    acc = {"均值": ([], []), "标准差": ([], [])}
    for _ in range(N_TRIALS):
        xl, yl, xq, xa = build(rng)
        if manip == "尺度":
            ya1 = sample_gp(rng, xa, ELL_WIGGLY) + NOISE_LOW * rng.standard_normal(N_ANCHOR)
            ya2 = sample_gp(rng, xa, ELL_SMOOTH) + NOISE_LOW * rng.standard_normal(N_ANCHOR)
        else:
            base = sample_gp(rng, xa, ELL_MID)
            ya1 = base + NOISE_LOW * rng.standard_normal(N_ANCHOR)
            ya2 = base + NOISE_HIGH * rng.standard_normal(N_ANCHOR)

        outs, anas = [], []
        for ya in (ya1, ya2):
            xc = np.concatenate([xl, xa])[None]
            yc = np.concatenate([yl, ya])[None]
            mu, sd = pfn_out(model, xc, yc, xq[None])
            outs.append((mu[0], sd[0]))
            m, v = mixture_posterior(xc[0], yc[0], xq, ells, noises)
            anas.append((m, np.sqrt(v)))
        for k, key in ((0, "均值"), (1, "标准差")):
            acc[key][0].append(outs[0][k] - outs[1][k])
            acc[key][1].append(anas[0][k] - anas[1][k])

    out = {}
    for key, (p_list, a_list) in acc.items():
        p, a = np.concatenate(p_list), np.concatenate(a_list)
        corr = float(np.corrcoef(a, p)[0, 1]) if a.std() > 1e-12 else 0.0
        slope = float(np.polyfit(a, p, 1)[0]) if a.std() > 1e-12 else 0.0
        out[key] = (corr, slope, float(np.abs(p).mean()), float(np.abs(a).mean()))
    return out


def main():
    print("\n" + "=" * 92)
    print("错误归因检验：同一个操纵在不同先验的模型上留下不同的签名")
    print("=" * 92)
    print(f"""
  {N_TRIALS} 次重复。每个操纵都同时看两个读数：预测均值的变化、预测标准差的变化。
  括号里是该模型自己先验下精确贝叶斯的相应变化幅度。
""")
    for manip in ("尺度", "噪声"):
        print(f"  --- 操纵：{manip}（anchor 的{'波动程度' if manip == '尺度' else '观测噪声'}） ---")
        print(f"    {'模型':<15}{'读数=均值':>30}{'读数=标准差':>32}")
        print(f"    {'':<13}{'相关':>9}{'斜率':>8}{'PFN(贝叶斯)':>16}"
              f"{'相关':>10}{'斜率':>8}{'PFN(贝叶斯)':>16}")
        for name, (ells, noises) in PRIORS.items():
            model = PFN()
            model.load_state_dict(torch.load(f"pfn_{name[0]}.pt", map_location="cpu"))
            model.eval()
            rng = np.random.default_rng(2024)
            r = both_readouts(model, ells, noises, manip, rng)
            m, s = r["均值"], r["标准差"]
            print(f"    {name:<13}{m[0]:>+9.3f}{m[1]:>+8.3f}"
                  f"{f'{m[2]:.3f}({m[3]:.3f})':>16}"
                  f"{s[0]:>+10.3f}{s[1]:>+8.3f}{f'{s[2]:.3f}({s[3]:.3f})':>16}")
        print()

    print("""  读法：
    若「缺失的轴 = 无反应」，A 在噪声操纵下两个读数都该几乎不动。
    若「缺失的轴 = 错误归因」，A 会把噪声当成尺度变化，从而在噪声操纵下
    连均值一起带动；B 则会把波动当成噪声，在尺度操纵下把标准差带动起来。
    而无论哪种情况，只要 PFN 在每个格子上都跟住了自己的解析参照，
    就说明它始终是自己先验下忠实的贝叶斯推断者。""")


if __name__ == "__main__":
    main()
