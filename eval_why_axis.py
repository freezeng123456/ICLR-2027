"""2x2 掰动测试：三个先验不同的 PFN × 两种掰动轴。

anchor 一律远置（与查询区相隔很远，函数相关性实际为零），所以它们对
「查询点的函数值是多少」不含信息，只携带任务层面的全局属性。

  尺度掰动：anchor 分别取自波动函数与平滑函数，观测噪声都很低
           读数 = 查询点**预测均值**的移动幅度
  噪声掰动：anchor 都取自同样尺度的函数，但观测噪声一低一高
           读数 = 查询点**预测标准差**的移动幅度

两种 anchor 在各自的对比里行数、位置、特征分布完全相同，只有被掰的那个属性不同。

每种掰动都同时算出「在该模型自己的先验下，精确贝叶斯应该移动多少」，
用作参照——这样可以区分「模型不该动」和「模型该动却没动」。
"""

import numpy as np
import torch

from exp_why_axis import PFN, PRIORS, mixture_posterior, sample_gp

N_TRIALS = 200
N_Q, N_LOCAL, N_ANCHOR = 10, 5, 20
Q_LO, Q_HI, A_LO, A_HI = 1.2, 3.0, -3.0, -1.2

ELL_WIGGLY, ELL_SMOOTH, ELL_MID = 0.35, 3.0, 1.0
NOISE_LOW, NOISE_HIGH = 0.02, 0.5
ELL_TRUE_LOCAL = 1.0        # 本地上下文所用的真实尺度，居中且稀疏，故本身认不出来
NOISE_TRUE_LOCAL = 0.1


def pfn_out(model, xc, yc, xq):
    x = torch.tensor(np.concatenate([xc, xq], 1), dtype=torch.float32)
    y = torch.tensor(np.concatenate([yc, np.zeros_like(xq)], 1), dtype=torch.float32)
    with torch.no_grad():
        mu, logv = model(x, y, xc.shape[1])
    return mu.numpy(), np.exp(0.5 * logv.numpy())


def build(rng):
    """生成本地上下文与查询点，以及两组 anchor 的横坐标。"""
    xq = np.sort(rng.uniform(Q_LO, Q_HI, N_Q))
    xl = np.sort(rng.uniform(Q_LO, Q_HI, N_LOCAL))
    xa = np.sort(rng.uniform(A_LO, A_HI, N_ANCHOR))
    loc = np.concatenate([xl, xq])
    f = sample_gp(rng, loc, ELL_TRUE_LOCAL) + NOISE_TRUE_LOCAL * rng.standard_normal(len(loc))
    return xl, f[:N_LOCAL], xq, xa


def run_axis(model, ells, noises, axis, rng):
    """比较两种 anchor 造成的逐点变化：PFN 的 vs 精确贝叶斯的。
    返回相关系数、回归斜率，以及两者各自的平均移动幅度。"""
    d_pfn, d_ana = [], []
    for _ in range(N_TRIALS):
        xl, yl, xq, xa = build(rng)
        if axis == "尺度":
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

        k = 0 if axis == "尺度" else 1        # 尺度看均值，噪声看标准差
        d_pfn.append(outs[0][k] - outs[1][k])
        d_ana.append(anas[0][k] - anas[1][k])

    p, a = np.concatenate(d_pfn), np.concatenate(d_ana)
    corr = float(np.corrcoef(a, p)[0, 1]) if a.std() > 1e-12 else 0.0
    slope = float(np.polyfit(a, p, 1)[0]) if a.std() > 1e-12 else 0.0
    return corr, slope, float(np.abs(p).mean()), float(np.abs(a).mean())


def main():
    print("\n" + "=" * 88)
    print("2x2 掰动测试：先验里变化的轴，才掰得动")
    print("=" * 88)
    print(f"""
  {N_TRIALS} 次重复 / {N_LOCAL} 个本地上下文 / {N_ANCHOR} 个远置 anchor / {N_Q} 个查询点。
  两组 anchor 的行数、位置完全相同，只有被掰的那个属性不同。
  「精确贝叶斯」= 在该模型自己的先验下解析算出的应有移动量，用来区分
  「本就不该动」和「该动却没动」。
""")
    print(f"  {'模型':<15}{'先验里变化的':<14}{'尺度掰动(看均值)':>30}{'噪声掰动(看标准差)':>32}")
    print(f"  {'':<29}{'相关':>8}{'斜率':>8}{'PFN幅度':>9}{'相关':>10}{'斜率':>8}{'PFN幅度':>9}")
    print("  " + "-" * 90)

    table = {}
    for name, (ells, noises) in PRIORS.items():
        model = PFN()
        model.load_state_dict(torch.load(f"pfn_{name[0]}.pt", map_location="cpu"))
        model.eval()
        varies = ("尺度" if len(ells) > 1 else "") + \
                 ("+" if len(ells) > 1 and len(noises) > 1 else "") + \
                 ("噪声" if len(noises) > 1 else "")
        row = {}
        for axis in ("尺度", "噪声"):
            rng = np.random.default_rng(2024)      # 三个模型看到完全相同的数据
            row[axis] = run_axis(model, ells, noises, axis, rng)
        table[name] = row
        print(f"  {name:<13}{varies:<16}"
              f"{row['尺度'][0]:>+8.3f}{row['尺度'][1]:>+8.3f}{row['尺度'][2]:>9.4f}"
              f"{row['噪声'][0]:>+10.3f}{row['噪声'][1]:>+8.3f}{row['噪声'][2]:>9.4f}")

    print(f"""
  读法：相关系数衡量 PFN 的逐点变化是否跟着精确贝叶斯走，斜率衡量跟随的完整程度
  （1.0 = 完全跟随）。相关接近 0 说明模型对这个轴没有反应。

  假说预测：
    A(只有尺度变)  尺度轴相关高，噪声轴相关低
    B(只有噪声变)  尺度轴相关低，噪声轴相关高
    C(都变)        两个轴相关都高
""")
    a, b, c = table["A(只有尺度变)"], table["B(只有噪声变)"], table["C(都变)"]
    checks = [
        ("A 跟随尺度强于跟随噪声", a["尺度"][0] > a["噪声"][0]),
        ("B 跟随噪声强于跟随尺度", b["噪声"][0] > b["尺度"][0]),
        ("尺度轴上 A 强于 B", a["尺度"][0] > b["尺度"][0]),
        ("噪声轴上 B 强于 A", b["噪声"][0] > a["噪声"][0]),
        ("C 在两个轴上都不弱于对应单轴模型的弱项",
         c["尺度"][0] > b["尺度"][0] and c["噪声"][0] > a["噪声"][0]),
    ]
    print("  逐条检验：")
    for label, ok in checks:
        print(f"    {'通过' if ok else '未通过'}   {label}")
    n_ok = sum(ok for _, ok in checks)
    print(f"\n  {n_ok}/{len(checks)} 条通过。"
          f"{'假说成立：先验里会变化的轴，才掰得动。' if n_ok >= 4 else '假说未获支持，需要另找解释。'}")


if __name__ == "__main__":
    main()
