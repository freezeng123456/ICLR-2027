"""分开两个混淆因素：C 跟随精确贝叶斯更差，是因为「先验组合数多」还是
「两个隐变量互相纠缠」？

C 有 12 个尺度 x 12 个噪声 = 144 种组合。这里再训一个模型 A144：
沿单一维度铺 144 个长度尺度，噪声固定。两者组合数相同、训练步数相同，
唯一区别是 A144 只有一个隐变量，C 有两个且互相混淆
（「平滑但吵」与「波动但干净」会产生相似的观测）。

  若 A144 明显强于 C  -> 差距来自隐变量纠缠，而非组合数
  若两者相当          -> 差距只是任务多样性带来的训练难度，与纠缠无关
"""

import sys

import numpy as np
import torch

from eval_why_axis import run_axis
from exp_why_axis import PFN, PRIORS, train

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
ELLS_144 = np.exp(np.linspace(np.log(0.3), np.log(3.0), 144))
NOISE_FIXED = np.array([0.1])

if __name__ == "__main__":
    if "--eval-only" not in sys.argv:
        print(f"  训练 A144：144 个长度尺度、噪声固定、{STEPS} 步", flush=True)
        train("Z", ELLS_144, NOISE_FIXED, STEPS)   # 存为 pfn_Z.pt

    model = PFN()
    model.load_state_dict(torch.load("pfn_Z.pt", map_location="cpu"))
    model.eval()
    out = {}
    for axis in ("尺度", "噪声"):
        rng = np.random.default_rng(2024)
        out[axis] = run_axis(model, ELLS_144, NOISE_FIXED, axis, rng)

    print(f"""
    组合数与步数都相同（144 种组合，{STEPS} 步）的两个模型：

    {'模型':<26}{'尺度轴相关':>12}{'尺度轴斜率':>12}{'噪声轴相关':>14}{'噪声轴斜率':>12}
    {'A144（单隐变量，144 尺度）':<20}{out['尺度'][0]:>+12.3f}{out['尺度'][1]:>+12.3f}"""
          f"{out['噪声'][0]:>+14.3f}{out['噪声'][1]:>+12.3f}")
    print(f"    {'C（双隐变量，12x12）':<22}{'+0.636':>12}{'+0.347':>12}"
          f"{'+0.519':>14}{'+0.357':>12}   (先前记录)")
    print(f"""
    参照：A（单隐变量，仅 12 种组合，15000 步）尺度轴 +0.863 / +0.683

    判定：A144 若明显强于 C，说明差距来自两个隐变量的纠缠而非组合数；
          若与 C 相当，说明只是任务多样性带来的训练难度。""")
