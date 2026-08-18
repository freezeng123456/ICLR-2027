"""排除混淆：模型 C 跟随精确贝叶斯更差，是因为先验丰富导致推断本身难，
还是仅仅因为同样步数下它收敛得更少？

C 的先验有 12x12=144 种组合，A 和 B 各只有 12 种，所以同为 15000 步时
C 见到每种组合的次数少得多。这里把 C 训到 45000 步（每种组合的曝光量
反超 A/B 在 15000 步时的水平），再测一次。

  若相关系数回升到 A/B 的水平 -> 之前的差距只是训练不足，该观察作废
  若仍显著偏低              -> 支持「先验越丰富，摊销推断偏离精确贝叶斯越多」
"""

import sys

import numpy as np
import torch

from exp_why_axis import PRIORS, train
from eval_why_axis import run_axis
from exp_why_axis import PFN

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 45000

if __name__ == "__main__":
    ells, noises = PRIORS["C(都变)"]
    print(f"  把 C 训到 {STEPS} 步（原为 15000）", flush=True)
    train("C_long", ells, noises, STEPS)

    print("\n  重测 C 的两个轴：\n")
    print(f"    {'检查点':<16}{'尺度轴相关':>12}{'尺度轴斜率':>12}"
          f"{'噪声轴相关':>14}{'噪声轴斜率':>12}")
    for tag, path in [("C 训 15000 步", "pfn_C.pt"), (f"C 训 {STEPS} 步", "pfn_C_long.pt")]:
        model = PFN()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        out = {}
        for axis in ("尺度", "噪声"):
            rng = np.random.default_rng(2024)
            out[axis] = run_axis(model, ells, noises, axis, rng)
        print(f"    {tag:<14}{out['尺度'][0]:>+12.3f}{out['尺度'][1]:>+12.3f}"
              f"{out['噪声'][0]:>+14.3f}{out['噪声'][1]:>+12.3f}")

    print("""
    参照（15000 步时的单轴模型）：
      A  尺度轴 +0.863 / 斜率 +0.683      噪声轴 +0.899 / 斜率 +0.918
      B  尺度轴 +0.861 / 斜率 +0.681      噪声轴 +0.835 / 斜率 +0.741""")
