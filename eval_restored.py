"""对补训/备份的检查点做与原论文表同一套轴测试。"""

import hashlib
import os

import numpy as np
import torch

from eval_why_axis import run_axis
from exp_entangle import ELLS_144, NOISE_FIXED
from exp_why_axis import PFN, PRIORS


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eval_ckpt(path, ells, noises, tag):
    if not os.path.exists(path):
        print(f"  skip {tag}: {path} missing")
        return None
    model = PFN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    out = {}
    for axis in ("尺度", "噪声"):
        rng = np.random.default_rng(2024)
        out[axis] = run_axis(model, ells, noises, axis, rng)
    print(f"  {tag:<22} {path:<16} md5={md5(path)[:8]}  "
          f"尺度 {out['尺度'][0]:+.3f}/{out['尺度'][1]:+.3f}  "
          f"噪声 {out['噪声'][0]:+.3f}/{out['噪声'][1]:+.3f}")
    return out


if __name__ == "__main__":
    print("\n  补训检查点对照（相关/斜率；种子 2024，与原表相同）\n")
    ells_c, noises_c = PRIORS["C(都变)"]
    eval_ckpt("pfn_A.pt", *PRIORS["A(只有尺度变)"], "A 12ℓ")
    eval_ckpt("pfn_B.pt", *PRIORS["B(只有噪声变)"], "B 12σ")
    eval_ckpt("pfn_C.pt", ells_c, noises_c, "C 12x12 @当前pfn_C")
    eval_ckpt("pfn_C_15k.pt", ells_c, noises_c, "C 12x12 @15k")
    eval_ckpt("pfn_C_45k.pt", ells_c, noises_c, "C 12x12 @45k")
    eval_ckpt("pfn_Z.pt", ELLS_144, NOISE_FIXED, "A144 单隐变量")
