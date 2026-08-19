"""补训被覆盖或从未提交的检查点。

  C_15k   12x12 先验、15000 步 -> pfn_C_15k.pt，成功后复制为 pfn_C.pt
  Z       A144：144 个长度尺度、噪声固定、15000 步 -> pfn_Z.pt
  gp      最早的 GP-PFN（仅长度尺度网格）15000 步 -> pfn_gp.pt

当前仓库里的 pfn_C.pt 是 45000 步的权重，已备份为 pfn_C_45k.pt。
不要在未备份的情况下重跑 exp_why_axis.py --force。
"""

import hashlib
import shutil
import sys

import numpy as np
import torch

from exp_gp_prior_tilt import train as train_gp_pfn
from exp_why_axis import PRIORS, train


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_c_15k():
    ells, noises = PRIORS["C(都变)"]
    print("  训练 C @ 15000 步 -> pfn_C_15k.pt", flush=True)
    train("C_15k", ells, noises, 15000, save_path="pfn_C_15k.pt")
    shutil.copy2("pfn_C_15k.pt", "pfn_C.pt")
    print(f"  已复制为 pfn_C.pt  md5={md5('pfn_C_15k.pt')}", flush=True)


def train_z():
    ells = np.exp(np.linspace(np.log(0.3), np.log(3.0), 144))
    noises = np.array([0.1])
    print("  训练 A144 / Z @ 15000 步 -> pfn_Z.pt", flush=True)
    train("Z", ells, noises, 15000, save_path="pfn_Z.pt")
    print(f"  pfn_Z.pt md5={md5('pfn_Z.pt')}", flush=True)


def train_gp():
    print("  训练 GP-PFN @ 15000 步 -> pfn_gp.pt", flush=True)
    train_gp_pfn(15000)
    print(f"  pfn_gp.pt md5={md5('pfn_gp.pt')}", flush=True)


JOBS = {
    "C_15k": train_c_15k,
    "Z": train_z,
    "gp": train_gp,
}


if __name__ == "__main__":
    want = sys.argv[1:] or ["all"]
    if want == ["all"]:
        want = list(JOBS)
    for name in want:
        if name not in JOBS:
            raise SystemExit(f"unknown job {name}, choose from {list(JOBS)} / all")
        JOBS[name]()
    print("  requested jobs done")
