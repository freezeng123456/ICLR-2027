import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# 判别两种解释：
#   更新不足     隐含隐变量被压向先验均值，与真值在先验均值的哪一侧无关，两侧都小于 1
#   方差膨胀假象 网络的方差要覆盖自身均值误差，于是隐含的隐变量一律朝一个方向被推；
#                真值在先验均值一侧时看起来像收缩，另一侧看起来像反向扩张
#
# 先验均值：log ell 是 log(0.2)，log sigma 是 log(0.1)。每个坐标要按自己那一维分组。

PRIOR_MEAN = {"ell": 0.2, "sigma": 0.1}
COORD_KEY = {0: "ell", 1: "sigma"}


def load(path):
    return json.loads(Path(path).read_text())["rows"]


def signed_beta(rows, coord):
    """按坐标算收缩系数，并按该坐标的真值落在先验均值的哪一侧分组。"""
    key = COORD_KEY[coord]
    mid = PRIOR_MEAN[key]
    out = {}
    for side, sel in (("真值在先验均值下方", lambda r: r[key] < mid),
                      ("真值在先验均值上方", lambda r: r[key] > mid)):
        sub = [r for r in rows if sel(r)]
        zn = np.vstack([np.array(r["z_net"]) for r in sub])[:, coord]
        ze = np.vstack([np.array(r["z_exact"]) for r in sub])[:, coord]
        out[side] = (float((ze @ zn) / (ze @ ze)), len(sub), float(ze.mean()))
    return out


def main(path):
    rows = load(path)
    print(f"    {path}，{len(rows)} 个格子\n")

    for coord, name in ((0, "log ell"), (1, "log sigma")):
        res = signed_beta(rows, coord)
        print(f"    {name} 的收缩系数（按该坐标自己那一维分组）")
        for k, (b, n, mean_ze) in res.items():
            print(f"      {k}：{b:.3f}（{n} 个格子，精确后验相对先验均值的偏移均值 {mean_ze:+.3f}）")
        print()

    print("    两种解释的判据：更新不足要求两侧都小于 1；")
    print("    方差膨胀假象要求下方那一侧小于 1、上方那一侧大于 1。\n")

    # 两种参数化的拟合残差，看每个坐标上哪一种成立
    zn = np.vstack([np.array(r["z_net"]) for r in rows])
    ze = np.vstack([np.array(r["z_exact"]) for r in rows])
    print(f"    {'坐标':<10}{'乘性 beta':>11}{'乘性残差':>11}"
          f"{'加性偏移':>11}{'加性残差':>11}{'哪一种更好':>13}")
    for coord, name in ((0, "log ell"), (1, "log sigma")):
        a, b = ze[:, coord], zn[:, coord]
        beta = float((a @ b) / (a @ a))
        res_mul = float(np.std(b - beta * a))
        delta = float(np.mean(b - a))
        res_add = float(np.std(b - a - delta))
        better = "乘性收缩" if res_mul < res_add else "加性偏移"
        print(f"    {name:<10}{beta:>11.3f}{res_mul:>11.4f}"
              f"{delta:>+11.3f}{res_add:>11.4f}{better:>13}")
    print()

    # 网络的方差是否恰好覆盖了它自己的均值误差
    dv = np.array([r["dlogvar"] for r in rows])
    ms = np.array([r["mean_slope"] for r in rows])
    gap = np.array([r["gap"] for r in rows])
    print(f"    方差对数之差与预测均值斜率的 Spearman：{spearmanr(dv, ms).statistic:+.3f}")
    print(f"    方差对数之差与差距的 Spearman：{spearmanr(dv, gap).statistic:+.3f}")

    # 逐格子收缩系数与各个变量的关系
    betas = np.array([float((np.array(r["z_exact"]) * np.array(r["z_net"])).sum()
                            / (np.array(r["z_exact"]) ** 2).sum()) for r in rows])
    for key in ("n_ctx", "ell", "sigma", "trF", "trG"):
        v = np.array([r[key] for r in rows])
        print(f"    beta 与 {key} 的 Spearman：{spearmanr(v, betas).statistic:+.3f}")

    for design in ("uniform", "paired"):
        b = np.array([b for b, r in zip(betas, rows) if r["design"] == design])
        print(f"    {design} 设计的 beta 中位数：{np.median(b):.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/conditioning_pfn_cond.json")
