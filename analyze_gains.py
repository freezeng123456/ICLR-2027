import json
import sys
from pathlib import Path

import numpy as np

# 正向侧的两条：容量与预算换来的实测降幅，以及上下文设计这条零成本规则。

SETTINGS = [("宽 64 @20k", "results/conditioning_pfn_cond_w64.json"),
            ("宽 64 @40k", "results/conditioning_pfn_cond_w64_40k.json"),
            ("宽 128 @20k", "results/conditioning_pfn_cond.json"),
            ("宽 128 @40k", "results/conditioning_pfn_cond_40k.json"),
            ("宽 128 @20k 加权", "results/conditioning_pfn_cond_rw.json"),
            ("宽 128 等算力蒸馏", "results/conditioning_pfn_cond_distill.json")]

# 参数量与步数，用来算算力标度。加权与蒸馏是干预对照，不进标度拟合。
COMPUTE = {"宽 64 @20k": (0.30, 20000), "宽 64 @40k": (0.30, 40000),
           "宽 128 @20k": (1.19, 20000), "宽 128 @40k": (1.19, 40000)}


def load(path):
    p = Path(path)
    return json.loads(p.read_text())["rows"] if p.exists() else None


def scaling_table():
    print(f"    {'设置':<20}{'超额 KL 均值':>14}{'最差格子':>11}{'中位数':>11}"
          f"{'n=8':>10}{'n=16':>10}{'n=24':>10}")
    base = None
    out = {}
    for name, path in SETTINGS:
        rows = load(path)
        if rows is None:
            continue
        g = np.array([r["gap"] for r in rows])
        by = [float(np.mean([r["gap"] for r in rows if r["n_ctx"] == n])) for n in (8, 16, 24)]
        out[name] = {"mean": float(g.mean()), "max": float(g.max()),
                     "median": float(np.median(g)), "by_n": by}
        if base is None:
            base = out[name]
        print(f"    {name:<20}{g.mean():>14.4f}{g.max():>11.4f}{np.median(g):>11.4f}"
              + "".join(f"{v:>10.4f}" for v in by))

    print(f"\n    相对第一行的比值")
    for name, v in out.items():
        print(f"    {name:<20}{v['mean'] / base['mean']:>14.3f}{v['max'] / base['max']:>11.3f}"
              f"{v['median'] / base['median']:>11.3f}"
              + "".join(f"{a / b:>10.3f}" for a, b in zip(v["by_n"], base["by_n"])))
    return out


def design_rule():
    """同样的点数、同样的先验，只改上下文点的位置。这是零成本的选用规则。"""
    print(f"\n    上下文设计的选用规则（成对紧邻相对均匀铺开的超额 KL 比值）")
    print(f"    {'设置':<20}{'n=8':>10}{'n=16':>10}{'n=24':>10}"
          f"{'成对更好的格子':>16}{'小 ell 一侧':>14}")
    out = {}
    for name, path in SETTINGS:
        rows = load(path)
        if rows is None:
            continue
        pairs = {}
        for r in rows:
            pairs.setdefault((r["n_ctx"], r["ell"], r["sigma"]), {})[r["design"]] = r["gap"]
        ratios_by_n, better, small_better, small_tot = [], 0, 0, 0
        for (n, ell, _), d in pairs.items():
            if len(d) < 2:
                continue
            better += int(d["paired"] < d["uniform"])
            if ell <= 0.1:
                small_tot += 1
                small_better += int(d["paired"] < d["uniform"])
        for n in (8, 16, 24):
            vals = [d["paired"] / d["uniform"] for (nn, _, _), d in pairs.items()
                    if nn == n and len(d) == 2]
            ratios_by_n.append(float(np.median(vals)))
        tot = sum(1 for d in pairs.values() if len(d) == 2)
        out[name] = {"ratio_by_n": ratios_by_n, "n_better": better, "n_total": tot,
                     "small_ell_better": small_better, "small_ell_total": small_tot}
        print(f"    {name:<20}" + "".join(f"{v:>10.3f}" for v in ratios_by_n)
              + f"{f'{better}/{tot}':>16}{f'{small_better}/{small_tot}':>14}")
    return out


def design_total_error():
    """设计要能用，改善的必须是总误差。

    总误差 = 贝叶斯 NLL（不可约那一部分）+ 超额 KL（网络多付的那一部分）。
    上下文点摆在哪里会同时动这两项：成对紧邻能把长度尺度与噪声分开，
    但占据的不同位置只有一半，覆盖更粗。
    """
    from scipy.stats import ttest_rel
    print(f"\n    总误差（贝叶斯 NLL + 超额 KL）与配对显著性")
    print(f"    {'设置':<20}{'n':>4}{'均匀总误差':>13}{'成对总误差':>13}"
          f"{'之差':>10}{'配对 t':>9}{'超额 KL 之差':>14}{'t':>9}")
    out = {}
    for name, path in SETTINGS:
        rows = load(path)
        if rows is None:
            continue
        pairs = {}
        for r in rows:
            pairs.setdefault((r["n_ctx"], r["ell"], r["sigma"]), {})[r["design"]] = r
        for n in (8, 16, 24):
            sel = [(d["uniform"], d["paired"]) for (nn, _, _), d in pairs.items()
                   if nn == n and len(d) == 2]
            tu = np.array([u["bayes_nll"] + u["gap"] for u, _ in sel])
            tp = np.array([p["bayes_nll"] + p["gap"] for _, p in sel])
            gu = np.array([u["gap"] for u, _ in sel])
            gp = np.array([p["gap"] for _, p in sel])
            t_tot = ttest_rel(tp, tu).statistic
            t_gap = ttest_rel(gp, gu).statistic
            out[f"{name} n={n}"] = {"total_uniform": float(tu.mean()),
                                    "total_paired": float(tp.mean()),
                                    "t_total": float(t_tot), "t_gap": float(t_gap),
                                    "gap_delta": float((gp - gu).mean())}
            print(f"    {name:<20}{n:>4}{tu.mean():>13.4f}{tp.mean():>13.4f}"
                  f"{tp.mean() - tu.mean():>+10.4f}{t_tot:>+9.2f}"
                  f"{(gp - gu).mean():>+14.4f}{t_gap:>+9.2f}")
    print("\n    负号表示成对更好。")
    return out


def scaling_law(scal):
    """超额 KL 随算力的标度。

    算力取参数量乘步数。这里能算这条曲线，是因为参照物是精确的贝叶斯最优解，
    而不是另一个近似方法——通常这个量是测不出来的。
    """
    pts = [(p * s, scal[k]["mean"], scal[k]["max"], k)
           for k, (p, s) in COMPUTE.items() if k in scal]
    if len(pts) < 3:
        print("\n    标度拟合需要至少三个点")
        return {}
    pts.sort()
    c = np.log([p[0] for p in pts])
    print(f"\n    算力标度（算力 = 参数量 x 步数）")
    print(f"    {'设置':<16}{'相对算力':>10}{'超额 KL 均值':>14}{'最差格子':>11}")
    for comp, mean, mx, k in pts:
        print(f"    {k:<16}{comp / pts[0][0]:>10.2f}{mean:>14.4f}{mx:>11.4f}")
    out = {}
    for label, idx in (("均值", 1), ("最差格子", 2)):
        y = np.log([p[idx] for p in pts])
        A = np.vstack([c, np.ones(len(c))]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - ((y - A @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        alpha = -coef[0]
        need = 2 ** (1 / alpha)
        target = 0.01
        extra = (pts[-1][idx] / target) ** (1 / alpha)
        out[label] = {"alpha": float(alpha), "r2": float(r2), "compute_to_halve": float(need),
                      "compute_to_reach_0.01": float(extra)}
        print(f"    {label}：超额 KL 正比于 算力^(-{alpha:.3f})，R^2 = {r2:.3f}，"
              f"要减半需要 {need:.1f} 倍算力，"
              f"要降到 {target} nat 需要在最大那一档之上再加 {extra:.0f} 倍")

    # 两个干预落在曲线之上还是之下
    print(f"\n    干预相对标度律的位置（算力按等价的参数量乘步数算）")
    y = np.log([p[1] for p in pts])
    coef = np.polyfit(c, y, 1)
    for name, comp in (("宽 128 @20k 加权", 1.19 * 20000), ("宽 128 等算力蒸馏", 1.19 * 20000)):
        if name not in scal:
            continue
        pred = float(np.exp(np.polyval(coef, np.log(comp))))
        got = scal[name]["mean"]
        out[name] = {"predicted": pred, "measured": got, "ratio": got / pred}
        print(f"    {name:<18}标度律预测 {pred:.4f}，实测 {got:.4f}，"
              f"比值 {got / pred:.3f}")
    return out


if __name__ == "__main__":
    scal = scaling_table()
    law = scaling_law(scal)
    des = design_rule()
    tot = design_total_error()
    Path("results/gains.json").write_text(
        json.dumps({"scaling": scal, "scaling_law": law, "design": des, "design_total": tot},
                   ensure_ascii=False, indent=1))
    print(f"\n    结果写到 results/gains.json")
