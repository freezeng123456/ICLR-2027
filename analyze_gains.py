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


JUMP = {"jump w64 @20k": ("results/jump_pfn_jump_w64.json", 0.30 * 20000),
        "jump w64 @40k": ("results/jump_pfn_jump_w64_40k.json", 0.30 * 40000),
        "jump w128 @20k": ("results/jump_pfn_jump.json", 1.19 * 20000),
        "jump w128 @40k": ("results/jump_pfn_jump_40k.json", 1.19 * 40000)}


def jump_scaling_law():
    """标度指数是否跨先验稳定。跳变过程先验上重复同一个 2x2 网格。"""
    pts = []
    for k, (p, c) in JUMP.items():
        if not Path(p).exists():
            continue
        g = np.array([x["gap"] for x in load(p)])
        pts.append((c, float(g.mean()), float(g.max()), k))
    if len(pts) < 3:
        print("\n    跳变先验的标度拟合需要至少三个点")
        return {}
    pts.sort()
    print(f"\n    跳变过程先验上的算力标度")
    print(f"    {'设置':<18}{'相对算力':>10}{'超额 KL 均值':>14}{'最差格子':>11}")
    for comp, mean, mx, k in pts:
        print(f"    {k:<18}{comp / pts[0][0]:>10.2f}{mean:>14.4f}{mx:>11.4f}")
    c = np.log([p[0] for p in pts])
    out = {}
    for label, idx in (("均值", 1), ("最差格子", 2)):
        y = np.log([p[idx] for p in pts])
        A = np.vstack([c, np.ones(len(c))]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - ((y - A @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        alpha = -coef[0]
        out[label] = {"alpha": float(alpha), "r2": float(r2),
                      "compute_to_halve": float(2 ** (1 / alpha))}
        print(f"    {label}：超额 KL 正比于 算力^(-{alpha:.3f})，R^2 = {r2:.3f}，"
              f"要减半需要 {2 ** (1 / alpha):.1f} 倍算力")

    # 算力在哪些格子上买不到东西：检验是否由预测分布的多峰性造成
    from scipy.stats import spearmanr
    lo = {(x["rate"], x["sigma"], x["n_ctx"], x["design"]): x
          for x in load(JUMP["jump w64 @20k"][0])}
    hi = {(x["rate"], x["sigma"], x["n_ctx"], x["design"]): x
          for x in load(JUMP["jump w128 @40k"][0])}
    keys = [k for k in lo if k in hi]
    ratio = np.array([hi[k]["gap"] / lo[k]["gap"] for k in keys])
    second = np.array([hi[k]["second_weight"] for k in keys])
    rate = np.array([k[0] for k in keys])
    print(f"\n    8 倍算力带来的逐格子改善（比值越小越好）")
    print(f"    中位数 {np.median(ratio):.3f}，四分位区间 "
          f"{np.percentile(ratio, 25):.3f} – {np.percentile(ratio, 75):.3f}，"
          f"变好的格子 {int((ratio < 1).sum())}/{len(ratio)}")
    print(f"    改善比值与次峰权重的 Spearman {spearmanr(second, ratio).statistic:+.3f}"
          f"（负号表示预测分布越多峰，算力买到的越多）")
    print(f"    改善比值与跳变速率的 Spearman {spearmanr(rate, ratio).statistic:+.3f}")
    # 真正的解释：后验越尖锐，KL 里的均值误差被 1/方差放得越大，算力越买不到东西
    bnll = np.array([hi[k]["bayes_nll"] for k in keys])
    print(f"    改善比值与贝叶斯 NLL 的 Spearman {spearmanr(bnll, ratio).statistic:+.3f}"
          f"（负号表示后验越尖锐、算力越买不到东西）")
    q = np.percentile(bnll, [0, 33, 67, 100])
    print(f"\n    {'贝叶斯 NLL 区间':<20}{'格子数':>8}{'改善比值中位数':>16}{'次峰权重中位数':>16}")
    for a, b in zip(q[:-1], q[1:]):
        m = (bnll >= a) & (bnll <= b)
        if m.sum():
            print(f"    {f'[{a:+.2f}, {b:+.2f}]':<20}{int(m.sum()):>8}"
                  f"{np.median(ratio[m]):>16.3f}{np.median(second[m]):>16.3f}")
    # 更精确的解释：跳变先验的最优预测要做一个离散判断（查询点落在哪个平台），
    # 判错就付一整个水平间距，误差因此重尾。重尾的格子靠算力压不下去。
    n_tasks = 40
    cv = np.array([hi[k]["gap_se"] * np.sqrt(n_tasks) / hi[k]["gap"] for k in keys])
    print(f"\n    逐任务误差的重尾程度（标准差比均值）：中位数 {np.median(cv):.2f}")
    print(f"    与改善比值的 Spearman {spearmanr(cv, ratio).statistic:+.3f}"
          f"（正号表示越重尾、算力越买不到东西）")

    # 同一关系在高斯过程先验上是否也成立
    glo = {(x["ell"], x["sigma"], x["n_ctx"], x["design"]): x
           for x in load("results/conditioning_pfn_cond_w64.json")}
    ghi = {(x["ell"], x["sigma"], x["n_ctx"], x["design"]): x
           for x in load("results/conditioning_pfn_cond_40k.json")}
    gk = [k for k in glo if k in ghi]
    gratio = np.array([ghi[k]["gap"] / glo[k]["gap"] for k in gk])
    gbnll = np.array([ghi[k]["bayes_nll"] for k in gk])
    print(f"\n    同一关系在高斯过程先验上：改善比值与贝叶斯 NLL 的 Spearman "
          f"{spearmanr(gbnll, gratio).statistic:+.3f}，"
          f"改善比值中位数 {np.median(gratio):.3f}")
    print(f"    两个先验的贝叶斯 NLL 范围：跳变 {bnll.min():+.2f} – {bnll.max():+.2f}，"
          f"高斯过程 {gbnll.min():+.2f} – {gbnll.max():+.2f}")
    gcv = np.array([ghi[k]["gap_se"] * np.sqrt(n_tasks) / ghi[k]["gap"] for k in gk])
    print(f"    逐任务误差的重尾程度：跳变中位数 {np.median(cv):.2f}，"
          f"高斯过程中位数 {np.median(gcv):.2f}")

    out["insensitive_cells"] = {
        "spearman_second_vs_ratio": float(spearmanr(second, ratio).statistic),
        "spearman_bayes_nll_vs_ratio": float(spearmanr(bnll, ratio).statistic),
        "gp_spearman_bayes_nll_vs_ratio": float(spearmanr(gbnll, gratio).statistic),
        "ratio_median": float(np.median(ratio)),
        "gp_ratio_median": float(np.median(gratio)),
        "n_improved": int((ratio < 1).sum()), "n_cells": len(ratio)}
    return out


if __name__ == "__main__":
    scal = scaling_table()
    law = scaling_law(scal)
    jump = jump_scaling_law()
    des = design_rule()
    tot = design_total_error()
    Path("results/gains.json").write_text(
        json.dumps({"scaling": scal, "scaling_law": law, "jump_scaling_law": jump,
                    "design": des, "design_total": tot}, ensure_ascii=False, indent=1))
    print(f"\n    结果写到 results/gains.json")
