import json
import sys
from pathlib import Path

import numpy as np
import torch

from exp_conditioning import load_pfn
from identifiability import gauss_kl

# 已测出：算力单调地让网络更自信。这在网络原本不够自信的地方是改进，
# 在原本已经过度自信的地方是变坏，而 KL 对过度自信的惩罚很重。
#
# 于是有一个不含任何可调参数的做法：**取大模型的均值、小模型的方差**。
# 大模型的均值确实更准（KL 的均值项随算力下降），小模型的方差没那么激进。
#
# 判据分两层：
#   选哪个变体，只能看先验采样上的留出 NLL（部署时唯一能看到的东西）
#   报告时同时给出相对精确贝叶斯的超额 KL，两者都列出来

N_EVAL_TASKS = 40
N_FIT_TASKS = 400


def load_prior(kind):
    """两个先验共用同一套评估流程，只换掉采样与精确后验。"""
    if kind == "gp":
        import exp_conditioning as E
        from eval_conditioning import quad_grids
        from identifiability import mixture_posterior
        return {"draw_design": E.draw_design, "draw_latent": E.draw_latent,
                "sample": lambda rng, x, a, b: E.sample_gp(rng, x, a) + b * rng.standard_normal(len(x)),
                "cells": E.sweep_cells(), "grids": quad_grids(),
                "posterior": mixture_posterior, "n_query": E.N_QUERY, "x_half": E.X_HALF,
                "keys": ("ell", "sigma")}
    import exp_jump as J
    from eval_jump import quad_grids as jq
    from prior_jump import mixture_posterior as jmp
    return {"draw_design": J.draw_design, "draw_latent": J.draw_latent,
            "sample": J.sample_task, "cells": J.sweep_cells(), "grids": jq(),
            "posterior": jmp, "n_query": J.N_QUERY, "x_half": J.X_HALF,
            "keys": ("rate", "sigma")}


def pfn_predict(model, xc, yc, xq):
    x = torch.tensor(np.concatenate([xc, xq])[None], dtype=torch.float32)
    y = torch.tensor(np.concatenate([yc, np.zeros_like(xq)])[None], dtype=torch.float32)
    with torch.no_grad():
        mu, logv = model(x, y, len(xc))
    return mu[0].numpy().astype(np.float64), np.exp(logv[0].numpy().astype(np.float64))


NOMINAL = 0.90
Z_NOMINAL = 1.6448536269514722  # 标准正态的 95% 分位点


def variants(small, big, threshold=None):
    """几种组合，除覆盖率门限外都不含可调参数。

    覆盖率门限那一支只在网络自报方差偏小的地方把方差抬起来。
    门限由先验采样上的覆盖率定出，不需要精确后验。
    """
    (ms, vs), (mb, vb) = small, big
    out = {"small": (ms, vs), "big": (mb, vb),
           "big_mean_small_var": (mb, vs),
           "big_mean_max_var": (mb, np.maximum(vs, vb)),
           "big_mean_geo_var": (mb, np.sqrt(vs * vb))}
    if threshold is not None:
        # threshold 是覆盖率不足的那些方差区间，只在这些区间里抬方差
        inside = np.zeros_like(vb, dtype=bool)
        for lo, hi in threshold:
            inside |= (vb >= lo) & (vb <= hi)
        out["coverage_gated"] = (mb, np.where(inside, np.maximum(vs, vb), vb))
    return out


def pick_threshold(models, P, n_tasks=N_FIT_TASKS, seed=2468, n_bins=8):
    """按先验采样上的覆盖率定门限：网络自报方差落在哪一档时区间盖不住名义比例。

    只用到先验采样与网络自身的输出，不需要精确后验，所以部署时能算。
    """
    rng = np.random.default_rng(seed)
    vb_all, hit_all = [], []
    for _ in range(n_tasks):
        n_ctx = int(rng.integers(8, 25))
        design = "uniform" if rng.random() < 0.5 else "paired"
        xc, xq, yc, yq = draw_task(rng, P, n_ctx, design)
        mb, vb = pfn_predict(models[1], xc, yc, xq)
        vb_all.append(vb)
        hit_all.append(np.abs(yq - mb) <= Z_NOMINAL * np.sqrt(vb))
    vb_all = np.concatenate(vb_all)
    hit_all = np.concatenate(hit_all).astype(float)

    order = np.argsort(vb_all)
    bins = np.array_split(order, n_bins)
    print(f"    先验采样上的覆盖率（名义 {NOMINAL:.2f}），按网络自报方差分档")
    bad = []
    for b in bins:
        cov = hit_all[b].mean()
        lo, hi = vb_all[b].min(), vb_all[b].max()
        flag = "偏小" if cov < NOMINAL else ""
        print(f"      方差 {lo:8.4f} – {hi:8.4f}：覆盖率 {cov:.3f} {flag}")
        if cov < NOMINAL:
            bad.append((float(lo), float(hi)))
    if not bad:
        print(f"    每一档的覆盖率都不低于名义值，不做任何抬升")
    else:
        print(f"    覆盖率不足的区间共 {len(bad)} 段，只在这些区间里把方差抬到两个模型的较大者")
    return bad


def draw_task(rng, P, n_ctx, design, latent=None):
    xc = P["draw_design"](rng, n_ctx, design)
    xq = rng.uniform(-P["x_half"], P["x_half"], P["n_query"])
    a, b = latent if latent else P["draw_latent"](rng)
    y = P["sample"](rng, np.concatenate([xc, xq]), a, b)
    return xc, xq, y[:n_ctx], y[n_ctx:]


def held_out_nll(models, P, threshold, n_tasks=N_FIT_TASKS, seed=4321):
    """先验采样上的平均留出 NLL，不需要精确后验。"""
    rng = np.random.default_rng(seed)
    acc = None
    for _ in range(n_tasks):
        n_ctx = int(rng.integers(8, 25))
        design = "uniform" if rng.random() < 0.5 else "paired"
        xc, xq, yc, yq = draw_task(rng, P, n_ctx, design)
        outs = variants(pfn_predict(models[0], xc, yc, xq),
                        pfn_predict(models[1], xc, yc, xq), threshold)
        if acc is None:
            acc = {k: [] for k in outs}
        for k, (m, v) in outs.items():
            acc[k].append(float(np.mean(0.5 * (np.log(2 * np.pi * v) + (yq - m) ** 2 / v))))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def run_cell(models, P, cell, threshold, n_tasks=N_EVAL_TASKS, seed=0):
    a, b, n_ctx, design = cell
    rng = np.random.default_rng(seed)
    acc = None
    for _ in range(n_tasks):
        xc, xq, yc, yq = draw_task(rng, P, n_ctx, design, latent=(a, b))
        mu_e, var_e, _ = P["posterior"](xc, yc, xq, *P["grids"])
        outs = variants(pfn_predict(models[0], xc, yc, xq),
                        pfn_predict(models[1], xc, yc, xq), threshold)
        if acc is None:
            acc = {k: {"kl": [], "nll": []} for k in outs}
        for k, (m, v) in outs.items():
            acc[k]["kl"].append(float(np.mean(gauss_kl(mu_e, var_e, m, v))))
            acc[k]["nll"].append(float(np.mean(0.5 * (np.log(2 * np.pi * v)
                                                      + (yq - m) ** 2 / v))))
    row = {P["keys"][0]: a, P["keys"][1]: b, "n_ctx": n_ctx, "design": design}
    for k, d in acc.items():
        row[k] = {"kl": float(np.mean(d["kl"])),
                  "kl_se": float(np.std(d["kl"]) / np.sqrt(n_tasks)),
                  "nll": float(np.mean(d["nll"]))}
    return row


def worst_regime_nll(models, P, threshold, n_tasks=20, seed=8765):
    """逐区域的留出 NLL 里最差的那个。

    这个判据同样只需要先验采样，不需要精确后验，所以部署时能算；
    它对应的是「跨区域的可靠性」而不是「平均似然」。
    """
    acc = None
    for cell in P["cells"]:
        a, b, n_ctx, design = cell
        rng = np.random.default_rng(seed)
        per = None
        for _ in range(n_tasks):
            xc, xq, yc, yq = draw_task(rng, P, n_ctx, design, latent=(a, b))
            outs = variants(pfn_predict(models[0], xc, yc, xq),
                            pfn_predict(models[1], xc, yc, xq), threshold)
            if per is None:
                per = {k: [] for k in outs}
            for k, (m, v) in outs.items():
                per[k].append(float(np.mean(0.5 * (np.log(2 * np.pi * v)
                                                   + (yq - m) ** 2 / v))))
        if acc is None:
            acc = {k: [] for k in per}
        for k, v in per.items():
            acc[k].append(float(np.mean(v)))
    return {k: float(np.max(v)) for k, v in acc.items()}


def report(rows, held, worst):
    names = list(held)
    print(f"\n    {'变体':<22}{'平均留出 NLL':>14}{'最差区域 NLL':>14}"
          f"{'超额 KL 均值':>14}{'相对 big':>10}{'最差格子':>11}{'相对 big':>10}{'变好的格子':>12}")
    base_kl = np.array([r["big"]["kl"] for r in rows])
    stats = {}
    for k in names:
        kl = np.array([r[k]["kl"] for r in rows])
        better = int((kl < base_kl).sum())
        stats[k] = {"held_nll": held[k], "worst_nll": worst[k],
                    "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
                    "kl_ratio": float(kl.mean() / base_kl.mean()),
                    "kl_max_ratio": float(kl.max() / base_kl.max()),
                    "n_better": better}
        print(f"    {k:<22}{held[k]:>14.4f}{worst[k]:>14.4f}{kl.mean():>14.4f}"
              f"{kl.mean() / base_kl.mean():>10.3f}{kl.max():>11.4f}"
              f"{kl.max() / base_kl.max():>10.3f}{f'{better}/{len(rows)}':>12}")

    for label, crit in (("平均留出 NLL", held), ("最差区域 NLL", worst)):
        pick = min(names, key=lambda k: crit[k])
        print(f"\n    按{label}选：{pick}"
              f"（超额 KL 均值 {stats[pick]['kl_ratio']:.3f}，"
              f"最差格子 {stats[pick]['kl_max_ratio']:.3f}）")
        stats[f"picked_by_{label}"] = pick
    print("    两个判据都只需要先验采样，不需要精确后验，所以部署时都能算。")
    return stats


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "jump"
    small = sys.argv[2] if len(sys.argv) > 2 else "pfn_jump_w64.pt"
    big = sys.argv[3] if len(sys.argv) > 3 else "pfn_jump_40k.pt"

    P = load_prior(kind)
    models = [load_pfn(small)[0], load_pfn(big)[0]]
    print(f"    先验 {kind}，小模型 {small}，大模型 {big}", flush=True)

    threshold = pick_threshold(models, P)
    held = held_out_nll(models, P, threshold)
    print(f"    平均留出 NLL：" + "，".join(f"{k} {v:.4f}" for k, v in held.items()), flush=True)
    worst = worst_regime_nll(models, P, threshold)
    print(f"    最差区域 NLL：" + "，".join(f"{k} {v:.4f}" for k, v in worst.items()), flush=True)

    rows = []
    for i, cell in enumerate(P["cells"]):
        rows.append(run_cell(models, P, cell, threshold))
        print(f"    格子 {i + 1}/{len(P['cells'])} 完成", flush=True)
    out = Path(f"results/hybrid_{kind}.json")
    payload = {"rows": rows, "held": held, "worst": worst, "threshold": threshold}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    payload["stats"] = report(rows, held, worst)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"\n    结果写到 {out}")
