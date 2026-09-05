"""只用训练点拟合温度 T，不消耗额外 ID 标签。

exp_calib_baselines 里温度缩放用了 60 个额外 ID 点，对 far-anchor 不公平。
这里 T 只在 30 个训练点的预测上拟合（模型已经在上下文里见过这些标签，
会偏乐观，这正是这个基线的真实用法），再与 far-anchor η=50% 成对比较。
"""

import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, log_loss
from tabicl import TabICLClassifier

from exp_calib_baselines import (D, FAR_SHIFT, LABELERS, N_ANCHOR, N_CTX, N_SEEDS,
                                 N_TEST, TARGET_NOISES, apply_T, fit_temperature,
                                 flip, metrics, paired)

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

OUT_PATH = Path("results/temp_on_train.json")


def run():
    pfn = TabICLClassifier(device="cpu", verbose=False)
    keys = ["base", "temp_train", "temp_T2", "far_50"]
    res = {tn: {k: {"ll": [], "ece": [], "acc": [], "conf": []} for k in keys} | {"T": []}
           for tn in TARGET_NOISES}
    n_total = len(TARGET_NOISES) * len(LABELERS) * N_SEEDS
    done_all = 0
    for tn in TARGET_NOISES:
        for target in LABELERS:
            done, seed = 0, 0
            while done < N_SEEDS:
                seed += 1
                rng = np.random.default_rng(50_000 + seed + int(tn * 1000))
                lab = LABELERS[target](rng)
                X = rng.normal(0, 1, (N_CTX + N_TEST, D))
                y = flip(lab(X), tn, rng)
                Xtr, ytr, Xte, yte = X[:N_CTX], y[:N_CTX], X[N_CTX:], y[N_CTX:]
                if ytr.mean() < 0.15 or ytr.mean() > 0.85:
                    continue
                pfn.fit(Xtr, ytr)
                p_te = pfn.predict_proba(Xte)
                p_tr = pfn.predict_proba(Xtr)
                rec = res[tn]
                for k, v in metrics(p_te, yte).items():
                    rec["base"][k].append(v)
                T = fit_temperature(p_tr[:, 1], ytr)
                rec["T"].append(T)
                for k, v in metrics(apply_T(p_te, T), yte).items():
                    rec["temp_train"][k].append(v)
                for k, v in metrics(apply_T(p_te, 2.0), yte).items():
                    rec["temp_T2"][k].append(v)
                other = [k for k in LABELERS if k != target][0]
                g = LABELERS[other](rng)
                Xa0 = rng.normal(0, 1, (N_ANCHOR, D))
                ya0, Xa = g(Xa0), Xa0 + FAR_SHIFT
                pfn.fit(np.vstack([Xtr, Xa]), np.concatenate([ytr, flip(ya0, 0.50, rng)]))
                for k, v in metrics(pfn.predict_proba(Xte), yte).items():
                    rec["far_50"][k].append(v)
                done += 1
                done_all += 1
                if done_all % 10 == 0:
                    print(f"  [{done_all}/{n_total}] tn={tn:.2f} {target}", flush=True)
    return res


def report(res):
    print("\n" + "=" * 88)
    print("温度缩放只在训练点上拟合 T，vs far-anchor η=50%")
    print("=" * 88)
    summary = {}
    for tn in TARGET_NOISES:
        print(f"\n  --- 目标噪声 {int(tn*100)}%  n={len(res[tn]['base']['ll'])}  "
              f"平均 T={np.mean(res[tn]['T']):.2f} ---")
        print(f"    {'方法':<16}{'log loss':>10}{'ECE':>8}{'准确率':>8}{'vs far Δll':>22}")
        block = {}
        for m in ("base", "temp_train", "temp_T2", "far_50"):
            ll, ece_m, acc = (np.mean(res[tn][m][k]) for k in ("ll", "ece", "acc"))
            d, se, t = paired(res[tn][m]["ll"], res[tn]["far_50"]["ll"])
            print(f"    {m:<16}{ll:>10.4f}{ece_m:>8.4f}{acc:>8.4f}{d:>+8.4f}(t={t:+5.1f})")
            block[m] = {"ll": float(ll), "ece": float(ece_m), "acc": float(acc),
                        "vs_far": {"delta": d, "t": t}}
        block["mean_T"] = float(np.mean(res[tn]["T"]))
        summary[str(tn)] = block
    return summary


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    res = run()
    summary = report(res)
    raw = {str(tn): {k: ( {kk: [float(x) for x in vv] for kk, vv in v.items()} if isinstance(v, dict)
                          else [float(x) for x in v] )
                     for k, v in block.items()} for tn, block in res.items()}
    OUT_PATH.write_text(json.dumps({"summary": summary, "raw": raw}, indent=2))
    print(f"\n  写到 {OUT_PATH}")
