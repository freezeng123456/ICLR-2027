"""远置 anchor 校准 vs 经典校准基线。

如果温度缩放或「把噪声直接打进真实上下文」就能追上 far-anchor 的 log loss / ECE，
方法半篇站不住，只保留机制半篇。本实验把这几件事放在同一批种子上成对比较。

同一份 60 个 ID 标注点有两种用法（对 far-anchor 不公平的强基线）：
  temp_cal60 / platt_cal60  只用来拟合一个校准参数，不进上下文
  id_extra                  当作额外的真实上下文行

far-anchor 用的是另外 60 个远离数据云、标签来自无关函数的点，不消耗 ID 标注。

ctx_noise 不增加任何点，只翻转真实训练标签——若它在校准上追上 far-anchor，
同时准确率明显更差，说明「远」和「任务无关」是在保护决策边界；
若准确率也不差，则 far 这个设计是多余的。
"""

import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from tabicl import TabICLClassifier

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

D = 8
N_CTX, N_CAL, N_TEST, N_ANCHOR = 30, 60, 400, 60
FAR_SHIFT = 6.0
TARGET_NOISES = [0.03, 0.30]
ANCHOR_ETAS = [0.0, 0.15, 0.35, 0.50]
CTX_ETAS = [0.15, 0.35, 0.50]
N_SEEDS = 25
OUT_PATH = Path("results/calib_baselines.json")


def make_rule_labeler(rng):
    i = rng.permutation(D)
    t = rng.normal(0, 0.4, 3)
    return lambda X: (((X[:, i[0]] > t[0]) & (X[:, i[1]] > t[1]))
                      | (X[:, i[2]] > t[2] + 0.8)).astype(int)


def make_smooth_labeler(rng):
    w, v = rng.normal(0, 1, D), rng.normal(0, 1, D)
    return lambda X: (np.tanh(X @ w) + 0.6 * np.sin(X @ v) > 0).astype(int)


LABELERS = {"规则型": make_rule_labeler, "平滑型": make_smooth_labeler}


def flip(y, eta, rng):
    return np.where(rng.random(len(y)) < eta, 1 - y, y)


def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def logits_from_p(p1):
    p1 = np.clip(p1, 1e-6, 1 - 1e-6)
    return np.log(p1) - np.log(1 - p1)


def apply_T(p, T):
    z = logits_from_p(p[:, 1]) / T
    z = np.clip(z, -30, 30)
    p1 = 1.0 / (1.0 + np.exp(-z))
    p1 = np.clip(p1, 1e-6, 1 - 1e-6)
    return np.column_stack([1 - p1, p1])


def fit_temperature(p1, y):
    z0 = logits_from_p(p1)

    def nll(T):
        z = np.clip(z0 / T, -30, 30)
        p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def apply_platt(p, coef, intercept):
    z = coef * logits_from_p(p[:, 1]) + intercept
    z = np.clip(z, -30, 30)
    p1 = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)
    return np.column_stack([1 - p1, p1])


def fit_platt(p1, y):
    if len(np.unique(y)) < 2:
        return 1.0, 0.0
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
    lr.fit(logits_from_p(p1).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def metrics(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "ll": float(log_loss(y, p[:, 1], labels=[0, 1])),
        "ece": ece(p, y),
        "acc": float(accuracy_score(y, p.argmax(1))),
        "conf": float(p.max(1).mean()),
    }


def empty_bucket():
    return {"ll": [], "ece": [], "acc": [], "conf": []}


def method_names():
    names = ["base", "temp_cal60", "platt_cal60", "temp_T2", "id_extra"]
    names += [f"ctx_{int(e * 100):02d}" for e in CTX_ETAS]
    names += [f"far_{int(e * 100):02d}" for e in ANCHOR_ETAS]
    return names


def run():
    pfn = TabICLClassifier(device="cpu", verbose=False)
    res = {tn: {m: empty_bucket() for m in method_names()} | {"T": [], "platt_coef": []}
           for tn in TARGET_NOISES}

    n_total = len(TARGET_NOISES) * len(LABELERS) * N_SEEDS
    done_all = 0
    for tn in TARGET_NOISES:
        for target in LABELERS:
            done, seed = 0, 0
            while done < N_SEEDS:
                seed += 1
                rng = np.random.default_rng(40_000 + seed + int(tn * 1000))
                lab = LABELERS[target](rng)
                X = rng.normal(0, 1, (N_CTX + N_CAL + N_TEST, D))
                y = flip(lab(X), tn, rng)
                Xtr, ytr = X[:N_CTX], y[:N_CTX]
                Xcal, ycal = X[N_CTX:N_CTX + N_CAL], y[N_CTX:N_CTX + N_CAL]
                Xte, yte = X[N_CTX + N_CAL:], y[N_CTX + N_CAL:]
                if min(ytr.mean(), ycal.mean(), yte.mean()) < 0.15:
                    continue
                if max(ytr.mean(), ycal.mean(), yte.mean()) > 0.85:
                    continue

                pfn.fit(Xtr, ytr)
                p_te = pfn.predict_proba(Xte)
                p_cal = pfn.predict_proba(Xcal)
                rec = res[tn]
                for k, v in metrics(p_te, yte).items():
                    rec["base"][k].append(v)

                T = fit_temperature(p_cal[:, 1], ycal)
                rec["T"].append(T)
                for k, v in metrics(apply_T(p_te, T), yte).items():
                    rec["temp_cal60"][k].append(v)
                for k, v in metrics(apply_T(p_te, 2.0), yte).items():
                    rec["temp_T2"][k].append(v)

                coef, intercept = fit_platt(p_cal[:, 1], ycal)
                rec["platt_coef"].append(coef)
                for k, v in metrics(apply_platt(p_te, coef, intercept), yte).items():
                    rec["platt_cal60"][k].append(v)

                pfn.fit(np.vstack([Xtr, Xcal]), np.concatenate([ytr, ycal]))
                for k, v in metrics(pfn.predict_proba(Xte), yte).items():
                    rec["id_extra"][k].append(v)

                for e in CTX_ETAS:
                    pfn.fit(Xtr, flip(ytr.copy(), e, rng))
                    for k, v in metrics(pfn.predict_proba(Xte), yte).items():
                        rec[f"ctx_{int(e * 100):02d}"][k].append(v)

                other = [k for k in LABELERS if k != target][0]
                g = LABELERS[other](rng)
                Xa0 = rng.normal(0, 1, (N_ANCHOR, D))
                ya0, Xa = g(Xa0), Xa0 + FAR_SHIFT
                for e in ANCHOR_ETAS:
                    pfn.fit(np.vstack([Xtr, Xa]), np.concatenate([ytr, flip(ya0, e, rng)]))
                    for k, v in metrics(pfn.predict_proba(Xte), yte).items():
                        rec[f"far_{int(e * 100):02d}"][k].append(v)

                done += 1
                done_all += 1
                if done_all % 5 == 0 or done_all == n_total:
                    print(f"  [{done_all}/{n_total}] tn={tn:.2f} {target} seed={done}",
                          flush=True)
    return res


def paired(a, b):
    d = np.array(a) - np.array(b)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    t = d.mean() / se if se > 0 else 0.0
    return float(d.mean()), float(se), float(t)


def summarize(res):
    methods = method_names()
    out = {}
    for tn in TARGET_NOISES:
        base = res[tn]["base"]
        block = {"n": len(base["ll"]), "mean_T": float(np.mean(res[tn]["T"]))}
        for m in methods:
            block[m] = {k: float(np.mean(res[tn][m][k])) for k in ("ll", "ece", "acc", "conf")}
            d, se, t = paired(res[tn][m]["ll"], base["ll"])
            block[m]["ll_vs_base"] = {"delta": d, "se": se, "t": t}
            d, se, t = paired(res[tn][m]["ece"], base["ece"])
            block[m]["ece_vs_base"] = {"delta": d, "se": se, "t": t}
        # head-to-head vs far_50
        far = res[tn]["far_50"]
        for m in ("temp_cal60", "platt_cal60", "temp_T2", "id_extra", "ctx_50"):
            d, se, t = paired(res[tn][m]["ll"], far["ll"])
            block[m]["ll_vs_far50"] = {"delta": d, "se": se, "t": t}
            d, se, t = paired(res[tn][m]["acc"], far["acc"])
            block[m]["acc_vs_far50"] = {"delta": d, "se": se, "t": t}
        out[str(tn)] = block
    return out


def report(res):
    summary = summarize(res)
    labels = {
        "base": "基准（无校准）",
        "temp_cal60": "温度缩放（60 个 ID 校准点）",
        "platt_cal60": "Platt 缩放（同上 60 点）",
        "temp_T2": "固定 T=2（无额外数据）",
        "id_extra": "60 个 ID 点直接进上下文",
        "ctx_15": "真实上下文注噪 15%",
        "ctx_35": "真实上下文注噪 35%",
        "ctx_50": "真实上下文注噪 50%",
        "far_00": "far-anchor 噪声 0%",
        "far_15": "far-anchor 噪声 15%",
        "far_35": "far-anchor 噪声 35%",
        "far_50": "far-anchor 噪声 50%",
    }
    print("\n" + "=" * 96)
    print("far-anchor 校准 vs 温度缩放 / 上下文注噪 / 额外 ID 样本")
    print("=" * 96)
    print(f"""
  每格 {N_SEEDS} 种子 × 2 种目标函数 / 训练 {N_CTX} / 校准或对照 {N_CAL} /
  测试 {N_TEST} / far-anchor {N_ANCHOR}。log loss 越低越好。
  温度缩放与 Platt 使用与 id_extra 完全同一批 60 个 ID 标注点。
""")
    for tn in TARGET_NOISES:
        s = summary[str(tn)]
        print(f"  --- 目标任务真实噪声 {int(tn * 100)}%  （n={s['n']}，平均拟合 T={s['mean_T']:.2f}） ---")
        print(f"    {'方法':<28}{'log loss':>10}{'相对基准':>18}{'ECE':>8}{'准确率':>8}{'置信度':>8}")
        for m in method_names():
            vs = s[m]["ll_vs_base"]
            print(f"    {labels[m]:<28}{s[m]['ll']:>10.4f}"
                  f"{vs['delta']:>+8.4f}(t={vs['t']:+5.1f})"
                  f"{s[m]['ece']:>8.4f}{s[m]['acc']:>8.4f}{s[m]['conf']:>8.4f}")
        print()
        print("    与 far-anchor η=50% 成对比较（Δ = 该方法 − far_50，log loss 负值表示该方法更好）：")
        for m in ("temp_cal60", "platt_cal60", "temp_T2", "id_extra", "ctx_50"):
            ll = s[m]["ll_vs_far50"]
            acc = s[m]["acc_vs_far50"]
            print(f"      {labels[m]:<28} Δll {ll['delta']:+.4f} (t={ll['t']:+5.1f})"
                  f"   Δacc {acc['delta']:+.4f} (t={acc['t']:+5.1f})")
        print()

    print("""  判据：
    1) temp/platt 的 log loss 追上 far_50
       -> 方法半篇降为「不需要 ID 校准标签的校准器」，机制半篇仍成立
    2) ctx_50 校准追上 far_50，但准确率明显更差
       -> 远置 + 任务无关是在保护决策边界，不是摆设
    3) ctx_50 在校准和准确率上都追上
       -> far 这个设计是多余的，直接给真实上下文加噪即可
    4) id_extra 显著更好
       -> 有 ID 标注时应该把点放进上下文，far-anchor 只在没有 ID 标签时有意义""")
    return summary


def to_jsonable(res):
    out = {}
    for tn, block in res.items():
        out[str(tn)] = {}
        for k, v in block.items():
            if isinstance(v, dict):
                out[str(tn)][k] = {kk: [float(x) for x in vv] for kk, vv in v.items()}
            else:
                out[str(tn)][k] = [float(x) for x in v]
    return out


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    res = run()
    summary = report(res)
    payload = {"summary": summary, "raw": to_jsonable(res),
               "config": {"N_CTX": N_CTX, "N_CAL": N_CAL, "N_TEST": N_TEST,
                          "N_ANCHOR": N_ANCHOR, "N_SEEDS": N_SEEDS,
                          "TARGET_NOISES": TARGET_NOISES,
                          "ANCHOR_ETAS": ANCHOR_ETAS, "CTX_ETAS": CTX_ETAS}}
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\n  原始结果已写到 {OUT_PATH}")
