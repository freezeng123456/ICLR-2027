"""真实（或接近真实）表格数据上的校准对照。

合成高斯云不能单独撑一篇论文。这里用 sklearn / OpenML 的小规模二分类表，
在同一份划分上比较：基准、温度缩放、真实上下文注噪、far-anchor、额外 ID 行。

far-anchor 的构造与合成实验对齐：特征标准化后平移 +6，标签来自与任务无关的
随机线性规则，再按 η=0.50 翻转。它们对决策边界几乎没有信息，只携带噪声水平。
"""

import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.datasets import load_breast_cancer, make_classification, make_moons
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabicl import TabICLClassifier

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

N_SEEDS = 10
N_TRAIN, N_CAL, N_ANCHOR = 50, 60, 60
MIN_TEST = 80
FAR_SHIFT = 6.0
ETA = 0.50
OUT_PATH = Path("results/real_calib.json")


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
    z = np.clip(logits_from_p(p[:, 1]) / T, -30, 30)
    p1 = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)
    return np.column_stack([1 - p1, p1])


def fit_temperature(p1, y):
    z0 = logits_from_p(p1)

    def nll(T):
        z = np.clip(z0 / T, -30, 30)
        p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def flip(y, eta, rng):
    return np.where(rng.random(len(y)) < eta, 1 - y, y)


def metrics(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    # log_loss needs both classes present in labels= to avoid crash
    return {
        "ll": float(log_loss(y, p[:, 1], labels=[0, 1])),
        "ece": ece(p, y),
        "acc": float(accuracy_score(y, p.argmax(1))),
        "conf": float(p.max(1).mean()),
    }


def load_openml(data_id, name):
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(data_id=data_id, as_frame=False, parser="liac-arff")
    X = np.asarray(ds.data, dtype=float)
    y = np.asarray(ds.target)
    if X.ndim != 2:
        raise ValueError(f"{name} has unexpected X shape {X.shape}")
    # keep numeric columns only
    numeric = []
    for j in range(X.shape[1]):
        col = X[:, j]
        try:
            col_f = col.astype(float)
        except (TypeError, ValueError):
            continue
        if np.isfinite(col_f).mean() < 0.9:
            continue
        numeric.append(col_f)
    if not numeric:
        raise ValueError(f"{name} has no numeric features")
    X = np.column_stack(numeric)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"{name} is not binary ({len(classes)} classes)")
    y = (y == classes[1]).astype(int)
    mask = np.isfinite(X).all(1)
    return name, X[mask], y[mask]


def builtin_datasets():
    out = []
    bc = load_breast_cancer()
    out.append(("breast_cancer", bc.data.astype(float), (bc.target == 1).astype(int)))
    rng = np.random.default_rng(0)
    X, y = make_classification(n_samples=800, n_features=20, n_informative=8,
                               n_redundant=4, n_clusters_per_class=2,
                               class_sep=1.0, flip_y=0.05, random_state=0)
    out.append(("make_classification", X.astype(float), y.astype(int)))
    X, y = make_moons(n_samples=800, noise=0.25, random_state=0)
    # pad moons to 8-d so far-shift still has a place to go
    pad = rng.normal(0, 1, (len(X), 6))
    out.append(("make_moons", np.hstack([X, pad]).astype(float), y.astype(int)))
    return out


def try_openml():
    catalog = [
        (37, "diabetes"),
        (31, "credit-g"),
        (44, "spambase"),
        (1462, "banknote"),
        (1489, "phoneme"),
        (1464, "blood-transfusion"),
        (1063, "kc2"),
        (1510, "wdbc"),
    ]
    out = []
    for data_id, name in catalog:
        try:
            out.append(load_openml(data_id, name))
            print(f"  loaded openml {name} ({data_id})", flush=True)
        except Exception as exc:
            print(f"  skip openml {name}: {exc}", flush=True)
    return out


def split_xy(X, y, rng):
    n = len(y)
    need = N_TRAIN + N_CAL + MIN_TEST
    if n < need:
        return None
    Xtr, Xrest, ytr, yrest = train_test_split(
        X, y, train_size=N_TRAIN, stratify=y, random_state=int(rng.integers(1e9)))
    Xcal, Xte, ycal, yte = train_test_split(
        Xrest, yrest, train_size=N_CAL, stratify=yrest,
        random_state=int(rng.integers(1e9)))
    if len(yte) < MIN_TEST:
        return None
    if len(yte) > 400:
        Xte, _, yte, _ = train_test_split(
            Xte, yte, train_size=400, stratify=yte,
            random_state=int(rng.integers(1e9)))
    if min(ytr.mean(), ycal.mean(), yte.mean()) < 0.10:
        return None
    if max(ytr.mean(), ycal.mean(), yte.mean()) > 0.90:
        return None
    return Xtr, ytr, Xcal, ycal, Xte, yte


def run_dataset(name, X, y, pfn):
    print(f"  dataset {name}: n={len(y)} d={X.shape[1]} pos={y.mean():.2f}", flush=True)
    buckets = {k: {"ll": [], "ece": [], "acc": [], "conf": []}
               for k in ("base", "temp_cal", "id_extra", "ctx_50", "far_50")}
    used = 0
    for seed in range(N_SEEDS * 3):
        if used >= N_SEEDS:
            break
        rng = np.random.default_rng(1000 + seed)
        spl = split_xy(X, y, rng)
        if spl is None:
            continue
        Xtr, ytr, Xcal, ycal, Xte, yte = spl
        scaler = StandardScaler().fit(Xtr)
        Xtr, Xcal, Xte = scaler.transform(Xtr), scaler.transform(Xcal), scaler.transform(Xte)

        pfn.fit(Xtr, ytr)
        p_te = pfn.predict_proba(Xte)
        p_cal = pfn.predict_proba(Xcal)
        for k, v in metrics(p_te, yte).items():
            buckets["base"][k].append(v)
        T = fit_temperature(p_cal[:, 1], ycal)
        for k, v in metrics(apply_T(p_te, T), yte).items():
            buckets["temp_cal"][k].append(v)

        pfn.fit(np.vstack([Xtr, Xcal]), np.concatenate([ytr, ycal]))
        for k, v in metrics(pfn.predict_proba(Xte), yte).items():
            buckets["id_extra"][k].append(v)

        pfn.fit(Xtr, flip(ytr.copy(), ETA, rng))
        for k, v in metrics(pfn.predict_proba(Xte), yte).items():
            buckets["ctx_50"][k].append(v)

        Xa = rng.normal(0, 1, (N_ANCHOR, Xtr.shape[1])) + FAR_SHIFT
        w = rng.normal(0, 1, Xtr.shape[1])
        ya = flip((Xa @ w > 0).astype(int), ETA, rng)
        pfn.fit(np.vstack([Xtr, Xa]), np.concatenate([ytr, ya]))
        for k, v in metrics(pfn.predict_proba(Xte), yte).items():
            buckets["far_50"][k].append(v)
        used += 1
    if used < 3:
        print(f"    not enough valid splits ({used})", flush=True)
        return None
    mean = {m: {k: float(np.mean(v)) for k, v in b.items()} | {"n": used}
            for m, b in buckets.items()}
    print(f"    n_splits={used}  "
          + "  ".join(f"{m} ll={mean[m]['ll']:.4f} ece={mean[m]['ece']:.4f} acc={mean[m]['acc']:.3f}"
                      for m in mean), flush=True)
    return {"mean": mean, "raw": {m: {k: [float(x) for x in vv] for k, vv in b.items()}
                                 for m, b in buckets.items()}}


def paired_delta(raw_a, raw_b, key="ll"):
    d = np.array(raw_a[key]) - np.array(raw_b[key])
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    t = d.mean() / se if se > 0 else 0.0
    return {"delta": float(d.mean()), "t": float(t)}


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    pfn = TabICLClassifier(device="cpu", verbose=False)
    datasets = builtin_datasets() + try_openml()
    results = {}
    for name, X, y in datasets:
        try:
            row = run_dataset(name, X, y, pfn)
        except Exception as exc:
            print(f"  fail {name}: {exc}", flush=True)
            row = None
        if row:
            results[name] = row
    print("\n" + "=" * 88)
    print("真实/半真实表：far-anchor vs 温度缩放 vs 上下文注噪")
    print("=" * 88)
    print(f"  {'数据集':<22}{'base ll':>9}{'temp':>9}{'id_extra':>10}{'ctx_50':>9}{'far_50':>9}"
          f"{'far-temp':>10}")
    for name, row in results.items():
        m = row["mean"]
        dt = paired_delta(row["raw"]["far_50"], row["raw"]["temp_cal"], "ll")
        print(f"  {name:<22}{m['base']['ll']:>9.4f}{m['temp_cal']['ll']:>9.4f}"
              f"{m['id_extra']['ll']:>10.4f}{m['ctx_50']['ll']:>9.4f}{m['far_50']['ll']:>9.4f}"
              f"{dt['delta']:>+7.4f}t={dt['t']:+4.1f}")
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n  写到 {OUT_PATH}")
