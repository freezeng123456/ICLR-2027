"""能不能把先验掰动反过来用：靠 anchor 的噪声水平校正模型的置信度。

上一个实验证明 anchor 的噪声会把模型在无关目标任务上的置信度整体拉低。
如果这真的是「模型推断了一个全局噪声水平」，那它就应该是可控的：

  目标任务本身噪声很大时，模型往往过度自信 -> 用高噪声 anchor 把它压下来
  目标任务本身很干净时                     -> 用低噪声 anchor 让它保持锐利

预期出现交叉：最优的 anchor 噪声随目标任务的真实噪声上升而上升。
若出现交叉，说明这个杠杆是可用的，不只是一个漏洞。

主指标用 log loss（严格适当评分规则，同时反映准确与校准），
辅以 ECE（纯校准）和准确率（纯决策）。
"""

import logging
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from tabicl import TabICLClassifier

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

D = 8
N_CTX, N_TEST, N_ANCHOR = 30, 400, 60
FAR_SHIFT = 6.0
TARGET_NOISES = [0.03, 0.30]
ANCHOR_ETAS = [0.0, 0.15, 0.35, 0.50]
N_SEEDS = 25


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
    return e


def run():
    pfn = TabICLClassifier(device="cpu")
    res = {tn: {"base": {"ll": [], "ece": [], "acc": []},
                **{e: {"ll": [], "ece": [], "acc": []} for e in ANCHOR_ETAS}}
           for tn in TARGET_NOISES}

    for tn in TARGET_NOISES:
        for target in LABELERS:
            done, seed = 0, 0
            while done < N_SEEDS:
                seed += 1
                rng = np.random.default_rng(30_000 + seed)
                lab = LABELERS[target](rng)
                X = rng.normal(0, 1, (N_CTX + N_TEST, D))
                y = flip(lab(X), tn, rng)
                Xtr, ytr, Xte, yte = X[:N_CTX], y[:N_CTX], X[N_CTX:], y[N_CTX:]
                if ytr.mean() < 0.15 or ytr.mean() > 0.85:
                    continue

                def measure(xc, yc, b):
                    pfn.fit(xc, yc)
                    p = np.clip(pfn.predict_proba(Xte), 1e-6, 1 - 1e-6)
                    b["ll"].append(log_loss(yte, p[:, 1], labels=[0, 1]))
                    b["ece"].append(ece(p, yte))
                    b["acc"].append(accuracy_score(yte, p.argmax(1)))

                measure(Xtr, ytr, res[tn]["base"])
                other = [k for k in LABELERS if k != target][0]
                g = LABELERS[other](rng)
                Xa0 = rng.normal(0, 1, (N_ANCHOR, D))
                ya0, Xa = g(Xa0), Xa0 + FAR_SHIFT
                for e in ANCHOR_ETAS:
                    measure(np.vstack([Xtr, Xa]), np.concatenate([ytr, flip(ya0, e, rng)]),
                            res[tn][e])
                done += 1
    return res


def paired(a, b):
    d = np.array(a) - np.array(b)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), se, (d.mean() / se if se > 0 else 0.0)


def report(res):
    print("\n" + "=" * 84)
    print("能否用 anchor 的噪声水平校正模型置信度")
    print("=" * 84)
    print(f"""
  每格 {N_SEEDS} 个种子 × 2 种目标函数 / {N_CTX} 行真实上下文 / {N_ANCHOR} 行远置 anchor /
  {N_TEST} 行测试。anchor 的标签来自与目标任务无关的另一个函数，只有噪声水平不同。
  log loss 越低越好。
""")
    best = {}
    for tn in TARGET_NOISES:
        b = res[tn]["base"]
        print(f"  --- 目标任务真实噪声 {int(tn * 100)}% ---")
        print(f"    基准（不加 anchor）: log loss {np.mean(b['ll']):.4f}   "
              f"ECE {np.mean(b['ece']):.4f}   准确率 {np.mean(b['acc']):.4f}")
        print(f"    {'anchor 噪声':>12}{'log loss':>12}{'相对基准':>16}{'ECE':>10}{'准确率':>10}")
        lls = []
        for e in ANCHOR_ETAS:
            d, _, t = paired(res[tn][e]["ll"], b["ll"])
            lls.append(np.mean(res[tn][e]["ll"]))
            mark = ""
            print(f"    {int(e * 100):>10}%{np.mean(res[tn][e]['ll']):>12.4f}"
                  f"{d:>+10.4f}(t={t:+5.1f}){np.mean(res[tn][e]['ece']):>10.4f}"
                  f"{np.mean(res[tn][e]['acc']):>10.4f}{mark}")
        best[tn] = ANCHOR_ETAS[int(np.argmin(lls))]
        print(f"    -> log loss 最优的 anchor 噪声：{int(best[tn] * 100)}%\n")

    lo, hi = TARGET_NOISES[0], TARGET_NOISES[-1]
    cross = best[hi] > best[lo]
    print(f"""  关键判据：最优 anchor 噪声是否随目标任务真实噪声上升
    目标噪声 {int(lo * 100)}% -> 最优 anchor 噪声 {int(best[lo] * 100)}%
    目标噪声 {int(hi * 100)}% -> 最优 anchor 噪声 {int(best[hi] * 100)}%
    {'出现交叉：这个杠杆是可控的，能按目标任务的性质调节。' if cross
     else '未出现交叉：掰动是单向的副作用，还谈不上可控。'}""")


if __name__ == "__main__":
    report(run())
