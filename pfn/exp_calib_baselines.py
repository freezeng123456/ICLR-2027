"""anchor 噪声校准 vs 那些一行代码就能做的对照。

这是决定「方法」那半边论文成不成立的实验。论文主张「往上下文塞远处 anchor、
调它的标签噪声率，可以改善校准」。任何审稿人第一个问题都是：

    那我直接做温度缩放不就行了？

温度缩放是一个参数、校准领域三十年的默认基线。这里把所有一个旋钮的对照都摆出来：

  A. anchor 噪声 eta          论文的方法。塞 60 行远处 anchor，标签来自无关函数，
                              只调它们自己的噪声率。**不需要任何额外的真实标签。**
  B. 概率收缩 lambda          p -> (1-lambda)*p + lambda*0.5。一个旋钮，零数据。
  C. 温度缩放 T（扫）          logit -> logit / T。一个旋钮，零数据。
  D. 温度缩放（保留集拟合）     从 30 行上下文里切出 10 行做校准集，剩 20 行做上下文。
                              数据用量公平，但基座模型少看了 10 行。
  E. 温度缩放（额外真实数据）   另给 60 行**带真实标签**的校准数据。
                              不公平（用了 anchor 方法不需要的真实标签），作为上界参照。
  F. 上下文注噪 eta            不加 anchor，直接把自己那 30 行真实标签按 eta 翻转。
                              消融：anchor 的「远」到底重不重要。

A、B、C、F 都是「一个旋钮、零额外数据」，属于同一量级的对手，可以直接比。
每一族都按**在测试集上最优**来取（oracle 选择），这样对各方都一样宽松。

主指标 log loss（严格适当评分规则），辅以 ECE（纯校准）与准确率（纯决策）。
"""

import logging
import sys
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
ETAS = [0.0, 0.15, 0.35, 0.50]          # anchor 噪声 / 上下文注噪 共用
LAMBDAS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
TEMPS = [1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
EPS = 1e-6


def make_rule(rng):
    i, t = rng.permutation(D), rng.normal(0, 0.4, 3)
    return lambda X: (((X[:, i[0]] > t[0]) & (X[:, i[1]] > t[1]))
                      | (X[:, i[2]] > t[2] + 0.8)).astype(int)


def make_smooth(rng):
    w, v = rng.normal(0, 1, D), rng.normal(0, 1, D)
    return lambda X: (np.tanh(X @ w) + 0.6 * np.sin(X @ v) > 0).astype(int)


LABELERS = {"规则型": make_rule, "平滑型": make_smooth}


def flip(y, eta, rng):
    return np.where(rng.random(len(y)) < eta, 1 - y, y)


def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    return sum(m.mean() * abs(correct[m].mean() - conf[m].mean())
               for lo, hi in zip(edges[:-1], edges[1:])
               if (m := (conf > lo) & (conf <= hi)).sum())


def scores(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    p = p / p.sum(1, keepdims=True)
    return {"ll": log_loss(y, p[:, 1], labels=[0, 1]),
            "ece": ece(p, y), "acc": accuracy_score(y, p.argmax(1))}


def shrink(p, lam):
    return (1 - lam) * p + lam * 0.5


def temper(p, T):
    """对二分类概率做温度缩放：logit / T 再过 sigmoid。"""
    p1 = np.clip(p[:, 1], EPS, 1 - EPS)
    z = np.log(p1 / (1 - p1)) / T
    q = 1 / (1 + np.exp(-z))
    return np.stack([1 - q, q], 1)


def fit_temperature(p_cal, y_cal):
    """在校准集上用一维网格搜最优温度（比梯度下降更稳，且够快）。"""
    grid = np.exp(np.linspace(np.log(0.5), np.log(10.0), 60))
    best, bestT = np.inf, 1.0
    for T in grid:
        q = np.clip(temper(p_cal, T)[:, 1], EPS, 1 - EPS)
        nll = -np.mean(y_cal * np.log(q) + (1 - y_cal) * np.log(1 - q))
        if nll < best:
            best, bestT = nll, T
    return bestT


def run():
    clf = TabICLClassifier(device="cpu")
    # res[目标噪声][方法名] = list of dict(ll/ece/acc)
    res = {tn: {} for tn in TARGET_NOISES}

    def rec(tn, name, s):
        res[tn].setdefault(name, []).append(s)

    for tn in TARGET_NOISES:
        for lname, maker in LABELERS.items():
            done, seed = 0, 0
            while done < N_SEEDS:
                seed += 1
                rng = np.random.default_rng(80_000 + seed)
                lab = maker(rng)
                X = rng.normal(0, 1, (N_CTX + N_TEST + N_ANCHOR, D))
                y = flip(lab(X), tn, rng)
                Xtr, ytr = X[:N_CTX], y[:N_CTX]
                Xte, yte = X[N_CTX:N_CTX + N_TEST], y[N_CTX:N_CTX + N_TEST]
                Xex, yex = X[N_CTX + N_TEST:], y[N_CTX + N_TEST:]   # 额外真实数据
                if ytr.mean() < 0.15 or ytr.mean() > 0.85:
                    continue

                # ---------- 基准 ----------
                clf.fit(Xtr, ytr)
                p_base = clf.predict_proba(Xte)
                rec(tn, "基准（不加 anchor）", scores(p_base, yte))

                # ---------- B 概率收缩（后处理，零成本） ----------
                for lam in LAMBDAS:
                    rec(tn, f"B 收缩 lam={lam}", scores(shrink(p_base, lam), yte))

                # ---------- C 温度缩放（后处理，零成本） ----------
                for T in TEMPS:
                    rec(tn, f"C 温度 T={T}", scores(temper(p_base, T), yte))

                # ---------- A anchor 噪声（论文方法） ----------
                other = [k for k in LABELERS if k != lname][0]
                g = LABELERS[other](rng)
                Xa0 = rng.normal(0, 1, (N_ANCHOR, D))
                ya0, Xa = g(Xa0), Xa0 + FAR_SHIFT
                for e in ETAS:
                    clf.fit(np.vstack([Xtr, Xa]), np.concatenate([ytr, flip(ya0, e, rng)]))
                    rec(tn, f"A anchor eta={e}", scores(clf.predict_proba(Xte), yte))

                # ---------- F 上下文注噪（消融：远不远重要吗） ----------
                for e in ETAS:
                    if e == 0.0:
                        rec(tn, f"F 注噪 eta={e}", scores(p_base, yte))
                        continue
                    clf.fit(Xtr, flip(ytr, e, rng))
                    rec(tn, f"F 注噪 eta={e}", scores(clf.predict_proba(Xte), yte))

                # ---------- D 温度缩放（从 30 行里切 10 行做校准） ----------
                clf.fit(Xtr[:20], ytr[:20])
                p_cal = clf.predict_proba(Xtr[20:])
                T_hat = fit_temperature(p_cal, ytr[20:])
                p_sub = clf.predict_proba(Xte)
                rec(tn, "D 温度(切分保留集)", scores(temper(p_sub, T_hat), yte))

                # ---------- E 温度缩放（额外 60 行真实标签，上界参照） ----------
                clf.fit(Xtr, ytr)
                T_hat2 = fit_temperature(clf.predict_proba(Xex), yex)
                rec(tn, "E 温度(额外真实数据)", scores(temper(p_base, T_hat2), yte))

                done += 1
                print(f"    [{tn:.2f}/{lname}] {done}/{N_SEEDS}", end="\r", flush=True)
    return res


def paired(a, b):
    d = np.array(a) - np.array(b)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), se, (d.mean() / se if se > 0 else 0.0)


def report(res):
    print("\n" + "=" * 92)
    print("anchor 噪声校准 vs 一行代码的对照")
    print("=" * 92)
    print(f"""
  每格 {N_SEEDS} 个种子 x 2 种目标函数 / {N_CTX} 行上下文 / {N_ANCHOR} 行远置 anchor /
  {N_TEST} 行测试 / {D} 个特征。log loss 与 ECE 越低越好。
""")
    for tn in TARGET_NOISES:
        base = res[tn]["基准（不加 anchor）"]
        bll = np.mean([s["ll"] for s in base])
        print(f"\n  ══ 目标任务真实噪声 {int(tn*100)}% ══"
              f"   基准 log loss {bll:.4f}  ECE {np.mean([s['ece'] for s in base]):.4f}"
              f"  准确率 {np.mean([s['acc'] for s in base]):.4f}\n")
        print(f"    {'方法':<24}{'log loss':>10}{'相对基准':>18}{'ECE':>9}{'准确率':>9}")
        print("    " + "-" * 70)
        for name in res[tn]:
            if name == "基准（不加 anchor）":
                continue
            v = res[tn][name]
            ll = np.mean([s["ll"] for s in v])
            d, _, t = paired([s["ll"] for s in v], [s["ll"] for s in base])
            print(f"    {name:<24}{ll:>10.4f}{d:>+11.4f}(t={t:+5.1f})"
                  f"{np.mean([s['ece'] for s in v]):>9.4f}{np.mean([s['acc'] for s in v]):>9.4f}")

        # 每一族取 oracle 最优
        print(f"\n    ── 各方法族的 oracle 最优（对各方一样宽松）──")
        fams = {"A anchor（论文方法，零额外真实标签）": "A anchor",
                "B 概率收缩（零数据）": "B 收缩",
                "C 温度缩放（零数据）": "C 温度",
                "F 上下文注噪（零额外真实标签）": "F 注噪"}
        best = {}
        for label, pref in fams.items():
            cands = {k: np.mean([s["ll"] for s in res[tn][k]]) for k in res[tn]
                     if k.startswith(pref)}
            k = min(cands, key=cands.get)
            best[label] = (k, cands[k],
                           np.mean([s["ece"] for s in res[tn][k]]),
                           np.mean([s["acc"] for s in res[tn][k]]),
                           [s["ll"] for s in res[tn][k]])
        for label in ("D 温度(切分保留集)", "E 温度(额外真实数据)"):
            v = res[tn][label]
            best[label] = (label, np.mean([s["ll"] for s in v]),
                           np.mean([s["ece"] for s in v]),
                           np.mean([s["acc"] for s in v]), [s["ll"] for s in v])
        print(f"    {'方法族':<34}{'最优设置':<18}{'log loss':>10}{'ECE':>9}{'准确率':>9}")
        print("    " + "-" * 80)
        for label, (k, ll, e, a, _) in sorted(best.items(), key=lambda kv: kv[1][1]):
            setting = k.split()[-1] if k.startswith(("A ", "B ", "C ", "F ")) else "—"
            print(f"    {label:<34}{setting:<18}{ll:>10.4f}{e:>9.4f}{a:>9.4f}")

        # 直接配对：anchor 对上最强的零数据对手
        a_key = best["A anchor（论文方法，零额外真实标签）"]
        rivals = {k: v for k, v in best.items() if k.startswith(("B ", "C "))}
        r_label = min(rivals, key=lambda k: rivals[k][1])
        d, se, t = paired(a_key[4], rivals[r_label][4])
        verdict = ("anchor 显著更好" if d < 0 and abs(t) > 2 else
                   "对手显著更好" if d > 0 and abs(t) > 2 else "打平（无显著差异）")
        print(f"""
    关键判定（配对比较，log loss）：
      anchor 最优 − {r_label.split('（')[0]}最优 = {d:+.4f} ± {se:.4f}  (t = {t:+.1f})
      -> {verdict}""")


if __name__ == "__main__":
    report(run())
