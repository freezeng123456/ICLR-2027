"""诊断上一个负面结果：远处的 anchor 到底有没有被用上？

上一个实验发现结构化 anchor 与随机标签 anchor 带来同样的增益，说明结构信息
没有传进去。有两种可能的解释：
  (甲) 模型确实读了 anchor，但「规则型 vs 平滑型」不是它的先验会变化的那个轴
  (乙) 远处的行压根没被真正使用，上下文对它而言是一个局部的证据库

这个实验直接把两者分开：把**确实含有目标任务信息**的 anchor（标签由目标任务
本身的规则生成）放到离测试数据不同远近的位置，看增益随距离如何衰减。

  若近处增益大、随距离迅速衰减到与随机标签无异 -> 支持 (乙)：上下文是局部证据库，
    放在邻域之外的信息会被丢弃，因此「全局先验掰动」这条路在当前表格 PFN 上不通。
  若增益不随距离衰减 -> 支持 (甲)：远处的行是被用的，问题出在结构轴选错了。

对照组用同样位置、同样行数、但标签随机的 anchor，扣掉「多了一批远处数据」
本身带来的副作用。
"""

import logging
import warnings

import numpy as np
from sklearn.metrics import accuracy_score

from tabicl import TabICLClassifier

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

D = 8
N_CTX, N_TEST, N_ANCHOR = 30, 400, 60
LABEL_NOISE = 0.08
SHIFTS = [0.0, 1.0, 2.0, 3.0, 4.5, 6.0]
N_SEEDS = 30


def make_rule_labeler(rng):
    i = rng.permutation(D)
    t = rng.normal(0, 0.4, 3)
    return lambda X: (((X[:, i[0]] > t[0]) & (X[:, i[1]] > t[1]))
                      | (X[:, i[2]] > t[2] + 0.8)).astype(int)


def make_smooth_labeler(rng):
    w, v = rng.normal(0, 1, D), rng.normal(0, 1, D)

    def f(X):
        s = np.tanh(X @ w) + 0.6 * np.sin(X @ v)
        return (s > 0).astype(int)

    return f


LABELERS = {"规则型": make_rule_labeler, "平滑型": make_smooth_labeler}


def run():
    pfn = TabICLClassifier(device="cpu")
    res = {t: {"base": [], "info": {s: [] for s in SHIFTS}, "rand": {s: [] for s in SHIFTS}}
           for t in LABELERS}

    for target in LABELERS:
        done, seed = 0, 0
        while done < N_SEEDS:
            seed += 1
            rng = np.random.default_rng(90_000 + seed)
            lab = LABELERS[target](rng)
            X = rng.normal(0, 1, (N_CTX + N_TEST, D))
            y = np.where(rng.random(len(X)) < LABEL_NOISE, 1 - lab(X), lab(X))
            Xtr, ytr, Xte, yte = X[:N_CTX], y[:N_CTX], X[N_CTX:], y[N_CTX:]
            if ytr.mean() < 0.15 or ytr.mean() > 0.85:
                continue

            pfn.fit(Xtr, ytr)
            res[target]["base"].append(accuracy_score(yte, pfn.predict(Xte)))

            # anchor 的原始坐标固定，只改平移量，保证「信息量」在各距离下完全相同
            Xa0 = rng.normal(0, 1, (N_ANCHOR, D))
            ya_info = np.where(rng.random(N_ANCHOR) < LABEL_NOISE, 1 - lab(Xa0), lab(Xa0))
            ya_rand = rng.integers(0, 2, N_ANCHOR)

            for s in SHIFTS:
                Xa = Xa0 + s
                for key, ya in (("info", ya_info), ("rand", ya_rand)):
                    pfn.fit(np.vstack([Xtr, Xa]), np.concatenate([ytr, ya]))
                    res[target][key][s].append(accuracy_score(yte, pfn.predict(Xte)))
            done += 1
    return res


def paired(a, b):
    d = np.array(a) - np.array(b)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), se, (d.mean() / se if se > 0 else 0.0)


def report(res):
    print("\n" + "=" * 88)
    print("诊断：远处的 anchor 到底有没有被用上")
    print("=" * 88)
    print(f"""
  {N_SEEDS} 个种子 / {N_CTX} 行真实上下文 / {N_ANCHOR} 行 anchor / {N_TEST} 行测试 / {D} 个特征。
  anchor 的原始坐标在所有距离下完全相同，只改平移量，因此各行含的信息量严格一致。
  「含信息」= 标签由目标任务本身的规则生成；「随机」= 标签纯随机。
""")
    for t in LABELERS:
        base = res[t]["base"]
        print(f"  --- {t}目标（基准准确率 {np.mean(base):.4f}）---")
        print(f"    {'平移量':>8}{'含信息 anchor 的增益':>22}{'随机标签的增益':>18}"
              f"{'净结构增益':>16}")
        for s in SHIFTS:
            di, _, ti = paired(res[t]["info"][s], base)
            dr, _, tr = paired(res[t]["rand"][s], base)
            dn, sn, tn = paired(res[t]["info"][s], res[t]["rand"][s])
            star = "  <-- 显著" if abs(tn) > 2 else ""
            print(f"    {s:>8.1f}{di:>+15.4f}(t={ti:+4.1f}){dr:>+12.4f}(t={tr:+4.1f})"
                  f"{dn:>+11.4f}(t={tn:+4.1f}){star}")
        print()

    print("""  读法：最后一列「净结构增益」= 含信息 anchor 减去随机标签 anchor，
  已经扣掉「多了一批远处数据」本身的副作用，剩下的是真正被用上的信息。

  若该列随平移量迅速衰减到零，说明上下文是一个**局部**证据库：
  放在邻域之外的信息会被丢弃，「全局先验掰动」这条路在当前表格 PFN 上走不通。""")


if __name__ == "__main__":
    report(run())
