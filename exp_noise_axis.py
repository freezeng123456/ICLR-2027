"""换一个轴再测先验掰动：噪声水平。

上一个实验说明「规则型 vs 平滑型」这个轴上传不进结构信息，但远处 anchor
本身是被读取的。所以问题可能出在轴选错了——「函数是否平滑」未必是表格 PFN
的先验里显式存在的隐变量，而「这批数据有多少噪声」几乎一定是。

设计成剂量反应：anchor 一律远置、标签来自与目标任务**无关**的另一个函数，
唯一变化的是 anchor 自身的标签噪声水平 η。若模型会从上下文推断一个全局的
噪声水平并据此调整预测，那么随着 η 增大，它在目标任务上应当变得更保守。

读数用两个：
  准确率        直接但迟钝
  平均置信度    直接反映后验的锐利程度，对先验的移动灵敏得多

若置信度随 η 单调下降，说明模型确实从上下文推断了一个任务层面的隐变量，
并把它应用到了预测上——那就是先验被掰动，方向对了，只是轴要换。
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
TARGET_NOISE = 0.08
FAR_SHIFT = 6.0
ETAS = [0.0, 0.10, 0.25, 0.40, 0.50]
N_SEEDS = 30


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


def run():
    pfn = TabICLClassifier(device="cpu")
    out = {t: {"base": {"acc": [], "conf": []},
               **{e: {"acc": [], "conf": []} for e in ETAS}} for t in LABELERS}

    for target in LABELERS:
        done, seed = 0, 0
        while done < N_SEEDS:
            seed += 1
            rng = np.random.default_rng(70_000 + seed)
            lab = LABELERS[target](rng)
            X = rng.normal(0, 1, (N_CTX + N_TEST, D))
            y = flip(lab(X), TARGET_NOISE, rng)
            Xtr, ytr, Xte, yte = X[:N_CTX], y[:N_CTX], X[N_CTX:], y[N_CTX:]
            if ytr.mean() < 0.15 or ytr.mean() > 0.85:
                continue

            def measure(xc, yc, bucket):
                pfn.fit(xc, yc)
                p = pfn.predict_proba(Xte)
                bucket["acc"].append(accuracy_score(yte, p.argmax(1)))
                bucket["conf"].append(p.max(1).mean())

            measure(Xtr, ytr, out[target]["base"])

            # anchor 的函数与目标任务无关；各 eta 共用同一批特征与同一个基础标签
            other = [k for k in LABELERS if k != target][0]
            g = LABELERS[other](rng)
            Xa0 = rng.normal(0, 1, (N_ANCHOR, D))
            ya0 = g(Xa0)
            Xa = Xa0 + FAR_SHIFT
            for e in ETAS:
                ya = flip(ya0, e, rng)
                measure(np.vstack([Xtr, Xa]), np.concatenate([ytr, ya]), out[target][e])
            done += 1
    return out


def paired(a, b):
    d = np.array(a) - np.array(b)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), se, (d.mean() / se if se > 0 else 0.0)


def report(out):
    print("\n" + "=" * 82)
    print("换轴再测：anchor 的噪声水平能否掰动模型对目标任务的判断")
    print("=" * 82)
    print(f"""
  {N_SEEDS} 个种子 / {N_CTX} 行真实上下文（噪声 {int(TARGET_NOISE * 100)}%）/
  {N_ANCHOR} 行远置 anchor / {N_TEST} 行测试 / {D} 个特征。
  anchor 的标签来自与目标任务无关的另一个函数，各 eta 之间只有噪声水平不同。
""")
    for t in LABELERS:
        b = out[t]["base"]
        print(f"  --- {t}目标 ---")
        print(f"    基准：准确率 {np.mean(b['acc']):.4f}   平均置信度 {np.mean(b['conf']):.4f}")
        print(f"    {'anchor 噪声':>12}{'准确率':>12}{'相对基准':>14}"
              f"{'平均置信度':>12}{'相对基准':>16}")
        for e in ETAS:
            da, _, ta = paired(out[t][e]["acc"], b["acc"])
            dc, _, tc = paired(out[t][e]["conf"], b["conf"])
            print(f"    {int(e * 100):>10}%{np.mean(out[t][e]['acc']):>12.4f}"
                  f"{da:>+10.4f}(t={ta:+4.1f}){np.mean(out[t][e]['conf']):>12.4f}"
                  f"{dc:>+10.4f}(t={tc:+5.1f})")
        confs = [np.mean(out[t][e]["conf"]) for e in ETAS]
        drop, _, td = paired(out[t][ETAS[-1]]["conf"], out[t][ETAS[0]]["conf"])
        mono = all(confs[i] >= confs[i + 1] - 1e-4 for i in range(len(confs) - 1))
        print(f"    置信度 从 eta=0 到 eta={ETAS[-1]}：{drop:+.4f} (t={td:+.1f})，"
              f"{'单调下降' if mono else '非单调'}\n")

    print("""  读法：若置信度随 anchor 噪声单调下降且幅度显著，说明模型从上下文里
  推断出了一个全局的噪声水平，并把它用到了与该 anchor 无关的目标任务上——
  这就是先验被掰动，只不过起作用的轴是噪声而不是平滑度。""")


if __name__ == "__main__":
    report(run())
