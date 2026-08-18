"""把「构造 anchor 掰动先验」搬到真实的表格 PFN（TabICLv2）上。

没有解析解可以对答案，所以改用一个信息上封死的对照设计：

  anchor 行的标签来自一条与目标任务**完全不同**的规则，因此不含任何关于
  目标任务答案的信息；它们唯一携带的是「这个任务是什么结构」——
  硬阈值规则型，还是平滑型。

  两种 anchor 的行数、特征分布、放置位置全部相同，只有生成标签的结构不同。
  归一化、上下文长度等副作用因此被完全抵消，剩下的差异只能来自结构信息。

anchor 一律放在特征空间的远处。这既让近邻检索必然把它们丢弃，
也是机制成立的前提——放在测试点附近的话，来自别的规则的标签会直接构成矛盾监督
（先导实验里近处 anchor 使准确率下降 8 到 19 个百分点）。

对照组：
  随机标签 anchor(远)  同样位置同样行数，但标签纯随机 -> 结构信息为零
  额外真实数据        同样行数的真实样本，来自目标任务本身 -> 增益的上界参照

统计上做配对：同一个种子下各条件共用同一份数据，只比较条件之间的差，
这样种子间的方差被消掉，小效应也能测出来。
"""

import logging
import warnings

import numpy as np
from sklearn.metrics import accuracy_score

from tabicl import TabICLClassifier

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

D = 8
N_CTX, N_TEST, N_ANCHOR = 30, 400, 60      # 上下文压到 30 行，留出提升空间
LABEL_NOISE = 0.08
FAR_SHIFT = 6.0
N_SEEDS = 60


def make_rule_labeler(rng):
    """硬阈值规则：处处不连续。"""
    i = rng.permutation(D)
    t = rng.normal(0, 0.4, 3)

    def f(X):
        return (((X[:, i[0]] > t[0]) & (X[:, i[1]] > t[1])) | (X[:, i[2]] > t[2] + 0.8)).astype(int)

    return f


def make_smooth_labeler(rng):
    """平滑非线性：处处可导。"""
    w, v = rng.normal(0, 1, D), rng.normal(0, 1, D)

    def f(X):
        s = np.tanh(X @ w) + 0.6 * np.sin(X @ v)
        return (s > np.median(s)).astype(int)

    return f


LABELERS = {"规则型": make_rule_labeler, "平滑型": make_smooth_labeler}
CONDS = ["不加 anchor", "规则型 anchor", "平滑型 anchor", "随机标签 anchor", "额外真实数据"]


def noisy(y, rng):
    flip = rng.random(len(y)) < LABEL_NOISE
    return np.where(flip, 1 - y, y)


def far_anchor(rng, kind):
    """远处 anchor：特征整体平移，标签由指定结构的**另一条**规则生成。"""
    X = rng.normal(0, 1, (N_ANCHOR, D))
    if kind == "随机标签":
        y = rng.integers(0, 2, N_ANCHOR)
    else:
        y = LABELERS[kind](rng)(X)
    return X + FAR_SHIFT, y


def run():
    pfn = TabICLClassifier(device="cpu")
    per_seed = {t: {c: [] for c in CONDS} for t in LABELERS}
    dists = {"真实上下文": [], "远 anchor": []}

    for target in LABELERS:
        done = 0
        seed = 0
        while done < N_SEEDS:
            seed += 1
            rng = np.random.default_rng(50_000 + seed)
            labeler = LABELERS[target](rng)
            X = rng.normal(0, 1, (N_CTX + N_TEST + N_ANCHOR, D))
            y = noisy(labeler(X), rng)
            Xtr, ytr = X[:N_CTX], y[:N_CTX]
            Xte, yte = X[N_CTX:N_CTX + N_TEST], y[N_CTX:N_CTX + N_TEST]
            Xex, yex = X[N_CTX + N_TEST:], y[N_CTX + N_TEST:]
            if len(np.unique(ytr)) < 2 or ytr.mean() < 0.15 or ytr.mean() > 0.85:
                continue

            for cond in CONDS:
                if cond == "不加 anchor":
                    xc, yc = Xtr, ytr
                elif cond == "额外真实数据":
                    xc, yc = np.vstack([Xtr, Xex]), np.concatenate([ytr, yex])
                else:
                    kind = cond.replace(" anchor", "")
                    Xa, ya = far_anchor(rng, kind)
                    xc, yc = np.vstack([Xtr, Xa]), np.concatenate([ytr, ya])
                pfn.fit(xc, yc)
                per_seed[target][cond].append(accuracy_score(yte, pfn.predict(Xte)))

            if target == "规则型" and done < 5:
                Xa, _ = far_anchor(rng, "规则型")
                dists["真实上下文"].append(
                    np.linalg.norm(Xtr[:, None] - Xte[None, :30], axis=-1).min(0).mean())
                dists["远 anchor"].append(
                    np.linalg.norm(Xa[:, None] - Xte[None, :30], axis=-1).min(0).mean())
            done += 1
    return per_seed, dists


def paired(a, b):
    """配对差：均值、标准误、t 值。"""
    d = np.array(a) - np.array(b)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), se, d.mean() / se if se > 0 else 0.0


def report(per_seed, dists):
    print("\n" + "=" * 86)
    print("真实表格 PFN（TabICLv2）上的先验掰动实验")
    print("=" * 86)
    print(f"""
  {N_SEEDS} 个种子 / {N_CTX} 行真实上下文 / {N_ANCHOR} 行构造 anchor / {N_TEST} 行测试 /
  {D} 个特征 / {int(LABEL_NOISE * 100)}% 标签噪声。
  anchor 的标签来自与目标任务无关的另一条规则，不含任何答案信息。

  最近邻距离  真实上下文→测试点 {np.mean(dists['真实上下文']):.2f}   "
  远 anchor→测试点 {np.mean(dists['远 anchor']):.2f}   （检索类方法必然丢弃 anchor）
""")
    print(f"  {'条件':<20}" + "".join(f"{t + '目标':>22}" for t in LABELERS))
    print("  " + "-" * 64)
    for c in CONDS:
        row = f"  {c:<18}"
        for t in LABELERS:
            m = np.mean(per_seed[t][c])
            if c == "不加 anchor":
                row += f"{m:>13.4f} (基准)"
            else:
                dm, se, tv = paired(per_seed[t][c], per_seed[t]["不加 anchor"])
                row += f"{m:>11.4f} {dm:+.4f}(t={tv:+.1f})"
        print(row)

    print("\n  关键判据：结构匹配的 anchor 是否显著优于结构不匹配的 anchor（直接配对）\n")
    ok = True
    for t in LABELERS:
        other = [k for k in LABELERS if k != t][0]
        dm, se, tv = paired(per_seed[t][f"{t} anchor"], per_seed[t][f"{other} anchor"])
        verdict = "显著" if abs(tv) > 2 else "不显著"
        if not (dm > 0 and abs(tv) > 2):
            ok = False
        print(f"    {t}目标：结构匹配 − 结构不匹配 = {dm:+.4f} ± {se:.4f}  "
              f"(t = {tv:+.1f}，{verdict})")

        dr, _, tr = paired(per_seed[t][f"{t} anchor"], per_seed[t]["随机标签 anchor"])
        print(f"             结构匹配 − 随机标签   = {dr:+.4f}  (t = {tr:+.1f})")
        de, _, te = paired(per_seed[t]["额外真实数据"], per_seed[t]["不加 anchor"])
        print(f"             参照：{N_ANCHOR} 行额外真实数据带来 {de:+.4f}  (t = {te:+.1f})\n")

    print(f"  结论：机制在真实表格 PFN 上{'成立' if ok else '未通过'}。")
    if not ok:
        print("  （构造 anchor 未能带来可检测的、方向正确的增益，需要重新设计或另寻切入点）")


if __name__ == "__main__":
    ps, ds = run()
    report(ps, ds)
