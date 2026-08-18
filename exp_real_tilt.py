"""在真实数据集上复现噪声轴的先验掰动，并在两个先验来源不同的 PFN 上对照。

  TabICLv2  纯合成先验（SCM 生成器）
  TabDPT    在真实数据上预训练

anchor 一律远置（标准化空间里整体平移 6 个标准差），标签由与目标数据集
毫无关系的一条随机阈值规则生成，再按噪声率 eta 打乱。因此 anchor
不含任何关于目标任务的信息，只携带「这批数据有多吵」这一全局属性。

三个待检验的命题：
  1. 置信度是否随 anchor 噪声单调下降（掰动是否在真实数据上复现）
  2. 两个 PFN 是否都出现（是否与先验来源无关）
  3. 增益是否集中在基准就过度自信的数据集上（能否事先预测何时该用）
"""

import logging
import pickle
import sys
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

N_CTX, N_TEST, N_ANCHOR = 50, 500, 60
FAR_SHIFT = 6.0
ETAS = [0.0, 0.25, 0.50]
N_SEEDS = 5


def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1)
    ok = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return e


def make_anchor(rng, d, n_class, eta):
    """远置 anchor：标准化空间里平移 6 个标准差，标签来自任意阈值规则再加噪。"""
    X = rng.normal(0, 1, (N_ANCHOR, d))
    i = rng.permutation(d)
    y = ((X[:, i[0]] > 0.2) & (X[:, i[1 % d]] > -0.3)).astype(int)
    y = np.where(rng.random(N_ANCHOR) < eta, 1 - y, y)
    return X + FAR_SHIFT, np.clip(y, 0, n_class - 1)


def get_model(which):
    if which == "TabICL":
        from tabicl import TabICLClassifier
        return TabICLClassifier(device="cpu")
    from tabdpt import TabDPTClassifier
    return TabDPTClassifier(device="cpu")


def run(which, data):
    model = get_model(which)
    rows = []
    for name, (X0, y0) in data.items():
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(4000 + seed)
            idx = rng.permutation(len(y0))
            tr, te = idx[:N_CTX], idx[N_CTX:N_CTX + N_TEST]
            if len(np.unique(y0[tr])) < 2:
                continue
            mu, sd = X0[tr].mean(0), X0[tr].std(0) + 1e-9
            Xtr, Xte = (X0[tr] - mu) / sd, (X0[te] - mu) / sd
            ytr, yte = y0[tr], y0[te]
            n_class = int(y0.max()) + 1
            labels = list(range(n_class))

            def measure(xc, yc, tag):
                model.fit(xc, yc)
                p = np.clip(model.predict_proba(Xte), 1e-6, 1 - 1e-6)
                p = p / p.sum(1, keepdims=True)
                if p.shape[1] < n_class:            # 上下文里缺某些类时补齐
                    q = np.full((len(p), n_class), 1e-6)
                    q[:, np.unique(yc)] = p
                    p = q / q.sum(1, keepdims=True)
                rows.append(dict(dataset=name, seed=seed, cond=tag,
                                 acc=accuracy_score(yte, p.argmax(1)),
                                 conf=float(p.max(1).mean()),
                                 ll=log_loss(yte, p, labels=labels),
                                 ece=ece(p, yte)))

            measure(Xtr, ytr, "base")
            for e in ETAS:
                Xa, ya = make_anchor(rng, Xtr.shape[1], n_class, e)
                measure(np.vstack([Xtr, Xa]), np.concatenate([ytr, ya]), f"eta{e}")
        print(f"    {which} / {name} 完成", flush=True)
    return rows


if __name__ == "__main__":
    with open("datasets.pkl", "rb") as f:
        data = pickle.load(f)
    which = sys.argv[1] if len(sys.argv) > 1 else "TabICL"
    print(f"  在 {len(data)} 个真实数据集上跑 {which}"
          f"（{N_CTX} 行上下文 / {N_TEST} 行测试 / {N_ANCHOR} 行远置 anchor）", flush=True)
    rows = run(which, data)
    with open(f"real_{which}.pkl", "wb") as f:
        pickle.dump(rows, f)
    print(f"  已保存 real_{which}.pkl（{len(rows)} 条记录）")
