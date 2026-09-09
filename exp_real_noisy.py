"""真实数据 + 带噪训练标签：先验掰动在这里能不能真的派上用场。

上一轮发现真实数据集上模型本就校准良好（ECE 约 0.05），没什么可修，
掰动只剩破坏。但工业场景里训练标签有噪声几乎是常态，而标签噪声正是
让模型过度自信的典型原因。

这里给**上下文行**注入标签噪声（测试标签保持干净，对应「你的训练标注有错」
这个真实情形），再看远置 anchor 的噪声能否把置信度拉回正确水平。

  预期：噪声率越高，基准越过度自信，噪声 anchor 的收益越大。
  若成立，这个杠杆就有了真实的用武之地；若不成立，它只是一个漏洞。
"""

import logging
import pickle
import sys
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from exp_real_tilt import ece, get_model, make_anchor

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

N_CTX, N_TEST, N_ANCHOR = 50, 500, 60
CTX_NOISES = [0.0, 0.20, 0.40]     # 注入到上下文标签里的噪声
ANCHOR_ETAS = [0.0, 0.50]
N_SEEDS = 3


def run(which, data):
    model = get_model(which)
    rows = []
    for name, (X0, y0) in data.items():
        n_class = int(y0.max()) + 1
        labels = list(range(n_class))
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(6000 + seed)
            idx = rng.permutation(len(y0))
            tr, te = idx[:N_CTX], idx[N_CTX:N_CTX + N_TEST]
            mu, sd = X0[tr].mean(0), X0[tr].std(0) + 1e-9
            Xtr, Xte = (X0[tr] - mu) / sd, (X0[te] - mu) / sd
            yte = y0[te]

            for rho in CTX_NOISES:
                ytr = y0[tr].copy()
                flip = rng.random(N_CTX) < rho
                ytr[flip] = rng.integers(0, n_class, flip.sum())
                if len(np.unique(ytr)) < 2:
                    continue

                def measure(xc, yc, tag):
                    model.fit(xc, yc)
                    p = np.clip(model.predict_proba(Xte), 1e-6, 1 - 1e-6)
                    if p.shape[1] < n_class:
                        q = np.full((len(p), n_class), 1e-6)
                        q[:, np.unique(yc)] = p
                        p = q
                    p = p / p.sum(1, keepdims=True)
                    rows.append(dict(dataset=name, seed=seed, rho=rho, cond=tag,
                                     acc=accuracy_score(yte, p.argmax(1)),
                                     conf=float(p.max(1).mean()),
                                     ll=log_loss(yte, p, labels=labels),
                                     ece=ece(p, yte)))

                measure(Xtr, ytr, "base")
                for e in ANCHOR_ETAS:
                    Xa, ya = make_anchor(rng, Xtr.shape[1], n_class, e)
                    measure(np.vstack([Xtr, Xa]), np.concatenate([ytr, ya]), f"eta{e}")
        print(f"    {which} / {name} 完成", flush=True)
    return rows


def paired(a, b):
    d = np.array(a) - np.array(b)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), se, (d.mean() / se if se > 0 else 0.0)


def report(which):
    with open(f"noisy_{which}.pkl", "rb") as f:
        rows = pickle.load(f)
    tab = {(r["dataset"], r["seed"], r["rho"], r["cond"]): r for r in rows}
    print(f"\n  ===== {which} =====")
    print(f"    {'上下文标签噪声':>14}{'条件':>12}{'准确率':>10}{'置信度':>10}"
          f"{'过度自信':>10}{'log loss':>11}{'相对基准':>15}{'ECE':>9}")
    for rho in CTX_NOISES:
        keys = sorted({(d, s) for d, s, r, c in tab if r == rho and c == "base"})

        def col(cond, m):
            return [tab[(d, s, rho, cond)][m] for d, s in keys if (d, s, rho, cond) in tab]

        for cond, tag in [("base", "不加 anchor")] + \
                         [(f"eta{e}", f"anchor噪声{int(e * 100)}%") for e in ANCHOR_ETAS]:
            if not col(cond, "ll"):
                continue
            acc, conf = np.mean(col(cond, "acc")), np.mean(col(cond, "conf"))
            ll, ec = np.mean(col(cond, "ll")), np.mean(col(cond, "ece"))
            if cond == "base":
                extra = "        (基准)"
            else:
                d, _, t = paired(col(cond, "ll"), col("base", "ll"))
                extra = f"{d:>+9.4f}(t={t:+5.1f})"
            head = f"{int(rho * 100)}%" if cond == "base" else ""
            print(f"    {head:>14}{tag:>12}{acc:>10.4f}{conf:>10.4f}"
                  f"{conf - acc:>+10.4f}{ll:>11.4f}{extra:>15}{ec:>9.4f}")
        print()


if __name__ == "__main__":
    if sys.argv[1] == "report":
        print("\n" + "=" * 96)
        print("真实数据 + 带噪训练标签：掰动能否派上用场")
        print("=" * 96)
        for w in ("TabICL", "TabDPT"):
            try:
                report(w)
            except FileNotFoundError:
                print(f"  {w} 结果尚未生成")
    else:
        with open("datasets.pkl", "rb") as f:
            data = pickle.load(f)
        which = sys.argv[1]
        print(f"  {which}：{len(data)} 个数据集 x {len(CTX_NOISES)} 个噪声水平", flush=True)
        rows = run(which, data)
        with open(f"noisy_{which}.pkl", "wb") as f:
            pickle.dump(rows, f)
        print(f"  已保存 noisy_{which}.pkl（{len(rows)} 条）")
