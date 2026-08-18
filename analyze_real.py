"""汇总真实数据集上的结果，检验三个命题：
  1. 置信度是否随 anchor 噪声单调下降（掰动是否在真实数据上复现）
  2. 两个先验来源不同的 PFN 是否都出现
  3. 增益是否集中在基准就过度自信的数据集上（能否事先预测何时该用）
"""

import pickle

import numpy as np

ETAS = [0.0, 0.25, 0.50]


def load(which):
    with open(f"real_{which}.pkl", "rb") as f:
        return pickle.load(f)


def by_key(rows):
    d = {}
    for r in rows:
        d[(r["dataset"], r["seed"], r["cond"])] = r
    return d


def paired(vals_a, vals_b):
    d = np.array(vals_a) - np.array(vals_b)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), se, (d.mean() / se if se > 0 else 0.0)


def analyze(which):
    rows = load(which)
    tab = by_key(rows)
    keys = sorted({(r["dataset"], r["seed"]) for r in rows})
    keys = [k for k in keys if (k[0], k[1], "base") in tab]

    def col(cond, metric):
        return [tab[(d, s, cond)][metric] for d, s in keys if (d, s, cond) in tab]

    print(f"\n  ===== {which} =====")
    print(f"    {len(keys)} 个 (数据集, 种子) 组合，"
          f"{len({k[0] for k in keys})} 个数据集")
    base = {m: np.mean(col("base", m)) for m in ("acc", "conf", "ll", "ece")}
    print(f"    基准：准确率 {base['acc']:.4f}  置信度 {base['conf']:.4f}  "
          f"log loss {base['ll']:.4f}  ECE {base['ece']:.4f}")
    print(f"    {'anchor 噪声':>12}{'准确率':>10}{'置信度':>12}{'相对基准':>16}"
          f"{'log loss':>11}{'ECE':>9}")
    confs = []
    for e in ETAS:
        c = f"eta{e}"
        dc, _, tc = paired(col(c, "conf"), col("base", "conf"))
        confs.append(np.mean(col(c, "conf")))
        print(f"    {int(e * 100):>10}%{np.mean(col(c, 'acc')):>10.4f}"
              f"{np.mean(col(c, 'conf')):>12.4f}{dc:>+10.4f}(t={tc:+5.1f})"
              f"{np.mean(col(c, 'll')):>11.4f}{np.mean(col(c, 'ece')):>9.4f}")
    d_tilt, se, t = paired(col(f"eta{ETAS[-1]}", "conf"), col(f"eta{ETAS[0]}", "conf"))
    mono = all(confs[i] >= confs[i + 1] - 1e-4 for i in range(len(confs) - 1))
    print(f"    置信度 eta=0 -> eta={ETAS[-1]}：{d_tilt:+.4f} ± {se:.4f} "
          f"(t={t:+.1f})，{'单调下降' if mono else '非单调'}")

    # 命题 3：按数据集看，基准的过度自信程度能否预测增益
    ds = sorted({k[0] for k in keys})
    over, gain = [], []
    for name in ds:
        sel = [(d, s) for d, s in keys if d == name]
        b_conf = np.mean([tab[(d, s, "base")]["conf"] for d, s in sel])
        b_acc = np.mean([tab[(d, s, "base")]["acc"] for d, s in sel])
        b_ll = np.mean([tab[(d, s, "base")]["ll"] for d, s in sel])
        best = min(np.mean([tab[(d, s, f"eta{e}")]["ll"] for d, s in sel]) for e in ETAS)
        over.append(b_conf - b_acc)
        gain.append(b_ll - best)
    r = float(np.corrcoef(over, gain)[0, 1])
    print(f"    过度自信程度(置信度−准确率) 与 log loss 增益 的相关：{r:+.3f}"
          f"   （{len(ds)} 个数据集）")
    top = np.argsort(over)[::-1][:4]
    print(f"    最过度自信的 4 个数据集：" +
          "  ".join(f"{ds[i]}(过度{over[i]:+.3f},增益{gain[i]:+.3f})" for i in top))
    bot = np.argsort(over)[:4]
    print(f"    最不过度自信的 4 个：" +
          "  ".join(f"{ds[i]}(过度{over[i]:+.3f},增益{gain[i]:+.3f})" for i in bot))
    return dict(tilt=d_tilt, t=t, mono=mono, corr=r)


if __name__ == "__main__":
    print("\n" + "=" * 84)
    print("真实数据集上的先验掰动：两个先验来源不同的 PFN")
    print("=" * 84)
    res = {}
    for which in ("TabICL", "TabDPT"):
        try:
            res[which] = analyze(which)
        except FileNotFoundError:
            print(f"\n  {which} 结果尚未生成")
    if len(res) == 2:
        a, b = res["TabICL"], res["TabDPT"]
        print(f"""
  =====小结=====
    命题 1 掰动在真实数据上复现        TabICL {a['tilt']:+.4f}(t={a['t']:+.1f})
                                       TabDPT {b['tilt']:+.4f}(t={b['t']:+.1f})
    命题 2 与先验来源无关              {'两者都出现' if a['t'] < -2 and b['t'] < -2 else '并非两者都出现'}
           （TabICL 纯合成先验，TabDPT 在真实数据上预训练）
    命题 3 过度自信可预测增益          TabICL 相关 {a['corr']:+.3f}，TabDPT 相关 {b['corr']:+.3f}""")
