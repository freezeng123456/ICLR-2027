"""评估三件事：
  A. PFN 是不是「它自己先验下的解析后验」的化身
  B. 数据落在先验支撑之外时，它错得有没有方向（可预测）
  C. 在远离查询点的位置放 anchor 样本，能不能把它的先验掰动，
     且移动方向与解析解的预测一致

B 要双向测。只测「函数比先验更波动」这一侧会被一个假象骗到：
剧烈波动函数的正确答案几乎处处为 0，于是一个什么都没学会、把什么都预测成 0 的模型
反而显得「最接近正确答案」。所以必须同时测「函数比先验更平滑」那一侧，
那一侧的正确答案是非平凡的，并且要把「全预测 0」作为对照一起列出来。

C 的设计要点：anchor 全部放在离查询点很远的区域。在真实长度尺度下，
那个距离上的函数相关性实际为零——anchor 对查询点的函数值不含任何信息，
只携带「这个函数有多波动」这一全局信息。任何基于近邻检索的方法都会把它们
当无关样本丢掉。如果 PFN 的预测仍被改变，起作用的就只能是先验被掰动。

两种 anchor 方向相反，构成双向对照：
  truthful = 该函数在远处的真实观测（波动）  -> 应把先验掰向「小长度尺度」
  smooth   = 人为构造的平滑假点（不是真实数据）-> 应把先验掰向「大长度尺度」
"""

import numpy as np
import torch

from exp_gp_prior_tilt import (ELL_GRID_P, NOISE, PFN, gp_posterior, mixture_posterior,
                               sample_gp)

N_TRIALS = 300
N_Q, N_LOCAL, N_ANCHOR = 10, 5, 20
Q_LO, Q_HI = 1.2, 3.0
A_LO, A_HI = -3.0, -1.2
ELL_TRUE_C = 0.35          # C 部分真实长度尺度：在先验支撑内，但本地上下文太稀疏、认不出来
MISSPEC = [("比先验更平滑", 8.0), ("比先验更波动", 0.08)]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def pfn_predict(model, xc, yc, xq):
    x = torch.tensor(np.concatenate([xc, xq], 1), dtype=torch.float32)
    y = torch.tensor(np.concatenate([yc, np.zeros_like(xq)], 1), dtype=torch.float32)
    with torch.no_grad():
        mu, _ = model(x, y, xc.shape[1])
    return mu.numpy()


def banner(t):
    print("\n" + "=" * 76 + f"\n{t}\n" + "=" * 76)


def sample_eval_set(rng, ell, n=30, n_ctx=20):
    XC, YC, XQ, mix, tru = [], [], [], [], []
    for _ in range(N_TRIALS):
        e = rng.choice(ELL_GRID_P) if ell is None else ell
        x = np.sort(rng.uniform(-3, 3, n))
        f = sample_gp(rng, x, e) + NOISE * rng.standard_normal(n)
        xc, yc, xq = x[:n_ctx], f[:n_ctx], x[n_ctx:]
        XC.append(xc); YC.append(yc); XQ.append(xq)
        mix.append(mixture_posterior(xc, yc, xq, ELL_GRID_P)[0])
        tru.append(gp_posterior(xc, yc, xq, e)[0])
    return (np.array(XC), np.array(YC), np.array(XQ), np.array(mix), np.array(tru))


def main():
    model = PFN()
    model.load_state_dict(torch.load("pfn_gp.pt", map_location="cpu"))
    model.eval()
    rng = np.random.default_rng(123)

    # ------------------------------------------------------------------ A
    banner("A. PFN 是否等于「它自己先验下的解析后验」（数据来自先验 P）")
    XC, YC, XQ, mix, tru = sample_eval_set(rng, None)
    pred = pfn_predict(model, XC, YC, XQ)
    refs = {
        "P 下的混合后验（理论答案）": mix,
        "用真实长度尺度算的后验": tru,
        "固定 ell=1.0 的后验": np.array([gp_posterior(XC[i], YC[i], XQ[i], 1.0)[0]
                                        for i in range(N_TRIALS)]),
        "全预测 0（平凡基线）": np.zeros_like(mix),
    }
    print("\n  PFN 的预测与各个解析参照的均方根差（越小 = PFN 越接近它）：\n")
    for k, v in refs.items():
        print(f"    {k:<28}{rmse(pred, v):.4f}")
    best = min(refs, key=lambda k: rmse(pred, refs[k]))
    print(f"\n  最接近的是：{best}")
    print("  若是「P 下的混合后验」，就证实了 PFN 是它自己先验下那个最优预测的化身。")

    # ------------------------------------------------------------------ B
    banner("B. 数据落在先验支撑之外时，它错向哪一边")
    print(f"\n  先验 P 的长度尺度支撑是 [{ELL_GRID_P[0]:.2f}, {ELL_GRID_P[-1]:.2f}]。"
          f"下面两行的真实长度尺度都在支撑之外。\n")
    print(f"    {'错配方向':<14}{'真实ell':>8}{'PFN↔P后验':>12}{'PFN↔正确答案':>14}"
          f"{'P后验↔正确答案':>15}{'全0↔正确答案':>14}")
    print("    " + "-" * 77)
    for name, ell_q in MISSPEC:
        XC, YC, XQ, mix, tru = sample_eval_set(rng, ell_q)
        pred = pfn_predict(model, XC, YC, XQ)
        print(f"    {name:<14}{ell_q:>8.2f}{rmse(pred, mix):>12.4f}{rmse(pred, tru):>14.4f}"
              f"{rmse(mix, tru):>15.4f}{rmse(0 * tru, tru):>14.4f}")
    print("""
    读法：若「PFN↔P后验」显著小于「PFN↔正确答案」，说明它算的是自己先验下的答案，
    而不是真实机制下的答案——错误是系统性的、有方向的。
    最后一列是「什么都不预测」的成绩，用来确认这个比较本身不平凡。""")

    # ------------------------------------------------------------------ C
    banner("C. 用远处的 anchor 掰动先验")
    XL, YL, XQ = [], [], []
    anc = {"truthful": [], "smooth": []}
    for _ in range(N_TRIALS):
        xq = np.sort(rng.uniform(Q_LO, Q_HI, N_Q))
        xl = np.sort(rng.uniform(Q_LO, Q_HI, N_LOCAL))
        xa = np.sort(rng.uniform(A_LO, A_HI, N_ANCHOR))
        allx = np.concatenate([xl, xq, xa])
        f = sample_gp(rng, allx, ELL_TRUE_C) + NOISE * rng.standard_normal(len(allx))
        XL.append(xl); YL.append(f[:N_LOCAL]); XQ.append(xq)
        anc["truthful"].append((xa, f[N_LOCAL + N_Q:]))
        anc["smooth"].append((xa, sample_gp(rng, xa, ELL_GRID_P[-1])))  # 人造，非真实观测
    XL, YL, XQ = np.array(XL), np.array(YL), np.array(XQ)

    pred_none = pfn_predict(model, XL, YL, XQ)
    base = [mixture_posterior(XL[i], YL[i], XQ[i], ELL_GRID_P) for i in range(N_TRIALS)]
    ana_none = np.array([b[0] for b in base])
    w_none = np.mean([b[2] for b in base], 0)
    ell_none = float(np.exp(w_none @ np.log(ELL_GRID_P)))

    gap = (Q_LO + Q_HI) / 2 - (A_LO + A_HI) / 2
    print(f"""
  布局：查询点 {N_Q} 个在 x∈[{Q_LO},{Q_HI}]，本地上下文 {N_LOCAL} 个也在那里（很稀疏），
        anchor {N_ANCHOR} 个全部放在 x∈[{A_LO},{A_HI}]，与查询区中心相隔 {gap:.1f}。
  真实长度尺度 {ELL_TRUE_C}，在这个距离上的函数相关性是
  {np.exp(-0.5 * gap ** 2 / ELL_TRUE_C ** 2):.1e}——实际为零。

  也就是说 anchor 对「查询点的函数值是多少」不含任何信息，近邻检索会把它们全丢掉。
  能起作用的只剩「先验被掰动」这一条路径。
""")
    results = {}
    for name, alist in anc.items():
        XA = np.array([a[0] for a in alist])
        YA = np.array([a[1] for a in alist])
        xc, yc = np.concatenate([XL, XA], 1), np.concatenate([YL, YA], 1)
        pred_with = pfn_predict(model, xc, yc, XQ)
        got = [mixture_posterior(xc[i], yc[i], XQ[i], ELL_GRID_P) for i in range(N_TRIALS)]
        ana_with = np.array([g[0] for g in got])
        w_with = np.mean([g[2] for g in got], 0)
        ell_with = float(np.exp(w_with @ np.log(ELL_GRID_P)))

        d_pfn, d_ana = (pred_with - pred_none).ravel(), (ana_with - ana_none).ravel()
        slope = float(np.polyfit(d_ana, d_pfn, 1)[0])
        corr = float(np.corrcoef(d_ana, d_pfn)[0, 1])
        results[name] = dict(corr=corr, slope=slope, ell=ell_with,
                             mp=float(np.abs(d_pfn).mean()), ma=float(np.abs(d_ana).mean()))

        tag = {"truthful": "真实观测（波动）", "smooth": "人造平滑假点"}[name]
        arrow = "变小 = 先验被掰得更不平滑" if ell_with < ell_none else "变大 = 先验被掰得更平滑"
        print(f"  --- anchor 类型：{tag} ---")
        print(f"    先验隐含的长度尺度   加之前 {ell_none:.3f}  ->  加之后 {ell_with:.3f}   ({arrow})")
        print(f"    解析解预测的移动 vs PFN 实际的移动："
              f"相关 {corr:+.3f}   斜率 {slope:+.3f}   (斜率 1.0 = 完全跟随理论)")
        print(f"    平均移动幅度   PFN {np.abs(d_pfn).mean():.4f}   解析解 {np.abs(d_ana).mean():.4f}\n")

    banner("结论")
    t, s = results["truthful"], results["smooth"]
    opposite = (t["ell"] - ell_none) * (s["ell"] - ell_none) < 0
    strong = t["corr"] > 0.3 and s["corr"] > 0.3
    print(f"""
  C 的三个判据：
    1. 两种 anchor 是否把先验掰向相反方向     {'是' if opposite else '否'}
       （{ell_none:.3f} -> 真实观测 {t['ell']:.3f} / 人造假点 {s['ell']:.3f}）
    2. PFN 的移动是否与解析解同向且相关       真实观测 {t['corr']:+.3f}，人造假点 {s['corr']:+.3f}
    3. 移动幅度是否与理论量级相当             真实观测 {t['mp']:.4f} vs {t['ma']:.4f}，
                                              人造假点 {s['mp']:.4f} vs {s['ma']:.4f}

  机制{'成立' if (opposite and strong) else '未通过'}。""")
    print("  " + ("上下文样本确实起到了掰动先验的作用，方向与解析解一致，且这条路径与近邻检索无关。"
                 if (opposite and strong) else
                 "移动方向或强度与理论不符，需要进一步检查再决定是否继续这个方向。"))


if __name__ == "__main__":
    main()
