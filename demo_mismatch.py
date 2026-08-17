"""先验错配的「旋钮」实验：让数据生成机制从「符合 PFN 先验」连续滑向「违背 PFN 先验」，
观察 PFN 相对树模型的优势如何被吃掉。这就是论文主图的雏形。

alpha = 1  标签完全由平滑的因果组合决定  -> 和 PFN 预训练时的假设同族
alpha = 0  标签完全由阈值规则的布尔组合决定 -> 处处不连续，不在它的假设里
中间是两者的混合。树模型没有平滑先验，在这里充当参照系。
"""

import logging
import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from tabicl import TabICLClassifier

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

N_TRAIN, N_TEST, N_FEAT, N_SEEDS = 150, 400, 6, 8
ALPHAS = [1.0, 0.85, 0.7, 0.5, 0.3, 0.15, 0.0]


def gen(rng, n, alpha):
    X = rng.normal(0, 1, (n, N_FEAT))

    # 平滑成分：随机权重的连续非线性组合，处处可导
    w = rng.normal(0, 1, N_FEAT)
    smooth = np.tanh(X @ w) + 0.5 * np.sin(X[:, 0] * 1.5)

    # 不连续成分：多条硬阈值规则的布尔组合，处处是断崖
    i = rng.permutation(N_FEAT)
    rules = ((X[:, i[0]] > 0.3).astype(float)
             + (X[:, i[1]] > -0.2).astype(float)
             - (X[:, i[2]] > 0.5).astype(float)
             + ((X[:, i[3]] > 0) & (X[:, i[4]] > 0)).astype(float))

    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    score = alpha * z(smooth) + (1 - alpha) * z(rules) + rng.normal(0, 0.45, n)
    return X, (score > np.median(score)).astype(int)


pfn = TabICLClassifier(device="cpu")
print(f"训练 {N_TRAIN} 行 / 测试 {N_TEST} 行 / {N_FEAT} 个特征 / 每点 {N_SEEDS} 个种子\n")
print(f"{'旋钮 alpha':>10}  {'数据性质':<22}{'PFN':>8}{'梯度提升树':>11}{'差值':>9}")
print("-" * 66)

rows = []
for alpha in ALPHAS:
    a_pfn, a_gbdt = [], []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(1000 + seed)
        X, y = gen(rng, N_TRAIN + N_TEST, alpha)
        Xtr, ytr, Xte, yte = X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]
        pfn.fit(Xtr, ytr)
        a_pfn.append(accuracy_score(yte, pfn.predict(Xte)))
        g = GradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
        a_gbdt.append(accuracy_score(yte, g.predict(Xte)))

    mp, mg = np.mean(a_pfn), np.mean(a_gbdt)
    desc = {1.0: "完全平滑（先验匹配）", 0.0: "完全规则（先验错配）"}.get(alpha, "混合")
    print(f"{alpha:>10.2f}  {desc:<20}{mp:>8.3f}{mg:>11.3f}{mp - mg:>+9.3f}")
    rows.append((alpha, mp, mg))

print("\n把差值画成条形图（PFN 减去树模型，向右为 PFN 领先）：\n")
for alpha, mp, mg in rows:
    d = mp - mg
    n = int(round(abs(d) * 300))
    bar = (" " * (30 - n) + "#" * n + "|") if d < 0 else (" " * 30 + "|" + "#" * n)
    print(f"  alpha={alpha:.2f} {bar} {d:+.3f}")
print(" " * 22 + "^ 零线")
