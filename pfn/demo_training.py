"""PFN 是怎么训的：样本从哪来、一步训练里发生什么、损失函数为什么是它。

全部用真实数字，不做任何新实验。
"""

import numpy as np
import torch
import torch.nn as nn

from exp_why_axis import (PFN, PRIORS, gp_posterior, log_marginal, mixture_posterior,
                          rbf, sample_gp)

np.set_printoptions(precision=3, suppress=True, linewidth=110)
torch.manual_seed(0)


def banner(t):
    print("\n" + "=" * 86 + f"\n{t}\n" + "=" * 86)


# =========================================================== 1. 样本从哪来
banner("1. 样本从哪来：没有数据集，每一步现场生成")

ELLS, NOISES = PRIORS["A(只有尺度变)"]
print(f"""
  模型 A 的先验 P 由两个东西定义：
    长度尺度候选 ell  {np.round(ELLS, 3)}
                      （12 个，在 0.3 到 3.0 之间对数均匀；小 = 函数抖，大 = 函数平缓）
    噪声候选          {NOISES}   （只有一个值，即 A 的先验里噪声是固定的）

  生成一个任务 = 掷两次骰子 + 采一条函数。下面把这个过程拆开走一遍。
""")

rng = np.random.default_rng(42)

# --- 第 1 步：掷骰子
ell = rng.choice(ELLS)
noise = rng.choice(NOISES)
print(f"  [第 1 步] 从候选里各抽一个：ell = {ell:.3f}，noise = {noise}")

# --- 第 2 步：随机取 x
n = 6                                   # 演示用 6 个点，真实训练是 40 个
x = np.sort(rng.uniform(-3, 3, n))
print(f"  [第 2 步] 在 [-3,3] 上均匀抽 {n} 个横坐标（真实训练是 40 个）")
print(f"           x = {x}")

# --- 第 3 步：算核矩阵
K = rbf(x, x, ell) + 1e-6 * np.eye(n)
print(f"""
  [第 3 步] 用 RBF 核算协方差矩阵 K，K[i,j] = exp(-0.5*(x_i-x_j)^2 / ell^2)
           ell 越大，远处的点也高度相关，函数就越平缓。这里 ell={ell:.3f}：""")
for row in np.round(K, 3):
    print(f"           {row}")

# --- 第 4 步：Cholesky × 标准正态 = 一条函数
L = np.linalg.cholesky(K)
z = rng.standard_normal(n)
f = L @ z
print(f"""
  [第 4 步] Cholesky 分解 K = L Lᵀ，再乘一个标准正态向量，得到一条函数样本
           z（标准正态）  = {z}
           f = L @ z      = {f}
           这一步等价于「从 N(0, K) 里采样」，是高斯过程采样的标准做法。""")

# --- 第 5 步：加观测噪声
eps = noise * rng.standard_normal(n)
y = f + eps
print(f"""
  [第 5 步] 加观测噪声 eps ~ N(0, {noise}^2)
           eps = {eps}
           y = f + eps = {y}

  这就是一个完整任务。**ell 和 noise 用完就扔，模型永远看不到。**
  代码原文（exp_why_axis.py 的 make_batch）：

      for b in range(bs):
          ell, noise = rng.choice(ells), rng.choice(noises)
          x = rng.uniform(-3, 3, N_POINTS)
          xs[b] = x
          ys[b] = sample_gp(rng, x, ell) + noise * rng.standard_normal(N_POINTS)

  **没有数据集，没有 train/val/test 划分，没有 epoch。** 每一步现采 48 个全新任务，
  15000 步就是 72 万个互不重复的任务。训练数据在数学上是无限的。
""")

# =========================================================== 2. 一步训练
banner("2. 一步训练里到底发生了什么")

model = PFN()
model.load_state_dict(torch.load("pfn_A.pt", map_location="cpu"))
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

BS, N_POINTS = 4, 40                     # 演示用 batch 4，真实训练是 48
rng = np.random.default_rng(1)
xs, ys = np.zeros((BS, N_POINTS)), np.zeros((BS, N_POINTS))
task_ells = []
for b in range(BS):
    e = rng.choice(ELLS)
    task_ells.append(e)
    xx = rng.uniform(-3, 3, N_POINTS)
    xs[b], ys[b] = xx, sample_gp(rng, xx, e) + 0.1 * rng.standard_normal(N_POINTS)
xt = torch.tensor(xs, dtype=torch.float32)
yt = torch.tensor(ys, dtype=torch.float32)

n_ctx = int(rng.integers(4, N_POINTS - 8))
print(f"""
  [a] 采一个 batch：{BS} 个任务 × {N_POINTS} 个点（真实训练 48 × 40）
      这 {BS} 个任务的隐变量各不相同：ell = {np.round(task_ells, 3)}

  [b] **随机**决定切分点 n_ctx = {n_ctx}
      前 {n_ctx} 个点当上下文，后 {N_POINTS - n_ctx} 个当查询。
      每一步都重新随机（代码：n_ctx = int(rng.integers(4, N_POINTS - 8))），
      这样模型在任意上下文长度下都可用；否则换个长度评估就不准了。
""")

mu, logv = model(xt, yt, n_ctx)
tgt = yt[:, n_ctx:]
per_point = 0.5 * (logv + (tgt - mu) ** 2 / logv.exp())
loss = per_point.mean()

print(f"""  [c] 前向：model(x, y, n_ctx) -> (mu, logv)，形状都是 {tuple(mu.shape)}
      = ({BS} 个任务, {N_POINTS - n_ctx} 个查询点)。每个查询点输出一个高斯的均值和 log 方差。

      第 0 个任务的前 5 个查询点：
        预测均值 mu    {mu[0, :5].detach().numpy()}
        预测标准差     {logv[0, :5].exp().sqrt().detach().numpy()}
        真实答案 y     {tgt[0, :5].numpy()}

  [d] 算损失（只在查询点上算，上下文点不参与）：
        逐点损失      {per_point[0, :5].detach().numpy()}
        整个 batch 平均 = {loss.item():.4f}
""")

opt.zero_grad()
loss.backward()
gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
with torch.no_grad():
    mu2, logv2 = model(xt, yt, n_ctx)
    loss2 = (0.5 * (logv2 + (tgt - mu2) ** 2 / logv2.exp())).mean()
print(f"""  [e] 反向 + 裁剪 + 更新：
        梯度范数（裁剪前） {gn:.3f}   上限 1.0（超了就按比例缩回去）
        走一步之后，同一个 batch 的损失 {loss.item():.4f} -> {loss2.item():.4f}

      注意这里损失**变差了**，这是演示的假象不是真实现象：我是在一个已经训好的
      模型上套了个全新的 AdamW，没有动量历史、也没有 OneCycle 的预热，直接用峰值
      学习率踹了一脚。真实训练是从随机初始化开始、带预热跑满 15000 步的。

  [f] 学习率调度：OneCycleLR，峰值 3e-4，前 10% 预热。优化器 AdamW，weight_decay 0.01。
      重复 15000 步，结束。
""")

# =========================================================== 3. 损失函数
banner("3. 损失函数是什么，以及为什么是它")

print("""
  公式（高斯负对数似然，省掉了与参数无关的常数 0.5*log(2*pi)）：

      L = E_task E_query [ 0.5 * ( log v + (y - mu)^2 / v ) ]

  代码一行：
      loss = (0.5 * (logv + (tgt - mu) ** 2 / logv.exp())).mean()

  两项各自的作用：
      (y - mu)^2 / v   预测不准就罚 —— 但可以靠把 v 调大来减轻
      log v            v 调大要付代价 —— 防止模型靠「我不确定」蒙混过关
  两项互相牵制，所以模型必须同时报对「答案是多少」和「我有多确定」。
  这也是为什么这个损失能训出校准，而不只是训出点估计。

  **损失是负数很正常。** 这是连续分布的负对数密度，不是概率；预测密度大于 1 时
  它就是负的。日志里 A 从 -0.92 收敛到 -1.18，就是密度越来越集中的意思。
""")

# 经验验证：谁的损失最低
banner("3b. 用数字验证「这个损失的最优解 = 先验 P 下的贝叶斯后验」")

model.eval()
N_TASK, N_CTX_E, N_TOT = 300, 20, 30
scores = {k: [] for k in ["PFN", "P 下的混合后验（理论最优）", "固定 ell=1.0 的后验",
                          "常数预测 N(0,1)", "上帝视角（用了特权信息）"]}
rng = np.random.default_rng(2026)


def nll(mu_, var_, y_):
    return float(np.mean(0.5 * (np.log(var_) + (y_ - mu_) ** 2 / var_)))


for _ in range(N_TASK):
    e = rng.choice(ELLS)
    xx = np.sort(rng.uniform(-3, 3, N_TOT))
    yy = sample_gp(rng, xx, e) + 0.1 * rng.standard_normal(N_TOT)
    xc, yc, xq, yq = xx[:N_CTX_E], yy[:N_CTX_E], xx[N_CTX_E:], yy[N_CTX_E:]

    xin = torch.tensor(np.concatenate([xc, xq])[None], dtype=torch.float32)
    yin = torch.tensor(np.concatenate([yc, np.zeros_like(xq)])[None], dtype=torch.float32)
    with torch.no_grad():
        m, lv = model(xin, yin, len(xc))
    scores["PFN"].append(nll(m[0].numpy(), lv[0].exp().numpy(), yq))

    mm, vv = mixture_posterior(xc, yc, xq, ELLS, NOISES)
    scores["P 下的混合后验（理论最优）"].append(nll(mm, vv, yq))
    mf, vf = gp_posterior(xc, yc, xq, 1.0, 0.1)
    scores["固定 ell=1.0 的后验"].append(nll(mf, vf, yq))
    scores["常数预测 N(0,1)"].append(nll(np.zeros_like(yq), np.ones_like(yq), yq))
    mt, vt = gp_posterior(xc, yc, xq, e, 0.1)
    scores["上帝视角（用了特权信息）"].append(nll(mt, vt, yq))

print(f"""
  在 {N_TASK} 个**全新**任务上算同一个损失（越低越好）。
  前四个都只能看到上下文；最后一个额外知道真实 ell，属于特权信息，不是公平对手。
""")
order = sorted(scores, key=lambda k: np.mean(scores[k]))
for k in order:
    star = "  <- 只看上下文的选手里最低" if k == "P 下的混合后验（理论最优）" else ""
    print(f"    {k:<28}{np.mean(scores[k]):>9.4f}{star}")

gap = np.mean(scores["PFN"]) - np.mean(scores["P 下的混合后验（理论最优）"])
print(f"""
  读法：
    「P 下的混合后验」在**只看上下文**的选手里损失最低——这不是巧合，
    它就是这个损失的理论最优解（在任务来自 P 的前提下）。
    PFN 比它高 {gap:+.4f}，这个差就是「摊销近似」的代价，不算小。
    「上帝视角」更低，但它偷看了 ell，不在同一个赛道上。

  顺带看一眼最后一名，它其实是这篇论文主张的一个缩影：
    「固定 ell=1.0」({np.mean(scores['固定 ell=1.0 的后验']):.4f}) 比「什么都不预测的常数」
    ({np.mean(scores['常数预测 N(0,1)']):.4f}) 还要差得多。原因是当真实函数很抖时，
    一个假设 ell=1.0 的 GP 会给出**又错又自信**的预测，而这个损失对自信的错误
    罚得极重。换成只看准确率/RMSE，它不会输得这么难看。
    **先验错了的代价主要记在校准上，而不是点估计上**——这正是整篇论文的论点。

  这就是整个 PFN 范式的地基：
      训练目标的最优解 = 先验 P 下的贝叶斯后验预测分布
      所以训出来的网络 ≈ 「在 P 下做贝叶斯推断」的一个摊销近似。
      它逼近的是 P 下的最优，不是真相。P 错了，它就跟着错，而且错得很自信。
""")

# =========================================================== 4. 训练曲线
banner("4. 实际的训练曲线（来自 train_why.log）")

print("""
  模型 A（12 档尺度、噪声固定）        模型 B（尺度固定、12 档噪声）
    2500 步   loss -0.9158              2500 步   loss -0.9428
    5000 步   loss -1.0933              5000 步   loss -1.0873
   10000 步   loss -1.1654             10000 步   loss -1.1874
   15000 步   loss -1.1802             15000 步   loss -1.2320
   用时 682 秒（CPU）                   用时 632 秒

  模型 C（12x12 = 144 种组合）
    2500 步   loss -0.6612
   15000 步   loss -1.0486        <- 明显高于 A/B，因为要同时分辨两个纠缠的隐变量
   45000 步   loss -1.0995（另跑）

  没有验证集，也不需要——每一步的数据都是没见过的，训练损失本身就是泛化损失的
  无偏估计。这是 PFN 相对普通监督学习的一个结构性便利。
""")
