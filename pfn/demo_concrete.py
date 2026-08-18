"""把 PFN 项目在做什么，用真实数字摊开给人看。

不做任何新实验，只是把已有代码里的数据、张量、损失、模型输出打印出来。

**运行位置**：这个脚本依赖 `exp_why_axis.py` 和训练好的 `pfn_A.pt`，
两者都在分支 `cursor/iclr-2027-research-direction-8ba3` 上。用法：

    git checkout cursor/iclr-2027-research-direction-8ba3
    cp <本分支>/pfn/demo_concrete.py .
    python3 demo_concrete.py

只需要 numpy 和 torch（CPU 即可），跑完约 2 秒。
本目录下的 demo_output.txt 是一份已经跑好的完整输出。
"""

import numpy as np
import torch

from exp_why_axis import PFN, PRIORS, gp_posterior, mixture_posterior, sample_gp

np.set_printoptions(precision=3, suppress=True, linewidth=120)


def banner(t):
    print("\n" + "=" * 88)
    print(t)
    print("=" * 88)


def predict_raw(model, xc, yc, xq_):
    """给一段上下文和一批查询点，返回 PFN 的预测均值与标准差。"""
    xx = torch.tensor(np.concatenate([xc, xq_])[None], dtype=torch.float32)
    yy = torch.tensor(np.concatenate([yc, np.zeros_like(xq_)])[None], dtype=torch.float32)
    with torch.no_grad():
        m, lv = model(xx, yy, len(xc))
    return m[0].numpy(), lv[0].exp().sqrt().numpy()


# ============================================================ 1. 一个任务长什么样
banner("1. 预训练时，PFN 看到的一个「任务」长什么样")

rng = np.random.default_rng(7)
ELL_TRUE, NOISE_TRUE = 0.8, 0.1
x = np.sort(rng.uniform(-3, 3, 12))
f_clean = sample_gp(rng, x, ELL_TRUE)
y = f_clean + NOISE_TRUE * rng.standard_normal(12)

print(f"""
从先验里采一个任务：先掷骰子定隐变量，再按它生成数据。
  这次掷到的隐变量：长度尺度 ell = {ELL_TRUE}（函数有多平缓），噪声 = {NOISE_TRUE}
  **模型永远看不到这两个数**，它只能从数据里猜。
""")
print(f"  x（输入）  {x}")
print(f"  y（标签）  {y}")
print(f"""
  前 8 个点当「上下文」（有标签，模型能看到 x 和 y）
  后 4 个点当「查询」（只给 x，要模型预测 y）

  上下文 x  {x[:8]}
  上下文 y  {y[:8]}
  查询   x  {x[8:]}
  真答案 y  {y[8:]}     <- 训练时用来算损失，评估时用来打分
""")

# ============================================================ 2. 张量长什么样
banner("2. 喂给网络的张量")

model = PFN()
model.load_state_dict(torch.load("pfn_A.pt", map_location="cpu"))
model.eval()

xt = torch.tensor(x[None], dtype=torch.float32)
yt = torch.tensor(np.concatenate([y[:8], np.zeros(4)])[None], dtype=torch.float32)
print(f"""
  x: {tuple(xt.shape)}  (批大小, 序列长度)  —— 12 个点排成一个序列
  y: {tuple(yt.shape)}  查询位置的 y 填 0，靠位置和一个特殊 token 区分
  n_ctx = 8            —— 前 8 个是上下文，后 4 个是查询

  注意力掩码：查询点之间互相看不见（否则会互相泄题），只能看上下文。
  参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M（6 层，128 维）
""")

with torch.no_grad():
    mu, logv = model(xt, yt, 8)
sd = logv.exp().sqrt()
print(f"  输出是每个查询点的一个高斯：均值 + log 方差")
print(f"    预测均值  {mu[0].numpy()}")
print(f"    预测标准差 {sd[0].numpy()}")
print(f"    真实答案  {y[8:]}")

# ============================================================ 3. 损失函数
banner("3. 损失函数")

tgt = torch.tensor(y[8:][None], dtype=torch.float32)
per_point = 0.5 * (logv + (tgt - mu) ** 2 / logv.exp())
print(f"""
  预训练损失（唯一一个真正在被优化的东西）—— 高斯负对数似然：

      loss = mean over query points of  0.5 * ( log v + (y_true - mu)^2 / v )

  代码原文（exp_why_axis.py / exp_gp_prior_tilt.py）：
      loss = (0.5 * (logv + (tgt - mu) ** 2 / logv.exp())).mean()

  拿上面这个任务实际算一遍，逐点的损失：
      {per_point[0].numpy()}
      平均 = {per_point.mean().item():.4f}

  这就是全部。优化器 AdamW，lr 3e-4，OneCycle，15000 步，batch 48。
  每一步重新采 48 个**全新的合成任务**——数据是无限的，不存在过拟合到某个数据集。

  为什么用这个损失：最小化它的最优解，恰好是先验 P 下的贝叶斯后验预测分布。
  所以训练出来的网络 = 「在 P 下做贝叶斯推断」的一个摊销近似。这是 PFN 的全部理论基础。

  ---- 分类版本（TabICL 那条线）----
  表格 PFN 的预训练损失是交叉熵，但那是 TabICL 作者做的，我们没参与，也没有重训。
  我们只调用它的推理接口。
""")

# ============================================================ 4. 我们在估计什么
banner("4. 我们在估计什么")

mu_true, var_true = gp_posterior(x[:8], y[:8], x[8:], ELL_TRUE, NOISE_TRUE)
ells_A, noises_A = PRIORS["A(只有尺度变)"]
mu_mix, var_mix = mixture_posterior(x[:8], y[:8], x[8:], ells_A, noises_A)

print(f"""
  四个不同的东西，别混（还是上面那个任务）：

  (a) 真实答案              {y[8:]}
  (b) 用真实隐变量算的后验    {mu_true}
      —— 上帝视角，知道 ell={ELL_TRUE} noise={NOISE_TRUE}
  (c) 模型自己先验 P 下的后验  {mu_mix}
      —— 在 P 的隐变量网格上按边际似然加权，这是「P 下能做到的最好」
  (d) PFN 实际输出           {mu[0].numpy()}

  这个项目关心的是 (d) 离 (c) 有多近，而不是 (d) 离 (a) 有多近。
  单个任务的数字全是噪声，必须多任务平均。下面跑 {200} 个任务：
""")

# 单个任务不足以说明问题，取平均。数据一律从 A 自己的先验里采（即「先验正确」的情形）。
rng2 = np.random.default_rng(2024)
acc = {"PFN↔P下后验": [], "PFN↔上帝视角": [], "PFN↔固定ell=1.0": [],
       "PFN↔全预测0": [], "P下后验↔上帝视角": []}
for _ in range(200):
    e = rng2.choice(ells_A)
    xx = np.sort(rng2.uniform(-3, 3, 30))
    ff = sample_gp(rng2, xx, e) + 0.1 * rng2.standard_normal(30)
    xc_, yc_, xq_ = xx[:20], ff[:20], xx[20:]
    p, _ = predict_raw(model, xc_, yc_, xq_)
    mmix, _ = mixture_posterior(xc_, yc_, xq_, ells_A, noises_A)
    mtru, _ = gp_posterior(xc_, yc_, xq_, e, 0.1)
    mfix, _ = gp_posterior(xc_, yc_, xq_, 1.0, 0.1)
    acc["PFN↔P下后验"].append(np.sqrt(np.mean((p - mmix) ** 2)))
    acc["PFN↔上帝视角"].append(np.sqrt(np.mean((p - mtru) ** 2)))
    acc["PFN↔固定ell=1.0"].append(np.sqrt(np.mean((p - mfix) ** 2)))
    acc["PFN↔全预测0"].append(np.sqrt(np.mean(p ** 2)))
    acc["P下后验↔上帝视角"].append(np.sqrt(np.mean((mmix - mtru) ** 2)))

print("    均方根差（越小 = 越接近，200 个任务平均）：\n")
for k, v in acc.items():
    tag = "  <- 参照之间的距离，不是 PFN 的成绩" if k.startswith("P下后验") else ""
    print(f"      {k:<22}{np.mean(v):.4f}{tag}")
pfn_only = {k: v for k, v in acc.items() if k.startswith("PFN")}
print(f"""
  PFN 最接近的是「{min(pfn_only, key=lambda k: np.mean(pfn_only[k]))}」，
  比「上帝视角」更近，更远远好过「固定 ell」和「全预测 0」这两个平凡解释。

  这就是整篇论文的出发点：PFN 不是在逼近「正确答案」，而是在逼近
  「它自己那个先验下的最优答案」。

  注意这里两个参照本身只差 {np.mean(acc['P下后验↔上帝视角']):.4f}——因为数据就是从
  先验里采的，先验是对的，两者当然接近，这一节的区分度有限。真正拉开差距要靠
  「错配」实验：喂进先验支撑之外的数据，两个参照分道扬镳，看 PFN 跟谁走。
  那是 eval_gp_prior_tilt.py 的 B 部分。
""")

# ============================================================ 5. 掰动实验的真实数据
banner("5. 掰动实验：真实的上下文行 vs anchor 行")

Q_LO, Q_HI, A_LO, A_HI = 1.2, 3.0, -3.0, -1.2
rng = np.random.default_rng(11)
xq = np.sort(rng.uniform(Q_LO, Q_HI, 4))
xl = np.sort(rng.uniform(Q_LO, Q_HI, 5))
xa = np.sort(rng.uniform(A_LO, A_HI, 8))
allx = np.concatenate([xl, xq, xa])
f = sample_gp(rng, allx, 0.35) + 0.1 * rng.standard_normal(len(allx))
yl, ya_low = f[:5], f[9:]
ya_high = ya_low + 0.5 * rng.standard_normal(len(xa))   # 同一批点，只是更吵

gap = (Q_LO + Q_HI) / 2 - (A_LO + A_HI) / 2
print(f"""
  查询点在 x∈[{Q_LO},{Q_HI}]，本地上下文也在那里（只有 5 个，很稀疏），
  anchor 全部放在 x∈[{A_LO},{A_HI}]，与查询区中心相隔 {gap:.1f}。

  在真实长度尺度 0.35 下，隔 {gap:.1f} 的两点函数相关性 = {np.exp(-0.5 * gap**2 / 0.35**2):.1e}
  —— **实际为零**。anchor 对「查询点的 y 是多少」不含任何信息。

  本地上下文  x {xl}
              y {yl}
  查询点      x {xq}
  anchor      x {xa}
       低噪声 y {ya_low}
       高噪声 y {ya_high}    <- 同一批底层函数值，只是加了更多噪声
""")


def predict(xc, yc, xq_):  # 兼容旧调用
    xx = torch.tensor(np.concatenate([xc, xq_])[None], dtype=torch.float32)
    yy = torch.tensor(np.concatenate([yc, np.zeros_like(xq_)])[None], dtype=torch.float32)
    with torch.no_grad():
        m, lv = model(xx, yy, len(xc))
    return m[0].numpy(), lv[0].exp().sqrt().numpy()


m0, s0 = predict(xl, yl, xq)
m1, s1 = predict(np.concatenate([xl, xa]), np.concatenate([yl, ya_low]), xq)
m2, s2 = predict(np.concatenate([xl, xa]), np.concatenate([yl, ya_high]), xq)

print("  用 pfn_A 实际跑三遍（A 的先验里噪声是固定的，它「没有噪声这个概念」）：\n")
print(f"    {'条件':<22}{'预测均值':>34}{'预测标准差':>34}")
for tag, m, s in (("不加 anchor", m0, s0), ("加低噪声 anchor", m1, s1), ("加高噪声 anchor", m2, s2)):
    print(f"    {tag:<20}{str(np.round(m, 3)):>36}{str(np.round(s, 3)):>36}")
print(f"""
    高噪声 − 低噪声   均值移动 {np.abs(m2 - m1).mean():+.4f}   标准差移动 {(s2 - s1).mean():+.4f}

  读法：anchor 里没有任何关于查询点的信息，但把它们变吵之后，
  模型在查询点上的预测**变了**。这就是「先验被掰动」。
  而且 A 连均值都动了——它没有噪声这个隐变量，只好把「变吵」当成
  「函数其实更波动」来解释，这就是论文里说的「错误归因」。
""")

# ============================================================ 6. 表格那条线
banner("6. 表格 PFN 那条线的数据长什么样（TabICLv2，未安装，这里只造数据）")

D, N_CTX, N_ANCHOR, FAR = 8, 30, 60, 6.0
rng = np.random.default_rng(3)
i, t = rng.permutation(D), rng.normal(0, 0.4, 3)
rule = lambda X: (((X[:, i[0]] > t[0]) & (X[:, i[1]] > t[1])) | (X[:, i[2]] > t[2] + 0.8)).astype(int)
w, v = rng.normal(0, 1, D), rng.normal(0, 1, D)
other = lambda X: (np.tanh(X @ w) + 0.6 * np.sin(X @ v) > 0).astype(int)

Xtr = rng.normal(0, 1, (N_CTX, D))
ytr = rule(Xtr)
Xa0 = rng.normal(0, 1, (N_ANCHOR, D))
ya0 = other(Xa0)                       # 标签来自**另一条完全无关的规则**
Xa = Xa0 + FAR
ya_noisy = np.where(rng.random(N_ANCHOR) < 0.5, 1 - ya0, ya0)

print(f"""
  目标任务：{D} 个特征，标签由一条硬阈值规则生成，{N_CTX} 行上下文。

  真实上下文的前 3 行（特征 + 标签）：""")
for k in range(3):
    print(f"    {np.round(Xtr[k], 2)}  -> y={ytr[k]}")
print(f"""
  anchor 的前 3 行（特征整体 +{FAR}，标签来自另一条无关规则）：""")
for k in range(3):
    print(f"    {np.round(Xa[k], 2)}  -> y={ya0[k]}  (加 50% 噪声后 y={ya_noisy[k]})")

Xte = rng.normal(0, 1, (200, D))
d_ctx = np.linalg.norm(Xtr[:, None] - Xte[None], axis=-1).min(0).mean()
d_anc = np.linalg.norm(Xa[:, None] - Xte[None], axis=-1).min(0).mean()
print(f"""
  到测试点的最近邻距离：真实上下文 {d_ctx:.2f}   anchor {d_anc:.2f}
  —— anchor 远了 {d_anc / d_ctx:.1f} 倍，任何近邻检索都会把它们丢掉。

  实验里唯一变化的量：ya 的噪声率 eta ∈ {{0, 10%, 25%, 40%, 50%}}。
  特征、行数、位置、底层规则全部固定不变。
""")

# ============================================================ 7. 有没有预训练模型
banner("7. 预训练模型")

print("""
  有两类，别混：

  (1) 自己训的小 PFN —— pfn_A.pt / pfn_B.pt / pfn_C.pt，各 4.8 MB，1.2M 参数
      已提交在分支上（虽然 .gitignore 里写了 pfn_*.pt，这三个是被强制加进去的）。
      三者只有先验不同：
        A  长度尺度变（12 档），噪声固定 0.1
        B  长度尺度固定 1.0，噪声变（12 档）
        C  两者都变（12 x 12 = 144 种组合）
      各训 15000 步。用途是「有解析解可以对答案」的受控实验。
      重训命令：python exp_why_axis.py 15000  （CPU 上约几十分钟）

  (2) 真实表格 PFN —— TabICLv2，别人预训练好的，pip 装 tabicl 后自动下载权重。
      我们完全没有训练它，只调 fit / predict_proba 的推理接口。
      本机没装，所以第 6 节只造了数据没跑模型。

  另外要讲清楚一件事：**「方法」本身没有损失函数，也不训练任何东西。**
  方法就是「往上下文里多塞 60 行构造出来的 anchor」，一次前向传播，完事。
  log loss 和 ECE 是**评估指标**，不是训练目标——目前 anchor 的噪声率是网格扫出来的
  ({0, 15%, 35%, 50%})，不是优化出来的。
""")
