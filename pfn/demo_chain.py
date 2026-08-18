"""完整的链条：ell -> 协方差 -> 对观测值的加权 -> 预测；再到 anchor 怎么插进来。"""

import numpy as np

from exp_why_axis import PRIORS, log_marginal, rbf, sample_gp

np.set_printoptions(precision=3, suppress=True, linewidth=120)

NOISE = 0.1
ELLS, _ = PRIORS["A(只有尺度变)"]


def banner(t):
    print("\n" + "=" * 84 + f"\n{t}\n" + "=" * 84)


def weights_and_var(xc, xq, ell, noise=NOISE):
    """预测 = 对观测 y 的加权求和。这里返回那个权重向量，以及预测方差。"""
    Kcc = rbf(xc, xc, ell) + noise ** 2 * np.eye(len(xc))
    kqc = rbf(np.atleast_1d(xq), xc, ell)          # (n_q, n_c)
    w = kqc @ np.linalg.inv(Kcc)                   # (n_q, n_c) 权重
    var = 1.0 + noise ** 2 - np.sum(kqc * (w), axis=1)
    return w, np.maximum(var, 1e-9)


# =============================================== 0. 先摆好一个极小的例子
banner("0. 一个极小的例子（3 个观测点，1 个查询点）")

xc = np.array([-1.0, 0.0, 0.5])
yc = np.array([0.8, -0.3, 1.2])
xq = np.array([0.2])

print(f"""
  已知（上下文）：
      x = {xc}
      y = {yc}
  要预测：
      x = {xq}   的 y 是多少，以及有多确定
""")

# =============================================== 1. ell -> 协方差
banner("1. ell 决定协方差矩阵")

for ell in (0.3, 1.0, 3.0):
    K = rbf(xc, xc, ell)
    kq = rbf(xq, xc, ell)
    print(f"  ell = {ell}")
    print(f"    观测点之间的相关 K：")
    for row in np.round(K, 3):
        print(f"        {row}")
    print(f"    查询点与各观测点的相关 k_q： {np.round(kq[0], 3)}\n")

print("""  注意最后一行 k_q 随 ell 变化：
    ell=0.3  查询点只和最近的 x=0.0、x=0.5 有一点相关，和 x=-1.0 几乎无关
    ell=3.0  三个观测点全都强相关，连 x=-1.0 都有 0.94
  这是整条链的起点。""")

# =============================================== 2. 协方差 -> 权重 -> 预测
banner("2. 关键一步：预测就是对观测 y 的加权求和")

print("""
  高斯过程说：所有函数值（观测的 + 要预测的）服从一个联合高斯分布，
  协方差就是上面那个 K。**预测 = 把这个联合分布对已观测的部分取条件。**
  条件高斯有闭式解：

      预测均值  mu_q  = k_q · (K + sigma^2 I)^(-1) · y      <- 记作 w · y
      预测方差  var_q = 1 + sigma^2 - k_q · (K + sigma^2 I)^(-1) · k_q^T

  第一行右边的 w = k_q (K+sigma^2 I)^(-1) 是一个**只依赖 x 和 ell、与 y 无关**
  的权重向量。所以预测就是把观测到的 y 按 w 加权平均。把 w 打出来看：
""")

print(f"    {'ell':>6}{'w(x=-1.0)':>12}{'w(x=0.0)':>12}{'w(x=0.5)':>12}"
      f"{'预测均值':>12}{'预测标准差':>12}")
print("    " + "-" * 66)
for ell in (0.3, 0.7, 1.0, 2.0, 3.0):
    w, var = weights_and_var(xc, xq, ell)
    mu = float(w[0] @ yc)
    print(f"    {ell:>6.1f}{w[0][0]:>12.3f}{w[0][1]:>12.3f}{w[0][2]:>12.3f}"
          f"{mu:>12.3f}{np.sqrt(var[0]):>12.3f}")

print(f"""
  y = {yc}，逐行验算一下 ell=0.3 那行：
      w = {np.round(weights_and_var(xc, xq, 0.3)[0][0], 3)}
      mu = {weights_and_var(xc, xq, 0.3)[0][0][0]:.3f}*{yc[0]} +
           {weights_and_var(xc, xq, 0.3)[0][0][1]:.3f}*{yc[1]} +
           {weights_and_var(xc, xq, 0.3)[0][0][2]:.3f}*{yc[2]}
         = {float(weights_and_var(xc, xq, 0.3)[0][0] @ yc):.3f}

  **这就是「然后呢」的答案**：ell 通过协方差决定了「每个观测点对这次预测有多大发言权」，
  预测值就是按这个发言权把观测到的 y 加权平均起来。

  两头的直觉：
    ell 小 -> 只有紧挨着查询点的观测有发言权，远的权重≈0；
              而且总权重小，预测被拉回先验均值 0，方差大（"我不知道"）
    ell 大 -> 所有观测都有发言权，预测是它们的一个平滑综合，方差小（"我挺确定"）
""")

# =============================================== 3. ell 未知 -> 12 个预测的混合
banner("3. 但 ell 是未知的：于是做 12 个预测，再按可信度混合")

lml = np.array([log_marginal(xc, yc, e, NOISE) for e in ELLS])
wgt = np.exp(lml - lml.max())
wgt /= wgt.sum()

print(f"\n    {'ell':>7}{'该 ell 的预测':>14}{'该 ell 的标准差':>16}"
      f"{'这批数据有多像它':>18}")
print("    " + "-" * 57)
mus, vars_ = [], []
for e, p in zip(ELLS, wgt):
    w, var = weights_and_var(xc, xq, e)
    mu = float(w[0] @ yc)
    mus.append(mu); vars_.append(float(var[0]))
    print(f"    {e:>7.3f}{mu:>14.3f}{np.sqrt(var[0]):>16.3f}{p:>16.3f}  {'█' * int(p * 40)}")

mus, vars_ = np.array(mus), np.array(vars_)
mu_mix = wgt @ mus
var_mix = wgt @ (vars_ + mus ** 2) - mu_mix ** 2
print(f"""
    最终预测 = 按最后一列加权平均：
        均值   = sum(权重 * 各自的预测) = {mu_mix:.3f}
        方差   = 两部分之和：
                 各自方差的加权平均      {float(wgt @ vars_):.4f}   （每个 ell 内部的不确定）
                 各自均值之间的分散程度  {float(wgt @ mus**2 - mu_mix**2):.4f}   （不知道该信哪个 ell）
                 合计 {var_mix:.4f}  ->  标准差 {np.sqrt(var_mix):.3f}

  第二部分很关键：**「不知道该用哪个 ell」本身就贡献了一块方差。**
  这就是 ell 和「我们要预测的东西」之间最直接的联系——
  你对 ell 越没把握，预测的不确定度就越大。
""")

# =============================================== 4. anchor 怎么插进来
banner("4. anchor 怎么插进来：两条完全不同的通路")

rng = np.random.default_rng(3)
xa = np.linspace(-8.0, -6.0, 8)                    # 极远
ya_clean = sample_gp(rng, xa, 1.0) * 0.5
ya_noisy = ya_clean + 0.8 * rng.standard_normal(len(xa))

print(f"""
  在离查询点很远的地方（x 从 {xa[0]} 到 {xa[-1]}，查询点在 x={xq[0]}）放 8 个 anchor。
  两个版本：一个干净、一个很吵，底层函数完全相同。

      anchor x       {np.round(xa, 2)}
      干净版 y       {np.round(ya_clean, 2)}
      吵版   y       {np.round(ya_noisy, 2)}
""")

for tag, ya in (("干净 anchor", ya_clean), ("吵 anchor", ya_noisy)):
    xc2 = np.concatenate([xc, xa])
    yc2 = np.concatenate([yc, ya])

    # 通路一：anchor 在加权求和里占多大权重
    w_all, _ = weights_and_var(xc2, xq, 1.0)
    w_anchor_share = np.abs(w_all[0][3:]).sum() / np.abs(w_all[0]).sum()

    # 通路二：anchor 怎么改变对 ell 的信念
    lml2 = np.array([log_marginal(xc2, yc2, e, NOISE) for e in ELLS])
    wgt2 = np.exp(lml2 - lml2.max()); wgt2 /= wgt2.sum()
    mus2, vars2 = [], []
    for e in ELLS:
        w, var = weights_and_var(xc2, xq, e)
        mus2.append(float(w[0] @ yc2)); vars2.append(float(var[0]))
    mus2, vars2 = np.array(mus2), np.array(vars2)
    mu2 = wgt2 @ mus2
    var2 = wgt2 @ (vars2 + mus2 ** 2) - mu2 ** 2
    ell_hat = float(np.exp(wgt2 @ np.log(ELLS)))

    print(f"  --- {tag} ---")
    print(f"    通路一（直接进加权求和）：8 个 anchor 在权重里只占 "
          f"{w_anchor_share:.2e} —— **实际为零**")
    print(f"    通路二（改变对 ell 的信念）：ell 的后验均值 "
          f"{float(np.exp(wgt @ np.log(ELLS))):.3f} -> {ell_hat:.3f}")
    print(f"    结果：预测均值 {mu_mix:.3f} -> {mu2:.3f}   "
          f"预测标准差 {np.sqrt(var_mix):.3f} -> {np.sqrt(var2):.3f}\n")

print("""  这就是整篇论文的机制：

    通路一是所有人都懂的「近邻检索」——远处的点权重为零，扔掉就行。
    通路二是被忽略的那条——远处的点虽然对「这个查询点的 y 是多少」不含信息，
    却携带了「这个任务是什么性质」的信息，从而改变了模型对 ell（或噪声）的信念，
    进而改变了 12 个候选预测的混合比例。

  而且注意上面两行数字的对比：**预测均值几乎没动，预测标准差动得很明显。**
  这正是论文的核心论点——先验被掰动的代价主要记在「有多确定」上，
  而不是「答案是多少」上。只看准确率的评测体系，永远发现不了这件事。""")
