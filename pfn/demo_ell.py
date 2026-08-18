"""ell 到底是什么：用同一个随机种子、只改 ell，看函数怎么变。"""

import numpy as np

from exp_why_axis import rbf

np.set_printoptions(precision=3, suppress=True, linewidth=110)

W, H = 74, 13                      # ASCII 画布尺寸
X = np.linspace(-3, 3, W)


def sample_with_fixed_z(ell, z):
    """用同一个 z 向量采样：这样不同 ell 之间的差异纯粹来自 ell 本身。"""
    K = rbf(X, X, ell) + 1e-8 * np.eye(W)
    return np.linalg.cholesky(K) @ z


def ascii_plot(y, lo, hi):
    rows = [[" "] * W for _ in range(H)]
    zero = int(round((0 - lo) / (hi - lo) * (H - 1)))
    for c in range(W):
        rows[H - 1 - max(0, min(H - 1, zero))][c] = "·"
    for c, v in enumerate(y):
        r = int(round((v - lo) / (hi - lo) * (H - 1)))
        rows[H - 1 - max(0, min(H - 1, r))][c] = "█"
    return ["  |" + "".join(r) + "|" for r in rows]


print("""
==========================================================================
ell（长度尺度）是什么
==========================================================================

  ell 是高斯过程里唯一控制「函数抖不抖」的那个参数。它出现在核函数里：

      K(x, x') = exp( -0.5 * (x - x')^2 / ell^2 )

  K 衡量「相距 (x-x') 的两个点，函数值有多相关」。
    ell 小 -> 稍微离开一点相关性就掉到 0 -> 函数各处互相独立 -> 看起来很抖
    ell 大 -> 隔很远还高度相关          -> 函数被拉平    -> 看起来很缓

  下面四条曲线用的是**同一个随机数向量 z**，唯一的区别就是 ell。
  所以形状的差异百分之百来自 ell。
""")

rng = np.random.default_rng(5)
z = rng.standard_normal(W)
ELLS = [0.3, 0.7, 1.5, 3.0]
curves = {e: sample_with_fixed_z(e, z) for e in ELLS}
lo = min(c.min() for c in curves.values()) - 0.2
hi = max(c.max() for c in curves.values()) + 0.2

for e in ELLS:
    y = curves[e]
    # 用「相邻点差值的平均绝对值」当抖动程度的一个粗略读数
    wig = np.abs(np.diff(y)).mean()
    tag = {0.3: "最抖", 0.7: "较抖", 1.5: "较缓", 3.0: "最缓"}[e]
    print(f"  ell = {e:<4}（{tag}）   相邻点平均跳动 {wig:.3f}")
    for line in ascii_plot(y, lo, hi):
        print(line)
    print(f"   x = -3{' ' * (W - 10)}x = +3\n")

print("""==========================================================================
换个角度看：相关性随距离怎么衰减
==========================================================================
""")
dists = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
print(f"  {'两点间距离':>10}" + "".join(f"{f'ell={e}':>12}" for e in ELLS))
print("  " + "-" * 58)
for d in dists:
    row = f"  {d:>10.2f}"
    for e in ELLS:
        c = np.exp(-0.5 * d ** 2 / e ** 2)
        row += f"{c:>12.4f}" if c > 1e-4 else f"{c:>12.1e}"
    print(row)

print("""
  读这张表：ell=0.3 时，隔开 1.0 的两个点相关性只剩 0.0038，基本无关；
  ell=3.0 时，隔开 4.0 还有 0.41，仍然强相关。

==========================================================================
在这个项目里，ell 扮演什么角色
==========================================================================

  它是**隐变量**——每个任务被生成时随机抽一个 ell，然后就扔掉，模型看不到。
  模型必须从上下文数据里反推「这个任务的函数大概有多抖」，才能做好预测。

  这正是整个项目要研究的东西：
    · 模型 A 的先验里 ell 会变（12 档），所以 A 必须学会推断它
    · 模型 B 的先验里 ell 固定为 1.0，所以 B 根本没有「函数抖不抖」这个概念
    · 「掰动先验」实验做的事，就是往上下文里塞远处的 anchor，
      看能不能改变模型对 ell 的推断

  一个具体例子：远处 anchor 里的数据如果很抖，模型会认为「这个任务的 ell 小」，
  于是在**完全无关的查询点**上也变得更不确定。anchor 里没有任何关于查询点的信息，
  但它携带了「这个任务是什么性质」这一全局判断，而 ell 就是承载这个判断的变量。

  另一个隐变量是**噪声**。两者容易混，但可以区分：
    ell 小   -> 底层函数本身就在剧烈起伏（信号真的在动）
    噪声大   -> 底层函数很平缓，但每个观测点被随机扰动了（信号没动，读数在抖）
  单看少量数据，这两者产生的散点图长得很像——这就是论文里说的「隐变量纠缠」，
  也是模型 C（两个都变）逼近精确贝叶斯明显更差的原因。
""")
