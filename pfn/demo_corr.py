"""k(x,x') 里的「相关性」到底是什么意思：它是一群随机函数上的统计量。"""

import numpy as np

np.set_printoptions(precision=3, suppress=True, linewidth=110)

XQ = 0.2
XC = np.array([-1.0, 0.0, 0.5])
N_DRAW = 4000


def rbf(a, b, ell):
    return np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell ** 2)


def banner(t):
    print("\n" + "=" * 80 + f"\n{t}\n" + "=" * 80)


def scatter(u, v, w=54, h=15):
    """把两组数画成 ASCII 散点图。"""
    lo, hi = -3.2, 3.2
    grid = [[" "] * w for _ in range(h)]
    for a, b in zip(u, v):
        c = int((a - lo) / (hi - lo) * (w - 1))
        r = int((b - lo) / (hi - lo) * (h - 1))
        if 0 <= c < w and 0 <= r < h:
            ch = grid[h - 1 - r][c]
            grid[h - 1 - r][c] = "·" if ch == " " else ("+" if ch == "·" else "█")
    return ["   |" + "".join(r) + "|" for r in grid]


banner("1. 先说清楚：相关的是「函数值」，不是 x")

print(f"""
  高斯过程不是在描述一条曲线，而是在描述**一整族随机曲线**。
  你可以想成：大自然每次随机抽一条歪歪扭扭的曲线出来。

  k(x, x') 回答的问题是：
      「在这一族随机曲线上，位置 x 处的高度 和 位置 x' 处的高度，有多同步？」

  它是两个**随机变量** f(x) 和 f(x') 之间的相关系数，取值 0 到 1。
      = 1  两处高度完全同步（知道一个就等于知道另一个）
      = 0  两处高度毫无关系（知道一个对另一个一点帮助没有）

  公式只依赖两点的**距离**：

      k(x, x') = exp( -0.5 * (x - x')^2 / ell^2 )

  合理，因为曲线是连续的：挨得近的两点高度必然差不多，隔得远就各走各的。
  ell 就是「多远算远」的那把尺子。
""")

banner("2. 手算一遍那三个数")

print(f"""  查询点 x_q = {XQ}，三个观测点 x = {XC}
  距离     |x_q - x| = {np.abs(XQ - XC)}
""")
for ell in (0.3, 1.0, 3.0):
    d = np.abs(XQ - XC)
    k = np.exp(-0.5 * d ** 2 / ell ** 2)
    print(f"  ell = {ell}")
    for di, ki in zip(d, k):
        print(f"      距离 {di:.1f}:  exp(-0.5 * {di:.1f}^2 / {ell}^2) "
              f"= exp({-0.5 * di**2 / ell**2:>7.3f}) = {ki:.3f}")
    print()

print("""  看第一列（距离 1.2 那个点）随 ell 的变化：
      ell=0.3 -> 0.000     1.2 相对于尺子 0.3 是「非常远」
      ell=1.0 -> 0.487     1.2 相对于尺子 1.0 是「中等」
      ell=3.0 -> 0.923     1.2 相对于尺子 3.0 是「很近」
  同一个物理距离 1.2，换把尺子量，结论完全不同。这就是 ell 的全部作用。""")

banner("3. 这个数是真的能测出来的：采 4000 条随机曲线看散点")

rng = np.random.default_rng(0)
pts = np.array([XC[0], XQ])              # 只看 x=-1.0 和 x=0.2 这两处

for ell in (0.3, 1.0, 3.0):
    K = rbf(pts, pts, ell) + 1e-9 * np.eye(2)
    L = np.linalg.cholesky(K)
    samples = (L @ rng.standard_normal((2, N_DRAW)))     # (2, N) 每列是一条曲线在两处的高度
    emp = np.corrcoef(samples[0], samples[1])[0, 1]
    theo = float(rbf(np.array([XC[0]]), np.array([XQ]), ell)[0, 0])
    print(f"\n  ell = {ell}    理论 k = {theo:.3f}    "
          f"实测相关系数 = {emp:.3f}   （{N_DRAW} 条随机曲线）")
    print(f"   横轴 = 该曲线在 x=-1.0 处的高度，纵轴 = 同一条曲线在 x=0.2 处的高度")
    for line in scatter(samples[0], samples[1]):
        print(line)
    print("    -3" + " " * 50 + "+3")

print(f"""
  三张图讲的是同一件事：
    ell=0.3  散成一团圆云 —— 知道 x=-1.0 处的高度，对 x=0.2 处毫无预测力
    ell=3.0  挤成一条细斜线 —— 知道一个就基本知道另一个

  **「相关性 0.923」的字面含义就是：这张散点图的相关系数是 0.923。**
  它是一个关于「大自然会抽出什么样的曲线」的陈述，与你手上这批数据无关。
""")

banner("4. 为什么这个数直接决定预测")

print(f"""
  现在把它接回预测。假设你观测到 x=-1.0 处 y = 0.8，要猜 x={XQ} 处的 y。

  在二元标准高斯下，条件期望有个初中就能写出来的形式：

      E[ f(x_q) | f(x_c)=y ] = k * y            <- k 就是那个相关系数
      Var[ f(x_q) | f(x_c)=y ] = 1 - k^2

  代入具体数字（只用这一个观测点）：
""")
y_obs = 0.8
for ell in (0.3, 1.0, 3.0):
    k = float(rbf(np.array([XQ]), np.array([XC[0]]), ell)[0, 0])
    print(f"      ell={ell}:  k={k:.3f}  ->  预测 {k:.3f} * {y_obs} = {k * y_obs:+.3f}   "
          f"预测方差 1 - {k:.3f}^2 = {1 - k**2:.3f}")

print(f"""
  读法：
    ell=0.3  k≈0，预测 ≈ 0（退回先验均值，等于「我不知道」），方差 ≈ 1（最大）
    ell=3.0  k=0.923，预测 ≈ 0.738（几乎照抄观测值），方差 0.148（很确定）

  **这就是 ell 和「我们要预测的东西」之间的全部联系**：
  k 既是「照抄多少」的系数，也是「能减掉多少不确定」的系数。

  多个观测点时公式变成矩阵形式 w = k_q (K + sigma^2 I)^(-1)，
  多出来的 (K+sigma^2 I)^(-1) 是在做「去重」——两个观测点如果彼此也高度相关，
  它们提供的是重复信息，不能各算一份。但核心还是这件事：
  **相关性决定发言权，发言权决定预测。**
""")
