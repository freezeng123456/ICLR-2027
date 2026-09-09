# 随机小批量 Feynman–Kac 校正：两日理论审查

日期：2026-09-09。本文只讨论一个瞬时 Feynman–Kac 权重；它不把经典的无偏乘积恒等式包装成新方法。

## 决定性判断

若每一步的 potential 估计无偏但有状态相关方差，直接把它放进指数权重会产生一阶累积的有限步偏差。对步长 \(h\)、总时间 \(T=Kh\)、小批量大小 \(M\)，在有界三阶矩和稳定归一化的条件下，未校正权重的相对偏差通常为

\[
  O\left(\sum_{k=1}^K h^2\frac{\sigma^2(X_k)}{M}\right)
  =O\left(T h\,\bar\sigma^2/M\right).
\]

因此减小步长可以降低小批量指数偏差，但会增加步骤数；粒子数 \(P\) 还要承担 \(O(T/(hP))\) 的重采样方差。这个候选只有在可以稳定估计并消除局部二阶 cumulant，且由此带来的额外计算小于减小 \(h\) 或增大 \(M,P\) 的代价时才可能形成方法贡献。

## 1. 指数化无偏估计仍然有偏

设目标权重为

\[
  W(x)=\exp[-h\phi(x)],\qquad
  \widehat\phi=\phi+\varepsilon,
  \quad E[\varepsilon\mid x]=0,
  \quad E[\varepsilon^2\mid x]=\sigma^2(x)/M.
\]

若 \(\varepsilon\) 条件高斯，则

\[
 E[\exp(-h\widehat\phi)\mid x]
 =\exp(-h\phi)\exp\!\left(\frac{h^2\sigma^2(x)}{2M}\right).
\]

一般情形由 cumulant 展开给出

\[
 \log E[e^{-h\varepsilon}\mid x]
 =\frac{h^2\sigma^2(x)}{2M}
 -\frac{h^3\kappa_3(x)}{6M^{3/2}}+O(h^4M^{-2}).
\]

所以“potential 估计无偏”并不意味着 Feynman–Kac 权重无偏。归一化后，粒子分布的测试函数 (f) 的一阶偏差主项是

\[
 \frac{h^2}{2M}\operatorname{Cov}_{\eta_k}
 \bigl(f(X),\sigma^2(X)\bigr),
\]

并在 \(K=T/h\) 步后累积到 \(O(Th/M)\)。若方差与状态无关，它主要表现为公共归一化因子；状态相关方差会改变归一化后的分布，不能靠除以总权重消除。若小批量噪声在不同粒子之间共享，协方差项还会进入该展开，不能套用独立粒子公式。

## 2. 可实施的二阶校正

若能从同一个小批量机制得到条件方差估计 \(\widehat v(x)\)，并且

\[
 E[\widehat v\mid x]=\sigma^2(x)/M,
 \qquad E|\widehat v-\sigma^2/M|^2<\infty,
\]

可使用

\[
 \widehat W_2=\exp(-h\widehat\phi)\left(1-\frac{h^2}{2}\widehat v\right).
\]

在 \(\widehat v\) 与 \(\widehat\phi\) 独立，或用独立的交叉批量构造协方差校正时，\(\widehat W_2\) 的期望误差从 \(O(h^2/M)\) 降为 \(O(h^3M^{-3/2}+h^4M^{-2})\)。但它可能为负；普通 bootstrap 或截断会重新引入偏差。若强制使用正权重，应采用

\[
 \widehat W_{\rm G}=\exp(-h\widehat\phi-h^2\widehat v/2),
\]

这在条件高斯噪声且方差已知时精确无偏，在非高斯情形只消除二阶 cumulant，剩余三阶项为 \(O(h^3M^{-3/2})\)。估计 \(\widehat v\) 的误差本身通过指数放大，因此需要独立批量或解析控制变量，不能用同一小批量的样本方差直接宣称无偏。

这两种校正都不是自动成立的通用算法：二阶多项式校正的负权重会破坏标准 SMC；指数校正依赖条件高斯或可控 cumulant 假设。实验上必须同时报告负权重比例、有效样本数和归一化前后相对方差。

## 3. Poisson 构造：无偏但通常不适合作为正权重方案

若可获得独立的无偏估计 \(\widehat a_i\)，令 \(N\sim\mathrm{Poisson}(\lambda)\)，则

\[
 Z=\prod_{i=1}^{N}\left(1-\frac{h\widehat a_i}{\lambda}\right)
 \quad\Longrightarrow\quad E[Z]=e^{-ha},
 \qquad a=E[\widehat a_i].
\]

这是 Poisson 泛函的无偏化。若因子允许为负，它一般是带符号的，且

\[
 E[Z^2]=\exp\left(-2ha+\frac{h^2E[\widehat a^2]}{\lambda}\right)
\]

（在独立同分布条件下），因此 \(\lambda\) 过小或 potential 方差较大时相对方差爆炸。要得到正权重，可要求 \(\widehat a_i\in[0,\lambda/h]\) 且对 \(a=\phi-c\ge0\) 使用

\[
 Z_+=e^{-hc}\prod_{i=1}^{N}\left(1-\frac{h\widehat a_i}{\lambda}\right).
\]

这个条件对无界 Gaussian likelihood 或 score-derived potential 通常不满足；截断 (widehat a_i) 会损失无偏性。Poisson 方法适合作为解析 sanity check，不应在两日实验中作为主算法。

## 4. 组数、批量、步长和粒子数

在每一步每粒子使用 (M) 个组、(P) 个粒子，令每个组贡献的条件方差有上界 (v_0)，则小批量 potential 方差约为 (v_0/M)。在独立重采样、有限四阶矩和稳定混合条件下，合理的误差账本是

\[
 \text{bias}_{\rm mini}=O(T h v_0/M),\qquad
 \text{variance}_{\rm mini}=O(T h v_0/M),
\]

\[
 \text{variance}_{\rm particle}=O(T/(hP)),\qquad
 \text{discretization}=O(h^q),
\]

其中 (q=1) 或 (2) 取决于时间积分器与权重实现。若二阶 cumulant 校正成功，第一项可降为 (O(T h^2 M^{-3/2}))，但校正估计量的方差与负权重风险必须单独计入。增加组总数 (G) 不会自动改善误差；若每个组贡献方差不衰减，计算量只按 (M) 控制，而遗漏组的随机方差仍由 (v_0/M) 决定。

用总的粒子计算量 (C\asymp P M T/h) 粗略消去 (P) 后，粒子方差项约为 (T^2M/C)，小批量项约为 (T h v_0/M)。这说明不存在只凭无偏性得到的免费优势；必须在固定 wall-clock 或固定 (C) 下测量 Pareto 曲线，并与全量校正、减小 (h)、增大 (P) 的基线比较。

## 5. 两日内可证伪的检验

先用解析 Gaussian 条件：比较全量 potential、普通小批量权重、(widehat W_2)、Gaussian cumulant 校正和 Poisson 构造。对每个 (G,M,h,P) 报告终点均值偏差、方差偏差、ESS、负权重比例和运行时间。再用一个有界非 Gaussian potential（例如有限状态跳变模型）检查三阶 cumulant 项是否按 (M^{-3/2}) 衰减。

若 Gaussian 校正不能在相同计算预算下同时降低终点 KL 与 ESS 损失，候选应终止。若它有效，也只能主张“在具有可估计二阶 cumulant 的小批量 Feynman–Kac 权重上获得可验证有限步误差”，不能主张一般的无偏粒子后验。新颖性还必须与已有随机权重 SMC、Feynman–Kac 校正和控制变量方法逐项比较。

## 6. 随机 Euler 一步算子的二阶差

固定状态 \(x\)，设

\[
 Q_h f(x)=E\left[e^{h\widehat g}f(x+h\widehat b)\right],
\qquad
 D_h f(x)=e^{hg}f(x+hb),
\]

并假设 \(E[\widehat b\mid x]=b(x)\)、\(E[\widehat g\mid x]=g(x)\)。若 \(f\in C^3\)，在所考虑的紧集上 \(f\) 的三阶导数有界，\(\widehat b,\widehat g\) 的条件三阶绝对矩有统一上界，且 \(h|\widehat g|\le c\) 的事件外尾部贡献为 \(O(h^3)\)，Taylor 展开给出

\[
 Q_h f-D_h f
 =h^2e^{hg}\left[
 \frac12\operatorname{Cov}(\widehat b):\nabla^2 f
 \operatorname{Cov}(\widehat b,\widehat g)\cdot\nabla f
 \frac12\operatorname{Var}(\widehat g)f
 \right]+O(h^3).
\]

余项常数只依赖于 \(c\)、三阶矩上界、\(\sup\|D^3f\|\) 以及紧集上的 \(f,\nabla f,\nabla^2f\) 上界。若只要求有限三阶矩而不要求有界变量，可以用截断事件和 Hölder 不等式得到同一阶数，但余项常数还包含尾部的 uniform-integrability 界；不能仅由二阶方差存在推出该结论。

归一化 Feynman–Kac 算子写成

\[
 \mathcal T_h f=\frac{Q_hf}{Q_h1},\qquad
 \mathcal T_h^{\,0}f=\frac{D_hf}{D_h1}.
\]

令
\[
 A f=\frac12\operatorname{Cov}(\widehat b):\nabla^2 f+
 \operatorname{Cov}(\widehat b,\widehat g)\cdot\nabla f+
 \frac12\operatorname{Var}(\widehat g)f.
\]

在同一有界性条件下，

\[
 \mathcal T_h f-\mathcal T_h^{\,0}f
 =h^2\left[A f-f\,A1\right]+O(h^3)
 =h^2\left[
 \frac12\operatorname{Cov}(\widehat b):\nabla^2 f+
 \operatorname{Cov}(\widehat b,\widehat g)\cdot\nabla f
 \right]+O(h^3),
\]

这里 \(A1=\frac12\operatorname{Var}(\widehat g)\)，所以权重方差在归一化后作为公共因子抵消；随机漂移的 covariance 与 diffusion 的 Hessian 项仍然存在。若把参考点取为 \(x+hb\)，参考点变换只改变 \(O(h^3)\) 余项。因此只改正 \(\operatorname{Var}(\widehat g)\) 的 cumulant，不能消除 \(\operatorname{Cov}(\widehat b,\widehat g)\cdot\nabla f\) 与 \(\operatorname{Cov}(\widehat b):\nabla^2f\)。

## 7. 标准正态 prior 的组合路径恒等式

令 prior score 为 \(-x\)，第 \(g\) 个局部 posterior score 为 \(s_g\)，并设

\[
 r_g=s_g+x,\qquad R=\sum_{g=1}^G r_g.
\]

组合路径密度满足

\[
 \rho_t(x)\propto \operatorname{prior}_t(x)^{1-G}\prod_{g=1}^Gq_{g,t}(x),
\]

所以

\[
 \nabla\log\rho_t(x)
 =(1-G)(-x)+\sum_gs_g
 =R-x.
\]

将 \(s_g=r_g-x\) 代入由局部路径生成元相加后得到的平方项，所有只含 \(x\) 的项与 prior 基线项抵消，剩余 Feynman–Kac potential 为

\[
 g_{\mathrm{FK}}(x,t)
 =\frac{\beta(t)}2\left(\|R\|^2-\sum_{g=1}^G\|r_g\|^2\right),
\]

而相对于 prior reverse dynamics 的漂移校正为

\[
 b_{\mathrm{corr}}(x,t)=\frac{\beta(t)}2R.
\]

这两个公式是代数恒等式，前提是所有 score 使用同一个 VP marginal、同一个噪声时间 \(t\)，且 prior 为标准正态不变分布。它们不意味着把 \(R\) 或 \(g_{\mathrm{FK}}\) 换成小批量估计后仍保持精确：令 \(\widehat R\) 和 \(\widehat g\) 无偏，只能保证一阶项无偏，上一节的随机 Euler 展开仍产生漂移 covariance、扩散 Hessian 和指数 cumulant 偏差。

## 8. 可证明的分批控制变量

把每个局部 score 写成 \(r_g=a_g+\delta_g\)，其中 \(a_g\) 是可精确计算的 anchor，\(\delta_g\) 是随机小批量残差。使用

\[
 \widehat R=R_a+\frac{G}{M}\sum_{j=1}^M\widehat\delta_{I_j},
\qquad E[\widehat R\mid x]=R,
\]

并用两份条件独立批量 \(A,B\) 构造

\[
 E[\widehat R_A^\top\widehat R_B\mid x]=\|R\|^2.
\]

若每份估计协方差为 \(\Sigma_R\)，则

\[
 \operatorname{Var}(\widehat R_A^\top\widehat R_B\mid x)
 =2R^\top\Sigma_RR+\operatorname{tr}(\Sigma_R^2).
\]

这只校正平方项的瞬时期望；它不校正随机 Euler 中的 \(\operatorname{Cov}(\widehat b,\widehat g)\) 和 \(\operatorname{Cov}(\widehat b):\nabla^2f\)。若 anchor 使 \(\Sigma_R\) 从 \(O(G/M)\) 降到 \(O(V_\delta/M)\)，则平方项方差按同一 \(V_\delta\) 缩减；这是一项可检验的控制变量效应，不是无偏性本身的新颖性。最终实验必须把全量 \(R,g\)、普通单批量、独立双批量平方校正和 anchor 残差校正放在相同 \(P,M,h\) 与 wall-clock 下比较。
