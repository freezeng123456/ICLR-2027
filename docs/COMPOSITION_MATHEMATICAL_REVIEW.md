# Composition manuscript: independent mathematical review

日期：2026-09-09。审查范围为 manuscript/main.tex、manuscript/appendix.tex、gaussian_integrability.py 与 composition_benchmark.py。本次只读，没有启动 GPU 或扩大实验。

## 经核验成立的部分

1. **Gaussian 一步递推。** 对条件状态 \(N(\mu,V)\)、quadratic potential \(\widehat c x^2+O(|x|)\)、affine drift \(-((1+\kappa)\widehat A+\kappa)x/2+O(1)\)，正权重 Gaussian 积分的严格条件确实是
   \[
   D=1-2h\widehat cV>0.
   \]
   有限时
   \[
   V^+=
   \frac{[1-h((1+\kappa)\widehat A+\kappa)/2]^2V}{D}
   +\kappa h.
   \]
   均值和线性项不影响 denominator。gaussian_integrability.py 的 denominator、multiplier、next_variance 与该公式一致。

2. **普通 U / naive 系数。** 当前论文使用 \(s^2=M^{-1}\sum_i(a_i-m)^2\) 的 population variance，公式
   \[
   c_U=\frac12\left[G(G-1)m^2-\left(\frac{G^2}{M-1}+G\right)s^2\right],
   \qquad
   c_N=\frac12[G(G-1)m^2-Gs^2]
   \]
   与实现一致。若改用分母 \(M-1\) 的 sample variance，系数必须相应变化；不能混用两种 convention。

3. **U-estimator 无偏性。** 有放回独立抽样下
   \[
   E\left[\left(\sum_i a_{I_i}\right)^2-\sum_i a_{I_i}^2\right]
   =
   M(M-1)\left(\frac{\sum_g a_g}{G}\right)^2.
   \]
   因此代码中的第一项期望为 \((\sum_g a_g)^2\)，第二项期望为 \(-\sum_g a_g^2\)，恰好得到 full potential。这个推导不适用于 without-replacement 抽样。

4. **两极值递推。** 无控制的 \(c_B\) 在固定 batch mean \(m\) 时随 \(s^2\) 递减，故最大 \(c\) 由 constant batch 给出。令
   \[
   f(m)=\frac{V(a-bm)^2}{1-dm^2}+\kappa h,
   \]
   在公共严格有效域内
   \[
   \frac{V(a-bm)^2}{1-dm^2}
   =
   V\sup_z\{2z(a-bm)-z^2(1-dm^2)\}.
   \]
   右侧对 \(m\) 是 convex quadratic 的上确界，所以最大值在 \(a_{\min}\) 或 \(a_{\max}\)。constant batch 对任何有限 \(M\) 都有正概率。gaussian_integrability.py 的 verify() 对 \(M=2,3,4\)、多个 steps 和两种 estimator 逐 history 与极值递推比较，逻辑上覆盖了这一点。

5. **四因子反例。** fractions 核验通过：
   \[
   a=\left(\frac1{21},\frac1{11},\frac12,\frac23\right),\quad
   V=\frac{154}{355},\quad h=\frac12,
   \]
   full denominator 为 \(2503/3195>0\)，全抽最强 group 的事件上 \(\widehat c=8/3\)，denominator 为 \(-167/1065<0\)。该事件对任意有限 \(M\) 的概率为 \(4^{-M}>0\)，Tonelli 给出总体正权重 normalizer 发散。

6. **tail envelope。** 附录的标量证明使用
   \[
   \log p(x)=-x^2/(2V)+O(|x|)
   \]
   与 \(g(x)=cx^2+O(|x|)\) 的上下 Gaussian 包络。\(D>0\) 时 variance parameter 变为 \(V/D\)；再经过 bounded displacement 和 \(\kappa h>0\) Gaussian convolution，输出 quadratic parameter 为 \(A^2V/D+\kappa h\)。证明不需要 bounded displacement 可微。\(D<0\) 发散、\(D=0\) 留白，范围正确。

7. **paired correction。** paired weight 的负平方差修正只会减小正权重。去掉修正后的方差递推在 \((A,c)\) 的正 denominator 域上是 jointly convex，因此 unpaired strict-positive certificate 给出 paired 的上界。反向使用每一步相同的 maximizing constant batch 时，平方差修正为零，故严格 negative certificate 也传给 paired。这个正/负两侧论证成立；等号在 mixture 情况应继续留白。

## 需要收紧的表述

- main theorem 的“if some \(D_z\le0\), population normalizer is infinite”依赖于当前时刻的 \(V_{\max}\) 已由此前所有 history 的最大值递推得到，并且每个 batch 在每步有正条件概率。正文应保留这两个条件；单独对任意候选 \(V\) 代入极值不能推出全局结论。

- arbitrary affine control proposition 使用 \(\widetilde c\) 删除非正的 \(s^2\) 项。其凸性证明是正确的，但必须明确：这是上界；在 simplex vertices 上 \(s^2=0\)，所以极值处上界取等，才可把上界变成 exact certificate。正文目前已基本这样写，不能删掉“vertex 上取等”这一句。

- “same strict integrability domain”只指离散 weighted Euler population operator 的 strict positive/negative denominator domain。它不表示 paired、tail-CV 与 full 的 finite-step endpoint law 相同，也不覆盖 \(D=0\)、一般非对角 covariance、without-replacement batch 或 learned score tails。

- mixture tail theorem 的 induction 需要有限 batch-history 集合、每个历史的 finite positive normalizer，以及每一步 bounded residual 常数在有限时间网格上统一有界。当前附录提到 finite histories，但这些 uniformity 条件应在 theorem 假设或 proof 中明确。

- main 中“arbitrary affine control”只适用于 residual tail slopes 已被精确分解、且控制和 residual 都给出标量 affine tails 的情形。对 moment-matched mixture score，残差通常仍有线性项；其 batch quadratic coefficient 会改变，不能自动套用 tail-CV 的 batch-independent conclusion。

## 最终判断

核心 Proposition 的 convex-perspective 证明、two-extreme recursion、paired strict-domain transfer、Gaussian tail envelope 与四因子 exact counterexample 在当前假设下没有发现数学反例。主要风险是适用范围和 convention 被读者扩大解释：结果必须限定为 with-replacement finite batches、非负 scalar/diagonal Gaussian tails、固定 weighted Euler、strict inequalities。若论文保留这些限定，当前证书可以作为可核验的 population-integrability 结果；它不能支持一般随机权重 SMC 或一般 learned score 的可积性结论。
