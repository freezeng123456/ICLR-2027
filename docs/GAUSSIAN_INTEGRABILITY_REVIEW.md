# Gaussian 随机批量权重的可积性审查

日期：2026-09-09。审查对象为 gaussian_integrability.py。本文只核验其 Gaussian population 一步递推、Tonelli 论证和有限批量反例，不修改实现。

## 结论

当前递推是正确的，且比逐条枚举 batch history 更强：在每一步所有 batch history 都有正概率、一步映射在有效域对输入方差单调递增时，最大的可达方差可以递推得到。若该最大方差对应的某个下一步 denominator 不正，则总体 Feynman–Kac normalizer 已经是无穷大。

对 \(\lambda=(0.1,0.2,2,4)\)，maximum_u=20、\(M=2\) 的独立计算得到：

| steps | full | naive / unbiased 首次失败 |
|---:|---:|---:|
| 16 | finite | 14 |
| 32 | finite | 27 |
| 64 | finite | 55 |
| 128 | finite | 111 |
| 256 | finite | 221 |
| 512 | finite | 443 |

128 步的失败点是 \(u=0.3955078125\)，\(h=0.042724609375\)，最大可达的前一步方差为 \(13.0706247822\)，最小 denominator 为 \(-8.1364890933\)。full 六个步数均有限；这只说明这些网格下 full 路径有限，不说明任意步长或任意总时间都有限。

## 一步 Gaussian 递推

在 \(u\)-time VP Euler 步中，给定 batch 的 \(\widehat A\) 与 \(\widehat c\)，代码使用

\[
 \widehat b(x)=-(\widehat A+\tfrac12)x,\qquad
 \widehat w(x)=\exp(h\widehat c x^2).
\]

若当前粒子是 \(N(0,V)\)，则 Gaussian 积分分母为

\[
 d=1-2h\widehat cV.
\]

当 \(d>0\) 时，归一化后的方差为

\[
 F_{\widehat A,\widehat c}(V)
 =\frac{(1-h(\widehat A+\tfrac12))^2V}{1-2h\widehat cV}+h.
\]

这与 audit_path 的 denominator、multiplier 和 next_variance 三行逐项一致。若 \(d\le0\)，Gaussian 二次指数积分发散；权重始终非负，所以这不是正负抵消造成的数值问题。

在有效域 \(d>0\) 内，

\[
 F'(V)=\frac{(1-h(\widehat A+\tfrac12))^2}{(1-2h\widehat cV)^2}\ge0.
\]

因此若 \(V_k^\star\) 是第 \(k\) 步所有正概率 history 的最大可达方差，那么对每个当前 batch 只需代入同一个 \(V_k^\star\)；取结果最大值即可得到 \(V_{k+1}^\star\)。若某个 batch 在该输入上 \(d\le0\)，它与达到 \(V_k^\star\) 的历史拼接后仍是正概率 history，故总体 normalizer 由 Tonelli 定理为无穷大。这里需要的假设是 batch 抽样在各步有正概率、批量与此前 history 条件独立，且权重非负；不需要枚举全部历史。

## \(\widehat A,\widehat c\) 的核验

写 \(a_g\) 为当前 \(u\) 下的局部 score 强度，\(G=4\)，\(M=2\)。有放回 batch 的

\[
 \widehat A=\frac{G}{M}\sum_{j=1}^M a_{I_j},
\]

是 \(\sum_g a_g\) 的无偏估计。两两项的当前代码估计为

\[
 \widehat c
 =\frac12\left[
 \frac{G^2}{M(M-1)}
 \left\{\left(\sum_j a_{I_j}\right)^2-\sum_j a_{I_j}^2\right\}
 -\frac{G}{M}\sum_j a_{I_j}^2
 \right].
\]

直接取期望：有放回独立抽样下
\[
 E\left[\left(\sum_j a_{I_j}\right)^2-\sum_j a_{I_j}^2\right]
 =M(M-1)\left(\frac{\sum_g a_g}{G}\right)^2.
\]
因此第一项的系数 \(G^2/[M(M-1)]\) 恰好给出 \((\sum_g a_g)^2\)，第二项给出 \(-\sum_g a_g^2\)，所以

\[
 E[\widehat c]
 =\frac12\left[(\sum_g a_g)^2-2\sum_g a_g^2\right],
\]

这正好等于 full 中当前代码的 \(\frac12[(\sum_g a_g)^2-\sum_g a_g^2]\)。因此该 U-estimator 对 full target 无偏；此前将第一项误写成排除重复组的有限总体平方项是不正确的。naive 与 unbiased 在 \(M=2\) 的失败轨迹仍可能相同，是因为当 batch 两次抽到同一组时，两者的实现公式逐项相同。

这项算术差异不影响 integrability 反例，因为两种方法在全抽到最强组的事件上都有相同的

\[
 \widehat A=Ga_{\max},\qquad
 \widehat c=\frac12G(G-1)a_{\max}^2>0.
\]

但它必须在任何“unbiased potential”表述中修正或降级为“对指定平方组合的估计”。

## 任意有限 \(M\) 的正概率失败事件

令每一步 batch 都抽到强度最大的 group。该事件在有放回 uniform batch 下的概率为

\[
 (1/G)^{M}
\]

每一步均为正；固定 \(K\) 步的 history 概率为 \(G^{-MK}>0\)。在该 history 上，\(\widehat A=Ga_{\max}\)，而 naive 与当前 unbiased 公式均给出

\[
 \widehat c=\frac12G(G-1)a_{\max}^2.
\]

当输入方差 \(V\) 与步长满足

\[
 2h\widehat cV\ge1,
\]

该单条 history 的 Gaussian normalizer 已发散。于是任意有限 \(M\) 都没有自动的可积性保证；增大 \(M\) 只降低失败 history 的概率，不能把正概率无限贡献变成有限贡献。不能据此声称所有 step schedule 都失败：若每一步 \(h\) 足够小、\(V\) 受控，denominator 可以始终为正。

一个完全有理数的单步反例直接使用代码的 VP path。取 \(u=\log 2\)、\(h=1/2\)、\(G=4\)、\(\lambda=(0.1,0.2,2,4)\)。此时

\[
 a_g=\frac{\lambda_g}{2+\lambda_g}
 =\left(\frac1{21},\frac1{11},\frac12,\frac23\right),
\qquad
 A=\sum_g a_g=\frac{201}{154}.
\]

取输入 path \(\pi_u=N(0,V)\)，其中 \(V=154/355\)。full potential 为
\(c_{\mathrm{full}}=346/693\)，且其 denominator（这里 \(2h=1\)）为

\[
 1-c_{\mathrm{full}}V=\frac{2503}{3195}>0.
\]

对任意有限 \(M\ge2\)，考虑 iid with-replacement batch 全部抽到最后一个 group 的事件。其概率为 \(4^{-M}>0\)，并且该事件上的批量估计满足
\(\widehat A=G(2/3)\)，
\[
 \widehat c=\frac12G(G-1)\left(\frac23\right)^2=\frac83.
\]

因此该事件上的 denominator 为

\[
 1-\widehat cV
 =1-\frac83\frac{154}{355}
 =-\frac{167}{1065}<0.
\]

所以单步 normalizer 已经无穷大，且这一结论对每个有限 \(M\) 都成立；drift 的数值不会改变 \(Q_h1\) 的发散性。这个反例同时显示：瞬时 drift/potential 可以无偏、batch variance 可以随 \(M\) 降低，但每个有限 \(M\) 的指数权重仍可能不可积。

若只需要一个直观的浮点反例，也可取 \(G=4\)、\(a_{\max}=4\)、\(V=1\)、\(h=0.01\)。全抽到最强 group 时 \(\widehat c=96\)，故

\[
 1-2h\widehat cV=1-1.92=-0.92<0.
\]

该事件对任何有限 \(M\) 都有概率 \(4^{-M}>0\)。若需要两步反例，固定第一步得到任何有限 \(V_1\ge1\) 的有效状态，再在第二步使用同一事件；只要 \(h\ge(192V_1)^{-1}\)，第二步立即发散。这个构造是固定 switch schedule 的可证明反例，不依赖对所有 Euler 步数的全局断言。

## 与普通有限方差分析的边界

有限方差或无偏 score 只能控制多项式函数的期望，例如 \(E\widehat A\) 或在独立双批量下的平方项；它不能保证 \(E\exp(h\widehat cX^2)\) 有限。这里的障碍是 exponential integrability，判据是每条正概率 history 的 quadratic denominator，而非 \(\operatorname{Var}(\widehat A)\) 是否有限。

若希望获得可实施保证，必须显式强制 \(\widehat c\le c_{\max}\)、选择 \(h<1/(2c_{\max}V_{\max})\)，或采用截断/tempering 并量化由此产生的偏差。仅增加粒子数 \(P\) 不能修复 population operator 的无穷 normalizer；粒子数只影响有限粒子近似的方差。

## 设计定理：共享 covariance mixture 的 tail control variable

考虑第 \(g\) 个局部 posterior 是若干 component 的 Gaussian mixture，所有 component 共享 covariance \(v_g(u)I\)，component mean 为 \(\alpha(u)\mu_{g,k}\)，并假设 \(\mu_{g,k}\) 的集合有界。令 \(\bar\mu_g\) 为 component mean 的参考平均，选择

\[
 s_{0,g}(x)=-\frac{x-\alpha\bar\mu_g}{v_g(u)}.
\]

混合 score 满足

\[
 s_g(x)-s_{0,g}(x)
 =\frac{\alpha}{v_g(u)}
 \left(E[\mu_{g,K}\mid x]-\bar\mu_g\right).
\]

由于条件平均仍位于 component mean 的凸包内，若
\(\max_k\|\mu_{g,k}-\bar\mu_g\|\le B_g\)，则

\[
 \|s_g-s_{0,g}\|\le \frac{|\alpha|B_g}{v_g(u)}
\]

在整个 \(x\)-空间一致成立。这个界要求 \(v_g(u)>0\)；若 \(v_g\) 随 \(u\) 变化，需在所用时间区间给出正下界。

设组合 potential 的 score 平方差分使用 U-estimator，并把每个 score 写成 \(s_{0,g}+\delta_g\)，其中 \(\delta_g=s_g-s_{0,g}\) 有界。任何 batch 上的 quadratic term 只来自确定的 \(s_0\) 部分；含 \(\delta_g\) 的交叉项至多为 \(O(|x|)\)，残差平方项为 \(O(1)\)。因此每条 batch history 的 potential 都有共同的尾部形式

\[
 \widehat c(x)=c_0x^2+O(|x|)+O(1),
\]

其中二次系数 \(c_0\) 与 batch 无关。若输入 density 满足

\[
 \log p(x)=-\frac{x^2}{2V}+O(|x|),
\]

则对固定 \(h>0\)：

\[
 1-2hc_0V>0
\quad\Longrightarrow\quad
 \int p(x)e^{h\widehat c(x)}\,dx<\infty
\]

对 full 权重和所有 tail-CV batch 权重同时成立；而

\[
 1-2hc_0V<0
\]

时二次项已经压倒所有线性余项，二者同时不可积。等号时线性项、常数项和 density 的次二阶项共同决定结论，不能对一般 mixture 作统一判断。

证明只用尾部夹逼：对任意 \(\epsilon>0\)，存在 \(C_\epsilon\) 使
\[
 -\frac{x^2}{2V}-(L+\epsilon)|x|-C_\epsilon
 \le \log p(x)
 \le -\frac{x^2}{2V}+(L+\epsilon)|x|+C_\epsilon,
\]
且 \(|\widehat c-c_0x^2|\le L|x|+C\)。严格正的二次系数差给出 Gaussian 型可积上界；严格负时给出发散下界。由于每个 batch history 的权重非负，这一结论也可用 Tonelli 推到 history 混合的 normalizer。

这个保证不适用于直接对 moment-matched Gaussian score 使用同一 control：其 score 残差通常仍含未消失的线性 \(x\) 项，batch 间 quadratic coefficient 会改变，必须重新计算各 history 的 tail coefficient，不能套用共享-covariance 定理。

## 9. 非负 strength 下的 batch-size-insensitive 上界

还可以避免 \(G^M\) 个 batch 的枚举。令 batch 中的 strength 为 \(z_1,\ldots,z_M\ge0\)，
\[
 m=\frac1M\sum_jz_j,\qquad
 s^2=\frac1M\sum_j(z_j-m)^2.
\]
将代码中的 U potential 直接化简，得到
\[
 \widehat c_U
 =\frac12\left[
 G(G-1)m^2-
 \left(\frac{G^2}{M-1}+G\right)s^2
 \right],
\]
而 naive potential 为
\[
 \widehat c_N
 =\frac12\left[G(G-1)m^2-Gs^2\right].
\]
这里 \(s^2\) 使用分母 \(M\) 的 batch population variance。两个恒等式可由 \(\sum_jz_j^2=M(m^2+s^2)\) 直接验证；数值枚举由 `gaussian_integrability.py` 保存。

固定 \(m\) 时，\(\widehat c\) 随 \(s^2\) 递减。若 \(\kappa\ge0\) 为扩散系数，递推写成
\[
 F(m,s^2;V)
 =\frac{\left(1-h((1+\kappa)Gm+\kappa)/2\right)^2V}
 {1-2h\widehat c(m,s^2)V}+\kappa h.
\]
在 denominator 为正的共同有效域内，\(F\) 随 \(\widehat c\) 递增，所以固定 \(m\) 时最大值由 \(s^2=0\) 给出。此时
\[
 F_0(m;V)=\frac{(a-bm)^2V}{1-dm^2}+\kappa h,
\quad
 a=1-\frac{\kappa h}{2},\
 b=\frac{(1+\kappa)Gh}{2},\
 d=hG(G-1)V.
\]
只要 \(1-dm^2>0\)，有恒等式
\[
 \frac{(a-bm)^2V}{1-dm^2}
 =V\sup_z\{2z(a-bm)-z^2(1-dm^2)\}.
\]
右侧对 \(m\) 是凸函数的上确界，因此 \(F_0\) 在 \([a_{\min},a_{\max}]\) 上取最大值于端点。端点 \(m=a_{\min}\) 和 \(m=a_{\max}\) 分别由 batch 全部抽到最小或最大 strength 的事件实现，二者对任意有限 \(M\ge2\) 都有正概率。

由此，在每一步的共同 denominator 对所有 batch 都为正时，最大可达方差只需把上一层最大方差分别代入两个端点；若某个端点 denominator 不正，则该端点 history 已使总体 normalizer 发散。对 U 与 naive，各步只需计算两个极值，复杂度为 \(O(G)\)（若 strength 已排序则为 \(O(1)\) 每步），不需要 \(G^M\) 枚举。这个结论要求 strengths 非负、\(\kappa\ge0\)、输入方差 \(V>0\)，以及在递推时考察的 denominator 处于共同有效域；一旦某个候选 denominator 不正，发散结论直接成立。

这也覆盖非零 Gaussian mixture mean 的 tail-CV 场景：均值只改变线性和常数尾项，不改变上述 quadratic coefficient 的端点判据。若 control 残差含未受控的线性 strength 或二次尾项，则不能使用本节结论。
