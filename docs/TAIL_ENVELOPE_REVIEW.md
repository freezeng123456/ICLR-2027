# 二次尾包络与随机 Euler 可积性

日期：2026-09-09。本文给出一维标量 lemma，并说明它对共享 component covariance 的 Gaussian mixture factors 的适用范围。多维只延伸到 diagonal/product 情形；本文不声称一般协方差或任意非线性 drift 的结论。

## Lemma

设 \(X\) 有密度 \(p\)，并存在 \(V>0\)、\(L,C<\infty\)，使得

\[
 -\frac{x^2}{2V}-L|x|-C
 \le \log p(x)
 \le
 -\frac{x^2}{2V}+L|x|+C.
\tag{1}
\]

设一个 batch 的 potential 与 drift 满足

\[
 g(x)=cx^2+r(x),\quad |r(x)|\le L_g(1+|x|),
\qquad
 b(x)=Bx+q(x),\quad |q(x)|\le L_b.
\tag{2}
\]

令 \(h>0\)，\(\kappa>0\)，先乘非负权重 \(\exp(hg(X))\) 再归一化；记

\[
 D=1-2hcV.
\]

若 \(D>0\)，加权后的密度仍满足 (1) 的同类二次包络，且其 quadratic variance parameter 为

\[
 V_{\mathrm{tilt}}=\frac{V}{D}.
\tag{3}
\]

再令

\[
 Y=X+h\,b(X)+\sqrt{\kappa h}\,\xi
 =(1+hB)X+\widetilde q(X)+\sqrt{\kappa h}\,\xi,
\]

其中 \(\xi\sim N(0,1)\) 独立，\(\widetilde q=hq\) 有界。令 \(A=1+hB\)，则 \(Y\) 的密度满足

\[
 \log p_Y(y)
 =
 -\frac{y^2}{2V_Y}+O(|y|),
\qquad
 V_Y=A^2V_{\mathrm{tilt}}+\kappa h.
\tag{4}
\]

若 \(D<0\)，加权 normalizer 无穷大。若 \(D=0\)，仅凭 (1)–(2) 不能作统一判断。\(A=0\) 时，(4) 给出 \(V_Y=\kappa h\)，仍然成立。

## Proof

由 (1)–(2)，未归一化 tilted density 的对数上下界为

\[
 -\frac{D}{2V}x^2-(L+hL_g)|x|-C'
 \le \log[p(x)e^{hg(x)}]
 \le
 -\frac{D}{2V}x^2+(L+hL_g)|x|+C'.
\tag{5}
\]

若 \(D>0\)，两端都是可积的 Gaussian quadratic envelope。配方说明归一化后的上下包络都具有 quadratic coefficient \(D/(2V)\)，即 (3)，线性项只改变 \(O(|x|)\)。若 \(D<0\)，右侧权重在 \(|x|\to\infty\) 时至少按正二次指数增长，左侧下界给出发散积分。\(D=0\) 时只剩线性上下界，线性项的符号和更高阶项决定可积性，不能统一判定。

对输出密度，写 \(K=\|\widetilde q\|_\infty\)、\(\tau^2=\kappa h\)。条件 Gaussian 密度为

\[
 p_Y(y)=\int p_X(x)\,
 \varphi_\tau\!\left(y-Ax-\widetilde q(x)\right)\,dx.
\]

由于 \(|\widetilde q(x)|\le K\)，令 \(z=y-Ax\)，有逐点上下界

\[
 e^{-K|z|/\tau^2-K^2/(2\tau^2)}
 \le \frac{\varphi_\tau(z-\widetilde q(x))}{\varphi_\tau(z)}
 \le e^{K|z|/\tau^2}.
\]

将 tilted density 的包络乘入，基准 Gaussian 联合密度为 \(\varphi_{\sqrt{V_{\mathrm{tilt}}}}(x)\varphi_\tau(y-Ax)\)。其 \(Y\) 边际方差是

\[
 W=A^2V_{\mathrm{tilt}}+\tau^2.
\]

条件分布 \(X\mid Y=y\) 为 Gaussian，其均值是 \(AV_{\mathrm{tilt}}y/W\)，方差是 \(V_{\mathrm{tilt}}\tau^2/W\)。对任意固定非负 \(L_1,L_2\)，Gaussian 的绝对值指数矩与三角不等式给出
\[
 E[e^{L_1|X|+L_2|y-AX|}\mid Y=y]\le e^{C_1|y|+C_2}.
\]
下界由 Jensen 不等式给出
\[
 E[e^{-L_1|X|-L_2|y-AX|}\mid Y=y]
 \ge e^{-L_1E|X|-L_2E|y-AX|}
 \ge e^{-C_3|y|-C_4}.
\]
所以 \(p_Y(y)/\varphi_{\sqrt W}(y)\) 被 \(e^{\pm(C|y|+C')}\) 夹住，得到 (4)。当 \(A=0\) 时同一个论证仍然成立。证明只需要 \(q\) 有界可测及 \(\kappa h>0\)，无需连续性或可微性。

## 对共享 covariance Gaussian mixtures 的推论

若第 \(g\) 个 factor 的 component covariance 是 \(v_g(u)I\)，component mean 为 \(\alpha(u)\mu_{g,k}\)，且 component means 有界，则

\[
 r_g(x)=s_g(x)+x=-a_g(u)x+\rho_g(x),
\qquad \|\rho_g(x)\|_\infty<\infty.
\]

因此每条 finite batch history 的 U potential 都有

\[
 \widehat g_{\mathrm{U}}(x)=\widehat c_{\mathrm{U}}x^2+O(|x|),
\]

并且 drift 为 \(B_{\mathrm{batch}}x+O(1)\)。若 tail control variable 使用共同的 exact Gaussian part，batch 残差只产生线性或常数项，二次系数改为与 batch 无关的 \(c_{\mathrm{full}}\)。于是，在严格不等式

\[
 1-2hc_{\mathrm{full}}V>0
\]

下，full 和所有 tail-CV batch history 都处于同一个一步严格可积性域；先 tilt 再加 \(\kappa h\) Gaussian Euler noise 后，仍可用 (3)–(4) 递推。若严格小于零，full 和 tail-CV 的每条 history 都不可积；等号必须留白。

对 naive/U 未作 tail control 时，二次系数随 batch 改变，不能使用这个 batch-independent 推论。此时仍可使用此前的 strength min/max 端点递推：在共同严格有效域内，最大下一步 variance 由 \(s^2=0\) 且 \(m\in\{a_{\min},a_{\max}\}\) 给出。该端点判据对每个有限 \(M\ge2\) 都只需检查两个正概率事件。

## 多维范围

若 \(x\in\mathbb R^d\) 且 density、potential、drift 在同一坐标系中完全 diagonal/product，以上结论逐坐标成立；需要每个坐标的 \(D_j>0\)，输出 covariance 为

\[
 V_{Y,j}=A_j^2V_{\mathrm{tilt},j}+\kappa_j h.
\]

一般非对角 covariance 需要矩阵条件
\(I-2hV^{1/2}CV^{1/2}\succ0\)，并重新控制线性尾项；本文不把标量证明直接推广到该情形。
