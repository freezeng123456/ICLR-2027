# 随机 Feynman–Kac 算子的理论核验

日期：2026-09-09。完整定理、假设与证明见 `manuscript/main.tex` 和 `manuscript/appendix.tex`。本文件给出与实现一致的局部算子公式及核验入口。

## 局部归一化算子

在固定时间，令随机漂移和势满足条件无偏性，分别记为 \(\widehat b\) 和 \(\widehat g\)。随机批量在状态、粒子和时间步骤之间独立重新抽取。定义

\[
 Q_hf(x)=E[e^{h\widehat g(x)}f(x+h\widehat b(x)+\sqrt{\kappa h}\xi)],
 \qquad \Phi_h(\eta)(f)=\frac{\eta Q_hf}{\eta Q_h1}.
\]

使用全量系数的 Euler 算子记为 \(Q_h^0\)。设

\[
 C_b=\operatorname{Cov}_B(\widehat b),\quad
 c_{bg}=\operatorname{Cov}_B(\widehat b,\widehat g),\quad
 v_g=\operatorname{Var}_B(\widehat g).
\]

在足够光滑且存在可积支配函数、允许展开至三阶的条件下，

\[
 \Phi_h(\eta)(f)-\Phi_h^0(\eta)(f)
 =h^2\left[\frac12\eta(C_b:\nabla^2f)
 +\eta(c_{bg}^{\mathsf T}\nabla f)
 +\frac12\operatorname{Cov}_{\eta}(f,v_g)\right]+O(h^3).
\]

最后一项是输入分布上的协方差。此处的归一化在整个输入分布上进行。有限的势方差本身不足以保证指数权重可积，因此必须先验证 \(0<\eta Q_h1<\infty\)。该局部恒等式不提供多步误差率或有限粒子收敛率。

两份独立批量的漂移取平均，使用权重

\[
 \exp\left[\frac h2(\widehat g_1+\widehat g_2)
 -\frac{h^2}{8}(\widehat g_1-\widehat g_2)^2\right]
\]

可以消去二阶势方差项。剩余系数为

\[
 \frac14\eta(C_b:\nabla^2f)+\frac12\eta(c_{bg}^{\mathsf T}\nabla f).
\]

## 总体可积性

Gaussian 条件状态方差为 \(V\)，势二次系数为 \(c\)，则指数权重可积当且仅当 \(D=1-2hcV>0\)。本文针对固定 weighted Euler 步骤证明：普通独立有放回估计只需比较最弱、最强因子构成的两个常量批量；任意 affine control 只需比较所有因子对应的常量批量。这些可达方差递推的判定与有限批量大小无关。

对于分量方差相同的 Gaussian mixture，精确保留共同二次尾部的控制变量与全量更新具有相同的严格可积域。该结论要求固定有限网格、正扩散系数、有限因子集合和标量或对角乘积结构。一般 mixture 的零分母边界没有分类。

## 可执行核验

- `gaussian_integrability.py`：精确分数恒等式、两极值递推与完全枚举比较、任意 affine control 递推与完全枚举比较，以及无放回的一步比较。
- `operator_audit.py`：四点输入分布、完整批量枚举、相同边际误差下的不同漂移和势耦合，检查三阶余项。
- `composition_benchmark.py` 的 `self_check()`：无偏估计、Gaussian control 精确性、尾部残差界和 Feynman–Kac 恒等式。
- `results/composition_theory_20260909/`：上述代数和数值检查、Gaussian population 递推及独立参考积分结果。

文献范围与新颖性判断见 `COMPOSITION_FINAL_PRIOR_ART_REVIEW.md`。Gaussian 积分、U-statistic 无偏恒等式、控制变量和局部 Taylor 展开均作为经典背景使用。
