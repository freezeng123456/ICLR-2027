# 无放回 prefix/suffix certificate 审查

日期：2026-09-10。本文审查 scalar nonnegative Gaussian residual precisions 的均匀无放回 batch；不修改论文或代码，不讨论一般 affine control。

## 结论

命题成立：固定 batch size \(M\) 时，单步最大可达 variance 只需检查 \(M+1\) 个集合

\[
 S_k=\{1,\ldots,k\}\cup\{G-(M-k)+1,\ldots,G\},
\qquad k=0,\ldots,M,
\]

其中 \(a_1\le\cdots\le a_G\)。这包含全最小、全最大以及所有 prefix/suffix 混合。结论允许重复的 \(a_i\)：排序按 index 任意打破 ties，重复值只会令若干候选数值相同。它不自动推广到任意 affine control，因为排序变量不再只由单个 strength 决定。

## 单步证明

固定当前输入 variance \(V>0\)，并先假设所有 \(M\)-subsets 的 denominator 都严格正。固定 \(M-1\) 个已选元素，令最后一个元素的 strength 为 \(z\)。写

\[
 A(z)=A_0+\alpha z,\qquad c(z)=c_0+\gamma z.
\]

这是因为 \(A\) 是 subset sum，pair sum 中含 \(z\) 的项只有 \(z\sum_{\mathrm{selected}}a_i\)，没有 \(z^2\)。于是

\[
 F(z)=
 \frac{V(\ell_0-\ell_1z)^2}{d_0-d_1z}+\kappa h
\]

其中 denominator 在当前未选集合上为正。直接微分，或使用 quadratic-perspective 表示

\[
 \frac{V(\ell_0-\ell_1z)^2}{d_0-d_1z}
 =
 V\sup_{t\in\mathbb R}
 \{2t(\ell_0-\ell_1z)-t^2(d_0-d_1z)\},
\]

可知 \(F(z)\) 是可行区间上的凸函数。因此固定其余 \(M-1\) 个元素时，把 \(z\) 替换为未选元素中的最小或最大 strength，不会降低 \(F\)。

重复此交换：若一个已选 index 位于当前最小未选 index \(l\) 与最大未选 index \(r\) 之间，则把它替换为 \(l\) 或 \(r\) 中使 \(F\) 较大的一个。替换后，被移出的 index 进入未选集合，且所选集合的 interior “夹层”减少；有限次后，已选 index 集合没有未选 index 位于其两侧的 interior gap，只能是 prefix 与 suffix 的并。选中 prefix 长度为 \(k\) 时正是 \(S_k\)。每次交换保持 subset 大小和无放回性质，因此最终候选不超过 \(M+1\) 个。

若某 subset 的 denominator 非正，则同一交换先应用于 coefficient \(c\)：固定 \(M-1\) 个元素时 \(c(z)\) 是 affine，故其最大值也在未选元素的两个端点取得。递归交换得到某个 \(S_k\) 的 \(c\) 不小于该 subset 的 \(c\)，于是该候选 denominator 也非正。正权重 Tonelli 立即给出发散。

因此单步严格结论是：

\[
 \max_{\lvert S\rvert=M}F(S)
 =
 \max_{0\le k\le M}F(S_k)
\]

在所有 denominator 严格正时成立；若存在非正 denominator，则只需检查同一 \(M+1\) 个候选即可发现发散。

## 多步递推

令 \(V_k^\star\) 为第 \(k\) 步所有正概率 batch histories 的最大 conditional variance。给定 \(V_k^\star\)，上面的单步结果应用于每个当前 subset，得到

\[
 V_{k+1}^\star
 =
 \max_{0\le j\le M}F_{S_j}(V_k^\star).
\]

每个候选 \(S_j\) 在均匀无放回抽样中都有正概率。达到 \(V_k^\star\) 的历史与该当前候选可以拼接，所以最大值可达；同时一步映射在 denominator 正的区域内关于输入 \(V\) 单调递增，因此任何其他历史都不超过该递推。归纳得到 exact population certificate。若某候选在 \(V_k^\star\) 上 denominator 非正，该 history 的非负权重积分发散，Tonelli 给出 population normalizer 发散。

## 数值核验

在 CPU 上对 \(G=4,\ldots,10\)、所有 \(2\le M<G\)、随机排序后的非负 strengths，以及随机 \(V,h,\kappa\) 做完整 \(\binom GM\) 枚举；逐次比较 all-subset 最大值与 \(M+1\) 个 \(S_k\) 候选，未发现差异。另对 \(G=4\)、\(M=2,3,4\) 逐项核验 denominator failure step 和 maximum variance recursion，结果一致。数值核验仅作为 proof 的实现检查，不替代上面的交换证明。

## 重复 strengths、Gaussian means 与 diagonal 情形

重复 strengths 不影响证明：只需对 index 排序；若交换产生相同 strength，目标值相同。Gaussian factor 的非零 mean 只会给 potential 添加 linear/constant terms。完成平方后，linear terms 只改变均值，variance 仍然等于 V/(1-2hcV)。因此可积性与最大条件 variance 的递推对非零均值同样成立；若需要计算完整 endpoint law，则还须单独跟踪均值和 mixture weights。

若模型是 diagonal/product Gaussian，固定 batch history 后各坐标条件独立。逐坐标使用 \(M+1\) 候选给出精确的联合可积性判定：一个坐标失败，其对应历史使联合非负积分发散；全部坐标通过，则每条历史均可积。不同坐标的最大 variance 可以由不同历史达到，判定不要求同时达到这些最大值。

## 无放回 U-estimator

本结论使用的 potential 是

\[
 \widehat g_U
 =
 \frac{G(G-1)}{2M(M-1)}
 \sum_{\substack{i,j\in S\\i\ne j}}r_i^\mathsf Tr_j.
\]

无放回 inclusion probabilities 给出条件无偏性；有放回的 pair probability 不能代入这里。prefix/suffix proof 只依赖 subset 上 \(A\) 与 pair sum 关于最后一个 strength 的 affine 性，不依赖 with-replacement 重复抽样。

该证书不适用于 arbitrary affine control 的一般排序变量；tail-matched control 若二次 coefficient 对所有 batch 完全相同，则可直接使用其已有 tail envelope proof，而非把本节的 scalar subset exchange 误套过去。
