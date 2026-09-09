# Prefix/suffix certificate: final proof review

日期：2026-09-10。审查对象为 manuscript/main.tex 的无放回段落、manuscript/appendix.tex 的 Without-replacement comparison，以及相关 Gaussian implementation。未修改代码或论文。

## 审查结论

当前 Prefix/suffix certificate 定理的证明在声明的范围内成立，没有发现剩余的实质性数学反例。范围是 scalar Gaussian factors、\(a_g\ge0\)、均匀无放回固定大小 \(M\)、固定有限时间网格、\(\kappa>0\)，以及非正 denominator 通过非负权重 Tonelli 判定发散。

无放回势的系数当前正确：

\[
 \widehat g_{\mathrm{WOR}}
 =
 \frac{G(G-1)}{M(M-1)}
 \sum_{i<j,\ i,j\in S}r_i^\mathsf Tr_j.
\]

这是 unordered pair 写法。等价的 ordered-pair 写法需使用
\[
 \frac{G(G-1)}{2M(M-1)}
 \sum_{i\ne j,\ i,j\in S}r_i^\mathsf Tr_j.
\]
它不能与 with-replacement U 公式混用。无放回 inclusion probabilities 给出条件无偏性。

## 证明核验

固定 \(M-1\) 个已选 strength，改变最后一个未选 strength \(z\)。subset sum 与 distinct pair sum 中 \(z\) 都只出现一次，因此 \(\widehat A(z)\)、\(\widehat c(z)\) 都是 affine。若 denominator 严格为正，

\[
 F(z)-\kappa h
 =
 \frac{V n(z)^2}{D(z)}
 =
 V\sup_t\{2tn(z)-t^2D(z)\}
\]

是 affine functions 的上确界，因而在可行 strength 区间的端点取最大值。对 denominator failure，只需先最大化 affine 的 \(\widehat c(z)\)，同样在端点取得。

反复把 selected interior index 换成当前未选 index 的最小端点或最大端点，目标不降低，且未选 index 的包围区间严格缩小。有限次后，未选 index 连续，selected set 必为

\[
 S_j=\{1,\ldots,j\}\cup\{G-(M-j)+1,\ldots,G\}.
\]

这给出单步 \(M+1\) 候选。由于一步递推 \(F_S(V)\) 在 denominator 正时满足

\[
 \frac{\partial F_S}{\partial V}
 =
 \frac{n_S^2}{(1-2h\widehat c_SV)^2}\ge0,
\]

把上一时刻最大 reachable variance 与当前最大候选拼接，归纳给出多步 exact recursion。每个 prefix/suffix subset 都有正概率，因此最大值可达；某个候选 denominator 非正时，正概率 history 的非负权重积分发散，Tonelli 判定 population normalizer 发散。

重复 strengths 只造成 ties，不影响按 index 排序或交换终止。Gaussian nonzero means 和 linear potential terms 只改变 Gaussian tilted mean；quadratic precision、denominator 与 variance recursion 不变。

## 对角与 mixture 范围

对于 diagonal/product Gaussian，坐标 conditional integrability 可逐坐标判定。若一个坐标失败，其 witnessing history 直接使联合非负积分发散；若所有坐标都通过，则有限 history 下每个联合积分有限。不同坐标的最大值可以由不同 history 取得，这不影响“存在任一失败坐标即失败”和“所有坐标通过即通过”的联合判定。

equal-component-variance Gaussian mixtures 只可使用论文已给出的 tail-envelope extension，并保留 strict positive/negative denominator 范围；zero denominator 不作一般结论。一般 affine control 不应套用 prefix/suffix 排序，因为 residual control 后不再由单一 strength 变量决定。tail-matched control 的 batch-independent quadratic coefficient 可直接使用其独立 tail proof。

## 最终范围判断

当前证明支持论文对 finite uniform without-replacement subsets 的 \(M+1\) certificate。它不覆盖 adaptive subset selection、非均匀抽样、without-replacement 与 with-replacement 的混合策略、非对角协方差或一般 learned score tails。数值 exhaustive checks 可以作为实现复核，但理论结论依赖上述交换和单调性条件，而不是依赖枚举结果。
