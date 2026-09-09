# 一手文献与研究边界

核查日期：2026-09-09。此文件服务于 `RESEARCH_PROGRAM.md`，记录已经读到的能力与待解决问题；“没有搜索到”不构成新颖性证据。

## R1：组合推断与误差抑制

Arruda et al., *Compositional amortized inference for large-scale hierarchical Bayesian models*，ICLR 2026；作者 arXiv 页面注明正式发表。当前核查版本 v5，2026-04-07。

- [作者全文](https://arxiv.org/html/2505.14429v5)
- [官方代码](https://github.com/bayesflow-org/hierarchical-abi)
- [作者维护的 BayesFlow 教程](https://bayesflow.org/v2.0.11/_examples/Compositional_Diffusion.html)

已核查摘要、引言、第 3.2 节、第 4.1 节相关正文。文章明确讨论复合 score 的近似误差累积，提出沿 diffusion 路径变化的 damping、mini-batch estimator 及噪声调度，并在大型分层模型中验证。其已展示能力覆盖独立组增多、可组合推断和数值稳定性；不能把这些概念重新命名为 PFN 创新。

第 3.2 节式 (7)–(9) 中 damping 满足 d(0)=1，在高噪声端减弱累积贡献；文中强调直接永久缩放会改变目标后验。第 4.1 节报告大数据规模下校准和 KL 仍会变差，调整调度有所缓解。这是作者结果，尚未复现。

官方代码已固定于 `f01a3add0b02e420fcf8b61e629f6b202954b4b6`。针对其允许的指数 damping，已建立局部 score 完全精确时的终点偏差反例，见 `ORACLE_COMPOSITION_AUDIT.md`。这是特定解析条件的独立审计，未复现作者的神经模型与全部超参数实验。

## R2：分层上下文学习

Zhu, Oermann, Cho, *Multi-Task Bayesian In-Context Learning*，ICML 2026；[官方会议列表](https://icml.cc/Downloads/2026)、[作者全文 v1](https://arxiv.org/html/2606.20538v1)。

第 5.2.2 节已有正确层级推断与错误 pooling 对照；附录 G.2 已有组数变化与超过训练范围的实验。其误差增长不能未经复现就归因于错误证据计数。若比较，双方必须得到相同组身份、先验信息和观测。

## R3：PFN 统计基础

Nagler, *Statistical Foundations of Prior-Data Fitted Networks*，ICML 2023；[正式论文页面](https://proceedings.mlr.press/v202/nagler23a.html)。

建立了关于预测器敏感性、方差和局部化偏差的统计框架。已有历史 Gaussian regret 与敏感性结果不能重新作为未探索的问题。新的依赖数据问题应明确它在哪些假设上超出已有理论。

## R4：贝叶斯上下文聚合

Volpp et al., *Bayesian Context Aggregation for Neural Processes*，ICLR 2021；[官方页面](https://iclr.cc/virtual/2021/poster/3148)。

研究助手已核查其 Bayesian aggregation 与 task ambiguity 设定。需要进一步精读才可声称具体定理不覆盖当前候选。仅使用乘积或 Bayesian aggregation 不是足够的方法创新。

## R5：真实扩散后验的修正

Linhart et al., *Diffusion posterior sampling for simulation-based inference in tall data settings*，TMLR 2026，[作者全文 v3](https://arxiv.org/html/2404.07593v3)。

已核查第 2.3、3.1–3.3 节与附录 M。式 (12) 给出 backward-kernel correction，GAUSS/JAC 对真实 Gaussian components 可恢复精确目标 score。JAC 的状态依赖协方差、停止相应梯度与残差 F，以及 GAUSS 的固定协方差近似构成具体边界。独立代数核验 297 个条件通过。一般的 Gaussian path correction 已被覆盖。

## R6：一般组合路径的粒子校正

[Soiffer et al., arXiv:2606.23920v1](https://arxiv.org/html/2606.23920)，2026-06-22 预印本。已核查第 4.1 节与附录 C 的关键公式：已有高斯 ODE 组合偏差分析，并明确区分 inference-time approximation 与 score estimation error；Feynman–Kac 校正已有前作。本项目不把“完美 score 仍存在组合偏差”作为首次发现。

[Lee et al., ACE, arXiv:2512.10339v2](https://arxiv.org/html/2512.10339)，2026-06-01 版本。已核查第 2.4 节 Theorem 2.3，时变指数需要相应粒子权重修正。一般时变 damping 加粒子修正已有先例。尚未核查其全部实现和实验，也没有复现。

## 候选判定

当前没有已通过新颖性检验的新算法。正在集中论证方差可控的小批量后验校正。已建立的障碍是无偏 score 代入平方校正会额外偏置；交叉批量估计可以恢复瞬时期望，但其方差和指数权重偏差仍待处理。经典控制变量或随机权重 SMC 若已完整覆盖拟议构造，则结束该候选。具体论证要求见 `ORACLE_COMPOSITION_AUDIT.md`。
