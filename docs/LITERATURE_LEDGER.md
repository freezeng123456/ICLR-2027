# 一手文献与研究边界

核查日期：2026-09-09。此文件服务于 `RESEARCH_PROGRAM.md`，记录已经读到的能力与待解决问题；“没有搜索到”不构成新颖性证据。

## R1：组合推断与误差抑制

Arruda et al., *Compositional amortized inference for large-scale hierarchical Bayesian models*，ICLR 2026；作者 arXiv 页面注明正式发表。当前核查版本 v5，2026-04-07。

- [作者全文](https://arxiv.org/html/2505.14429v5)
- [官方代码](https://github.com/bayesflow-org/hierarchical-abi)
- [作者维护的 BayesFlow 教程](https://bayesflow.org/v2.0.11/_examples/Compositional_Diffusion.html)

已核查摘要、引言、第 3.2 节、第 4.1 节相关正文。文章明确讨论复合 score 的近似误差累积，提出沿 diffusion 路径变化的 damping、mini-batch estimator 及噪声调度，并在大型分层模型中验证。其已展示能力覆盖独立组增多、可组合推断和数值稳定性；不能把这些概念重新命名为 PFN 创新。

第 3.2 节式 (7)–(9) 中 damping 满足 d(0)=1，在高噪声端减弱累积贡献；文中强调直接永久缩放会改变目标后验。第 4.1 节报告大数据规模下校准和 KL 仍会变差，调整调度有所缓解。这是作者结果，尚未复现。

待研究：数值轨迹稳定性与零噪声端学习 score 的条件系统偏差是否可分离；我们的高斯偏差分解只是理解这一问题的基础，尚无超出该文和相关 SBI 理论的贡献。需核查完整证明和官方代码后再作判断。

## R2：分层上下文学习

Zhu, Oermann, Cho, *Multi-Task Bayesian In-Context Learning*，ICML 2026；[官方会议列表](https://icml.cc/Downloads/2026)、[作者全文 v1](https://arxiv.org/html/2606.20538v1)。

第 5.2.2 节已有正确层级推断与错误 pooling 对照；附录 G.2 已有组数变化与超过训练范围的实验。其误差增长不能未经复现就归因于错误证据计数。若比较，双方必须得到相同组身份、先验信息和观测。

## R3：PFN 统计基础

Nagler, *Statistical Foundations of Prior-Data Fitted Networks*，ICML 2023；[正式论文页面](https://proceedings.mlr.press/v202/nagler23a.html)。

建立了关于预测器敏感性、方差和局部化偏差的统计框架。已有历史 Gaussian regret 与敏感性结果不能重新作为未探索的问题。新的依赖数据问题应明确它在哪些假设上超出已有理论。

## R4：贝叶斯上下文聚合

Volpp et al., *Bayesian Context Aggregation for Neural Processes*，ICLR 2021；[官方页面](https://iclr.cc/virtual/2021/poster/3148)。

研究助手已核查其 Bayesian aggregation 与 task ambiguity 设定。需要进一步精读才可声称具体定理不覆盖当前候选。仅使用乘积或 Bayesian aggregation 不是足够的方法创新。

## 候选判定

当前没有已通过新颖性检验的主线。优先检查 R1 的残余统计误差与其已解决的数值误差之间是否存在有价值、可证明的区别，同时追溯 simulation-based calibration 与 approximate likelihood 的相关理论。若强基线已完整覆盖，则结束当前候选；保留解析参照用于下一个有依据的问题。
