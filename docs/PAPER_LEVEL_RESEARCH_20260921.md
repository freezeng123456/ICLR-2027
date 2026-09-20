# 方法定位与原始文献核查

## 组合扩散研究的具体问题

组合推断的目标是重复使用分别获得的条件分布，对多个条件的联合结果进行推断。Feynman–Kac Correctors 从密度演化方程推导粒子权重和重采样，对模型组合及其他目标变换提供采样构造。该工作已包含分子生成和图像条件生成等应用。因此，本项目的贡献不能描述为首次提出组合扩散、加权扩散或一般性的模型组合。

依据：[Skreta 等，2025，作者提供的摘要与版本信息](https://arxiv.org/abs/2503.02819)。本轮核查范围是摘要和出版信息，具体方程对应关系另由论文附录中的逐项推导给出。

本项目聚焦指定的指数加权 Euler 更新：每一步仅抽取部分因子时，形成的总体分布是否可以归一化。标量极端批次递推与不放回前缀/后缀递推是应重点说明的结构性结果。矩阵尾部推广使用经典高斯积分，贡献范围是把确定尾部曲率的设计与相关变量的严格可积域连接起来。

## 锚点控制的已有基础

利用控制变量减少随机梯度估计方差已有明确研究。Baker 等讨论随机梯度 MCMC 中的控制变量与计算复杂度，相关复杂度结论有目标分布假设。因此，锚点本身应定位为已有控制变量思想在本问题中的具体实现。本文需要说明额外要求：保留真实尾部斜率，并同时控制漂移和指数势函数；条件无偏不等于完整正核无偏。

依据：[Baker 等，Control Variates for Stochastic Gradient MCMC](https://arxiv.org/abs/1706.05439)。本轮核查范围是摘要和版本信息。

## 应当采用的问题求解对照

退火 SMC 使用一系列中间分布，将重要性加权、重采样和 Markov 更新组合起来。它适合本轮可计算观测似然的传感器问题。这个对照可以回答最终目标的求解成本，不能被省略为与组合扩散不同路径而不比较。

依据：[Del Moral、Doucet 与 Jasra，Sequential Monte Carlo Samplers](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf)，原论文第 1–4 节。

pCN 提议保持高斯参考测度；在本实验中，温度为 beta 时，接受率仅需使用 beta 倍的对数似然差。我们同时使用对称全局符号翻转提议，两者都保留当前温度的目标分布。有限粒子误差依然需要实验测量。

依据：[Cotter、Roberts、Stuart 与 White，MCMC Methods for Functions](https://arxiv.org/html/1202.0709v3)。

## 论文证据与价值边界

可证明的总体可积性与观测到的采样效率分开书写。新传感器问题是具有真实依赖结构的可核查诊断族，规模为十二个观测；它本身可通过 4096 个符号组合精确求解。实际应用收益需要更多观测、真实数据或不可直接计算似然的模型证据。这个边界不妨碍用它检验完整矩阵理论，但禁止据此声称实际部署加速。

研究技能要求追溯原始来源，因此本轮使用作者论文和正式出版信息。审稿技能及代理审查用于作者侧质量检查，不构成人类同行评审或接收概率估计。
# 新增方法来源核查

已阅读 Whiteley and Lee 的 [Twisted Particle Filters](https://people.maths.bris.ac.uk/~manpw/filtering_aos.pdf) 中路径换测度与归一化常数无偏性构造，以及 Heng, Bishop, Deligiannidis and Doucet 的 [Controlled Sequential Monte Carlo](https://www.stats.ox.ac.uk/~doucet/HengBishopDeligiannidisDoucet_controlledSMC.pdf) 第 3.1 节和补充材料第 7.1 节。后者明确包含线性二次高斯控制、Riccati 后向递推和高斯零方差特例。这些思想不能作为本项目的一般性原创贡献。

本轮进一步检查的是：组合扩散共享尾部二次型能否给出一个明确的路径级所有正阶权重矩有限结论，并保持指定随机 Euler 核的增广目标。实际非线性误差、有限样本效率及重复采样稳定性仍需分别验证。
