# Oral 论文在做什么

完整的 317 篇标题（ICLR 2026 的 224 篇 + NeurIPS 2025 的 93 篇）按主题分组列在
[`analysis/results/oral-themes.md`](../analysis/results/oral-themes.md)，由
[`oral_themes.py`](../analysis/oral_themes.py) 生成。这里只讲从中读出来的东西。

---

## 一、先看"放大倍数"，不要看篇数

**放大倍数 = 该主题在 Oral 里的占比 ÷ 在录用论文里的占比。** 直接数 Oral 篇数会被基数
带偏——LLM 相关的 Oral 最多，但那只是因为 LLM 投稿最多。放大倍数回答的是：
**在够格录用的论文里，哪些主题更容易被抬成 Oral。**

（对照池用录用论文而非全体投稿，是为了让两个会议可比：NeurIPS 公开数据基本只含录用论文。）

### ICLR 2026（Oral 占录用的 4.2%）

| 主题 | Oral 篇数 | 占 Oral | 占录用 | 放大倍数 |
|---|---:|---:|---:|---:|
| 扩散语言模型 | 4 | 1.8% | 1.2% | **1.52×** ⚠ |
| **优化器与训练方法** | **33** | 14.7% | 10.8% | **1.37×** |
| 架构与序列建模 | 24 | 10.7% | 9.0% | 1.20× |
| Agent 与工具使用 | 21 | 9.4% | 8.1% | 1.15× |
| 安全 / 对齐 / 监督 | 29 | 12.9% | 11.6% | 1.12× |
| 理论 / 学习动力学 | 29 | 12.9% | 11.9% | 1.09× |
| 推理 / test-time compute | 33 | 14.7% | 13.7% | 1.08× |
| 扩散 / 流匹配生成 | 29 | 12.9% | 12.1% | 1.07× |
| 具身 / 机器人 | 15 | 6.7% | 6.3% | 1.06× |
| RL 与后训练 | 35 | 15.6% | 15.2% | 1.02× |
| 可解释性 | 12 | 5.4% | 5.4% | 0.99× |
| 效率 / 推理系统 | 19 | 8.5% | 9.0% | 0.94× |
| 视频与图像生成 | 18 | 8.0% | 8.6% | 0.93× |
| 评测与 benchmark | 20 | 8.9% | 9.6% | 0.93× |
| 数据 / 记忆 / 归因 | 11 | 4.9% | 5.4% | 0.92× |
| 图 / 时序 / 表格 | 13 | 5.8% | 6.8% | 0.85× |
| **多模态 / 视觉语言** | 28 | 12.5% | **15.1%** | **0.83×** |
| 科学与生物医学应用 | 12 | 5.4% | 6.8% | 0.79× |

⚠ = Oral 少于 8 篇，噪声大，只当线索。

**两个反直觉的地方：**

- **"优化器与训练方法"是最可靠的高放大主题（1.37×，33 篇，样本足够）。** 这不是热门话题，
  但它是最容易被抬成 Oral 的。
- **多模态 / 视觉语言是录用论文里最大的一块（15.1%），却是放大倍数最低的之一（0.83×）。**
  投得最多，被抬得最少。评测、效率、数据这几个大池子也都在 1.0 以下。

### 半年前的 NeurIPS 2025，画风完全不同（Oral 占录用的 1.6%）

| 主题 | NeurIPS 2025 | ICLR 2026 | 变化 |
|---|---:|---:|---|
| **Agent 与工具使用** | 0.21× ⚠ | 1.15× | ↑↑ +0.95 |
| **推理 / test-time compute** | 0.45× ⚠ | 1.08× | ↑↑ +0.63 |
| 优化器与训练方法 | 0.96× | 1.37× | ↑ +0.41 |
| 扩散 / 流匹配生成 | 0.70× | 1.07× | ↑ +0.37 |
| 安全 / 对齐 / 监督 | 0.83× | 1.12× | ↑ +0.29 |
| 具身 / 机器人 | 0.79× | 1.06× | ↑ +0.27 |
| 评测与 benchmark | 0.69× | 0.93× | ↑ +0.24 |
| 可解释性 | 0.78× | 0.99× | ↑ +0.22 |
| 理论 / 学习动力学 | 1.06× | 1.09× | — +0.03 |
| RL 与后训练 | 1.08× | 1.02× | — −0.06 |
| 效率 / 推理系统 | 1.08× | 0.94× | — −0.14 |
| **图 / 时序 / 表格** | 1.22× | 0.85× | ↓ −0.37 |
| 架构与序列建模 | 1.74× | 1.20× | ↓ −0.54 |
| 多模态 / 视觉语言 | 1.45× | 0.83× | ↓↓ −0.62 |

**NeurIPS 2025 的 93 篇 Oral 里，agent 只有 1 篇，推理只有 4 篇。** 半年后的 ICLR 2026，
两者都翻到 1.0 以上。这是个很大的摆动。

但要**谨慎解读**：NeurIPS 那边 agent 和推理的 Oral 篇数分别是 1 和 4，统计噪声极大；
而且两个会议的 Oral 选择性差 2.6 倍（1.6% vs 4.2%），不是同一个筛选强度。方向是可信的
（agent/推理从被压制变成被接受），幅度不要当真。

**唯一稳定的是理论**：1.06× → 1.09×，两届都在 1 以上，且样本都够（15 篇 / 29 篇）。
它不是最高的，但它是**唯一不随风向摆动的**。

---

## 二、比主题更重要的是形态

把最高放大的那一组（优化器与训练方法，33 篇）的标题摊开看，会发现**它们几乎没有一篇是
"我们提出了一个新优化器"**。真正的构成是四类：

**（一）标准做法其实是错的**
- How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining
- WSM: Decay-Free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training
- Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation
- Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention

**（二）把大家天天用的东西的数学做对**
- The Polar Express: Optimal Matrix Sign Methods and their Application to the Muon Algorithm（Muon 里的极分解，做出最优多项式逼近）
- Global Resolution: Optimal Multi-Draft Speculative Sampling via Convex Optimization
- Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks

**（三）证伪一个方法的理论根基**
- Why DPO is a Misspecified Estimator and How to Fix It
- Extending Sequence Length is Not All You Need
- GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning

**（四）给经验规律一个可外推的刻画**
- Pre-training under infinite compute
- The Coverage Principle: How Pre-Training Enables Post-Training
- The Art of Scaling Reinforcement Learning Compute for LLMs
- Scaling Laws and Spectra of Shallow Neural Networks in the Feature Learning Regime

理论组（29 篇）也是同样的味道：表达能力（Softmax Transformers are Turing-Complete、
Quantitative Bounds for Length Generalization in Transformers）、谱与 scaling
（Scaling Laws and Spectra…、Fast Escape, Slow Convergence: Learning Dynamics of Phase
Retrieval under Power-Law Data、Overparametrization bends the landscape: BBP transitions
at initialization）、以及解码本身的理论（$p$-less Sampling: A Robust Hyperparameter-Free
Approach for LLM Decoding）。

**一句话概括：Oral 奖励的不是"新方法"，是"你们一直这么做，但其实不对 / 其实可以算清楚"。**
这和两个顶会今年 Outstanding Paper 的选择完全一致（见 [01](01-发表趋势调研.md) 第六节，
七篇里零篇刷 SOTA）。

---

## 三、和当前主推方案的关系

[06](06-logits方向.md) 的方案（logit 谱 × 后训练）落在这几个主题的交叉处：

- **优化器与训练方法（1.37×，最高）**——后训练怎么改变模型，正是这一组的核心议题
- **理论 / 学习动力学（1.09×，最稳）**——谱的收缩定理
- **可解释性（0.99×）**——logit 谱作为观察模型内部的量

而且形态上正好命中第（三）和第（四）类：证伪"熵是该看的量"，并给 RL 后多样性坍缩这个
人尽皆知的经验现象一个可外推的刻画。

值得单独留意的几篇 Oral，它们和这个方案在同一片地里，写作时应该引用并划清边界：

| 论文 | 关系 |
|---|---|
| Sequences of Logits Reveal the Low Rank Structure of Language Models | **直接前作**，必须在引言第二段划清边界 |
| $p$-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding | 说明社区在意"分布的尾部怎么截"这件事 |
| Navigating the Latent Space Dynamics of Neural Models | 同为"把模型内部当几何对象研究" |
| Temporal superposition and feature geometry of RNNs under memory demands | 同上，且跨架构 |
| The Coverage Principle: How Pre-Training Enables Post-Training | 预训练与后训练的关系，是这个方案的邻居 |
| Intrinsic Entropy of Context Length Scaling in LLMs | 熵视角的对照——你的主张之一正是"秩比熵更该看" |
| The Art of Scaling Reinforcement Learning Compute for LLMs | RL 后训练的经验规律，可作为验证素材 |

---

## 四、怎么自己重跑

```bash
cd analysis
python3 fetch_data.py data          # ICLR 2025/2026 + ICML 2026 + NeurIPS 2025
python3 oral_themes.py data results # 生成 results/oral-themes.md
```

主题判定规则写在 `oral_themes.py` 的 `THEMES` 里，是标题 + 关键词 + TL;DR 上的正则。
要换分类口径直接改那个列表重跑即可。**这套判定不完美**——一篇论文可能命中多个主题，
边界主题（比如"优化器"和"理论"）会重叠，所以放大倍数应该看量级和排序，不要抠小数点。
