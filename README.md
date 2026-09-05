# ICLR-2027

ICLR 2027 投稿的工作仓库。

## 关键日期（AOE）

- **摘要截止：2026-09-18**
- **正文截止：2026-09-25**
- 评审发布 2026-11-05，最终决定 2026-12-16

## 文档

- [`docs/scope.md`](docs/scope.md) — **先看这个**：收缩后的范围、结论清单、试错清单、下一个实验
- [`docs/pfn-vs-its-own-bayes.md`](docs/pfn-vs-its-own-bayes.md) — 全部数字与判据
- [`docs/submission-checklist.md`](docs/submission-checklist.md) — ICLR 2027 硬性规则（页数、配额、评审义务、AI 披露）
- [`docs/direction-shortlist-no-llm.md`](docs/direction-shortlist-no-llm.md) — 选题判据：单次实验 ≤10 分钟、模型能自己从零训出来
- [`docs/direction-shortlist.md`](docs/direction-shortlist.md) — 与方向无关的选型判据与排除项
- [`docs/theory-interventions.md`](docs/theory-interventions.md) — 推理期干预的对象与可达集
- [`docs/how-others-write.md`](docs/how-others-write.md) — 两篇该精读的稿：Müller ICLR 2022、SPN ICML 2026，以及九页该怎么套
- [`papers/`](papers/README.md) — 上述两篇 PDF

## 当前状态

收缩后的一条线：**PFN 的预测不确定性不是它的后验不确定性；上下文越多这件事越糟；
而放大模型会单调地让它更自信——在原本不够自信的地方是改进，
在结构清晰、噪声低的地方是变坏。**

先验完全正确、数据严格来自先验，此时：

- 上下文越多、偏离越大。跳变先验 30/32 个组合上升，高斯过程 20/32。
- 网络把自己的逼近误差算进噪声那一维，隐含噪声偏高 1.19 倍（高斯过程）到 1.53 倍（跳变）。
- 算力单调地让网络更自信（两个先验上 81/96 与 94/96 个格子的方差对数之差变小）。
- 算力在尖锐区域买不到东西：跳变先验最差格子 8 倍算力只降 4%，
  那里 KL 的方差项随算力上升（0.1809 → 0.2614）而均值项下降（0.0227 → 0.0112）。
- 偏离的价格可以量出来但不是先验不变的：高斯过程指数 0.371（$R^2$ 0.983），跳变 0.146。
- 五条看起来该管用的修法逐一不成立。

结论清单、试错清单与下一个实验见 [`docs/scope.md`](docs/scope.md)，
全部数字见 [`docs/pfn-vs-its-own-bayes.md`](docs/pfn-vs-its-own-bayes.md)。

先前那一轮先验错配实验（隐变量纠缠、错误归因、远置探针对照温度缩放）的记录在
[`docs/exp-log.md`](docs/exp-log.md)，其中的机制结果是同一现象在隐变量维数不足时的情形。
