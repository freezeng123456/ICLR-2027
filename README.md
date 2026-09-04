# ICLR-2027

ICLR 2027 投稿的工作仓库。

## 关键日期（AOE）

- **摘要截止：2026-09-18**
- **正文截止：2026-09-25**
- 评审发布 2026-11-05，最终决定 2026-12-16

## 文档

- [`docs/pfn-vs-its-own-bayes.md`](docs/pfn-vs-its-own-bayes.md) — **主方案**：PFN 与它自己的贝叶斯最优解相差多少
- [`docs/submission-checklist.md`](docs/submission-checklist.md) — ICLR 2027 硬性规则（页数、配额、评审义务、AI 披露）
- [`docs/direction-shortlist-no-llm.md`](docs/direction-shortlist-no-llm.md) — 选题判据：单次实验 ≤10 分钟、模型能自己从零训出来
- [`docs/direction-shortlist.md`](docs/direction-shortlist.md) — 与方向无关的选型判据与排除项
- [`docs/theory-interventions.md`](docs/theory-interventions.md) — 推理期干预的对象与可达集
- [`docs/how-others-write.md`](docs/how-others-write.md) — 两篇该精读的稿：Müller ICLR 2022、SPN ICML 2026，以及九页该怎么套
- [`papers/`](papers/README.md) — 上述两篇 PDF

## 当前状态

主线是 PFN 与它自己的贝叶斯最优解之间的偏离。先验完全正确、数据严格来自先验，
此时上下文数据越多，网络离自己的最优解越远：96 个格子上超额 KL 落在 0.0101 – 0.2799 nat，
32 个组合里 20 个随上下文点数上升，最大 11.4 倍。
偏离在隐变量空间里有确定的形状：长度尺度那一维是乘性更新不足（$\beta = 0.853$，与方向无关），
噪声那一维是加性上偏（$+0.174$，即高 19%），网络把自己的逼近误差算进了噪声。
信息几何那一族预测量被否掉，记录与判据见 [`docs/pfn-vs-its-own-bayes.md`](docs/pfn-vs-its-own-bayes.md)。

先前那一轮先验错配实验（隐变量纠缠、错误归因、远置探针对照温度缩放）的记录在
[`docs/exp-log.md`](docs/exp-log.md)，其中的机制结果是同一现象在隐变量维数不足时的情形。
