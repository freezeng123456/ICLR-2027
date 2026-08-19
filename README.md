# ICLR-2027

ICLR 2027 投稿的工作仓库。

## 关键日期（AOE）

- **摘要截止：2026-09-18**
- **正文截止：2026-09-25**
- 评审发布 2026-11-05，最终决定 2026-12-16

## 文档

- [`docs/direction-shortlist-no-llm.md`](docs/direction-shortlist-no-llm.md) — **主方案**：非大模型的快迭代方向候选（单次实验 ≤10 分钟）
- [`docs/direction-shortlist.md`](docs/direction-shortlist.md) — 备选：大模型路线，以及与方向无关的选型判据与排除项
- [`docs/submission-checklist.md`](docs/submission-checklist.md) — ICLR 2027 硬性规则（页数、配额、评审义务、AI 披露）
- [`docs/theory-interventions.md`](docs/theory-interventions.md) — 推理期干预的对象、可达集、和还没做的格子
- [`docs/exp-log.md`](docs/exp-log.md) — 校准对照与检查点补训的实验记录

## 当前状态

主线是 PFN 先验错配。机制半篇（隐变量纠缠 / 错误归因）在补训后的检查点上可以复现。方法半篇已经对照过温度缩放：有 ID 校准点时温度赢；few-shot 没有校准集时，在训练点上拟合 T 会把 log loss 打爆，far-anchor 不受影响。记录见 [`docs/exp-log.md`](docs/exp-log.md)。
