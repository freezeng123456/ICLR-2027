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

## 当前状态

方向待拍板。推荐首选：**PFN 的先验错配——把上下文样本理解为先验倾斜算子**（见主方案 §1）。
第一个关卡是 8/21 前完成 48 小时验证实验：用解析可算的 GP 先验，验证追加 anchor 样本是否让隐含后验朝理论预测的方向移动。
方向不符就换题，不要调参。
