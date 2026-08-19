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

主线已经从「选题」变成「把 PFN 先验错配做成可投稿实验」。机制半篇（隐变量纠缠 / 错误归因）在 GP-PFN 上已经有可复现的证据；方法半篇能不能活，取决于 far-anchor 校准是否只是温度缩放的马甲。对照脚本：

- `exp_calib_baselines.py` — 温度缩放 / Platt / 上下文注噪 / 额外 ID 行 vs far-anchor
- `exp_real_calib.py` — 同一对照搬到 sklearn / OpenML 小表
- `train_missing.py` — 补训被覆盖的 C@15k、从未提交的 A144 和 `pfn_gp.pt`

实验记录见 [`docs/exp-log.md`](docs/exp-log.md)。
