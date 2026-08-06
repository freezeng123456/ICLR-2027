# 数据分析

报告里的每一个自算数字都由这里的脚本生成，可复现。

```bash
cd analysis
python3 fetch_data.py data          # 下载约 213 MB 原始数据（data/ 已 gitignore）
python3 analyze.py data results     # 生成 results/conference-stats.md
python3 oral_themes.py data results # 生成 results/oral-themes.md
python3 novelty_check.py            # 对候选选题做 arXiv 查重
python3 hardware_budget.py          # 按 roofline 估算算力预算（默认 2x H20）
```

无第三方依赖，Python 3.8+ 即可。

`novelty_check.py` 内置两组检索式：默认跑 [选题建议](../docs/02-选题建议.md) 里的候选，
`--theory` 跑 [理论方向](../docs/05-理论方向.md) 里的（会自动加 cs 分类限制，否则
"self-consistency"、"modal"之类的词会被物理论文淹没），`--all` 两组都跑。也可以传自己的
query：`python3 novelty_check.py 'abs:"..." AND abs:"..."'`。建议在定稿前重跑 ——
从这次的结果看，2026 年一个"明显的空白"被占掉大约只需要两个月。

`hardware_budget.py` 回答三个排期问题：大 batch 下解码卡在带宽还是算力、一次 prefill
要多久、到截止日能跑出多少 token。支持 h20 / h100 / h200 / a100、任意模型规模、
以及 MoE 与量化：

```bash
python3 hardware_budget.py --params 32                      # dense 32B BF16
python3 hardware_budget.py --params 30 --active 3 --bits 8  # Qwen3-30B-A3B FP8
python3 hardware_budget.py --gpu h100 --params 8 --mfu 0.3  # 换卡、用实测 MFU
```

MoE 的处理是关键：**算力只按激活参数走，访存按总参数走**。这解释了为什么一个
30B-A3B 的 MoE 在 H20 上比 32B dense 快一个数量级——显存大算力弱的卡天然偏爱 MoE。

**输出是理论上限，实测通常只有 30–60%**，第一周测完真实吞吐后应该把 `--mfu` 换成实测值
重算。结论见 [04-算力约束下的方案.md](../docs/04-算力约束下的方案.md) 与
[05-理论方向.md](../docs/05-理论方向.md) 第五节。

## 数据来源

[Paper Copilot](https://github.com/papercopilot/paperlists) 从 OpenReview 抓取的投稿
记录，每篇含标题、摘要、关键词、primary area、每位审稿人的评分/confidence/soundness/
contribution/presentation，以及最终状态（Oral / Poster / Reject / Withdraw / Desk Reject）。

| 文件 | 记录数 | 说明 |
|---|---:|---|
| `iclr2025.json` | 11,677 | 全部投稿，含被拒和撤回 |
| `iclr2026.json` | 19,814 | 全部投稿，含被拒和撤回；224 篇 Oral |
| `icml2026.json` | 6,341 | **仅录用论文**（ICML 不公开被拒稿件） |
| `nips2025.json` | 6,212 | 录用论文 + 自愿公开的 400 篇拒稿；93 篇 Oral |

ICLR 的文件在仓库里是 Git LFS 指针，`fetch_data.py` 走 `media.githubusercontent.com`
端点取真实内容。

## 口径

ICLR 官方公布的 27.4% 接收率，分母是**有效投稿数**（含中途撤稿，不含 desk reject）。
脚本区分两个口径：

- **投稿口径** = 接收 / (全部投稿 − desk reject)。对齐官方数字，回答"我投这个方向，
  中稿概率多大"。
- **决策口径** = 接收 / (接收 + 拒稿)。剔除撤稿，回答"撑到出结果的论文里多少中了"。

两者的差由撤稿率驱动，而撤稿率本身是信号：某方向撤稿多，说明该方向的稿子在评审中普遍
被打穿。计算机视觉应用是最典型的例子 —— 决策口径 43.1% 看着不错，投稿口径只有 28.1%，
因为 33.8% 的作者中途撤了。

## 已知局限

- **关键词是作者自填的**，没有受控词表。脚本用一张别名表把最明显的变体合并
  （`LLM` / `LLMs` / `large language model` → `llm(s)` 等），但仍然存在拆分和重叠。
  一篇论文会计入它的所有关键词，所以各行不互斥。
- **关键词与接收率之间存在混淆因素**。写 `deep learning` 当关键词的作者接收率低，更可能
  是因为这批作者对社区惯例不熟，而不是"deep learning"这个词本身有害。跨方向比较时应该
  看趋势和量级，不要当成因果效应。
- **ICML 只有录用论文**，算不了接收率。脚本用 spotlight 率（spotlight / 该领域录用数）
  作为替代信号，回答"同样是录用论文，哪个领域更被当回事"。**NeurIPS 2025 同理**：
  它的对照池实际上就是录用论文，所以 `oral_themes.py` 的放大倍数统一用录用论文做分母，
  否则两个会议不可比。
- **主题聚类是正则匹配，不是语义聚类**。一篇论文可命中多个主题，边界主题（如"优化器"
  与"理论"）会重叠。放大倍数应看量级和排序，不要抠小数点；Oral 少于 8 篇的行噪声很大。
- **2025 与 2026 的 ICLR 评分不可直接比较**：刻度从 {1,3,5,6,8,10} 改成了
  {0,2,4,6,8,10}。要比的是分数所处的百分位。
- **抓取时点差异**导致本地统计与官方数字有小幅出入（如本地 19,814 篇 vs 官方 19,525 篇
  有效投稿）。结论用的是比例和排序，不受此影响。
- 数据里的 `wc_reply_authors`、`github` 等字段在这一版快照中为空，因此无法分析
  rebuttal 长度或开源与否对结果的影响。
