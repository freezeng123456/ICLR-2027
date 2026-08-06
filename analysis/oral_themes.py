"""把顶会 Oral 论文按主题聚类，并算出每个主题相对全体投稿的"放大倍数"。

放大倍数 = 该主题在 Oral 中的占比 ÷ 在全体投稿中的占比。
> 1 表示同样一篇论文做这个主题更容易被评为 Oral；< 1 表示更难。这个量比单纯的
Oral 篇数更有用：LLM 相关的 Oral 最多，但那只是因为 LLM 的投稿基数最大。

用法：
    python3 fetch_data.py data
    python3 oral_themes.py data results
"""

import collections
import json
import os
import re
import sys

# 主题定义。用正则匹配 标题 + 关键词 + TL;DR（不匹配摘要全文，噪声太大）。
# 一篇论文可以命中多个主题，所以各主题占比之和大于 1。
THEMES = [
    ("推理 / test-time compute", r"\breasoning|chain[- ]of[- ]thought|\bcot\b|test[- ]time (scaling|compute|training)|self[- ]consistency|best[- ]of[- ]n|inference[- ]time|overthink|think(ing)? (longer|budget)|verifier"),
    ("RL 与后训练", r"\brl\b|reinforcement learning|rlhf|rlvr|\bdpo\b|\bgrpo\b|\bppo\b|preference optimization|post[- ]training|reward (model|hacking|shaping)|policy (gradient|optimization)"),
    ("Agent 与工具使用", r"\bagent|tool[- ](use|calling)|computer[- ]use|\bgui\b|web (agent|navigation)|multi[- ]agent|long[- ]horizon"),
    ("效率 / 推理系统", r"quantiz|\bkv[- ]cache|speculative (decoding|sampling|action)|sparse attention|prun|distillation|efficient (inference|reasoning|training)|throughput|\bflops\b|low[- ]precision|kernel|\bcuda\b|memory[- ]efficient|mixture[- ]of[- ]experts|\bmoe\b"),
    ("扩散 / 流匹配生成", r"diffusion|flow matching|rectified flow|score[- ]based|denois|consistency model"),
    ("扩散语言模型", r"(diffusion|masked diffusion|discrete diffusion).{0,25}(language model|\bllm\b)|\bdllm\b|language diffusion"),
    ("视频与图像生成", r"video (generation|diffusion|model)|image (generation|editing|synthesis)|text[- ]to[- ](image|video|3d)|novel view|gaussian splatting|\b3d\b"),
    ("架构与序列建模", r"transformer|attention|mamba|state space|\bssm\b|\brnn\b|recurren|positional (encoding|embedding)|\brope\b|tokeniz|architecture|linear attention"),
    ("理论 / 学习动力学", r"\btheor|provab|\bbound(s)?\b|convergence|complexity|expressiv|generaliz|learning dynamics|scaling law|phase transition|random matrix|optimal(ity)?\b|\bnp[- ]hard|turing|succinct|lower bound"),
    ("优化器与训练方法", r"optimiz(er|ation)|\badam\b|\bmuon\b|learning rate|gradient|curriculum|pre[- ]?training (recipe|data|dynamics)|checkpoint merging|initializ"),
    ("可解释性", r"interpretab|mechanistic|sparse autoencoder|\bsae\b|prob(e|ing)\b|circuit|feature geometry|latent space|representation (analysis|geometry)|explain"),
    ("评测与 benchmark", r"benchmark|\beval(uation|uating)?\b|leaderboard|\bjudge\b|\barena\b|测评|micro[- ]benchmark|rank(ing)? (models|systems)"),
    ("安全 / 对齐 / 监督", r"safety|align(ment)?\b|jailbreak|adversarial|red[- ]team|watermark|fingerprint|deception|honest|monitor|unlearn|privacy|differential privacy|harmful|steganograph|poison"),
    ("多模态 / 视觉语言", r"multimodal|vision[- ]language|\bvlm\b|\bmllm\b|audio|speech|cross[- ]modal|visual"),
    ("具身 / 机器人", r"robot|embodied|manipulation|imitation learning|\bvla\b|locomot|control"),
    ("科学与生物医学应用", r"protein|molecul|chemistr|physic|biolog|medical|clinical|genom|neuroscience|\bfmri\b|material|climate|\bpde\b|astro"),
    ("数据 / 记忆 / 归因", r"\bdata (recipe|selection|curation|attribution|pruning)|synthetic data|memoriz|corpus|dataset|contamination|attribution"),
    ("图 / 时序 / 表格", r"graph neural|\bgnn\b|time series|tabular|forecast|temporal|clustering"),
]

COMPILED = [(name, re.compile(pat, re.I)) for name, pat in THEMES]


def norm_status(raw):
    s = (raw or "").strip().lower()
    if "desk" in s:
        return "desk"
    if s.startswith("withdraw"):
        return "withdraw"
    if s.startswith("reject"):
        return "reject"
    if any(t in s for t in ("poster", "oral", "spotlight", "accept")):
        return "accept"
    return "other"


def text_of(paper):
    parts = [paper.get("title") or "", paper.get("keywords") or "", paper.get("tldr") or ""]
    return " ".join(parts)


def themes_of(paper):
    blob = text_of(paper)
    return [name for name, pat in COMPILED if pat.search(blob)]


def analyse(papers, oral_pred, label, out):
    """对照池用**录用论文**（含 Oral 自身），而不是全体投稿。

    理由：NeurIPS 的公开数据基本只有录用论文（拒稿只有自愿公开的那部分），
    用全体投稿做分母会让两个会议不可比。统一成"在够格录用的论文里，哪些主题
    更容易被抬成 Oral"，两边口径才一致。
    """
    orals = [p for p in papers if oral_pred(p)]
    pool = [p for p in papers if norm_status(p.get("status")) == "accept"]
    out.append(f"## {label}\n")
    out.append(f"Oral {len(orals)} 篇；对照池（全部录用论文）{len(pool):,} 篇，"
               f"Oral 占录用的 {len(orals)/len(pool)*100:.1f}%。\n")

    oc, pc = collections.Counter(), collections.Counter()
    for p in orals:
        oc.update(themes_of(p))
    for p in pool:
        pc.update(themes_of(p))

    rows = []
    for name, _ in COMPILED:
        o_share = oc[name] / len(orals) if orals else 0
        p_share = pc[name] / len(pool) if pool else 0
        rows.append((o_share / p_share if p_share else float("nan"), name, oc[name], o_share, p_share))
    rows.sort(reverse=True, key=lambda r: (r[0] if r[0] == r[0] else -1))

    out.append("| 主题 | Oral 篇数 | 占 Oral | 占录用 | 放大倍数 |")
    out.append("|---|---:|---:|---:|---:|")
    for lift, name, n, o_share, p_share in rows:
        cell = f"**{lift:.2f}×**" if lift >= 1.3 else f"{lift:.2f}×"
        if n < 8:
            cell += " ⚠"
        out.append(f"| {name} | {n} | {o_share*100:.1f}% | {p_share*100:.1f}% | {cell} |")
    out.append("\n⚠ = Oral 篇数少于 8，放大倍数的统计噪声很大，只能当线索不能当结论。\n")
    return orals, rows


def list_by_theme(orals, out, label, per_theme=None):
    out.append(f"### {label} 的 Oral 标题（按主题分组）\n")
    seen = collections.Counter()
    for name, _ in COMPILED:
        hits = [p for p in orals if name in themes_of(p)]
        if not hits:
            continue
        out.append(f"**{name}**（{len(hits)} 篇）\n")
        for p in sorted(hits, key=lambda x: x.get("title", ""))[:per_theme]:
            out.append(f"- {p.get('title')}")
            seen[p.get("title")] += 1
        if per_theme and len(hits) > per_theme:
            out.append(f"- …… 另有 {len(hits)-per_theme} 篇")
        out.append("")
    uncat = [p for p in orals if not themes_of(p)]
    if uncat:
        out.append(f"**未归类**（{len(uncat)} 篇）\n")
        for p in sorted(uncat, key=lambda x: x.get("title", "")):
            out.append(f"- {p.get('title')}")
        out.append("")


def main():
    datadir = sys.argv[1] if len(sys.argv) > 1 else "data"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "results"
    os.makedirs(outdir, exist_ok=True)
    load = lambda n: json.load(open(os.path.join(datadir, n)))

    out = ["# 顶会 Oral 论文的主题分布", "",
           "> 由 `analysis/oral_themes.py` 生成。**放大倍数** = 该主题在 Oral 中的占比 ÷ "
           "在**录用论文**中的占比。也就是回答：在够格录用的论文里，哪些主题更容易被抬成 "
           "Oral。这个量比 Oral 篇数有用得多——LLM 相关的 Oral 最多，但那只是因为基数最大。",
           "", "> 对照池用录用论文而非全体投稿，是为了让 ICLR 和 NeurIPS 可比："
           "NeurIPS 的公开数据基本只含录用论文（拒稿只有自愿公开的那部分）。", "",
           "> 主题由标题、关键词、TL;DR 的正则匹配判定，一篇论文可命中多个主题，"
           "所以各主题占比之和大于 1。判定规则见脚本里的 `THEMES`。", ""]

    iclr26 = load("iclr2026.json")
    orals26, _ = analyse(iclr26, lambda p: "oral" in (p.get("status") or "").lower(),
                         "ICLR 2026（决定于 2025-12 ～ 2026-03）", out)

    nips_path = os.path.join(datadir, "nips2025.json")
    orals_n = []
    if os.path.exists(nips_path):
        nips25 = load("nips2025.json")
        orals_n, _ = analyse(nips25, lambda p: (p.get("status") or "").strip().lower() == "oral",
                             "NeurIPS 2025（决定于 2025-09，比 ICLR 2026 早约半年）", out)
        out.append("### 两届之间的变化（同一口径）\n")
        out.append("| 主题 | NeurIPS 2025 | ICLR 2026 | 变化 |")
        out.append("|---|---:|---:|---|")
        n_orals = orals_n
        n_pool = [p for p in nips25 if norm_status(p.get("status")) == "accept"]
        i_pool = [p for p in iclr26 if norm_status(p.get("status")) == "accept"]

        def lift(orals, pool, name):
            oc = sum(1 for p in orals if name in themes_of(p))
            pcn = sum(1 for p in pool if name in themes_of(p))
            if not oc or not pcn:
                return float("nan")
            return (oc / len(orals)) / (pcn / len(pool))

        deltas = []
        for name, _ in COMPILED:
            a, b = lift(n_orals, n_pool, name), lift(orals26, i_pool, name)
            if a == a and b == b:
                deltas.append((b - a, name, a, b))
        deltas.sort(reverse=True)
        for d, name, a, b in deltas:
            arrow = "↑↑" if d > 0.5 else ("↑" if d > 0.15 else ("↓↓" if d < -0.5 else ("↓" if d < -0.15 else "—")))
            out.append(f"| {name} | {a:.2f}× | {b:.2f}× | {arrow} {d:+.2f} |")
        out.append("")

    out.append("---\n")
    list_by_theme(orals26, out, "ICLR 2026")
    if orals_n:
        out.append("---\n")
        list_by_theme(orals_n, out, "NeurIPS 2025")

    path = os.path.join(outdir, "oral-themes.md")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {path} ({len(out)} lines)")


if __name__ == "__main__":
    main()
