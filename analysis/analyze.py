"""从 ICLR/ICML 原始投稿数据生成趋势分析表格（Markdown）。

用法：
    python3 fetch_data.py data
    python3 analyze.py data results

所有输出表格写入 results/ 目录，报告正文引用的每一个数字都由本脚本生成。

口径说明
--------
ICLR 官方公布的 27.4% 接收率 = 接收数 / 有效投稿数，分母**包含**中途撤稿的论文、
但不含 desk reject。本脚本区分两个口径：

* 投稿口径 = 接收 / (全部投稿 - desk reject)    —— 对齐官方数字，回答"我投这个方向，中稿概率多大"
* 决策口径 = 接收 / (接收 + 拒稿)               —— 剔除撤稿，回答"撑到出结果的论文里多少中了"

两者差值由撤稿率驱动。撤稿率本身是信号：撤稿多说明该方向的稿子在评审中普遍被打穿。
"""

import collections
import json
import os
import re
import sys

# ---------------------------------------------------------------- 基础工具

ACCEPT_TOKENS = ("poster", "oral", "spotlight", "accept")


def norm_status(raw):
    s = (raw or "").strip().lower()
    if "desk" in s:
        return "desk"
    if s.startswith("withdraw"):
        return "withdraw"
    if s.startswith("reject"):
        return "reject"
    if any(t in s for t in ACCEPT_TOKENS):
        return "accept"
    return "other"


def is_oral(raw):
    return "oral" in (raw or "").strip().lower()


def rating_avg(paper):
    """rating_avg 在原始数据里可能是 list，也可能是 list 的字符串形式。"""
    v = paper.get("rating_avg")
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except (ValueError, TypeError):
            return None
    return v if isinstance(v, list) and v else None


def split_keywords(paper):
    seen = []
    for k in re.split(r"[;,]", paper.get("keywords") or ""):
        k = k.strip().lower()
        if k and len(k) <= 60 and k not in seen:
            seen.append(k)
    return seen


# 把同义关键词并到一起，否则 "LLM" / "LLMs" / "large language model" 会把信号打散。
ALIASES = {
    "llm": "llm(s)", "llms": "llm(s)", "large language model": "llm(s)",
    "large language models": "llm(s)", "large language models (llms)": "llm(s)",
    "language model": "llm(s)", "language models": "llm(s)",
    "rl": "reinforcement learning",
    "diffusion model": "diffusion models",
    "generative model": "generative models",
    "graph neural network": "graph neural networks",
    "multimodal large language model": "multimodal llm",
    "multimodal large language models": "multimodal llm",
    "mllm": "multimodal llm", "mllms": "multimodal llm",
    "vision-language model": "vision-language models",
    "vision language models": "vision-language models",
    "vlm": "vision-language models", "vlms": "vision-language models",
    "llm agent": "llm agents", "llm-based agents": "llm agents",
    "agent": "agents", "ai agent": "agents", "ai agents": "agents",
    "time series forecasting": "time series",
}


def canon(k):
    return ALIASES.get(k, k)


class Tally(collections.Counter):
    @property
    def decided(self):
        return self["accept"] + self["reject"]

    @property
    def submitted(self):
        """对齐官方口径：全部投稿去掉 desk reject。"""
        return self["all"] - self["desk"]

    def acc_decided(self):
        return self["accept"] / self.decided * 100 if self.decided else float("nan")

    def acc_submitted(self):
        return self["accept"] / self.submitted * 100 if self.submitted else float("nan")

    def wd_rate(self):
        return self["withdraw"] / self["all"] * 100 if self["all"] else float("nan")


def tally(papers, key_fn):
    out = collections.defaultdict(Tally)
    for p in papers:
        st = norm_status(p.get("status"))
        for k in key_fn(p):
            t = out[k]
            t[st] += 1
            t["all"] += 1
            if is_oral(p.get("status")):
                t["oral"] += 1
    return out


def pct(v, nd=1):
    return "n/a" if v != v else f"{v:.{nd}f}%"


def signed(v, nd=1):
    return "n/a" if v != v else f"{v:+.{nd}f}"


# ---------------------------------------------------------------- 各张表

def table_overview(d25, d26, out):
    out.append("## ICLR 2025 → 2026 总体盘子\n")
    out.append("| 指标 | ICLR 2025 | ICLR 2026 | 变化 |")
    out.append("|---|---:|---:|---:|")
    rows = []
    for label, data in (("2025", d25), ("2026", d26)):
        c = collections.Counter(norm_status(p.get("status")) for p in data)
        c["all"] = len(data)
        rows.append(c)
    a, b = rows
    def line(name, k, as_pct_of=None):
        va, vb = a[k], b[k]
        if as_pct_of:
            sa = f"{va:,} ({va/a[as_pct_of]*100:.1f}%)"
            sb = f"{vb:,} ({vb/b[as_pct_of]*100:.1f}%)"
        else:
            sa, sb = f"{va:,}", f"{vb:,}"
        out.append(f"| {name} | {sa} | {sb} | {vb/va-1:+.1%} |")
    line("总投稿", "all")
    line("接收", "accept", "all")
    line("拒稿", "reject", "all")
    line("作者撤稿", "withdraw", "all")
    line("desk reject", "desk", "all")
    for label, c in (("2025", a), ("2026", b)):
        pass
    out.append(
        f"| 接收率（投稿口径） | {a['accept']/(a['all']-a['desk'])*100:.1f}% "
        f"| {b['accept']/(b['all']-b['desk'])*100:.1f}% | "
        f"{b['accept']/(b['all']-b['desk'])*100 - a['accept']/(a['all']-a['desk'])*100:+.1f}pp |"
    )
    out.append(
        f"| 接收率（决策口径） | {a['accept']/(a['accept']+a['reject'])*100:.1f}% "
        f"| {b['accept']/(b['accept']+b['reject'])*100:.1f}% | "
        f"{b['accept']/(b['accept']+b['reject'])*100 - a['accept']/(a['accept']+a['reject'])*100:+.1f}pp |"
    )
    out.append("")


def table_scores(d25, d26, out):
    out.append("## 评分分布与录用分数线\n")
    out.append("ICLR 2026 换了打分刻度：2025 年是 {1,3,5,6,8,10}，2026 年改成等距的 "
               "{0,2,4,6,8,10}。所以两年的分数不可直接比较，要比的是"
               "**分数落在哪个百分位**。\n")
    for name, data in (("ICLR 2025", d25), ("ICLR 2026", d26)):
        vals = collections.Counter()
        for p in data:
            for r in (p.get("rating") or "").split(";"):
                r = r.strip()
                if r:
                    vals[float(r)] += 1
        total = sum(vals.values())
        dist = "，".join(f"{int(k)}分 {v/total*100:.1f}%" for k, v in sorted(vals.items()))
        out.append(f"**{name}** 单条评分分布（共 {total:,} 条评审）：{dist}\n")

    out.append("平均分对应的录用概率（仅统计拿到 accept/reject 决定的论文）：\n")
    out.append("| 平均分 | ICLR 2025 论文数 | 2025 录用率 | ICLR 2026 论文数 | 2026 录用率 |")
    out.append("|---:|---:|---:|---:|---:|")
    buckets = {}
    for name, data in (("2025", d25), ("2026", d26)):
        b = collections.defaultdict(Tally)
        for p in data:
            st = norm_status(p.get("status"))
            r = rating_avg(p)
            if st not in ("accept", "reject") or not r:
                continue
            key = round(r[0] * 2) / 2
            b[key][st] += 1
            b[key]["all"] += 1
        buckets[name] = b
    for k in sorted(set(buckets["2025"]) | set(buckets["2026"])):
        if k < 2.5:
            continue
        c25, c26 = buckets["2025"].get(k, Tally()), buckets["2026"].get(k, Tally())
        f = lambda c: (f"{c['all']:,}", pct(c.acc_decided())) if c["all"] else ("—", "—")
        n25, r25 = f(c25)
        n26, r26 = f(c26)
        out.append(f"| {k:.1f} | {n25} | {r25} | {n26} | {r26} |")
    out.append("")


def table_areas(d26, out):
    out.append("## ICLR 2026 分领域接收率（按作者自选的 primary area）\n")
    t = tally(d26, lambda p: [(p.get("primary_area") or "(未填)").strip().lower()])
    rows = [(k, v) for k, v in t.items() if v.decided >= 30]
    rows.sort(key=lambda kv: -kv[1].acc_submitted())
    out.append("| Primary area | 投稿 | 接收 | 接收率(投稿口径) | 接收率(决策口径) | 撤稿率 | Oral | 每千投稿 Oral |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k, v in rows:
        out.append(
            f"| {k} | {v['all']:,} | {v['accept']:,} | {pct(v.acc_submitted())} | "
            f"{pct(v.acc_decided())} | {pct(v.wd_rate())} | {v['oral']} | "
            f"{v['oral']/v['all']*1000:.1f} |"
        )
    out.append("")


def table_keyword_yoy(d25, d26, out):
    out.append("## 关键词同比：拥挤度 vs 回报率\n")
    n25, n26 = len(d25), len(d26)
    growth = n26 / n25 - 1
    out.append(
        f"ICLR 总投稿从 {n25:,} 涨到 {n26:,}（{growth:+.1%}）。所以看绝对投稿量增长会误导——"
        f"一个方向必须涨过 {growth:.0%} 才算真正在**抢占份额**，涨得比这慢的其实在被稀释。\n"
    )
    t25 = tally(d25, lambda p: [canon(k) for k in split_keywords(p)])
    t26 = tally(d26, lambda p: [canon(k) for k in split_keywords(p)])
    rows = []
    for k, v26 in t26.items():
        if v26["all"] < 100:
            continue
        v25 = t25.get(k, Tally())
        share25 = v25["all"] / n25 if v25["all"] >= 10 else 0
        share26 = v26["all"] / n26
        rows.append({
            "kw": k,
            "n25": v25["all"], "n26": v26["all"],
            "share_growth": (share26 / share25 - 1) * 100 if share25 else float("nan"),
            "acc25": v25.acc_decided() if v25.decided >= 30 else float("nan"),
            "acc26": v26.acc_decided(),
            "wd26": v26.wd_rate(),
        })
    rows.sort(key=lambda r: -(r["share_growth"] if r["share_growth"] == r["share_growth"] else -1e9))
    out.append("| 关键词 | 2025 投稿 | 2026 投稿 | 份额变化 | 2025 接收率 | 2026 接收率 | 接收率变化 | 2026 撤稿率 |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        delta = r["acc26"] - r["acc25"]
        out.append(
            f"| {r['kw']} | {r['n25']:,} | {r['n26']:,} | {signed(r['share_growth'])}% | "
            f"{pct(r['acc25'])} | {pct(r['acc26'])} | "
            f"{signed(delta) + 'pp' if delta == delta else 'n/a'} | {pct(r['wd26'])} |"
        )
    out.append("")


def table_blue_ocean(d26, out):
    out.append("## 中等体量、高接收率的细分方向\n")
    out.append(
        "筛选条件：ICLR 2026 拿到决策的论文在 25–220 篇之间（不是无人区，也还没被淹没），"
        "按决策口径接收率排序。撤稿率低说明这类稿子在评审里普遍站得住。\n"
    )
    t = tally(d26, split_keywords)
    rows = [(k, v) for k, v in t.items() if 25 <= v.decided <= 220]
    rows.sort(key=lambda kv: -kv[1].acc_decided())
    out.append("| 关键词 | 投稿 | 决策 | 接收 | 接收率(决策口径) | 撤稿率 |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for k, v in rows[:45]:
        out.append(
            f"| {k} | {v['all']} | {v.decided} | {v['accept']} | "
            f"{pct(v.acc_decided())} | {pct(v.wd_rate())} |"
        )
    out.append("")


def table_disagreement(d26, out):
    out.append("## 评审分歧与结果\n")
    b = collections.defaultdict(Tally)
    for p in d26:
        st = norm_status(p.get("status"))
        r = rating_avg(p)
        if st not in ("accept", "reject") or not r or len(r) < 2:
            continue
        key = round(r[1] * 2) / 2
        b[key][st] += 1
        b[key]["all"] += 1
    out.append("| 评分标准差 | 论文数 | 接收率(决策口径) |")
    out.append("|---:|---:|---:|")
    for k in sorted(b):
        v = b[k]
        if v["all"] < 100:
            continue
        out.append(f"| {k:.1f} | {v['all']:,} | {pct(v.acc_decided())} |")
    out.append("")


def table_icml(icml, out):
    out.append("## ICML 2026 录用论文的领域构成与 spotlight 率\n")
    out.append(
        "ICML 的公开数据只有录用论文，算不了接收率；但 spotlight 率（spotlight / 该领域录用数）"
        "能反映**同样是录用论文，哪个领域更被评审当回事**。\n"
    )
    top = collections.defaultdict(Tally)
    sub = collections.defaultdict(Tally)
    for p in icml:
        area = (p.get("primary_area") or "?")
        st = (p.get("status") or "").lower()
        for bucket, key in ((top, area.split("->")[0]), (sub, area)):
            bucket[key]["all"] += 1
            if "spotlight" in st:
                bucket[key]["spotlight"] += 1
    total = len(icml)
    out.append(f"共 {total:,} 篇录用论文。一级领域：\n")
    out.append("| 一级领域 | 录用数 | 占比 | spotlight | spotlight 率 |")
    out.append("|---|---:|---:|---:|---:|")
    for k, v in sorted(top.items(), key=lambda kv: -kv[1]["all"]):
        out.append(
            f"| {k} | {v['all']:,} | {v['all']/total*100:.1f}% | {v['spotlight']} | "
            f"{v['spotlight']/v['all']*100:.1f}% |"
        )
    out.append("")
    out.append("细分领域（录用 ≥ 40 篇），按 spotlight 率排序：\n")
    out.append("| 细分领域 | 录用数 | spotlight | spotlight 率 |")
    out.append("|---|---:|---:|---:|")
    rows = [(k, v) for k, v in sub.items() if v["all"] >= 40]
    rows.sort(key=lambda kv: -kv[1]["spotlight"] / kv[1]["all"])
    for k, v in rows:
        out.append(f"| {k} | {v['all']} | {v['spotlight']} | {v['spotlight']/v['all']*100:.1f}% |")
    out.append("")


def table_orals(d26, out):
    out.append("## ICLR 2026 Oral 的领域集中度\n")
    orals = [p for p in d26 if is_oral(p.get("status"))]
    kc = collections.Counter()
    for p in orals:
        for k in split_keywords(p):
            kc[canon(k)] += 1
    out.append(f"共 {len(orals)} 篇 Oral。出现最多的关键词：\n")
    out.append("| 关键词 | Oral 篇数 |")
    out.append("|---|---:|")
    for k, n in kc.most_common(30):
        out.append(f"| {k} | {n} |")
    out.append("")


# ---------------------------------------------------------------- 入口

def main():
    datadir = sys.argv[1] if len(sys.argv) > 1 else "data"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "results"
    os.makedirs(outdir, exist_ok=True)

    load = lambda n: json.load(open(os.path.join(datadir, n)))
    d25, d26, icml = load("iclr2025.json"), load("iclr2026.json"), load("icml2026.json")

    out = ["# 会议数据分析结果", "",
           "> 本文件由 `analysis/analyze.py` 自动生成，数据来自 Paper Copilot 的 OpenReview 抓取"
           "（https://github.com/papercopilot/paperlists）。", ""]
    table_overview(d25, d26, out)
    table_scores(d25, d26, out)
    table_areas(d26, out)
    table_keyword_yoy(d25, d26, out)
    table_blue_ocean(d26, out)
    table_disagreement(d26, out)
    table_orals(d26, out)
    table_icml(icml, out)

    path = os.path.join(outdir, "conference-stats.md")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {path} ({len(out)} lines)")


if __name__ == "__main__":
    main()
