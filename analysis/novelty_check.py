"""对候选选题做 arXiv 查重，按投稿时间倒序列出最相关的近期工作。

用法：
    python3 novelty_check.py              # 跑内置的候选选题清单
    python3 novelty_check.py 'abs:"..."'  # 跑自定义 query

query 语法见 https://info.arxiv.org/help/api/user-manual.html#query_details
（前缀 ti: / abs: / all: / cat:，可用 AND OR ANDNOT 组合）。

结果只能当作第一道筛子：arXiv 的检索是字面匹配，换个说法的同类工作查不出来。
真正定稿前还要在 OpenReview 上搜 ICLR 2026 / ICML 2026 / NeurIPS 2025 的全量投稿，
**包括被拒的** —— 被拒论文的公开审稿意见会直接告诉你评审在意什么。
"""

import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}

CANDIDATES = {
    "A1-a  agent memory vs 等预算长上下文":
        'all:"agent memory" AND all:"long context" AND '
        '(all:"token budget" OR all:"fair comparison" OR all:"controlled")',
    "A1-b  RLVR verifier 的覆盖边界与钻空子":
        'abs:"RLVR" AND (abs:"reward hacking" OR abs:"verifier" AND abs:"exploit")',
    "A1-c  test-time scaling 收益何时转负":
        'abs:"test-time scaling" AND (abs:"when does" OR abs:"diminishing" OR '
        'abs:"overthinking" OR abs:"negative")',
    "A2-a  LLM-as-judge 的统计效力与样本量":
        'abs:"LLM-as-a-judge" AND (abs:"statistical power" OR abs:"sample size" OR '
        'abs:"significance")',
    "A2-b  agent benchmark 的运行间方差":
        'abs:"agent" AND abs:"benchmark" AND (abs:"variance" OR abs:"reproducibility" OR '
        'abs:"run-to-run")',
    "A2-c  污染对排名的实际扰动量级":
        'abs:"contamination" AND abs:"benchmark" AND (abs:"ranking" OR abs:"leaderboard") '
        'AND abs:"impact"',
    "A3-a  推理模型场景下 KV 压缩的失效":
        'abs:"KV cache" AND (abs:"reasoning model" OR abs:"long chain-of-thought" OR '
        'abs:"long CoT")',
    "A3-b  低比特量化对推理能力（而非困惑度）的影响":
        'abs:"quantization" AND abs:"reasoning" AND (abs:"degradation" OR '
        'abs:"perplexity is not" OR abs:"misleading")',
    "A4    diffusion LM 的推理能力边界":
        'abs:"diffusion language model" AND (abs:"reasoning" OR abs:"limitation" OR '
        'abs:"when does")',
    "A5-a  agent 状态治理：审计、遗忘、回滚":
        'abs:"agent" AND (abs:"rollback" OR abs:"revoke" OR abs:"forgetting") AND '
        '(abs:"memory" OR abs:"state") AND abs:"governance"',
    "A5-b  agent 撤销已失效的用户偏好":
        'abs:"agent" AND abs:"preference" AND (abs:"revoke" OR abs:"retract" OR '
        'abs:"outdated" OR abs:"stale")',
}


def search(query, n=12):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": query, "start": 0, "max_results": n,
        "sortBy": "submittedDate", "sortOrder": "descending",
    })
    with urllib.request.urlopen(url, timeout=60) as resp:
        root = ET.fromstring(resp.read())
    return [(
        e.find("a:published", NS).text[:10],
        e.find("a:id", NS).text.rsplit("/", 1)[-1],
        e.find("a:title", NS).text.strip().replace("\n", " "),
    ) for e in root.findall("a:entry", NS)]


def run(label, query):
    print("=" * 100)
    print(label)
    print(f"  query: {query}")
    print("-" * 100)
    try:
        hits = search(query)
    except Exception as exc:
        print(f"  检索失败: {exc}")
        return
    if not hits:
        print("  无结果")
    for date, arxiv_id, title in hits:
        print(f"  {date}  {arxiv_id:<13} {title[:95]}")


def main():
    if len(sys.argv) > 1:
        run("自定义 query", sys.argv[1])
        return
    for i, (label, query) in enumerate(CANDIDATES.items()):
        if i:
            time.sleep(3)  # arXiv API 要求请求之间留间隔
        run(label, query)


if __name__ == "__main__":
    main()
