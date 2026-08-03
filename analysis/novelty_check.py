"""对候选选题做 arXiv 查重，按投稿时间倒序列出最相关的近期工作。

用法：
    python3 novelty_check.py              # 跑内置的候选选题清单
    python3 novelty_check.py --theory     # 跑理论方向清单（自动限定 cs 分类）
    python3 novelty_check.py --all        # 两组都跑
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

# 理论方向（见 docs/05-理论方向.md）。这些查询会自动加上 cs 分类限制，
# 否则 "self-consistency"、"modal" 之类的词会被物理论文淹没。
THEORY = {
    "T1   best-of-N / self-consistency 的理论界":
        '(abs:"best-of-n" OR abs:"self-consistency" OR abs:"majority voting") AND '
        '(abs:"bound" OR abs:"provable" OR abs:"theoretical analysis" OR abs:"asymptotic")',
    "T2   不完美 verifier 的极限":
        '(abs:"imperfect verifier" OR abs:"noisy verifier" OR '
        'abs:"reward model" AND abs:"overoptimization") AND '
        '(abs:"theory" OR abs:"bound" OR abs:"limit")',
    "T3   test-time compute 的理论":
        'abs:"test-time" AND (abs:"scaling" OR abs:"compute") AND '
        '(abs:"theory" OR abs:"theoretical" OR abs:"provably" OR abs:"upper bound")',
    "T4   长时程误差累积 vs 自我纠错":
        '(abs:"compounding error" OR abs:"error propagation" OR abs:"horizon") AND '
        '(abs:"self-correction" OR abs:"recovery") AND (abs:"agent" OR abs:"LLM")',
    "T5   成功率随任务时长衰减的刻画":
        '(abs:"long-horizon" OR abs:"multi-step") AND (abs:"success rate" OR abs:"failure") '
        'AND (abs:"exponential" OR abs:"decay" OR abs:"phase transition")',
    "T6   任务时长曲线（METR 那条）的理论解释":
        '(abs:"task length" OR abs:"time horizon") AND '
        '(abs:"doubling" OR abs:"exponential trend" OR abs:"50% success")',
    "T7   agent 的 Markov / MDP 抽象与界":
        'abs:"LLM agent" AND (abs:"Markov" OR abs:"MDP" OR abs:"regret" OR '
        'abs:"sample complexity")',
    "T8   Transformer 表达能力（拥挤度对照组）":
        'abs:"transformer" AND (abs:"expressivity" OR abs:"expressive power" OR '
        'abs:"circuit complexity")',
}

CS_FILTER = "(cat:cs.LG OR cat:cs.AI OR cat:cs.CL OR cat:stat.ML OR cat:cs.CC)"


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
    args = sys.argv[1:]
    if args and args[0] not in ("--theory", "--all"):
        run("自定义 query", args[0])
        return

    groups = []
    if not args or args[0] == "--all":
        groups.append(("选题候选", CANDIDATES, False))
    if args and args[0] in ("--theory", "--all"):
        groups.append(("理论方向", THEORY, True))

    first = True
    for title, queries, cs_only in groups:
        print(f"\n{'#' * 102}\n# {title}\n{'#' * 102}")
        for label, query in queries.items():
            if not first:
                time.sleep(3)  # arXiv API 要求请求之间留间隔
            first = False
            run(label, f"{CS_FILTER} AND ({query})" if cs_only else query)


if __name__ == "__main__":
    main()
