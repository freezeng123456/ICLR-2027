import argparse
import hashlib
import json
from pathlib import Path


def report(root):
    analysis_path = root / "confirmation/analysis.json"
    analysis = json.loads(analysis_path.read_text())
    if not analysis["audit_passed"] or not analysis["formal_matrix"] or analysis["audited_cells"] != 20000:
        raise ValueError("A complete audited confirmation is required")
    output = root / "decision_review"
    output.mkdir(exist_ok=False)
    lines = [
        "# 新种子确认：完整结果", "",
        f"审计单元：{analysis['audited_cells']}；已计费查询：{analysis['total_queries']}。", "",
        f"预定应用探索门槛：{'通过' if analysis['application_exploration_gate'] else '未通过'}。", "",
        "## 冻结主问题的三项比较", "",
        "主问题为开发阶段选定的 rare_p0.002_b0.2；每种方法500个新种子。差值使用候选减去对照。", "",
        "| 对照 | 查询节省 | 成对平均差 | Bonferroni单侧t上界 | Bonferroni bootstrap上界 | 判定率下降 | 判定率下降bootstrap 95%上界 | 门槛 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in analysis["primary_comparisons"]:
        lines.append(f"| {row['control']} | {row['relative_query_reduction']:.2%} | {row['paired_mean_difference']:.3f} | {row['bonferroni_one_sided_t_upper']:.3f} | {row['bonferroni_bootstrap_upper']:.3f} | {row['observed_certification_rate_loss']:.2%} | {row['certification_loss_bootstrap_upper_95']:.2%} | {'通过' if row['protocol_comparison_pass'] and row['bootstrap_supports_reduction'] else '未通过'} |")
    lines.extend([
        "", "## 全部问题与方法", "",
        "查询数为4096预算下的截断均值。零差值题中的任何方向证书均计为误判。误判率区间为逐配置Clopper–Pearson 95%区间，未作40配置同时覆盖声明。", "",
        "| 问题 | 方法 | 平均查询 | 查询中位数 | 有效判定 | 误判 | 误判率95%区间 | 平均循环耗时（秒） |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ])
    for row in analysis["statistics"]:
        low, high = row["wrong_interval_95"]
        lines.append(f"| {row['problem']} | {row['method']} | {row['mean_calls']:.3f} | {row['median_calls']:.1f} | {row['certified']}/{row['repetitions']} | {row['wrong']}/{row['repetitions']} | [{low:.3%}, {high:.3%}] | {row['mean_runtime_seconds']:.6f} |")
    lines.extend([
        "", "## 审计与结论边界", "",
        "全部20000个单元核验原始查询、成本、区间算术或财富累积、检查频率、停止逻辑与汇总。官方区间仅对每题预定种子9000、9249、9499执行财富端点和局部单调性数值复核，共24个单元；其他官方单元没有逐端点独立重算。",
        f"本轮官方端点复核覆盖{analysis['official_replay_checkpoints']}个检查点。浮点求根复核不能替代精确算术的数学证明。", "",
        "记录的循环耗时包含采样、策略计算、证据更新与停止检查，排除配置写入、原始轨迹压缩和最终文件保存；四个工作进程共用一个节点。该数据不代表真实昂贵函数应用的端到端加速，也未测量独占硬件下的性能。", "",
        "确认支持的范围限于已经选择的合成问题族在新随机种子上的可重复性。新颖性、可实现学习效率、真实应用和端到端成本需要单独验证。判定率下降条件是观测值门槛，未证明统计非劣效。",
    ])
    (output / "analysis.md").write_text("\n".join(lines) + "\n")
    provenance = {
        "analysis_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256((output / "analysis.md").read_bytes()).hexdigest(),
        "source": "confirmation/analysis.json",
        "scope": "deterministic rendering of the frozen audit report; no new experiment",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"application_exploration_gate": analysis["application_exploration_gate"], "primary_comparisons": analysis["primary_comparisons"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    report(parser.parse_args().root.resolve())
