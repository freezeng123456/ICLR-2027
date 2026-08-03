"""按 roofline 估算给定 GPU 上的推理吞吐、prefill 成本和实验预算。

用来回答三个排期问题：
  1. 大 batch 离线推理时，解码吞吐会卡在带宽还是算力上？封顶多少？
  2. 一次 prefill 要多久？上下文开到多长会把预算吃光？
  3. 到截止日为止，能跑出多少 token？

用法：
    python3 hardware_budget.py                    # 默认 2x H20，8B 模型
    python3 hardware_budget.py --gpu h100 --n 8
    python3 hardware_budget.py --params 32 --gpus 2 --ctx 8192 32768

**这些是理论上限的推算，实测通常只有 30–60%。** 第一周务必用真实推理栈测一遍
（例如 `vllm bench throughput`，固定输入/输出长度、扫 batch、记录饱和吞吐），
再用实测值替换 --mfu 重跑本脚本。宁可低估也不要拿理论值做排期承诺。
"""

import argparse
from datetime import date

# 显存(GB), 带宽(TB/s), BF16 稠密算力(TFLOPS)
GPUS = {
    "h20":  (96,  4.00, 148),
    "h100": (80,  3.35, 989),   # 989 为稠密值；NVIDIA 标称 1979 是含 sparsity
    "h200": (141, 4.80, 989),
    "a100": (80,  2.04, 312),
}

# 常见模型的层数与隐层维度，用于估算注意力项
SHAPES = {8: (32, 4096), 14: (48, 5120), 32: (64, 5120), 70: (80, 8192)}


def decode(params_b, gpu, n_gpus, mfu, active_b=None, bytes_per_param=2):
    """解码阶段的 roofline：读权重是固定开销，算数随 batch 线性增长。

    MoE 的关键区别：**算力只随激活参数走，访存却要按整个模型算**（大 batch 下不同
    token 会点亮不同专家，最坏情况要读全部权重）。H20 显存大带宽高、算力弱，
    这个组合恰好对 MoE 特别有利。
    """
    _, bw, tflops = GPUS[gpu]
    active_b = active_b or params_b
    # 访存按全部权重估（大 batch 下的保守上界）；算力按激活参数估
    t_bandwidth = params_b * 1e9 * bytes_per_param / (bw * 1e12 * n_gpus)
    t_compute_per_batch = 2 * active_b * 1e9 / (tflops * 1e12 * mfu * n_gpus)
    return {
        "t_bandwidth_ms": t_bandwidth * 1e3,
        "t_compute_per_batch_ms": t_compute_per_batch * 1e3,
        "crossover_batch": t_bandwidth / t_compute_per_batch,
        "peak_tok_s_total": 1 / t_compute_per_batch,
    }


def prefill(params_b, ctx, gpu, mfu):
    """prefill 是纯算力活儿，注意力项随上下文长度平方增长。"""
    _, _, tflops = GPUS[gpu]
    layers, d_model = SHAPES.get(params_b, (32, 4096))
    linear = 2 * params_b * 1e9 * ctx
    attention = 4 * layers * ctx * ctx * d_model
    total = linear + attention
    return {
        "tflops": total / 1e12,
        "seconds": total / (tflops * 1e12 * mfu),
        "attention_share": attention / total,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", default="h20", choices=sorted(GPUS))
    p.add_argument("--gpus", type=int, default=2, help="卡数")
    p.add_argument("--params", type=float, default=8, help="模型总参数量（十亿）")
    p.add_argument("--active", type=float, default=None,
                   help="MoE 的激活参数量（十亿）。dense 模型留空")
    p.add_argument("--bits", type=int, default=16, choices=[16, 8, 4],
                   help="权重精度，影响显存与访存")
    p.add_argument("--mfu", type=float, default=0.40, help="有效算力利用率，实测后替换")
    p.add_argument("--ctx", type=int, nargs="+", default=[8192, 32768, 131072])
    p.add_argument("--deadline", default="2026-09-25")
    p.add_argument("--overhead", type=float, default=0.58,
                   help="写作/搭建/调试/重跑占掉的比例，默认 0.58")
    a = p.parse_args()

    mem, bw, tflops = GPUS[a.gpu]
    bpp = a.bits / 8
    kind = f"MoE {a.params:g}B-A{a.active:g}B" if a.active else f"dense {a.params:g}B"
    print(f"=== {a.gpus} x {a.gpu.upper()} | {kind} | {a.bits}-bit | MFU {a.mfu:.0%} ===")
    print(f"单卡 {mem} GB / {bw} TB/s / {tflops} TFLOPS BF16 稠密")
    print(f"合计 {mem*a.gpus} GB / {bw*a.gpus:.1f} TB/s / {tflops*a.gpus} TFLOPS")
    print(f"算术强度 {tflops/bw:.0f} FLOP/byte（对照 H100 {989/3.35:.0f}）")

    w = a.params * bpp
    free = mem * a.gpus - w
    verdict = "装不下" if free < 0 else ("KV 空间紧张，并发上不去" if free < 40 else "宽裕")
    print(f"\n权重占 {w:.0f} GB，{a.gpus} 卡剩 {free:.0f} GB 给 KV cache 与激活 —— {verdict}")

    d = decode(a.params, a.gpu, a.gpus, a.mfu, a.active, bpp)
    print("\n--- 解码（大 batch 离线推理，两卡合计）---")
    print(f"  读权重耗时         {d['t_bandwidth_ms']:.2f} ms/步（与 batch 无关）")
    print(f"  算数耗时           {d['t_compute_per_batch_ms']:.3f} ms/步/batch")
    print(f"  带宽/算力 crossover  batch ≈ {d['crossover_batch']:.0f}"
          f"  —— 超过它就是算力瓶颈，带宽优势不再兑现")
    print(f"  封顶吞吐           {d['peak_tok_s_total']:,.0f} tok/s")
    if a.active:
        print(f"  （MoE 的算力只按 {a.active:g}B 激活参数走，访存按 {a.params:g}B 全量估——"
              f"这正是 H20 这种显存大、算力弱的卡偏爱 MoE 的原因）")

    print("\n--- prefill（纯算力，注意力项 O(n²)）---")
    print(f"  {'上下文':>10} {'FLOPs':>12} {'单卡耗时':>12} {'注意力占比':>10}")
    for ctx in a.ctx:
        r = prefill(a.params, ctx, a.gpu, a.mfu)
        secs = r["seconds"]
        pretty = f"{secs:.1f} s" if secs < 90 else f"{secs/60:.1f} min"
        print(f"  {ctx//1024:>8}k {r['tflops']:>11,.0f}T {pretty:>12} {r['attention_share']:>9.0%}")

    days = (date.fromisoformat(a.deadline) - date.today()).days
    raw = days * 24 * a.gpus
    usable = raw * (1 - a.overhead)
    tokens = d["peak_tok_s_total"] * (usable / a.gpus) * 3600
    print(f"\n--- 到 {a.deadline} 的预算（还剩 {days} 天）---")
    print(f"  理论上限           {raw:,.0f} GPU·h")
    print(f"  扣除 {a.overhead:.0%} 开销后   {usable:,.0f} GPU·h"
          f" ≈ {usable/a.gpus:,.0f} 小时墙钟 ≈ {usable/a.gpus/24:.0f} 天不间断")
    print(f"  可产出输出 token   约 {tokens/1e9:.1f} B（按封顶吞吐算，实际会更少）")


if __name__ == "__main__":
    main()
