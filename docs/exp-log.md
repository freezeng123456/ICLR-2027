# 实验记录：校准对照 + 丢失检查点

> 2026-08-19。用户说「继续完成实验」。本轮只补必须补的对照，不展开新轴。

## 这一轮要回答的问题

1. **方法半篇能不能活**：far-anchor 降 ECE / log loss，是不是温度缩放或「给真实上下文加噪」也能做到？
   - 脚本：`exp_calib_baselines.py`
   - 若温度缩放追上，方法降格为「不需要 ID 校准标签的校准器」，机制半篇仍在。
   - 若上下文注噪追上且准确率不掉，far 这个设计是多余的。
2. **纠缠结论能不能复现**：`pfn_C.pt` 实际是 45k 步，15k 的 C 和 A144（`pfn_Z.pt`）从未进仓库。
   - 先把当前权重备份为 `pfn_C_45k.pt`
   - `train_missing.py` 重训 C@15k、A144、`pfn_gp.pt`
   - `eval_restored.py` 用与原表相同的种子 2024 重测
3. **合成高斯云之外**：`exp_real_calib.py` 在 sklearn / OpenML 小表上重复同一对照。

## 存盘 bug

`train()` 曾经用 `pfn_{name[0]}.pt`。`C_long` 的首字母也是 C，45k 跑把 15k 的 `pfn_C.pt` 覆盖了。
现改为 `ckpt_path(name)`：取括号前的整段名字，并允许传入 `save_path`。已有检查点存在时，`exp_why_axis.py` 默认跳过，除非 `--force`。

## 检查点

| 文件 | 含义 | 状态 |
| --- | --- | --- |
| `pfn_A.pt` | 12 个长度尺度，噪声 0.1，15k | 原仓库，保留 |
| `pfn_B.pt` | ℓ=1.0，12 档噪声，15k | 原仓库，保留 |
| `pfn_C_45k.pt` | 12×12，45k（原 `pfn_C.pt`） | 本轮备份 |
| `pfn_C.pt` / `pfn_C_15k.pt` | 12×12，15k | 重训中 |
| `pfn_Z.pt` | A144，15k | 重训中 |
| `pfn_gp.pt` | 最早 GP-PFN，15k | 重训中 |

## 如何跑

```bash
python exp_calib_baselines.py      # ~20 min CPU，TabICL 对照
python exp_real_calib.py           # sklearn + OpenML
python train_missing.py all        # C_15k、Z、gp，各约 11 min
python eval_restored.py
python eval_gp_prior_tilt.py       # 需要 pfn_gp.pt
```

## 结果

数字进 `results/calib_baselines.json` 和 `results/real_calib.json`，跑完后写回本节。
