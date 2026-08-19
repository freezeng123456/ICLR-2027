# PFN 相关的脚本与结果

这些脚本服务于分支 `cursor/iclr-2027-research-direction-8ba3` 上的 PFN 先验掰动工作
（PR #2）。它们依赖那个分支的 `exp_why_axis.py`、`eval_why_axis.py` 和训练好的
`pfn_A.pt` / `pfn_B.pt` / `pfn_C.pt`，所以要这样跑：

```bash
git checkout cursor/iclr-2027-research-direction-8ba3
cp <本分支>/pfn/*.py .
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn tabicl      # 只有表格侧实验需要 tabicl
python3 <脚本名>.py
```

每个脚本旁边都有一份 `*_output.txt`，是已经跑好的完整输出，不想重跑可以直接看。

## 实验脚本

| 脚本 | 做什么 | 耗时（CPU） |
|---|---|---|
| `exp_calib_baselines.py` | **决定性对照**：anchor 噪声校准 vs 温度缩放 / 概率收缩 / 上下文注噪 / 切保留集 / 额外真实数据。默认 30 种子 | 约 60 分钟 |
| `exp_gp_vs_rescale.py` | GP 侧：anchor 的改变能不能被单个常数复现（逐点结构 vs 均匀缩放） | 约 10 秒 |
| `rebuild_checkpoints.py` | 补回丢失的 `pfn_Z.pt`（A144）与 `pfn_C15000.pt`，并修掉存盘与随机种子两个 bug | 约 40 分钟 |

结论见 [docs/10-温度缩放对照实验结果.md](../docs/10-温度缩放对照实验结果.md)。

## 讲解用的演示脚本

这几个不做新实验，只是把已有的东西用真实数字摊开，用于理解项目在做什么：

| 脚本 | 讲什么 |
|---|---|
| `demo_concrete.py` | 一个任务长什么样、喂进网络的张量、损失函数、我们在估计什么 |
| `demo_training.py` | 样本从哪来（逐步拆开生成过程）、一步训练里发生什么、为什么是这个损失 |
| `demo_ell.py` | ell（长度尺度）是什么，同一随机向量只改 ell 的 ASCII 曲线对比 |
| `demo_grid.py` | 那 12 个 ell 意味着什么——它就是先验本身；支撑之外会顶到边界 |
| `demo_corr.py` | `k(x,x')` 里的"相关性"指什么，采 4000 条随机曲线做散点验证 |
| `demo_chain.py` | 完整链条：ell → 协方差 → 对观测的加权 → 预测；anchor 的两条通路 |

对应的讲解见 [docs/09-PFN项目全景.md](../docs/09-PFN项目全景.md)。

## 已知问题（跑之前先看）

- **`exp_why_axis.py` 的 `train()` 没设 `torch.manual_seed`**，训练不可复现。
  实测同配置重训，相关系数差 0.09–0.11。`rebuild_checkpoints.py` 里已修，
  但源脚本还没改。
- **`train()` 用 `f"pfn_{name[0]}.pt"` 存盘**，只取首字母。这个 bug 已经让
  `exp_c_longer.py` 的 "C_long" 覆盖掉了 15000 步的 `pfn_C.pt`。改动前先备份。
