# PFN posterior-predictive approximation audit

研究有限 PFN 在自身训练先验下的预测近似误差，以及这种误差对不确定性排序的影响。
目前主要证据来自 RBF GP 和离散水平跳变过程两个受控先验。

## 研究证据

| 先验 | 宽 64、20k 步：平均 gap | 宽 128、40k 步：平均 gap | 两端最差格子 gap |
| --- | ---: | ---: | ---: |
| GP | 0.1048 | 0.0477 | 0.4569 → 0.1761 |
| 跳变过程 | 0.1081 | 0.0768 | 0.3100 → 0.2965 |

单位为 nat。每个检查点在 96 个条件上评估，每个条件的原实验配置为 40 个任务。
这些是已保存的探索性结果，未包含独立训练种子的置信区间。
`gap` 计算数值后验的矩匹配高斯到网络高斯的 KL；它等于相对最佳高斯预测器的条件期望额外 NLL。
连续先验积分使用数值求积，困难区域的求积收敛需要单独检查。

- [当前研究范围](docs/scope.md)：允许的结论、数学定义和补充实验。
- [代码与论文审计](docs/RESEARCH_AUDIT.md)：代码地图、证据缺口、相关工作与 SCNet 实验安排。
- [历史数值记录](docs/pfn-vs-its-own-bayes.md)：原实验表格与探索解释。
- [历史先验错配实验](docs/exp-log.md)：远置上下文、温度缩放和检查点恢复。
- [投稿规则](docs/submission-checklist.md)：摘要 2026-09-18、正文 2026-09-25，均为 AOE；以 [ICLR 官方 CFP](https://iclr.cc/Conferences/2027/CallForPapers) 为准。

## 安装与验收

Python 3.12。在仓库根目录运行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python check_jump_prior.py
.venv/bin/python audit_artifacts.py --output work/artifact-audit.json
```

`audit_artifacts.py` 核对 17 个已有检查点的有限参数、8 个主实验模型的前向和 768 个结果格子，
并生成 SHA-256 清单。它不重新训练历史模型，也无法恢复未记录的历史训练种子。
TabICL、TabDPT、OpenML 实验另装 `requirements-tabular.txt`，其模型权重和数据由各项目提供。

## 训练与决策评估

```bash
.venv/bin/python train_repro.py --prior gp --seed 0 --steps 20000 --width 128 --device cpu --output work/gp-w128-s0
.venv/bin/python train_repro.py --prior jump --seed 0 --steps 20000 --width 128 --device cuda --output work/jump-w128-s0
.venv/bin/python eval_decision.py --prior gp --checkpoints pfn_cond_w64.pt pfn_cond_40k.pt --tasks 200 --output work/decision-gp.json
.venv/bin/python eval_decision.py --prior jump --checkpoints pfn_jump_w64.pt pfn_jump_40k.pt --tasks 200 --output work/decision-jump.json
.venv/bin/python check_quadrature.py --prior gp --output work/quadrature-gp.json
.venv/bin/python check_quadrature.py --prior jump --output work/quadrature-jump.json
```

每次训练使用新的输出目录；已有目录会触发错误，防止覆盖。
训练固定 NumPy 与 PyTorch 种子，保存配置、源码 hash、检查点 hash、步数、用时、硬件和状态。
CPU 与 CUDA 之间的逐位一致性不作保证。CUDA 需要安装适配服务器驱动的 PyTorch。

决策评估使用所有任务和查询点合并后的 coverage，按任务执行 cluster bootstrap。
分别记录网络方差排序、固定网络均值下的 oracle 排序、Bayes 均值与方差的 oracle 曲线。
保存逐任务输入和预测，便于重新计算置信区间与求积敏感性。

## 分支整合

主研究继承 `cursor/amortization-conditioning-5ca4`，包含校准对照分支的历史；
真实表格实验继承 `cursor/iclr-2027-research-direction-8ba3` 的独立末端提交。
选题趋势分支 `cursor/iclr-2027-research-trends-and-topic-proposals-83c0` 保留为历史材料。
当前 main 的研究入口和数据定义以上述文档为准。
