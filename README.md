# ICLR 2027 research

当前论文：[Integrability of Subsampled Compositional Diffusions](output/pdf/ICLR_2027_research_draft_continued_20260921.pdf)。论文研究小批量组合扩散的指数权重可积性，包含有放回与无放回的完整证明、原 SCNet 4,060 个采样实验单元、五次完整条件密度网络训练及后续 H20 实验。另有 936 个确定性批量判定配置。最新一轮增加 512 个开发配置，两组新方案均未通过预设联合门槛，全部负面结果保留。论文采用 ICLR 2027 格式，状态为研究稿、尚未投稿；当前证据不支持通用高性能采样算法的主张。

- [2026-09-21 最新结果、限制与完整成果入口](docs/CONTINUED_ITERATION_RESULTS_20260921.md)
- [本轮分析、数值审计、测试与发布回执](results/continued_iteration_20260921/)
- [完整原始记录分支](https://github.com/freezeng123456/ICLR-2027/tree/results/complete-research-20260921/results)：原始样本、检查点、逐步诊断与失败记录，文件级 SHA-256 清单随各批成果保留。

- [复现入口与归档校验](docs/COMPOSITION_REPRODUCIBILITY.md)
- [无放回和学习型密度扩展的复现入口](docs/EXTENSION_REPRODUCIBILITY.md)
- [完整扩展结果与独立审计](results/composition_extension_20260910/)
- [研究状态与科学结论](docs/RESEARCH_STATUS.md)
- [主实验核验](results/composition_main_20260909/verification.json)与[补充实验核验](results/composition_supplement_20260909/verification.json)
- [LaTeX 主文件](manuscript/main.tex)与[数学审查](docs/COMPOSITION_MATHEMATICAL_REVIEW.md)

## PFN posterior-predictive approximation audit

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
