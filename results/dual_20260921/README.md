# 2026-09-21 实验与验证索引

论文：Integrability of Subsampled Compositional Diffusions。

中文解读见 `../../docs/DUAL_RESULTS_20260921.md`；作者侧科学自查见 `../../docs/REVIEW_THEORY_20260921.md`。本目录仅包含可纳入版本管理的汇总、逐单元指标、审计、测试记录和图表。完整原始样本及源码归档保留在本工作目录的 `work/` 中。

## 完成的采样与审计

| 阶段 | 单元数 | 汇总目录 | 本地恢复验证 | 运行源码提交 |
|---|---:|---|---|---|
| 探索开发 | 140 | `explore-development/` | `recovery-explore-development.json` | `261566a82092cdffbcdfcec7d1705855e5adab3e` |
| 探索确认 | 250 | `explore-confirmation/` | `recovery-explore-confirmation.json` | `261566a82092cdffbcdfcec7d1705855e5adab3e` |
| 可分离结构诊断 | 40 | `factorization/` | `recovery-factorization.json` | `48f7ad350ea3d11d60c9020679ccb17cf3727ff9` |
| 锚点 CPU 开发 | 24 | `anchored-pilot/` | `recovery-anchored-pilot.json` | `ad030e6467635871dc5f69f0ceaf2d29e5fd4c8c` |
| 锚点 GPU 确认 | 150 | `anchored-confirmation/` | `recovery-anchored-confirmation.json` | `39814c1192ff5ad02744d3d332c4088ffe2663f7` |
| 同一确认集的等价实现复跑 | 150 | `anchored-replay/` | `recovery-anchored-replay.json` | `175b3097759c84e395714103f20f0eba49de283c` |

每个汇总目录包含 `statistics.json`、`cells.csv`、服务器独立检查 `audit.json`。每份恢复检查均核验运行版本中实际记录的源码文件、原始结果校验值、数据集重建、训练模型严格加载及主要指标。两组确认分别使用 400–409、700–709 数据种子，均交叉同一组五个正式模型；不能把运行单元计数解释为独立模型数量。

六份本地恢复检查均通过，合计 604 个采样单元与 150 个同数据复跑单元。`equivalence-local.json` 额外逐项比较全部 150 对原始粒子、权重和对数归一化常数，并核对服务器等价性报告。复跑不增加独立数据集数量。

## 机制与用途诊断

- `conditional/`：固定锚点和非锚点位置的漂移/势函数估计方差；独立 NumPy/SciPy 检查。
- `path-moments/`：高斯路径一阶有限与二阶发散边界，保留失败二阶证书。
- `posterior-explore/`：探索确认全部 250 个采样单元的参数误差和区间覆盖事后诊断。
- `posterior-anchored/`：锚点确认全部 150 个单元的同类诊断；不替代预定 W1/时间门槛。
- `figures/`：SVG 与 PNG 图表。运行时间包含方法准备，图中误差棒为训练模型 × 数据交叉 bootstrap 的双侧 95% 区间。

## 完整原始归档

根目录：`work/recovered-dual-20260921/`。

| 归档文件 | SHA-256 |
|---|---|
| `recovery-explore-factorization.tar.gz` | `37980a2eae0adf8f508ae29361d4fe90de29e47a4bff9db969b75a08584538ab` |
| `recovery-anchored-pilot.tar.gz` | `e53309e65d63f31e4c0d7e78da20ecba05c681799886c295095fbcedbed9ff71` |
| `recovery-anchored-confirmation.tar.gz` | `166b238e587ac0dc0b684ca266d73ead117ec01c841dfa9683d1ee9c5695f3ee` |
| `recovery-anchored-replay.tar.gz` | `75a062982cee50419e4edd97c880a5e87bade448af082de7de1548ce99a2ff94` |

以上归档服务器与本地 SHA-256 相同。展开目录 `new/` 与 `old/` 含原始粒子、权重、输入、参考、逐步记录、源码归档、完成凭证、独立审计和运行日志。局部机制原始数据另存于 `work/anchored-conditional-run-20260921-final-v3/`；高斯路径数据位于 `work/path-moment-diagnostic-v2/`。

恢复等价复跑时，必须将锚点确认与复跑两个归档展开到同一个父目录。复跑的 `assets` 是指向 `../run-anchored-confirmation/assets` 的相对符号链接，两个阶段共享完全相同的固定输入。

五个正式训练检查点及其训练记录另存于同一恢复目录的 `training.tar.gz`，SHA-256 为 `87735b1ce159535f0886cd60f1994746a47be9818cf3118a0002c659b20d5dbc`。该归档与实验启动时发送到新机器的输入相同；每个 `final.pt` 的身份由阶段 manifest 和严格模型加载检查核验。

本地完整测试为 74 项通过、两项 CUDA 检查按设备条件跳过、六项既有 PyTorch 提示。服务器锚点及等价实现 CUDA 检查共十项全部通过，没有跳过。独立数值审计和图表检查与单元测试分开记录。

本轮未更新投稿内容，也未推送远程仓库。科学结论保留各阶段的模型范围、统计限制和未通过项。
