# 概率决策实验复现

## 隔离依赖

使用Python3.10或3.12，单独建立环境。原项目测试使用NumPy2；作者官方 `confseq` 的当前固定版本使用NumPy1.x，不能把两套依赖安装进同一个共享环境。

```bash
python3 -m venv work/probability-venv
work/probability-venv/bin/python -m pip install -r requirements-decision-evidence.txt
git clone https://github.com/gostevehoward/confseq.git work/confseq
git -C work/confseq rev-parse HEAD
export PYTHONPATH="$PWD/work/confseq/src"
export MPLCONFIGDIR="$PWD/work/matplotlib"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

官方库的提交必须等于 `5ffe733ca2447a2e28c2c91f3b00086173f2ab2c`，并保持工作目录无修改；若上游HEAD变化，使用该固定提交的独立检出。实验入口会核验提交和工作区状态。Python源文件必须已经提交，入口还会将实际文件的SHA-256与所记录Git提交逐一核对。

```bash
work/probability-venv/bin/python -m pytest -q tests/test_decision_evidence_20260922.py tests_probability
```

## 重现开发实验

原项目的默认 `pytest` 仅收集 `tests/`，使用原项目NumPy2环境；官方库相关测试位于 `tests_probability/`，按上述命令在NumPy1环境显式执行。已提交SCNet作业使用冻结提交9b037725中的测试路径与46项服务器测试，后续测试目录组织不改变该作业的源码。

每次指定全新的结果目录，禁止覆盖已有输出。以下命令以复现目录为例；原始开发结果保留在同名20260922目录。

```bash
work/probability-venv/bin/python run_decision_evidence_20260922.py --output results/reproduce_ordinary --suite ordinary --repetitions 100 --seed-start 0 --workers 4
work/probability-venv/bin/python run_decision_evidence_20260922.py --output results/reproduce_rare --suite rare --repetitions 100 --seed-start 2000 --workers 4
work/probability-venv/bin/python official_cs_control_20260922.py --output results/reproduce_official_betting --method betting --workers 4
work/probability-venv/bin/python official_cs_control_20260922.py --output results/reproduce_official_empbern --method empbern --workers 4
work/probability-venv/bin/python analyze_decision_evidence_20260922.py results/reproduce_ordinary
work/probability-venv/bin/python analyze_decision_evidence_20260922.py results/reproduce_rare
work/probability-venv/bin/python audit_official_cs_20260922.py results/reproduce_official_betting
work/probability-venv/bin/python audit_official_cs_20260922.py results/reproduce_official_empbern
```

首两批原始开发实验使用NumPy2.5.3；官方开发对照使用NumPy1.26.4/SciPy1.13.1。原始软件信息保存于每个目录的 `provenance.json`。跨环境重跑不承诺逐位一致；原始开发计时不用于方法间端到端加速结论。

## 冻结的新种子确认

```bash
work/probability-venv/bin/python run_matched_confirmation_20260922.py --output results/reproduce_matched --workers 4 --repetitions 500 --seed-start 9000
work/probability-venv/bin/python audit_matched_confirmation_20260922.py results/reproduce_matched
```

运行完毕不自动表示通过：必须核对 `done`、`completion.json`、20000个完整单元以及审计生成的 `analysis.json`。`formal_matrix` 检查完整问题/方法/种子集合；`application_exploration_gate` 按预先固定的终点判断是否值得进入应用研究，不能据此宣称达到论文录用标准。

SCNet启动脚本位于 `scripts/prepare_probability_confirmation_scnet_20260922.sh` 和 `scripts/run_probability_confirmation_scnet_20260922.sh`。它们绑定本次已核验账户、解释器、队列和唯一目录，供审计追溯；其他账户应先重新核实资源，避免直接照搬路径。任务状态、准备日志、作业编号和输出位于 `/work/home/zenghang/probability_confirmation_20260922`。共享Python环境未被修改，额外依赖安装在该任务自己的目录中。

实验依赖此前观察的SCNet资源快照：Python3.10.18、Git1.8.3.1，账户目录50GB总量、约7.5GB可用；队列和剩余空间均可能变化。已有研究数据未删除。代码与开发成果发布到 `research/iclr-new-direction-20260922`，保留原研究分支与main。

## 本次结果的独立审计与回收

审计入口为 `audit_matched_confirmation_20260922.py`，服务器包装程序为 `scripts/audit_probability_confirmation_scnet_20260922.sh`。它按顺序审计smoke和confirmation，输出保存在两个原始目录的 `analysis.json` 与 `SHA256.json`，总状态和资源日志保存在任务根目录的 `audit.status`、`audit.log`。只附着已经启动的审计；状态RUNNING时不重复启动。

`scripts/finalize_probability_confirmation_scnet_20260922.sh` 等待这次已知的审计进程完成，随后使用 `report_probability_confirmation_20260922.py` 生成包含全部40项配置、三项主比较及审计范围的 `decision_review/analysis.md`，并由 `scripts/package_probability_confirmation_20260922.py` 验证完整配置矩阵、Slurm正常退出、4CPU/8GB/90分钟资源请求以及源码归档哈希。此流程不会运行新的实验。

回收包名称为 `probability_confirmation_20260922.recovery.tar.gz`，与同名校验回执保存在服务器用户目录；流程状态和日志为 `probability_confirmation_recovery_20260922.status` 与 `.log`。各步骤拒绝覆盖已有结果，异常时先检查持久状态和日志。完整数据包包含两个原始结果目录、日志、源码归档、独立审计、报告和逐文件SHA-256清单；环境安装目录 `code/` 不重复归档，其实验源码由固定提交的 `source.tar.gz` 保存。

已结束的审计不再运行。归档续接入口为 `scripts/resume_probability_packaging_scnet_20260922.sh`，使用独立状态 `probability_confirmation_packaging_20260922.status` 和日志 `.log`；旧流程的FAILED状态及原始日志保持不变。续接要求审计为COMPLETE、既有报告存在、归档与 `recovery_evidence` 均不存在，并将旧流程文件保存到 `recovery_history/`。服务器部署文件为 `package_probability_confirmation_20260922_timelimit.py`，SHA-256为 `5e2bac53a11dfc9e5bcf0f05c153b25c40f0f437955103e66d54025f72957047`；续接入口SHA-256为 `62ee330ea8b24347f8956aa1d2e20a2e38233908351c406be53bd26d3d8a239a`，来源提交为 `c60690f6a719fb2c2023e703019d16ead079a567`。资源核验使用实际 `sacct` 输出中的 `Timelimit` 列名，并检查主作业及两个步骤的完整集合。真实记录及破坏条件测试共10项通过，命令为 `python -m pytest -q tests/test_probability_recovery_20260922.py`。此测试只检查保存的资源记录，不生成实验样本。

后续跟进首先读取新的归档状态和日志；若状态文件、归档或恢复目录已存在，只检查进度，禁止再次启动。归档进程日志在原始结果目录之外，以保持包内清单稳定；完成后还须单独保留续接状态与日志。

本次续接已于2026-09-22 07:45:55（北京时间）启动，09:27:51完成，最终状态COMPLETE、退出0；禁止再次启动。服务器已核对两个部署文件的SHA-256，资源读取函数也通过同一真实记录。GitHub原始文件域和Git传输当时不可用，使用GitHub官方文件API获取了上述固定提交的原文，未使用第三方镜像或提供认证材料。该传输仅部署归档程序，实验源码提交保持不变。

完成回执记录归档为1185568247字节、80057个文件、20005个单元，SHA-256为 `9b9d0ffe3eeff6ec537b1d8a4d9f26d9faced70c2df1bfda541fc64f318998f9`；服务器完成后的独立哈希计算与回执一致。包内逐文件清单 `recovery_evidence/files.json` 的SHA-256为 `bde240eb8daa7e9720324093be37148ae4351fb98c1eb89e9d9206e78cce120a`；它记录80056个文件，清单自身作为第80057个文件归档。回执、结构核验记录和包外续接状态/日志均已按服务器SHA-256恢复到[工作流证据目录](../results/probability_confirmation_remote_review_20260922/workflow_evidence/)。本地尚未获得完整归档，已请求用户下载到Downloads并提供路径。

该数据包仅用于传输。收到后必须核对归档哈希、逐文件清单、全部20005个单元的配置/原始轨迹/汇总/完成标记，再将展开后的完整结果通过普通Git推送至指定仓库。只有远端引用、文件清单及本地恢复均核实后，才能将本轮标记为已完整发布。服务器目前没有可用的GitHub推送认证；本地仓库可以正常推送，完整数据回收尚未完成。
