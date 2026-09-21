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
work/probability-venv/bin/python -m pytest -q tests/test_decision_evidence_20260922.py tests/test_official_cs_control_20260922.py tests/test_matched_confirmation_20260922.py tests/test_audit_matched_confirmation_20260922.py
```

## 重现开发实验

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
