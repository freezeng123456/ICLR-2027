# Tail-informed annealed SMC for the sensor posterior

本实现将传感器因子的尾部精度用于标准 SMC 的参考分布与移动构造。SMC 的几何退火路径、增量权重和重采样沿用 [Del Moral、Doucet、Jasra (2006), *Sequential Monte Carlo samplers*](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf)。Gaussian-reference pCN 的依据见 [Cotter、Roberts、Stuart、White (2013), *MCMC Methods for Functions*](https://arxiv.org/html/1202.0709v3)。本文件给出本实现的混合参考、responsibility 移动和符号翻转的具体推导；这些内容用于说明目标正确性，不主张一般 SMC 或 pCN 的原创性。

## 调用接口

```python
from tail_bridge_smc_20260921 import sample

x, w, logZ, timing, counts, metadata = sample(
    problem,
    particles=8192,
    seed=123,
    reference="mixture",          # "prior", "gaussian", "mixture"
    components=2,                 # mixture 支持 1--4；其余固定一个分量
    covariance_scale=1.0,          # 有限且 >= 1；prior 始终使用单位协方差
    ess_target=0.8,
    moves=3,
    proposal_scale=0.5,            # pCN innovation scale，0 < scale <= 1
    sign_flip=True,               # 每个阶段追加一次全局符号翻转
    mode_steps=32,
    max_stages=256,
    temperatures=None,            # 可选的固定 [0,...,1] 严格递增温度表
    device="cpu",                # "cuda" 或 "cuda:0" 使用同一实现
    return_numpy=True,
)
```

`problem` 可以是现有 `SensorProblem`，也可以是包含 `directions[G,d]`、`noise_std[G]`、`positive_probability[G]`、`observations[G]` 的 dict。读取范围仅为这四项；`truth`、exact posterior、reference samples 和归一化常数均不参与构造或采样。支持 (1\le G\le16,d\ge1\)，所有计算使用 float64。噪声标准差严格为正，方向非零，参数有限。符号概率允许闭区间 `[0,1]`；端点给出已知符号的真实 Gaussian 传感器消融，使用 log-space 精确处理。

返回六元 tuple：

1. `x[N,d]` 和 `w[N]` 默认是 NumPy float64。`return_numpy=False` 返回选定设备上的 Torch float64。输出统计仍会执行一次 CPU 转换并计时。
2. `logZ` 是联合观测证据 `log p(y)` 的 SMC 估计，包含 prior 与 likelihood 的全部 Gaussian 常数。
3. `timing` 包括 `total_seconds`、`preparation_seconds`、`initialization_seconds`、`annealing_seconds`、`output_seconds`。四个分项之和等于总时间。
4. `counts` 给出完整评价计数，定义见下文。
5. `metadata` 保存配置、设备、随机流、输入 hash、参考参数、模式搜索诊断、逐阶段记录与输出统计。所有 metadata 可由 `json.dumps(..., allow_nan=False)` 序列化。

`build_reference(problem, reference=..., components=..., mode_steps=..., covariance_scale=...)` 可单独检查确定性参考构造，返回 `ReferenceSpec`。正式计时调用 `sample`，其内部重新构造参考并计入准备时间。

## 真实目标与参考构造

目标直接使用现有模型的联合密度

\[
\widetilde p(x)=\phi_d(x;0,I)\prod_{g=1}^G
\big[(1-\pi_g)\phi(y_g;-a_g^Tx,\sigma_g^2)
+\pi_g\phi(y_g;a_g^Tx,\sigma_g^2)\big].
\]

定义

\[
P=I+\sum_g\frac{a_ga_g^T}{\sigma_g^2},\quad
c_g=\frac{y_ga_g}{\sigma_g^2},\quad V=P^{-1}.
\]

其 log density 可写为

\[
\log\widetilde p(x)=C-\tfrac12x^TPx+
\sum_g\log\big[(1-\pi_g)e^{-c_g^Tx}+\pi_ge^{c_g^Tx}\big].
\]

`prior` 使用 (q=N(0,I))，不执行模式搜索。`gaussian` 与 `mixture` 共用以下固定的确定性搜索：

- 初值为零点、原 affine 中心及其相反数、全正符号解及其相反数，以及对每个 sensor direction 构造的符号一致解及其相反数，共 `5 + 2G` 个初值。
- 每次迭代计算 (r_g(x)=\Pr(s_g=+1\mid x,y_g))，执行 latent-sign EM 更新 (x_{new}=P^{-1}\sum_g(2r_g(x)-1)c_g)。这是针对同一真实目标的确定性模式搜索。
- 固定执行 `mode_steps` 次，随后按目标 log density 排序。以原始 (P) 的平方 Mahalanobis 距离 `<=1e-8` 合并重复中心。Gaussian 选最高密度的一个中心；mixture 依次选至多 `components` 个中心。
- 使用共同协方差 (\Sigma_q=\texttt{covariance_scale}\,V)，分量权重正比于所选中心的目标密度。由于所有协方差相同，这个权重规则为归一化的峰高启发式。它不会把模式搜索结果当作精确模式概率。

模式搜索只有固定计算预算；可能尚未到达驻点，可能遗漏模式。metadata 保存所有最终 score norm、log density、选中初值索引和实际分量数。多个初值收敛到相同位置时，实际分量数可以小于请求值。该规则同样适用于真正的 Gaussian 目标。

`tail_precision` 始终记录未缩放的 (P)；prior 分支记录参考精度 (I)。`covariance_scale_applied` 在 prior 为 1，其余等于请求值。参考在整个 SMC 过程中固定。所有分量均为正定 Gaussian，因此参考在整个有限维空间具有正密度。参考密度以完整的 `logsumexp(log_weight + Gaussian_logpdf)` 计算。

## 退火权重、移动与归一化估计

设 (r(x)=\log\widetilde p(x)-\log q(x))，桥接密度为

\[
\gamma_\beta(x)=q(x)e^{\beta r(x)},\quad
\pi_\beta(x)=\gamma_\beta(x)/Z_\beta,\quad \beta\in[0,1].
\]

于是 (Z_0=1,Z_1=p(y))。从均匀权重的 (q) 样本开始；从 \(\beta\) 到 \(\beta'\) 的权重增量为 \(e^{(\beta'-\beta)r(x)}\)。代码以 `logsumexp` 更新归一化权重，累积每次的 log normalizer increment。

自适应温度取满足 `ESS >= ess_target * N` 的最大下一温度；若 1 可行则直接终止。二分固定 48 次，中心化残差以避免常数影响 ESS，二分比较留在计算设备上。每个非末阶段执行 systematic resampling 并恢复均匀权重，然后移动粒子；末阶段保留加权粒子并执行相同的移动。达到 `max_stages` 仍未到达 1 会直接报错，不返回完成状态。`temperatures` 可指定固定路径用于复核。

对混合参考 (q(x)=\sum_c w_c\phi_c(x))，先抽取

\[
c\sim r_c(x)=w_c\phi_c(x)/q(x),\qquad
y=m_c+\sqrt{1-s^2}(x-m_c)+sL\xi,
\quad LL^T=\Sigma_q,\quad\xi\sim N(0,I).
\]

每个条件提议 (K_c) 对 \(\phi_c\) 可逆，因此

\[
q(x)K(x,dy)=\sum_cw_c\phi_c(x)K_c(x,dy)
=\sum_cw_c\phi_c(y)K_c(y,dx)=q(y)K(y,dx).
\]

完整参考提议随后接受概率为

\[
\alpha_\beta(x,y)=\min\{1,\exp[\beta(r(y)-r(x))]\}.
\]

此核对固定 \(\pi_\beta\) 满足详细平衡。分量选择必须使用当前点 responsibilities。pCN 参数 `proposal_scale=1` 仍执行 conditional-component 提议；分量选择由当前位置决定。

每阶段默认追加一次全局符号翻转 \(y=-x\)。这是对称提议，其完整 log MH ratio 为

\[
(1-\beta)[\log q(-x)-\log q(x)]
+\beta[\log\widetilde p(-x)-\log\widetilde p(x)].
\]

代码等价地使用 `logq_difference + beta * residual_difference`。该表达式适用于不对称 Gaussian/mixture 参考。

这些恒等式刻画目标与移动核的正确性。有限粒子输出是 SMC 近似；自适应路径的有限样本归一化估计无偏性在本实现中不作保证，`logZ` 本身也不声明无偏。不同阶段的自适应选择与有限粒子模式覆盖需要实际重复实验评估。

## 权重矩的适用范围

有限传感器给出 \(\log\widetilde p(x)=-\tfrac12x^TPx+O(\|x\|)+O(1)\)。共同协方差混合参考同样满足 \(\log q(x)=-\tfrac12x^TQx+O(\|x\|)+O(1)\)，tail 参考的 \(Q=P/s\)，其中 \(s=\texttt{covariance_scale}\ge1\)；prior 的 \(Q=I\preceq P\)。

在精确的桥接分布 \(\pi_\beta\) 下，正增量权重的任意有限正阶 \(k\) 矩具有二次精度

\[
Q+(\beta+k\Delta\beta)(P-Q)\succ0.
\]

因此这些精确分布下的权重矩有限，特别包括二阶矩。这个尾部计算并不量化常数大小，也不保证有限粒子方差足够小、模式覆盖充分、或测试达到论文的精度/时间门槛。

## 计数、计时与复现

一次因子评价定义为在一个点上完整评价一个 sensor factor；值与 score 由同一次责任概率计算得到时计一次。计数规则：

- `factor_preparation_calls = G`：一次参数系数准备，包含 tail 几何所需的因子系数。
- `mode_point_evaluations = (5+2G)*(mode_steps+1)`，`mode_factor_calls` 为该值乘以 (G)；prior 两项均为零。
- `target_point_evaluations = N * [1 + stages * (moves + int(sign_flip))]`，`particle_factor_calls` 再乘以 (G)。包含初始粒子评价、全部 pCN proposals 和 sign proposals，包括被拒绝的提议。
- `factor_calls` 是以上三个 factor-call 分项之和。`sign_factor_calls` 已包含在 `particle_factor_calls`，用作明细，不能再次相加。
- `reference_component_evaluations` 单独计入实际 Gaussian component log-density 计算；`reference_samples=N`。SMC 残差复用、ESS 二分、重采样与纯线性代数不虚增 factor calls，其耗时进入总时间。

三个独立设备 generator 的种子为 `seed`、`seed+1000003`、`seed+2000003`（模 \(2^{63}\)），分别用于 Gaussian motion、分量选择/MH uniforms、resampling。没有全局 RNG seed 修改，也没有在每次移动中创建 generator。

计时在 CPU 参数准备之前开始；包括模式搜索、参考分解、CPU→device copy、reference sampling、全部 SMC 计算、逐阶段诊断、最终 CPU copy 和加权统计。CUDA 在各计时边界同步；调用入口先同步以排除之前排队的设备工作。调用者在进入函数前的网络预测、文件 I/O、warmup、环境导入与程序启动不包含在这里。公平 runner 应逐方法执行独立 warmup，并记录外部并发；本方法文档不使用本地测试秒数作 GPU 性能结论。

`metadata["records"]` 包括 `beta_previous/beta/delta_beta`、重采样前的 `ess_fraction`、`resampled`、pCN `acceptance/mh_proposals/mh_accepted`、`sign_acceptance/sign_proposals/sign_accepted/sign_factor_calls`、本阶段总 `factor_calls`、logZ 增量/累计值，以及移动前的加权残差均值/方差。`moves=0` 时 pCN acceptance 为 JSON `null`；禁用 sign flip 时该接受率亦为 `null`。

最终 statistics 包含加权均值、总体协方差、每坐标 `[.025,.05,.95,.975]` 加权分位数、ESS 和权重范围。reference-responsibility 加权质量单独记录。样本精度/覆盖率的独立评估由外部实验代码计算，采样函数完全不读取 truth。

## 本地验证与解释范围

测试使用真实传感器模型和独立 SciPy 密度/积分，不使用 mock。覆盖独立 prior+likelihood 密度、已知符号且非零均值的 Gaussian 目标、零观测 Gaussian 目标、异质非可分二维目标、真实双峰目标、八维真实模型、精确混合密度、提议详细平衡、完整 sign-flip density ratio、权重望远镜恒等式、模式合并、全评价计数、失败路径和 RNG 复现。CUDA 检查需显式设置 `ICLR_TEST_DEVICE=cuda`，CPU 检验不声称已验证 CUDA。

二维统计诊断使用四次独立采样重复，报告全部每次结果、证据比均值与标准误、均值标准误和 W1；有限重复的统计阈值仅检查明显实现错误。固定开发/确认 cohorts 与研究 gate 由主线程 runner 负责，本地测试不使用这些 cohorts 选择超参数。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ICLR_TEST_DEVICE=cpu \
/Users/zenghang/Documents/Codex/2026-09-09/https-github-com-freezeng123456-iclr-2027/work/venv/bin/python \
-B -m pytest -p no:cacheprovider -q -s tests/test_tail_bridge_smc_20260921.py
```

2026-09-21 指定本地 Python 的最终实测为 **48 passed, 2 skipped in 2.80s**。环境为 NumPy 2.5.3、SciPy 1.18.1、Torch 2.14.0；两项跳过均要求显式 `ICLR_TEST_DEVICE=cuda`。主线程另行运行相同测试，反馈为 **48 passed, 2 skipped in 3.34s**。

异质二维小问题 `G=4,N=4096` 的四次固定独立重复（采样 seeds 360--363，`ess_target=.9`，默认 pCN 与 sign flip）结果如下。W1 是两个坐标的平均值，参考为该小问题的 32768 个独立精确样本；该统计不等于正式研究的 32-direction W1。

- prior：四次 W1 为 `[0.007591920, 0.009671750, 0.010647335, 0.009531005]`，均值 `0.009360502`；证据比 `exp(logZ_est-logZ_exact)` 的均值 `1.002055478`，重复间标准误 `0.008226660`。四次 logZ error 为 `[-0.014049240, 0.007656620, -0.008125605, 0.022328452]`。每次 3 stages、212996 factor calls。
- tail Gaussian：四次 W1 为 `[0.013073915, 0.009691715, 0.019526680, 0.008534529]`，均值 `0.012706710`；证据比均值 `0.995419233`，标准误 `0.004485688`。四次 logZ error 为 `[-0.000050342, -0.004146072, 0.003134487, -0.017425589]`。每次 2 stages、149176 factor calls。
- tail mixture：此小问题的搜索中心合并为一个分量，结果与 Gaussian 分支逐粒子一致。保留这一结果；该小样本中 tail 两分支的平均 W1 大于 prior。

另一个真实非可分双峰二维问题 `G=3,N=16384,seed=194` 实际保留两个参考分量。`covariance_scale=1` 时 1 stage，均值误差范数 `0.005624214`、协方差误差 Frobenius 范数 `0.017736053`、三个投影 CDF probes 的最大误差 `0.005585048`、logZ error `+0.000472546`，总 factor calls `246852`。`covariance_scale=4` 时 3 stages，相应为 `0.007348579`、`0.036788566`、`0.006831177`、`-0.001744003`，总 factor calls `640068`。测试输出同时保留全部计时分项、pCN/sign 接受数与准备成本。

已知符号且非零均值的 Gaussian 小问题验证了参考中心/协方差、均匀终端权重、单阶段完成、pCN 全接受以及与独立 observation-space Gaussian evidence 的 `1e-12` 绝对误差检查。八维 `G=12` 测试对三个参考分支运行真实算法，并通过修改未读取的 truth 字段验证相同输入参数下逐粒子结果可复现。上述检查是数值证据；理论恒等式由前文公式独立说明。

最终采样源码与测试已停止编辑，供主线程开始 GPU smoke。冻结 SHA-256：

```text
fa423f2875ca4f42d97de6998d6147aaaf934e4147b22c1b98b41a0065607a9f  tail_bridge_smc_20260921.py
17b891dbd2a50ec4a3dbe88ce287626a592f32f9b90920f3e4e8f877794fd490  tests/test_tail_bridge_smc_20260921.py
```

GPU 性能、固定开发/确认 cohorts 的精度差单侧 95% 上界和包含准备成本的时间比上界仍待主线程实验。当前不据本地测试秒数声明加速，也不据有限样本通过声明模式覆盖充分。
