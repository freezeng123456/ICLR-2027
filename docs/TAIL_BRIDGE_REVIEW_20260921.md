# Tail-informed annealed SMC：数学、证据与后续算例审查

AI-assisted internal working assessment · 2026-09-21

作者授权的内部研究审查。本文包含自行核对的数学推导、源码的静态检查、原始文献查证，以及尚未执行的小实验建议；不构成人类独立同行评审。使用 research 与 math-paper-writing 的证据和数学表述要求。唯一写入文件为本报告；没有修改实现、执行新采样实验、连接实验服务器、提交或推送。初始任务的基准 HEAD 为作者提供的 `e9e35c3`；最终核对对象为作者标定冻结于 `b7aa4f0ee8d2d80eecbcf9967b820e62597e2d76` 的 sampler、runner 与协议。本次以读取内容及 SHA-256 识别文件，未运行 Git 验证提交与文件之间的映射。

审查时间顺序保留如下：A，先读取尚无全局翻转的实现快照并形成初审；B，作者通知加入 covariance scale、sign flip 和冻结协议，本次补充 §3.1、§4.3 的重点推导；C，作者通知上述冻结提交及 GPU 已启动，本次重读最终 sampler、runner、协议，更新 §5.2 与文件哈希。A 的“新框架缺少翻转”仅属于早期快照，不能作为 C 的当前缺陷。§7、§8 保留最初请求下的研究建议，本次 C 核对没有扩展任务或要求更改正在运行的冻结实验。

## 1. 当前判断

这个方向在数学上可以成立。当前实现中的 responsibility 选分量、分量 pCN、完整混合密度残差及 MH 接受率相互匹配；权重与 logZ 的累计也符合其“每个中间温度重采样、最终温度保留权重”的设计。最终带权粒子逐粒通过目标不变 MCMC 后保留各自权重是合法操作，证明见 §4.3。有限 covariance scale≥1 的单高斯及有限高斯混合参考满足全部固定正阶 IS 矩条件，精确范围见 §3.1。新加入的符号翻转使用完整桥密度比，公式正确。静态检查没有发现调用真实后验参考或生成参数 truth 的路径。

已重读 `run_tail_bridge_study_20260921.py` 及 `docs/TAIL_BRIDGE_STUDY_PROTOCOL_20260921.md`：所有方法 N=16384，26 个设置×16 个开发问题；候选须同时对开发选择基线和固定 64-stage、scale=0.15 基线通过门槛。12 个 SMC 基线全部包含 sign flip，12 个候选也包含同类完整校正。根据作者最新提供的状态，GPU 实验已启动；本次没有连接服务器核实进度。本次没有发现上述两项重点中的致命数学错误；运行结果及方法优越性尚未验证。分层 +0.004 在冻结文本中的开发/确认适用范围见 §5.2。

最重要的研究边界有五项：

1. **现有 sensor 的先验重要性权重已经具有所有正阶矩。** 高斯观测似然有界，因此新参考在这个算例上的贡献必须体现为精度、模式覆盖或总成本改善。它不能被表述为恢复该静态后验重要性采样原本不存在的高阶矩。
2. **Gaussian-mixture + SMC + pCN 已有直接原始文献。** Lu、Jia、Meng 的 2025 年预印本是必须正面比较的近邻工作。当前 responsibility 核有清楚的可逆性证明，但通用混合 pCN 或 Gaussian-reference SMC 的首创主张缺乏支持。
3. **混合 pCN 的尺度等于 1 仍可能难以跨模式。** 此时分量选择继续依赖当前位置；独立从完整 q 抽样需要另一个明确的 proposal。
4. **粒子自适应温度下，正确的 logZ 累计不自动给出有限粒子无偏性。** 当前 metadata 明确不作无偏声明，范围合适。正阶矩有限也不能证明重采样相对方差小或长期稳定；scale>1 时可得到的额外路径上界见 §3.1。
5. **真正有辨识力的下一步是独立检查点组成的非可分离 learned posterior。** 第 7 节给出二维、五个不同训练检查点、32 分量评价参考的小实验。它能检验仅访问已训练因子的组合任务；其合成数据仍有解析评价真值，应用表述必须保留这个限制。

## 2. 目标、参考与混合 pCN 的详细平衡

### 2.1 明确目标和可用信息

固定一个问题，令未归一化目标为 \(\widetilde p(x)>0\)，参考为已归一化密度

\[
q(x)=\sum_{j=1}^{J}\alpha_j\phi_j(x),\qquad
\phi_j=\mathcal N(m_j,S_j),\quad
J<\infty,\quad \alpha_j>0,\quad \sum_j\alpha_j=1,\quad S_j\succ0.
\]

参考参数须在正式粒子系统启动前固定。允许使用观测、模型参数、可计算的未归一化目标及固定预算的模式搜索；本节不涵盖用同一运行中的粒子不断重拟合 q 后仍沿用原权重公式的算法。

定义

\[
R(x)=\log\widetilde p(x)-\log q(x),\qquad
\gamma_\beta(x)=q(x)e^{\beta R(x)},\qquad
\pi_\beta=\gamma_\beta/Z_\beta,\quad 0\leq\beta\leq1.
\]

只要 \(0<Z_1=\int\widetilde p<\infty\)，Hölder 不等式给出 \(Z_\beta\leq Z_1^\beta\)，而 \(Z_0=1\)。这里的终点是静态后验。论文中的有限 Euler 网格 Feynman–Kac 路径目标、随机批量核目标与这个静态目标需要分别命名；路径 twisting 的定理不能直接作为本算法的重采样方差定理。

### 2.2 可逆性证明

在当前位置 x 选择

\[
\rho_j(x)=\frac{\alpha_j\phi_j(x)}{q(x)},\qquad J_x\sim\rho(x),
\]

然后用固定分量尺度 \(0<s_j\leq1\) 提议

\[
y=m_j+a_j(x-m_j)+s_j\xi_j,\qquad
a_j=\sqrt{1-s_j^2},\qquad \xi_j\sim\mathcal N(0,S_j).
\]

记分量核为 \(K_j\)。在 \(x\sim\phi_j\) 下，\((x,y)\) 的联合高斯均值为 \((m_j,m_j)\)，协方差为

\[
\begin{pmatrix}S_j&a_jS_j\\a_jS_j&S_j\end{pmatrix}.
\]

该联合分布交换 x、y 后不变，故 \(\phi_j(dx)K_j(x,dy)=\phi_j(dy)K_j(y,dx)\)。于是混合核

\[
K(x,dy)=\sum_j\rho_j(x)K_j(x,dy)
\]

满足

\[
q(dx)K(x,dy)
=\sum_j\alpha_j\phi_j(dx)K_j(x,dy)
=q(dy)K(y,dx).
\]

因此以 \(\pi_\beta\) 为目标的 MH 接受率恰为

\[
A_\beta(x,y)=1\wedge\exp\{\beta[R(y)-R(x)]\}.
\]

这是完整证明；无需假定各分量协方差对易。当前实现进一步限制为所有参考分量共享一个协方差。高斯 pCN 的基础可逆性及 Gaussian-reference MH 的背景见 [Cotter–Roberts–Stuart–White，2013，§4.2、§6](https://eprints.maths.manchester.ac.uk/2215/1/1202.0709v3.pdf)。上面的责任概率混合推导是本次对所提算法的直接核对。

必要实现条件如下：每次 proposal 重新计算当前点的 q-responsibility；R 使用完整 `logsumexp` 混合密度；接受率乘当前温度 \(\beta\)，权重增量乘温度差 \(\Delta\beta\)。用硬分配、目标后验 responsibility、随 x 改变的尺度，或保存过期分量标签后继续套用同一边缘核证明，都需要重新推导。若明确维护增广状态，联合目标 \(\alpha_j\phi_j(x)e^{\beta R(x)}\) 也能支持相应 Gibbs/MH 构造，状态和权重必须与之配套。

尤其在 \(s_j=1\) 时，

\[
K(x,dy)=\sum_j\rho_j(x)\phi_j(dy),
\]

通常仍依赖 x。真正的全局独立参考 proposal 是 \(K_{\rm ind}(x,dy)=q(dy)\)，其分量按固定 \(\alpha_j\) 抽取。它同样对 q 可逆，可用同一个残差 MH 接受率，并可与 K 按预先固定概率混合。单高斯的尺度 1 特例才直接等同于该独立 proposal。当前测试覆盖尺度 1 的详细平衡，但没有因此证明跨模式迁移速度。

关于混合权重分母还需保留一个限定：本几何桥必须使用 q。另一个独立重要性采样构造若先抽 \(J\sim\alpha\)、再抽 \(X\sim\phi_J\)，采用 \(\widetilde p(X)/\phi_J(X)\) 在各分量覆盖目标支持集时也可以有正确的一阶积分均值；它对应不同的增广构造和方差，不能用来替换这里的 R。完整混合分母的经典重要性采样背景见 [He–Owen，2014，§3.2，式 (11)](https://arxiv.org/html/1411.3954v1)。

## 3. 尾部能保证什么，当前算例已经具备什么

假设固定问题满足全局双侧包络

\[
\log\widetilde p(x)=-\tfrac12x^T\Lambda x+r(x),\qquad
\Lambda\succ0,\qquad |r(x)|\leq c_0+c_1\|x\|.
\]

若参考的有限个分量共享 \(S_j=\Lambda^{-1}\)，则

\[
\log q(x)=-\tfrac12x^T\Lambda x+
\log\sum_j\exp\{b_j+x^T\Lambda m_j\}.
\]

有限 `logsumexp` 可由任一项下界、由最大项加 \(\log J\) 上界；因此第二项的绝对值被 \(c_2+c_3\|x\|\) 控制。得到 \(|R(x)|\leq c_4+c_5\|x\|\)。对每个固定 \(a>0\)，

\[
\mathbb E_q[(\widetilde p(X)/q(X))^a]
\leq e^{ac_4}\mathbb E_q e^{ac_5\|X\|}<\infty.
\]

最后一步来自有限维高斯具有所有线性范数指数矩。理想桥分布 \(\pi_\beta\) 同样有高斯二次包络，故每个固定桥步的增量 \(e^{\Delta\beta R}\) 在理想源分布下有所有固定正阶矩。独立 q-IS 对有界函数可得到通常的自归一化一致性和 CLT；重采样后的相关粒子不属于这个独立样本结论。

常数依赖问题、参考、模式位置和权重；这里没有维数、因子数、温度步数或随机参考拟合过程的统一方差界。一个完全解析的警示例为

\[
p=\tfrac12\mathcal N(-m,\sigma^2)+\tfrac12\mathcal N(m,\sigma^2),\quad
q=\mathcal N(m,\sigma^2),\quad
\operatorname{Var}_q[p(X)/q(X)]
=\tfrac14\{e^{4m^2/\sigma^2}-1\}.
\]

所有正阶矩有限与极大的实际相对方差可以同时成立。高 ESS、正常 logZ 和低 MH 拒绝率也不能单独排除未发现模式。

如果用局部 Laplace 协方差替代尾部协方差，须重新检查矩条件。单高斯参考精度为 \(\Lambda_q\) 时，a 阶矩的严格二次条件是

\[
a\Lambda-(a-1)\Lambda_q\succ0.
\]

负特征值在上述双侧线性余项条件下意味着该矩发散；半正定边界取决于余项。有限混合中有一个固定正权重、尾部匹配的防御分量，就足以用 \(q\geq\alpha_j\phi_j\) 保证 a≥1 的 IS 矩；0<a<1 可由 Hölder 控制。这种充分条件没有赋予 mode-height 权重最优性。

**当前 sensor 的特殊事实。** 现有目标为

\[
\widetilde p(x)=\varphi_d(x)\prod_g
\{\pi_g\varphi_{\sigma_g}(y_g-a_g^Tx)
+(1-\pi_g)\varphi_{\sigma_g}(y_g+a_g^Tx)\}.
\]

尾部精度是 \(\Lambda=I+\sum_ga_ga_g^T/\sigma_g^2\)。直接从因子参数求这个矩阵完全合法，无需枚举真实后验模式。另一方面，每个似然因子最多为 \((2\pi\sigma_g^2)^{-1/2}\)，所以以标准高斯先验为参考的权重也有界。这已经给出所有正阶矩。由此，本静态桥的尾部定理是一项充分性说明；在现有 sensor 上，评价重点应是参考质量、有限预算模式覆盖及效率。不能把有限 Euler 路径权重的问题移植成先验静态 IS 的问题。

### 3.1 covariance scale≥1 的完整范围

令 c 表示乘在协方差上的尺度，与 pCN proposal scale 区分。对一个或有限多个正权重、有限均值的参考分量，取共同协方差 \(c\Lambda^{-1}\)，其中 \(1\leq c<\infty\)。此时

\[
\log q_c(x)=-\frac1{2c}x^T\Lambda x+O(1+\|x\|),\qquad
R_c(x)=-\frac12(1-c^{-1})x^T\Lambda x+O(1+\|x\|).
\]

对任意固定 \(a>0\)，IS 矩被积函数 \(\widetilde p^a q_c^{1-a}\) 的二次精度为

\[
H_a=\left[a-\frac{a-1}{c}\right]\Lambda
=\left[\frac1c+a\left(1-\frac1c\right)\right]\Lambda\succ0.
\]

因此全部固定正阶矩成立，包括 a<1、a=2 和任意大的固定 a。c=1 对应二次项完全抵消后的线性残差；c>1 时 R 含负定二次项，具有有限全局上界，通常不具有双侧“绝对值至多线性”的性质。证明不可把这两种余项表述混用。c=4 时系数为 \((3a+1)/4\)，仍严格为正。

对理想桥分布 \(\pi_\beta\) 下的增量 a 阶矩，令 \(t=\beta+a\delta\geq0\)，其分子为 \(\int q_c e^{tR_c}\)，二次精度是

\[
\left[\frac1c+t\left(1-\frac1c\right)\right]\Lambda\succ0.
\]

这些结论条件于固定问题及固定参考参数。没有对随机拟合过程、模式间距或不断增长维数作统一积分保证。有限多个不同尺度 \(c_j\geq1\) 的混合也能用任一正权重防御分量和 Hölder 证明全部正阶 IS 矩；当前代码仅实现共同尺度。

对于 c>1，若 \(B=\sup_x R_c(x)<\infty\)，任一完成到 β=1 的运行还满足逐路径上界

\[
\widehat Z=\prod_k\sum_iW_{k-1,i}e^{\delta_kR_c(X_{k-1,i})}
\leq\prod_ke^{\delta_kB}=e^B.
\]

这个界允许温度自适应和中间重采样，直接限制完成路径上的 normalizer 估计值；它不能给出有用的小相对方差、自动完成概率或自适应无偏性。c=1 时 R 可能向上无界，不能使用此界。失败运行没有产出完整 Z 估计，必须保持失败状态。

代码 `tail_bridge_smc_20260921.py:116–117,195` 拒绝非有限尺度及 c<1，实际将 c 乘在协方差上。`:121–125` 的 prior 分支始终返回 N(0,I)，不受 covariance scale 改动；`:536–537` 分别记录请求值与实际应用值。它的矩结论应使用前述有界 sensor likelihood，而不能把 prior 分支 metadata 中的 I 当作真实目标尾部精度 Λ。

## 4. 温度、重采样、权重和 logZ

### 4.1 任意归一化旧权重下的正确公式

设当前位置及权重为 \((x_i,W_i)\)，\(\sum_iW_i=1\)，从 \(\beta\) 前进到 \(\beta'=\beta+\delta\)。先在移动前的粒子上计算

\[
\ell_i=\log W_i+\delta R(x_i),\quad
c=\operatorname{LSE}(\ell_1,\ldots,\ell_N),\quad
\log W'_i=\ell_i-c,\quad
\log\widehat Z\leftarrow\log\widehat Z+c.
\]

若重采样，按 \(W'_i\) 抽祖先，同时索引所有缓存的状态和残差，然后将权重重置为 \(1/N\)。重采样本身不增加 logZ。之后用 \(\pi_{\beta'}\)-不变的 MH 移动；移动本身也不增加 logZ。若未重采样，移动后的粒子继续携带原来的 \(W'_i\)，不能因执行了 MCMC 就设为等权。最终可以返回带权粒子，或另作一次明确记录的最终重采样。几何桥、增量权重和归一化常数积估计的原始框架见 [Del Moral–Doucet–Jasra，2006，§2.3.1、§3.2.1、§3.3.2.3](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf)。

logq 必须含混合权重、高斯行列式和归一化常数。给 \(\log\widetilde p\) 加常数 c 不改变最终归一化目标、ESS 或 MH 决策，但应使最终 logZ 加 c。sensor 的 prior×likelihood 归一化常数是观测证据；归一化因子后验之积再除先验幂得到的 composition normalizer 与它相差因子边缘证据之和，见 `nonseparable_sensor_model_20260921.py:145–155`。learned posterior 因子的边缘证据通常未知，组合 logZ 应保持其自身语义。

### 4.2 adaptive ESS 与 adaptive resampling 要分别说明

令 \(u_i=e^{\delta R_i}\)。一般旧权重下，实际新 ESS 和 conditional ESS 为

\[
\mathrm{ESS}_{\rm new}=\frac{(\sum_iW_i u_i)^2}{\sum_iW_i^2u_i^2},\qquad
\mathrm{CESS}=N\frac{(\sum_iW_i u_i)^2}{\sum_iW_i u_i^2}.
\]

旧权重等权时二者相同。一般情况下，CESS(0)=N，且对非负温度增量单调不增：令 \(F(t)=\log\sum_iW_ie^{tR_i}\)，则 \(\log(\mathrm{CESS}/N)=2F(\delta)-F(2\delta)\)，导数为 \(2F'(\delta)-2F'(2\delta)\leq0\)。这使二分温度有明确依据。

实际新 ESS 未必单调。例如 \(W=(0.9,0.1)\)、\(R=(0,\log9)\)，从 δ=0 到 δ=1 时，实际 ESS 从 \(1/0.82\) 增加到 2。若将来改为只在 ESS 低于阈值时重采样，就应使用含旧权重的 CESS 选择温度，再用实际 ESS 决定重采样。相关定义与自适应温度讨论见 [Zhou–Johansen–Aston，2016，§3.3.2–3.4，式 (3.15)–(3.16)](https://api.repository.cam.ac.uk/server/api/core/bitstreams/c53d7c7a-8376-4ec8-bedb-71a9d08ea2b4/content)。

**当前实现没有这个不等权温度错误。** `tail_bridge_smc_20260921.py:463–476` 在所有中间温度重采样；最终阶段保留权重，完成移动后便结束。因此每次调用 `:321–342` 的温度二分时旧权重都是均匀的。当前名称应描述为“自适应温度、所有中间阶段系统重采样”，不宜描述成已实现一般 ESS-triggered resampling。

固定参考、固定温度和适当不变核，配合条件无偏的重采样，可使用通常的 \(\widehat Z\) 无偏性论证；\(\log\widehat Z\) 本身通常有 Jensen 偏差。由同一粒子系统自适应选取温度时，一般不能保留有限 N 的前一项无偏结论。Zhou 等的 §3.3.2 明确提醒这个边界。独立 pilot 冻结温度后另启生产粒子系统，可以回到条件于冻结设计的标准框架；事后保存同次运行产生的温度不具备这种效果。当前 `:552` 的 `normalizer_unbiasedness_claim=False` 与这个边界一致。

### 4.3 最终带权粒子逐粒 MCMC 后保留权重是正确的

令最终目标为 π，K 为固定的 π-不变核；给定移动前粒子信息 \(\mathcal F\)，对每个 i 抽 \(Y_i\sim K(X_i,\cdot)\)，保留 \(W_i\)。则

\[
\mathbb E\left[\sum_iW_if(Y_i)\mid\mathcal F\right]
=\sum_iW_iKf(X_i).
\]

这使加权经验测度经 K 传播；目标层面有 \(\pi K=\pi\)。在固定设计的标准无偏未归一化测度条件下，若 \(\widehat\gamma_N=\widehat Z\sum_iW_i\delta_{X_i}\) 满足 \(\mathbb E\widehat\gamma_N(g)=Z\pi(g)\)，则

\[
\mathbb E\left[\widehat Z\sum_iW_if(Y_i)\right]
=\mathbb E\widehat\gamma_N(Kf)=Z\pi(Kf)=Z\pi(f).
\]

整个论证没有要求旧权重等权，也没有要求粒子原本独立。亦可通过增广目标 \(\pi(dx)K(x,dy)\) 理解：向前传播与目标使用同一个条件核，旧重要性权重原样保留。K 只需不变；多个分别可逆的 MH 步顺序组合，一般只保证组合核不变，无需把组合核也称为可逆。

当前 `tail_bridge_smc_20260921.py:466–489,508–512` 在 β=1 完成重加权后不重采样，用 β=1 做逐粒 pCN-MH 和完整密度符号翻转，随后返回未改动的权重，符合这个构造。`run_tail_bridge_study_20260921.py:79–90` 保存并传递这些权重；`run_sensor_study_20260921.py:46–67` 的 W1、均值、协方差与分位数也实际使用权重，没有退化成不加权指标。

边界有三项：有限 N 的归一化经验测度一般仍有偏；有限次移动不把每个粒子变成目标独立样本；最终权重代表路径上的加权传播，通常不等于最终位置的 \(\widetilde p(Y_i)/q(Y_i)\)。因此既不能把返回权重重置为 1/N，也不能在审计中重算最后一个密度比替代它们。只根据权重计算的 ESS 在这段移动中保持不变，无法度量移动带来的模式覆盖改善。自适应温度的有限 N 无偏性限制仍由 §4.2 单独处理。

## 5. 当前源码的有界检查结论

行号对应第 9 节记录的实际读取版本。以下“相符”均指源码与上述数学契约相符，不代表本次重新运行通过。

| Location | Observation / evidence | Importance / action |
|---|---|---|
| `tail_bridge_smc_20260921.py:13–15,49–67,204–227` | 目标只复制 directions、noise、sign probability、observations；自行计算含常数的 prior×likelihood。没有导入 sensor evaluator，也没有访问 truth。 | 本次静态检查未见参考泄漏。保留这个受限输入接口，未来 learned adapter 只接受训练头参数。 |
| 同文件 `:127–196` | 精度直接由观测模型得到；固定 5+2G 个初值、固定次数 latent-sign EM；按目标高度选最多四个有限中心；保存终止梯度范数。 | 不需要真实模式。32 次迭代不保证找到驻点或所有模式；称为候选中心，报告梯度范数、去重数量和失败情况。高度归一化只是一种可评价的 proposal 权重规则。 |
| 同文件 `:261–311,362–390` | 完整混合 log density、当前 responsibility、pCN 和 βR-MH 对应第 2 节；sign flip 使用完整桥密度。 | 可逆性契约相符。固定权重全局 q-refresh 与当前 responsibility-pCN 必须区分。 |
| 同文件 `:345–351,463–515` | 使用旧 logweights 累计增量 normalizer；中间重采样同步复制 residual；最终保留权重并通过目标不变核移动；达到最大阶段数但 β<1 会报错。 | 权重/logZ 顺序相符。任何运行失败应计入协议结果，不能只汇总成功子集。 |
| 同文件 `:442–461,519–560` | 总计时包含 CPU 参考构造、初始化、GPU 同步、输出及统计；各类 factor/reference 调用单独记录。 | 计时范围较完整。基线也需采用相同范围；factor_calls 不能独自替代 CPU 搜索、混合密度与输出成本。 |
| `tests/test_tail_bridge_smc_20260921.py:109–223,226–254,366–405` | 已写真实密度比较、q/MH 详细平衡、完整符号翻转比率、协方差缩放、增量权重望远镜、解析高斯证据及数值积分测试。 | 已有针对性覆盖；本次只读测试，没有重跑，也未查看对应新执行回执。 |
| 同测试 `:257–328` | 保留从真实后验初始化的五步不变性检查；增加两种 scale 的双峰完整 sampler、八维真模型输入隔离及矩精度检查。 | 双峰/八维覆盖实际存在，不能把早期版本的缺项沿用为当前缺陷。有限案例仍不保证所有重要模式的恢复速度。 |
| 同测试 `:332–363` | 四个随机种子对一个小异质问题比较矩、W1 和证据比。 | 这是具体回归测试；四次重复和固定宽容差不足以证明 adaptive logZ 无偏性或通用性能。 |

### 5.1 易被其他接口引入的泄漏

现有 `SensorProblem.log_density` 在 `nonseparable_sensor_model_20260921.py:197–198` 调用精确 composition normalizer；`log_sensor_posterior` 的 `:200–225` 路径也调用枚举证据。即使相关常数在归一化采样中抵消，参考构造或 sampler 使用这些接口仍会取得原本属于 evaluator 的指数枚举信息，并扭曲成本。当前新实现避开了这些接口，应继续保持。

允许从已知因子精度计算 Λ。禁止从 `exact_posterior()`、保存的 `reference.npz` / `true_reference.npz`、真实参数、评价网格或确认集最优 W1 中挑选中心、分量数或温度。目标高度优化本身可以使用全部因子；这不构成真值泄漏，但必须计费。理论上可取得精确混合与实际 sampler 是否取得它是两个不同事实。

### 5.2 基线核对：保留时间顺序，最终冻结版本为准

**A，初审快照。** 当时读取的 sampler 哈希为 `6303fd76…`，新框架尚无全局符号翻转，而旧 `nonseparable_sensor_sampler_20260921.py:130–166` 含该操作。因此初审记录了 mutation 不可直接对齐的风险。这个观察只适用于该早期快照。

**C，冻结版本复核。** 作者提供冻结提交 `b7aa4f0ee8d2d80eecbcf9967b820e62597e2d76` 后，本次重读实际文件。`tail_bridge_smc_20260921.py:424` 已默认 `sign_flip=True`，`:488–493` 每阶段执行并计费；`run_tail_bridge_study_20260921.py:59–64` 调用时没有关闭它。因此三个 adaptive-prior 基线与十二个新候选均执行翻转。另九个 fixed-stage 基线在 runner `:66–69` 调用旧 `annealed_smc`，该函数 `:156–160` 同样执行翻转。冻结协议 `:7–9` 与这条调用链相符：**12 个 SMC 基线全部共享 sign flip，缺少翻转不再是当前风险。** 两个 diffusion 配置另列为机制诊断，不属于这 12 个 SMC 基线。

一般 q 桥的对称翻转 \(y=-x\) 使用的正确 log 接受比是

\[
(1-\beta)[\log q(-x)-\log q(x)]
+\beta[\log\widetilde p(-x)-\log\widetilde p(x)].
\]

冻结实现 `tail_bridge_smc_20260921.py:379–388` 计算 \(\Delta\log q+\beta\Delta R\)，与上式完全相同。β=1 时化为完整 posterior log density 差，β=0 时化为 q 的 log density 差。标准先验对称时，它与旧基线的 β likelihood 差一致。逐粒局部 MH 和翻转的顺序组合保持桥目标；无需追加权重修正。

**选择与优越性范围。** runner `:97–130` 检查完整开发矩阵，按平均时间选择精度达标的基线，再要求候选同时对该基线及 `smc_k64_s0.15` 满足 mean W1 +0.002、每个 stratum +0.004、mean time ratio<0.8。协议 `:16` 的确认主张要求对两个比较对象分别通过配对单侧 95% 精度及时间界；20 个共享数据种子是聚类单位，维数/噪声组合和 sampler repeats 留在簇内。任何时间比及 bootstrap 比值均应使用平均总时间之比。开发挑中的胜者尚不能称为确认获益。

冻结文本中的 +0.004 明确出现在开发筛选；确认段只明确要求报告 stratum，并没有写出 stratum +0.004 的确认均值或置信界门槛。按当前文本只能如此解释，不应在观察结果后宣称已经预声明了更强的分层确认保证。本审查不修改协议，也不要求更改已启动的冻结运行。确认统计实现与执行结果本次均未检查。

原初审建议的同框架 prior control 已纳入冻结协议，旧 64-stage 强基线也被保留。比较仍需要区分“冻结整体算法达到收益”和“尾部几何单一因素导致收益”；同参考 direct IS 的建议属于初始机制研究提案，未纳入当前 26 配置，不作为要求本轮扩展的新任务。全局符号翻转能连接正负镜像，仍不能保证探索一般多于两个模式的结构。

当前结果账本 `docs/PAPER_LEVEL_RESULTS_20260921.md:49–66` 记录旧 sensor prior-SMC 平均投影 W1 为 0.007149，旧 Gaussian-path pilot 的最佳候选仍未通过联合筛选。其 `:60` 排除了有并发负载的旧耗时，`:72–75` 已记录相应审计完成。这些是本次读到的现有记录，未在本次重算；它们没有包含本新方向的获益证据。新静态 bridge 结果不能改写这些路径方法的负面诊断。

## 6. 原始文献与可主张的新颖性

以下链接均为本次实际阅读的作者论文或作者托管全文。文献事实与本报告的推论分开列出。

| 原始来源 | 实际相关内容 | 对本项目主张的影响 |
|---|---|---|
| [Del Moral, Doucet, Jasra (2006), Sequential Monte Carlo samplers](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf), §2.3.1、§3 | 给出静态分布序列的 SMC、几何退火、不变移动核及归一化常数估计。 | 从可计算 q 到目标的退火、校正及重采样属于既有框架。 |
| [Cotter, Roberts, Stuart, White (2013), MCMC Methods for Functions](https://eprints.maths.manchester.ac.uk/2215/1/1202.0709v3.pdf), §4.2、§6 | Gaussian-reference pCN 与参考测度校正的 MH 结构。 | pCN 的 Gaussian invariance、残差接受率没有一般性首创空间。 |
| [Zhou, Johansen, Aston (2016), Toward Automatic Model Comparison](https://api.repository.cam.ac.uk/server/api/core/bitstreams/c53d7c7a-8376-4ec8-bedb-71a9d08ea2b4/content), §3.3.2–3.4 | CESS 控制退火，并区分自适应设计与标准 normalizer 无偏性。 | ESS/CESS 温度选择应作为既有方法引用；新的方差理论须单独证明。 |
| [Lu, Jia, Meng (2025), Sequential Monte Carlo with Gaussian Mixture Distributions for Infinite-Dimensional Statistical Inverse Problems, v1](https://arxiv.org/html/2503.16028v1), §2.3–2.4、§5.3 | 明确研究 SMC-pCN-GM；另有用混合近似代替 mutation 的 SMC-GM，并讨论自适应温度。 | 已经直接覆盖“Gaussian mixture + pCN + SMC”这个组合方向。本文核应逐式比较，不能宣称首先结合三者。 |
| [He, Owen (2014), Optimal mixture weights in multiple importance sampling](https://arxiv.org/html/1411.3954v1), §3.2、§4、§5.3 | 完整混合分母、多 proposal 的选择及构造成本核算。 | 模式混合参考和完整 mixture denominator 本身属于已有重要性采样设计。 |

最邻近的 Lu 等 v1 需要准确区分：其 §2.3 的混合 proposal 及完整 MH 比率与这里的 responsibility-选择核不同；§2.4 另讨论接受全部的近似 mutation。当前代码保留精确 MH，不能继承其近似版本的计算开销，也不应把该文的收敛陈述直接拿来证明本算法。本文确认的是直接的先行研究关系，没有断言两者逐项相同。

本项目可能保留的具体贡献是：从独立因子可读的共同二次尾部构造参考，给出可检查的矩充分条件，将 proposal 的尾部有效性与有限预算模式覆盖分开，并在只访问已训练因子的任务上完成公平、可复核的评估。这里的 Gaussian 积分、线性余项指数矩以及 latent-component 可逆核构造都属于标准数学工具。当前证据不足以把它们合称为新的通用采样原理，也不足以主张解决了 resampling variance。

若后续只有当前解析 sensor 的正面数字，可支持的层级仍是结构化合成问题上的实现与机制结果。当前 prior 权重本来有界这一事实，使“高阶矩修复”尤其不适合作为该实验的性能解释。

## 7. 一个可以直接形成小协议的 learned-factor 实验

本节保留初始研究任务要求的小实验提案，尚未执行，也不属于已经冻结并启动的 sensor study。最终冻结复核没有要求主线程并行开展本节任务。

### 7.1 科学问题和现有模型的差别

建议只做一个设计：**五个独立训练的标量 posterior 模型，沿不同非正交投影组合成二维后验。** 核心问题是：在 sampler 仅得到冻结模型输出、共同先验和观测 context 时，尾部参考是否以相当的 learned-target 精度降低总成本或改善模式覆盖？

现有 `run_dual_20260921.py:72–78` 在一个问题内用同一检查点预测所有因子；五个训练种子是重复实验维度。建议的新设计把五个不同训练检查点分别指定给五个因子，确实改变因子模型独立训练的结构。`learned_sbi.py:18–32` 的头提供可精确评价的两分量密度，共享标量方差 \(0.02\leq v_g\leq0.98\)；`:35–47` 的独立模拟流和 NLL 训练可复用。每个头对其 context 只运行一次；后续因子密度计算全部使用冻结输出，所有基线享有相同访问能力。

### 7.2 数据、目标与解析检查

取 \(d=2,G=5\)，\(\theta\sim\mathcal N(0,I_2)\)，固定

\[
a_g=(\cos t_g,\sin t_g)^T,\quad t_g=(g+1/4)\pi/5,\quad
\sigma_g=0.55+0.10g,\quad o_g=0.40+0.15g,
\quad g=0,\ldots,4.
\]

给定 θ，五个观测条件独立：\(y_g=a_g^T\theta+s_go_g+\sigma_g\epsilon_g\)，\(s_g\in\{-1,1\}\) 等概率，\(\epsilon_g\sim\mathcal N(0,1)\)。这些 σ、o 落在现有训练范围内。令第 g 个独立检查点只处理 \(c_g=(y_g,\sigma_g,o_g)\)，输出标量密度

\[
\widehat f_g(z\mid c_g)=\sum_{k=1}^2\omega_{gk}\mathcal N(z;\mu_{gk},v_g).
\]

将该头提升为二维因子

\[
F_g(\theta)=\varphi_2(\theta)
\frac{\widehat f_g(a_g^T\theta\mid c_g)}{\varphi_1(a_g^T\theta)}.
\]

单位投影确保 \(F_g\) 已归一化：沿投影方向用该标量后验，正交方向保留标准高斯。组合目标是

\[
\widetilde p_{\rm learned}(\theta)
=\frac{\prod_gF_g(\theta)}{\varphi_2(\theta)^{G-1}}
=\varphi_2(\theta)\prod_g
\frac{\widehat f_g(a_g^T\theta\mid c_g)}{\varphi_1(a_g^T\theta)}.
\]

这个先验修正不可遗漏。对于精确单因子后验且观测条件独立的生成过程，它就是联合后验的未归一化形式；对于学习头，它定义了应当评价的 learned target。其已知尾部精度为

\[
\Lambda=I_2+\sum_g(v_g^{-1}-1)a_ga_g^T\succ0.
\]

各因子二维分量协方差为 \(I_2+(v_g-1)a_ga_g^T\)；不同非正交方向通常不对易，组合也无法由逐坐标 filter 精确表示。当前 sensor 的 sign-EM 均值更新不能直接用于这组 additive-offset learned heads；所需 adapter 应由实现者针对上述目标写出正确密度及梯度，通用 reference/SMC 部分可以复用。

评价者可以枚举 \(2^5=32\) 个分量。对 \(s=(s_0,\ldots,s_4)\in\{1,2\}^5\)，令

\[
b_s=\sum_g\frac{\mu_{g,s_g}}{v_g}a_g,\qquad
d_s=\sum_g\left[\log\omega_{g,s_g}-\tfrac12\log v_g
-\frac{\mu_{g,s_g}^2}{2v_g}\right].
\]

分量共同协方差是 \(\Lambda^{-1}\)，均值是 \(\Lambda^{-1}b_s\)，未归一化质量是

\[
\exp\{d_s+\tfrac12b_s^T\Lambda^{-1}b_s-\tfrac12\log\det\Lambda\}.
\]

先独立检查枚举密度与直接 prior-corrected product 在固定点的比例常数；再以二维数值积分检查 Z 和矩。精确一维投影 CDF 可由这 32 个分量直接得到。评价参考必须在 sampler 之外构造，禁止把其分量均值拿来当 q 的搜索初值。再用 `learned_sbi.exact_parameters` 只在 evaluator 侧构造 true reference，分别记录对 learned target 的采样误差以及 learned/true 之间的模型误差。

### 7.3 固定的小规模执行规格

建议主线程把下列设置正式冻结后执行；本次没有启动该实验，也没有宣称其结果。

- **模型与数据：** 使用已有、严格重载并验明哈希的 `training_0/final.pt` 至 `training_4/final.pt`，按 g 固定分配。四个开发数据种子和十二个全新确认数据种子；建议预留 1600–1603、1700–1711，协议落盘前核对未被当前任务使用。每个数据种子三次独立 sampler 重复。五个检查点共同构成一个固定 factor bank，不能充作五个训练重复。
- **参考构造：** 从先验生成 16 个固定种子的起点，对 learned log target 每点至多 50 次优化迭代；使用解析 mixture gradient。按 Λ-Mahalanobis 距离合并接近中心，最多保留四个，协方差统一为 \(\Lambda^{-1}\)，权重由目标高度正则化后归一化。明确固定去重阈值、最小混合权重和优化停止规则，保存终止梯度范数。单高斯参考使用同一搜索结果的最高目标中心。所有准备成本计入该方法。
- **五个固定比较项：** A 为 prior-reference SMC；B 为单尾部高斯 SMC；C 为四分量上限的责任概率 pCN-SMC；D 为 C 的同一 q 直接 IS；E 为 C 加固定 10% 概率全局独立 q-MH proposal。E 的全局分量按 α 抽取。参考缺少四个不同中心时保留实际数量，禁止用评价者真值补齐。
- **公共 SMC 设置：** N=4096，ESS 温度目标 0.8，全部中间阶段 systematic resampling、最后保留权重，每阶段三次 mutation，pCN scale=0.5，最多 128 阶段；失败单元原样保留。A/B/C/E 的温度、重采样及计时范围一致。D 仅从其 q 独立抽 N 个点并用完整混合分母校正。本实验共 240 个单元，其中确认部分为 180 个；这是待批准的执行规模，不是已完成计数。
- **评价与决定：** 预先固定 32 个投影方向，用精确 learned CDF 评价带权 projected W1；报告 learned posterior 均值误差、log composition-Z 误差、模式区域概率误差、阶段数、最大权重、接受率、总时间和准备时间。模式区域由独立 evaluator 固定，q-responsibility mass 单独作为描述诊断，不能充作真模式质量。以 C 对 A 为唯一主要比较，建议联合标准为 W1 差的一侧 95% 上界 ≤0.002，平均总时间比的一侧 95% 上界 <0.8；B/D/E 解释机制，未经额外冻结不可把最优者临时晋升为主要比较。
- **统计单位：** 先在同一数据问题内平均三次 sampler 重复，再按十二个数据种子配对 bootstrap 10000 次，固定 seed=20260930。所有方法和差值共享重抽索引；时间比始终为同一索引下的平均时间之比。十二个数据问题、一个固定 factor bank 只能给出有限规模的条件证据，不支持总体校准或跨训练分布的普遍 95% 覆盖主张。

它比现有 sensor 更贴近“获取独立已训练因子，无法调用联合似然”的接口条件，但生成器在数学上仍有解析似然。应准确称为 checkpoint-only 的受控合成任务，不宜称为真实不可解似然应用。所有方法都能评价 learned product，所以普通 SMC 同样可用。另一个必须保留的负面可能性是：现有 \(v_g<1\) 加上投影覆盖二维空间，使 prior-reference 的静态权重也有良好尾部；新方法仍需通过实际成本与误差证明用途。

## 8. 冻结前最有价值的五组测试

1. **核密度与实际抽样的一致性。** 保留当前详细平衡测试，补充 proposal 的条件矩或分布与独立 SciPy 计算比较，确保 `propose` 和 `transition_log_prob` 真正描述同一个核。用两个远隔高斯检查尺度 1 的核依赖 x，同时检查独立 q-refresh 不依赖 x；不把后者混入前者的证明。
2. **normalizer 与权重生命周期。** 现有 helper 已有望远镜及常数移位检查。针对最终 weighted mutation，一个精确的回归规格是：在同一真实 Gaussian sensor、同一 seed、q=prior、固定温度 `[0,1]` 下比较零次移动与若干次最终 MH，要求初次生成的非均匀最终权重及 logZ 一致、粒子位置允许变化，再用解析目标检查带权统计。不能要求有限样本移动前后统计值逐项相等。若未来支持跳过中间重采样，先加入非均匀旧权重的 CESS 与权重保留测试。这些是建议，当前未执行新增测试。
3. **实际模式恢复。** 在远隔双峰和至少一个多于四个重要模式的真实小 sensor 上，从明显失衡的粒子模式质量出发，记录目标模式概率和跨区域迁移。当前从真后验初始化的不变性测试继续保留，但不能替代这项。ESS、高接受率与末端责任概率均不能单独判定通过。
4. **输入隔离和目标身份。** 仅用四字段 mapping 运行 sensor sampler；对同一输入改变或完全不提供 truth，验证参考与粒子输出一致。learned adapter 单独验证先验修正和 32 分量评价密度；sampler 的输入包不含 oracle 文件。使用真实模型与实际参数，避免模拟成功的替身。
5. **同目标、同预算的成本复核。** 至少保存 prior/single/mixture/direct-IS 的 q 参数、温度、阶段记录、权重、种子、源码与输入哈希，独立重算 R、每步 logZ 和输出指标；明确所有 setup、模式优化、完整混合密度、同步及输出计时。成功判定采用实际参考误差及平均总时间比较，保留所有失败和负面结果。

## 9. 本次已检查与未检查

已检查：上述数学契约和反例；早期 sampler/测试全文及冻结 sampler、相关新增测试；最终 `run_tail_bridge_study_20260921.py` 与 `docs/TAIL_BRIDGE_STUDY_PROTOCOL_20260921.md`；sensor 目标、旧 SMC 基线及带权 metrics；`learned_sbi.py`、`run_dual_20260921.py` 的训练头与资产构造；先验修正的参考构造；论文实验、协议及 paper-level 结果说明；第 6 节原始来源。research 技能的文献工作由本次 AI 直接完成，未委派额外任务。math-paper-writing 的要求落实为假设、目标、矩阶、常数依赖和实验状态的显式区分。

未检查：本新 sampler 的 CPU/CUDA 实际测试回执、任何新 GPU 结果、完整运行原始文件、GitHub 发布状态；未重新执行已有 paper-level 审计，未严格重载训练检查点，未执行第 7 节实验。本文不把测试源码中的断言视作已经发生的测试通过。

阶段 A 的 sampler/测试 SHA-256 分别为 `6303fd76e2e3408bfdebd95ef7fe2870b54980e66cfe8a3194e24ffbb9a8db90`、`e82d5d692f60326d62ee4e4431a76c431056d42ce4db09e7b261ecd8d9089bd6`，仅用于保留初审时间顺序。

下表为阶段 C 最终复核实际读取的 SHA-256。作者标定冻结提交为 `b7aa4f0ee8d2d80eecbcf9967b820e62597e2d76`；表中的哈希是本次读取时独立计算的文件标识，不额外证明服务器执行档案、冻结时间或提交映射。GPU 已启动仅为作者提供的状态，尚无本次核实的新性能结果。

| 文件 | SHA-256 |
|---|---|
| `tail_bridge_smc_20260921.py` | `fa423f2875ca4f42d97de6998d6147aaaf934e4147b22c1b98b41a0065607a9f` |
| `tests/test_tail_bridge_smc_20260921.py` | `17b891dbd2a50ec4a3dbe88ce287626a592f32f9b90920f3e4e8f877794fd490` |
| `run_tail_bridge_study_20260921.py` | `3279171a0929d84e7f7319b372f6952506833f4ccfb5423ba1eb8171b9902888` |
| `docs/TAIL_BRIDGE_STUDY_PROTOCOL_20260921.md` | `7f29b925321eb927eb3230fa425caa9f30b88da9edf189192400f185629de731` |
| `learned_sbi.py` | `b07b39aab748921c79d62a60b6588fc8925389275e1ad428a12788605ab667ac` |
| `nonseparable_sensor_model_20260921.py` | `cdabcb627dc9f22a69e3e1c72c7e6c52da700e5b1d60d05126ddafea1da120ad` |
| `docs/PAPER_LEVEL_RESULTS_20260921.md` | `b6dcb6fc7eab77c14cbe0b035a2499b6867b5dbc157884e862a02254ff2e3a2b` |

优先级最高的条件是保留精确目标/参考密度与权重生命周期、阻断 oracle 输入、补足跨模式检查，并用已有强基线和总成本判定价值。当前实现具有正确的数学构件；是否产生值得论文主张的增量，取决于这组可证伪验证。
