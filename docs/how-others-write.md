# 别人怎么写这个方向：两篇该精读的稿

PDF 在 `papers/`。这两篇不是随便找的引用，是和你们这篇**同一套叙事**的两个模板：

| | 论文 | 看它学什么 |
| --- | --- | --- |
| 1 | Müller et al., **Transformers Can Do Bayesian Inference**, ICLR 2022 | 「PFN = 先验 $P$ 下的后验」怎么立住：解析 GP 对照、损失就是 KL、先机制后应用 |
| 2 | Lv et al., **When Tabular Foundation Models Meet Strategic Tabular Data**, ICML 2026 | 「先验错配 + 推理期改上下文、不重训」怎么写成投稿稿：先证失败，再给构造，再问三个实验问题 |

Xie et al. ICLR 2022（ICL = 隐式贝叶斯）是 LLM 孪生篇，结构很像 Müller，不必先看。TabPFN Nature 2025 是产品论文，写法帮不上你们。

---

## 1. Müller 2022：机制半篇该长这样

[PDF](../papers/muller2022-transformers-can-do-bayesian-inference.pdf) · [arXiv:2112.10510](https://arxiv.org/abs/2112.10510) · OpenReview `KSugKcbNf9`

### 摘要公式（可直接套）

> Currently it is hard to … We present PFNs. The only requirement is … We restate posterior approximation as a supervised problem: sample a task from the prior, mask a label, predict it from the rest. At test time, one forward pass approximates Bayesian inference. We demonstrate that PFNs near-perfectly mimic GPs, give 200× speedups, and work on GP / BNN / small tables / few-shot images.

四段：痛点 → 对象 → 训练/测试在干什么 → 证据清单。没有「我们提出一个新损失 / 新架构很复杂」。复杂度藏在「唯一要求是能从先验里采样」。

### 正文骨架（9 页里真正干活的部分）

1. **Introduction + Figure 1**  
   Figure 1 就是整篇论文：左采样先验数据集，中训一个 $q_\theta(y\mid x,D)$，右一次前向 ≈ $p(y\mid x,D)$。引言最后三条贡献：架构能逼近 PPD、GP/BNN 比 MCMC 快两个数量级、真实小表上有用。
2. **Background**（不叫 Related Work）  
   把 meta-learning、Neural Process、MCMC/VI、simulation-based inference 一笔收掉，马上落到 PPD 公式  
   $p(y\mid x,D)=\int p(y\mid x,t)\,p(t\mid D)\,dt$。
3. **Insight 而不是定理墙**  
   他们最重要的理论一句话就说完：Prior-Data NLL **等于** 真实 PPD 和 $q_\theta$ 的交叉熵（再写成期望 KL）。可实现时 $q_{\theta^\star}=p$。  
   评审读得下去，是因为这句话直接解释后面所有实验的纵轴。
4. **§5 Posterior approximation studies 先于真实表**  
   顺序是死的：  
   - 固定超参 GP（PPD 有解析解）→ 图上看均值和 95% 区间几乎重合；  
   - 有超先验的 GP（解析解没了）→ 比 MLE-II / NUTS 更接近真 PPD，且快 200×；  
   - BNN（完全不可积）→ 同样的纵轴 Prior-Data NLL。  
   **有解析解的玩具先验，是这篇能投 ICLR 的原因，不是附录。**
5. **§6 真实表是「这个先验居然有用」**  
   20 个 OpenML 子集、30 个训练点、AUC + **ECE** + 墙钟。BNN 先验的 PFN 校准比 XGB / 真 BNN 都好。他们把校准当成「它真的在做贝叶斯」的现场证据，不是附属指标。
6. **结尾的 future work 几乎是在点你们的题**  
   (i) 现在能逼近以前逼近不了的先验；(ii) 架构；(iii) 放大；(iv) amortized SBI。他们没写 $Q\neq P$ 时算的是什么。

### 该偷的句子级习惯

- 先画「和解析解比」，再上真实表。你们的 `eval_gp_prior_tilt.py` / A·B·C 检查点就是他们的 §5。
- 主指标跟理论对象一致：他们是 Prior-Data NLL（= KL）；你们该是「PFN 相对解析混合后验的相关/斜率」，校准实验才是 log loss / ECE，准确率必须当对照。
- 贡献写成能力，不写成系统名堆砌。
- ECE 放主表，不放附录。

### 不该偷的

- 他们的 claim 是「能逼近 $P$ 下的 PPD」。你们的 claim 是「$Q\neq P$ 时它仍然在逼近 $P$，并且会用自己仅有的 $z$ 乱解释」。不要写成 Müller 的续作「我们逼近得更好」。
- 真实表 30 个点 + 自造先验能赢 2019 年的 GBDT，2026 年赢不了 TabPFN-2.5 / TabArena。应用段必须换。

---

## 2. SPN 2026：方法半篇该长这样

[PDF](../papers/lv2026-spn-strategic-prior-alignment.pdf) · [arXiv:2605.19662](https://arxiv.org/abs/2605.19662) · ICML 2026

这是目前和你们**方法叙事最近**的一篇：预训练先验对不上部署分布 → 系统偏差 → **不改权重、构造上下文** → 推理期对齐。必须精读，也必须在引言里主动收编，否则评审会当成 SPN 的弱仿写。

### 摘要公式（和 Müller 同一套，只是对象从「逼近 $P$」换成「$P$ 错了怎么修」）

> Tabular foundation models based on PFNs generalize on diverse tasks, **but they are typically designed for** non-strategic settings. **We show that** strategic manipulation creates a mismatch between the non-strategic prior and the post-manipulation prior, **which leads to systematic prediction bias**. **To address this, we propose** SPN, an **inference-time** framework that **constructs strategic in-context examples** and aligns PFN predictions **without retraining**. Experiments on real-world and synthetic data show …

句式就是：SOTA 背景 → but 边界 → we show 失败机制 → we propose 推理期构造 → 实验。你们的摘要如果还在写「一种新的校准算法」，就输在这一句上。

### 正文骨架

1. **引言以一个问题收束**  
   *Are PFN-style tabular foundation models capable of generalizing on strategic tabular data?*  
   然后三条贡献：刻画边界 / 证明偏差；提出推理期框架；实验既修战略设定、又不把非战略设定搞坏。
2. **Related work 两刀切开**  
   表格学习（树、TabPFN/TabICL/TabDPT）vs 战略分类。他们的位置是「这两摊还没人接」。你们对应的两刀是：PFN/表格基础模型 vs 校准 / ICL 先验倾斜。SPN 必须出现在第一刀里，写成「已知、可命名的错配 + 成对的、对 $f(x)$ 有信息的上下文」。
3. **§4 先写失败，再写方法**  
   定义非战略 / 战略两个元先验 → 支撑不交的集合 $\mathcal{S}_0$ → 未覆盖质量 $\delta$ → TV 下界 → Proposition：只要 $\delta>0$，任何估计器的误差 $\liminf \mathcal{E}_n \ge c\delta$（样本量 $n\to\infty$ 也消不掉）。  
   理论并不深，但**版面上**是「不可约偏差」，后面的方法才有存在理由。你们对应的位置是：错误归因 + 纠缠，不是 TV。
4. **§5 方法前先有一个 case study**  
   微调和 ICL 的时间/样本成本。用来挡住「你为什么不微调」——这是 2026 年 PFN 方法文的标准防火墙。你们对应的防火墙已经有了：温度缩放、训练点拟合 $T$、真实上下文注噪。
5. **构造写进 Definition + Algorithm 1**  
   战略上下文 = $\{(x_i,y_i),\,(b_f(x_i),y_i)\}$。命题：注意力更新 ≈ 一步梯度战略更新。算法输入输出写死。  
   你们对应：far-anchor 的 $(x_q \perp D_\mathrm{aux}\mid z)$、$\eta$ 网格、伪代码。
6. **实验是三个问题，不是一张大表**  
   - 战略操纵如何毁掉非战略 PFN，SPN 能不能救；  
   - 非战略基准上会不会变差；  
   - ICL 条数和操纵机制变了还稳不稳。  
   主图按数据集分面，主表是 AUC，消融是上下文条数 10–50。

### 和你们的差别（写进 Related Work 的那一段该怎么写）

| | SPN | 你们 |
| --- | --- | --- |
| 错配 | 有名字：战略操纵 $b_f$ | 泛泛的 $Q\neq P$，先验里的 $z$ 对不上 |
| 上下文 | 成对、局部、**同一标签**、对 $f(x)$ 有信息 | 远置、任务无关、**只携带 $z$** |
| 失败模式 | 准确率 / 假阳性 | 校准（准确率故意不动） |
| 理论对象 | 元先验支撑的未覆盖质量 $\delta$ | $p(z\mid D)$ 的倾斜、纠缠、错误归因 |
| 对照 | 微调成本、XGBoost、TabPFN、TabDPT | 温度 / Platt / 注噪 / 额外 ID 行 |
| 会不会被说成同一篇 | 会，如果你们也只写「构造上下文修错配」 | 不会，如果第一句话就是「它不报错，它用自己仅有的隐变量重新解释数据」 |

SPN 的上下文是类 II（动局部似然）：$(b_f(x),y)$ 就在查询点附近，标签还是真的。你们的 far-anchor 是类 I。这是 related work 里最值钱的一句话。

### 该偷的

- 引言用一个是非题收束。
- 失败理论放在方法前面。
- 「不重训」要有一个显式对照（他们用微调，你们用温度）。
- 实验写成问题列表；必须报告「原来那个设定有没有被搞坏」。
- 构造过程有名字、有定义、有算法框。

### 不该偷的

- 他们的理论是支撑不交 ⇒ 不可识别，和你们「支撑内但拧错维」不是一回事。不要把 $\delta$ 那套 TV 搬过来充数。
- 他们主指标是 AUC。你们若把主指标写成准确率，方法半篇已经被温度缩放杀掉了。
- 他们假设知道 $b_f$。你们不能假设知道 $Q$。这是 claim 强弱的来源，别往「我们也假设一个操纵模型」上靠。

---

## 3. 九页稿如果按这两篇的写法来

对照 Müller 的 §5 + SPN 的 §4–6：

1. **Figure 1**（整页故事）  
   左：从 $P$ 采样、一次前向 = $p_P(y\mid x,D)$（Müller Fig.1）。  
   右：数据来自 $Q$ 时同一前向仍输出 $p_P$；远探针只改 $p(z\mid D)$（你们的机制图）。
2. **引言**  
   PFN 是 $P$ 的化身（Müller）→ 部署时 $Q\neq P$ 它不报错（SPN 的 but）→ 它用仅有的 $z$ 重新解释（你们独有）→ 代价在校准不在准确率 → 研究问题：*When the prior is wrong, what does a PFN actually compute, and which interventions can change that object?*
3. **§2 失败机制**（对应 SPN §4，内容换你们的）  
   解析 GP：$Q$ 在支撑外则堆边界；支撑内则忠实于自己的混合后验。  
   2×2 先验：缺失的轴 = 错误归因，不是无反应。  
   A144 vs C：组合数不是原因，纠缠才是。
4. **§3 干预**（对应 SPN §5）  
   四类对象（见 `theory-interventions.md`）。分类噪声混合 ≈ 温度，所以必须把温度放进主实验，不能藏附录。Algorithm 1：far-anchor。
5. **§4 实验问题**（对应 SPN §6 的三个问句）  
   Q1 机制是否跟着解析贝叶斯走？  
   Q2 分类上类 I 是不是类 III 的马甲？（已有答案：hold-out 温度赢；训练点拟合 $T$ 会炸；准确率被 ctx_noise 毁掉。）  
   Q3 没有合法校准集、模型已经过自信时，类 I 还剩什么？
6. **Related work**  
   PFN / TabPFN / TabICL；ICL as Bayes（Xie）；校准（温度、Platt）；**SPN 一段写清楚成对局部上下文 vs 远置全局 $z$**；LoCalPFN/kNN 作为「丢掉 $z$ 通道」的对偶。

---

## 4. 读这两篇时带着看的三页

精读不必从头到尾：

**Müller**：摘要、Figure 1、Insight 1（损失 = KL）、§5.1 固定 GP 的图、Table 1 的 ECE 行。

**SPN**：摘要、引言最后那个问题、§4.3 不可约偏差、Definition 5.1 + Algorithm 1、§6 开头三个问题、他们怎么评价非战略基准（Table 1 AUC）。

读完应能回答：他们的**一句话 claim**是什么、**失败**写在方法前还是后、**主图**在证明哪一件事、**主指标**和 claim 是否同一个对象。
