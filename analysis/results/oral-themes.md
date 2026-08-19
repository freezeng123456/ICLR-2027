# 顶会 Oral 论文的主题分布

> 由 `analysis/oral_themes.py` 生成。**放大倍数** = 该主题在 Oral 中的占比 ÷ 在**录用论文**中的占比。也就是回答：在够格录用的论文里，哪些主题更容易被抬成 Oral。这个量比 Oral 篇数有用得多——LLM 相关的 Oral 最多，但那只是因为基数最大。

> 对照池用录用论文而非全体投稿，是为了让 ICLR 和 NeurIPS 可比：NeurIPS 的公开数据基本只含录用论文（拒稿只有自愿公开的那部分）。

> 主题由标题、关键词、TL;DR 的正则匹配判定，一篇论文可命中多个主题，所以各主题占比之和大于 1。判定规则见脚本里的 `THEMES`。

## ICLR 2026（决定于 2025-12 ～ 2026-03）

Oral 224 篇；对照池（全部录用论文）5,358 篇，Oral 占录用的 4.2%。

| 主题 | Oral 篇数 | 占 Oral | 占录用 | 放大倍数 |
|---|---:|---:|---:|---:|
| 扩散语言模型 | 4 | 1.8% | 1.2% | **1.52×** ⚠ |
| 优化器与训练方法 | 33 | 14.7% | 10.8% | **1.37×** |
| 架构与序列建模 | 24 | 10.7% | 9.0% | 1.20× |
| Agent 与工具使用 | 21 | 9.4% | 8.1% | 1.15× |
| 安全 / 对齐 / 监督 | 29 | 12.9% | 11.6% | 1.12× |
| 理论 / 学习动力学 | 29 | 12.9% | 11.9% | 1.09× |
| 推理 / test-time compute | 33 | 14.7% | 13.7% | 1.08× |
| 扩散 / 流匹配生成 | 29 | 12.9% | 12.1% | 1.07× |
| 具身 / 机器人 | 15 | 6.7% | 6.3% | 1.06× |
| RL 与后训练 | 35 | 15.6% | 15.2% | 1.02× |
| 可解释性 | 12 | 5.4% | 5.4% | 0.99× |
| 效率 / 推理系统 | 19 | 8.5% | 9.0% | 0.94× |
| 视频与图像生成 | 18 | 8.0% | 8.6% | 0.93× |
| 评测与 benchmark | 20 | 8.9% | 9.6% | 0.93× |
| 数据 / 记忆 / 归因 | 11 | 4.9% | 5.4% | 0.92× |
| 图 / 时序 / 表格 | 13 | 5.8% | 6.8% | 0.85× |
| 多模态 / 视觉语言 | 28 | 12.5% | 15.1% | 0.83× |
| 科学与生物医学应用 | 12 | 5.4% | 6.8% | 0.79× |

⚠ = Oral 篇数少于 8，放大倍数的统计噪声很大，只能当线索不能当结论。

## NeurIPS 2025（决定于 2025-09，比 ICLR 2026 早约半年）

Oral 93 篇；对照池（全部录用论文）5,812 篇，Oral 占录用的 1.6%。

| 主题 | Oral 篇数 | 占 Oral | 占录用 | 放大倍数 |
|---|---:|---:|---:|---:|
| 扩散语言模型 | 1 | 1.1% | 0.5% | **2.08×** ⚠ |
| 架构与序列建模 | 15 | 16.1% | 9.3% | **1.74×** |
| 多模态 / 视觉语言 | 18 | 19.4% | 13.3% | **1.45×** |
| 数据 / 记忆 / 归因 | 7 | 7.5% | 5.7% | **1.32×** ⚠ |
| 图 / 时序 / 表格 | 9 | 9.7% | 7.9% | 1.22× |
| RL 与后训练 | 11 | 11.8% | 10.9% | 1.08× |
| 效率 / 推理系统 | 7 | 7.5% | 7.0% | 1.08× ⚠ |
| 理论 / 学习动力学 | 15 | 16.1% | 15.2% | 1.06× |
| 优化器与训练方法 | 10 | 10.8% | 11.2% | 0.96× |
| 安全 / 对齐 / 监督 | 8 | 8.6% | 10.3% | 0.83× |
| 视频与图像生成 | 7 | 7.5% | 9.1% | 0.83× ⚠ |
| 具身 / 机器人 | 4 | 4.3% | 5.5% | 0.79× ⚠ |
| 可解释性 | 3 | 3.2% | 4.1% | 0.78× ⚠ |
| 扩散 / 流匹配生成 | 7 | 7.5% | 10.8% | 0.70× ⚠ |
| 评测与 benchmark | 5 | 5.4% | 7.8% | 0.69× ⚠ |
| 科学与生物医学应用 | 4 | 4.3% | 7.0% | 0.61× ⚠ |
| 推理 / test-time compute | 4 | 4.3% | 9.6% | 0.45× ⚠ |
| Agent 与工具使用 | 1 | 1.1% | 5.2% | 0.21× ⚠ |

⚠ = Oral 篇数少于 8，放大倍数的统计噪声很大，只能当线索不能当结论。

### 两届之间的变化（同一口径）

| 主题 | NeurIPS 2025 | ICLR 2026 | 变化 |
|---|---:|---:|---|
| Agent 与工具使用 | 0.21× | 1.15× | ↑↑ +0.95 |
| 推理 / test-time compute | 0.45× | 1.08× | ↑↑ +0.63 |
| 优化器与训练方法 | 0.96× | 1.37× | ↑ +0.41 |
| 扩散 / 流匹配生成 | 0.70× | 1.07× | ↑ +0.37 |
| 安全 / 对齐 / 监督 | 0.83× | 1.12× | ↑ +0.29 |
| 具身 / 机器人 | 0.79× | 1.06× | ↑ +0.27 |
| 评测与 benchmark | 0.69× | 0.93× | ↑ +0.24 |
| 可解释性 | 0.78× | 0.99× | ↑ +0.22 |
| 科学与生物医学应用 | 0.61× | 0.79× | ↑ +0.18 |
| 视频与图像生成 | 0.83× | 0.93× | — +0.10 |
| 理论 / 学习动力学 | 1.06× | 1.09× | — +0.03 |
| RL 与后训练 | 1.08× | 1.02× | — -0.06 |
| 效率 / 推理系统 | 1.08× | 0.94× | — -0.14 |
| 图 / 时序 / 表格 | 1.22× | 0.85× | ↓ -0.37 |
| 数据 / 记忆 / 归因 | 1.32× | 0.92× | ↓ -0.40 |
| 架构与序列建模 | 1.74× | 1.20× | ↓↓ -0.55 |
| 扩散语言模型 | 2.08× | 1.52× | ↓↓ -0.56 |
| 多模态 / 视觉语言 | 1.45× | 0.83× | ↓↓ -0.63 |

---

### ICLR 2026 的 Oral 标题（按主题分组）

**推理 / test-time compute**（33 篇）

- Actions Speak Louder than Prompts: A Large-Scale Study of LLMs for Graph Inference
- DepthLM: Metric Depth from Vision Language Models
- EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning
- Energy-Based Transformers are Scalable Learners and Thinkers
- GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models
- Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments
- Generative Universal Verifier as Multimodal Meta-Reasoner
- In-Place Test-Time Training
- Is it Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effort
- LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts
- MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains
- MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science
- On the Reasoning Abilities of Masked Diffusion Language Models
- OpenThoughts: Data Recipes for Reasoning Models
- Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks
- Overthinking Reduction with Decoupled Rewards and Curriculum Data Scheduling
- Premise Selection for a Lean Hammer
- RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following in Large Reasoning Models with Preserved Thinking Format
- Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment
- Reasoning with Sampling: Your Base Model is Smarter Than You Think
- Reducing Belief Deviation in Reinforcement Learning for Active Reasoning
- Reliable Weak-to-Strong Monitoring of LLM Agents
- Scaling Atomistic Protein Binder Design with Generative Pretraining and Test-Time Compute
- Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People
- The Art of Scaling Reinforcement Learning Compute for LLMs
- The Coverage Principle: How Pre-Training Enables Post-Training
- ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models
- Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs
- UALM: Unified Audio Language Model for Understanding, Generation and Reasoning
- Verifying Chain-of-Thought Reasoning via Its Computational Graph
- Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning
- Vid-LLM: A Compact Video-based 3D Multimodal LLM with Reconstruction–Reasoning Synergy
- Visual symbolic mechanisms: Emergent symbol processing in Vision Language Models

**RL 与后训练**（35 篇）

- AgentGym-RL: An Open-Source Framework to Train LLM Agents for Long-Horizon Decision Making via Multi-Turn RL
- Differentiable Model Predictive Control on the GPU
- DiffusionNFT: Online Diffusion Reinforcement with Forward Process
- EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning
- Enhancing Generative Auto-bidding with Offline Reward Evaluation and Policy Search
- Exploratory Diffusion Model for Unsupervised Reinforcement Learning
- GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning
- Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments
- Half-order Fine-Tuning for Diffusion Model: A Recursive Likelihood Ratio Optimizer
- In-The-Flow Agentic System Optimization for Effective Planning and Tool Use
- Is it Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effort
- LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning
- LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts
- Mastering Sparse CUDA Generation through Pretrained Models and Deep Reinforcement Learning
- Mean Flow Policy with Instantaneous Velocity Constraint for One-step Action Generation
- MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent
- Multiplayer Nash Preference Optimization
- Omni-Reward: Towards Generalist Omni-Modal Reward Modeling with Free-Form Preferences
- OpenApps: Simulating Environment Variations to Measure UI Agent Reliability
- Optimistic Task Inference for Behavior Foundation Models
- P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling
- Q-RAG: Long Context Multi‑Step Retrieval via Value‑Based Embedder Training
- Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment
- Reducing Belief Deviation in Reinforcement Learning for Active Reasoning
- SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety
- Semi-Supervised Preference Optimization with Limited Feedback
- TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning
- TROLL: Trust Regions Improve Reinforcement Learning for Large Language Models
- The Art of Scaling Reinforcement Learning Compute for LLMs
- The Coverage Principle: How Pre-Training Enables Post-Training
- Token-Importance Guided Direct Preference Optimization
- Triple-BERT: Do We Really Need MARL for Order Dispatch on Ride-Sharing Platforms?
- What's In My Human Feedback? Learning Interpretable Descriptions of Preference Data
- Why DPO is a Misspecified Estimator and How to Fix It
- cadrille: Multi-modal CAD Reconstruction with Reinforcement Learning

**Agent 与工具使用**（21 篇）

- Agent Data Protocol
- AgentGym-RL: An Open-Source Framework to Train LLM Agents for Long-Horizon Decision Making via Multi-Turn RL
- AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite
- Compositional Diffusion with Guided search for Long-Horizon Planning
- CyberGym: Evaluating AI Agents' Real-World Cybersecurity Capabilities at Scale
- GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning
- Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments
- Huxley-G\"odel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine
- In-The-Flow Agentic System Optimization for Effective Planning and Tool Use
- MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains
- MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science
- MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent
- OpenApps: Simulating Environment Variations to Measure UI Agent Reliability
- RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments
- Reducing Belief Deviation in Reinforcement Learning for Active Reasoning
- Reliable Weak-to-Strong Monitoring of LLM Agents
- ScaleCUA: Scaling Open-Source Computer Use Agents with Cross-Platform Data
- Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People
- SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents
- Speculative Actions: A Lossless Framework for Faster AI Agents
- To Infinity and Beyond: Tool-Use Unlocks Length Generalization in State Space Models

**效率 / 推理系统**（19 篇）

- A Representer Theorem for Hawkes Processes via Penalized Least Squares Minimization
- A Scalable Distributed Framework for Multimodal GigaVoxel Image Registration
- Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss
- DTO-KD: Dynamic Trade-off Optimization for Effective Knowledge Distillation
- FlashWorld: High-quality 3D Scene Generation within Seconds
- GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models
- Global Resolution: Optimal Multi-Draft Speculative Sampling via Convex Optimization
- Mastering Sparse CUDA Generation through Pretrained Models and Deep Reinforcement Learning
- Mixture-of-Experts Can Surpass Dense LLMs Under Strictly Equal Resource
- Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks
- Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding
- Overthinking Reduction with Decoupled Rewards and Curriculum Data Scheduling
- Probabilistic Kernel Function for Fast Angle Testing
- Speculative Actions: A Lossless Framework for Faster AI Agents
- Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation
- ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models
- TileLang: Bridge Programmability and Performance in Modern Neural Kernels
- Universal Inverse Distillation for Matching Models with Real-Data Supervision (No GANs)
- Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention

**扩散 / 流匹配生成**（29 篇）

- Compositional Diffusion with Guided search for Long-Horizon Planning
- DCFold: Efficient Protein Structure Generation with Single Forward Pass
- Diffusion Language Model Knows the Answer Before It Decodes
- DiffusionNFT: Online Diffusion Reinforcement with Forward Process
- Exploratory Diffusion Model for Unsupervised Reinforcement Learning
- FALCON: Few-step Accurate Likelihoods for Continuous Flows
- FlashWorld: High-quality 3D Scene Generation within Seconds
- GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models
- Half-order Fine-Tuning for Diffusion Model: A Recursive Likelihood Ratio Optimizer
- Improving Diffusion Models for Class-imbalanced Training Data via Capacity Manipulation
- Latent Fourier Transform
- Let Features Decide Their Own Solvers: Hybrid Feature Caching for Diffusion Transformers
- Monocular Normal Estimation via Shading Sequence Estimation
- NextStep-1: Toward Autoregressive Image Generation with Continuous Tokens at Scale
- On the Reasoning Abilities of Masked Diffusion Language Models
- Pareto-Conditioned Diffusion Models for Offline Multi-Objective Optimization
- Partition Generative Modeling: Masked Modeling Without Masks
- Planner Aware Path Learning in Diffusion Language Models Training
- Quotient-Space Diffusion Model
- SAFETY-GUIDED FLOW (SGF): A UNIFIED FRAMEWORK FOR NEGATIVE GUIDANCE IN SAFE GENERATION
- SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer
- Scaling Atomistic Protein Binder Design with Generative Pretraining and Test-Time Compute
- Spherical Watermark: Encryption-Free, Lossless Watermarking for Diffusion Models
- Structured Flow Autoencoders: Learning Structured Probabilistic Representations with Flow Matching
- TRACE: Your Diffusion Model is Secretly an Instance Edge Detector
- Text-to-3D by Stitching a Multi-view Reconstruction Network to a Video Generator
- The Spacetime of Diffusion Models: An Information Geometry Perspective
- Universal Inverse Distillation for Matching Models with Real-Data Supervision (No GANs)
- VibeVoice: Expressive Podcast Generation with Next-Token Diffusion

**扩散语言模型**（4 篇）

- Diffusion Language Model Knows the Answer Before It Decodes
- On the Reasoning Abilities of Masked Diffusion Language Models
- Partition Generative Modeling: Masked Modeling Without Masks
- Planner Aware Path Learning in Diffusion Language Models Training

**视频与图像生成**（18 篇）

- $PhyWorldBench$: A Comprehensive Evaluation of Physical Realism in Text-to-Video Models
- FlashWorld: High-quality 3D Scene Generation within Seconds
- Generative Human Geometry Distribution
- Latent Particle World Models: Self-supervised Object-centric Stochastic Dynamics Modeling
- Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation
- Monocular Normal Estimation via Shading Sequence Estimation
- MotionStream: Real-Time Video Generation with Interactive Motion Controls
- Neon: Negative Extrapolation From Self-Training Improves Image Generation
- NextStep-1: Toward Autoregressive Image Generation with Continuous Tokens at Scale
- On the Generalization Capacities of MLLMs for Spatial Intelligence
- Radiometrically Consistent Gaussian Surfels for Inverse Rendering
- SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer
- Stable Video Infinity: Infinite-Length Video Generation with Error Recycling
- Text-to-3D by Stitching a Multi-view Reconstruction Network to a Video Generator
- True Self-Supervised Novel View Synthesis is Transferable
- Vid-LLM: A Compact Video-based 3D Multimodal LLM with Reconstruction–Reasoning Synergy
- WoW!: World Models in a Closed-Loop World
- cadrille: Multi-modal CAD Reconstruction with Reinforcement Learning

**架构与序列建模**（24 篇）

- Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series
- Efficient Resource-Constrained Training of Transformers via Subspace Optimization
- Energy-Based Transformers are Scalable Learners and Thinkers
- FlashRNN: Unlocking Parallel Training of Nonlinear RNNs for Large Language Models
- From Markov to Laplace: How Mamba In-Context Learns Markov Chains
- From movement to cognitive maps: recurrent neural networks reveal how locomotor development shapes hippocampal spatial coding
- HATSolver: Learning Gröbner Bases with Hierarchical Attention Transformers
- How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Interpretability
- InfoTok: Adaptive Discrete Video Tokenizer via Information-Theoretic Compression
- Latent Speech-Text Transformer
- Let Features Decide Their Own Solvers: Hybrid Feature Caching for Diffusion Transformers
- Mamba-3: Improved Sequence Modeling using State Space Principles
- MrRoPE: Mixed-radix Rotary Position Embedding
- On the Reasoning Abilities of Masked Diffusion Language Models
- Pinet: Optimizing hard-constrained neural networks with orthogonal projection layers
- Quantitative Bounds for Length Generalization in Transformers
- Rodrigues Network for Learning Robot Actions
- SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer
- Softmax Transformers are Turing-Complete
- TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning
- TRACE: Your Diffusion Model is Secretly an Instance Edge Detector
- To Infinity and Beyond: Tool-Use Unlocks Length Generalization in State Space Models
- Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention
- mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules

**理论 / 学习动力学**（29 篇）

- $p\textrm{-less}$ Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding
- A Representer Theorem for Hawkes Processes via Penalized Least Squares Minimization
- Cross-Domain Lossy Compression via Rate- and Classification-Constrained Optimal Transport
- Differentiable Model Predictive Control on the GPU
- Difficult Examples Hurt Unsupervised Contrastive Learning: A Theoretical Perspective
- FRABench and UFEval: Unified Fine-grained Evaluation with Task and Aspect Generalization
- Fast Escape, Slow Convergence: Learning Dynamics of Phase Retrieval under Power-Law Data
- Gaussian certified unlearning in high dimensions: A hypothesis testing approach
- Global Resolution: Optimal Multi-Draft Speculative Sampling via Convex Optimization
- Huxley-G\"odel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine
- Hyperparameter Trajectory Inference with Conditional Lagrangian Optimal Transport
- InfoTok: Adaptive Discrete Video Tokenizer via Information-Theoretic Compression
- Navigating the Latent Space Dynamics of Neural Models
- Non-Asymptotic Analysis of (Sticky) Track-and-Stop
- On the Generalization Capacities of MLLMs for Spatial Intelligence
- On the Reasoning Abilities of Masked Diffusion Language Models
- On the Wasserstein Geodesic Principal Component Analysis of probability measures
- Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks
- Pre-training under infinite compute
- Premise Selection for a Lean Hammer
- Quantitative Bounds for Length Generalization in Transformers
- Scaling Laws and Spectra of Shallow Neural Networks in the Feature Learning Regime
- Softmax Transformers are Turing-Complete
- Steering the Herd: A Framework for LLM-based Control of Social Learning
- The Coverage Principle: How Pre-Training Enables Post-Training
- The Polar Express: Optimal Matrix Sign Methods and their Application to the Muon Algorithm
- To Infinity and Beyond: Tool-Use Unlocks Length Generalization in State Space Models
- Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning
- VibeVoice: Expressive Podcast Generation with Next-Token Diffusion

**优化器与训练方法**（33 篇）

- A Scalable Distributed Framework for Multimodal GigaVoxel Image Registration
- AutoEP: LLMs-Driven Automation of Hyperparameter Evolution for Metaheuristic Algorithms
- Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training
- Conformal Robustness Control: A New Strategy for Robust Decision
- DTO-KD: Dynamic Trade-off Optimization for Effective Knowledge Distillation
- Differentiable Model Predictive Control on the GPU
- Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces
- Efficient Resource-Constrained Training of Transformers via Subspace Optimization
- FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability–Plasticity Tradeoff
- Fast Escape, Slow Convergence: Learning Dynamics of Phase Retrieval under Power-Law Data
- Fast training of accurate physics-informed neural networks without gradient descent
- GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning
- Global Resolution: Optimal Multi-Draft Speculative Sampling via Convex Optimization
- Half-order Fine-Tuning for Diffusion Model: A Recursive Likelihood Ratio Optimizer
- How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Interpretability
- How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining
- In-The-Flow Agentic System Optimization for Effective Planning and Tool Use
- Learning to Segment for Vehicle Routing Problems
- Multiplayer Nash Preference Optimization
- Non-Convex Federated Optimization under Cost-Aware Client Selection
- Overparametrization bends the landscape: BBP transitions at initialization in simple Neural Networks
- Overthinking Reduction with Decoupled Rewards and Curriculum Data Scheduling
- Pareto-Conditioned Diffusion Models for Offline Multi-Objective Optimization
- Pinet: Optimizing hard-constrained neural networks with orthogonal projection layers
- RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following in Large Reasoning Models with Preserved Thinking Format
- SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety
- Semi-Supervised Preference Optimization with Limited Feedback
- Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation
- Task-free Adaptive Meta Black-box Optimization
- The Polar Express: Optimal Matrix Sign Methods and their Application to the Muon Algorithm
- Token-Importance Guided Direct Preference Optimization
- WSM: Decay-Free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training
- Why DPO is a Misspecified Estimator and How to Fix It

**可解释性**（12 篇）

- Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces
- EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning
- Exploratory Causal Inference in SAEnce
- How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Interpretability
- Navigating the Latent Space Dynamics of Neural Models
- On the Reasoning Abilities of Masked Diffusion Language Models
- Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability
- Temporal superposition and feature geometry of RNNs under memory demands
- The Shape of Adversarial Influence: Characterizing LLM Latent Spaces with Persistent Homology
- Verifying Chain-of-Thought Reasoning via Its Computational Graph
- Visual symbolic mechanisms: Emergent symbol processing in Vision Language Models
- What's In My Human Feedback? Learning Interpretable Descriptions of Preference Data

**评测与 benchmark**（20 篇）

- $PhyWorldBench$: A Comprehensive Evaluation of Physical Realism in Text-to-Video Models
- AdAEM: An Adaptively and Automated Extensible Evaluation Method of LLMs' Value Difference
- AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite
- BIRD-INTERACT: Re-imagining Text-to-SQL Evaluation via Lens of Dynamic Interactions
- Benchmarking Empirical Privacy Protection for Adaptations of Large Language Models
- CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of Large Language Models in Mental Health Question Answering
- CyberGym: Evaluating AI Agents' Real-World Cybersecurity Capabilities at Scale
- EditBench: Evaluating LLM Abilities to Perform Real-World Instructed Code Edits
- Enhancing Generative Auto-bidding with Offline Reward Evaluation and Policy Search
- FRABench and UFEval: Unified Fine-grained Evaluation with Task and Aspect Generalization
- Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments
- How Reliable is Language Model Micro-Benchmarking?
- MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains
- RealPDEBench: A Benchmark for Complex Physical Systems with Real-World Data
- RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments
- SWINGARENA: Adversarial Programming Arena for Long-context GitHub Issue Solving
- SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents
- TTSDS2: Resources and Benchmark for Evaluating Human-Quality Text to Speech Systems
- Train-before-Test Harmonizes Language Model Rankings
- WebDevJudge: Evaluating (M)LLMs as Critiques for Web Development Quality

**安全 / 对齐 / 监督**（29 篇）

- AdAEM: An Adaptively and Automated Extensible Evaluation Method of LLMs' Value Difference
- Benchmarking Empirical Privacy Protection for Adaptations of Large Language Models
- Beyond Prompt-Induced Lies: Investigating LLM Deception on Benign Prompts
- CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of Large Language Models in Mental Health Question Answering
- Differentially Private Domain Discovery
- EigenBench: A Comparative Behavioral Measure of Value Alignment
- Every Language Model Has a Forgery-Resistant Signature
- GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models
- Gaussian certified unlearning in high dimensions: A hypothesis testing approach
- Hubble: a Model Suite to Advance the Study of LLM Memorization
- Invisible Safety Threat: Malicious Finetuning for LLM via Steganography
- Is it Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effort
- LLM DNA: Tracing Model Evolution via Functional Representations
- LLM Fingerprinting via Semantically Conditioned Watermarks
- Latent Speech-Text Transformer
- Learning with Dual-level Noisy Correspondence for Multi-modal Entity Alignment
- Modality-free Graph In-context Alignment
- Omni-Reward: Towards Generalist Omni-Modal Reward Modeling with Free-Form Preferences
- P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling
- PetaGAIL++: Utility Optimized Private Trajectory Generation with Imitation Learning
- RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments
- Reliable Weak-to-Strong Monitoring of LLM Agents
- SAFETY-GUIDED FLOW (SGF): A UNIFIED FRAMEWORK FOR NEGATIVE GUIDANCE IN SAFE GENERATION
- SWINGARENA: Adversarial Programming Arena for Long-context GitHub Issue Solving
- SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety
- Spherical Watermark: Encryption-Free, Lossless Watermarking for Diffusion Models
- The Shape of Adversarial Influence: Characterizing LLM Latent Spaces with Persistent Homology
- Token-Importance Guided Direct Preference Optimization
- Watch your steps: Dormant Adversarial Behaviors that Activate upon LLM Finetuning

**多模态 / 视觉语言**（28 篇）

- A Scalable Distributed Framework for Multimodal GigaVoxel Image Registration
- BioX-Bridge: Model Bridging for Unsupervised Cross-Modal Knowledge Transfer across Biosignals
- Depth Anything 3: Recovering the Visual Space from Any Views
- DepthLM: Metric Depth from Vision Language Models
- EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning
- Extending Sequence Length is Not All You Need: Effective Integration of Multimodal Signals for Gene Expression Prediction
- FlashVID: Efficient Video Large Language Models via Training-free Tree-based Spatiotemporal Token Merging
- Generative Universal Verifier as Multimodal Meta-Reasoner
- Latent Fourier Transform
- Latent Speech-Text Transformer
- Learning to See Before Seeing: Demystifying LLM Visual Priors from Language Pre-training
- MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains
- MetaEmbed: Scaling Multimodal Retrieval at Test-Time with Flexible Late Interaction
- MomaGraph: State-Aware Unified Scene Graphs with Vision-Language Models for Embodied Task Planning
- Multimodal Aligned Semantic Knowledge for Unpaired Image-text Matching
- On the Generalization Capacities of MLLMs for Spatial Intelligence
- Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment
- Seeing Through the Brain: New Insights from Decoding Visual Stimuli with fMRI
- TTSDS2: Resources and Benchmark for Evaluating Human-Quality Text to Speech Systems
- Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs
- UALM: Unified Audio Language Model for Understanding, Generation and Reasoning
- VibeVoice: Expressive Podcast Generation with Next-Token Diffusion
- Vid-LLM: A Compact Video-based 3D Multimodal LLM with Reconstruction–Reasoning Synergy
- Visual Planning: Let's Think Only with Images
- Visual symbolic mechanisms: Emergent symbol processing in Vision Language Models
- WAVE: Learning Unified & Versatile Audio-Visual Embeddings with Multimodal LLM
- cadrille: Multi-modal CAD Reconstruction with Reinforcement Learning
- mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules

**具身 / 机器人**（15 篇）

- Conformal Robustness Control: A New Strategy for Robust Decision
- Differentiable Model Predictive Control on the GPU
- Exploratory Causal Inference in SAEnce
- From movement to cognitive maps: recurrent neural networks reveal how locomotor development shapes hippocampal spatial coding
- Improving Diffusion Models for Class-imbalanced Training Data via Capacity Manipulation
- Latent Fourier Transform
- Latent Particle World Models: Self-supervised Object-centric Stochastic Dynamics Modeling
- MomaGraph: State-Aware Unified Scene Graphs with Vision-Language Models for Embodied Task Planning
- MotionStream: Real-Time Video Generation with Interactive Motion Controls
- On the Generalization Capacities of MLLMs for Spatial Intelligence
- PetaGAIL++: Utility Optimized Private Trajectory Generation with Imitation Learning
- Rodrigues Network for Learning Robot Actions
- SAFETY-GUIDED FLOW (SGF): A UNIFIED FRAMEWORK FOR NEGATIVE GUIDANCE IN SAFE GENERATION
- Steering the Herd: A Framework for LLM-based Control of Social Learning
- WoW!: World Models in a Closed-Loop World

**科学与生物医学应用**（12 篇）

- $PhyWorldBench$: A Comprehensive Evaluation of Physical Realism in Text-to-Video Models
- DCFold: Efficient Protein Structure Generation with Single Forward Pass
- Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series
- Exploring Synthesizable Chemical Space with Iterative Pathway Refinements
- Fast training of accurate physics-informed neural networks without gradient descent
- Information Shapes Koopman Representation
- MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science
- Planner Aware Path Learning in Diffusion Language Models Training
- RealPDEBench: A Benchmark for Complex Physical Systems with Real-World Data
- Scaling Atomistic Protein Binder Design with Generative Pretraining and Test-Time Compute
- Seeing Through the Brain: New Insights from Decoding Visual Stimuli with fMRI
- mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules

**数据 / 记忆 / 归因**（11 篇）

- CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data
- Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training
- FRABench and UFEval: Unified Fine-grained Evaluation with Task and Aspect Generalization
- High-dimensional Analysis of Synthetic Data Selection
- Hubble: a Model Suite to Advance the Study of LLM Memorization
- Navigating the Latent Space Dynamics of Neural Models
- OpenThoughts: Data Recipes for Reasoning Models
- Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks
- TabStruct: Measuring Structural Fidelity of Tabular Data
- Verifying Chain-of-Thought Reasoning via Its Computational Graph
- What's In My Human Feedback? Learning Interpretable Descriptions of Preference Data

**图 / 时序 / 表格**（13 篇）

- CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data
- Causal Structure Learning in Hawkes Processes with Complex Latent Confounder Networks
- Compactness and Consistency: A Conjoint Framework for Deep Graph Clustering
- Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series
- Exchangeability of GNN Representations with Applications to Graph Retrieval
- FlashVID: Efficient Video Large Language Models via Training-free Tree-based Spatiotemporal Token Merging
- Multi-Domain Transferable Graph Gluing for Building Graph Foundation Models
- One for Two: A Unified Framework for Imbalanced Graph Classification via Dynamic Balanced Prototype
- SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents
- TabStruct: Measuring Structural Fidelity of Tabular Data
- Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability
- Temporal superposition and feature geometry of RNNs under memory demands
- Uncover Underlying Correspondence for Robust Multi-view Clustering

**未归类**（19 篇）

- Addressing divergent representations from causal interventions on neural networks
- AnyUp: Universal Feature Upsampling
- Characterizing the Discrete Geometry of ReLU Networks
- Distributional Equivalence in Linear Non-Gaussian Latent-Variable Cyclic Causal Models: Characterization and Learning
- EditVerse: Unifying Image and Video Editing and Generation with In-Context Learning
- Generating metamers of human scene understanding
- Hallucination Begins Where Saliency Drops
- InfoNCE Induces Gaussian Distribution
- Instilling an Active Mind in Avatars via Cognitive Simulation
- Intrinsic Entropy of Context Length Scaling in LLMs
- It's All Just Vectorization: einx, a Universal Notation for Tensor Operations
- LLMs Get Lost In Multi-Turn Conversation
- On The Surprising Effectiveness of a Single Global Merging in Decentralized Learning
- Online Learning and Equilibrium Computation with Ranking Feedback
- Plug-and-Play Compositionality for Boosting Continual Learning with Foundation Models
- RefineStat: Efficient Exploration for Probabilistic Program Synthesis
- Revela: Dense Retriever Learning via Language Modeling
- Sequences of Logits Reveal the Low Rank Structure of Language Models
- WAFT: Warping-Alone Field Transforms for Optical Flow

---

### NeurIPS 2025 的 Oral 标题（按主题分组）

**推理 / test-time compute**（4 篇）

- Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?
- Envisioning Beyond the Pixels: Benchmarking Reasoning-Informed Visual Editing
- NOVA: A Benchmark for Rare Anomaly Localization and Clinical Reasoning in Brain MRI
- SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing

**RL 与后训练**（11 篇）

- 1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities
- A Clean Slate for Offline Reinforcement Learning
- A Snapshot of Influence: A Local Data Attribution Framework for Online Reinforcement Learning
- Adaptive Surrogate Gradients for Sequential Reinforcement Learning in Spiking Neural Networks
- Breaking the Performance Ceiling in Reinforcement Learning requires Inference Strategies
- Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?
- Does Stochastic Gradient really succeed for bandits?
- EvoLM: In Search of Lost Language Model Training Dynamics
- PRIMT: Preference-based Reinforcement Learning with Multimodal Feedback and Trajectory Synthesis from Foundation Models
- QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training
- State Entropy Regularization for Robust Reinforcement Learning

**Agent 与工具使用**（1 篇）

- WebGen-Bench: Evaluating LLMs on Generating Interactive and Functional Websites from Scratch

**效率 / 推理系统**（7 篇）

- Advancing Expert Specialization for Better MoE
- ElasticMM: Efficient Multimodal LLMs Serving with Elastic Multimodal Parallelism
- HyperET: Efficient Training in Hyperbolic Space for Multi-modal Large Language Models
- KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction
- On Linear Mode Connectivity of Mixture-of-Experts Architectures
- The emergence of sparse attention: impact of data distribution and benefits of repetition
- Tighter CMI-Based Generalization Bounds via Stochastic Projection and Quantization

**扩散 / 流匹配生成**（7 篇）

- Adjoint Schrödinger Bridge Sampler
- Deep Compositional Phase Diffusion for Long Motion Sequence Generation
- Exploring Diffusion Transformer Designs via Grafting
- Large Language Diffusion Models
- On the Closed-Form of Flow Matching: Generalization Does Not Arise from Target Stochasticity
- Representation Entanglement for Generation: Training Diffusion Transformers Is Much Easier Than You Think
- Why Diffusion Models Don’t Memorize: The Role of Implicit Dynamical Regularization in Training

**扩散语言模型**（1 篇）

- Large Language Diffusion Models

**视频与图像生成**（7 篇）

- BEDLAM2.0: Synthetic humans and cameras in motion
- Dynam3D: Dynamic Layered 3D Tokens Empower VLM for Vision-and-Language Navigation
- Envisioning Beyond the Pixels: Benchmarking Reasoning-Informed Visual Editing
- InfinityStar: Uniﬁed Spacetime AutoRegressive Modeling for Visual Generation
- Interactive Cross-modal Learning for Text-3D Scene Retrieval
- PlayerOne: Egocentric World Simulator
- SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing

**架构与序列建模**（15 篇）

- A multiscale analysis of mean-field transformers in the moderate interaction regime
- Auto-Compressing Networks
- Boosting Knowledge Utilization in Multimodal Large Language Models via Adaptive Logits Fusion and Attention Reallocation
- Exploring Diffusion Transformer Designs via Grafting
- From Condensation to Rank Collapse: A Two-Stage Analysis of Transformer Training Dynamics
- Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free
- Generalized Linear Mode Connectivity for Transformers
- High-dimensional neuronal activity from low-dimensional latent dynamics: a solvable model
- In Search of Adam’s Secret Sauce
- Learning Linear Attention in Polynomial Time
- Learning long range dependencies through time reversal symmetry breaking
- On Linear Mode Connectivity of Mixture-of-Experts Architectures
- Representation Entanglement for Generation: Training Diffusion Transformers Is Much Easier Than You Think
- Task-Optimized Convolutional Recurrent Networks Align with Tactile Processing in the Rodent Brain
- The emergence of sparse attention: impact of data distribution and benefits of repetition

**理论 / 学习动力学**（15 篇）

- Agnostic Active Learning Is Always Better Than Passive Learning
- An Optimized Franz-Parisi Criterion and its Equivalence with SQ Lower Bounds
- Depth-Bounds for Neural Networks via the Braid Arrangement
- Dynamical Decoupling of Generalization and Overfitting in Large Two-Layer Networks
- Generalized Gradient Norm Clipping & Non-Euclidean $(L_0,L_1)$-Smoothness
- Generalized Linear Mode Connectivity for Transformers
- GnnXemplar: Exemplars to Explanations - Natural Language Rules for Global GNN Interpretability
- Improved Regret Bounds for Gaussian Process Upper Confidence Bound in Bayesian Optimization
- Learning Linear Attention in Polynomial Time
- On the Closed-Form of Flow Matching: Generalization Does Not Arise from Target Stochasticity
- Optimal Mistake Bounds for Transductive Online Learning
- SAGE: A Unified Framework for Generalizable Object State Recognition with State-Action Graph Embedding
- Spectral Perturbation Bounds for Low-Rank Approximation with Applications to Privacy
- Superposition Yields Robust Neural Scaling
- Tighter CMI-Based Generalization Bounds via Stochastic Projection and Quantization

**优化器与训练方法**（10 篇）

- Adaptive Surrogate Gradients for Sequential Reinforcement Learning in Spiking Neural Networks
- Advancing Expert Specialization for Better MoE
- Analog In-memory Training on General Non-ideal Resistive Elements: The Impact of Response Functions
- Does Stochastic Gradient really succeed for bandits?
- From Condensation to Rank Collapse: A Two-Stage Analysis of Transformer Training Dynamics
- Generalized Gradient Norm Clipping & Non-Euclidean $(L_0,L_1)$-Smoothness
- Improved Regret Bounds for Gaussian Process Upper Confidence Bound in Bayesian Optimization
- In Search of Adam’s Secret Sauce
- Learning (Approximately) Equivariant Networks via Constrained Optimization
- PhySense: Sensor Placement Optimization for Accurate Physics Sensing

**可解释性**（3 篇）

- A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders
- GnnXemplar: Exemplars to Explanations - Natural Language Rules for Global GNN Interpretability
- ImageNet-trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression

**评测与 benchmark**（5 篇）

- A Clean Slate for Offline Reinforcement Learning
- BEDLAM2.0: Synthetic humans and cameras in motion
- Envisioning Beyond the Pixels: Benchmarking Reasoning-Informed Visual Editing
- NOVA: A Benchmark for Rare Anomaly Localization and Clinical Reasoning in Brain MRI
- WebGen-Bench: Evaluating LLMs on Generating Interactive and Functional Websites from Scratch

**安全 / 对齐 / 监督**（8 篇）

- Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)
- Dynamical Low-Rank Compression of Neural Networks with Robustness under Adversarial Attacks
- Machine Unlearning Doesn't Do What You Think: Lessons for Generative AI Policy and Research
- More effort is needed to protect pedestrian privacy in the era of AI
- Position: Bridge the Gaps between Machine Unlearning and AI Regulation
- Spectral Perturbation Bounds for Low-Rank Approximation with Applications to Privacy
- Stop the Nonconsensual Use of Nude Images in Research
- Task-Optimized Convolutional Recurrent Networks Align with Tactile Processing in the Rodent Brain

**多模态 / 视觉语言**（18 篇）

- Boosting Knowledge Utilization in Multimodal Large Language Models via Adaptive Logits Fusion and Attention Reallocation
- ControlFusion: A Controllable Image Fusion Network with Language-Vision Degradation Prompts
- CoralVQA: A Large-Scale Visual Question Answering Dataset for Coral Reef Image Understanding
- Dynam3D: Dynamic Layered 3D Tokens Empower VLM for Vision-and-Language Navigation
- ElasticMM: Efficient Multimodal LLMs Serving with Elastic Multimodal Parallelism
- Envisioning Beyond the Pixels: Benchmarking Reasoning-Informed Visual Editing
- High-dimensional neuronal activity from low-dimensional latent dynamics: a solvable model
- InfinityStar: Uniﬁed Spacetime AutoRegressive Modeling for Visual Generation
- Interactive Cross-modal Learning for Text-3D Scene Retrieval
- MokA: Multimodal Low-Rank Adaptation for MLLMs
- NOVA: A Benchmark for Rare Anomaly Localization and Clinical Reasoning in Brain MRI
- OpenHOI: Open-World Hand-Object Interaction Synthesis with Multimodal Large Language Model
- PRIMT: Preference-based Reinforcement Learning with Multimodal Feedback and Trajectory Synthesis from Foundation Models
- Perception Encoder: The best visual embeddings are not at the output of the network
- QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training
- Rethinking Joint Maximum Mean Discrepancy for Visual Domain Adaptation
- Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion
- SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing

**具身 / 机器人**（4 篇）

- Adaptive Surrogate Gradients for Sequential Reinforcement Learning in Spiking Neural Networks
- ControlFusion: A Controllable Image Fusion Network with Language-Vision Degradation Prompts
- ImageNet-trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression
- PRIMT: Preference-based Reinforcement Learning with Multimodal Feedback and Trajectory Synthesis from Foundation Models

**科学与生物医学应用**（4 篇）

- Learning long range dependencies through time reversal symmetry breaking
- NOVA: A Benchmark for Rare Anomaly Localization and Clinical Reasoning in Brain MRI
- PhySense: Sensor Placement Optimization for Accurate Physics Sensing
- QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training

**数据 / 记忆 / 归因**（7 篇）

- A Snapshot of Influence: A Local Data Attribution Framework for Online Reinforcement Learning
- CoralVQA: A Large-Scale Visual Question Answering Dataset for Coral Reef Image Understanding
- NOVA: A Benchmark for Rare Anomaly Localization and Clinical Reasoning in Brain MRI
- On the Closed-Form of Flow Matching: Generalization Does Not Arise from Target Stochasticity
- Stop the Nonconsensual Use of Nude Images in Research
- Tighter CMI-Based Generalization Bounds via Stochastic Projection and Quantization
- Why Diffusion Models Don’t Memorize: The Role of Implicit Dynamical Regularization in Training

**图 / 时序 / 表格**（9 篇）

- A multiscale analysis of mean-field transformers in the moderate interaction regime
- Adaptive Surrogate Gradients for Sequential Reinforcement Learning in Spiking Neural Networks
- Discovering Opinion Intervals from Conflicts in Signed Graphs
- FuXi-Ocean: A Global Ocean Forecasting System with Sub-Daily Resolution
- GnnXemplar: Exemplars to Explanations - Natural Language Rules for Global GNN Interpretability
- Learning long range dependencies through time reversal symmetry breaking
- QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training
- Task-Optimized Convolutional Recurrent Networks Align with Tactile Processing in the Rodent Brain
- TransferTraj: A Vehicle Trajectory Learning Model for Region and Task Transferability

**未归类**（16 篇）

- Class-wise Balancing Data Replay for Federated Class-Incremental Learning
- High-Dimensional Calibration from Swap Regret
- Identifiability of Deep Polynomial Neural Networks
- Learning to Learn with Contrastive Meta-Objective
- MaxSup: Overcoming Representation Collapse in Label Smoothing
- Mean Flows for One-step Generative Modeling
- Memory Mosaics at scale
- NeurIPS should lead scientific consensus on AI policy
- OrthoLoC: UAV 6-DoF Localization and Calibration Using Orthographic Geodata
- Pan-LUT: Efficient Pan-sharpening via Learnable Look-Up Tables
- Position: If Innovation in AI systematically Violates Fundamental Rights, Is It Innovation at All?
- Position: Machine Learning Conferences Should Establish a "Refutations and Critiques" Track
- RAG4GFM: Bridging Knowledge Gaps in Graph Foundation Models through Graph Retrieval Augmented Generation
- Real-Time Hyper-Personalized Generative AI Should Be Regulated to Prevent the Rise of "Digital Heroin"
- Stop DDoS Attacking the Research Community with AI-Generated Survey Papers
- Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference

