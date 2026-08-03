# 会议数据分析结果

> 本文件由 `analysis/analyze.py` 自动生成，数据来自 Paper Copilot 的 OpenReview 抓取（https://github.com/papercopilot/paperlists）。

## ICLR 2025 → 2026 总体盘子

| 指标 | ICLR 2025 | ICLR 2026 | 变化 |
|---|---:|---:|---:|
| 总投稿 | 11,677 | 19,814 | +69.7% |
| 接收 | 3,703 (31.7%) | 5,358 (27.0%) | +44.7% |
| 拒稿 | 4,911 (42.1%) | 8,453 (42.7%) | +72.1% |
| 作者撤稿 | 2,993 (25.6%) | 5,098 (25.7%) | +70.3% |
| desk reject | 70 (0.6%) | 905 (4.6%) | +1192.9% |
| 接收率（投稿口径） | 31.9% | 28.3% | -3.6pp |
| 接收率（决策口径） | 43.0% | 38.8% | -4.2pp |

## 评分分布与录用分数线

ICLR 2026 换了打分刻度：2025 年是 {1,3,5,6,8,10}，2026 年改成等距的 {0,2,4,6,8,10}。所以两年的分数不可直接比较，要比的是**分数落在哪个百分位**。

**ICLR 2025** 单条评分分布（共 46,751 条评审）：1分 2.2%，3分 24.7%，5分 28.0%，6分 31.4%，8分 13.3%，10分 0.4%

**ICLR 2026** 单条评分分布（共 75,859 条评审）：0分 1.7%，2分 26.2%，4分 39.2%，6分 26.0%，8分 6.6%，10分 0.2%

平均分对应的录用概率（仅统计拿到 accept/reject 决定的论文）：

| 平均分 | ICLR 2025 论文数 | 2025 录用率 | ICLR 2026 论文数 | 2026 录用率 |
|---:|---:|---:|---:|---:|
| 2.5 | 67 | 0.0% | 521 | 0.4% |
| 3.0 | 211 | 0.0% | 739 | 1.2% |
| 3.5 | 338 | 0.0% | 1,428 | 3.4% |
| 4.0 | 746 | 1.2% | 2,265 | 11.9% |
| 4.5 | 574 | 1.4% | 2,794 | 28.6% |
| 5.0 | 1,462 | 8.1% | 2,123 | 53.9% |
| 5.5 | 1,007 | 19.8% | 1,825 | 78.8% |
| 6.0 | 2,044 | 65.9% | 1,005 | 93.2% |
| 6.5 | 887 | 93.6% | 465 | 97.0% |
| 7.0 | 758 | 94.9% | 171 | 98.8% |
| 7.5 | 337 | 98.5% | 72 | 100.0% |
| 8.0 | 123 | 100.0% | 17 | 100.0% |
| 8.5 | 10 | 100.0% | 1 | 100.0% |
| 9.0 | 7 | 100.0% | — | — |
| 9.5 | 1 | 100.0% | — | — |
| 10.0 | 1 | 100.0% | — | — |

## ICLR 2026 分领域接收率（按作者自选的 primary area）

| Primary area | 投稿 | 接收 | 接收率(投稿口径) | 接收率(决策口径) | 撤稿率 | Oral | 每千投稿 Oral |
|---|---:|---:|---:|---:|---:|---:|---:|
| learning theory | 516 | 190 | 38.6% | 46.3% | 15.9% | 12 | 23.3 |
| foundation or frontier models, including llms | 2,646 | 831 | 32.9% | 43.7% | 23.7% | 40 | 15.1 |
| probabilistic methods (bayesian methods, variational inference, sampling, uq, etc.) | 380 | 116 | 32.1% | 37.9% | 14.5% | 5 | 13.2 |
| applications to neuroscience & cognitive science | 383 | 114 | 31.7% | 40.4% | 20.4% | 4 | 10.4 |
| generative models | 1,680 | 496 | 31.0% | 43.2% | 27.0% | 25 | 14.9 |
| applications to robotics, autonomy, planning | 598 | 178 | 30.7% | 42.3% | 26.4% | 5 | 8.4 |
| reinforcement learning | 1,065 | 308 | 30.2% | 38.7% | 21.0% | 8 | 7.5 |
| datasets and benchmarks | 1,593 | 443 | 29.6% | 41.1% | 26.4% | 18 | 11.3 |
| applications to computer vision, audio, language, and other modalities | 2,711 | 737 | 28.1% | 43.1% | 33.8% | 21 | 7.7 |
| applications to physical sciences (physics, chemistry, biology, etc.) | 861 | 221 | 26.8% | 37.1% | 26.5% | 11 | 12.8 |
| optimization | 739 | 191 | 26.8% | 34.3% | 21.1% | 10 | 13.5 |
| interpretability and explainable ai | 788 | 199 | 26.5% | 35.7% | 24.4% | 9 | 11.4 |
| alignment, fairness, safety, privacy, and societal considerations | 1,696 | 423 | 26.3% | 36.4% | 26.4% | 21 | 12.4 |
| infrastructure, software libraries, hardware, systems, etc. | 178 | 43 | 25.4% | 32.3% | 20.2% | 3 | 16.9 |
| causal reasoning | 195 | 47 | 25.4% | 33.1% | 22.1% | 2 | 10.3 |
| unsupervised, self-supervised, semi-supervised, and supervised representation learning | 1,163 | 265 | 23.7% | 32.6% | 26.0% | 12 | 10.3 |
| learning on time series and dynamical systems | 467 | 101 | 22.7% | 30.1% | 23.3% | 1 | 2.1 |
| learning on graphs and other geometries & topologies | 534 | 113 | 22.6% | 30.6% | 24.5% | 5 | 9.4 |
| other topics in machine learning (i.e., none of the above) | 825 | 177 | 22.4% | 31.8% | 28.1% | 7 | 8.5 |
| neurosymbolic & hybrid ai systems (physics-informed, logic & formal reasoning, etc.) | 226 | 47 | 22.3% | 29.6% | 23.0% | 1 | 4.4 |
| transfer learning, meta learning, and lifelong learning | 570 | 118 | 22.0% | 31.2% | 27.7% | 4 | 7.0 |

## 关键词同比：拥挤度 vs 回报率

ICLR 总投稿从 11,677 涨到 19,814（+69.7%）。所以看绝对投稿量增长会误导——一个方向必须涨过 70% 才算真正在**抢占份额**，涨得比这慢的其实在被稀释。

| 关键词 | 2025 投稿 | 2026 投稿 | 份额变化 | 2025 接收率 | 2026 接收率 | 接收率变化 | 2026 撤稿率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| llm reasoning | 10 | 113 | +565.9% | n/a | 47.5% | n/a | 27.4% |
| multi-agent systems | 16 | 110 | +305.2% | n/a | 26.5% | n/a | 28.2% |
| llm agents | 49 | 186 | +123.7% | 45.0% | 39.4% | -5.6pp | 24.2% |
| reasoning | 125 | 455 | +114.5% | 48.5% | 45.7% | -2.8pp | 23.5% |
| agents | 84 | 265 | +85.9% | 56.1% | 44.7% | -11.4pp | 23.8% |
| vision-language models | 176 | 485 | +62.4% | 57.6% | 41.5% | -16.1pp | 29.7% |
| flow matching | 73 | 199 | +60.7% | 53.1% | 49.7% | -3.4pp | 13.1% |
| reinforcement learning | 506 | 1,379 | +60.6% | 41.3% | 42.8% | +1.6pp | 21.6% |
| multimodal llm | 159 | 421 | +56.0% | 47.7% | 49.1% | +1.4pp | 31.6% |
| video generation | 65 | 170 | +54.1% | 53.5% | 57.4% | +3.9pp | 30.0% |
| multimodal learning | 65 | 143 | +29.7% | 52.5% | 43.6% | -8.9pp | 30.1% |
| retrieval-augmented generation | 53 | 112 | +24.5% | 51.4% | 29.7% | -21.7pp | 33.9% |
| benchmark | 263 | 545 | +22.1% | 46.8% | 41.3% | -5.6pp | 25.0% |
| image generation | 70 | 145 | +22.1% | 54.3% | 46.5% | -7.8pp | 27.6% |
| uncertainty quantification | 71 | 147 | +22.0% | 47.4% | 37.3% | -10.1pp | 19.7% |
| code generation | 58 | 108 | +9.7% | 38.1% | 44.7% | +6.6pp | 25.0% |
| robotics | 55 | 102 | +9.3% | 48.8% | 44.3% | -4.5pp | 19.6% |
| safety | 83 | 153 | +8.6% | 34.3% | 46.2% | +11.8pp | 26.1% |
| llm(s) | 2,137 | 3,815 | +5.2% | 42.8% | 37.8% | -5.0pp | 25.1% |
| machine unlearning | 67 | 117 | +2.9% | 46.2% | 33.3% | -12.8pp | 29.1% |
| ai safety | 79 | 136 | +1.5% | 34.9% | 37.0% | +2.1pp | 21.3% |
| time series | 141 | 241 | +0.7% | 34.7% | 29.4% | -5.3pp | 24.1% |
| efficiency | 64 | 108 | -0.6% | 47.9% | 43.5% | -4.4pp | 15.7% |
| mechanistic interpretability | 84 | 141 | -1.1% | 43.8% | 40.0% | -3.8pp | 14.9% |
| quantization | 75 | 124 | -2.6% | 34.4% | 45.6% | +11.1pp | 23.4% |
| representation learning | 235 | 369 | -7.5% | 40.5% | 38.1% | -2.4pp | 22.5% |
| machine learning | 108 | 169 | -7.8% | 34.2% | 30.6% | -3.7pp | 27.2% |
| knowledge distillation | 91 | 141 | -8.7% | 39.7% | 38.9% | -0.8pp | 30.5% |
| continual learning | 133 | 203 | -10.0% | 34.4% | 39.7% | +5.3pp | 31.0% |
| diffusion | 113 | 171 | -10.8% | 46.2% | 38.1% | -8.2pp | 18.7% |
| evaluation | 124 | 185 | -12.1% | 46.6% | 38.3% | -8.3pp | 25.9% |
| multimodal | 76 | 113 | -12.4% | 55.6% | 38.2% | -17.3pp | 34.5% |
| interpretability | 223 | 328 | -13.3% | 44.5% | 40.2% | -4.3pp | 21.0% |
| federated learning | 168 | 240 | -15.8% | 29.5% | 25.6% | -3.8pp | 30.8% |
| foundation models | 87 | 122 | -17.4% | 34.3% | 42.5% | +8.2pp | 23.8% |
| dataset | 75 | 105 | -17.5% | 41.5% | 43.7% | +2.2pp | 28.6% |
| computer vision | 113 | 156 | -18.6% | 37.5% | 37.4% | -0.1pp | 30.1% |
| robustness | 134 | 181 | -20.4% | 39.8% | 34.4% | -5.4pp | 24.9% |
| contrastive learning | 117 | 158 | -20.4% | 34.9% | 30.0% | -4.9pp | 27.2% |
| optimization | 141 | 188 | -21.4% | 39.6% | 27.9% | -11.8pp | 20.2% |
| fine-tuning | 90 | 119 | -22.1% | 43.3% | 35.4% | -8.0pp | 27.7% |
| diffusion models | 657 | 811 | -27.3% | 51.0% | 43.5% | -7.5pp | 28.0% |
| deep learning | 274 | 333 | -28.4% | 32.8% | 25.5% | -7.3pp | 24.0% |
| transformers | 157 | 190 | -28.7% | 37.8% | 32.0% | -5.8pp | 16.8% |
| generative models | 299 | 359 | -29.2% | 49.2% | 43.3% | -5.8pp | 23.1% |
| generalization | 127 | 144 | -33.2% | 41.6% | 44.0% | +2.5pp | 19.4% |
| in-context learning | 148 | 165 | -34.3% | 37.9% | 35.9% | -1.9pp | 18.8% |
| alignment | 154 | 169 | -35.3% | 50.0% | 36.2% | -13.8pp | 20.1% |
| self-supervised learning | 140 | 153 | -35.6% | 40.2% | 29.1% | -11.1pp | 19.6% |
| graph neural networks | 302 | 321 | -37.4% | 39.5% | 27.0% | -12.5pp | 21.2% |
| transformer | 182 | 163 | -47.2% | 44.9% | 37.6% | -7.3pp | 25.2% |

## 中等体量、高接收率的细分方向

筛选条件：ICLR 2026 拿到决策的论文在 25–220 篇之间（不是无人区，也还没被淹没），按决策口径接收率排序。撤稿率低说明这类稿子在评审里普遍站得住。

| 关键词 | 投稿 | 决策 | 接收 | 接收率(决策口径) | 撤稿率 |
|---|---:|---:|---:|---:|---:|
| diffusion transformer | 35 | 27 | 20 | 74.1% | 20.0% |
| mllms | 42 | 27 | 18 | 66.7% | 35.7% |
| efficient reasoning | 46 | 32 | 19 | 59.4% | 26.1% |
| scaling laws | 49 | 39 | 23 | 59.0% | 6.1% |
| 3d gaussian splatting | 68 | 46 | 27 | 58.7% | 30.9% |
| discrete diffusion | 40 | 36 | 21 | 58.3% | 7.5% |
| robotic manipulation | 42 | 26 | 15 | 57.7% | 28.6% |
| video generation | 170 | 108 | 62 | 57.4% | 30.0% |
| image editing | 76 | 51 | 29 | 56.9% | 31.6% |
| robot learning | 39 | 30 | 17 | 56.7% | 20.5% |
| offline rl | 30 | 25 | 14 | 56.0% | 13.3% |
| 3d reconstruction | 58 | 40 | 22 | 55.0% | 29.3% |
| diffusion language models | 41 | 33 | 18 | 54.5% | 14.6% |
| pretraining | 66 | 54 | 29 | 53.7% | 15.2% |
| language model | 58 | 41 | 22 | 53.7% | 24.1% |
| spatial reasoning | 58 | 45 | 24 | 53.3% | 17.2% |
| rlvr | 49 | 34 | 18 | 52.9% | 28.6% |
| neuroscience | 45 | 38 | 20 | 52.6% | 13.3% |
| exploration | 48 | 38 | 20 | 52.6% | 20.8% |
| sparse attention | 52 | 37 | 19 | 51.4% | 21.2% |
| test-time adaptation | 58 | 39 | 20 | 51.3% | 29.3% |
| theory | 52 | 41 | 21 | 51.2% | 15.4% |
| rl | 83 | 66 | 33 | 50.0% | 15.7% |
| training dynamics | 36 | 32 | 16 | 50.0% | 8.3% |
| large reasoning model | 34 | 26 | 13 | 50.0% | 23.5% |
| finetuning | 38 | 26 | 13 | 50.0% | 23.7% |
| safety alignment | 56 | 40 | 20 | 50.0% | 19.6% |
| novel view synthesis | 49 | 38 | 19 | 50.0% | 22.4% |
| flow matching | 199 | 165 | 82 | 49.7% | 13.1% |
| multimodal large language models | 174 | 111 | 55 | 49.5% | 33.9% |
| deep research | 40 | 31 | 15 | 48.4% | 17.5% |
| variational inference | 39 | 27 | 13 | 48.1% | 23.1% |
| multimodal reasoning | 82 | 52 | 25 | 48.1% | 34.1% |
| neural tangent kernel | 27 | 25 | 12 | 48.0% | 3.7% |
| agent | 161 | 105 | 50 | 47.6% | 28.0% |
| llm reasoning | 113 | 80 | 38 | 47.5% | 27.4% |
| memorization | 49 | 40 | 19 | 47.5% | 14.3% |
| vision language models | 91 | 57 | 27 | 47.4% | 29.7% |
| learning theory | 42 | 38 | 18 | 47.4% | 2.4% |
| personalization | 59 | 36 | 17 | 47.2% | 30.5% |
| multimodal large language model | 94 | 68 | 32 | 47.1% | 27.7% |
| imitation learning | 89 | 66 | 31 | 47.0% | 22.5% |
| uncertainty estimation | 51 | 32 | 15 | 46.9% | 25.5% |
| image generation | 145 | 101 | 47 | 46.5% | 27.6% |
| generative models | 239 | 185 | 86 | 46.5% | 18.8% |

## 评审分歧与结果

| 评分标准差 | 论文数 | 接收率(决策口径) |
|---:|---:|---:|
| 0.0 | 802 | 29.6% |
| 1.0 | 5,423 | 39.1% |
| 1.5 | 4,974 | 38.8% |
| 2.0 | 1,748 | 41.9% |
| 2.5 | 678 | 38.3% |
| 3.0 | 153 | 43.8% |

## ICLR 2026 Oral 的领域集中度

共 224 篇 Oral。出现最多的关键词：

| 关键词 | Oral 篇数 |
|---|---:|
| llm(s) | 49 |
| reinforcement learning | 15 |
| diffusion models | 13 |
| agents | 8 |
| reasoning | 8 |
| flow matching | 7 |
| transformers | 6 |
| interpretability | 6 |
| benchmark | 6 |
| multimodal llm | 5 |
| representation learning | 5 |
| generative models | 5 |
| deep learning | 4 |
| generative modeling | 4 |
| optimization | 4 |
| evaluation | 4 |
| scaling laws | 4 |
| code generation | 4 |
| optimal transport | 4 |
| attention | 3 |
| pre-training | 3 |
| graph neural networks | 3 |
| mamba | 3 |
| speculative decoding | 3 |
| in-context learning | 3 |
| differential privacy | 3 |
| imitation learning | 3 |
| rlhf | 3 |
| video diffusion model | 3 |
| video generation | 3 |

## ICML 2026 录用论文的领域构成与 spotlight 率

ICML 的公开数据只有录用论文，算不了接收率；但 spotlight 率（spotlight / 该领域录用数）能反映**同样是录用论文，哪个领域更被评审当回事**。

共 6,341 篇录用论文。一级领域：

| 一级领域 | 录用数 | 占比 | spotlight | spotlight 率 |
|---|---:|---:|---:|---:|
| deep_learning | 2,370 | 37.4% | 192 | 8.1% |
| applications | 1,517 | 23.9% | 104 | 6.9% |
| general_machine_learning | 703 | 11.1% | 68 | 9.7% |
| social_aspects | 531 | 8.4% | 48 | 9.0% |
| theory | 452 | 7.1% | 56 | 12.4% |
| reinforcement_learning | 359 | 5.7% | 35 | 9.7% |
| optimization | 230 | 3.6% | 14 | 6.1% |
| probabilistic_methods | 179 | 2.8% | 19 | 10.6% |

细分领域（录用 ≥ 40 篇），按 spotlight 率排序：

| 细分领域 | 录用数 | spotlight | spotlight 率 |
|---|---:|---:|---:|
| theory->deep_learning | 53 | 10 | 18.9% |
| probabilistic_methods->monte_carlo_and_sampling_methods | 40 | 7 | 17.5% |
| theory->game_theory | 44 | 7 | 15.9% |
| reinforcement_learning->deep_rl | 77 | 11 | 14.3% |
| general_machine_learning->evaluation | 121 | 17 | 14.0% |
| deep_learning->theory | 75 | 10 | 13.3% |
| optimization->discrete_and_combinatorial_optimization | 40 | 5 | 12.5% |
| theory->learning_theory | 123 | 15 | 12.2% |
| general_machine_learning | 76 | 9 | 11.8% |
| deep_learning->sequential_models_time_series | 60 | 7 | 11.7% |
| general_machine_learning->causality | 87 | 10 | 11.5% |
| social_aspects->accountability_transparency_and_interpretability | 140 | 16 | 11.4% |
| applications->robotics | 158 | 18 | 11.4% |
| reinforcement_learning->multiagent | 45 | 5 | 11.1% |
| deep_learning | 91 | 10 | 11.0% |
| applications->neuroscience_cognitive_science | 92 | 10 | 10.9% |
| deep_learning->everything_else | 58 | 6 | 10.3% |
| deep_learning->algorithms | 69 | 7 | 10.1% |
| deep_learning->foundation_models | 109 | 11 | 10.1% |
| theory->online_learning_and_bandits | 50 | 5 | 10.0% |
| reinforcement_learning | 121 | 12 | 9.9% |
| deep_learning->graph_neural_networks | 112 | 11 | 9.8% |
| applications | 93 | 9 | 9.7% |
| applications->chemistry_physics_and_earth_sciences | 202 | 18 | 8.9% |
| probabilistic_methods->bayesian_models_and_methods | 47 | 4 | 8.5% |
| social_aspects->safety | 109 | 9 | 8.3% |
| social_aspects->alignment | 50 | 4 | 8.0% |
| deep_learning->large_language_models | 1155 | 90 | 7.8% |
| social_aspects->privacy | 78 | 6 | 7.7% |
| general_machine_learning->clustering | 40 | 3 | 7.5% |
| deep_learning->generative_models_and_autoencoders | 349 | 26 | 7.4% |
| deep_learning->selfsupervised_learning | 46 | 3 | 6.5% |
| social_aspects->security | 67 | 4 | 6.0% |
| general_machine_learning->representation_learning | 84 | 5 | 6.0% |
| applications->time_series | 68 | 4 | 5.9% |
| general_machine_learning->transfer_multitask_and_metalearning | 86 | 5 | 5.8% |
| reinforcement_learning->batchoffline | 53 | 3 | 5.7% |
| applications->health_medicine | 163 | 9 | 5.5% |
| applications->computer_vision | 515 | 28 | 5.4% |
| deep_learning->attention_mechanisms | 74 | 4 | 5.4% |
| optimization | 56 | 3 | 5.4% |
| probabilistic_methods | 43 | 2 | 4.7% |
| deep_learning->robustness | 86 | 4 | 4.7% |
| applications->everything_else | 97 | 4 | 4.1% |
| applications->language_speech_and_dialog | 108 | 4 | 3.7% |
| deep_learning->other_representation_learning | 86 | 3 | 3.5% |
| general_machine_learning->unsupervised_and_semisupervised_learning | 41 | 1 | 2.4% |

