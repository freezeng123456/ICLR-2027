# Checkpoint-only nonseparable posterior composition

Frozen before observing any result on this task. This small applicability experiment follows the negative 416-cell sensor development study. Its budget comes from the same four-GPU-hour iteration allowance; it does not authorize a new unbounded sweep. Each phase stops after one hour of sampling/evaluation wall time. No performance confirmation opens unless the predefined development gate passes.

## Question and target

Five independently trained posterior heads are assigned, one per sensor, to five nonorthogonal unit projections of a two-dimensional parameter with standard Gaussian prior. Directions have angles `(g+0.25)*pi/5`, noise standard deviations `0.55+0.10*g`, and additive ambiguity offsets `0.40+0.15*g`, for g=0,...,4. Observations are `a_g dot theta + sign_g*offset_g + noise_g*epsilon_g`. The simulator seed is `920000+dataset_seed`.

Each frozen neural head supplies a two-component scalar posterior f_g. The composed target is `phi_2(theta) * product_g [f_g(a_g dot theta)/phi_1(a_g dot theta)]`. The prior division is essential. Samplers receive only directions and the predicted variances, means and weights. They cannot read the true parameter, generating signs, analytic posterior, reference samples or evaluation results. The generative likelihood remains analytically available to the evaluator, so this is a controlled checkpoint-only interface experiment, not a real intractable-likelihood application.

The five fixed checkpoints are `training_0/final.pt` through `training_4/final.pt`, from the previously completed learned-SBI run. The bank is one fixed collection, not five independent repetitions of this experiment. Strict reload, finite-parameter checks and the following SHA-256 values are required:

| Training seed | SHA-256 |
|---|---|
| 0 | ce9344dce5f56b4cda176bcdb36371f48911618aa600da1465617419b4995b69 |
| 1 | 642c9ac8b37cc879a46c5dc332c9d99eb4ae9d0bd2c4b38a1ce3bbe1af22951a |
| 2 | 4ad56f1f2de8ac7e99a997dc798fcc9a82d5e41bcaa01483fa8161ba24fa2737 |
| 3 | 40d2f60f7b01b721909df18cdd26cc0bda1e28ef62dd47bb3bc9b7acfaa1fbc4 |
| 4 | 4f8f7d024dee2ba1d205fcf59f39c4f32d87524ab703a9051d4ad4e7ca38f59f |

## Methods and fixed design

All methods return 4096 weighted particles. Adaptive SMC uses ESS target 0.8, systematic resampling at every nonfinal temperature, retained final weights, three pCN moves and one complete-bridge Metropolis sign flip per stage, and at most 128 stages. The common tail precision is `I + sum_g (1/v_g-1) a_g a_g^T`. Reference means use 16 standard-prior starts (fixed seed 20261020), 50 latent-component EM steps, a squared tail-Mahalanobis merging threshold of 1e-8 and normalized target heights with a 1e-6 pre-normalization weight floor. This search has no global-mode guarantee.

Eight development settings are fixed:

- prior-reference SMC at pCN scales 0.15, 0.35 and 0.65;
- single tail-Gaussian SMC, pCN scale 0.5;
- at most four-component tail-mixture SMC, pCN scale 0.5 (the single primary candidate);
- direct importance sampling from the same mixture reference;
- mixture SMC with 10% probability of an independent full-q refresh at each mutation;
- exact 32-component enumeration followed by direct sampling, with enumeration cost included.

The last baseline is deliberately included because exact composition is feasible for five factors. A gain over prior-SMC alone does not establish an advantage over all available methods. Mixture SMC, pCN, ESS adaptation and mode-height proposals are not claimed as general methodological firsts.

Development uses data seeds 1600–1603, crossed with three paired sampler repeats and eight settings: 96 cells. Sampling seed is `9300000+10*dataset_seed+repeat`. Execution order seed is 20261022. Select the fastest prior-SMC setting whose mean projected W1 is within 0.001 of the best prior-SMC error. The mixture candidate expands only if its mean W1 is at most 0.002 worse and mean total time is below 80% of that baseline. Neither the direct-IS nor global-refresh diagnostic may be promoted after seeing this cohort.

If the gate passes, confirmation uses new data seeds 1700–1711, the selected prior baseline and all five non-prior settings, with three paired repeats: 216 cells. Execution-order seed is 20261023. The primary comparison is the frozen mixture against the selected prior baseline. First average the three repeats within each data seed, then paired-bootstrap the twelve data seeds 10000 times with seed 20261024. Claim a conditional accuracy/cost improvement only if the one-sided 95% upper W1 difference is at most 0.002 and the one-sided 95% upper ratio of mean times is below 0.8. Report exact enumeration and all diagnostics regardless of this decision.

## Evaluation, costs and limits

The learned target expands to 32 Gaussian components with common covariance. The evaluator separately checks the prior-corrected product and this mixture. Projection directions are the 32 evenly spaced unit vectors with angles `j*pi/32`. Weighted empirical-versus-mixture W1 is evaluated through the analytic primitive of a Gaussian CDF, resolving interval crossings numerically; report root-location error separately from floating-point roundoff. Thus this evaluation avoids a random finite-reference-sample floor. Report posterior mean/covariance error, composition-normalizer error, effective sample size, factor calls, mode-search diagnostics and any available region-mass discrepancy. Separately report learned-versus-generating posterior error; it cannot be assigned to the sampler.

Time includes reference optimization, density calculations, output transfer and the common five network predictions per query. Loading the fixed checkpoint bank is a separate one-time initialization cost. All comparison methods receive the same frozen network outputs. GPU measurements require an isolated device and record process lists before and after each cell. Evaluation is CPU-side and outside sampling time. Exact enumeration uses CPU and is timed from its own factor-parameter input, with no cached oracle object.

This experiment is two-dimensional, has five factors and one fixed bank. It cannot establish high-dimensional scaling, external application value, broad posterior calibration, or universal benefits. Complete negative results, failed gates, source/input hashes, original samples, diagnostics and logs are retained and published.
