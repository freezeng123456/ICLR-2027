# Compositional diffusion experiment protocol

Frozen before SCNet submission, 9 September 2026. The two-day research deadline is 11 September 2026, 19:49 China Standard Time.

## Question and claim boundary

Does unbiased estimation of a compositional Feynman–Kac potential suffice for accurate finite-step posterior sampling? The study separates random drift covariance, drift–potential covariance, and exponential-weight variance. The Gaussian control variate and second-order cumulant correction are evaluated as classical statistical constructions applied to this problem. Their generic use is not claimed as new.

The targets have exact factor scores and numerically normalized reference marginals. These experiments isolate sampling error from learned-score approximation. They do not establish performance on learned simulators or image generation.

## Distribution and sampler

The prior is a standard multivariate normal. Each group posterior is a product across coordinates of a two-component Gaussian mixture. The composed target is the product of the group posterior densities divided by the prior to power G-1. Three predefined families cover Gaussian factors, separated mixtures, and weak mixture factors. Parameters are deterministic functions of group and coordinate indices in `FactorModel`; they are not selected using reported outcomes.

Noise time u has alpha squared exp(-u). The path is the product of the forward VP group marginals with the same prior correction. With residual scores r_g=s_g+x and R=sum r_g, the reverse probability-flow drift is R/2 and the Feynman–Kac potential is (||R||^2-sum ||r_g||^2)/2. Adding a Langevin term with diffusion coefficient one preserves the target path in continuous time. Discretization uses Euler–Maruyama with u values on a fixed quadratic grid from 20 to zero, exponential weights at the left endpoint, and systematic resampling whenever ESS falls below half of P.

Initialization is N(0,I) at finite u=20. It is an approximation; an endpoint sensitivity check is required before manuscript claims about discretization limits. One-dimensional reference densities use 65,537 quadrature nodes on [-12,12], fail on appreciable boundary density, and are cross-checked by grid refinement. The dimension-eight reference factorizes; marginal metrics do not characterize all possible cross-coordinate dependence.

## Frozen comparison

Full Feynman–Kac, full unweighted composition, naive small-batch substitution, unbiased U-statistic potential, paired cumulant correction, Gaussian control variate, and the combination of control variate with paired cumulant correction.

Groups: 16 and 64. Dimensions: 1 and 8. Steps: 128, 512, 2,048. Particles: 8,192. Seeds: 0–4. Batch size: four draws with replacement independently for each particle and step. Paired cumulant methods use two batches and incur twice the group evaluations. This gives 1,260 cells. All results, including poor or divergent outcomes, belong to the coverage report.

Primary metric: average marginal 1-Wasserstein error against the quadrature reference. Secondary metrics: marginal Kolmogorov distance, mean error, variance relative error, positive-mode mass error, minimum/final ESS, resampling count, sampling time, group evaluations, peak GPU memory. Cost comparisons count both batches and include control-variate arithmetic in measured sampling time. Evaluation quadrature and disk serialization are outside sampling time. The analytic Gaussian control preparation is cheap here; no claim about its cost for a learned model is implied.

Report all five seeds and uncertainty intervals. The same predefined families and numerical settings apply to every method. No favorable subset replaces the full matrix. Finite-step gains require comparisons at equal group-evaluation budgets and measured wall time; a gain at unequal work is descriptive only.

## Compute and validation

SCNet resource check: `xhhgnormal`, 57 nodes, four RTX 3080 GPUs per node. Parent association `acu722p2q8` reports GPU cap 40 and submit cap 200; the account had no active jobs at preflight. This experiment uses at most four concurrent one-GPU tasks, each with four CPU threads and OMP/MKL threads set to one. Twenty-eight array tasks pack 45 cells each. The fixed wall-time ceiling is 30 minutes per task, hence 14 allocated GPU-hours maximum for the matrix. A three-cell single-GPU smoke is separate and has a five-minute ceiling. Submission requires smoke success; no automatic expansion of resources or time on timeout.

Source is a local Git commit exported to a unique server directory because the previously verified server Git version lacks worktree support. Archive SHA-256, full commit, actual interpreter, Slurm allocation, model dtype float64, and CUDA visibility are recorded. Each cell saves configuration, metrics, log, summary, samples, quadrature reference, and completion marker. A failed subprocess fails its array task. The final audit checks all expected cell IDs, exit codes, finite outputs, hashes, and independent metric recomputation.

Required additional checks before interpreting the matrix: Gaussian exactness of the control variate; enumeration of every batch on a four-group problem; Feynman–Kac identity by direct automatic differentiation; reference-grid convergence; particle/step/initial-noise sensitivity on predefined representative targets; inspection of distribution plots. These checks are validation of the methods and numerical references, not grounds for excluding matrix outcomes.
