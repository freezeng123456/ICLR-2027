# Integrability supplement protocol

Fixed before supplementary SCNet submission on 9 September 2026. This supplement tests the algebraic integrability result; the original 1,260-cell matrix remains unchanged and will be reported in full.

## Tail-curvature control

For each Gaussian mixture group with a shared component covariance, use that covariance for the Gaussian score control, with the mixture mean as its center. The resulting score residual is bounded throughout state space. Its control-variate U-statistic potential retains the full potential's quadratic tail coefficient. Compare this estimator (`tail`) and its paired cumulant variant (`tail_cumulant`) with the already frozen full, unbiased, and moment-matched CV results.

Use both non-Gaussian families, G=16/64, d=1/8, K=128/512/2048, P=8192, M=4, seeds 0–4, and the same noise grid, diffusion coefficient, resampling rule and metrics. These 240 cells are a matched extension of the original matrix. A guarantee of finite normalization is distinct from lower finite-particle distribution error. Every outcome is retained.

## Exact Gaussian one-step counterexample

Use four zero-mean Gaussian group posteriors with likelihood precisions (0.1,0.2,2,4). At noise time u=log(2), the residual strengths are (1/21,1/11,1/2,2/3). Initialize exactly from the composed path at that time, N(0,154/355), and use h=1/2 with diffusion one.

The full potential has coefficient 346/693 and finite denominator 2503/3195. For every finite iid-with-replacement batch size M>=2, the event that every index selects the strongest factor has positive probability 4^(-M) and coefficient 8/3; its denominator is -167/1065. Therefore the random weighted population normalizer is infinite although both instantaneous estimators are unbiased. This statement is an exact rational calculation, not an inference from sampled weights.

Measure full and unbiased-U one-step experiments at P=1024,16384,262144,1048576, seeds 0–19. Unbiased batches use M=2,4,8,16,32. The full operator is evaluated once per P/seed; its Gaussian CV is algebraically identical. These give 480 cells. Metrics include sampled log normalizer, weighted propagated variance, ESS fraction, maximum normalized weight, score/potential MSE and realized counts of nonintegrable batch configurations. Compare against the exact full *discrete* operator; h=1/2 is not asserted to accurately resolve the continuous path.

Save configuration, runtime, summary, metrics and completion marker for every cell. Store a fixed weighted histogram and the 1,024 largest-weight records as diagnostics; all particles remain reproducible from the recorded seed, software version, source commit, chunk size and configuration. These one-step cells do not store all P particle arrays. This avoids making storage proportional to the particle-count sweep. The original full-path matrix retains all samples.

## Resource ceiling and checks

The supplement contains 720 cells, packed across 12 one-GPU tasks with four CPU cores each, at most four concurrent tasks, and a twenty-minute wall-time limit per task (four allocated GPU-hours maximum). Start after the original matrix to preserve the four-GPU experiment concurrency limit. A new-source smoke checks one tail-CV run and one one-step run, with a five-minute ceiling. Validate source/archive hashes, exact counterexample arithmetic, batch enumeration, Slurm exit codes, expected cell IDs, recorded metrics and representative independent reruns before writing scientific conclusions.
