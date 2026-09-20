# Tail-informed reference SMC study

This extends `CONTINUED_ITERATION_PROTOCOL_20260921.md` before any new GPU result is observed. All methods use 16384 particles; particle count cannot explain a measured difference. The task is the existing exact-reference nonseparable sensor inverse problem, with 12 factors, dimensions 2/8 and ambiguous/regular noise regimes. Development seeds are 1400–1403. Confirmation seeds are 1500–1519 with two paired sampler repeats. Confirmation consists of 80 problems, clustered by 20 data seeds.

## Frozen development configurations

- Tail reference: a single Gaussian or a mixture with at most 2 or 4 components; covariance scales 1 or 4; pCN proposal scales 0.5 or 1.0. Fixed 32-step latent-sign mode search, ESS target 0.8, three pCN moves and one fully corrected global sign flip per stage. These are 12 candidate settings.
- Adaptive-prior reference control: the same annealing/move implementation starting from N(0,I), with proposal scales 0.15, 0.35 or 0.65, ESS 0.8 and three local moves plus sign flip. These are three baselines.
- Fixed-stage likelihood SMC: 32, 64 or 128 equally spaced temperatures crossed with proposal scales 0.15, 0.35 or 0.65; three pCN moves and one sign flip. These are nine baselines.
- Full-factor diffusion and the existing anchored tail method are mechanism diagnostics, both at 2048 Euler steps, with anchor batch 8.

Twenty-six settings crossed with sixteen development problems give 416 cells. The baseline is selected as the fastest baseline whose mean projected W1 is within 0.001 of the best baseline mean. A candidate is eligible only if, against both this baseline and fixed-stage SMC (64 temperatures, scale 0.15), its mean projected W1 is at most 0.002 worse, each stratum is at most 0.004 worse, and its mean time is below 80% of each comparison. Choose the fastest eligible candidate; ties are resolved by setting identifier. With no eligible candidate, retain all outputs and do not open this candidate's performance confirmation.

## Confirmation and costs

Confirmation retains the selected candidate, selected baseline, the fixed 64-stage baseline, matched Gaussian/prior reference ablations and the two diffusion diagnostics. Identical baseline identifiers are evaluated once. Every setting uses the same particle count. The primary claim requires both paired one-sided 95% W1-degradation bounds at most 0.002 and both time-ratio bounds below 0.8. Bootstrap 10000 times with seed 20261012 over the twenty shared data-seed clusters, retaining dimensions, noise strata and the two paired sampling repeats. Report each stratum and normalizer error separately.

Sampling time includes reference construction, mode search, CPU/GPU transfer, initial draws and returned output. Shared exact-reference construction and metric computation are outside sampler timing. GPU synchronization brackets each measurement. CPU evaluation of saved particles overlaps subsequent GPU sampling but receives no access from the sampling algorithm. Raw outputs and isolated-device checks are retained for every cell. Configuration, source and input hashes identify the executed study.

The reference construction must not read truth, exact mixture enumeration, evaluation samples or confirmation metrics. The observation likelihood is available in this controlled task. Performance here does not establish advantages for arbitrary neural score models or unavailable-likelihood inference. Adaptive temperatures do not by themselves justify finite-sample unbiased normalizer claims.
