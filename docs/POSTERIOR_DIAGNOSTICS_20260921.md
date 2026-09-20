# Posterior decision diagnostics — 2026-09-21

## Scope and status

This is an AI-assisted internal exploratory/post hoc diagnostic of the completed synthetic sampler run. It does not alter the predeclared W1 or time-gate protocol, does not select methods, and is not an official application-performance or calibration claim.

The audit uses the completed `confirmation` phase manifest as the sole cohort definition. The manifest contains 250 sampler cells: five methods crossed with five training seeds and ten dataset seeds. The methods are `full`, `tail_fixed`, `surrogate_only`, `surrogate_full`, and `certified`. Each sampler cell contains saved samples and weights, a receipt, a `done` marker, and a saved numerical reference.

All 250 manifest cells were checked for completed status, exact `done` content, cell receipt hashes, and finite samples and weights. The phase completion marker was checked for exact `completed` content. The 50 corresponding learned assets were separately checked against their asset receipts before their data and saved references were cached and reused. The manifest SHA-256 and phase-marker SHA-256 are recorded in `summary.json`. No GPU, server, git operation, or external upload was used.

## Metric definition

For each sampler cell, the saved weighted sample cloud is evaluated directly. For every coordinate of the saved generating parameter `theta`, the posterior mean is the normalized weighted sample mean. The 90% and 95% marginal intervals are weighted empirical equal-tail intervals. The empirical inverse CDF is implemented by removing zero-weight entries only after validating that samples and weights are finite and weights are nonnegative, sorting the retained samples, forming the normalized cumulative sum, and using `searchsorted`; it does not linearly interpolate empirical observations.

The scalar cell metric is the mean over the saved problem coordinates. Coordinates are not counted as independent datasets. Results are aggregated over the manifest-defined training-seed × dataset-seed cells.

Two saved reference rows are reported:

- `learned_reference`: the numerical reference stored in each asset's `reference.npz`;
- `true_reference`: the independently verified true numerical reference stored in the corresponding standard asset's `true_reference.npz`.

Both references are evaluated from their saved density/CDF integrals and inverse CDFs. These assets contain the intended prior-corrected compositional targets. The `true_reference` row is checked for identical values across the five training-model rows of each dataset seed.

## Overall descriptive results

| method | mean MSE | mean RMSE | 90% coverage | 95% coverage | mean width 90% | mean width 95% |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.015160 | 0.118889 | 0.8425 | 0.8850 | 0.333362 | 0.396358 |
| tail_fixed | 0.031787 | 0.174583 | 0.5525 | 0.6300 | 0.277474 | 0.330284 |
| surrogate_only | 0.015595 | 0.120452 | 0.8425 | 0.9000 | 0.333470 | 0.396413 |
| surrogate_full | 0.015915 | 0.121338 | 0.8325 | 0.8875 | 0.332273 | 0.395093 |
| certified | 0.015164 | 0.118852 | 0.8375 | 0.9050 | 0.332179 | 0.395057 |
| learned_reference | 0.015126 | 0.118285 | 0.8775 | 0.9325 | 0.362686 | 0.432195 |
| true_reference | 0.015102 | 0.118455 | 0.8750 | 0.9375 | 0.362783 | 0.432316 |

These are realized descriptive frequencies from the finite synthetic cohort. They are not claims of nominal 90% or 95% coverage in a population. The cohort has ten dataset seeds and five training seeds, so it is not sufficient for a general calibration statement.

## Paired comparison with the full baseline

For every non-full method, the audit uses the same crossed bootstrap index draws as the full baseline. The reported difference is method minus `full`, with training-seed and dataset-seed indices resampled independently but shared across the paired method and baseline matrices. The JSON artifact contains the 10,000-replicate 5%–95% intervals for every metric.

| method − full | MSE difference | RMSE difference | 90% coverage difference | 95% coverage difference | width-90 difference | width-95 difference |
|---|---:|---:|---:|---:|---:|---:|
| certified | +0.0000045 | −0.0000371 | −0.0050 | +0.0200 | −0.0011830 | −0.0013010 |
| surrogate_only | +0.0004357 | +0.0015630 | 0.0000 | +0.0150 | +0.0001075 | +0.0000549 |
| surrogate_full | +0.0007549 | +0.0024492 | −0.0100 | +0.0025 | −0.0010892 | −0.0012646 |
| tail_fixed | +0.0166274 | +0.0556938 | −0.2900 | −0.2550 | −0.0558881 | −0.0660740 |

These paired differences are diagnostics of this saved confirmation cohort. They do not establish superiority, robustness, or real-application validity.

## Per-seed results and artifacts

The generated `summary.json` contains method means for every training seed and every dataset seed, as well as the crossed-bootstrap intervals for every method and reference row. Every summary method uses the same deterministic crossed index draws for each metric; every method-minus-`full` difference uses the same deterministic difference draws for that metric. `cell_metrics.csv` contains exactly 350 rows: 250 sampler rows plus one `learned_reference` and one `true_reference` row per training-seed×dataset-seed asset, with no reference row duplicated once per sampler method.

The `true_reference` identity check passes per dataset seed: the same dataset's true-reference MSE is identical across its five training-model rows. The number of distinct datasets remains ten.

## Limitations

This diagnostic concerns only the saved synthetic simulator and the completed confirmation manifest. It does not demonstrate real-application calibration, generalization outside the simulator, or a theorem about any sampler. The saved reference objects are numerical integrals, and the sampler metrics inherit the finite particle count and the recorded weights.

## Reproducibility

- Script: `analyze_posterior_decisions_20260921.py`
- Cell-level output: `results/dual_20260921/posterior-explore/cell_metrics.csv`
- Summary and paired bootstrap output: `results/dual_20260921/posterior-explore/summary.json`
- Invocation: `analyze_posterior_decisions_20260921.py --run work/recovered-dual-20260921/new/run-explore --phase confirmation --output results/dual_20260921/posterior-explore`
- Runtime: local CPU with one thread; no GPU, server, git, or external upload.
