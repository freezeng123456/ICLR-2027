# Paper-level evidence package, 2026-09-21

Status: all four study lines completed and audited. This is a research draft and an internal evidence package, not a submission or a human peer-review certificate.

The Chinese interpretation is in `docs/PAPER_LEVEL_RESULTS_20260921.md`. The paper source is `manuscript/main.tex`. `verification-index.json` records raw archive digests and audit report digests.

`ICLR_2027_research_draft_20260921.pdf` is the final compiled draft: 35 pages in total, main text through page 8, references beginning later on page 8, followed by proofs and full results. It is marked as not submitted. Rendering and software verification are recorded in `docs/PAPER_QA_20260921.md`; this does not certify theorem novelty or conference acceptance.

## Cohorts and claims

| Study | Run cells | Independent grouping | Status and supported interpretation |
|---|---:|---|---|
| Learned development | 72 | 2 checkpoints × 4 new datasets | Frozen nine-setting selection; all raw inputs and references audited |
| Learned confirmation | 400 | 5 checkpoints × 20 new datasets | Selected anchor passes original joint baseline gate and fails original coordinatewise baseline gate |
| Equal-budget ablation | 300 | Same 100 confirmation targets | Post-hoc; contains 100 exactly equal anchor replays; no new-data claim |
| Nonseparable sensor development/confirmation | 576 | 8 development targets; 80 confirmation problems clustered by 20 seeds | All 88 reference assets and 576 cells audited; no formal speed claim due external GPU overlap |
| Gaussian path proposal development | 56 | 8 fresh development targets | All checked; no candidate eligible for expansion; no held-out improvement claim |

Total: 1404 run cells, including 100 equivalent replays. Counts of run cells are not counts of independent datasets.

## Files

- `*-statistics.json`: means, paired intervals, inherited gate diagnostics and explicit evidence status.
- `*-cells.csv`: all cell-level observations for the learned cohorts, equal-budget ablation and sensor confirmation.
- `*-audit.json`: input/source checks, independent reference and metric checks, checkpoint reloads where applicable.
- `sensor-cross-platform-diagnostic.json`: retained local diagnostic. Its samplewise Gaussian replay differs across numerical libraries for some eight-dimensional assets. All local density/moment/metric/matrix checks agree. The original-runtime audit reproduces every reference and passes all checks.
- `twist-development-audit.json`: all eight references, 56 cells, matrix checks, log-weight normalization and failed expansion gate.
- `verification-index.json`: package-level checks and exact local raw archive SHA-256 values.
- `tests-*-cuda.log` and `tests-repository-final.log`: actual remote GPU test results and the final local repository regression result; local CUDA skips are retained.

Raw archives are retained under `work/recovered-paper-20260921/old/` and `work/recovered-paper-20260921/new/`. The matched run has a relative `assets` link to the original anchor run; both roots are recovered together. Do not detach the matched root from that base without preserving its input files.

## Frozen sampling source commits

- Original learned study: `e77042f59d62fd6908108e6a0496cd4a0188a32d`.
- Nonseparable sensor study: `451b73f06df59cc0e74c10ece6aff737cf497fa5`.
- Gaussian path development: `c1d8203d23936128cca46103145712803fd780f3`.
- Equal-budget ablation: `f1cf3409a2f28d7d10b54fab974bc5410196a80e`.

The frozen source archive is included with each raw run archive. Audit and reporting code may be newer and identifies its own digest. The equal-budget audit reuses already independently checked same-target references and same-grid certificates; this reuse is explicit in its scope.

## Interpretation limits

The mathematical results concern a specified exponential Euler operator and known shared Gaussian tail classes. Finite normalizers, finite independent-path weight moments, particle error and application-level benefit remain separate claims. Gaussian twisting and Riccati recursion build on established prior work. Neither the sensor task nor the learned mixture experiments establish large-neural-score or real-world deployment gains. The paper retains all failed gates and all tested candidate configurations.
