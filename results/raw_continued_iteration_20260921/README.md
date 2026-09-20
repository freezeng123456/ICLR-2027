# Complete continued-iteration records, 2026-09-21

This append-only archive contains all 512 completed development cells from the tail-reference sensor study (416 cells, 16 targets) and the checkpoint-only learned-factor composition study (96 cells, 4 data targets with 3 paired sampler repeats). Neither primary development screen passed; neither opened its reserved confirmation cohort. All method settings and negative results remain present.

`FILE_MANIFEST.json` records 3,726 original paths, their byte counts and SHA-256 hashes. These files occupy 390,036,922 logical bytes, with 375,767,102 bytes of distinct SHA-256 content. The manifest and this README are additional archive-index files. Completeness requires the final research-branch publication receipt, not just this manifest's presence.

- `sensor/run`: original assets, configurations, raw samples and weights, diagnostics, GPU process snapshots, receipts, completion markers, aggregate rows and frozen selection.
- `sensor/audits`: independent original-environment audit of all 416 cells and 16 assets.
- `learned/run`: original learned/true reference pairs, network outputs, observations, all 96 raw cell records, source/checkpoint receipts and failed development decision.
- `learned/audit-independent`: independent direct-density, normalization, projected-W1 and integrity audit of all 96 cells and four assets.
- The two task roots retain production and audit test logs, launcher logs, completion exits and post-run runtime records.

The five full model-training directories are already preserved in the sibling historical archive at `results/raw_research_20260921/extension/run/training/training_0` through `training_4`; their exact checkpoint hashes appear in the learned protocol and run manifest. They are the same fixed bank, not a new round of five trainings. Transport tarballs, redundant top-level copies of the same aggregate rows, runtime caches and dependencies are excluded; their unique scientific contents are retained in the canonical directories.

Executed production revisions in this repository:

- Sensor: `b7aa4f0ee8d2d80eecbcf9967b820e62597e2d76`.
- Learned composition: `8f656b41557962ba69fe9c64049e6379ad1c98a1`.

Protocols, post-hoc auditors, analysis and the updated manuscript live on [`research/iclr-review-dual-20260921`](https://github.com/freezeng123456/ICLR-2027/tree/research/iclr-review-dual-20260921), with the current ledger in `docs/CONTINUED_ITERATION_RESULTS_20260921.md`. Source bytes are also identified by every run manifest. Later audit additions do not alter the frozen sampling source or original records.

All observations are simulated. Development cell counts are not independent-data counts. Numerical verification does not imply performance superiority, a successful confirmation screen, external application validation, independent human peer review or conference submission. Preserve the audit reports' numerical-integration, floating-point, checkpoint-reload and snapshot-only GPU-isolation qualifications when reusing these results.
