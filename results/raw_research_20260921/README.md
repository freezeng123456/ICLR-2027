# Complete compositional-diffusion research records

This branch publishes complete canonical experiment directories for the original oracle study, learned-density extension, H20 replication, dual-track experiments and paper-level confirmation. It also retains mechanism diagnostics and unsuccessful development attempts under explicitly named directories. Scientific interpretation and audits are on branch `research/iclr-review-dual-20260921`.

`FILE_MANIFEST.json` contains every retained file path, byte count and SHA-256 digest, plus the two relative asset links shared between original and equivalent-replay studies. Every source revision used for sampling is reachable in this branch's Git history. The manifest is published with the first transfer batch; completeness requires the final publication receipt on the research branch, not merely the presence of this README.

All per-cell outputs are retained: configurations, raw samples and weights, exact/numerical references, step records, metrics, logs and completion markers. The learned extension includes all five full training histories and checkpoints. Source-transfer archives and duplicated packed recovery archives are redundant transport containers; their original contents and frozen source commits are retained rather than duplicating those containers. Runtime environments and caches are excluded.

Negative results and unsuccessful development attempts are evidence about the tested configuration, not part of the formal confirmation cohort. Run counts are not independent-dataset counts. The original one-step experiment did not retain all generated particles; its retained histograms and largest-weight records remain the available evidence. This publication does not invent missing historical output.

To reproduce a study, retain its directory structure and use the corresponding protocol and audit script on the research branch. In particular, keep `dual/new/run-anchored-confirmation` together with `dual/new/run-anchored-replay`, and `paper/new/run-anchor` together with `paper/new/run-matched`, so the relative asset links resolve.

All data in these controlled studies are simulated. Publication does not establish practical superiority, human peer review or conference submission.
