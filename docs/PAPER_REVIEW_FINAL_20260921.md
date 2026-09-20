# Paper-level internal review: Integrability of Subsampled Compositional Diffusions

Final revision note: the review below is retained as its at-review-time snapshot. The subsequent learned, sensor, Gaussian-pilot and equal-budget audits have completed; see `docs/PAPER_LEVEL_RESULTS_20260921.md` and `results/paper_20260921/verification-index.json`. The revised theorem uses primitive Gaussian/bounded-correction assumptions and an exact transition-density-ratio proof. Gaussian prior art is cited in the main related work. The mixture discussion now distinguishes the marginal-mixture weight from a separate valid augmented-component construction. The same-budget ablation removes an unqualified 4.9-times algorithmic speed claim. Application-level superiority remains unsupported, and human mathematical review is still outstanding.

**Status.** This is an AI-assisted internal working assessment, prepared for the author’s revision process. It is not an independent human review, not a conference review, and it does not assign an acceptance probability.

The central mathematical question is worthwhile: an unbiased estimator of a quadratic potential can have an exponential with an infinite population normalizer. The paper also does a good job separating population integrability from finite-particle error in several places. The present rejection risk is chiefly claim calibration and evidence closure, rather than the absence of a mathematically meaningful phenomenon.

## Evidence status

Observed in the current files:

- The main paper explicitly defines a discrete weighted Euler operator and says that its finite-step target is distinct from the continuous flow target (`manuscript/main.tex:61`). It also states that conditional unbiasedness does not imply an unbiased exponential kernel or a finite normalizer (`manuscript/main.tex:78`).
- The scalar with-replacement counterexample is concrete: the all-strongest batch has positive probability and makes the subsampled normalizer infinite, while the full-factor update is finite (`manuscript/main.tex:137-147`). The associated experiment records finite-particle invisibility of this event even at very large particle count (`manuscript/experiments_paper.tex:5`).
- The paper states useful negative results rather than claiming universal performance: tail control can restore the strict domain while failing to improve sampling accuracy, and no general speed--accuracy superiority is claimed (`manuscript/main.tex:38`, `manuscript/experiments_paper.tex:17`, `manuscript/dual_results.tex:30-32`).
- The current Gaussian path diagnostic is a path-level result: the first weight moment is finite and the second is infinite for the stated proposal, and the text explicitly supplies no resampled-particle variance theorem (`manuscript/dual_results.tex:34-35`).
- The matrix extension is scoped to common quadratic tail coefficients and says that the scalar extreme-batch reduction is not being extended to general random matrices (`manuscript/main.tex:183-187`; `manuscript/matrix_tail_extension.tex:21-35`).
- The saved sensor file is an observed, parseable confirmation summary with 20 data seeds, 480 cells, 10,000 bootstrap replicates, and `audit_status: separate independent numerical audit required` (`work/paper-sensor-confirmation/statistics.json:1-49`, `:10184-10194`). Its statistics are therefore preliminary saved benchmark results, not independently closed new evidence.

Unverified for this review:

- The new sensor and anchor numerical audits are still in progress. I do not use any new anchor number as a positive paper claim.
- I did not treat the saved aggregate statistics as proof that every underlying raw sample, receipt, source hash, checkpoint, and reference calculation has passed an independent audit.
- No claim below relies on deployment data. The sensor study is a controlled synthetic inverse problem with an exact reference, as the manuscript itself says (`manuscript/main.tex:200`).

## Five largest concrete rejection risks

### 1. The novelty boundary is still vulnerable to known Gaussian twisting and controlled-SMC prior art

The Gaussian backward recursion, Gaussian path twisting, Riccati structure, and zero-variance Gaussian special cases are established ingredients in controlled SMC and twisted particle filters. The project’s own research note records Whiteley and Lee, *Twisted Particle Filters*, and Heng, Bishop, Deligiannidis, and Doucet, *Controlled Sequential Monte Carlo*, including the linear-quadratic-Gaussian/Riccati material in Sections 3.1 and Supplement 7.1 (`docs/PAPER_LEVEL_RESEARCH_20260921.md:34`). The current Gaussian twist sidecar also correctly says that it claims no novelty for general twisting or zero-variance Gaussian constructions (`manuscript/gaussian_twist_theorem.tex:1-7`).

The defensible contribution is narrower: exact finite-step integrability certificates for the specified subsampled exponential Euler operator, including the two-extreme reduction for scalar/diagonal with-replacement tails, the prefix/suffix reduction without replacement, and deterministic common-tail preservation. The main paper mostly states this boundary (`manuscript/main.tex:195-200`), but a reviewer can still read the Gaussian machinery as a new twisting method unless the abstract, introduction, related work, and Gaussian-theorem transition all repeat the distinction.

**Required action:** make the novelty sentence operational: the Gaussian recursion is an analysis/reference construction; the contribution is the batch-history integrability decision and its counterexample/limitations. Cite the two prior primary sources directly in the main related-work discussion. Do not describe a Gaussian twist, Riccati recursion, or zero-variance special case as newly introduced.

### 2. The all-moments theorem needs primitive assumptions to prevent a substantially broader reading

The finite-horizon Gaussian theorem is mathematically plausible under its stated envelope, but its use in the paper will be rejected if readers infer a general all-moments theorem for arbitrary learned factors, arbitrary mixtures, or long horizons. The theorem requires finite (K), a finite history set, nondegenerate Gaussian initial law, positive-definite transition noise, strict backward precisions, common quadratic potential coefficients, and a uniformly bounded linear-log correction. The proof gives finite-dimensional Gaussian domination and explicitly does not provide a bound uniform in (K) (`manuscript/gaussian_twist_theorem.tex:8-60`).

This is consistent with the main matrix theorem, which assumes a tail envelope and leaves the semidefinite boundary unclassified (`manuscript/matrix_tail_extension.tex:21-51`). It is also consistent with the path-moment diagnostic: first-moment finiteness does not imply second-moment finiteness (`manuscript/dual_results.tex:34-35`). The risk is presentation: “all positive moments” can be mistaken for resampling variance control, a long-time stability result, or a universal guarantee for a learned score whose residual is not globally bounded.

**Required action:** state the primitive assumptions before the theorem and again at its application point: finite horizon, finite batch-history set, Gaussian initial/noise with positive-definite covariances, globally bounded residual/mean corrections after the common affine part, and common quadratic potential. State separately that independent-path importance sampling is covered, while resampling stability, uniform-in-(K) constants, and universal variance claims are not. For normalized estimators, require the relevant second moment explicitly.

### 3. The empirical section does not yet establish an application-level benefit, and the newest anchor evidence is not closed

The existing results are valuable as diagnostics, but they do not support a positive application claim. The difficult product-mixture baseline improves marginal W1 through coordinatewise resampling while retaining substantial joint sign-TV error (`manuscript/dual_results.tex:25`). In the learned study, fixed-mean tail control is much worse than full weighting, while anchored control is closer but still fails the predeclared accuracy allowance: the reported anchored-to-full W1 upper bound is (0.006076), above (0.002), despite a favorable time ratio (`manuscript/dual_results.tex:30`). The paper itself correctly says that finite-domain restoration alone does not solve accuracy (`manuscript/experiments_paper.tex:17`).

The sensor statistics file gives a controlled benchmark with an exact reference, but it is marked as requiring a separate independent numerical audit and includes observational timing status (`work/paper-sensor-confirmation/statistics.json:10184-10194`). The new anchor audit is explicitly still in progress for this review. Consequently, no current positive statement about anchor accuracy, runtime, calibration, or application value is closed evidence.

**Required action:** keep the application conclusion negative or conditional until the audit is complete. Present the scientific value as a validity diagnostic and failure detector. If the anchor audit later passes, report its exact cohort, selection protocol, raw-source receipts, and paired confidence intervals; do not convert a favorable benchmark result into a claim of real sensor or deployment improvement.

### 4. Provenance and cohort boundaries are a paper-level acceptance risk even where the mathematics is sound

The historical oracle baseline has a documented recovery/audit protocol, including source and configuration checks, while also warning that retained one-step particles are insufficient for complete raw recomputation (`docs/COMPOSITION_REPRODUCIBILITY.md:25-48`). The document explicitly limits the original theorem to the specified Euler operator, finite with-replacement batches, and scalar or diagonal product tails, and disclaims universal accuracy, coverage, and wall-clock gains (`docs/COMPOSITION_REPRODUCIBILITY.md:90`).

The newer learned, dual, matrix, sensor, and anchor artifacts have different cohorts and different status levels. Mixing the original 1,980-cell oracle matrix, extension cells, learned cells, preliminary sensor aggregates, and in-progress anchor audit in one “experimental validation” narrative would make the evidence look larger than the independently closed set. The saved sensor summary itself reports 480 cells and a separate-audit requirement, so it cannot silently serve as a completed replacement for raw-data verification.

**Required action:** add a compact evidence table in the paper or supplement with one row per study: operator, cohort, reference, method, completed audit status, and claim supported. Preserve source/archive hashes and distinguish “saved aggregate,” “independently recomputed,” and “unverified.” Do not report a new numerical audit as completed until its raw inputs, source hash, checkpoint reload, reference calculation, and all completion markers are checked.

### 5. Mixture and resampling qualifications remain easy places for a reviewer to find an overclaim

The tail-preservation theorem is for equal-component-variance Gaussian mixtures and common quadratic coefficients. The main paper correctly warns that fitting a Gaussian using total mixture variance generally does not preserve the bounded-residual property (`manuscript/main.tex:150-155`). The matrix extension likewise concerns shared covariance tails, not arbitrary correlated mixtures (`manuscript/main.tex:183-187`). A finite positive Gaussian mixture proposal also requires the complete mixture-density denominator; a selected-component density ratio is not a valid mixture importance weight. This qualification is present in the Gaussian twist sidecar, yet it is not currently a central limitation in the main narrative.

Separately, the Poisson/kernel correction result is a conditional unnormalized-kernel identity. It does not establish unbiased normalized particle estimates or a resampling variance bound (`manuscript/dual_results.tex:5`). The original random-batch kernel and the candidate positive Poisson kernel must remain distinct: a pointwise Poisson expectation identity does not retroactively make the original finite random-batch exponential unbiased, and finite conditional kernel preservation does not imply global SMC stability.

**Required action:** state three separate objects wherever methods are compared: (i) unnormalized kernel or normalizer existence, (ii) independent-path importance-sampling moments, and (iii) normalized/resampled particle error. Give the mixture denominator and component-weight qualification in the method definition, and restrict claims to the certified common-tail class. Do not label the candidate correction “SMC-stable” without a separate theorem and evidence.

## True value versus benchmark-only evidence

The exact Gaussian integrals, exact mixture posterior/reference calculations, and eigenvalue certificates are mathematical truth for the explicitly defined synthetic operator and discretization. They establish properties such as finite versus infinite population normalizer or path (p)-moment under stated assumptions. They do not establish that the discretization equals the continuous posterior (`manuscript/main.tex:61`), nor that a learned score is correct.

W1, state MSE, interval coverage, runtime, ESS, and the saved sensor summary are finite-cohort benchmark measurements. They are useful evidence about the tested synthetic tasks. They are not deployment value, population calibration, universal speedup, or proof of joint accuracy. The sensor study has an exact reference and should be called an exact-reference benchmark, not measured-sensor evidence. The current report therefore makes no positive claim that depends on new anchor numbers.

## Bottom line for revision

The strongest paper is a narrowly scoped theory-and-diagnostics paper: exponential factor subsampling can fail at the population-normalizer level despite local unbiasedness and vanishing local MSE; exact finite-horizon certificates can detect that failure; deterministic common tail curvature can preserve strict integrability in the stated scalar/diagonal and shared-covariance classes; finite-particle accuracy and computational benefit remain separate empirical questions. The five risks above should be resolved in that order before presenting any new anchor result as support for application value.
