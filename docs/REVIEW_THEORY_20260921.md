# AI-assisted internal working assessment

**Paper:** *Integrability of Subsampled Compositional Diffusions*  
**Date:** 2026-09-21  
**Scope:** internal scientific review of the author’s ICLR research draft; this is not a formal conference review and does not claim independent human review.  
**Review rule:** each issue is recorded as location, observation, evidence, importance, and action. No acceptance probability is assigned.

## Assessment boundary

The review read the working copy of `manuscript/main.tex`, `manuscript/appendix.tex`, `docs/SOLID_RESULTS_20260920.md`, the local test and audit entry points, and the read-only candidate implementation `/Users/zenghang/Documents/Codex/2026-09-12/wi/work/certified-composition/certified_composition.py`. The report was written only to this file. No manuscript, result, source, remote service, or server state was modified; no manuscript material was uploaded.

The manuscript itself defines a narrow object: a specified finite-step weighted Euler/Feynman–Kac operator. It explicitly says that the finite-step target need not equal the continuous path (`manuscript/main.tex:51-61`) and separates population integrability, finite-particle behavior, and learning error (`manuscript/main.tex:34-38`, `185-191`). That scope is scientifically defensible and should remain the central claim boundary.

## Overall scientific assessment

The strongest contribution is a focused validity certificate for a particular random-batch, score-composition Feynman–Kac Euler operator. The potentially distinctive mechanism is the combination of: (i) conditionally unbiased drift and potential estimators, (ii) exponentiation into a nonnegative weight, (iii) positive-probability minibatch histories, and (iv) an exact reachable-tail-variance recursion that can make the population normalizer infinite while every finite particle realization remains numerically finite. The local prior-art record supports this positioning: random-weight particle filters, unbounded-weight moment conditions, Feynman–Kac stability, and Gaussian quadratic-exponential integrals are established ingredients, while the claimed application-specific rare-history certificate is plausibly distinct (`docs/INTEGRABILITY_PRIOR_ART.md:13-21`, `23-65`).

The paper does not yet justify a broad claim that certified tail control is accurate, faster at matched accuracy, or valid for arbitrary learned scores. The current evidence supports an existence/validity result and a diagnostic methodology. It does not support an application-level superiority result. The supplemental results themselves report tail control W1 of `0.09859` versus `0.01817` for full-factor computation, with about `1/4.14` of the time and about `5.43` times the error, and explicitly state that this is not same-accuracy speed evidence (`docs/SOLID_RESULTS_20260920.md:13-25`).

## 1. Novelty and value of the theorem conditions

### Location

`manuscript/main.tex:80-134`, `111-129`, `149-180`; `manuscript/appendix.tex:56-100`, `128-183`.

### Observation

The Gaussian theorem and the tail-control theorem are mathematically meaningful under their stated restrictions. The theorem identifies a population-level existence condition before particle approximation: the one-step denominator is `D_B(V)=1-2h c_B V`, and a nonpositive denominator yields an infinite Gaussian integral (`manuscript/main.tex:94-107`). The extreme-batch recursion then reduces all with-replacement finite batch sizes to two attainable constant batches (`manuscript/main.tex:111-129`). The without-replacement extension has an explicit batch-size-dependent prefix/suffix recursion (`manuscript/main.tex:133-134`; full statement and proof `manuscript/appendix.tex:143-175`).

The value is strongest as a preconditioner for deciding whether a specified population operator exists. It is weaker as a general theory of compositional diffusion, SMC, or random-weight inference. The manuscript correctly limits the setting to scalar/diagonal product tails, fixed weighted Euler construction, and a shared-variance Gaussian-mixture class (`manuscript/main.tex:185-191`).

### Evidence

The exact four-factor example is an effective discriminator: the full update has `D_full=2503/3195>0`, while the all-strongest with-replacement batch has `D=-167/1065<0`, with probability `4^{-M}` for every finite `M` (`manuscript/main.tex:136-147`). The supplemental study adds threshold-side experiments, reporting that the certificate classifies the population operator while finite-particle W1 changes direction across families (`docs/SOLID_RESULTS_20260920.md:27-42`). This is the right separation of theorem evidence from particle evidence.

Public primary literature supplies the surrounding contract. Fearnhead et al. study positive random weights for continuous-time particle filtering, which supports treating positivity as a required construction property, but does not establish finite population integrability for this minibatch score-composition operator ([JRSS B 2010](https://doi.org/10.1111/j.1467-9868.2010.00744.x)). Mbalawata and Särkkä derive second- and fourth-moment conditions for convergence with unbounded importance weights, which supports the need for moment assumptions rather than a claim that unbounded weights alone are novel ([Signal Processing 2016](https://doi.org/10.1016/j.sigpro.2015.06.018)). Whiteley’s stability results rely on multiplicative-drift and exponential-moment conditions, so they are relevant after the underlying Feynman–Kac flow is well-defined, not a substitute for this certificate ([Annals of Applied Probability 2013](https://doi.org/10.1214/12-AAP878)).

### Importance

High. This is the paper’s clearest novelty boundary and the main place where an overstated title or abstract would damage the scientific contribution. The theorem conditions are useful because they are operational and falsifiable, not because they cover all compositional diffusion systems.

### Action

Keep the contribution phrased as an exact or certified integrability analysis for the stated random-batch weighted Euler operator. State explicitly that the new part is the reachable-history certificate in the score-derived compositional setting. Avoid general claims about random-weight SMC, arbitrary neural scores, non-diagonal covariance, adaptive sampling, or continuous-time validity unless separate theorems are supplied.

## 2. Kernel unbiasedness versus potential unbiasedness

### Location

`manuscript/main.tex:63-78`; `manuscript/appendix.tex:102-126`.

### Observation

The manuscript’s original random-batch kernel and the candidate’s positive Poisson correction are different constructions and must be analyzed separately.

For the manuscript’s random batch, the potential estimator is conditionally unbiased at fixed `(x,u)`, but the kernel is an expectation of an exponential multiplied by a state transition (`manuscript/main.tex:63-78`). In general,

\[
\mathbb E_B[\exp(h\widehat g_B)]\neq \exp(hg),
\]

and unbiasedness of the drift and potential separately does not imply unbiasedness of the random-batch kernel, of its normalized update, or of the target distribution. The manuscript’s local expansion makes the missing terms explicit: drift variance, drift–potential covariance, and the population covariance of the potential variance enter at order `h^2` (`manuscript/appendix.tex:105-115`). Paired weighting cancels the second-order potential-variance term under its local assumptions, but leaves drift variance and drift–potential covariance (`manuscript/appendix.tex:117-126`).

The candidate uses a separate positive Poisson product estimator for the residual exponential correction. The author-side contract note now fixes the exact candidate object and its restrictions (`docs/KERNEL_CORRECTION_CONTRACT_20260921.md:5-23`). Conditional on `(x,y)`, let `I_1,\ldots,I_N` be Poisson marks with `N\sim\operatorname{Poisson}(\lambda)` and `\Pr(I_j=i)=p_i`, and define

\[
Z=\prod_{j=1}^{N}\left(1+\frac{z_{I_j}}{\lambda p_{I_j}}\right).
\]

For a finite index set and finite `\lambda`, the compound-Poisson generating identity gives

\[
\mathbb E[Z\mid x,y]
=\exp\!\left(\lambda\sum_i p_i\left[\frac{z_i}{\lambda p_i}\right]\right)
=\exp\!\left(\sum_i z_i\right).
\]

Equivalently, conditioning on `N=n` gives the `n`th power of the marked-factor mean, and summing the Poisson series gives the exponential. The candidate’s positivity contract checks every realized factor `1+z_i/(\lambda p_i)>0` (`certified_composition.py:185-210`). The contract note also gives the exact conditional relative second moment and the bound `\exp(\eta)` under `\lambda\ge B^2/\eta` (`docs/KERNEL_CORRECTION_CONTRACT_20260921.md:11-23`). This establishes conditional kernel preservation for the specified candidate correction, with finite conditional first and second moments under the stated finite-`G`, finite-`x,y`, positive-finite-`\eta` restrictions. It must not be classified together with the manuscript’s raw random-batch exponential kernel.

For a finite horizon without resampling, conditioning on the complete proposal path and using independent auxiliary events at each step gives the path-weight identities in the contract note (`docs/KERNEL_CORRECTION_CONTRACT_20260921.md:25-34`): the conditional mean equals the exact full-Euler importance weight, and the conditional relative second moment is bounded by `\exp(\sum_k\eta_k)`. Tonelli then gives equality of the nonnegative unnormalized endpoint measure for `f\ge0`, provided the full-Euler path weight is the intended one and the corresponding integral is defined. The global second-moment conclusion still requires the additional assumption `\mathbb E[W^2]<\infty`; a first-moment certificate or a passing power strictly between one and two does not supply it. This preserves the finite-horizon unnormalized population operator, including the possibility that its total mass is infinite. It does not prove that the finite-`N` normalized particle empirical measure is unbiased: self-normalization introduces ratio bias, even when the unnormalized estimator is unbiased. The candidate’s adaptive systematic resampling retains the conditional one-step identities, but the no-resampling path bound does not establish resampling stability, genealogical moment bounds, normalized finite-`N` error rates, or a global SMC theorem (`docs/KERNEL_CORRECTION_CONTRACT_20260921.md:25-38`).

### Evidence

The main text already states the correct non-implication (`manuscript/main.tex:78`). The supplemental results provide a concrete empirical counterpart: tail control passes the population finite certificate in `100/100` cells, but has substantially worse W1 than the full-factor method (`docs/SOLID_RESULTS_20260920.md:13-25`). This shows that preserving an integrability domain is not equivalent to preserving sampling accuracy.

The Feynman–Kac and particle-filter literature also treats normalizing constants and moment conditions as separate objects. Whiteley’s primary result ties particle stability and normalizing-constant variance to exponential-moment and drift conditions ([Whiteley 2013](https://doi.org/10.1214/12-AAP878)); Mbalawata and Särkkä explicitly formulate moment conditions for convergence with unbounded weights ([Mbalawata and Särkkä 2016](https://doi.org/10.1016/j.sigpro.2015.06.018)). These sources support the paper’s distinction, while they do not prove the manuscript’s specific recursion.

### Importance

Critical. A reader could otherwise conflate the manuscript’s potential-unbiased batch estimator with the candidate’s conditionally kernel-preserving Poisson correction, or infer resampling stability from the no-resampling path identity. The candidate’s conditional kernel contract is now established under its stated restrictions; the remaining issue is the population-to-particle and no-resampling-to-resampling bridge.

### Action

Keep three separate claim boundaries near `eq:kernel` and the candidate description. For the manuscript batch estimator, conditional unbiasedness gives `E[\widehat b_B|x]=b` and `E[\widehat g_B|x]=g`; it does not give `E[e^{h\widehat g_B}f(Y_B)|x]=e^{hg}P_hf(x)`. For the candidate, the positive Poisson product preserves the specified one-step unnormalized kernel pointwise and, after no-resampling path conditioning, preserves the finite-horizon unnormalized population measure under the contract assumptions. The report should state separately the required global assumption `E[W^2]<\infty`, normalized finite-particle ratio bias, and the absence of a resampling-stability or finite-`N` error theorem.

## 3. Does `moment_certificate` support a second-moment or full random-correction guarantee?

### Location

Candidate implementation `/Users/zenghang/Documents/Codex/2026-09-12/wi/work/certified-composition/certified_composition.py:221-261`, `166-218`, `271-328`; manuscript discussion `manuscript/main.tex:167-180`; local test `tests/test_certified_composition.py` in the candidate checkout.

### Observation

`moment_certificate(parameters, grid, power=1.0, margin=0.0)` propagates a scalar/vector Gaussian quadratic-tail variance. It computes a denominator

\[
D_k=1-2\,p\,h_k\,c_k V_k
\]

and updates `V_k` through the deterministic quadratic-tail recursion (`certified_composition.py:221-236`). The `power` argument is used by multiplying the potential curvature by `power`; `prepare_grid` searches for a numerically feasible power above one and returns a grid plus a margin-based certificate (`certified_composition.py:239-261`). This is evidence for a fixed-grid Gaussian tail-envelope moment of the reference path under the stated affine-plus-bounded proposal class.

The candidate Poisson product has the exact first-moment identity derived above. Its relative second moment is controlled locally by the squared residual terms; for a fixed Poisson event construction, the exact conditional factor is

\[
\frac{\mathbb E[Z^2\mid x,y]}{\mathbb E[Z\mid x,y]^2}
=\exp\!\left(\sum_i\frac{z_i^2}{\lambda p_i}\right),
\]

when all factors and sums are finite. The contract note establishes the implementation-level bound `\exp(\eta)` from the envelope construction (`docs/KERNEL_CORRECTION_CONTRACT_20260921.md:11-23`). Along a no-resampling path, these conditional bounds multiply, but finiteness of the resulting unconditional second moment still requires `\mathbb E[W^2]<\infty` (`docs/KERNEL_CORRECTION_CONTRACT_20260921.md:25-34`). `sample` also uses table interpolation and an adaptive full-versus-randomized branch (`certified_composition.py:62-80`, `166-218`, `283-328`); the no-correction interpolant ablation is explicitly outside the pointwise identity.

The code contains a useful local runtime guard: it checks the conditional budget and raises if the computed inflation exceeds `eta` (`certified_composition.py:211-218`). The contract note now supplies the missing conditional Poisson-product derivation and the no-resampling path inequality. The remaining global requirement is explicit: `\mathbb E[W^2]<\infty`. The `power` parameter is a Gaussian-tail diagnostic for the deterministic full-Euler path; it should not be presented as a predictor that `power>1` implies the `power=2` certificate or as a substitute for the separate full-path assumption.

### Evidence

The manuscript is appropriately cautious for the tail-control theorem: it says that the theorem certifies existence of the discretized population flow and does not remove Euler error or the effect of exponentiating the bounded residual estimator (`manuscript/main.tex:167-180`). The new contract note supplies the candidate-specific pointwise factorization, conditional first and second moments, and no-resampling path conditioning (`docs/KERNEL_CORRECTION_CONTRACT_20260921.md:11-34`). It expressly leaves resampling stability, normalized finite-particle error, and a finite-`N` rate outside the claim (`docs/KERNEL_CORRECTION_CONTRACT_20260921.md:34-38`).

The supplemental study reports finite/infinite population certificates and finite-particle results, while the new contract note narrows the remaining mathematical gap to the full-Euler path moment and resampling layer (`docs/SOLID_RESULTS_20260920.md:44-72`; `docs/KERNEL_CORRECTION_CONTRACT_20260921.md:25-38`). The candidate unit tests verify the local positive-Poisson mean and second-moment identities numerically, together with exact kernel-mass and first two state-moment checks (`tests/test_certified_composition.py:58-80`, `137-152`). The newly added CPU test file is reported as 29 tests passed; GPU execution remains in progress and is not treated as completed evidence. These tests verify implementation behavior and do not create a resampling-stability theorem.

As a concrete boundary test, using the specified interpreter, `FactorParameters` with four one-dimensional zero-mean Gaussian factors and variances
`[0.75110836, 0.27855919, 0.48470856, 0.33205171]` on `grid=np.linspace(2,0,12)` gives:

| `power` | certificate result | evidence |
|---:|---|---|
| `1.0` | passed | minimum denominator `0.5173211952` |
| `1.5` | passed | minimum denominator `0.1195211830`; terminal tail variance `0.3170662543` |
| `2.0` | failed | first failure at step `10`; minimum denominator `-2.6983889315` |

This is a real boundary counterexample for the deterministic Gaussian-tail recursion. It shows that a first-moment/existence pass can coexist with failure at power two. It does not support an algorithmic rule that `power>1` predicts a finite second moment; the contract note explicitly requires the separate full-Euler assumption `\mathbb E[W^2]<\infty`.

### Importance

Critical if the method is described as “certified” in the sense of finite variance, unbiased random correction, or full-path Monte Carlo guarantees. Moderate if “certificate” is restricted to the stated Gaussian quadratic-tail population integrability test.

### Action

Use a four-level vocabulary:

1. **Certified by the current recursion:** strict Gaussian or tail-envelope population integrability for the specified fixed grid and control class.
2. **Conditionally established for the candidate:** pointwise positive Poisson correction preserves the specified one-step unnormalized kernel; finite-horizon no-resampling path conditioning preserves the unnormalized population measure under the contract assumptions.
3. **Runtime-checked:** realized conditional Poisson-factor budget and positivity assertions for sampled particles.
4. **Unproved unless added:** finiteness of the global second moment without an independent `E[W^2]<\infty` assumption, resampling-stable variance bounds, normalized finite-particle error rates, and end-to-end SMC convergence.

The candidate-specific conditional contract is already recorded in `docs/KERNEL_CORRECTION_CONTRACT_20260921.md`. Any stronger claim must add or cite a result that verifies `\mathbb E[W^2]<\infty` for the full-Euler path and separately analyzes adaptive systematic resampling. The current `moment_certificate` should not be used as a replacement for either condition.

## 4. Main theoretical risks

### Risk A: theorem scope may be read more broadly than its hypotheses

**Location:** `manuscript/main.tex:111-134`, `167-191`.  
**Observation:** The exact recursion is scalar or coordinatewise diagonal and assumes nonnegative residual precisions, fixed finite grids, fixed batch rules, and strict denominators.  
**Evidence:** The manuscript lists noncommuting covariances, adaptive batches, and broader learned models as outside scope (`manuscript/main.tex:185-191`); equality at the mixture-envelope boundary is explicitly left unresolved (`manuscript/main.tex:167-173`; `manuscript/appendix.tex:21-54`).  
**Importance:** High. These are structural hypotheses, not implementation details.  
**Action:** Repeat the hypothesis list in the abstract/contribution paragraph and label zero-denominator mixture cases as undecided rather than finite or divergent.

### Risk B: “strict integrability domain agrees with full-factor” is exact only for the stated control class

**Location:** `manuscript/main.tex:149-180`; `manuscript/appendix.tex:56-100`.  
**Observation:** The shared component-variance mixture argument depends on an affine part with the exact deterministic quadratic tail and a uniformly bounded residual.  
**Evidence:** The manuscript derives `r_g=r_g^0+e_g` with bounded `e_g` only for finite mixtures with shared component variance (`manuscript/main.tex:149-165`). It separately warns that a Gaussian fit using total mixture variance generally lacks this property (`manuscript/main.tex:155`).  
**Importance:** High for learned-score interpretation.  
**Action:** State that learned models need an independently verified tail representation; finite neural outputs or finite particle values do not establish the bounded-residual hypothesis.

### Risk C: empirical finite-particle behavior cannot validate an undefined population operator

**Location:** `manuscript/main.tex:136-147`, `182-191`; `docs/SOLID_RESULTS_20260920.md:27-42`, `62-80`.  
**Observation:** A finite particle run can miss a rare divergent history.  
**Evidence:** The four-factor history has probability `4^{-M}` per sampled batch, and the manuscript explicitly separates this from finite-particle observation (`manuscript/main.tex:143-147`). The supplement retains finite-particle results for infinite certificates (`docs/SOLID_RESULTS_20260920.md:62-68`).  
**Importance:** High.  
**Action:** Place the certificate status before ESS, W1, and runtime in every benchmark table; never use no-NaN output as existence evidence.

## 5. Most important falsifiable experiments

These are ordered by scientific information, not by convenience. Each should preserve the certificate-before-particle-analysis protocol.

### Experiment 1: matched-accuracy cost frontier

**Location:** motivation in `manuscript/main.tex:31-38`; current evidence `docs/SOLID_RESULTS_20260920.md:13-25`, `76-80`.  
**Falsifiable question:** At a pre-specified W1 or task-loss tolerance, does tail control reduce total cost relative to full-factor and naive without-replacement baselines?  
**Protocol:** Use identical factor evaluations, particle count search, time grid, precision, resampling rule, and reference target. Report cost including preparation, factor evaluation, weighting, resampling, and any increased particle count needed to meet the tolerance. Use paired seeds and report the cost–error frontier, not one unconstrained runtime ratio.  
**Falsifier:** If the controlled method cannot reach the full-factor error within the allowed cost, or requires more cost at every common accuracy level, its value remains validity protection rather than acceleration.

### Experiment 2: full-Euler second moment and resampling stability

**Location:** candidate contract `docs/KERNEL_CORRECTION_CONTRACT_20260921.md:25-38`; implementation `certified_composition.py:166-218`, `271-328`.  
**Falsifiable question:** Under an independently verified finite `\mathbb E[W^2]` condition for the full-Euler path, does the candidate’s local Poisson bound produce the predicted no-resampling path second-moment bound, and what changes after adaptive systematic resampling?  
**Protocol:** For small finite grids, compute or tightly verify the full-Euler path second moment `\mathbb E[W^2]`, then compare it with the random-correction estimate under independent auxiliary streams. Report the conditional relative second-moment product `\exp(\sum_k\eta_k)`, the unconditional second moment, and the ratio between them. Run the same configurations with adaptive systematic resampling and report normalized endpoint bias, variance, ESS, genealogical coalescence, and particle-count scaling. Keep the no-resampling identity and the resampling experiment as separate analyses.  
**Falsifier:** A failure of the Poisson-product identity or its local second-moment bound would challenge the candidate contract. A stable no-resampling result together with resampling-dependent bias or variance growth would establish the contract’s intended boundary, not refute it. The existing `power=1.5` versus `power=2` Gaussian counterexample remains a tail-recursion boundary test and is not a predictor claim for the candidate’s global second moment.

### Experiment 3: tail misspecification boundary and adversarial learned tails

**Location:** `manuscript/main.tex:149-180`; supplement `docs/SOLID_RESULTS_20260920.md:44-60`.  
**Falsifiable question:** How sensitive are finite certificates and accuracy to violations of the bounded-residual/shared-variance assumption?  
**Protocol:** Keep the oracle target fixed and introduce controlled tail slope errors, unequal component variances, correlated/rotated covariance, and learned score perturbations that agree on a central interval but differ in the tails. Pre-register certificate status, divergence/equality status, W1, and correction-factor diagnostics. Include zero-denominator cases separately.  
**Falsifier:** A small tail perturbation that passes the current certificate but produces divergent or unstable correction moments shows that the present robustness statement is too broad; a failure outside the hypothesis class is expected and should be reported as scope evidence.

### Experiment 4: rare-history detection and finite-particle deception

**Location:** `manuscript/main.tex:136-147`; `docs/SOLID_RESULTS_20260920.md:27-42`.  
**Falsifiable question:** Can the certificate predict a population failure when finite particle outputs remain finite and apparently accurate?  
**Protocol:** Use the four-factor analytic example and vary minibatch size, particle count, and number of independent runs. Compute the exact bad-history probability, observed bad-history rate, maximum log weight, ESS collapse, and finite-sample W1. Compare naive exponential weighting with a certified control.  
**Falsifier:** Failure to reproduce the exact denominator boundary, or a systematic disagreement between exact bad-history probability and observed batch-history frequencies, indicates an implementation or theorem-to-code mismatch.

### Experiment 5: anisotropic non-diagonal stress test with a clearly bounded claim

**Location:** limitation `manuscript/main.tex:185-191`; diagonal extension `manuscript/appendix.tex:177`.  
**Falsifiable question:** Does the coordinatewise certificate remain conservative or fail under rotated covariance?  
**Protocol:** Construct low-dimensional Gaussian factors with known non-diagonal covariance and compare numerical eigenvalue-based quadratic integrability against the coordinatewise certificate after rotation. This is a diagnostic experiment, not evidence for a theorem outside the current scope.  
**Falsifier:** If rotation changes the true boundary while the coordinatewise test reports the same result, the current coordinatewise extension cannot be presented as a general multivariate certificate.

## 6. Checked versus unchecked material

### Checked in this review

- Read `Agent.md` and the complete `math-paper-writing` skill plus required mathematical-exposition and numerical-methods references.
- Read the main manuscript and appendix theorem/proof passages, including the FK identity, estimator definitions, Gaussian recursion, mixture tail theorem, paired correction, and without-replacement theorem.
- Read `docs/SOLID_RESULTS_20260920.md`, including protocol, certificate counts, W1/runtime results, threshold experiments, tail perturbations, audit and recovery statements.
- Read the candidate implementation’s `FactorParameters`, `TimeFactors`, `random_correction`, `moment_certificate`, `prepare_grid`, and `sample` paths with exact line references.
- Read the author-side mathematical contract `docs/KERNEL_CORRECTION_CONTRACT_20260921.md`, including the pointwise Poisson mean, conditional second moment, no-resampling path conditioning, explicit `E[W^2]<\infty` requirement, and resampling limitation.
- Read the local prior-art audit `docs/INTEGRABILITY_PRIOR_ART.md` and cross-checked the relevant public primary sources: Fearnhead et al. (2010), Mbalawata and Särkkä (2016), and Whiteley (2013).
- Ran the candidate test and the本文 test suite with `/Users/zenghang/Documents/Codex/2026-09-09/https-github-com-freezeng123456-iclr-2027/work/venv/bin/python`: `39 passed, 6 warnings`.
- Ran a direct Gaussian boundary check with the same interpreter: `power=1` and `power=1.5` passed, while `power=2` failed at step `10` for variances `[0.75110836, 0.27855919, 0.48470856, 0.33205171]` on `np.linspace(2,0,12)`.
- Recorded the author-reported status of `tests/test_random_kernel_moments_20260921.py`: 29 CPU tests passed; GPU execution was still in progress at review time and is not counted as completed evidence.

### Not checked or not independently established

- No independent re-derivation of every appendix algebraic identity or complete proof reconstruction.
- No fresh execution of the H20 experiments, no server access, and no remote artifact verification.
- No independent human review; this is explicitly AI-assisted internal working assessment.
- No independent verification of the author-side contract note beyond reading it and the already completed CPU tests; in particular, no independent proof that the full-Euler path satisfies `E[W^2]<\infty` for the reported learned configurations.
- No resampling-stability theorem, normalized finite-particle error rate, or global SMC convergence result.
- No validation for arbitrary neural scores, non-diagonal covariances, adaptive batches, continuous-time limits, or zero-denominator mixture cases.
- No same-accuracy cost frontier has been completed; the existing runtime/error comparison is intentionally insufficient for that claim.

## Most important risks for the author

1. The paper’s most defensible contribution is narrower than “certified compositional diffusion”: it is a population-integrability certificate for a specified random-batch weighted Euler/Feynman–Kac operator.
2. The manuscript’s raw random-batch potential estimator is distinct from the candidate’s positive Poisson correction. The candidate’s pointwise conditional kernel-preserving identity is established under the new contract; normalized finite-particle bias, the independent `E[W^2]<\infty` requirement, and resampling stability remain separate questions.
3. `moment_certificate` is a tail-recursion diagnostic and cannot replace the full-Euler second-moment assumption. The real `power=1.5`/`power=2` counterexample should remain as a boundary test, without interpreting `power>1` as a predictor of `p=2` validity.
4. Tail control has strong validity evidence but materially worse current accuracy. The paper should present this as a validity–accuracy–cost tradeoff until a matched-accuracy experiment says otherwise.
5. The finite-particle results are useful precisely because they can look finite when the population operator is undefined. They cannot be used to promote a failed certificate.

## Recommended claim status

**Supported:** exact one-step Gaussian integrability criterion; reachable-variance extreme-batch recursion under stated hypotheses; prefix/suffix recursion for the stated without-replacement Gaussian operator; shared-component-variance tail-control integrability result under strict nonzero denominators; empirical separation of certificate status, finite-particle error, and learning error; candidate pointwise positive-Poisson correction identity and conditional first/second-moment contract under the note’s restrictions; finite-horizon no-resampling unnormalized path identity conditional on the explicit `E[W^2]<\infty` requirement for the second-moment bound.

**Conditionally supported:** application-specific novelty of the rare-history certificate, subject to precise comparison with surrounding random-weight/Feynman–Kac literature and strict scope language.

**Unverified or unsupported at present:** independent verification of `E[W^2]<\infty` for the full-Euler path on the learned benchmark configurations; resampling stability; normalized finite-particle unbiasedness or finite-`N` error rates; arbitrary learned-score validity; same-accuracy speedup; general multivariate/non-diagonal certificate; universal accuracy improvement from tail control.
