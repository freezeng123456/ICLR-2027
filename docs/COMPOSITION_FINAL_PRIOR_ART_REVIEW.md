# Final prior-art review: constant-batch integrability certificates

**Date:** 2026-09-09
**Scope:** bounded primary-source check of the manuscript's final claim: for an i.i.d. with-replacement factor batch, an unbiased U-statistic score-derived quadratic FK potential has its full finite-history Gaussian integrability decision determined by the two extreme constant batches at each step, independently of every finite batch size (M); affine controls extend this to (G) constant batches; tail-matched controls and paired cumulant correction preserve the full strict domain.

This review covers direct compositional diffusion/SBI papers, random-weight/subsampled SMC, particle-FK moment/stability results, and Gaussian quadratic-exponential integrability. It does not claim an exhaustive search of all numerical-analysis, convex-analysis, or SMC literature.

## Judgment

The manuscript's `main.tex` and `appendix.tex` state a specialized theorem for the fixed weighted Euler construction. A batch history remains Gaussian and the next variance has the form

\[
V_{k+1}=F_B(V_k)=\frac{V_k[1-h((1+\kappa)A_B+\kappa)/2]^2}{1-2h c_BV_k}+\kappa h.
\]

The denominator is the Gaussian exponential-quadratic integrability condition. Since every with-replacement batch has positive probability, a bad history makes the nonnegative population FK operator divergent by Tonelli. The appendix reduces maximization over the batch-frequency simplex to vertices using convexity/quadratic perspectives. Vertices are exactly constant batches, so two extrema in the no-control scalar theorem, and (G) constant batches for arbitrary affine controls, suffice. The paired correction has the same strict domain because its negative squared-difference term can only lower the weight, while the maximizing identical-batch history makes that correction vanish.

The important point is that this is a **reachable-integrability theorem for a particular random weighted Euler operator**, not a claim that U-statistics, Gaussian integration, convexity, or random-weight SMC are new individually.

## Search result

I found no primary paper that states an equivalent theorem combining all of these elements:

1. i.i.d. with-replacement minibatching over score factors;
2. an unbiased pairwise/U-statistic quadratic composition potential;
3. a Gaussian weighted-Euler variance recursion;
4. maximization over all finite batch histories via constant-batch vertices of the frequency simplex;
5. independence of the integrability decision from every finite (M);
6. extension to affine tail controls and paired cumulant correction; and
7. application to compositional VP diffusion/SBI.

This is a bounded negative result. It supports “we are unaware of prior work that gives this certificate for this operator and application,” not “the first theorem of this kind in probability/SMC.” General convexity facts and Gaussian moment conditions make the proof ingredients familiar; the potentially distinct object is their assembly into a batch-size-independent certificate for compositional score-derived FK weighting.

## Six direct sources

### 1. Linhart et al., tall-data diffusion posterior sampling

The TMLR paper develops GAUSS/JAC-style score composition for tall-data SBI and analyzes numerical stability and cost ([TMLR 2026 record](https://openreview.net/pdf/332aa5cc412cd42f0920ed8b48800991b60b50f8.pdf), [arXiv](https://arxiv.org/abs/2404.07593)). It does not study random factor minibatches inside a positive FK weight, reachable Gaussian variances, or a constant-batch simplex certificate. It is the direct method baseline for the application, not an equivalent integrability theorem.

### 2. Arruda et al., compositional amortized inference

Arruda et al. propose compositional score matching with adaptive solvers and an error-damping estimator for large hierarchical problems ([official ICLR 2026 page](https://proceedings.iclr.cc/paper_files/paper/2026/hash/54c306b0c044e2dc92124b6689a9574a-Abstract-Conference.html), [OpenReview PDF](https://openreview.net/pdf/N3XCVHZGW5)). Their stability issue is related to component accumulation, but they do not exponentiate a random U-statistic potential in an FK population operator or prove finite-history integrability by constant batches.

### 3. Fearnhead et al., random-weight particle filtering

Fearnhead et al. explicitly study random positive particle weights and use a martingale/Wald construction to ensure positivity ([JRSS B](https://doi.org/10.1111/j.1467-9868.2010.00744.x)). This establishes a close random-weight precedent and is why the manuscript must distinguish positivity from integrability. Their construction does not give the proposed rare-history Gaussian divergence certificate, batch-size-independent vertex reduction, or compositional score application.

### 4. Gunawan et al., subsampling SMC

Gunawan et al. use data subsampling in annealed SMC with approximately unbiased likelihood estimates and pseudo-marginal auxiliary updates ([Statistics and Computing 2020](https://doi.org/10.1007/s11222-020-09969-z), [arXiv version](https://arxiv.org/abs/1805.03317)). This is the closest subsampling-SMC baseline. It does not establish that a positive-probability minibatch history can make an exponentiated score-composition FK operator's population normalizer infinite while the full-batch discretization remains finite.

### 5. Whiteley, particle-filter stability

Whiteley proves particle stability and normalizing-constant variance bounds under multiplicative-drift and exponential-moment conditions ([Annals of Applied Probability](https://doi.org/10.1214/12-AAP878)). This supports treating the manuscript's certificate as a precondition for invoking standard FK stability, but it does not provide a simplex reachable-variance reduction or a random score-derived potential analysis.

### 6. Gaussian quadratic-exponential moments

The finite-dimensional Gaussian integral criterion is classical: the precision after adding a quadratic exponential must remain positive definite; equality and the wrong sign require separate treatment. A standard multivariate Gaussian reference is [Chapter 3 of the Springer multivariate Gaussian text](https://doi.org/10.1007/978-3-030-95864-0_3). This supports the denominator calculation but does not address random batch histories, U-statistics, affine controls, or FK propagation.

## Priority wording recommended for the paper

Use a bounded claim such as:

> To the best of our knowledge, this is the first analysis of the finite-history integrability of a with-replacement, score-derived quadratic Feynman–Kac Euler operator in compositional diffusion inference. For the scalar Gaussian model, we prove that the population integrability decision for every finite batch size is determined by two extreme constant batches per time step; we extend the certificate to arbitrary affine tail controls and show strict-domain preservation for the stated tail-matched and paired constructions.

Avoid “first analysis of random-weight SMC,” “first use of U-statistics in score composition,” “first Gaussian exponential-integrability criterion,” “all subsampling methods are invalid,” and “unbiasedness guarantees a valid FK operator.” Scope the theorem by the fixed weighted Euler discretization, scalar/diagonal Gaussian or stated Gaussian-mixture tail assumptions, i.i.d. with-replacement batches, finite time grid, and strict denominator domain. Equality at the denominator boundary must remain separately qualified exactly as in the appendix.

## References.bib audit

`gunawan2020` is **Statistics and Computing 30(6), 1741–1758 (2020)**, DOI `10.1007/s11222-020-09969-z`. The bibliography uses this journal metadata and retains the arXiv URL as an access link.

`fearnhead2010`, `mbalawata2016`, and `whiteley2013` have materially correct venue/DOI metadata. `linhart2026` is supported as TMLR 2026; add the published-review URL if exact bibliographic provenance is desired. `arruda2026` is supported as ICLR 2026; check author ordering against final proceedings BibTeX. `soiffer2026` is an arXiv preprint and should not be described as peer-reviewed without a separate venue record.

The independent review was read-only and did not submit a GPU job. Bibliographic metadata was checked against the primary records during manuscript preparation.
