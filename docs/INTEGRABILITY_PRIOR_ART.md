# Prior-art audit: integrability failure from random subsampled Feynman–Kac weights

**Date:** 2026-09-09  
**Narrow question:** Has prior work already identified the following phenomenon in compositional diffusion/SBI?

* the full-batch discrete Feynman–Kac (FK) operator at a fixed step size has a finite normalizing constant;
* each component drift and potential estimate is unbiased at the current state;
* a with-replacement minibatch creates histories with positive probability;
* on one such history, a Gaussian exponential-quadratic update has infinite next normalizer;
* because the weight is nonnegative, Tonelli's theorem makes the population operator's normalizer infinite, even though finite-particle runs often look stable;
* decreasing estimator MSE as minibatch size grows does not restore a finite population target for any fixed finite minibatch size.

## Verdict

I found strong prior art for each surrounding issue—random positive weights, moment conditions, FK stability, and Gaussian quadratic-exponential integrability—but no paper that states this exact **rare-minibatch-history integrability certificate for a subsampled compositional score-derived FK operator**, nor one that applies it to the proposed VP composition.

The application-level observation is therefore plausibly distinct. The mathematical core must be framed carefully: it is not simply “random weights have high variance” or “SMC can be unstable.” The candidate result is a sharper existence claim:

> unbiased local drift/potential estimates do not imply that the induced positive exponential FK operator has a finite normalizing constant; a single positive-probability Gaussian history crossing the quadratic-integrability boundary is enough to destroy the population normalizer.

This is a useful diagnostic/method contribution only if the paper proves the certificate, gives an exact reachable-variance recursion, and demonstrates the finite-population versus finite-particle discrepancy. It should not claim that all random-weight SMC is invalid: established random-weight particle filters impose positivity and integrability conditions or construct weights specifically to preserve them.

## Closest primary sources and their precise relationship

### 1. Fearnhead et al.: random-weight particle filtering

Fearnhead, Papaspiliopoulos, Roberts, and Stuart explicitly develop particle filters with random importance weights for continuous-time diffusion models ([JRSS B 2010](https://doi.org/10.1111/j.1467-9868.2010.00744.x), [author record](https://authors.library.caltech.edu/records/34d4q-7w428)). They emphasize that random-weight methods require weights positive almost surely and use a martingale/Wald-identity construction to ensure positivity. This is the closest direct prior art for random weights.

It does **not** analyze a Gaussian exponential-quadratic weight whose conditional integral becomes infinite on a rare minibatch history. Its positivity condition is necessary but weaker than the proposed integrability certificate: a weight can be positive almost surely and still have an infinite first moment or an undefined FK normalizer. The paper supports presenting positivity and moment checks as first-class validity conditions, not as novel terminology.

### 2. Mbalawata and Särkkä: moment conditions for unbounded particle-filter weights

Mbalawata and Särkkä derive moment conditions under which particle-filter estimates converge in mean square and in higher moments, while allowing unbounded importance weights ([Signal Processing 2016](https://doi.org/10.1016/j.sigpro.2015.06.018)). Their result directly blocks a claim that unbounded weights alone are unexplored.

The proposed construction is different in kind: it gives a simple Gaussian counter-certificate where the relevant conditional moment is not merely large or infinite in a tail asymptotic, but the entire next-state integral is infinite whenever

\[
1-2h\widehat c V_{\mathrm{next,prior}}\le 0
\]

in the scalar form used by the experiment. The prior paper supplies the language of required weight moments; it does not derive this reachable-history criterion for score-derived FK potentials.

### 3. Whiteley: FK stability and normalizing-constant variance

Whiteley proves particle-filter stability and relative-variance bounds for normalizing constants under multiplicative-drift and exponential-moment assumptions ([Annals of Applied Probability 2013](https://doi.org/10.1214/12-AAP878)). The key implication for this project is that a finite, well-defined FK flow and suitable exponential moments are assumptions behind stability results.

This paper does not cover a random auxiliary minibatch history that makes the underlying FK kernel non-integrable. Thus the proposed result should be positioned as an **integrability preconditioner/certificate** that must be checked before invoking standard particle stability or finite-variance conclusions.

### 4. Gunawan et al.: subsampling SMC

Gunawan et al. combine data subsampling with annealed SMC and use an approximately unbiased likelihood estimator with pseudo-marginal updates ([arXiv:1805.03317](https://arxiv.org/abs/1805.03317)). This is the closest subsampling-SMC precedent.

Their setting does not provide the proposed result. The estimator is designed around an annealed likelihood target and its stated approximations/limitations; it does not give a Gaussian score-composition recursion that checks every reachable minibatch history for finite exponential moments. The new certificate would be relevant as an additional precondition for subsampling SMC when the random estimator is exponentiated inside a diffusion/FK bridge.

### 5. Del Moral and collaborators: nonnegative FK particle measures

The FK/SMC framework represents unnormalized measures through nonnegative kernels and potentials, with normalizing constants obtained by integrating products of potentials. A representative treatment of convergence for particle approximations with unbounded weights is the moment-condition literature above; the standard FK formulation and unbiased unnormalized particle estimates are also summarized in the particle-filter theory used by Del Moral.

The proposed Tonelli step is elementary but consequential: if the auxiliary minibatch history is part of the population state and has positive probability, and the conditional nonnegative next-state integral is infinite on that event, then the joint integral is infinite. This is an existence failure before particle approximation. It should be stated as a short lemma, with all measurability and positivity assumptions explicit, rather than advertised as a new general FK theorem.

### 6. Gaussian quadratic-exponential integrability

The Gaussian calculation is classical: for a Gaussian state with covariance (V), an exponential quadratic moment is finite only when the resulting quadratic form remains negative definite; in the scalar update this gives the denominator boundary (1-2aV>0). A standard multivariate Gaussian integral formula is recorded in mathematical references such as the multivariate Gaussian chapter ([Springer reference](https://doi.org/10.1007/978-3-030-95864-0_3)).

The proposed value is not the Gaussian integral itself. It is the observation that the minibatch-dependent coefficient \(\widehat c\) has a positive-probability reachable history crossing the boundary, even while the full-batch coefficient remains below it and while the local estimator has finite, decreasing MSE. The exact variance recursion makes the certificate algorithmic rather than an exhaustive history enumeration.

## What appears genuinely different in the current application

The direct compositional score setting has

\[
r_g=s_g+x,\quad R=\sum_g r_g,\quad
b=\frac{\beta}{2}R,\quad
g=\frac{\beta}{2}\left(\|R\|^2-\sum_g\|r_g\|^2\right).
\]

For Gaussian factors, a fixed minibatch history remains Gaussian and its next variance can be propagated in closed form. In the scalar notation supplied for the current example,

\[
V_{\mathrm{next}}
 =\frac{\bigl(1-h(\widehat A+\tfrac12)\bigr)^2V}
 {1-2h\widehat c V}+h.
\]

If the denominator is nonpositive, the conditional Gaussian integral is infinite. Therefore:

1. a single history with positive probability is enough;
2. no union bound over all histories is needed;
3. the maximal reachable variance can be recursively computed by dynamic programming over the distinct minibatch transition types;
4. a finite-(P) particle run can miss the history and remain numerically finite;
5. (\operatorname{MSE}(\widehat A,\widehat c)\to0) as minibatch size increases does not imply finite normalization for every finite minibatch size.

The reported G4 example—Gaussian factors with \(\lambda\in[0.1,0.2,2,4]\), \(K=16,\ldots,512\)—is exactly the right kind of discriminator if the full-batch operator remains finite while every tested finite minibatch size has a reachable divergent history. It still needs an independently checked probability statement for the offending history and a proof that the recursion covers all transitions allowed by the sampling scheme.

## Required theorem and experiment boundary

The minimal theorem should have four parts:

1. **Conditional Gaussian integral:** derive the finite/infinite criterion for one history, including equality at the boundary.
2. **Tonelli lift:** prove that positive probability of a divergent history implies an infinite population normalizer for the nonnegative joint operator.
3. **Reachability recursion:** show that the maximum reachable variance is obtained by the stated finite recursion, without enumerating all history sequences.
4. **Separation from finite-particle behavior:** state why a finite-(P) run can be finite with high probability while the population operator is infinite, and report an empirical detection probability or lower bound for the bad history.

The numerical comparison should include full-batch FK, naive minibatch FK, increasing minibatch sizes, and finite particle counts. Report the certificate before interpreting ESS, finite sample estimates, or apparent agreement with the full-batch posterior. A control experiment should use an estimator with bounded/clipped quadratic coefficient or a proposal/weight construction that guarantees the denominator condition, making clear whether the method trades exactness for integrability.

## Claims to avoid

* “Unbiased drift and potential estimates imply an unbiased or valid exponential FK operator.”
* “Finite MSE guarantees finite normalizing constants.”
* “A finite run with no NaNs proves the population target exists.”
* “Random-weight SMC is invalid in general.”
* “The Gaussian denominator condition is a new Gaussian-integral theorem.”

## Bottom-line novelty assessment

The surrounding ingredients are established. The likely novel contribution is the **application-specific integrability failure certificate** for subsampled compositional score-derived FK operators, together with the exact reachable-variance recursion and the finite-population/finite-particle separation. This is strong enough for a focused diagnostic or methods paper if the proof and G4-style experiment are airtight. The paper should present it as a validity condition for a particular random-weight compositional diffusion construction, not as a new general theory of SMC or Gaussian exponential moments.

## Sources checked (6)

1. Fearnhead, Papaspiliopoulos, Roberts, and Stuart, *Random-weight particle filtering of continuous time processes*, JRSS B 2010: https://doi.org/10.1111/j.1467-9868.2010.00744.x
2. Mbalawata and Särkkä, *Moment conditions for convergence of particle filters with unbounded importance weights*, Signal Processing 2016: https://doi.org/10.1016/j.sigpro.2015.06.018
3. Whiteley, *Stability properties of some particle filters*, Annals of Applied Probability 2013: https://doi.org/10.1214/12-AAP878
4. Gunawan, Dang, Quiroz, Kohn, and Tran, *Subsampling Sequential Monte Carlo for Static Bayesian Models*, arXiv:1805.03317: https://arxiv.org/abs/1805.03317
5. Del Moral, standard particle/Feynman–Kac framework and unnormalized measures: https://doi.org/10.1007/978-1-4684-0491-0
6. Multivariate Gaussian exponential-integral reference: https://doi.org/10.1007/978-3-030-95864-0_3

