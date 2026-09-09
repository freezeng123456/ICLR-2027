# Two-day novelty review: Gaussian control variates and minibatch Feynman–Kac correction

**Review date:** 2026-09-09  
**Question:** Is “a Gaussian control variate plus a minibatch random Feynman–Kac correction for large-data compositional SBI” a sufficiently distinct ICLR direction, and what must be proved before spending the two-day experiment budget?

## Decision

The broad proposal is **already substantially covered**. A Gaussian control variate is a standard variance-reduction device; unbiased or approximately unbiased subsampling estimators are established in pseudo-marginal MCMC and SMC; and sequential control variates have been developed specifically for Feynman–Kac / Markov-process functionals. A compositional-SBI version is therefore not novel merely because it combines these words, uses minibatches, or reduces the variance of a particle estimate.

The direction is worth two days only if it produces a theorem and an implementation with all of the following properties:

1. the estimator targets the **composed tall-data SBI Feynman–Kac quantity**, rather than a generic likelihood or generic SMC normalizing constant;
2. the Gaussian control variate is usable with only quantities available at inference time and has a stated residual bias/variance bound;
3. the random minibatch correction is either exactly unbiased in the required nonnegative form, or its controlled bias is explicit and propagated through resampling/composition;
4. the result covers the non-Gaussian or cross-batch terms that existing Gaussian corrections leave out; and
5. the method beats a strong implementation of existing pseudo-marginal / controlled-SMC / sequential-control-variate baselines at fixed compute.

Without these conditions, stop after the literature check and reframe the idea as an engineering combination or benchmark.

## What prior work already removes from the novelty claim

### Gaussian control variates and subsampling

Quiroz et al. introduce control-variate estimators of the log likelihood for data subsampling, bias-correct the exponentiation step, and analyze the resulting perturbed posterior as a function of full-data size and minibatch size ([arXiv:1404.4178](https://arxiv.org/abs/1404.4178); the journal version is [JASA 2019](https://doi.org/10.1080/01621459.2018.1448827)). A claim such as “a Gaussian approximation provides a control variate for large-data Bayesian inference” is therefore not new by itself. The relevant distinction would have to be the structure of the Gaussian approximation for **compositional posterior scores / Feynman–Kac potentials**, together with a theorem that is stronger or more targeted than the perturbed-posterior analysis there.

### Minibatch SMC and pseudo-marginal correction

Gunawan et al. already combine data subsampling with annealed SMC for static Bayesian models, using an approximately unbiased likelihood estimator and block pseudo-marginal updates ([arXiv:1805.03317](https://arxiv.org/abs/1805.03317)). Thus “random minibatches inside SMC” and “approximately unbiased correction” cannot be the central contribution. The proposed work must identify a compositional-SBI-specific obstacle—such as a cross-component normalization or bridge term—and show why existing subsampling SMC does not resolve it.

### Controlled SMC

Heng et al. formulate controlled SMC through an approximate optimal-control problem and give fluctuation and stability analysis ([arXiv:1708.08396](https://arxiv.org/abs/1708.08396); [Annals of Statistics DOI](https://doi.org/10.1214/19-AOS1914)). A Gaussian proposal/control variate that improves particle stability is close to this literature unless the proposed correction targets a different object and has a distinct guarantee. Variance reduction, better proposals, and lower particle degeneracy are useful outcomes, but they do not establish algorithmic novelty without the new target or theorem.

### Sequential control variates for Feynman–Kac functionals

Gobet and Maire developed sequential control variates for Markov-process functionals represented by the Feynman–Kac formula, including diffusion and jump processes, and analyzed geometric reduction of bias and variance ([SIAM JNA DOI](https://doi.org/10.1137/040609124)). This is the strongest direct objection to the phrase “Feynman–Kac correction.” A new method must explain precisely which Feynman–Kac quantity is outside their setting: for example, a compositional score bridge with data-dependent potentials, a tall-data minibatch estimator that preserves nonnegativity, or a finite-particle correction with a new bound.

## The specific gap that may remain

The defensible gap is narrow:

> Given oracle or learned component scores for conditionally independent data blocks, construct a finite-compute, minibatch Feynman–Kac estimator for the composed SBI posterior whose Gaussian control variate has a provable residual error, while preserving the required positivity/normalization for particle weighting.

This is different from saying that Gaussian control variates, SMC subsampling, or sequential control variates are individually new. The key difficulty is that three operations need not commute:

\[
\text{compose component posteriors}\;\to\;\text{add noise / form a bridge}\;\to\;\text{subsample and exponentiate}.
\]

A control variate that is unbiased for a log potential need not yield an unbiased **positive potential** after exponentiation. A minibatch correction that is unbiased for one component may also fail after products, resampling, or a time-inhomogeneous score bridge. The reportable contribution would be a theorem that tracks these operations rather than an empirical reduction in variance alone.

## A concrete two-day theorem target

Use a simple compositional model first. Let the full potential at bridge time \(t\) be

\[
G_t(z)=\exp\!\left(\sum_{j=1}^{J}\ell_{j,t}(z)+r_t(z)\right),
\]

where \(r_t\) is a Gaussian control-variate term available from a tractable approximation and \(\ell_{j,t}-r_{j,t}\) are residual block contributions. Sample a minibatch \(B\) with known inclusion probabilities and define a corrected residual estimator \(\widehat R_t\). The minimum useful result would be one of:

* **Exact route:** construct a nonnegative unbiased estimator \(\widehat G_t\) of \(G_t\), and prove finite-particle consistency or a variance bound under explicit moment assumptions;
* **Controlled-bias route:** prove an explicit bound such as \(|\mathbb E\widehat G_t-G_t|\leq C_t/m^\alpha\), then propagate it through the normalized Feynman–Kac flow and state how the bound scales with composition count \(J\), minibatch size \(m\), and particle count \(N\);
* **Path-consistency route:** show that the correction targets the missing intermediate bridge term in compositional SBI, and characterize exactly when it is zero (for example, a Gaussian constant-covariance special case) and how the non-Gaussian remainder enters.

The theorem must compare against the existing Gaussian-exact special case. If the construction is exact only when every component is Gaussian with constant covariance, it is a baseline or sanity check, not the headline result.

## Two-day falsification plan

**Day 1: algebra and estimator audit.**

1. Write the full-data composed potential and the proposed minibatch estimator in one notation.
2. Check whether the estimator is unbiased before exponentiation, after exponentiation, and after normalized particle weighting.
3. Derive the Gaussian special case symbolically and identify the first nonzero non-Gaussian or cross-batch remainder.
4. Compare the resulting assumptions and bound line by line with Quiroz, Gunawan, Heng, and Gobet–Maire.
5. If no term remains that is specific to compositional SBI, stop.

**Day 2: minimal numerical discriminator.**

Use a one-dimensional Gaussian target, a mildly non-Gaussian target, and a two-component composition. Compare full-data particle estimates, naive minibatch composition, the proposed estimator, and the strongest existing-style control-variate/pseudo-marginal baseline at matched likelihood evaluations. Record bias, variance, effective sample size, normalization failure rate, and error versus \(J,m,N\). Do not interpret a speedup on the Gaussian case as evidence of novelty; it is a correctness control.

The direction passes only if the proposed correction is measurably better on the non-Gaussian/cross-batch case **and** the observed scaling follows the new bound. If it merely lowers variance in the Gaussian case, or if its advantage disappears against a sequential-control-variate baseline, terminate.

## Claims that should not appear in a paper proposal

* “We introduce control variates for large-data Bayesian inference.”
* “We are the first to use minibatches in SMC.”
* “We correct Feynman–Kac particle bias with a Gaussian approximation.”
* “The correction is unbiased” when only a log-likelihood estimate is unbiased.
* “The method is exact” when exactness holds only for a Gaussian constant-covariance toy case.
* “Lower particle variance proves a better posterior” without a normalization, bias, or downstream posterior-risk check.

## Final recommendation

Proceed for two days only as a **theorem-first audit** of a compositional, path-level correction. The default prior is that the broad idea is covered by existing methods. The project should be stopped if it cannot produce a new finite-minibatch positivity/bias result, a compositional path-consistency result beyond the Gaussian exact case, or a clearly superior non-Gaussian benchmark against sequential control variates and subsampling SMC. No experiment or code change was made in this review.

## Sources checked

1. Quiroz, Kohn, Villani, and Tran, *Speeding Up MCMC by Efficient Data Subsampling*, [arXiv:1404.4178](https://arxiv.org/abs/1404.4178).
2. Gunawan, Dang, Quiroz, Kohn, and Tran, *Subsampling Sequential Monte Carlo for Static Bayesian Models*, [arXiv:1805.03317](https://arxiv.org/abs/1805.03317).
3. Heng, Bishop, Deligiannidis, and Doucet, *Controlled Sequential Monte Carlo*, [arXiv:1708.08396](https://arxiv.org/abs/1708.08396).
4. Gobet and Maire, *Sequential Control Variates for Functionals of Markov Processes*, [SIAM JNA](https://doi.org/10.1137/040609124).
5. Linhart et al., *Diffusion Posterior Sampling for SBI in Tall Data Settings*, [arXiv:2404.07593](https://arxiv.org/abs/2404.07593), for the directly relevant Gaussian/path-correction baseline in diffusion SBI.
6. Arruda et al., *Compositional Diffusion Models for SBI* / R1, [arXiv:2505.14429](https://arxiv.org/abs/2505.14429), for the compositional score-bridge setting and its damping/path issue.

