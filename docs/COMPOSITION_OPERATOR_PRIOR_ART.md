# Prior-art audit: random Euler Feynman–Kac operators for compositional scores

**Date:** 2026-09-09  
**Scope:** bounded novelty check for the one-step operator

\[
\widehat{Q}_h f(x)=\mathbb E\left[e^{h\widehat g(x)}
 f\left(x+h\widehat b(x)\right)\right],
\]

with compositional-score quantities

\[
r_g=s_g+x,\qquad R=\sum_g r_g,\qquad
b=\frac{\beta}{2}R,\qquad
g=\frac{\beta}{2}\left(\|R\|^2-\sum_g\|r_g\|^2\right).
\]

The proposed local calculation separates the second-order contribution into

\[
h^2\left[
\frac12\operatorname{Cov}(\widehat b):\nabla^2f
+\operatorname{Cov}(\widehat b,\widehat g)\!\cdot\!\nabla f
+\frac12\operatorname{Var}(\widehat g)f
\right].
\]

The question is whether existing primary literature has already treated **all three terms together**—random drift, drift/potential correlation, and exponential-potential variance—in a compositional diffusion application with experiments.

## Judgment

I found no primary paper that explicitly presents this exact combination as a single compositional-diffusion analysis with the three displayed terms isolated and experimentally tested. The closest papers cover the pieces separately:

* weak-error and random-drift Euler analysis covers random drift, but usually studies an unweighted Markov transition;
* Feynman–Kac numerical analysis covers exponential potentials and splitting/Euler bias, but generally assumes a deterministic drift/potential pair or analyzes particle approximation;
* compositional SBI papers analyze score-composition error, Gaussian path corrections, or sampling quality, but do not expand a minibatch-random one-step Feynman–Kac operator into these three covariance terms.

This is a **real but narrow difference**. The second-order expansion itself is a standard multivariate Taylor/weak-error calculation and must not be claimed as a new theorem without a genuinely new setting, remainder bound, or consequence. The publishable unit could be a diagnostic/method paper if it proves the expansion under a clear filtration and moment condition, propagates it through a composed VP bridge, and shows that the three terms predict observed bias or instability better than total variance or score MSE alone.

## Directly relevant compositional diffusion sources

### 1. Geffner et al.: compositional score modeling for SBI

Geffner et al. factorize a multi-observation posterior into individual posteriors, aggregate learned individual scores, and sample with an annealing-style procedure ([PMLR paper](https://proceedings.mlr.press/v202/geffner23a.html), [PDF](https://proceedings.mlr.press/v202/geffner23a/geffner23a.pdf)). Their sections on F-NPSE/PF-NPSE establish the compositional target and sampling algorithm, but do not model random minibatch estimates \((\widehat b,\widehat g)\) inside a Feynman–Kac one-step operator, nor derive the covariance decomposition above. This is the correct application baseline, not a novelty claim.

### 2. Linhart et al.: tall-data SBI and GAUSS/JAC correction

Linhart et al. derive a correction for the score of a diffused tall-data posterior and give Gaussian approximations to the relevant backward kernels ([arXiv:2404.07593](https://arxiv.org/abs/2404.07593)). Their Gaussian constant-covariance case is an essential exact/sanity regime. The paper addresses path correction and score construction, not the finite-minibatch random-operator expansion. Any proposed method must compare against GAUSS and JAC and state whether its \(\widehat g\) is an estimator of a correction potential, a score-derived potential, or merely a control variate.

### 3. Arruda et al.: compositional amortized inference

Arruda et al. study compositional amortized inference and damping of aggregated scores ([arXiv:2505.14429](https://arxiv.org/abs/2505.14429)). Their motivation is accumulation and instability as the number of observations/components increases. The work does not isolate stochastic-drift variance, drift/potential covariance, and exponential-weight variance from a random Euler operator. It is therefore a direct experimental benchmark for whether the proposed three-term diagnostic explains failures that damping alone does not.

### 4. Touron et al.: error analysis of GAUSS

Touron et al. provide the closest recent theoretical analysis of compositional SBI: an MSE bound for GAUSS in terms of individual score errors, precision estimation errors, and the number of observations, with a Gaussian numerical example ([arXiv:2510.15817](https://arxiv.org/abs/2510.15817)). Their Proposition 2 and Gaussian experiments concern **score-estimation error**, not the weak error of a random weighted Euler transition. They explicitly separate score and precision errors and do not analyze the exponential-potential variance or the covariance between random drift and random potential. This makes the proposed operator audit meaningfully different, but also sets a strong baseline: the new paper must show why the three operator terms predict a downstream error not already explained by their score MSE bound.

## Numerical-analysis and Feynman–Kac sources that constrain the claim

### 5. Standard weak Euler/Talay–Tubaro framework

Weak Euler analysis derives local expansions by applying a generator/Taylor expansion to test functions. The classical smooth-coefficient result gives order-one weak error, while later work treats irregular coefficients. A useful primary reference for the relevant expansion machinery is Talay and Tubaro's weak approximation framework ([original journal record](https://doi.org/10.1016/0304-4149(90)90077-6)). The three covariance terms are therefore not new merely because they appear after expanding \(e^{h\widehat g}f(x+h\widehat b)\). Novelty must come from the jointly random, correlated drift/potential and its compositional VP interpretation.

### 6. Majka, Mijatović, and Szpruch: randomized/inaccurate drift

Majka et al. analyze Euler-type sampling with a randomized or inaccurate drift and prove Wasserstein and weak-convergence results; their headline observation is that random-drift variance does not change the weak-convergence rate in their setting ([arXiv:1808.07105](https://arxiv.org/abs/1808.07105)). This directly blocks any claim that “random drift variance in Euler is unexplored.” Their setting has no exponential Feynman–Kac potential correlated with the drift. The proposed work must explain whether the \(\operatorname{Cov}(\widehat b,\widehat g)\) and \(\operatorname{Var}(\widehat g)\) terms create a new bias mechanism precisely because the transition is weighted.

### 7. Jourdain and Menozzi: Euler weak error with irregular drift

Jourdain and Menozzi establish weak convergence rates for Euler schemes with low-regularity drifts, including a randomized time variable/cutoff construction ([Annals of Applied Probability](https://doi.org/10.1214/23-AAP2006)). This reinforces that an Euler expansion needs explicit regularity, moment, and conditioning assumptions. It does not treat a random potential or compositional score bridge. It is relevant for a rigorous remainder bound if score estimates are only locally regular.

### 8. Gobet and Maire: sequential control variates for Feynman–Kac functionals

Gobet and Maire develop sequential control variates for Markov-process functionals represented by Feynman–Kac formulae, including diffusion and jump processes, and analyze bias and variance reduction ([SIAM JNA](https://doi.org/10.1137/040609124)). This is the closest prior art for using a control variate in a weighted Markov/Feynman–Kac computation. A Gaussian moment-matched residual and a U-statistic estimator should be presented as an application/instantiation unless the paper proves a new compositional finite-minibatch bound or a new positivity property.

### 9. Del Moral and coauthors: particle Feynman–Kac error analysis

The particle-Feynman–Kac literature gives propagation-of-chaos, fluctuation, and variance bounds for normalized particle approximations. For a representative non-asymptotic result, see the linear variance-bound analysis for time-homogeneous Feynman–Kac formulae ([ScienceDirect record](https://doi.org/10.1016/j.spa.2012.05.008)). These results constrain any claim that a local \(O(h^2)\) term alone establishes global particle accuracy: one must propagate local operator error through normalization, resampling, and the number of time steps.

### 10. Splitting and exponential-potential discretization

Feynman–Kac semigroups are often approximated by Lie/Strang splitting, with separate Markov propagation and multiplication by an exponential potential. The resulting local commutator/quadrature error is established in the numerical-analysis literature; for example, Gobet and Maire's Feynman–Kac setting and subsequent exponential-potential splitting analyses distinguish transition error from exponential-weight integration error. This is another reason that \(\operatorname{Var}(\widehat g)\) is not novel in isolation. The candidate contribution is the **joint randomization and correlation** of both factors in a score-composed VP transition.

## What the existing evidence actually supports

The literature supports the following conservative statements:

1. A one-step weak expansion for a random weighted Euler operator is mathematically routine under suitable smoothness and moments.
2. Random/inaccurate drift is already analyzed in Euler sampling; drift variance alone is not a novelty gap.
3. Exponential Feynman–Kac weights, variance reduction, and particle error propagation are established topics.
4. Compositional SBI has direct baselines for score aggregation, Gaussian path correction, damping, and score/precision error accumulation.
5. The combination of a **random compositional score drift**, a **correlated random composition potential**, and an **exponential-weighted Euler operator** does not appear to have been explicitly decomposed into the three covariance terms and tested as a diagnostic in the direct compositional-diffusion papers checked here.

The fifth point is a bounded search conclusion, not a claim of absolute priority. It is sufficiently distinct for a focused method/diagnostic paper, provided the paper proves and tests the consequence rather than presenting the algebra as a new general numerical-analysis theorem.

## Minimum viable contribution

The strongest defensible formulation is:

> We derive and validate a conditional second-order local-error decomposition for a minibatch-random compositional VP Feynman–Kac Euler operator. The decomposition separates random-transition variance, drift/potential covariance, and exponential-potential variance. We show on controlled Gaussian and non-Gaussian compositions which term dominates, how the terms scale with the number of components and minibatch size, and whether their sum predicts global posterior/sample error better than score MSE or total estimator variance.

Required conditions:

* define the filtration: what is fixed at \(x\), what randomness generates \(\widehat b,\widehat g\), and whether the minibatch is redrawn per step;
* state centering assumptions and include first-order bias terms if \(\mathbb E[\widehat b]\neq b\) or \(\mathbb E[\widehat g]\neq g\);
* give a remainder bound with explicit dependence on \(h\), dimension, moments, and derivatives of \(f\);
* distinguish the local operator error from accumulated global weak error and particle approximation error;
* compare full-batch, independent component subsampling, shared minibatch randomness, and a decorrelated control experiment;
* include GAUSS/JAC, naive score composition, and a sequential-control-variate baseline where applicable.

## Stop/continue rule

Continue only if at least one of the following is observed in the controlled experiment:

* the covariance term \(\operatorname{Cov}(\widehat b,\widehat g)\) changes sign or magnitude in a way that predicts global composition error while score MSE does not;
* the exponential-weight term dominates in a regime where a Gaussian residual control variate reduces score variance but not weighted-operator bias;
* shared versus independent minibatch randomness produces a reproducible difference explained by the cross-covariance term;
* the decomposition yields a useful design rule or estimator with a bound that survives non-Gaussian composition and beats the direct baselines.

Stop if the terms collapse to an ordinary Taylor identity, are fully explained by total score MSE/variance, or only improve the Gaussian exact case. The Gaussian moment-matched residual control variate and U-statistic should remain classical implementation tools unless the work proves a new finite-minibatch positivity or propagation result.

## Sources checked (10; all primary or publisher records)

1. Geffner et al., *Compositional Score Modeling for Simulation-Based Inference*, PMLR 2023: https://proceedings.mlr.press/v202/geffner23a.html
2. Linhart et al., *Diffusion Posterior Sampling for SBI in Tall Data Settings*, arXiv:2404.07593: https://arxiv.org/abs/2404.07593
3. Arruda et al., *Compositional Amortized Inference for Large-Scale Hierarchical Bayesian Models*, arXiv:2505.14429: https://arxiv.org/abs/2505.14429
4. Touron et al., *Error Analysis of a Compositional Score-Based Algorithm for SBI*, arXiv:2510.15817: https://arxiv.org/abs/2510.15817
5. Talay and Tubaro, weak approximation framework: https://doi.org/10.1016/0304-4149(90)90077-6
6. Majka, Mijatović, and Szpruch, *Non-asymptotic Bounds for Sampling Algorithms Without Log-concavity*, arXiv:1808.07105: https://arxiv.org/abs/1808.07105
7. Jourdain and Menozzi, Euler weak convergence with low-regularity drift: https://doi.org/10.1214/23-AAP2006
8. Gobet and Maire, *Sequential Control Variates for Functionals of Markov Processes*: https://doi.org/10.1137/040609124
9. Del Moral et al., linear variance bounds for particle Feynman–Kac formulae: https://doi.org/10.1016/j.spa.2012.05.008
10. Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*: https://arxiv.org/abs/2011.13456
