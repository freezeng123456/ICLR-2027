# Scientific review of the completed compositional-diffusion manuscript

Date: 2026-09-10. Reviewed starting commit: `9b56e0889257f74c19a357e6f3432d26bb113784`. This is a sequential self-review by the authoring agent, not an independent referee report. It combines direct source inspection, primary-literature checks, algebraic verification, and the existing raw-data audits. No new GPU experiment is implied by this review.

## Result and correction

The principal actionable finding was a missing direct attribution. The manuscript described the compositional Feynman--Kac identity as background but omitted the paper that directly develops Feynman--Kac correctors for products of diffusion marginals. The revised related-work section now cites Skreta et al. and Thornton et al. The appendix explicitly identifies the connection to the weighted-product formula. No theorem, experimental cell, or reported numerical result was changed.

The scoped certificate arguments remain consistent with the implemented operator on this review. That is a self-review conclusion, not a proof that no mathematical error remains. A targeted search did not locate an identical extreme-batch or prefix/suffix certificate in the directly inspected sources; this is not a comprehensive novelty guarantee or an acceptance prediction.

## Primary-source comparison

| Source | Established ingredient relevant here | Consequence for this manuscript |
|---|---|---|
| [Skreta et al., ICML 2025](https://proceedings.mlr.press/v267/skreta25a.html), Sections 3.4 and 4, Proposition D.5 | Weighted diffusion and resampling for products of marginals; a general weighted-product score formula | Cite the construction directly. The new question must start with finite-step factor subsampling and existence of its normalizer. |
| [Thornton et al., AISTATS 2025](https://proceedings.mlr.press/v258/thornton25a.html) | Distilled energy models and Feynman--Kac SMC for composition and control | Energy-based diffusion composition with SMC is established background. |
| [Quiroz et al., version 6](https://arxiv.org/abs/1404.4178v6) | Unbiased control-variate log-likelihood estimates, bias-corrected likelihoods, and perturbed-posterior analysis | Unbiased log estimates and exponentiation bias are not new observations by themselves. |
| [Heng et al., version 3](https://arxiv.org/abs/1708.08396v3) | Proposal control through an approximate optimal-control problem and associated stability analysis | Our affine potential control is not a claim to introduce controlled SMC or to outperform its proposal optimization. |

The searches included combinations of “Feynman-Kac,” “subsampling,” “integrability,” “infinite normalizer,” “minibatch,” and the named diffusion-corrector papers. The official Skreta PDF was read for the product formula and simulation alternatives. Its lack of matching text keywords alone is not evidence of novelty. The comparison above concerns the stated objects and formulas.

## Exact connection to the continuous-flow background

In Proposition D.5 of the [official FKC paper](https://raw.githubusercontent.com/mlresearch/v267/main/assets/skreta25a/skreta25a.pdf), specialize the weighted product by including the stationary Gaussian factor with exponent `1-G` and each of the `G` posterior factors with exponent one. The following algebra is our verification of that specialization.

Let `s_0=-x`, `r_g=s_g+x`, `R=sum_g r_g`, and `Q=sum_g ||r_g||^2`. Then the exponent sum is one and

\[
 \sum_{i=0}^G\beta_i s_i=R-x,
 \qquad
 \sum_{i=0}^G\beta_i\|s_i\|^2
 =Q-2x^\mathsf TR+\|x\|^2.
\]

Consequently the weighted score-norm difference is `(||R||^2-Q)/2`, exactly the manuscript's full potential. With OU forward drift `-x/2`, the reverse drift is `R-x/2`, the manuscript's `kappa=1` case. For general positive `kappa`, the added drift and diffusion terms cancel in the density PDE through `div(rho grad log rho)=Delta rho`. Normalizability assumptions are still required. This continuous identity is not claimed as a new theorem.

A numerical self-check evaluated the specialization in 27 combinations of group count, dimension and noise level, with seven states per combination. Maximum absolute discrepancy was `9.094947017729282e-13`. This finite check supports the implementation correspondence; the identity follows from the algebra above. The local receipt is included with the review validation artifacts.

## Theorem-to-code and claim audit

| Contract | Source anchors | Review result |
|---|---|---|
| Potential evaluated before Euler propagation | `composition_benchmark.py:run_cell` | The increment uses the current state and current noise level, matching the specified kernel. Resampling follows propagation and accumulated weighting. |
| Drift convention | `FactorModel.exact`, `FactorModel.estimate`, `run_cell` | Model methods return `R/2`; the loop transforms this to `(1+kappa)R/2-kappa*x/2`. The apparent factor-of-two difference is accounted for. |
| WR versus WOR pair coefficients | `composition_benchmark.py:estimate`, `composition_extension.py:estimate` | WR includes the diagonal correction; WOR uses the distinct-pair inclusion coefficient. They are different unbiased estimators and are not interchanged. |
| Tail-control slope | `FactorModel.coefficients`, `run_cell` | The code's affine slope is `1-1/v`, the negative of the paper's nonnegative residual precision. Tail methods explicitly use component variance rather than total mixture variance. |
| Mixture boundary | `gaussian_integrability.py:audit_path`, `audit_extension_results.py:independent_certificate` | Gaussian nonpositive denominators are classified exactly. The formal mixture audit requires a strictly negative failing denominator; the paper leaves the general zero boundary open. |
| Learned density rather than arbitrary score network | `learned_sbi.py:PosteriorMDN`, `predict`; manuscript extension protocol | The trained network predicts shared-variance mixture parameters. Subsequent noised scores are analytic conditional on these parameters. The variance architecture supplies the tail assumption. |
| Training and evaluation separation | `learned_sbi.py:train`, `negative_log_likelihood`; training audit | Training uses simulated joint samples, fixed final checkpoints and no exact posterior labels. Exact posterior calculations are diagnostics. This does not remove the architectural restriction. |
| Distinct sources of error | `run_composition_extension.py:run_extension_cell`, raw-result audit | Learned-composition discrepancy and particle discrepancies against learned and true references are retained separately. They are not asserted to be an additive W1 decomposition. |
| Crossed seed interpretation | `run_composition_extension.py:matrix`; manuscript figures/tables | The 25 combinations are not 25 independent trainings. Reported standard deviations describe dispersion; the paper does not use them as independent-replicate standard errors. |
| Recorded timing | `run_cell`, `run_extension_cell` | Sampler time includes subset selection and resampling. The separate setup timer includes checkpoint loading, dataset construction and parameter prediction; neither timer establishes arbitrary neural-score inference speed. |
| Sensitivity interpretation | Oracle manifest and manuscript appendix | The U sweep changes both initialization and the complete grid at fixed K. Particle-count results and finite certificates do not establish monotonic accuracy improvement. |

## Remaining scientific boundaries and decision

The strongest supported contribution remains an existence certificate for a specified random-batch weighted Euler operator. The full-factor method can itself fail on a coarse grid; finite-particle accuracy cannot settle population integrability. Tail control preserves a strict finite domain for the stated tail class and does not guarantee low sampling error. The revised related-work text explicitly restricts conclusions to the stated operator, so the counterexamples are not presented as a refutation of every FKC discretization, active-interval correction, or jump-process implementation.

The learned-density study is controlled synthetic inference with a correct mixture family built into the architecture. It demonstrates that the mechanism persists after actual parameter learning, but it supplies neither a robustness result for arbitrary score networks nor a real-world SBI performance benchmark. Noncommuting covariance tails and the general-mixture zero boundary remain outside the proof. Independent human scientific review remains outstanding.

No additional GPU sweep is needed to resolve the attribution finding. Additional computational work should be tied to a new explicit claim, such as a broader certified tail class or a defined real-world inference task; simply increasing the current experiment count would not resolve the limitations above.
