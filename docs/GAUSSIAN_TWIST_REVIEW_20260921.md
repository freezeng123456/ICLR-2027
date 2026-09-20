# Gaussian twist theorem: internal audit

This is an AI-assisted internal working assessment. It is not an independent human review and is not a conference review.

## Supported theorem boundary

The design and implementation support a finite-horizon Gaussian backward-recursion construction under explicit assumptions:

- (K<\infty);
- finitely many admissible batch histories;
- nondegenerate Gaussian initial law;
- strictly positive-definite Gaussian transition noise covariances;
- globally bounded residual and mean corrections after the common affine Gaussian part is removed;
- a common quadratic potential coefficient at each step;
- strict positive definiteness of every backward precision and of the initial twisted precision;
- finite positive reference normalizer (Z_0).

Under these assumptions, the complete path log-density correction is at most linear in the stacked path norm. A Gaussian quadratic tail dominates every fixed positive multiple of this linear correction. Therefore every fixed positive weight moment is finite. Uniform constants exist over the finite history set. No bound uniform in (K) follows from this argument.

The theorem sidecar is [gaussian_twist_theorem.tex](/Users/zenghang/Documents/Codex/2026-08-24/https-github-com-freezeng123456-iclr-2027/work/review-dual-20260921/manuscript/gaussian_twist_theorem.tex).

## Backward recursion

The implementation in [gaussian_twisted_path_20260921.py](/Users/zenghang/Documents/Codex/2026-08-24/https-github-com-freezeng123456-iclr-2027/work/review-dual-20260921/gaussian_twisted_path_20260921.py) implements the quadratic backward recursion with

\[
H_k=Q_k^{-1}-J_{k+1},
\quad
T_k=Q_k^{-1}H_k^{-1}Q_k^{-1}-Q_k^{-1},
\quad
\nu_k=Q_k^{-1}H_k^{-1}j_{k+1}.
\]

The returned initial twisted Gaussian, marginal moments, and log normalizer are independently compared against dense joint Gaussian calculations in the existing test file. The current five tests cover four-step dense paths, a singular transition matrix, a one-step case where the original second moment diverges, affine backbone algebra, and invalid/non-symmetric input rejection. The reported implementation evidence is consistent with the finite-(K) recursion.

The backward recursion must retain strict positive-definiteness checks. A nonpositive backward precision is an integrability failure or boundary case for the proposed reference construction; it cannot be repaired by the finite-sample sampler.

## Importance-sampling consequence

For independent paths drawn from the normalized Gaussian reference, the corrected path weight has all fixed positive moments under the theorem assumptions. This supports consistency for bounded observables. A standard self-normalized importance-sampling CLT requires the relevant second moment, together with the usual nonzero finite target normalizer. The statement concerns independent paths and does not provide a theorem for resampling systems, genealogical dependence, or universal low variance.

The phrase “all moments” therefore means every fixed (p>0) for a fixed finite path problem under the stated strict certificate. It does not mean a uniform-in-time result, a uniform-in-(p) numerical bound, or an application-independent variance guarantee.

## Mixture reference boundary

For finitely many positive Gaussian reference components, the importance denominator must be the full mixture density. Using the density of the selected component creates a different weight and does not target the mixture law. If all components have common strict quadratic tail control and strictly positive mixture probabilities, a finite-horizon linear-envelope argument can be applied with the complete denominator. Component-specific tail curvature, vanishing mixture probabilities, and long-time accumulation require additional analysis.

## Position relative to prior work

Whiteley and Lee, *Twisted Particle Filters*, establish path twisting and normalized-constant constructions for particle filters. Heng, Bishop, Deligiannidis, and Doucet, *Controlled Sequential Monte Carlo*, Sections 3.1 and Supplement 7.1, include linear-quadratic Gaussian control, backward Riccati recursions, and Gaussian zero-variance cases. Those results are the relevant prior foundation. The present construction should not claim general Gaussian twisting, Riccati recursion, or zero-variance control as novel.

The defensible scope is the explicit finite-step application to the specified Gaussian reference and bounded correction class, together with a separately checked strict-domain condition. The existing design document already identifies this boundary and separates the candidate from general twisting and controlled SMC.

## Remaining scientific gaps

The current implementation tests establish algebraic recursion correctness for finite examples. They do not by themselves establish:

- a uniform long-time moment bound;
- resampling stability or genealogical moment bounds;
- universal variance reduction;
- validity of an approximate or state-dependent mixture denominator;
- finite moments when the strict backward precision certificate fails;
- real-sensor application calibration or deployment benefit.

The dense-path Gaussian numerical tests reported by the main task are useful implementation evidence. They should remain separate from the theorem assumptions and from claims about resampled particle systems.

No implementation, main paper file, or existing test file was modified in this audit. No remote job, commit, or external dissemination was performed.
