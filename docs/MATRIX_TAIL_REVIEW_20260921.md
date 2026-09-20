# Matrix tail extension: internal scientific review

This is an AI-assisted internal working assessment. It is not an independent human review and is not a conference review.

## Mathematical status

The extension has a clear strict-domain result for a fixed finite Gaussian path with shared full covariance inside each factor. For symmetric component tail matrices (A_g), the quadratic coefficient

\[
C=\tfrac12\big[(\sum_g A_g)^T(\sum_g A_g)-\sum_g A_g^TA_g\big]
\]

is symmetric and may be indefinite. With incoming covariance (V\succ0), the exact precision test is

\[
H=V^{-1}-2hC.
\]

The strict conclusions are:

- (H\succ0) gives a finite one-step Gaussian exponential integral and the covariance envelope (V^+=LH^{-1}L^T+\kappa hI).
- A strictly negative eigenvalue of (H) gives divergence along an eigenvector direction.
- The semidefinite boundary is unclassified by this extension.

The independent Gaussian update implemented in the certificate uses the exact precision, mean, and log-normalizer formulas. The covariance update remains well-defined when (L) is singular because it only pushes forward the finite Gaussian covariance and adds (kappa hI). The vector bounded-displacement argument uses norm bounds and Gaussian exponential moments; invertibility of (L) is not required.

## Scope of the claim

This is a vector tail-control extension for the specified finite-step operator. It does not claim a new general matrix Gaussian integral theorem. The Gaussian quadratic integral is standard linear algebra; the paper-level content is its application as a strict integrability certificate with bounded residual remainders.

When the quadratic coefficients are deterministic and common to all histories, every history has the same strict-domain decision and no branch enumeration is required. Exact enumeration remains available for a small explicitly listed collection of candidate full-tail arrays. The scalar extreme-batch reduction does not extend to noncommuting matrices. No claim is made for arbitrary (G), adaptive batch selection, continuous-time limits, finite-particle normalized bias, or resampling stability.

The strict path (p)-moment statement concerns a fixed no-resampling path under the corresponding (p)-scaled quadratic coefficient. It does not imply a resampling variance bound. Any resampling theorem would require separate genealogical and normalized-particle assumptions, including the relevant global moment conditions.

## Evidence produced locally

`tests/test_matrix_tail_certificate_20260921.py` passes nine CPU tests after the main-task numerical integration cross-check:

1. diagonal matrices reduce to the scalar coordinate formula;
2. nonsymmetric factor matrices are rejected;
3. a genuinely noncommuting symmetric pair is accepted and produces a symmetric precision;
4. finite and divergent examples agree with direct eigenvalue-based precision checks;
5. the $K=1$ mean and log-normalizer agree with independent NumPy linear algebra;
6. mixed-sign precision detects divergence from one negative eigenvalue;
7. a singular (L) preserves a symmetric finite covariance update;
8. invalid (kappa), mean, and covariance inputs are rejected;
9. independent SciPy two-dimensional quadrature agrees with the non-diagonal Gaussian normalizer.

The test suite contains no mocked sampler, no remote execution, and no GPU run. The current evidence validates the algebraic certificate and its local numerical implementation. It does not validate a full particle method or application-level accuracy.

## Scientific gaps that remain

The extension should retain the following boundaries in any manuscript-facing use:

- equality at the precision boundary remains unclassified;
- the bounded vector-envelope constants should be stated as finite-history constants when the result is used in a theorem;
- exact enumeration applies only to small explicitly listed candidate arrays;
- resampling variance and normalized finite-particle error require separate analysis;
- shared covariance within each factor is an assumption and does not cover arbitrary state-dependent or factor-dependent covariance fields;
- no application claim follows from this certificate alone. The sensor-model interpretation remains outside this extension's evidence.

## Files and verification

The bounded write set is:

- `manuscript/matrix_tail_extension.tex`
- `docs/MATRIX_TAIL_REVIEW_20260921.md`
- `matrix_tail_certificate_20260921.py`
- `tests/test_matrix_tail_certificate_20260921.py`

The local command

```text
/Users/zenghang/Documents/Codex/2026-09-09/https-github-com-freezeng123456-iclr-2027/work/venv/bin/python -m pytest -q tests/test_matrix_tail_certificate_20260921.py
```

completed with 8 passed tests. No other file was modified, and no external dissemination or remote job was used.
