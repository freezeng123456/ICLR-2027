# Positive random correction: mathematical contract

Status: author-side mathematical derivation, 2026-09-21. This note specifies the algorithm being tested; it is not an independent human verification or a new general particle-convergence theorem.

## Object and restrictions

Fix a finite decreasing time grid, diffusion coefficient one, and diagonal two-component mixture factors with positive weights and common component variance within each factor-coordinate. At a state x let residual scores be r_g=r_g^0+e_g, where the proposal residual r_g^0 contains the exact affine tail and optionally a bounded, piecewise-linear residual interpolant. Write R=sum_g r_g, R0=sum_g r_g^0, and g=(|R|²-sum_g|r_g|²)/2, with g0 defined similarly. The full Euler kernel has density exp(h g(x)) times a normal density with mean (1-h/2)x+hR(x), covariance hI. The proposal has the corresponding mean using R0 and covariance hI.

All statements here concern this discrete kernel. Its normalized endpoint may differ from the continuous compositional posterior.

## Exact factorization and positive Poisson correction

For A=y-(1-h/2)x define z_g=(A-h r_g^0)·e_g-h|e_g|²/2. Expanding the two Gaussian quadratic forms cancels the cross-factor squared-score terms and gives the exact identity

    Q(x,dy) = exp(h g0(x)) q0(x,dy) exp(sum_g z_g(x,y)).

At fixed (x,y), suppose certified deterministic numbers b_g satisfy |z_g|≤b_g. Put B=sum b_g. When B=0, every z_g is zero and the correction is one. Otherwise set p_g=b_g/B on the positive-b_g support and choose lambda≥1.1B and lambda≥B²/eta. Draw N~Poisson(lambda) and independent I_j~p; define C=product_{j=1}^N (1+z_{I_j}/(lambda p_{I_j})). Every factor is at least 1-1/1.1>0. The empty product equals one. The Poisson generating function gives

    E[C | x,y] = exp(sum_g z_g),
    E[C² | x,y] / E[C | x,y]²
      = exp(sum_g z_g²/(lambda p_g)) ≤ exp(B²/lambda) ≤ exp(eta).

Terms with b_g=0 have z_g=0 and contribute neither events nor exponent. Finite G, finite x,y and a positive finite eta make all conditional quantities finite. The implementation evaluates exp(sum z_g) deterministically if its predicted event rate exceeds a preset factor budget or x is outside the interpolation interval. This branch is a deterministic function of x,y and has the same conditional mean, with zero conditional auxiliary variance. It therefore preserves the kernel identity pointwise. The no-correction interpolant ablation does not satisfy this identity in general.

## Finite paths and the second-moment boundary

Without resampling, condition on a complete proposal path X0,...,XK. Use independent auxiliary events at each step, given that path. The random path weight W_hat is the product of exp(h_k g0_k) C_k. Consequently its conditional mean is the exact full-kernel importance weight W on the proposal path, and its conditional relative second moment is at most exp(sum eta_k). Tonelli's theorem for nonnegative integrands then yields

    E[W_hat f(XK)] = E[W f(XK)]  for f≥0,
    E[W_hat²] ≤ exp(sum eta_k) E[W²].

For signed f, require E[W |f(XK)|]<infinity. The first identity preserves the finite-horizon unnormalized full Euler measure, including its possible infinite total mass; it does not turn an undefined full Euler measure into a finite one. The second inequality requires an independently finite E[W²] to give a finite bound. A first-moment certificate or a passing power between one and two is insufficient. Uniform local budgets control additional Poisson randomness, while deterministic importance-weight concentration remains a separate limitation.

The implemented algorithm includes adaptive systematic resampling. Its kernel factors retain the conditional identities above, but the no-resampling path bound does not directly bound its final normalized estimate or its genealogical dependence. No claim of unbiased normalized particle estimates, universal second-moment control after resampling, or finite-N error rate is made here. The experiment measures these finite-particle effects separately.

## Required implementation evidence

Tests check the kernel density identity against the legacy full-score implementation; interpolated squared-residual sums; envelope bounds; actual Poisson first and second moments with nonzero event counts; Gaussian zero-residual reduction; full-computation branches; distinct random streams; and GPU execution. Large-N estimates and observed performance remain experimental statements. Source hashes, raw particles, development/confirmation separation and numerical references are part of that evidence.

## Prior work boundary

Random positive weights and Poisson estimators in particle methods are established ideas. See Paul Fearnhead, Omiros Papaspiliopoulos, Gareth O. Roberts, and Andrew Stuart (2010), *Random-weight particle filtering of continuous time processes*, JRSS B 72(4), 497–512, https://doi.org/10.1111/j.1467-9868.2010.00744.x (bibliographic record and abstract checked at https://authors.library.caltech.edu/records/34d4q-7w428).

The possible contribution being investigated is the score-derived Euler-kernel cancellation, computable tail/interpolation envelopes, and measured accuracy–cost behavior in compositional inference. No claim that Poissonization or random-weight SMC itself is new is made.
