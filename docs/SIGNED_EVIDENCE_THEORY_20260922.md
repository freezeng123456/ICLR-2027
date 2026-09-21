# Growth-optimal proposals for a signed integral

Status: a derived proposition and an implementable development experiment. Novelty is unresolved. Optimality below concerns a specific randomized importance-sampling and linear-betting class, not all integration, stratification, or decision algorithms.

## Mathematical setting

Let $(\Omega,\mathcal A,\mu)$ be a measure space and let $f$ be a measurable integrable real function. Define $f_+=\max(f,0)$, $f_-=\max(-f,0)$, $A=\int f_+d\mu$, and $B=\int f_-d\mu$. Assume $0<B<A<\infty$. The task is to establish the sign of $I=A-B$ through point evaluations of $f$.

For a probability density $q$ that is positive almost everywhere on $\{f\ne0\}$, draw $\theta\sim q$ and form $X=f(\theta)/q(\theta)$. Consider a linear evidence increment $1+\lambda X$, where $\lambda\ge0$ and $1+\lambda f/q>0$ almost everywhere. Its expected logarithmic growth is

\[
G(q,\lambda)=\int q\log(1+\lambda f/q)\,d\mu.
\]

An infinite negative integral is permitted and cannot maximize this objective. The positive part is integrable because $\log(1+x)\le x$ for $x\ge0$.

## Proposition: global optimum within the stated class

Among densities assigning no mass to $\{f=0\}$ and admissible scalar bets, the maximizing proposal and bet are

\[
q_{w_*}=w_*\frac{f_+}{A}+(1-w_*)\frac{f_-}{B},\qquad
w_*=\frac{A}{A-B}-\frac{1}{\log(A/B)},
\]

\[
\lambda_* = \frac{(A-B)w_*(1-w_*)}{AB}.
\]

The optimum is unique up to sets of measure zero. Allowing positive proposal mass on $\{f=0\}$ does not increase the maximum.

### Proof

Fix $w=q(f>0)\in(0,1)$. Conditional Jensen inequalities on the two sign regions give

\[
G(q,\lambda)\le
w\log\left(1+\lambda A/w\right)
+(1-w)\log\left(1-\lambda B/(1-w)\right).
\]

For a positive admissible bet, equality holds precisely when $f/q$ is constant on each region. This gives $q=q_w$. The right hand side is strictly concave in $\lambda$, and its unique stationary point is

\[
\lambda_*(w)=\frac{(A-B)w(1-w)}{AB}.
\]

It lies strictly between zero and $(1-w)/B$. Substitution yields

\[
g(w)=\log(A(1-w)+Bw)-w\log B-(1-w)\log A.
\]

The derivative is $(B-A)/(A(1-w)+Bw)+\log(A/B)$, and the second derivative is strictly negative. Setting the derivative to zero gives the displayed $w_*$. The logarithmic mean $(A-B)/\log(A/B)$ lies strictly between $B$ and $A$, which establishes $w_*\in(0,1)$ and uniqueness.

If a density puts mass $c$ on $\{f=0\}$, normalize its restriction to the complement as $\bar q=q/(1-c)$. Its objective equals $(1-c)G(\bar q,\lambda/(1-c))$. Since the optimum for $A>B$ is positive, allocating $c>0$ cannot improve it. This completes the optimization argument.

For $A=B$, the maximum is zero at $\lambda=0$; uniqueness does not hold. The continuous formula for $w_*$ tends to $1/2$. For $A<B$, apply the proposition to $-f$ and exchange the two regions.

## Difference from variance minimization

For ordinary randomized importance sampling, the variance-minimizing density is $q\propto |f|$, giving positive-region mass $w_{\rm var}=A/(A+B)$. The two-region second moment is $A^2/w+B^2/(1-w)$; differentiating proves this allocation directly. At $A/B=9$, the evidence-growth optimum is approximately0.670, whereas the variance optimum is0.900.

This comparison does not establish dominance over deterministic stratification. A paired estimator can remove the random choice of region, and known pointwise bounds can eliminate queries to one region. Both are explicit controls in the development experiment. Ideal $q_w$ also uses unknown $A,B$; its formula supplies a reference objective, not a deployable sampler with free information.

## Anytime validity and posterior decisions

Let $\mathcal F_{t-1}$ contain all previous queries. A proposal $q_t$ and bet $\lambda_t$ chosen using only this information satisfy

\[
\mathbb E[X_t\mid\mathcal F_{t-1}]=I.
\]

If $I\le0$ and $1+\lambda_tX_t\ge0$, then the product $M_t=\prod_{s\le t}(1+\lambda_sX_s)$ is a nonnegative supermartingale starting at one. Ville's inequality yields $\Pr(\sup_tM_t\ge1/\alpha)\le\alpha$. This is an application of established test-supermartingale theory. No novelty is claimed for this validity argument.

For an unnormalized posterior $\gamma$ with $0<Z=\int\gamma<\infty$, and integrable bounded action losses, set $f=\gamma(L_b-L_a)$. Then $I=Z(R_b-R_a)$, so its sign does not require computing $Z$. An anytime certificate concerns numerical risk ordering under the specified posterior. It does not certify the posterior model's correctness or real-world safety. For multiple actions, the error budget must be allocated across every comparison that can determine the announced action.

The implemented two-sided experiment uses two processes, each at threshold $2/\alpha$. Proposal and bet updates use past observations. Known global function bounds ensure nonnegative increments for every possible future observation. Empirical minima or maxima cannot replace these global bounds. There is no clipping of current observations or retrospective choice of their bets.

## Related-work boundary

Waudby-Smith and Ramdas develop bounded-mean inference through nonnegative martingales and confidence sequences. The sequential guarantee above belongs to this established framework. [Primary paper page](https://arxiv.org/abs/2010.09686).

Fischer and Ramdas use log-optimal betting for anytime-valid Monte-Carlo permutation and resampling tests. That is a close computational-testing precedent; this candidate optimizes the proposal for a signed integral. [Primary paper page](https://arxiv.org/abs/2401.07365).

Owen describes positivisation and estimating-equation approaches to self-normalized importance sampling. Signed decomposition and avoiding a separately estimated normalizer cannot be claimed as new ideas. [Primary paper page](https://arxiv.org/abs/2510.00389).

Active Statistical Inference adapts data collection while preserving valid uncertainty statements, and Active Importance Sampling explicitly targets difficult variational objectives. These precedents require comparison beyond a plain posterior-fitting baseline. [ICML 2024](https://proceedings.mlr.press/v235/zrnic24a.html); [MSML 2022](https://proceedings.mlr.press/v145/rotskoff22a.html).

This primary-source screening has not established that the displayed optimizer is unpublished. The remaining research question is whether optimizing valid evidence increments gives a practically useful method after accounting for proposal learning, stratification, known bounds, and wall time.
