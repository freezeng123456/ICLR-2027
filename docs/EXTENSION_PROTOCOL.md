# Composition extension protocol

Frozen before SCNet submission, 2026-09-10. This supplements the completed 1980-cell baseline. Compute is authorized without a monetary/GPU-hour cap by the user's 2026-09-09 23:51 CST instruction. Scheduler concurrency is capped by the observed account limit of 40 GPUs; each job requests one RTX 3080 and four CPU cores. The original deadline is 2026-09-11 19:49 CST.

## Hypotheses and fixed comparisons

1. Uniform sampling without replacement changes the random quadratic potential and may change population integrability. The unbiased potential is G(G-1)/(2M(M-1)) times the ordered distinct residual-pair sum. The exact scalar nonnegative-strength Gaussian certificate checks M+1 prefix/suffix subsets. Exhaustive enumeration independently checks the implementation. For equal-component-variance mixtures, the existing strict tail-envelope result applies.
2. Complete-trajectory finite-particle error and finite initial-noise error must be measured separately from population integrability. All results, including collapse and negative comparisons, are retained.
3. A learned conditional mixture density has an exact OU-noised score and known Gaussian tail. This permits a controlled synthetic neural-posterior study with no exact score or posterior labels used for training. Claims concern integrability conditional on the learned factors; posterior learning accuracy is measured separately.

## Oracle sampling and sensitivity cells

Sampling matrix: families Gaussian, mixture, weak mixture; G=64; d in {1,8}; K in {512,2048}; seeds 0..4; P=8192; U=20. Full baseline plus raw U and tail control, each with replacement and without replacement, each with M in {2,4,8}. Total 780 cells.

Sensitivity matrix: same three families, G=64,d=8,K=512; seeds 0..4; methods full, raw U with replacement, raw U without replacement, tail with replacement, tail without replacement; M=4. Baseline P=8192,U=20; independently vary P to 32768 and 131072, and U to 10 and 30. Baseline duplicates are shared with the sampling matrix. This adds 300 cells. Exactly 1080 unique oracle cells.

All samplers use diffusion coefficient 1, standard normal initialization, the squared-root U time grid, float64 arithmetic, and systematic resampling below ESS/P=0.5, with no terminal resampling. Sampling without replacement uses float64 independent uniform random keys and top-k selection; this O(PG) selection overhead is included in wall time. Score counts report actual selected factors and do not conceal selection overhead.

## Learned posterior experiment

Simulator: theta~N(0,1), independent sign in {-1,+1} with equal probability, y=theta+sign*b+sigma*epsilon, epsilon~N(0,1). Training contexts have sigma uniform [0.45,1.2] and b uniform [0.3,1.3]. A 3-128-128-128-4 SiLU MLP consumes (y/3,log sigma,b). Outputs are center, softplus separation, shared variance 0.02+0.96 sigmoid, and one logit for two positive mixture weights. Means are center±separation. The network receives no evolving latent state.

Train independent seeds 0..4 for exactly 40000 Adam updates, batch 2048, learning rate 3e-4 cosine-decayed to 3e-5. Fresh simulator pairs at every update; no oracle labels. Fixed final checkpoint, no early stopping or outcome-based selection. The same 100000 heldout pairs (seed 800000) are evaluated every 2000 updates. Conditional KL is independently integrated for 1024 heldout contexts over [-12,12] with 8193 points. Save initial/final checkpoints, finite/changed-parameter audit, loss history, config, runtime and checkpoint hashes.

Each heldout grouped dataset shares one theta per coordinate across observations. Five dataset seeds 0..4, RNG seed 1000+seed; G in {16,64}; d in {1,8}. Deterministic heteroskedastic contexts use phase=2pi(g+0.37d)/G, sigma=.45+.75(.5+.5sin phase), b=.8+.5cos(2 phase). Each observation receives an independent simulator sign and noise. True posterior factors follow Bayes' rule and are verified against the likelihood-prior identity.

For each checkpoint and dataset, evaluate K in {512,2048}, P=32768, U=20,M=4; methods full, raw U with/without replacement, tail with/without replacement. Fixed sampling seed is 100000+1000*checkpoint_seed+dataset_seed. Exactly 1000 learned sampling cells. All methods in a paired comparison share fixed learned parameters, true/learned quadrature references, and checkpoint hash. Reference densities use 65537 points on [-12,12]; independent audit doubles resolution and checks a wider domain. Report true-vs-learned composed KL/W1 separately from sample-vs-learned and sample-vs-true W1/KS/moments.

## Completion and interpretation

Every submitted task must have a successful scheduler exit, expected cell count, finite particle states/weights, normalized weights, recomputed diagnostics, source/config/checkpoint hashes, raw artifacts and independent quadrature checks. A successful finite-particle run does not establish a finite population normalizer. Certificate classifications and observed sampler scores are reported independently. Learned experiments are a structured synthetic study with known mixture architecture, not a general neural-score or real-data SBI claim. Submission failures or numerical failures are retained and investigated before resubmission to a new output directory.
