# Certified composition Torch implementation

## Interface

`FactorParameters(variance, means, weights)` stores `float64` tensors with shapes `(groups, dimension)`, `(groups, dimension, 2)`, and `(groups, dimension, 2)`. NumPy arrays and Torch tensors are accepted. `sample` also accepts a dictionary with these three keys.

```python
samples, weights, logZ, records = sample(
    params, grid, N, seed, method,
    eta_total=4.0, cost_fraction=0.25, batch=4,
    table_nodes=129, table_radius=8.0,
    device="cuda", return_type="torch",
)
```

`device` moves parameters and sampling tensors. `return_type="torch"` preserves the selected device; `return_type="numpy"` moves the first two outputs to CPU NumPy arrays. `logZ` is a Python float. `records` contains one dictionary per grid interval. `factor_calls` includes preparation calls, deterministic full-factor calls, and sampled Poisson events.

Supported methods are `full`, `tail_fixed`, `certified`, `surrogate_full`, `surrogate_only`, plus compatibility aliases `affine_certified` and `affine_full`. Motion, auxiliary Poisson/component sampling, and systematic resampling use independent `torch.Generator` streams derived from `seed`.

## Mathematical scope

The implementation uses the positive Poisson random-kernel conditional expectation estimator from the NumPy prototype. The deterministic full branch evaluates every factor. The interpolated residual table contributes to the proposal and its preparation cost is counted. Outside the interpolation interval, the certified correction selects the full branch. The envelope controls the conditional second-moment budget through `envelope**2 / rate <= eta`.

`prepare_grid` uses the supplied grid construction and the Gaussian quadratic-tail recursion. Its returned certificate is for `power=1.0`; this implementation does not claim a moment strictly above one or a second-moment theorem. The method follows the requested grid.

## GPU behavior and limitations

The Poisson event path performs `torch.repeat_interleave`, `torch.searchsorted`, vectorized residual evaluation, and `scatter_add_` on the selected device. It contains no per-event Python loop and no event-level GPU-to-CPU transfer. Event count can still dominate memory and runtime when the envelope is large; `records` exposes this through `poisson_events`, `full_particles`, and `preparation_calls`.

Tests cover kernel factorization, exact interpolation accumulation, bounded residuals, conditional Poisson positivity and budget, Gaussian zero residual, Torch full/certified consistency, grid certificate scope, independent-stream reproducibility, dictionary input, and required methods. CPU Torch execution is locally testable. CUDA execution, large `N=32768`, `K=2048`, 250-question workloads, numerical agreement with the legacy NumPy sampler for identical random streams, and multi-GPU behavior remain unverified.

Files written:

- `certified_composition_torch.py`
- `tests/test_certified_composition_torch.py`
- `docs/CERTIFIED_IMPLEMENTATION_20260921.md`
