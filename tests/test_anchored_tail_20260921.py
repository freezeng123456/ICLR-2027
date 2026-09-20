import itertools
import math

import numpy as np
import pytest
import torch

from anchored_tail_20260921 import AnchoredTailModel
from composition_extension import ExtensionModel


GRID = np.linspace(2.0, 0.0, 5)


def make_parameters(groups=4, dimension=2, gaussian=False):
    rng = np.random.default_rng(600 + groups + dimension)
    variance = rng.uniform(0.35, 0.8, size=(groups, dimension)).astype(np.float64)
    if gaussian:
        center = rng.normal(0.0, 0.6, size=(groups, dimension))
        means = np.stack((center, center), axis=-1).astype(np.float64)
    else:
        means = rng.normal(0.0, 0.8, size=(groups, dimension, 2)).astype(np.float64)
    weights_left = rng.uniform(0.2, 0.8, size=(groups, dimension))
    weights = np.stack((weights_left, 1 - weights_left), axis=-1).astype(np.float64)
    return {"variance": variance, "means": means, "weights": weights}


def test_all_wor_subsets_are_unbiased_for_drift_and_potential():
    model = AnchoredTailModel(4, 2, "mixture", "cpu", parameters=make_parameters(), grid=GRID)
    subsets = torch.tensor(list(itertools.combinations(range(4), 2)), dtype=torch.long)
    for u in [0.0, 1.0, 2.0]:
        for point in [[0.2, -0.4], [-1.1, 0.7], [1.3, 1.8]]:
            x = torch.tensor([point], dtype=torch.float64)
            estimates = []
            potential_values = []
            for subset in subsets:
                indexed = subset[None]
                drift, potential = model.estimate(x, u, 2, torch.Generator().manual_seed(2), control=True, indices=indexed)
                estimates.append(drift[0])
                potential_values.append(potential[0])
            exact_drift, exact_potential = model.exact(x, u)
            torch.testing.assert_close(torch.stack(estimates).mean(0), exact_drift[0], atol=2e-12, rtol=2e-12)
            torch.testing.assert_close(torch.stack(potential_values).mean(), exact_potential[0], atol=2e-12, rtol=2e-12)


def test_changed_intercept_preserves_exact_slopes_and_global_boundedness():
    parameters = make_parameters(5, 2)
    model = AnchoredTailModel(5, 2, "mixture", "cpu", parameters=parameters, grid=GRID)
    entry = model.anchor_parameters(1.0)
    x = torch.linspace(-4, 4, 101, dtype=torch.float64)[:, None].repeat(1, 2)
    original = ExtensionModel(5, 2, "mixture", "cpu", parameters=parameters)
    exact_residual, _ = original.residuals(x, 1.0, torch.arange(5)[None].expand(len(x), -1))
    a = torch.as_tensor(entry["a"], dtype=torch.float64)
    b = torch.as_tensor(entry["b_anchored"], dtype=torch.float64)
    anchored_residual = exact_residual - (a[None] * x[:, None] + b[None])
    assert torch.isfinite(anchored_residual).all()
    slopes_before = torch.as_tensor(model._numpy_coefficients(1.0)[0], dtype=torch.float64)
    torch.testing.assert_close(a, slopes_before, atol=1e-14, rtol=1e-14)
    shift = b - torch.as_tensor(entry["b_original"], dtype=torch.float64)
    e0 = torch.as_tensor(model._numpy_coefficients(1.0)[2], dtype=torch.float64)
    edelta = torch.as_tensor(model._numpy_coefficients(1.0)[3], dtype=torch.float64)
    endpoint_values = torch.stack((e0 - shift, e0 + edelta - shift))
    probes = torch.tensor([-1e6, 1e6], dtype=torch.float64)[:, None].repeat(1, 2)
    probe_residual, _ = original.residuals(probes, 1.0, torch.arange(5)[None].expand(len(probes), -1))
    probe_residual = probe_residual - (a[None] * probes[:, None] + b[None])
    lower, upper = endpoint_values.amin(0), endpoint_values.amax(0)
    for values in [anchored_residual, probe_residual]:
        assert torch.isfinite(values).all()
        assert (values >= lower[None] - 2e-9).all()
        assert (values <= upper[None] + 2e-9).all()


def test_gaussian_anchor_has_zero_residual():
    model = AnchoredTailModel(6, 3, "gaussian", "cpu", parameters=make_parameters(6, 3, True), grid=GRID)
    subsets = torch.tensor(list(itertools.combinations(range(6), 2)), dtype=torch.long)
    x = torch.randn(len(subsets), 3, dtype=torch.float64)
    residual, _ = model.residuals(x, 1.0, subsets[: len(x)])
    entry = model.anchor_parameters(1.0)
    a = torch.as_tensor(entry["a"], dtype=torch.float64)
    b = torch.as_tensor(entry["b_anchored"], dtype=torch.float64)
    torch.testing.assert_close(residual - (a[subsets[:len(x)]] * x[:, None] + b[subsets[:len(x)]]), torch.zeros_like(residual), atol=1e-12, rtol=1e-12)


def test_preparation_cost_and_fixed_protocol():
    model = AnchoredTailModel(7, 2, "mixture", "cpu", parameters=make_parameters(7, 2), grid=GRID)
    report = model.cost_report()
    assert report["preparation_calls"] == len(GRID) * 7 * 7
    assert report["preparation_factor_calls_per_grid_point"] == 49
    assert report["newton_iterations"] == 6


def test_learnedlike_conditional_variance_is_diagnostic_only():
    model = AnchoredTailModel(8, 2, "mixture", "cpu", parameters=make_parameters(8, 2), grid=GRID)
    x = torch.tensor([[0.35, -0.15]], dtype=torch.float64).repeat(4096, 1)
    keys = torch.rand((len(x), 8), generator=torch.Generator().manual_seed(603), dtype=torch.float64)
    indices = keys.topk(3, dim=1, largest=False).indices
    exact, _ = model.exact(x[:1], 1.0)
    estimates, _ = model.estimate(x, 1.0, 3, torch.Generator().manual_seed(602), control=True, indices=indices)
    variance = (estimates - exact).square().mean(0)
    assert torch.isfinite(variance).all()
    assert variance.numel() == 2


def test_cache_is_torch_native_and_runtime_costs_are_separate():
    model = AnchoredTailModel(5, 2, "mixture", "cpu", parameters=make_parameters(5, 2), grid=GRID)
    entry = model._torch_entry(1.0)
    assert all(isinstance(entry[key], torch.Tensor) for key in ["a", "b", "total_a", "total_b", "a2", "ab", "b2"])
    report = model.cost_report()
    assert report["preparation_seconds"] >= 0
    x = torch.zeros((16, 2), dtype=torch.float64)
    model.estimate(x, 1.0, 2, torch.Generator().manual_seed(31), control=True)
    assert model.cost_report()["preparation_seconds"] == report["preparation_seconds"]


@pytest.mark.parametrize("grid", [[-1], [0, 1], [1, np.nan], []])
def test_invalid_grid_is_rejected(grid):
    with pytest.raises(ValueError):
        AnchoredTailModel(4, 2, "mixture", "cpu", parameters=make_parameters(), grid=grid)
