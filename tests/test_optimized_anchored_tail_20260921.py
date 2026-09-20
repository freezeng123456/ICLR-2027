import itertools
import math
import os

import numpy as np
import pytest
import torch
from scipy.special import expit

from anchored_tail_20260921 import AnchoredTailModel
from certified_composition_torch import FactorParameters
from optimized_anchored_tail_20260921 import OptimizedAnchoredTailModel
from run_anchored_confirmation_20260921 import anchored_sample


def params(groups=8, dimension=3, gaussian=False):
    rng = np.random.default_rng(20260921 + groups * 17 + dimension)
    variance = rng.uniform(0.35, 0.85, (groups, dimension))
    if gaussian:
        center = rng.normal(0, 0.7, (groups, dimension))
        means = np.stack((center, center), -1)
    else:
        means = rng.normal(0, 0.8, (groups, dimension, 2))
    left = rng.uniform(0.2, 0.8, (groups, dimension))
    weights = np.stack((left, 1 - left), -1)
    return {"variance": variance, "means": means, "weights": weights}


def test_optimized_matches_parent_on_heterogeneous_points_times_and_subsets():
    parameters = params(8, 3)
    grid = np.array([4.0, 1.0, 0.2, 0.0])
    parent = AnchoredTailModel(8, 3, "learned", "cpu", "without_replacement", parameters, grid)
    optimized = OptimizedAnchoredTailModel(8, 3, "learned", "cpu", "without_replacement", parameters, grid)
    subsets = torch.tensor(list(itertools.combinations(range(8), 4)), dtype=torch.long)
    points = [torch.tensor([20.0, -20.0, 0.5], dtype=torch.float64), torch.tensor([-3.0, 4.0, 2.0], dtype=torch.float64), torch.tensor([0.1, -0.2, 0.4], dtype=torch.float64)]
    for u in [0.0, 0.2, 1.0, 4.0]:
        for point in points:
            x = point[None].repeat(len(subsets), 1)
            parent_drift, parent_potential = parent.estimate(x, u, 4, torch.Generator().manual_seed(7), control=True, indices=subsets)
            optimized_drift, optimized_potential = optimized.estimate(x, u, 4, torch.Generator().manual_seed(7), control=True, indices=subsets)
            torch.testing.assert_close(optimized_drift, parent_drift, atol=2e-12, rtol=2e-12)
            torch.testing.assert_close(optimized_potential, parent_potential, atol=2e-11, rtol=2e-11)


def test_optimized_residual_matches_analytic_endpoints_and_large_probes():
    parameters = params(8, 3)
    grid = np.array([4.0, 1.0, 0.2, 0.0])
    model = OptimizedAnchoredTailModel(8, 3, "learned", "cpu", "without_replacement", parameters, grid)
    for u in grid:
        index = model._time_index(u)
        entry = model._cache[float(u)]
        numpy_coefficients = model._numpy_coefficients(u)
        e0, edelta = numpy_coefficients[2], numpy_coefficients[3]
        shift = entry["b_anchored"] - entry["b_original"]
        endpoint = np.stack((e0 - shift, e0 + edelta - shift))
        x = torch.tensor([[-1e6, -1e6, -1e6], [1e6, 1e6, 1e6]], dtype=torch.float64)
        indices = torch.tensor([[0, 1, 2], [5, 6, 7]], dtype=torch.long)
        actual = model._torch_banks["e0shift"][index][indices] + model._torch_banks["edelta"][index][indices] * torch.sigmoid(x[:, None, :] * model._torch_banks["slope"][index][indices] + model._torch_banks["intercept"][index][indices])
        selected = indices.numpy()
        slope, intercept = numpy_coefficients[4:]
        reference = (e0 - shift)[selected] + edelta[selected] * expit(x.numpy()[:, None, :] * slope[selected] + intercept[selected])
        np.testing.assert_allclose(actual.numpy(), reference, atol=2e-13, rtol=2e-13)
        assert (actual.numpy() >= endpoint.min(0)[selected] - 2e-13).all()
        assert (actual.numpy() <= endpoint.max(0)[selected] + 2e-13).all()
        assert np.isfinite(endpoint).all()
        assert torch.isfinite(actual).all()
        assert float(actual.abs().max()) < 100


def test_gaussian_nonzero_identical_components_have_exact_zero_residual():
    parameters = params(8, 3, gaussian=True)
    model = OptimizedAnchoredTailModel(8, 3, "gaussian", "cpu", "without_replacement", parameters, np.array([4.0, 1.0, 0.2, 0.0]))
    x = torch.tensor([[20.0, -20.0, 0.7], [0.2, -0.4, 1.2]], dtype=torch.float64)
    indices = torch.tensor([[0, 2, 5, 7], [1, 3, 4, 6]], dtype=torch.long)
    for u in [0.0, 0.2, 1.0, 4.0]:
        drift, potential = model.estimate(x, u, 4, torch.Generator().manual_seed(17), control=True, indices=indices)
        parent = AnchoredTailModel(8, 3, "gaussian", "cpu", "without_replacement", parameters, np.array([4.0, 1.0, 0.2, 0.0]))
        expected_drift, expected_potential = parent.estimate(x, u, 4, torch.Generator().manual_seed(17), control=True, indices=indices)
        assert float(model._torch_banks["e0shift"][model._time_index(u)].abs().max()) <= 1e-12
        assert float(model._torch_banks["edelta"][model._time_index(u)].abs().max()) <= 1e-12
        torch.testing.assert_close(drift, expected_drift, atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(potential, expected_potential, atol=2e-11, rtol=2e-11)


def test_exact_numpy_anchor_and_optimized_banks_are_consistent():
    grid = np.array([4.0, 1.0, 0.2, 0.0])
    model = OptimizedAnchoredTailModel(8, 3, "learned", "cpu", "without_replacement", params(8, 3), grid)
    for index, u in enumerate(grid):
        entry = model._cache[float(u)]
        torch.testing.assert_close(model._torch_banks["a"][index], torch.as_tensor(entry["a"]), atol=0, rtol=0)
        torch.testing.assert_close(model._torch_banks["b"][index], torch.as_tensor(entry["b_anchored"]), atol=0, rtol=0)
        torch.testing.assert_close(model._torch_banks["total_a"][index], torch.as_tensor(entry["a"].sum(0)), atol=0, rtol=0)
        torch.testing.assert_close(model._torch_banks["total_b"][index], torch.as_tensor(entry["b_anchored"].sum(0)), atol=0, rtol=0)


def test_frozen_multistep_sampler_matches_optimized_cpu_and_optional_cuda():
    device_name = os.environ.get("ICLR_TEST_DEVICE", "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requested CUDA device is unavailable")
    device = torch.device(device_name)
    grid = np.linspace(math.sqrt(20.0), 0.0, 129, dtype=np.float64) ** 2
    parameters = params(8, 3)
    factor_parameters = FactorParameters(**parameters)
    if device.type == "cuda":
        factor_parameters = FactorParameters(factor_parameters.variance.to(device), factor_parameters.means.to(device), factor_parameters.weights.to(device))
    parent = AnchoredTailModel(8, 3, "learned", device, "without_replacement", parameters, grid)
    optimized = OptimizedAnchoredTailModel(8, 3, "learned", device, "without_replacement", parameters, grid)
    original = anchored_sample(factor_parameters, parent, grid, 2048, 2717, 4, 0.5, device)
    candidate = anchored_sample(factor_parameters, optimized, grid, 2048, 2717, 4, 0.5, device)
    assert sum(record["resampled"] for record in original[3]) > 1
    assert sum(record["resampled"] for record in candidate[3]) > 1
    assert torch.max(torch.abs(candidate[0] - original[0])).item() <= 1e-8
    assert torch.max(torch.abs(candidate[1] - original[1])).item() <= 1e-10
    assert abs(candidate[2] - original[2]) <= 1e-8
