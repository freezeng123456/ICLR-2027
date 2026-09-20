import itertools
import math
import os

import numpy as np
import pytest
import torch

from anchored_tail_20260921 import AnchoredTailModel
from certified_composition_torch import FactorParameters, sample as torch_sample
from composition_extension import ExtensionModel
from run_anchored_confirmation_20260921 import GRID, anchored_sample


def parameters(groups=4, dimension=2, gaussian=False):
    rng = np.random.default_rng(20260921 + groups + dimension)
    variance = rng.uniform(0.35, 0.8, (groups, dimension))
    if gaussian:
        center = rng.normal(0, 0.7, (groups, dimension))
        means = np.stack((center, center), -1)
    else:
        means = rng.normal(0, 0.9, (groups, dimension, 2))
    left = rng.uniform(0.2, 0.8, (groups, dimension))
    weights = np.stack((left, 1 - left), -1)
    return {"variance": variance, "means": means, "weights": weights}


def test_cpu_exhaustive_fixed_indices_matches_exact_mean():
    params = parameters()
    model = AnchoredTailModel(4, 2, "learned", "cpu", "without_replacement", params, np.array([2.0, 1.0, 0.0]))
    x = torch.tensor([[0.4, -0.8]], dtype=torch.float64)
    subsets = torch.tensor(list(itertools.combinations(range(4), 2)), dtype=torch.long)
    for u in [2.0, 1.0, 0.0]:
        drifts = []
        potentials = []
        for subset in subsets:
            drift, potential = model.estimate(x, u, 2, torch.Generator().manual_seed(5), control=True, indices=subset[None])
            drifts.append(drift[0])
            potentials.append(potential[0])
        exact_drift, exact_potential = model.exact(x, u)
        torch.testing.assert_close(torch.stack(drifts).mean(0), exact_drift[0], atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(torch.stack(potentials).mean(), exact_potential[0], atol=2e-12, rtol=2e-12)


@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("ICLR_TEST_DEVICE", "cuda") != "cuda", reason="CUDA confirmation test requires ICLR_TEST_DEVICE=cuda")
def test_cuda_model_estimate_and_gpu_wor_indices():
    device = "cuda"
    params = parameters(4, 2)
    model = AnchoredTailModel(4, 2, "learned", device, "without_replacement", params, np.array([2.0, 1.0, 0.0]))
    x = torch.randn((32, 2), dtype=torch.float64, device=device)
    keys = torch.rand((32, 4), dtype=torch.float64, device=device, generator=torch.Generator(device=device).manual_seed(717))
    indices = keys.topk(2, dim=1, largest=False).indices
    drift, potential = model.estimate(x, 1.0, 2, torch.Generator(device=device).manual_seed(718), control=True, indices=indices)
    assert drift.device.type == "cuda" and potential.device.type == "cuda"
    sorted_indices = indices.sort(dim=1).values
    assert torch.all(sorted_indices[:, 1:] > sorted_indices[:, :-1])
    exhaustive = torch.tensor(list(itertools.combinations(range(4), 2)), dtype=torch.long, device=device)
    point = torch.tensor([[0.4, -0.8]], dtype=torch.float64, device=device).repeat(len(exhaustive), 1)
    exact_drift, exact_potential = model.exact(point[:1], 1.0)
    drift_all, potential_all = model.estimate(point, 1.0, 2, torch.Generator(device=device).manual_seed(719), control=True, indices=exhaustive)
    torch.testing.assert_close(drift_all.mean(0), exact_drift[0], atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(potential_all.mean(), exact_potential[0], atol=2e-10, rtol=2e-10)
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("ICLR_TEST_DEVICE", "cuda") != "cuda", reason="CUDA confirmation test requires ICLR_TEST_DEVICE=cuda")
def test_cuda_gaussian_anchored_reduces_full_one_step_with_common_streams():
    device = "cuda"
    params = parameters(4, 2, gaussian=True)
    anchored = AnchoredTailModel(4, 2, "gaussian", device, "without_replacement", params, np.array([1.0, 0.0]))
    original = ExtensionModel(4, 2, "gaussian", device, "without_replacement", params)
    original.control_variance = original.variance
    x = torch.randn((4096, 2), dtype=torch.float64, device=device, generator=torch.Generator(device=device).manual_seed(717))
    keys = torch.rand((4096, 4), dtype=torch.float64, device=device, generator=torch.Generator(device=device).manual_seed(718))
    indices = keys.topk(2, dim=1, largest=False).indices
    original_drift, original_potential = original.estimate(x, 1.0, 2, torch.Generator(device=device).manual_seed(719), control=True, indices=indices)
    anchored_drift, anchored_potential = anchored.estimate(x, 1.0, 2, torch.Generator(device=device).manual_seed(719), control=True, indices=indices)
    torch.testing.assert_close(anchored_drift, original_drift, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(anchored_potential, original_potential, atol=1e-10, rtol=1e-10)
    torch.cuda.synchronize()


def test_cpu_one_step_driver_uses_float64_and_fixed_wor():
    params = parameters(4, 2)
    model = AnchoredTailModel(4, 2, "learned", "cpu", "without_replacement", params, GRID[:3])
    factor_params = FactorParameters(**params)
    samples, weights, logz, records = anchored_sample(
        factor_params,
        model,
        GRID[:3],
        32,
        12,
        2,
        0.5,
        torch.device("cpu"),
    )
    assert samples.dtype == torch.float64 and weights.dtype == torch.float64
    assert torch.isfinite(samples).all() and torch.isfinite(weights).all()
    assert len(records) == 2 and records[-1]["resampled"] is False


def test_gaussian_driver_matches_full_with_common_three_streams():
    device = os.environ.get("ICLR_TEST_DEVICE", "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requested CUDA device is unavailable")
    torch_device = torch.device(device)
    params = parameters(4, 2, gaussian=True)
    factor_params = FactorParameters(**params)
    if torch_device.type == "cuda":
        factor_params = FactorParameters(factor_params.variance.to(torch_device), factor_params.means.to(torch_device), factor_params.weights.to(torch_device))
    anchored = AnchoredTailModel(4, 2, "gaussian", torch_device, "without_replacement", params, np.array([1.0, 0.95, 0.9]))
    full_samples, full_weights, full_logz, _ = torch_sample(factor_params, np.array([1.0, 0.95, 0.9]), 512, 717, method="full", batch=2, device=torch_device, return_type="torch")
    anchored_samples, anchored_weights, anchored_logz, _ = anchored_sample(factor_params, anchored, np.array([1.0, 0.95, 0.9]), 512, 717, 2, 0.5, torch_device)
    torch.testing.assert_close(anchored_samples, full_samples, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(anchored_weights, full_weights, atol=1e-10, rtol=1e-10)
    assert abs(anchored_logz - full_logz) <= 1e-10
