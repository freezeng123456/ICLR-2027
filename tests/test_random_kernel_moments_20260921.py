import math
import os

import numpy as np
import pytest
import torch

from certified_composition_torch import FactorParameters, TimeFactors, sample
from composition_benchmark import FactorModel
from composition_extension import ExtensionModel


DEVICE = os.environ.get("ICLR_TEST_DEVICE", "cpu")


def params(groups=8, dimension=2, family="mixture"):
    model = FactorModel(groups, dimension, family, DEVICE)
    return FactorParameters(model.variance, model.means, model.weights)


@pytest.mark.parametrize("nodes", [0, 17, 65])
def test_positive_random_correction_first_and_second_moments(nodes):
    n = 160000
    p = params()
    f = TimeFactors(p, 2.0, nodes, 4)
    x = torch.full((n, 2), 0.3, dtype=torch.float64, device=DEVICE)
    y = torch.full((n, 2), 0.5, dtype=torch.float64, device=DEVICE)
    h, eta = 0.02, 0.3
    correction, report = f.random_correction(x, y, h, eta, torch.Generator(device=DEVICE).manual_seed(904 + nodes), 1.0)
    assert report["randomized_particles"] == n and report["poisson_events"] > 20
    z = f.correction_terms(x[:1], y[:1], h)[0]
    ratio = torch.exp(correction - z.sum())
    aa = y[0] - (1 - h / 2) * x[0]
    coeff = torch.cat((aa.abs(), h * x[0].abs(), torch.tensor([h], dtype=torch.float64, device=DEVICE)))
    bound = f.basis @ coeff
    total = bound.sum()
    rate = torch.maximum(1.1 * total, total.square() / eta)
    second = torch.exp((z.square() / (rate * bound / total)).sum())
    assert second <= math.exp(eta) * (1 + 1e-12)
    for values, expected in [(ratio, 1.0), (ratio.square(), second)]:
        standard_error = values.std(unbiased=True) / math.sqrt(n)
        assert abs(values.mean() - expected) <= 6 * standard_error + 1e-12


@pytest.mark.parametrize("family", ["gaussian", "mixture", "weak_mixture"])
def test_full_matches_legacy_model(family):
    legacy = ExtensionModel(16, 3, family, DEVICE, "without_replacement")
    p = FactorParameters(legacy.variance, legacy.means, legacy.weights)
    x = torch.randn(101, 3, dtype=torch.float64, device=DEVICE, generator=torch.Generator(device=DEVICE).manual_seed(134))
    for u in [0.0, 1.0, 20.0]:
        actual_drift, actual_potential = TimeFactors(p, u).full(x)
        expected_drift, expected_potential = legacy.exact(x, u)
        torch.testing.assert_close(actual_drift, 2 * expected_drift, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(actual_potential, expected_potential, atol=1e-9, rtol=1e-10)


def test_kernel_mass_and_first_two_state_moments():
    n, h = 160000, 0.02
    f = TimeFactors(params(4, 2), 1.0, 17, 4)
    x = torch.full((n, 2), 0.3, dtype=torch.float64, device=DEVICE)
    proposal, potential0 = f.proposal(x)
    generator = torch.Generator(device=DEVICE).manual_seed(1306)
    y = (1 - h / 2) * x + h * proposal + math.sqrt(h) * torch.randn(x.shape, dtype=torch.float64, device=DEVICE, generator=generator)
    correction, info = f.random_correction(x, y, h, 0.3, torch.Generator(device=DEVICE).manual_seed(1307), 1.0)
    weight = torch.exp(h * potential0 + correction)
    drift, potential = f.full(x[:1])
    mass = torch.exp(h * potential[0])
    mean = (1 - h / 2) * x[0] + h * drift[0]
    for values, expected in [(weight, mass), (weight * y[:, 0], mass * mean[0]),
                             (weight * y[:, 0].square(), mass * (h + mean[0].square())),
                             (weight * y[:, 0] * y[:, 1], mass * mean[0] * mean[1])]:
        assert abs(values.mean() - expected) < 6 * values.std(unbiased=True) / math.sqrt(n)
    assert info["poisson_events"] > 0


def test_envelope_extreme_states_and_exact_branch():
    generator = torch.Generator(device=DEVICE).manual_seed(309)
    for nodes in [0, 65]:
        f = TimeFactors(params(16, 3), 0.1, nodes, 8)
        x = torch.randn(97, 3, dtype=torch.float64, device=DEVICE, generator=generator) * 100
        y = torch.randn(97, 3, dtype=torch.float64, device=DEVICE, generator=generator) * 100
        h = 0.1
        result, info = f.random_correction(x, y, h, 1e-12, generator)
        expected = f.correction_terms(x, y, h).sum(1)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)
        assert info["full_particles"] == len(x)


@pytest.mark.parametrize("method", ["full", "tail_fixed", "certified", "surrogate_full", "surrogate_only"])
def test_actual_sampling_on_selected_device(method):
    grid = np.linspace(math.sqrt(20), 0, 129) ** 2
    samples, weights, logz, records = sample(params(16, 8), grid, 1024, 1356, method, table_nodes=65, device=DEVICE)
    assert samples.shape == (1024, 8) and np.isfinite(samples).all()
    assert np.isfinite(logz) and len(records) == 128
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-12)
