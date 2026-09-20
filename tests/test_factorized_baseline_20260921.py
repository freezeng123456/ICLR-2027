import math
import os

import numpy as np
import pytest
import torch

from composition_benchmark import FactorModel
from factorized_baseline_20260921 import factorized_sample, product_coupling
from gaussian_path_moments_20260921 import path_moment


def test_one_step_gaussian_factorized_distribution():
    torch.set_num_threads(1)
    device = os.environ.get("ICLR_TEST_DEVICE", "cpu")
    model = FactorModel(4, 3, "gaussian", "cpu")
    parameters = {key: getattr(model, key).numpy() for key in ["variance", "means", "weights"]}
    u, h = 1.0, 0.05
    alpha = math.exp(-u / 2)
    variance = 1 - alpha ** 2 + alpha ** 2 * parameters["variance"]
    a = 1 - 1 / variance
    b = alpha * parameters["means"][..., 0] / variance
    aa, bb = a.sum(0), b.sum(0)
    quadratic = (aa ** 2 - (a ** 2).sum(0)) / 2
    linear = aa * bb - (a * b).sum(0)
    tilted_variance = 1 / (1 - 2 * h * quadratic)
    tilted_mean = h * linear * tilted_variance
    multiplier = 1 - h / 2 + h * aa
    expected_mean = multiplier * tilted_mean + h * bb
    expected_variance = multiplier ** 2 * tilted_variance + h
    samples, weights, _, records = factorized_sample(parameters, [u, u - h], 65536, 461, device)
    assert np.all(weights == 1 / 65536)
    np.testing.assert_allclose(samples.mean(0), expected_mean, atol=0.02)
    np.testing.assert_allclose(samples.var(0), expected_variance, atol=0.03)
    assert records[0]["resampled_coordinates"] == 3
    covariance = np.cov(samples.T)
    assert np.max(abs(covariance - np.diag(np.diag(covariance)))) < 0.02


@pytest.mark.parametrize("ess_fraction", [0.0, 1.0])
def test_multistep_gaussian_moments_and_normalizer(ess_fraction):
    device = os.environ.get("ICLR_TEST_DEVICE", "cpu")
    model = FactorModel(4, 3, "gaussian", "cpu")
    params = {key: getattr(model, key).numpy() for key in ["variance", "means", "weights"]}
    params["variance"] = 0.8 + 0.2 * params["variance"]
    grid = np.linspace(1.0, 0.0, 17)
    assert path_moment(params, grid, 2)["finite"]
    mean, variance, logz = np.zeros(3), np.ones(3), np.zeros(3)
    for u, next_u in zip(grid[:-1], grid[1:]):
        h = u - next_u
        v = 1 - np.exp(-u) + np.exp(-u) * params["variance"]
        a, b = 1 - 1 / v, np.exp(-u / 2) * params["means"][..., 0] / v
        aa, bb = a.sum(0), b.sum(0)
        quadratic = (aa ** 2 - (a ** 2).sum(0)) / 2
        linear = aa * bb - (a * b).sum(0)
        constant = (bb ** 2 - (b ** 2).sum(0)) / 2
        denominator = 1 - 2 * h * quadratic * variance
        tilted_variance = variance / denominator
        tilted_mean = (mean / variance + h * linear) * tilted_variance
        logz += h * constant - np.log(denominator) / 2 - mean ** 2 / (2 * variance) + tilted_mean ** 2 / (2 * tilted_variance)
        multiplier = 1 - h / 2 + h * aa
        mean = multiplier * tilted_mean + h * bb
        variance = multiplier ** 2 * tilted_variance + h
    samples, _, actual_logz, records = factorized_sample(params, grid, 65536, 759, device, ess_fraction)
    np.testing.assert_allclose(samples.mean(0), mean, atol=0.025)
    np.testing.assert_allclose(samples.var(0), variance, atol=0.03)
    np.testing.assert_allclose(actual_logz, logz.sum(), atol=0.04)
    assert abs(np.prod(np.cos(samples), axis=1).mean() - np.prod(np.exp(-variance / 2) * np.cos(mean))) < 0.015
    if ess_fraction:
        assert sum(row["resampled_coordinates"] for row in records) > 3


def test_product_coupling_preserves_marginals_and_joint_expectation():
    values = torch.tensor([[-2, -1, 1], [-1, -0.5, 2], [1, 0.5, 3], [2, 1, 4]], dtype=torch.float64)
    generator = torch.Generator().manual_seed(178)
    observations = []
    for _ in range(2000):
        actual = product_coupling(values, generator)
        torch.testing.assert_close(actual.sort(dim=0).values, values.sort(dim=0).values)
        observations.append(torch.cos(actual).prod(dim=1).mean().item())
    target = torch.cos(values).mean(dim=0).prod().item()
    assert abs(np.mean(observations) - target) < 0.006


def test_finite_normalizer_does_not_imply_finite_path_second_moment():
    model = FactorModel(4, 3, "gaussian", "cpu")
    params = {key: getattr(model, key).numpy() for key in ["variance", "means", "weights"]}
    grid = np.linspace(1.0, 0.0, 17)
    first, second = path_moment(params, grid, 1), path_moment(params, grid, 2)
    assert first["finite"] and not second["finite"]
    np.testing.assert_allclose(first["log_moment"], 3.6156738442, atol=1e-9)
    np.testing.assert_allclose([row["minimum_eigenvalue"] for row in first["coordinates"]],
                               [0.3679553329, 0.3675610090, 0.3677065739], atol=1e-9)
    assert all(row["minimum_eigenvalue"] < -0.26 for row in second["coordinates"])
