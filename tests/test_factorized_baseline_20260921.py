import math
import os

import numpy as np
import torch

from composition_benchmark import FactorModel
from factorized_baseline_20260921 import factorized_sample


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
