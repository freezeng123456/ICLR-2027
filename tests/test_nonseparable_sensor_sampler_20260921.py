import itertools

import numpy as np
import pytest
import torch

from nonseparable_sensor_model_20260921 import make_problem
from nonseparable_sensor_sampler_20260921 import SensorScoreBank, annealed_smc, diffusion_sample, sensor_log_likelihood


def test_full_score_independent_log_density_difference():
    problem = make_problem(213, groups=4, dimension=2)
    bank = SensorScoreBank(problem, [1., 0.])
    points = np.array([[.1, -.2], [.8, .4], [-2., 1.]])
    residual, _ = bank.estimate(torch.from_numpy(points), 0, "full")
    derivative = np.empty_like(points)
    for d in range(2):
        delta = np.zeros(2)
        delta[d] = 1e-5
        derivative[:, d] = (problem.log_density(points + delta, time=1.) - problem.log_density(points - delta, time=1.)) / 2e-5
    np.testing.assert_allclose(residual.numpy() - points, derivative, atol=1e-8)


def test_exhaustive_without_replacement_unbiasedness_and_anchor():
    problem = make_problem(44, groups=4, dimension=2)
    bank = SensorScoreBank(problem, [1., .3, 0.])
    points = torch.tensor([[.1, .3], [-1., 2.]], dtype=torch.float64)
    for step in [0, 1]:
        exact_R, exact_g = bank.estimate(points, step, "full")
        for method in ["unbiased", "tail_fixed", "tail_anchored"]:
            values = [bank.estimate(points, step, method, torch.tensor([indices, indices]))
                      for indices in itertools.combinations(range(4), 2)]
            torch.testing.assert_close(torch.stack([v[0] for v in values]).mean(0), exact_R, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(torch.stack([v[1] for v in values]).mean(0), exact_g, atol=1e-12, rtol=1e-12)
        anchor = bank.banks["anchor"][step][None]
        anchor_values = [bank.estimate(anchor, step, "tail_anchored", torch.tensor([indices]))[1]
                         for indices in itertools.combinations(range(4), 2)]
        torch.testing.assert_close(torch.stack(anchor_values), anchor_values[0].expand(6, 1), atol=1e-12, rtol=1e-12)


def test_full_batch_control_matches_full_particle_arrays():
    problem = make_problem(51, groups=4, dimension=2)
    grid = np.linspace(np.sqrt(8), 0, 129) ** 2
    full = diffusion_sample(problem, grid, 512, 77, "full")
    for method in ["unbiased", "tail_fixed", "tail_anchored"]:
        controlled = diffusion_sample(problem, grid, 512, 77, method, batch=4)
        np.testing.assert_allclose(controlled[0], full[0], atol=1e-10)
        np.testing.assert_allclose(controlled[1], full[1], atol=1e-10)
        np.testing.assert_allclose(controlled[2], full[2], atol=1e-10)


def test_sensor_likelihood_ratio_and_tempered_smc():
    problem = make_problem(67, groups=4, dimension=2)
    x = torch.tensor([[.1, .2], [1., -.4]], dtype=torch.float64)
    args = [torch.as_tensor(v) for v in [problem.directions, problem.noise_std, problem.positive_probability, problem.observations]]
    logpost = sensor_log_likelihood(x, *args) - .5 * x.square().sum(1)
    truth = problem.log_density(x.numpy())
    np.testing.assert_allclose(float(logpost[0] - logpost[1]), truth[0] - truth[1], atol=1e-12)
    samples, weights, _, _ = annealed_smc(problem, 16384, 456, stages=64)
    probabilities, means, _ = problem.exact_posterior()
    np.testing.assert_allclose(weights @ samples, probabilities @ means, atol=.035)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA device")
def test_gpu_bank_parity_and_full_batch_trajectory():
    problem = make_problem(32, groups=4, dimension=2)
    cpu = SensorScoreBank(problem, [1., .3, 0.])
    gpu = SensorScoreBank(problem, [1., .3, 0.], "cuda")
    x = torch.tensor([[.1, -.2], [1., 2.]], dtype=torch.float64)
    indices = torch.tensor([[0, 2], [1, 3]])
    for method in ["full", "unbiased", "tail_fixed", "tail_anchored"]:
        expected = cpu.estimate(x, 0, method, indices)
        observed = gpu.estimate(x.cuda(), 0, method, indices.cuda())
        for a, b in zip(expected, observed):
            torch.testing.assert_close(a, b.cpu(), atol=1e-11, rtol=1e-11)
    grid = np.linspace(np.sqrt(8), 0, 129) ** 2
    full = diffusion_sample(problem, grid, 512, 77, "full", device="cuda")
    anchored = diffusion_sample(problem, grid, 512, 77, "tail_anchored", batch=4, device="cuda")
    np.testing.assert_allclose(anchored[0], full[0], atol=1e-10)
    np.testing.assert_allclose(anchored[1], full[1], atol=1e-10)
