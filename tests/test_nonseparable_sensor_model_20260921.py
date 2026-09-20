import itertools
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from nonseparable_sensor_model_20260921 import SensorProblem, make_problem


def _small_problem():
    directions = np.array([[1.0, 0.0], [0.6, 0.8], [-0.8, 0.6]])
    return SensorProblem(directions, np.array([0.7, 0.9, 0.6]),
                         np.array([0.65, 0.8, 0.65]), np.array([0.2, -0.4, 0.7]),
                         np.array([0.3, -0.5]))


def test_factor_shapes_full_covariance_and_noncommutation():
    problem = _small_problem()
    precision, means, weights = problem.factor_parameters()
    assert precision.shape == (3, 2, 2)
    assert means.shape == (3, 2, 2)
    assert weights.shape == (3, 2)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert not np.allclose(precision[0] @ precision[1], precision[1] @ precision[0])


def test_enumerated_posterior_has_normalized_weights_and_spd_covariance():
    problem = _small_problem()
    weights, means, covariance = problem.exact_posterior()
    assert weights.shape == (8,)
    assert means.shape == (8, 2)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights > 0.0)
    assert np.min(np.linalg.eigvalsh(covariance)) > 0.0


def test_direct_density_equals_enumerated_gaussian_mixture():
    problem = _small_problem()
    weights, means, covariance = problem.exact_posterior()
    points = np.array([[0.0, 0.0], [0.4, -0.2], [-1.0, 0.8]])
    mixture = np.asarray([logsumexp(np.log(weights) + np.array([
        -0.5 * (point - mean) @ np.linalg.solve(covariance, point - mean)
        - 0.5 * np.linalg.slogdet(covariance)[1] - np.log(2.0 * np.pi)
        for mean in means])) for point in points])
    assert np.allclose(problem.log_density(points), mixture, atol=1e-10)


def test_exact_normalizer_matches_density_integral_and_json_roundtrip():
    problem = _small_problem()
    assert np.isfinite(problem.log_composition_normalizer())
    assert np.isfinite(problem.log_evidence())
    restored = SensorProblem.from_json(problem.to_json())
    assert np.allclose(restored.directions, problem.directions)
    assert json.loads(problem.to_json())["truth"] == problem.truth.tolist()
    grid = np.linspace(-7.0, 7.0, 281)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    values = np.exp(problem.log_density(np.column_stack((xx.ravel(), yy.ravel())))).reshape(xx.shape)
    assert np.isclose(np.trapezoid(np.trapezoid(values, grid, axis=1), grid), 1.0, atol=2e-3)


def test_brute_grid_moments_match_exact_mixture():
    problem = _small_problem()
    weights, means, covariance = problem.exact_posterior()
    expected_mean = weights @ means
    expected_second = covariance + np.einsum("k,ki,kj->ij", weights, means, means)
    grid = np.linspace(-6.0, 6.0, 241)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    density = np.exp(problem.log_density(points)).reshape(xx.shape)
    dx = grid[1] - grid[0]
    grid_mean = np.array([(density * xx).sum(), (density * yy).sum()]) * dx * dx
    grid_second = np.array([[((density * xx * xx).sum()), (density * xx * yy).sum()],
                            [(density * xx * yy).sum(), (density * yy * yy).sum()]]) * dx * dx
    assert np.allclose(grid_mean, expected_mean, atol=3e-3)
    assert np.allclose(grid_second, expected_second, atol=8e-3)


def test_ou_factor_parameters_and_projected_cdf():
    problem = _small_problem()
    precision0, means0, _ = problem.factor_parameters(0.0)
    precision1, means1, _ = problem.factor_parameters(1.2)
    assert np.allclose(means1, np.exp(-0.6) * means0)
    assert not np.allclose(precision0, precision1)
    values = np.linspace(-3.0, 3.0, 11)
    cdf = problem.projected_cdf(np.array([1.0, 2.0]), values)
    assert np.all(np.diff(cdf) >= 0.0)
    assert np.all((cdf >= 0.0) & (cdf <= 1.0))
    try:
        problem.projected_cdf(np.array([1.0, 2.0]), np.zeros((2, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError("non-flat projected CDF points were accepted")


def test_sensor_likelihood_posterior_has_independent_normalization():
    problem = _small_problem()
    grid = np.linspace(-7.0, 7.0, 281)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    density = np.exp(problem.log_sensor_posterior(points)).reshape(xx.shape)
    integral = np.trapezoid(np.trapezoid(density, grid, axis=1), grid)
    assert np.isclose(integral, 1.0, atol=2e-3)
    variances = problem.noise_std ** 2 + np.sum(problem.directions ** 2, axis=1)
    expected = problem.log_composition_normalizer() + np.sum(
        -0.5 * (np.log(2.0 * np.pi * variances)
                + problem.observations ** 2 / variances)
    )
    assert np.isclose(problem.log_evidence(), expected)
    assert np.isclose(problem.log_evidence(), problem.log_sensor_evidence(), atol=1e-10)


def test_composition_and_sensor_posterior_densities_agree():
    problem = _small_problem()
    points = np.array([[0.0, 0.0], [0.4, -0.2], [-1.0, 0.8]])
    assert np.allclose(problem.log_density(points), problem.log_sensor_posterior(points), atol=1e-10)


def test_sampling_and_reproducible_generation():
    problem = make_problem(13, groups=5, dimension=3, regime="ambiguous")
    samples = problem.sample(np.random.default_rng(7), 4000)
    assert samples.shape == (4000, 3)
    assert np.all(np.isfinite(samples))
    same_a = make_problem(13, groups=5, dimension=3, regime="ambiguous")
    same_b = make_problem(13, groups=5, dimension=3, regime="ambiguous")
    assert np.array_equal(same_a.observations, same_b.observations)


def test_validation_and_time_finite_domain():
    problem = _small_problem()
    for invalid in (-1.0, np.nan, np.inf):
        try:
            problem.factor_parameters(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid time was accepted")
    try:
        SensorProblem(np.zeros((17, 2)), np.ones(17), np.full(17, 0.65), np.ones(17), np.zeros(2))
    except ValueError:
        pass
    else:
        raise AssertionError("G > 16 was accepted")
