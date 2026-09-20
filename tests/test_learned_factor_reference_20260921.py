import copy

import numpy as np
import pytest
from scipy.integrate import dblquad, quad
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm

from learned_factor_reference_20260921 import exact_mixture, metrics, model_error, projected_w1, weighted_cdf_w1


def _parameters():
    angle = (np.arange(5) + .25) * np.pi / 5
    return dict(directions=np.column_stack((np.cos(angle), np.sin(angle))),
                variance=np.array([.41, .53, .62, .46, .71]),
                means=np.array([[-.7, .6], [-.3, .9], [-.8, .2], [-.4, .7], [-.6, .5]]),
                weights=np.array([[.3, .7], [.6, .4], [.45, .55], [.8, .2], [.35, .65]]))


def _normal(mean, covariance):
    return dict(probabilities=np.ones(1), means=np.atleast_2d(mean).astype(float),
                covariance=np.atleast_2d(covariance).astype(float), log_normalizer=0.)


def _direct_log_product(p, points):
    points = np.atleast_2d(points)
    value = norm.logpdf(points).sum(axis=1)
    for direction, variance, means, weights in zip(p["directions"], p["variance"], p["means"], p["weights"]):
        projection = points @ direction
        density = sum(weight * norm.pdf(projection, mean, np.sqrt(variance))
                      for weight, mean in zip(weights, means))
        value += np.log(density) - norm.logpdf(projection)
    return value


def _quad_empirical(values, weights, probability, means, sigma):
    positive = weights > 0
    values, weights = values[positive], weights[positive]
    points = np.unique(values)
    bounds = np.concatenate(([-np.inf], points, [np.inf]))
    total = 0.
    for a, b in zip(bounds[:-1], bounds[1:]):
        level = weights[values <= a].sum()
        result, error = quad(lambda z: abs(level - sum(p * norm.cdf(z, m, sigma)
                            for p, m in zip(probability, means))), a, b, epsabs=2e-11, epsrel=2e-11, limit=200)
        assert error < 2e-9
        total += result
    return total


@pytest.mark.parametrize("probability,means,sigma", [
    (np.array([1.]), np.array([.7]), 1.2),
    (np.array([.2, .8]), np.array([-2., 1.3]), .55),
])
def test_analytic_empirical_cdf_integral_matches_independent_quad(probability, means, sigma):
    values = np.array([-3.1, -.2, .4, 1.5, 5.])
    weights = np.array([.1, .2, 0., .4, .3])
    result = weighted_cdf_w1(values, weights, probability, means, sigma)
    expected = _quad_empirical(values, weights, probability, means, sigma)
    assert abs(result["w1"] - expected) < 2e-9
    assert 0 <= result["quadrature_error_bound"] < 1e-15


def test_degenerate_empirical_distribution_has_exact_gaussian_stoploss_value():
    result = weighted_cdf_w1([0., 0., 99.], [.3, .7, 0.], [1.], [0.], 1.)
    assert result["w1"] == pytest.approx(np.sqrt(2 / np.pi), abs=2e-15)
    assert result["quadrature_error_bound"] == 0
    shifted = weighted_cdf_w1([20., 20., -99.], [.3, .7, 0.], [1.], [20.], 1.)
    assert shifted == result


def test_sorting_duplicate_splitting_and_direction_reversal_preserve_w1():
    exact = _normal([.4], [[.7]])
    x, w = np.array([[-1.], [.2], [2.], [.2]]), np.array([.2, .25, .4, .15])
    expected = projected_w1(x, w, exact, [[1.]])
    order = np.array([3, 1, 0, 2])
    shuffled = projected_w1(x[order], w[order], exact, [[1.]])
    reversed_direction = projected_w1(x, w, exact, [[-1.]])
    assert shuffled == expected
    assert reversed_direction["sliced_w1_32"] == pytest.approx(expected["sliced_w1_32"], abs=3e-15)


def test_single_factor_prior_correction_is_a_normalized_full_gaussian_mixture():
    direction = np.array([.6, .8])
    p = dict(directions=direction[None], variance=np.array([.4]), means=np.array([[-1.2, .7]]),
             weights=np.array([[.3, .7]]))
    exact = exact_mixture(p)
    np.testing.assert_allclose(exact["probabilities"], [.3, .7], atol=2e-15)
    np.testing.assert_allclose(exact["means"], p["means"][0, :, None] * direction, atol=2e-15)
    np.testing.assert_allclose(exact["covariance"], np.eye(2) - .6 * np.outer(direction, direction), atol=2e-15)
    assert abs(exact["log_normalizer"]) < 2e-15


def test_noncommuting_five_factor_exact_density_and_independent_two_dimensional_quadrature():
    p = _parameters()
    exact = exact_mixture(p)
    assert exact["probabilities"].shape == (32,)
    points = np.array([[0., 0.], [1., -.5], [-2., 1.3], [.1, .7]])
    components = np.column_stack([np.log(weight) + multivariate_normal.logpdf(points, mean, exact["covariance"])
                                 for weight, mean in zip(exact["probabilities"], exact["means"])])
    np.testing.assert_allclose(logsumexp(components, axis=1) + exact["log_normalizer"],
                               _direct_log_product(p, points), atol=7e-15, rtol=0)
    integral, error = dblquad(lambda y, x: float(np.exp(_direct_log_product(p, [x, y])[0])),
                              -7, 7, lambda _: -7, lambda _: 7, epsabs=1e-8, epsrel=1e-8)
    assert error < 1e-7
    assert np.exp(exact["log_normalizer"]) == pytest.approx(integral, rel=2e-9, abs=2e-9)
    standard_deviation = np.sqrt(np.diag(exact["covariance"]))
    outside_bound = (norm.cdf(-7, exact["means"], standard_deviation)
                     + norm.sf(7, exact["means"], standard_deviation)).sum(axis=1)
    assert exact["probabilities"] @ outside_bound < 1e-16
    first = np.outer(p["directions"][0], p["directions"][0])
    second = np.outer(p["directions"][1], p["directions"][1])
    assert np.linalg.norm(first @ second - second @ first) > .1


def test_independent_gaussian_proposal_quadrature_recovers_composition_normalizer():
    p = _parameters()
    exact = exact_mixture(p)
    proposal_mean = np.array([.3, -.2])
    proposal_covariance = np.array([[.6, .12], [.12, .5]])
    nodes, weights = np.polynomial.hermite.hermgauss(48)
    xx, yy = np.meshgrid(nodes, nodes, indexing="ij")
    points = proposal_mean + np.sqrt(2) * np.column_stack((xx.ravel(), yy.ravel())) @ np.linalg.cholesky(proposal_covariance).T
    log_proposal = multivariate_normal.logpdf(points, proposal_mean, proposal_covariance)
    importance_weight = np.exp(_direct_log_product(p, points) - log_proposal)
    integral = np.sum(np.outer(weights, weights).ravel() * importance_weight) / np.pi
    assert integral == pytest.approx(np.exp(exact["log_normalizer"]), rel=2e-12, abs=2e-12)


def test_one_component_limit_agrees_with_independent_gaussian_product_completion():
    p = _parameters()
    p["means"][:, 1] = p["means"][:, 0]
    exact = exact_mixture(p)
    precision = np.eye(2)
    linear = np.zeros(2)
    constant = 0.
    for a, v, m in zip(p["directions"], p["variance"], p["means"][:, 0]):
        precision += (1 / v - 1) * np.outer(a, a)
        linear += m / v * a
        constant -= .5 * (np.log(v) + m * m / v)
    covariance = np.linalg.inv(precision)
    expected_mean = np.linalg.solve(precision, linear)
    expected_logz = constant + .5 * linear @ expected_mean - .5 * np.linalg.slogdet(precision)[1]
    np.testing.assert_allclose(exact["means"], np.tile(expected_mean, (32, 1)), atol=1e-15)
    np.testing.assert_allclose(exact["covariance"], covariance, atol=1e-15)
    assert exact["log_normalizer"] == pytest.approx(expected_logz, abs=2e-15)


def test_metrics_use_weighted_population_moments():
    exact = _normal([.5, -.3], [[.7, .2], [.2, .9]])
    x = np.array([[-1., 2.], [.2, -.4], [1.5, .6], [33., -99.]])
    w = np.array([.2, .3, .5, 0.])
    directions = np.array([[1., 0.], [0., 1.]])
    actual = metrics(x, w, exact, directions)
    mean = sum(wi * xi for wi, xi in zip(w, x))
    covariance = sum(wi * np.outer(xi - mean, xi - mean) for wi, xi in zip(w, x))
    assert actual["mean_error"] == pytest.approx(np.linalg.norm(mean - exact["means"][0]) / np.sqrt(2))
    assert actual["covariance_error"] == pytest.approx(np.linalg.norm(covariance - exact["covariance"]) / 2)
    assert "region_mass_tv" not in actual


def test_model_error_matches_analytic_gaussian_shift_and_scale_values():
    standard = _normal([0.], [[1.]])
    shifted = _normal([3.], [[1.]])
    wide = _normal([0.], [[4.]])
    assert model_error(standard, shifted, [[1.]]) == pytest.approx(3., abs=2e-10)
    assert model_error(standard, wide, [[1.]]) == pytest.approx(np.sqrt(2 / np.pi), abs=2e-10)
    assert model_error(standard, standard, [[1.]]) == 0.
    assert model_error(shifted, standard, [[-1.]]) == pytest.approx(3., abs=2e-10)


def test_model_error_handles_far_narrow_gaussians_and_32_directions():
    first = _normal([1000., -2000.], [[.01, 0.], [0., .01]])
    second = _normal([1002., -2001.], [[.01, 0.], [0., .01]])
    angle = np.arange(32) * np.pi / 32
    directions = np.column_stack((np.cos(angle), np.sin(angle)))
    expected = np.mean(np.abs(directions @ np.array([2., -1.])))
    assert model_error(first, second, directions) == pytest.approx(expected, abs=1e-10)


def test_model_error_accepts_frozen_factor_payloads_without_loading_models():
    p = _parameters()
    q = copy.deepcopy(p)
    q["means"] += .2
    directions = np.eye(2)
    assert model_error(p, q, directions) == pytest.approx(model_error(exact_mixture(p), exact_mixture(q), directions), abs=1e-14)
    assert model_error(p, p, directions) == 0.


@pytest.mark.parametrize("values,weights", [
    ([0., 1.], [.5, -.5]), ([0., 1.], [0., 0.]), ([0., 1.], [.2, .2]),
    ([0., np.nan], [1., 0.]), ([0., np.inf], [1., 0.]), ([0., 1.], [1., np.nan]),
    ([0., 1.], [[.5, .5]]), ([], []),
])
def test_invalid_empirical_inputs_fail_before_zero_weight_discard(values, weights):
    with pytest.raises(ValueError):
        weighted_cdf_w1(values, weights, [1.], [0.], 1.)


@pytest.mark.parametrize("field,value", [
    ("variance", [0., .5, .5, .5, .5]), ("variance", [1., .5, .5, .5, .5]),
    ("variance", [np.nan] * 5), ("weights", np.ones((5, 2))),
    ("directions", np.ones((5, 2))), ("means", np.ones((5, 3))),
])
def test_invalid_factor_contract_fails(field, value):
    p = _parameters()
    p[field] = value
    with pytest.raises(ValueError):
        exact_mixture(p)


def test_invalid_reference_covariance_directions_and_sample_shape_fail():
    exact = _normal([0., 0.], [[1., 0.], [0., 1.]])
    with pytest.raises(ValueError):
        projected_w1([[0., 0.]], [1.], exact, [[2., 0.]])
    with pytest.raises(ValueError):
        projected_w1([[0.]], [1.], exact, [[1., 0.]])
    exact["covariance"] = np.array([[1., .3], [0., 1.]])
    with pytest.raises(ValueError):
        projected_w1([[0., 0.]], [1.], exact, [[1., 0.]])
    exact["covariance"] = np.array([[1., 0.], [0., -1.]])
    with pytest.raises(np.linalg.LinAlgError):
        projected_w1([[0., 0.]], [1.], exact, [[1., 0.]])
