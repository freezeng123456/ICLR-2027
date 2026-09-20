import numpy as np
import pytest
from scipy.stats import norm

from analyze_posterior_decisions_20260921 import empirical_quantile, metric_row, reference_decisions, sample_decisions


def test_weighted_empirical_inverse_cdf_and_zero_weights():
    actual = empirical_quantile([3., 999., 1.], [.25, 0., .75], [0., .5, .75, 1.])
    np.testing.assert_array_equal(actual, [1., 1., 1., 3.])


@pytest.mark.parametrize("weights, probabilities", [([-1., 2.], [.5]), ([np.nan, 1.], [.5]), ([0., 0.], [.5]), ([.5, .5], [1.01])])
def test_invalid_empirical_inputs_fail(weights, probabilities):
    with pytest.raises(ValueError):
        empirical_quantile([0., 1.], weights, probabilities)


def test_saved_sample_decisions_match_direct_values():
    samples = np.array([[1., 4.], [3., 8.], [-100., -100.]])
    weights = np.array([.25, .75, 0.])
    mean, low90, low95 = sample_decisions(samples, weights)
    np.testing.assert_allclose(mean, [2.5, 7.], atol=0, rtol=0)
    np.testing.assert_array_equal(low90, [[1., 3.], [4., 8.]])
    np.testing.assert_array_equal(low95, low90)
    row = metric_row("sample", np.array([2., 6.]), mean, low90, low95, {})
    assert row["mse"] == .625
    assert row["coverage90"] == row["coverage95"] == 1
    assert row["width90"] == row["width95"] == 3
    with pytest.raises(ValueError):
        sample_decisions(samples, weights[:, None])


def test_numerical_reference_matches_known_gaussian(tmp_path):
    grid = np.linspace(-10., 10., 100001)
    path = tmp_path / "normal-reference.npz"
    np.savez(path, grid=grid, density=norm.pdf(grid)[None], cdf=norm.cdf(grid)[None])
    mean, interval90, interval95 = reference_decisions(path)
    np.testing.assert_allclose(mean, [0.], atol=1e-13, rtol=0)
    np.testing.assert_allclose(interval90[0], norm.ppf([.05, .95]), atol=1e-8, rtol=0)
    np.testing.assert_allclose(interval95[0], norm.ppf([.025, .975]), atol=1e-8, rtol=0)
