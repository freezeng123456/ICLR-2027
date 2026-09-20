import os

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import norm
import torch

from learned_factor_bridge_20260921 import (LearnedTarget, construct_reference, log_target_numpy,
    parameter_hash, parameters, sample, tail_precision)
from tail_bridge_smc_20260921 import EvaluationCounts


def problem():
    angles = (np.arange(5) + .25) * np.pi / 5
    return dict(directions=np.column_stack((np.cos(angles), np.sin(angles))),
        variance=np.array([.24, .35, .47, .55, .65]),
        means=np.array([[-.6, .8], [-.8, .6], [-.4, .5], [-.8, .3], [-.5, .4]]),
        weights=np.array([[.4, .6], [.6, .4], [.35, .65], [.45, .55], [.2, .8]]))


def test_learned_target_matches_independent_prior_corrected_density():
    p = problem()
    x = np.random.default_rng(18).normal(size=(37, 2))
    independent = norm.logpdf(x).sum(1)
    for a, v, m, w in zip(p["directions"], p["variance"], p["means"], p["weights"]):
        z = x @ a
        independent += logsumexp(np.log(w) + norm.logpdf(z[:, None], m, np.sqrt(v)), axis=1) - norm.logpdf(z)
    np.testing.assert_allclose(log_target_numpy(p, x), independent, atol=1e-12)
    np.testing.assert_allclose(LearnedTarget(p).log_prob(torch.tensor(x)).numpy(), independent, atol=1e-12)


def test_em_reference_uses_only_learned_inputs_and_reports_real_cost():
    p = problem()
    counts = EvaluationCounts()
    weights, means, covariance, search = construct_reference(p, "mixture", counts)
    np.testing.assert_allclose(covariance @ tail_precision(p), np.eye(2), atol=2e-14)
    assert 1 <= len(means) <= 4 and weights.min() > 0
    assert search["starts"] == 16 and search["steps"] == 50
    assert counts.mode_factor_calls == 16 * 52 * 5
    p_with_oracle = dict(p, truth=[999, 999], exact_means=[[-100, 100]])
    other = construct_reference(p_with_oracle, "mixture", EvaluationCounts())
    for actual, expected in zip(other[:3], [weights, means, covariance]):
        np.testing.assert_array_equal(actual, expected)
    assert parameter_hash(p_with_oracle) == parameter_hash(p)


@pytest.mark.parametrize("reference,direct,global_probability", [
    ("prior", False, 0.), ("gaussian", False, 0.), ("mixture", False, 0.),
    ("mixture", True, 0.), ("mixture", False, .1)])
def test_real_gaussian_sampling_moments_normalizer_and_weight_lifecycle(reference, direct, global_probability):
    p = dict(directions=np.ones((1, 1)), variance=np.array([.4]),
             means=np.array([[.7, .7]]), weights=np.array([[.3, .7]]))
    x, w, logz, diagnostic = sample(p, 3127, reference=reference, particles=12000,
                                  direct_is=direct, global_probability=global_probability)
    assert abs((w @ x)[0] - .7) < .025
    assert abs(float(w @ ((x[:, 0] - .7) ** 2)) - .4) < .025
    assert abs(logz) < .04
    assert abs(w.sum() - 1) < 1e-12
    assert diagnostic["records"][-1]["beta"] == 1
    assert diagnostic["records"][-1]["resampled"] is False
    assert diagnostic["counts"]["factor_calls"] > 0
    assert diagnostic["seconds"] >= diagnostic["preparation_seconds"] > 0
    np.testing.assert_allclose(sum(r["log_normalizer_increment"] for r in diagnostic["records"]), logz, atol=1e-12)


def test_direct_importance_weights_match_complete_mixture_density():
    p = problem()
    x, w, logz, diagnostic = sample(p, 141, direct_is=True, particles=1024)
    q = diagnostic["reference_parameters"]
    inverse = np.linalg.inv(q["covariance"])
    delta = x[:, None] - np.asarray(q["means"])
    logq = logsumexp(np.log(q["weights"])[None] - .5 * (
        np.einsum("nki,ij,nkj->nk", delta, inverse, delta) + np.linalg.slogdet(q["covariance"])[1]
        + 2 * np.log(2 * np.pi)), axis=1)
    ratio = log_target_numpy(p, x) - logq
    np.testing.assert_allclose(w, np.exp(ratio - logsumexp(ratio)), rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(logz, logsumexp(ratio) - np.log(len(x)), atol=2e-13)


def test_reject_invalid_factors():
    p = problem()
    p["variance"][0] = 1.5
    with pytest.raises(ValueError):
        parameters(p)
    p = problem()
    p["directions"][0] *= 2
    with pytest.raises(ValueError):
        parameters(p)


@pytest.mark.skipif(os.environ.get("ICLR_TEST_DEVICE") != "cuda", reason="explicit CUDA validation required")
def test_cuda_target_and_real_sampler():
    p = problem()
    x = np.random.default_rng(918).normal(size=(51, 2))
    np.testing.assert_allclose(LearnedTarget(p, "cuda").log_prob(torch.tensor(x, device="cuda")).cpu(),
                               log_target_numpy(p, x), atol=1e-11)
    x, w, logz, diagnostic = sample(p, 92, particles=1024, global_probability=.1, device="cuda")
    assert diagnostic["device"] == "cuda:0" and np.isfinite(x).all() and np.isfinite(logz)
    assert abs(w.sum() - 1) < 1e-12
