import json
import math
import os

import numpy as np
import pytest
from scipy.integrate import simpson
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm, wasserstein_distance
import torch

from nonseparable_sensor_model_20260921 import SensorProblem, make_problem
from tail_bridge_smc_20260921 import (
    GaussianReference,
    SensorTarget,
    _next_beta,
    _resample,
    _reweight,
    build_reference,
    make_generators,
    metropolis_step,
    sample,
    sign_flip_step,
)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _heterogeneous():
    return SensorProblem(
        np.array([[1.0, 0.2], [0.3, 1.2], [-0.8, 0.6], [0.7, -0.4]]),
        np.array([0.7, 0.9, 0.8, 1.1]), np.array([0.65, 0.8, 0.65, 0.8]),
        np.array([0.4, -0.8, 0.9, 0.2]), np.array([8.0, -9.0]),
    )


def _gaussian_parameters():
    return dict(
        directions=np.array([[1.0, 0.2], [-0.4, 1.1], [0.6, 0.7]]),
        noise_std=np.array([0.7, 0.8, 1.2]),
        positive_probability=np.array([1.0, 0.0, 1.0]),
        observations=np.array([1.7, -0.8, -1.2]),
    )


def _numpy_sensor_joint(problem, points):
    prior = norm.logpdf(points).sum(axis=1)
    likelihood = np.zeros(len(points))
    for a, sigma, pi, y in zip(problem.directions, problem.noise_std,
                               problem.positive_probability, problem.observations):
        projection = points @ a
        terms = np.stack((math.log(1.0 - pi) + norm.logpdf(y, -projection, sigma),
                          math.log(pi) + norm.logpdf(y, projection, sigma)))
        likelihood += logsumexp(terms, axis=0)
    return prior + likelihood


def _moments(probabilities, means, covariance):
    mean = probabilities @ means
    centered = means - mean
    return mean, covariance + (centered * probabilities[:, None]).T @ centered


def _assert_counts_and_metadata(result):
    x, w, log_z, timing, counts, meta = result
    n, d, g = meta["particles"], meta["dimension"], meta["groups"]
    assert x.shape == (n, d)
    assert w.shape == (n,)
    assert np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(log_z)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-13, rtol=0)
    assert (w >= 0).all()
    assert meta["status"] == "completed"
    assert meta["temperatures"][0] == 0 and meta["temperatures"][-1] == 1
    assert np.all(np.diff(meta["temperatures"]) > 0)
    assert counts["reference_samples"] == n
    assert counts["target_point_evaluations"] == n * (1 + meta["stages"] * (meta["moves"] + int(meta["sign_flip"])))
    assert counts["particle_factor_calls"] == g * counts["target_point_evaluations"]
    assert counts["mode_factor_calls"] == g * counts["mode_point_evaluations"]
    assert counts["factor_calls"] == sum(counts[key] for key in (
        "factor_preparation_calls", "mode_factor_calls", "particle_factor_calls"
    ))
    assert counts["resampling_calls"] == meta["stages"] - 1
    assert not meta["records"][-1]["resampled"]
    assert counts["mh_proposals"] == meta["moves"] * n * meta["stages"]
    assert counts["mh_accepted"] == sum(row["mh_accepted"] for row in meta["records"])
    assert counts["sign_proposals"] == n * meta["stages"] * int(meta["sign_flip"])
    assert counts["sign_accepted"] == sum(row["sign_accepted"] for row in meta["records"])
    assert counts["sign_factor_calls"] == sum(row["sign_factor_calls"] for row in meta["records"])
    assert counts["particle_factor_calls"] == n * g + sum(row["factor_calls"] for row in meta["records"])
    if meta["temperature_strategy"] == "adaptive_ess":
        assert min(row["ess_fraction"] for row in meta["records"]) >= meta["ess_target"] - 1e-11
    np.testing.assert_allclose(sum(row["log_normalizer_increment"] for row in meta["records"]), log_z,
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(meta["statistics"]["weighted_mean"], w @ x, atol=1e-12, rtol=0)
    assert all(timing[key] >= 0 for key in ("preparation_seconds", "initialization_seconds",
                                           "annealing_seconds", "output_seconds"))
    assert timing["total_seconds"] == sum(timing[key] for key in (
        "preparation_seconds", "initialization_seconds", "annealing_seconds", "output_seconds"
    ))
    json.dumps([timing, counts, meta], allow_nan=False)


def test_sensor_target_matches_independent_joint_and_existing_posterior():
    problem = _heterogeneous()
    points = np.array([[0.0, 0.1], [-1.0, 0.7], [4.0, -3.0], [0.2, -0.3]])
    target = SensorTarget(problem)
    actual = target.log_prob(torch.from_numpy(points)).numpy()
    np.testing.assert_allclose(actual, _numpy_sensor_joint(problem, points), atol=2e-13, rtol=0)
    np.testing.assert_allclose(actual, problem.log_density(points) + problem.log_evidence(),
                               atol=2e-12, rtol=0)
    assert target.counts.particle_factor_calls == len(points) * problem.groups


@pytest.mark.parametrize("kind", ["prior", "gaussian", "mixture"])
def test_reference_tail_geometry_preparation_cost_and_no_truth_dependence(kind):
    problem = _heterogeneous()
    reference = build_reference(problem, reference=kind, components=3, mode_steps=12)
    problem.truth[:] = [1e30, -1e30]
    other = build_reference(problem, reference=kind, components=3, mode_steps=12)
    for field in ("weights", "means", "covariance", "tail_precision"):
        np.testing.assert_array_equal(getattr(reference, field), getattr(other, field))
    np.testing.assert_allclose(reference.weights.sum(), 1.0, atol=2e-15, rtol=0)
    expected_precision = np.eye(2)
    if kind != "prior":
        expected_precision += (problem.directions / problem.noise_std[:, None]).T @ (
            problem.directions / problem.noise_std[:, None]
        )
    np.testing.assert_allclose(reference.covariance @ expected_precision, np.eye(2), atol=3e-15, rtol=0)
    if kind == "prior":
        assert reference.evaluation_counts["mode_factor_calls"] == 0
        np.testing.assert_array_equal(reference.means, np.zeros((1, 2)))
    else:
        assert reference.evaluation_counts["mode_point_evaluations"] == (12 + 1) * (5 + 2 * problem.groups)
        assert reference.search["actual_components"] <= (1 if kind == "gaussian" else 3)


def test_exact_mixture_log_density_and_responsibilities():
    reference = GaussianReference(
        [0.31, 0.69], [[1.3, -0.4], [-1.0, 0.8]], [[0.6, 0.13], [0.13, 0.8]]
    )
    x = torch.tensor([[0.0, 0.2], [10.0, -12.0], [-20.0, 20.0]], dtype=torch.float64)
    logs = np.column_stack([math.log(float(w)) + multivariate_normal.logpdf(
        x.numpy(), mean.numpy(), reference.covariance.numpy()
    ) for w, mean in zip(reference.weights, reference.means)])
    np.testing.assert_allclose(reference.log_prob(x), logsumexp(logs, axis=1), atol=3e-13, rtol=0)
    np.testing.assert_allclose(reference.responsibilities(x),
                               np.exp(logs - logsumexp(logs, axis=1, keepdims=True)), atol=2e-14, rtol=0)


@pytest.mark.parametrize("scale", [0.2, 0.5, 1.0])
def test_q_reversibility_and_beta_mh_detailed_balance_density_identity(scale):
    target = SensorTarget(_heterogeneous())
    q = GaussianReference([0.3, 0.7], [[-1.0, 0.5], [0.8, -0.3]], [[0.8, 0.2], [0.2, 0.5]])
    rng = np.random.default_rng(711)
    x = torch.from_numpy(rng.normal(size=(31, 2)))
    y = torch.from_numpy(rng.normal(size=(31, 2)))
    qx, qy = q.log_prob(x), q.log_prob(y)
    kxy, kyx = q.transition_log_prob(x, y, scale), q.transition_log_prob(y, x, scale)
    torch.testing.assert_close(qx + kxy, qy + kyx, atol=2e-12, rtol=0)
    rx, ry = target.log_prob(x) - qx, target.log_prob(y) - qy
    for beta in (0.0, 0.37, 1.0):
        forward = qx + beta * rx + kxy + torch.minimum(beta * (ry - rx), torch.zeros_like(rx))
        reverse = qy + beta * ry + kyx + torch.minimum(beta * (rx - ry), torch.zeros_like(rx))
        torch.testing.assert_close(forward, reverse, atol=3e-12, rtol=0)


@pytest.mark.parametrize("beta", [0.0, 0.37, 1.0])
def test_actual_sign_flip_uses_complete_tempered_density(beta):
    target = SensorTarget(_heterogeneous())
    q = GaussianReference([0.3, 0.7], [[-1.0, 0.5], [0.8, -0.3]], [[0.8, 0.2], [0.2, 0.5]])
    points = torch.from_numpy(np.random.default_rng(21).normal(size=(512, 2)))
    residual = target.log_prob(points) - q.log_prob(points)
    _, auxiliary, _ = make_generators(61)
    actual, actual_residual, accepted, ratio = sign_flip_step(target, q, points, residual, beta, auxiliary)
    expected_ratio = ((1 - beta) * (q.log_prob(-points) - q.log_prob(points))
                      + beta * (target.log_prob(-points) - target.log_prob(points)))
    torch.testing.assert_close(ratio, expected_ratio, atol=1e-13, rtol=0)
    torch.testing.assert_close(actual, torch.where(accepted[:, None], -points, points), atol=0, rtol=0)
    torch.testing.assert_close(actual_residual, target.log_prob(actual) - q.log_prob(actual), atol=1e-13, rtol=0)
    assert bool(accepted.any()) and not bool(accepted.all())


@pytest.mark.parametrize("kind", ["prior", "gaussian", "mixture"])
def test_covariance_scale_preserves_tail_precision_and_prior(kind):
    problem = _heterogeneous()
    original = build_reference(problem, reference=kind, covariance_scale=1)
    wide = build_reference(problem, reference=kind, covariance_scale=4)
    np.testing.assert_array_equal(original.tail_precision, wide.tail_precision)
    np.testing.assert_array_equal(original.means, wide.means)
    np.testing.assert_array_equal(original.weights, wide.weights)
    np.testing.assert_allclose(wide.covariance, (1 if kind == "prior" else 4) * original.covariance,
                               atol=0, rtol=0)
    result = sample(problem, particles=2048, seed=41, reference=kind, covariance_scale=4)
    _assert_counts_and_metadata(result)
    assert result[5]["covariance_scale_applied"] == (1 if kind == "prior" else 4)


def test_incremental_weights_telescope_and_normalizer_constant_shift():
    problem = _heterogeneous()
    target = SensorTarget(problem)
    spec = build_reference(problem)
    q = GaussianReference(spec.weights, spec.means, spec.covariance)
    motion, auxiliary, _ = make_generators(92)
    points = q.sample(4096, motion, auxiliary)
    residual = target.log_prob(points) - q.log_prob(points)
    lw = torch.full((4096,), -math.log(4096), dtype=torch.float64)
    total = 0.0
    for increment in (0.1, 0.2, 0.3, 0.4):
        lw, log_ratio, _ = _reweight(lw, residual, increment)
        total += float(log_ratio)
    torch.testing.assert_close(lw, residual - torch.logsumexp(residual, 0), atol=5e-14, rtol=0)
    assert abs(total - float(torch.logsumexp(residual, 0)) + math.log(4096)) < 5e-14
    beta = _next_beta(residual, 0.0, 0.95)
    assert abs(beta - _next_beta(residual + 83.0, 0.0, 0.95)) < 1e-12
    shifted, shift_z, _ = _reweight(torch.full_like(lw, -math.log(4096)), residual + 83.0, 1.0)
    torch.testing.assert_close(shifted, lw, atol=5e-14, rtol=0)
    assert abs(float(shift_z) - total - 83.0) < 5e-14


def test_real_nonzero_gaussian_sample_exact_normalizer_and_uniform_weights():
    parameters = _gaussian_parameters()
    a, sigma, y = (parameters[key] for key in ("directions", "noise_std", "observations"))
    signs = 2 * parameters["positive_probability"] - 1
    covariance = np.linalg.inv(np.eye(2) + (a / sigma[:, None]).T @ (a / sigma[:, None]))
    mean = covariance @ (((signs * y) / sigma ** 2) @ a)
    observation_covariance = (signs[:, None] * a) @ (signs[:, None] * a).T + np.diag(sigma ** 2)
    expected_log_z = multivariate_normal.logpdf(y, cov=observation_covariance)
    result = sample(parameters, particles=8192, seed=71, reference="gaussian")
    x, w, log_z, _, _, meta = result
    _assert_counts_and_metadata(result)
    np.testing.assert_allclose(meta["reference_parameters"]["means"], mean[None], atol=2e-14, rtol=0)
    np.testing.assert_allclose(w, 1 / len(w), atol=1e-17, rtol=0)
    assert abs(log_z - expected_log_z) < 1e-12
    assert meta["stages"] == 1 and meta["records"][0]["acceptance"] == 1.0
    np.testing.assert_allclose(w @ x, mean, atol=0.03, rtol=0)
    empirical = (x - w @ x).T @ ((x - w @ x) * w[:, None])
    np.testing.assert_allclose(empirical, covariance, atol=0.025, rtol=0)


def test_zero_observation_gaussian_uses_real_model_and_merges_modes():
    problem = _heterogeneous()
    problem.observations[:] = 0.0
    result = sample(problem, particles=1024, seed=7, components=4)
    _assert_counts_and_metadata(result)
    assert result[5]["components_actual"] == 1
    assert result[5]["stages"] == 1
    np.testing.assert_allclose(result[1], 1 / 1024, atol=1e-16, rtol=0)
    assert abs(result[2] - problem.log_evidence()) < 1e-12


def test_actual_mh_mixture_invariant_distribution_empirical_diagnostic():
    problem = SensorProblem(np.array([[1.0]]), np.array([0.5]), np.array([0.65]),
                            np.array([2.5]), np.array([0.0]))
    probabilities, means, covariance = problem.exact_posterior()
    rng = np.random.default_rng(117)
    initial = problem.sample(rng, 32768)
    spec = build_reference(problem)
    q = GaussianReference(spec.weights, spec.means, spec.covariance)
    target = SensorTarget(problem)
    x = torch.from_numpy(initial)
    residual = target.log_prob(x) - q.log_prob(x)
    motion, auxiliary, _ = make_generators(917)
    for _ in range(5):
        x, residual, _ = metropolis_step(target, q, x, residual, 1.0, 0.7, motion, auxiliary)
    exact_mean, exact_covariance = _moments(probabilities, means, covariance)
    np.testing.assert_allclose(x.numpy().mean(0), exact_mean, atol=0.05, rtol=0)
    np.testing.assert_allclose(np.var(x.numpy(), axis=0), np.diag(exact_covariance), atol=0.06, rtol=0)
    expected_cdf = problem.projected_cdf(np.ones(1), np.array([-2.0, 0.0, 2.0]))
    observed_cdf = np.mean(x.numpy() <= np.array([-2.0, 0.0, 2.0])[None], axis=0)
    np.testing.assert_allclose(observed_cdf, expected_cdf, atol=0.012, rtol=0)


@pytest.mark.parametrize("covariance_scale", [1.0, 4.0])
def test_complete_bimodal_sensor_sampler_has_two_reference_components(covariance_scale):
    problem = SensorProblem(np.array([[1.0, 0.0], [0.6, 0.2], [0.4, 1.0]]),
                            np.array([0.5, 0.7, 0.8]), np.array([0.65, 0.8, 0.65]),
                            np.array([2.5, 1.4, 0.8]), np.array([0.0, 0.0]))
    result = sample(problem, particles=16384, seed=194, covariance_scale=covariance_scale)
    _assert_counts_and_metadata(result)
    x, w, log_z, timing, counts, meta = result
    probabilities, means, covariance = problem.exact_posterior()
    expected_mean, expected_covariance = _moments(probabilities, means, covariance)
    assert meta["components_actual"] == 2
    np.testing.assert_allclose(w @ x, expected_mean, atol=0.065, rtol=0)
    empirical_covariance = (x - w @ x).T @ ((x - w @ x) * w[:, None])
    np.testing.assert_allclose(empirical_covariance, expected_covariance, atol=0.09, rtol=0)
    assert abs(log_z - problem.log_evidence()) < 0.065
    observed_cdf = np.sum(w[:, None] * (x[:, 0, None] <= np.array([-2.0, 0.0, 2.0])), axis=0)
    expected_cdf = problem.projected_cdf(np.array([1.0, 0.0]), np.array([-2.0, 0.0, 2.0]))
    np.testing.assert_allclose(observed_cdf, expected_cdf, atol=0.018, rtol=0)
    print(json.dumps({"bimodal_covariance_scale": covariance_scale, "components": meta["components_actual"],
                      "stages": meta["stages"], "mean_error": float(np.linalg.norm(w @ x - expected_mean)),
                      "covariance_error": float(np.linalg.norm(empirical_covariance - expected_covariance)),
                      "cdf_max_error": float(np.max(np.abs(observed_cdf - expected_cdf))),
                      "logZ_error": log_z - problem.log_evidence(), "counts": counts,
                      "timing": timing}, allow_nan=False))


@pytest.mark.parametrize("kind", ["prior", "gaussian", "mixture"])
def test_eight_dimensional_real_sensor_smoke_and_no_truth_leak(kind):
    problem = make_problem(433, groups=12, dimension=8, regime="regular")
    result = sample(problem, particles=1024, seed=19, reference=kind, covariance_scale=4.0)
    _assert_counts_and_metadata(result)
    values = problem.to_dict()
    values["truth"] = [float("nan")] * 8
    values["reference"] = "unused"
    repeated = sample(values, particles=1024, seed=19, reference=kind, covariance_scale=4.0)
    for left, right in zip(result[:3], repeated[:3]):
        np.testing.assert_array_equal(left, right)
    assert result[4:] == repeated[4:]


def test_weight_moment_tail_quadratics_positive_for_all_supported_reference_scales():
    problem = _heterogeneous()
    tail = build_reference(problem).tail_precision
    for scale in (1.0, 4.0):
        q_precision = tail / scale
        for order in (1.0, 2.0, 4.0, 32.0):
            for beta in (0.0, 0.4, 0.9):
                delta = 1.0 - beta
                moment_precision = q_precision + (beta + order * delta) * (tail - q_precision)
                assert np.linalg.eigvalsh(moment_precision).min() > 0


@pytest.mark.parametrize("kind", ["prior", "gaussian", "mixture"])
def test_real_heterogeneous_cpu_runs_and_replicate_statistics(kind):
    problem = _heterogeneous()
    probabilities, means, covariance = problem.exact_posterior()
    expected_mean, expected_covariance = _moments(probabilities, means, covariance)
    exact_log_z = problem.log_evidence()
    exact_samples = problem.sample(np.random.default_rng(350), 32768)
    rows = []
    for seed in (360, 361, 362, 363):
        result = sample(problem, particles=4096, seed=seed, reference=kind, ess_target=0.9)
        _assert_counts_and_metadata(result)
        x, w, log_z, timing, counts, meta = result
        mean = w @ x
        cov = (x - mean).T @ ((x - mean) * w[:, None])
        w1 = np.mean([wasserstein_distance(x[:, index], exact_samples[:, index], u_weights=w)
                      for index in range(2)])
        rows.append({"seed": seed, "mean": mean.tolist(), "covariance": cov.tolist(),
                     "w1": float(w1), "logZ_error": log_z - exact_log_z,
                     "seconds": timing["total_seconds"], "factor_calls": counts["factor_calls"],
                     "stages": meta["stages"], "ess": meta["statistics"]["ess"]})
    errors = np.asarray([row["logZ_error"] for row in rows])
    evidence_ratio = np.exp(errors)
    diagnostic = {"reference": kind, "replicates": rows,
                  "evidence_ratio_mean": float(evidence_ratio.mean()),
                  "evidence_ratio_standard_error": float(evidence_ratio.std(ddof=1) / 2),
                  "mean_standard_errors": (np.std([r["mean"] for r in rows], axis=0, ddof=1) / 2).tolist(),
                  "w1_mean": float(np.mean([row["w1"] for row in rows]))}
    print(json.dumps(diagnostic, allow_nan=False))
    np.testing.assert_allclose(np.mean([row["mean"] for row in rows], axis=0), expected_mean, atol=0.04, rtol=0)
    np.testing.assert_allclose(np.mean([row["covariance"] for row in rows], axis=0), expected_covariance,
                               atol=0.045, rtol=0)
    assert abs(evidence_ratio.mean() - 1.0) < 0.04
    assert np.max(np.abs(errors)) < 0.1


def test_two_dimensional_bridge_integral_weight_identity():
    problem = _heterogeneous()
    spec = build_reference(problem)
    reference = GaussianReference(spec.weights, spec.means, spec.covariance)
    target = SensorTarget(problem)
    axis = np.linspace(-6.0, 6.0, 193)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    points = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel())))
    log_q = reference.log_prob(points)
    log_joint = target.log_prob(points)
    residual = log_joint - log_q
    for left, right in ((0.0, 0.4), (0.4, 1.0)):
        source = torch.exp(log_q + left * residual)
        weighted = source * torch.exp((right - left) * residual)
        destination = torch.exp(log_q + right * residual)
        torch.testing.assert_close(weighted, destination, atol=1e-15, rtol=1e-13)
    integrated = simpson(simpson(torch.exp(log_joint).numpy().reshape(xx.shape), x=axis), x=axis)
    assert abs(math.log(integrated) - problem.log_evidence()) < 1e-10


def test_reproducible_returns_final_weights_rng_isolation_and_fixed_schedule():
    problem = make_problem(431, groups=4, dimension=2)
    options = dict(particles=1024, seed=105, reference="prior", temperatures=[0.0, 0.2, 0.5, 1.0])
    first = sample(problem, **options)
    second = sample(problem.to_dict(), **options)
    _assert_counts_and_metadata(first)
    for left, right in zip(first[:3], second[:3]):
        np.testing.assert_array_equal(left, right)
    assert first[4:] == second[4:]
    tensor = sample(problem, **options, return_numpy=False)
    torch.testing.assert_close(tensor[0], torch.from_numpy(first[0]), atol=0, rtol=0)
    torch.testing.assert_close(tensor[1], torch.from_numpy(first[1]), atol=0, rtol=0)
    assert first[5]["records"][-1]["resampled"] is False
    a, b, c = make_generators(15)
    fresh, _, _ = make_generators(15)
    torch.rand((1000,), generator=b)
    torch.rand((1000,), generator=c)
    torch.testing.assert_close(torch.randn((20,), generator=a), torch.randn((20,), generator=fresh), atol=0, rtol=0)
    indices = _resample(torch.tensor([0.0, 0.4, 0.6, 0.0], dtype=torch.float64), c)
    assert ((indices > 0) & (indices < 3)).all()


@pytest.mark.parametrize("options", [
    {"particles": 1}, {"particles": True}, {"seed": -1}, {"moves": -1},
    {"reference": "unknown"}, {"components": 5}, {"mode_steps": 0},
    {"ess_target": 1.0}, {"ess_target": np.nan}, {"proposal_scale": 0},
    {"max_stages": 0}, {"temperatures": [0.0, 0.5]}, {"temperatures": [0, 0.5, 0.5, 1]},
    {"covariance_scale": 0.5}, {"covariance_scale": float("inf")}, {"covariance_scale": float("nan")},
    {"sign_flip": 1},
    {"covariance_scale": True},
])
def test_invalid_configuration_fails_fast(options):
    with pytest.raises((ValueError, FloatingPointError)):
        sample(_heterogeneous(), **options)


def test_bad_inputs_and_unfinished_annealing_raise():
    problem = _heterogeneous().to_dict()
    problem["noise_std"][0] = 0.0
    with pytest.raises(ValueError):
        sample(problem)
    problem["noise_std"][0] = float("nan")
    with pytest.raises(FloatingPointError):
        sample(problem)
    with pytest.raises(RuntimeError, match="max_stages"):
        sample(_heterogeneous(), particles=256, seed=11, reference="prior", ess_target=0.999, max_stages=1)


@pytest.mark.skipif(not os.environ.get("ICLR_TEST_DEVICE", "cpu").startswith("cuda"),
                    reason="explicit ICLR_TEST_DEVICE=cuda is required")
def test_requested_cuda_gaussian_and_device_cache():
    device = os.environ["ICLR_TEST_DEVICE"]
    result = sample(_gaussian_parameters(), particles=2048, seed=18, device=device, return_numpy=False,
                    reference="gaussian")
    assert result[0].is_cuda and result[0].dtype == torch.float64
    assert result[1].device == result[0].device
    assert result[5]["stages"] == 1
    torch.testing.assert_close(result[1], torch.full_like(result[1], 1 / 2048), atol=1e-16, rtol=0)


@pytest.mark.skipif(not os.environ.get("ICLR_TEST_DEVICE", "cpu").startswith("cuda"),
                    reason="explicit ICLR_TEST_DEVICE=cuda is required")
def test_requested_cuda_mixture_density_parity_and_sampling():
    device = os.environ["ICLR_TEST_DEVICE"]
    problem = _heterogeneous()
    spec = build_reference(problem, covariance_scale=4)
    cpu = GaussianReference(spec.weights, spec.means, spec.covariance)
    gpu = GaussianReference(spec.weights, spec.means, spec.covariance, device=device)
    points = torch.from_numpy(np.random.default_rng(7).normal(size=(512, 2)))
    torch.testing.assert_close(cpu.log_prob(points), gpu.log_prob(points.to(device)).cpu(), atol=1e-12, rtol=0)
    torch.testing.assert_close(SensorTarget(problem).log_prob(points),
                               SensorTarget(problem, device).log_prob(points.to(device)).cpu(), atol=1e-12, rtol=0)
    result = sample(problem, particles=1024, seed=11, device=device, covariance_scale=4)
    _assert_counts_and_metadata(result)
