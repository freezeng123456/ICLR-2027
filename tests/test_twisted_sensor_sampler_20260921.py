import os
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from nonseparable_sensor_model_20260921 import make_problem
from nonseparable_sensor_sampler_20260921 import SensorScoreBank
from twisted_sensor_sampler_20260921 import twisted_sensor_sample, sensor_references


@pytest.mark.parametrize("method,reference", [("full", "mean"), ("full", "mirror"), ("tail_anchored", "mirror")])
def test_complete_nonlinear_path_density_identity(method, reference):
    problem = make_problem(73, groups=4, dimension=2, regime="ambiguous")
    grid = np.linspace(np.sqrt(20.), 0., 65) ** 2
    output = twisted_sensor_sample(problem, grid, 9, 512, method=method, batch=2, reference=reference, return_paths=True,
                                   device=os.environ.get("ICLR_TEST_DEVICE", "cpu"))
    bank = SensorScoreBank(problem, grid, prepare_method="all")
    references, _, _ = sensor_references(bank, reference)
    for i, path in enumerate(output["paths"]):
        log_target = multivariate_normal.logpdf(path[0], np.zeros(2), np.eye(2))
        log_proposals = [multivariate_normal.logpdf(path[0], ref.initial_mean, ref.initial_covariance) for ref in references]
        for k, h in enumerate(-np.diff(grid)):
            x, y = path[k:k + 2]
            precision, means, probabilities = problem.factor_parameters(float(grid[k]))
            scores = []
            for g in range(problem.groups):
                logp = np.array([np.log(probabilities[g, c]) + multivariate_normal.logpdf(x, means[g, c], np.linalg.inv(precision[g])) for c in range(2)])
                responsibility = np.exp(logp - logsumexp(logp))
                scores.append(-precision[g] @ (x - responsibility @ means[g]) + x)
            r = np.array(scores)
            if method == "full":
                drift = r.sum(0)
                potential = sum(r[g] @ r[j] for g in range(4) for j in range(g + 1, 4))
            else:
                selected = output["batches"][k][i]
                r0 = -np.einsum("gij,j->gi", bank.numpy_banks["A"][k], x) + bank.numpy_banks["b_anchor"][k]
                e, R0 = r - r0, r0.sum(0)
                drift = R0 + 2 * e[selected].sum(0)
                potential = sum(r0[g] @ r0[j] for g in range(4) for j in range(g + 1, 4))
                potential += 2 * sum((R0 - r0[g]) @ e[g] for g in selected)
                potential += 6 * e[selected[0]] @ e[selected[1]]
            log_target += h * potential + multivariate_normal.logpdf(y, (1 - h / 2) * x + h * drift, h * np.eye(2))
            for j, ref in enumerate(references):
                log_proposals[j] += multivariate_normal.logpdf(y, ref.matrices[k] @ x + ref.shifts[k], ref.covariances[k])
        expected = log_target - logsumexp(log_proposals) + np.log(len(references))
        np.testing.assert_allclose(output["log_weights"][i], expected, atol=2e-12)
    np.testing.assert_allclose(output["weights"].sum(), 1., atol=1e-14)


def test_sampler_repeatable_and_nonnegative_weights():
    problem = make_problem(19, groups=4, dimension=2)
    a = twisted_sensor_sample(problem, [.2, .15, .1], 17, 8, batch=2)
    b = twisted_sensor_sample(problem, [.2, .15, .1], 17, 8, batch=2)
    np.testing.assert_array_equal(a["samples"], b["samples"])
    np.testing.assert_array_equal(a["weights"], b["weights"])
    assert np.all(a["weights"] >= 0) and 0 < a["ess_fraction"] <= 1


def test_nonintegrable_reference_is_rejected():
    problem = make_problem(73, groups=4, dimension=2, regime="ambiguous")
    with pytest.raises(np.linalg.LinAlgError):
        twisted_sensor_sample(problem, np.linspace(.3, 0., 5), 9, 512, batch=2)


def test_real_development_sensor_proposal_is_finite():
    problem = make_problem(1300, groups=12, dimension=8, regime="regular")
    grid = np.linspace(np.sqrt(20.), 0., 1025) ** 2
    output = twisted_sensor_sample(problem, grid, 64, 23, batch=8,
                                  device=os.environ.get("ICLR_TEST_DEVICE", "cpu"))
    assert np.isfinite(output["samples"]).all() and np.isfinite(output["log_weights"]).all()
    np.testing.assert_allclose(output["weights"].sum(), 1, atol=1e-12)
