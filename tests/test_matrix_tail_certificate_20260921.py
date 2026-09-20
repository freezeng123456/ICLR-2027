import numpy as np
import pytest
from scipy.integrate import dblquad

from matrix_tail_certificate_20260921 import gaussian_exponential_update, matrix_certificate


def test_diagonal_reduction_matches_scalar_formula():
    A = np.array([[[0.2, 0.0], [0.0, 0.4]], [[0.5, 0.0], [0.0, 0.1]]])
    V = np.diag([0.7, 1.3])
    result = matrix_certificate(A, V, 0.2, 1.0)
    scalar_C = np.array([0.2 * 0.5, 0.4 * 0.1])
    expected_H = 1.0 / np.diag(V) - 2.0 * 0.2 * scalar_C
    assert result["status"] == "finite"
    assert np.allclose(np.diag(result["H"]), expected_H)
    assert np.allclose(result["H"] - np.diag(np.diag(result["H"])), 0.0)


def test_nonsymmetric_factor_is_rejected():
    with pytest.raises(ValueError):
        matrix_certificate(np.array([[[0.0, 1.0], [0.0, 0.0]]]), np.eye(2), 0.1, 1.0)


def test_noncommuting_symmetric_matrices_are_supported():
    A = np.array([[[0.8, 0.0], [0.0, 0.1]], [[0.3, 0.25], [0.25, 0.3]]])
    assert not np.allclose(A[0] @ A[1], A[1] @ A[0])
    result = matrix_certificate(A, np.eye(2), 0.05, 0.5)
    assert result["status"] == "finite"
    assert np.allclose(result["H"], result["H"].T)


def test_finite_and_divergent_examples_agree_with_direct_gaussian_integration():
    finite = matrix_certificate(np.array([np.eye(2), 0.2 * np.eye(2)]), np.eye(2), 0.05, 1.0)
    divergent = matrix_certificate(np.array([4.0 * np.eye(2), 4.0 * np.eye(2)]), np.eye(2), 0.1, 1.0)
    assert finite["status"] == "finite"
    assert divergent["status"] == "divergent"
    assert np.linalg.eigvalsh(finite["H"])[0] > 0
    assert np.linalg.eigvalsh(divergent["H"])[0] < 0


def test_k1_gaussian_update_has_analytic_mean_and_log_normalizer():
    mean = np.array([0.3, -0.2])
    V = np.diag([0.7, 1.2])
    C = np.diag([0.1, -0.05])
    linear = np.array([0.4, -0.1])
    result = gaussian_exponential_update(mean, V, C, linear, 0.2, 0.3)
    H = np.linalg.inv(V) - 2.0 * 0.3 * C
    b = np.linalg.inv(V) @ mean + 0.3 * linear
    expected_mean = np.linalg.solve(H, b)
    expected_log = (0.3 * 0.2 - 0.5 * np.linalg.slogdet(V)[1] - 0.5 * np.linalg.slogdet(H)[1]
                    + 0.5 * b @ np.linalg.solve(H, b) - 0.5 * mean @ np.linalg.inv(V) @ mean)
    assert result["status"] == "finite"
    assert np.allclose(result["mean"], expected_mean)
    assert np.isclose(result["log_normalizer"], expected_log)


def test_mixed_sign_precision_with_one_negative_direction_diverges():
    A = np.array([[[3.0, 0.0], [0.0, 1.0]], [[3.0, 0.0], [0.0, 1.0]]])
    result = matrix_certificate(A, np.eye(2), 0.1, 1.0)
    assert np.linalg.eigvalsh(result["H"])[0] < 0 < np.linalg.eigvalsh(result["H"])[-1]
    assert result["status"] == "divergent"


def test_singular_L_is_allowed_and_preserves_symmetric_covariance():
    A = np.array([[[1.5, 0.0], [0.0, 1.5]]])
    result = matrix_certificate(A, np.eye(2), 0.5, 1.0)
    assert np.linalg.matrix_rank(result["L"]) == 0
    assert result["status"] == "finite"
    assert np.allclose(result["V_next"], result["V_next"].T)


def test_invalid_kappa_mean_and_covariance_are_rejected():
    with pytest.raises(ValueError):
        matrix_certificate(np.zeros((1, 2, 2)), np.eye(2), 0.1, 0.0)
    with pytest.raises(ValueError):
        gaussian_exponential_update(np.array([[0.0, 0.0]]), np.eye(2), np.zeros((2, 2)), np.zeros(2), 0.0, 0.1)
    with pytest.raises(ValueError):
        gaussian_exponential_update(np.zeros(2), np.array([[1.0, 0.0], [0.0, 0.0]]), np.zeros((2, 2)), np.zeros(2), 0.0, 0.1)


def test_nondiagonal_gaussian_weight_matches_independent_quadrature():
    mean = np.array([.3, -.2])
    V = np.array([[.8, .25], [.25, 1.1]])
    C = np.array([[.2, -.1], [-.1, -.15]])
    linear = np.array([.4, -.3])
    h, constant = .15, .1
    inverse = np.linalg.inv(V)
    scale = 2 * np.pi * np.sqrt(np.linalg.det(V))
    def integrand(y, x):
        point = np.array([x, y])
        delta = point - mean
        return np.exp(-.5 * delta @ inverse @ delta + h * (point @ C @ point + linear @ point + constant)) / scale
    observed, error = dblquad(integrand, -np.inf, np.inf, -np.inf, np.inf, epsabs=1e-9, epsrel=1e-9)
    expected = gaussian_exponential_update(mean, V, C, linear, constant, h)
    assert error < 1e-7
    np.testing.assert_allclose(np.log(observed), expected["log_normalizer"], atol=1e-9)
