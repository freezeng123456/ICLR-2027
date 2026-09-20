import numpy as np
import pytest
from scipy.stats import multivariate_normal

from gaussian_twisted_path_20260921 import build_gaussian_twist, affine_factor_backbone


def dense_path(m, V, L, d, Q, C, ell, c):
    K, dim = len(L), len(m)
    P, b = np.zeros(((K + 1) * dim,) * 2), np.zeros((K + 1) * dim)
    Vinv = np.linalg.inv(V)
    P[:dim, :dim], b[:dim] = Vinv, Vinv @ m
    constant = -.5 * (m @ Vinv @ m + np.linalg.slogdet(V)[1])
    for k in range(K):
        x, y = slice(k * dim, (k + 1) * dim), slice((k + 1) * dim, (k + 2) * dim)
        inverse = np.linalg.inv(Q[k])
        P[x, x] += L[k].T @ inverse @ L[k] - 2 * C[k]
        P[y, y] += inverse
        P[x, y] -= L[k].T @ inverse
        P[y, x] -= inverse @ L[k]
        b[x] += ell[k] - L[k].T @ inverse @ d[k]
        b[y] += inverse @ d[k]
        constant += c[k] - .5 * (d[k] @ inverse @ d[k] + np.linalg.slogdet(Q[k])[1])
    covariance = np.linalg.inv(P)
    assert np.linalg.eigvalsh(P).min() > 0
    return covariance @ b, covariance, constant - .5 * np.linalg.slogdet(P)[1] + .5 * b @ covariance @ b


@pytest.mark.parametrize("singular", [False, True])
def test_backward_recursion_matches_independent_joint_gaussian(singular):
    rng = np.random.default_rng(41)
    K, dim = 4, 3
    m, V = rng.normal(size=dim), np.diag([1., 1.4, .8])
    L, d = rng.normal(size=(K, dim, dim)) * .2, rng.normal(size=(K, dim)) * .1
    if singular:
        L[1] = 0
    q = rng.normal(size=(K, dim, dim))
    Q = q @ q.transpose(0, 2, 1) * .1 + .3 * np.eye(dim)
    C = np.broadcast_to(np.diag([.015, -.02, .01]), (K, dim, dim))
    ell, c = rng.normal(size=(K, dim)) * .1, rng.normal(size=K) * .1
    result = build_gaussian_twist(m, V, L, d, Q, C, ell, c)
    exact_m, exact_V, exact_logz = dense_path(m, V, L, d, Q, C, ell, c)
    means, variances = result.marginal_moments()
    np.testing.assert_allclose(means.ravel(), exact_m, atol=1e-12)
    for k in range(K + 1):
        np.testing.assert_allclose(variances[k], exact_V[k * dim:(k + 1) * dim, k * dim:(k + 1) * dim], atol=1e-12)
    np.testing.assert_allclose(result.log_normalizer, exact_logz, atol=1e-12)
    path = rng.normal(size=(K + 1, dim))
    actual_log = multivariate_normal.logpdf(path[0], result.initial_mean, result.initial_covariance)
    target_log = multivariate_normal.logpdf(path[0], m, V)
    for k in range(K):
        actual_log += multivariate_normal.logpdf(path[k + 1], result.matrices[k] @ path[k] + result.shifts[k], result.covariances[k])
        target_log += (path[k] @ C[k] @ path[k] + ell[k] @ path[k] + c[k]
                       + multivariate_normal.logpdf(path[k + 1], L[k] @ path[k] + d[k], Q[k]))
    np.testing.assert_allclose(target_log - actual_log, exact_logz, atol=1e-12)


def test_gaussian_twist_exists_when_original_second_moment_diverges():
    args = dict(initial_mean=np.zeros(1), initial_covariance=np.eye(1), matrices=np.ones((1, 1, 1)),
                shifts=np.zeros((1, 1)), noise_covariances=np.ones((1, 1, 1)),
                quadratic=np.array([[[.3]]]), linear=np.zeros((1, 1)), constant=np.zeros(1))
    result = build_gaussian_twist(**args)
    assert result.initial_covariance[0, 0] == pytest.approx(2.5)
    assert result.log_normalizer == pytest.approx(-.5 * np.log(.4))
    with pytest.raises(np.linalg.LinAlgError):
        build_gaussian_twist(**{**args, "quadratic": 2 * args["quadratic"]})


def test_affine_backbone_matches_direct_pair_potential():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(3, 4, 2, 2)) * .1
    A = (A + A.transpose(0, 1, 3, 2)) / 2
    b, grid = rng.normal(size=(3, 4, 2)), np.array([2., 1.8, 1.7, 1.4])
    result = affine_factor_backbone(A, b, grid)
    for k, h in enumerate(-np.diff(grid)):
        x = rng.normal(size=2)
        r = -np.einsum("gij,j->gi", A[k], x) + b[k]
        expected = h * .5 * (np.square(r.sum(0)).sum() - np.square(r).sum())
        observed = x @ result["quadratic"][k] @ x + result["linear"][k] @ x + result["constant"][k]
        np.testing.assert_allclose(observed, expected, atol=1e-13)
        np.testing.assert_allclose(result["matrices"][k] @ x + result["shifts"][k], (1 - h / 2) * x + h * r.sum(0), atol=1e-13)


def test_nonfinite_and_nonsymmetric_inputs_rejected():
    args = dict(initial_mean=np.zeros(2), initial_covariance=np.eye(2), matrices=np.zeros((1, 2, 2)),
                shifts=np.zeros((1, 2)), noise_covariances=np.eye(2)[None], quadratic=np.zeros((1, 2, 2)),
                linear=np.zeros((1, 2)), constant=np.zeros(1))
    with pytest.raises(ValueError):
        build_gaussian_twist(**{**args, "constant": np.array([np.nan])})
    with pytest.raises(ValueError):
        build_gaussian_twist(**{**args, "initial_covariance": np.array([[1., .1], [0., 1.]])})
