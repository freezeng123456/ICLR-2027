from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _symmetric(value):
    return (value + value.T) / 2


def _inverse_logdet(value):
    if not np.isfinite(value).all() or not np.allclose(value, value.T, atol=1e-12, rtol=1e-12):
        raise ValueError("a finite symmetric matrix is required")
    factor = cho_factor(value, lower=True)
    return cho_solve(factor, np.eye(len(value))), float(2 * np.log(np.diag(factor[0])).sum())


@dataclass
class GaussianTwistedPath:
    initial_mean: np.ndarray
    initial_covariance: np.ndarray
    matrices: np.ndarray
    shifts: np.ndarray
    covariances: np.ndarray
    log_normalizer: float
    backward_quadratic: np.ndarray
    backward_linear: np.ndarray
    backward_constant: np.ndarray

    def marginal_moments(self):
        mean, covariance = self.initial_mean.copy(), self.initial_covariance.copy()
        means, covariances = [mean], [covariance]
        for matrix, shift, noise in zip(self.matrices, self.shifts, self.covariances):
            mean = matrix @ mean + shift
            covariance = _symmetric(matrix @ covariance @ matrix.T + noise)
            means.append(mean)
            covariances.append(covariance)
        return np.stack(means), np.stack(covariances)


def build_gaussian_twist(initial_mean, initial_covariance, matrices, shifts,
                         noise_covariances, quadratic, linear, constant):
    m, V, L, d, Q, C, ell, c = [np.asarray(v, dtype=np.float64) for v in
        [initial_mean, initial_covariance, matrices, shifts, noise_covariances, quadratic, linear, constant]]
    if m.ndim != 1 or L.ndim != 3:
        raise ValueError("invalid Gaussian path dimensions")
    K, dim = L.shape[0], len(m)
    shapes = [(V, (dim, dim)), (L, (K, dim, dim)), (d, (K, dim)), (Q, (K, dim, dim)),
              (C, (K, dim, dim)), (ell, (K, dim)), (c, (K,))]
    if K < 1 or dim < 1 or any(a.shape != shape for a, shape in shapes):
        raise ValueError("inconsistent Gaussian path dimensions")
    if not all(np.isfinite(a).all() for a in [m, V, L, d, Q, C, ell, c]):
        raise ValueError("all Gaussian path inputs must be finite")
    if not np.allclose(C, C.transpose(0, 2, 1), atol=1e-12, rtol=1e-12):
        raise ValueError("quadratic potentials must be symmetric")
    Vinv, logdetV = _inverse_logdet(V)
    J, j, a = np.zeros((K + 1, dim, dim)), np.zeros((K + 1, dim)), np.zeros(K + 1)
    M, shift, Sbank = np.empty_like(L), np.empty_like(d), np.empty_like(Q)
    for k in reversed(range(K)):
        Qinv, logdetQ = _inverse_logdet(Q[k])
        S, logdetH = _inverse_logdet(_symmetric(Qinv - J[k + 1]))
        # 等价形式避免两个随步长倒数增大的矩阵相减。
        T = _symmetric(J[k + 1] + J[k + 1] @ S @ J[k + 1])
        nu = j[k + 1] + J[k + 1] @ S @ j[k + 1]
        J[k] = _symmetric(2 * C[k] + L[k].T @ T @ L[k])
        j[k] = ell[k] + L[k].T @ (T @ d[k] + nu)
        a[k] = (c[k] + a[k + 1] - .5 * (logdetQ + logdetH)
                + .5 * d[k] @ T @ d[k] + nu @ d[k] + .5 * j[k + 1] @ S @ j[k + 1])
        correction = np.eye(dim) + S @ J[k + 1]
        M[k], shift[k], Sbank[k] = correction @ L[k], correction @ d[k] + S @ j[k + 1], S
    covariance, logdetP = _inverse_logdet(_symmetric(Vinv - J[0]))
    b = Vinv @ m + j[0]
    mean = covariance @ b
    logz = a[0] - .5 * (logdetV + logdetP) + .5 * b @ mean - .5 * m @ Vinv @ m
    return GaussianTwistedPath(mean, covariance, M, shift, Sbank, float(logz), J, j, a)


def affine_factor_backbone(A, b, grid):
    A, b, grid = [np.asarray(v, dtype=np.float64) for v in [A, b, grid]]
    if A.ndim != 4 or b.shape != A.shape[:3] or len(grid) != len(A) + 1:
        raise ValueError("invalid affine factor bank dimensions")
    if A.shape[2] != A.shape[3] or not np.isfinite(grid).all() or not (np.diff(grid) < 0).all():
        raise ValueError("a finite decreasing grid and square matrices are required")
    if not np.allclose(A, A.transpose(0, 1, 3, 2), atol=1e-12, rtol=1e-12):
        raise ValueError("factor tail matrices must be symmetric")
    h = -np.diff(grid)
    total_A, total_b = A.sum(1), b.sum(1)
    C = .5 * (np.einsum("kji,kjl->kil", total_A, total_A) - np.einsum("kgji,kgjl->kil", A, A))
    ell = -np.einsum("kji,kj->ki", total_A, total_b) + np.einsum("kgji,kgj->ki", A, b)
    constant = .5 * (np.square(total_b).sum(1) - np.square(b).sum((1, 2)))
    identity = np.eye(A.shape[-1])
    return dict(matrices=identity - h[:, None, None] * (total_A + .5 * identity),
                shifts=h[:, None] * total_b, noise_covariances=h[:, None, None] * identity,
                quadratic=h[:, None, None] * C, linear=h[:, None] * ell, constant=h * constant)
