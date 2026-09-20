import numpy as np


def _symmetric(name, value, dimension):
    array = np.asarray(value, dtype=float)
    if array.shape != (dimension, dimension):
        raise ValueError(f"{name} shape must be {(dimension, dimension)}")
    if not np.isfinite(array).all() or not np.allclose(array, array.T, atol=1e-12, rtol=0):
        raise ValueError(f"{name} must be finite and symmetric")
    return array


def matrix_certificate(A, V, h, kappa):
    A = np.asarray(A, dtype=float)
    if A.ndim != 3 or A.shape[1] != A.shape[2] or not np.isfinite(A).all():
        raise ValueError("A must be a finite (G,d,d) array")
    G, dimension, _ = A.shape
    if G < 1 or dimension < 1:
        raise ValueError("A must have at least one factor and coordinate")
    if not all(np.allclose(item, item.T, atol=1e-12, rtol=0) for item in A):
        raise ValueError("all A_g must be symmetric")
    V = _symmetric("V", V, dimension)
    if not np.isfinite(h) or not np.isfinite(kappa) or h <= 0 or kappa <= 0:
        raise ValueError("h and kappa must be finite, with h and kappa positive")
    V_eigenvalues = np.linalg.eigvalsh(V)
    if V_eigenvalues[0] <= 0:
        raise ValueError("V must be positive definite")
    total_A = A.sum(axis=0)
    C = 0.5 * (total_A.T @ total_A - np.einsum("gji,gjk->ik", A, A))
    L = np.eye(dimension) - h * ((1.0 + kappa) * total_A + kappa * np.eye(dimension)) / 2.0
    precision = np.linalg.inv(V) - 2.0 * h * C
    eigenvalues = np.linalg.eigvalsh(precision)
    tolerance = 1e-11 * max(1.0, np.linalg.norm(precision, ord=2))
    if eigenvalues[0] > tolerance:
        status = "finite"
        V_next = L @ np.linalg.inv(precision) @ L.T + kappa * h * np.eye(dimension)
    elif eigenvalues[0] < -tolerance:
        status = "divergent"
        V_next = None
    else:
        status = "boundary_unclassified"
        V_next = None
    return {"status": status, "C": C, "L": L, "H": precision, "H_eigenvalues": eigenvalues, "V_next": V_next}


def gaussian_exponential_update(mean, V, C, linear, constant, h):
    mean = np.asarray(mean, dtype=float)
    if mean.ndim != 1 or not np.isfinite(mean).all():
        raise ValueError("mean must be a finite one-dimensional array")
    V = _symmetric("V", V, mean.size)
    if np.linalg.eigvalsh(V)[0] <= 0:
        raise ValueError("V must be positive definite")
    C = _symmetric("C", C, mean.size)
    linear = np.asarray(linear, dtype=float)
    if linear.shape != mean.shape or not np.isfinite(linear).all() or not np.isfinite(constant) or not np.isfinite(h) or h <= 0:
        raise ValueError("invalid Gaussian update inputs")
    V_inverse = np.linalg.inv(V)
    H = V_inverse - 2.0 * h * C
    eigenvalues = np.linalg.eigvalsh(H)
    if eigenvalues[0] <= 0:
        return {"status": "divergent", "H": H, "mean": None, "log_normalizer": np.inf}
    H_inverse = np.linalg.inv(H)
    b = V_inverse @ mean + h * linear
    updated_mean = H_inverse @ b
    log_normalizer = (h * constant - 0.5 * np.linalg.slogdet(V)[1] - 0.5 * np.linalg.slogdet(H)[1]
                      + 0.5 * b @ H_inverse @ b - 0.5 * mean @ V_inverse @ mean)
    return {"status": "finite", "H": 0.5 * (H + H.T), "mean": updated_mean, "log_normalizer": float(log_normalizer)}
