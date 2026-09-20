import numpy as np
from scipy.integrate import quad
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp, ndtr


def _array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _probabilities(value, name, *, positive=False):
    result = _array(value, name)
    if result.ndim != 1 or not len(result):
        raise ValueError(f"{name} must be a nonempty vector")
    if np.any(result < 0) or (positive and np.any(result <= 0)):
        raise ValueError(f"{name} has invalid weights")
    total = result.sum()
    if not np.isfinite(total) or total <= 0 or not np.isclose(total, 1, atol=1e-12, rtol=0):
        raise ValueError(f"{name} must sum to one")
    return result / total


def exact_mixture(p):
    """枚举 prior-corrected learned product；最多支持 16 个二分量因子。"""
    a, v, m, w = [_array(p[name], name) for name in ("directions", "variance", "means", "weights")]
    if a.ndim != 2 or not 1 <= len(a) <= 16 or a.shape[1] < 1:
        raise ValueError("directions must have shape (G,d), 1 <= G <= 16")
    groups, dimension = a.shape
    if v.shape != (groups,) or m.shape != (groups, 2) or w.shape != m.shape:
        raise ValueError("invalid factor parameter shapes")
    if not np.allclose(np.linalg.norm(a, axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError("factor directions must be unit vectors")
    if np.any((v <= 0) | (v >= 1)) or np.any(w <= 0):
        raise ValueError("factor variances must lie in (0,1) and weights must be positive")
    if not np.allclose(w.sum(axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError("factor weights must sum to one")
    precision = np.eye(dimension) + a.T @ ((1 / v - 1)[:, None] * a)
    precision = (precision + precision.T) / 2
    factor = cho_factor(precision, lower=True, check_finite=True)
    covariance = cho_solve(factor, np.eye(dimension))
    covariance = (covariance + covariance.T) / 2
    indices = (np.arange(2 ** groups, dtype=np.uint64)[:, None]
               >> np.arange(groups, dtype=np.uint64)) & 1
    chosen = m[np.arange(groups), indices]
    linear = (chosen / v) @ a
    means = linear @ covariance
    constant = (np.log(w[np.arange(groups), indices]) - .5 * np.log(v)
                - .5 * chosen ** 2 / v).sum(axis=1)
    log_mass = constant + .5 * np.einsum("ni,ni->n", linear, means)
    log_mass -= np.log(np.diag(factor[0])).sum()
    log_normalizer = float(logsumexp(log_mass))
    probabilities = np.exp(log_mass - log_normalizer)
    if not np.isfinite(means).all() or not np.isfinite(log_normalizer):
        raise FloatingPointError("nonfinite exact Gaussian mixture")
    return dict(probabilities=probabilities, means=means, covariance=covariance,
                log_normalizer=log_normalizer)


def _exact(exact):
    probability = _probabilities(exact["probabilities"], "mixture probabilities")
    means = _array(exact["means"], "mixture means")
    covariance = _array(exact["covariance"], "mixture covariance")
    if means.ndim != 2 or means.shape[0] != len(probability) or means.shape[1] < 1:
        raise ValueError("mixture means must have shape (M,d)")
    dimension = means.shape[1]
    if covariance.shape != (dimension, dimension):
        raise ValueError("invalid covariance shape")
    if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0):
        raise ValueError("covariance must be symmetric")
    np.linalg.cholesky(covariance)
    if "log_normalizer" in exact and not np.isfinite(exact["log_normalizer"]):
        raise ValueError("log_normalizer must be finite")
    return probability, means, covariance


def _directions(directions, dimension):
    result = _array(directions, "evaluation directions")
    if result.ndim != 2 or result.shape[1] != dimension or not len(result):
        raise ValueError("evaluation directions must have shape (K,d)")
    if not np.allclose(np.linalg.norm(result, axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError("evaluation directions must be unit vectors")
    return result


def _samples(x, w, dimension):
    x = _array(x, "samples")
    w = _probabilities(w, "sample weights")
    if x.ndim != 2 or x.shape != (len(w), dimension):
        raise ValueError("samples and weights must have shapes (N,d) and (N,)")
    return x, w


def _cdf(z, probability, means, sigma):
    standardized = (np.asarray(z)[..., None] - means) / sigma
    return ndtr(standardized) @ probability


def _stoploss(z, probability, means, sigma, *, upper=False):
    delta = np.asarray(z)[..., None] - means
    if upper:
        delta = -delta
    standardized = delta / sigma
    density = np.exp(-.5 * standardized ** 2) / np.sqrt(2 * np.pi)
    return (delta * ndtr(standardized) + sigma * density) @ probability


def weighted_cdf_w1(values, weights, probabilities, means, sigma):
    """返回 w1 与交点二分误差界；该界不包含浮点舍入误差。"""
    values = _array(values, "projected samples")
    weights = _probabilities(weights, "sample weights")
    probability = _probabilities(probabilities, "mixture probabilities")
    means = _array(means, "projected mixture means")
    sigma = _array(sigma, "projected standard deviation")
    if values.ndim != 1 or values.shape != weights.shape or means.shape != probability.shape:
        raise ValueError("invalid one-dimensional distribution shapes")
    if sigma.ndim != 0 or sigma <= 0:
        raise ValueError("projected standard deviation must be a positive scalar")
    # 先校验全部输入，再丢弃零权重；平移减小原函数相减的舍入误差。
    keep = weights > 0
    center = float(probability @ means)
    ordered = np.argsort(values[keep], kind="stable")
    points = values[keep][ordered] - center
    means = means - center
    points, first = np.unique(points, return_index=True)
    mass = np.add.reduceat(weights[keep][ordered], first)
    empirical = np.minimum(np.cumsum(mass)[:-1], 1.0)
    cdf = _cdf(points, probability, means, sigma)
    primitive = _stoploss(points, probability, means, sigma)
    tails = primitive[0] + _stoploss(points[-1], probability, means, sigma, upper=True)
    left, right = points[:-1], points[1:]
    width = right - left
    integral = np.empty(len(width), dtype=np.float64)
    below = empirical <= cdf[:-1]
    above = (~below) & (empirical >= cdf[1:])
    crossing = ~(below | above)
    signed = primitive[1:] - primitive[:-1] - empirical * width
    integral[below], integral[above] = signed[below], -signed[above]
    root_error = 0.0
    if np.any(crossing):
        lo, hi = left[crossing].copy(), right[crossing].copy()
        level = empirical[crossing]
        for _ in range(40):
            middle = lo + .5 * (hi - lo)
            below_root = _cdf(middle, probability, means, sigma) < level
            lo = np.where(below_root, middle, lo)
            hi = np.where(below_root, hi, middle)
        root = lo + .5 * (hi - lo)
        integral[crossing] = (primitive[:-1][crossing] + primitive[1:][crossing]
                              - 2 * _stoploss(root, probability, means, sigma)
                              + level * (2 * root - left[crossing] - right[crossing]))
        root_error = float(2 * np.sum((hi - lo) * np.abs(_cdf(root, probability, means, sigma) - level)))
    rounding_scale = 1 + np.abs(primitive[:-1]) + np.abs(primitive[1:]) + np.abs(empirical * width)
    if np.any(integral < -256 * np.finfo(float).eps * rounding_scale):
        raise FloatingPointError("negative interval integral beyond rounding tolerance")
    value = float(tails + np.maximum(integral, 0).sum())
    if not np.isfinite(value) or value < 0 or not np.isfinite(root_error):
        raise FloatingPointError("invalid projected Wasserstein integral")
    return dict(w1=value, quadrature_error_bound=root_error)


def projected_w1(x, w, exact, directions):
    """沿提供的单位方向平均 W1；生产配置为 32 个方向，测试可提供更少方向。"""
    probability, means, covariance = _exact(exact)
    x, w = _samples(x, w, means.shape[1])
    directions = _directions(directions, means.shape[1])
    results = [weighted_cdf_w1(x @ direction, w, probability, means @ direction,
                              np.sqrt(direction @ covariance @ direction)) for direction in directions]
    return dict(sliced_w1_32=float(np.mean([row["w1"] for row in results])),
                quadrature_error_bound=float(np.mean([row["quadrature_error_bound"] for row in results])))


def metrics(x, w, exact, directions):
    """均值 L2 误差除以 sqrt(d)，协方差 Frobenius 误差除以 d；不定义区域质量指标。"""
    probability, means, covariance = _exact(exact)
    dimension = means.shape[1]
    x, w = _samples(x, w, dimension)
    reference_mean = probability @ means
    centered_means = means - reference_mean
    reference_covariance = covariance + centered_means.T @ (centered_means * probability[:, None])
    mean = w @ x
    centered = x - mean
    observed_covariance = centered.T @ (centered * w[:, None])
    result = projected_w1(x, w, exact, directions)
    result.update(mean_error=float(np.linalg.norm(mean - reference_mean) / np.sqrt(dimension)),
                  covariance_error=float(np.linalg.norm(observed_covariance - reference_covariance) / dimension))
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError("nonfinite posterior metric")
    return result


def model_error(learned, true, directions):
    """两个 exact mixture 或因子参数字典的投影 W1；SciPy quad 的 epsabs 为 1e-8。"""
    learned = learned if "probabilities" in learned else exact_mixture(learned)
    true = true if "probabilities" in true else exact_mixture(true)
    lp, lm, lc = _exact(learned)
    tp, tm, tc = _exact(true)
    if lm.shape[1] != tm.shape[1]:
        raise ValueError("reference dimensions differ")
    directions = _directions(directions, lm.shape[1])
    errors = []
    for direction in directions:
        left, right = lm @ direction, tm @ direction
        ls, ts = np.sqrt(direction @ lc @ direction), np.sqrt(direction @ tc @ direction)
        center = .5 * (lp @ left + tp @ right)
        left, right = left - center, right - center
        # 分段覆盖各分量中心和尾部，避免无限区间积分漏过远处的窄峰。
        knots = np.unique(np.concatenate([left + k * ls for k in (-8, 0, 8)]
                                         + [right + k * ts for k in (-8, 0, 8)]))
        boundaries = np.concatenate(([-np.inf], knots, [np.inf]))

        def difference(z):
            if z >= 0:
                return abs(float(ndtr((right - z) / ts) @ tp - ndtr((left - z) / ls) @ lp))
            return abs(float(ndtr((z - left) / ls) @ lp - ndtr((z - right) / ts) @ tp))

        total = 0.0
        tolerance = 1e-8 / (len(boundaries) - 1)
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            outcome = quad(difference, a, b, epsabs=tolerance, epsrel=1e-10, limit=200, full_output=1)
            if len(outcome) != 3:
                raise RuntimeError(f"model-error quadrature did not converge: {outcome[3]}")
            total += outcome[0]
        if not np.isfinite(total) or total < 0:
            raise FloatingPointError("invalid model-error integral")
        errors.append(total)
    return float(np.mean(errors))
