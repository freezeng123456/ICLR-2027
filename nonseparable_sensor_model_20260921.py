from dataclasses import dataclass
import json
import numpy as np
from scipy.special import logsumexp, ndtr


def _validate_time(time):
    value = float(time)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("time must be finite and nonnegative")
    return value


def _log_normal(x, mean, covariance):
    delta = np.asarray(x, dtype=np.float64) - mean
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise np.linalg.LinAlgError("covariance is not positive definite")
    solved = np.linalg.solve(covariance, delta)
    return -0.5 * (delta.size * np.log(2.0 * np.pi) + logdet + delta @ solved)


def _sign_matrix(groups):
    count = 1 << groups
    bits = ((np.arange(count, dtype=np.uint64)[:, None] >> np.arange(groups, dtype=np.uint64)) & 1).astype(np.int64)
    return bits


@dataclass
class SensorProblem:
    directions: np.ndarray
    noise_std: np.ndarray
    positive_probability: np.ndarray
    observations: np.ndarray
    truth: np.ndarray

    def __post_init__(self):
        self.directions = np.asarray(self.directions, dtype=np.float64)
        self.noise_std = np.asarray(self.noise_std, dtype=np.float64)
        self.positive_probability = np.asarray(self.positive_probability, dtype=np.float64)
        self.observations = np.asarray(self.observations, dtype=np.float64)
        self.truth = np.asarray(self.truth, dtype=np.float64)
        if self.directions.ndim != 2:
            raise ValueError("directions must have shape (G, d)")
        groups, dimension = self.directions.shape
        if groups < 1 or groups > 16 or dimension < 1:
            raise ValueError("G must be in [1, 16] and d must be positive")
        if not np.all(np.isfinite(self.directions)) or np.any(np.linalg.norm(self.directions, axis=1) == 0.0):
            raise ValueError("directions must be finite and nonzero")
        for value, name in ((self.noise_std, "noise_std"), (self.positive_probability, "positive_probability"),
                            (self.observations, "observations")):
            if value.shape != (groups,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must have shape (G,) and finite values")
        if np.any(self.noise_std <= 0.0):
            raise ValueError("noise_std must be positive")
        if np.any((self.positive_probability <= 0.0) | (self.positive_probability >= 1.0)):
            raise ValueError("positive_probability must lie strictly between zero and one")
        if self.truth.shape != (dimension,) or not np.all(np.isfinite(self.truth)):
            raise ValueError("truth must have shape (d,) and finite values")

    @property
    def groups(self):
        return self.directions.shape[0]

    @property
    def dimension(self):
        return self.directions.shape[1]

    def to_dict(self):
        return {
            "directions": self.directions.tolist(),
            "noise_std": self.noise_std.tolist(),
            "positive_probability": self.positive_probability.tolist(),
            "observations": self.observations.tolist(),
            "truth": self.truth.tolist(),
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)

    def to_json(self):
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True)

    @classmethod
    def from_json(cls, payload):
        return cls.from_dict(json.loads(payload))

    def _factor_base(self):
        identity = np.eye(self.dimension, dtype=np.float64)
        covariance = np.empty((self.groups, self.dimension, self.dimension), dtype=np.float64)
        means = np.empty((self.groups, 2, self.dimension), dtype=np.float64)
        for group in range(self.groups):
            direction = self.directions[group]
            denominator = self.noise_std[group] ** 2 + direction @ direction
            covariance[group] = identity - np.outer(direction, direction) / denominator
            center = self.observations[group] * direction / denominator
            means[group, 0] = -center
            means[group, 1] = center
        weights = np.stack((1.0 - self.positive_probability, self.positive_probability), axis=1)
        return covariance, means, weights

    def factor_parameters(self, time=0.0):
        time = _validate_time(time)
        covariance, means, weights = self._factor_base()
        rho = np.exp(-0.5 * time)
        identity = np.eye(self.dimension, dtype=np.float64)
        noised_covariance = rho * rho * covariance + (1.0 - rho * rho) * identity
        precision = np.empty_like(noised_covariance)
        for group in range(self.groups):
            precision[group] = np.linalg.inv(noised_covariance[group])
        return precision, rho * means, weights

    def _posterior_terms(self, time=0.0):
        precision, means, weights = self.factor_parameters(time)
        identity = np.eye(self.dimension, dtype=np.float64)
        total_precision = np.sum(precision, axis=0) - (self.groups - 1) * identity
        eigenvalues = np.linalg.eigvalsh(total_precision)
        if np.min(eigenvalues) <= 0.0:
            raise np.linalg.LinAlgError("composed posterior precision is not positive definite")
        covariance = np.linalg.inv(total_precision)
        sign, logdet_precision = np.linalg.slogdet(total_precision)
        if sign <= 0:
            raise np.linalg.LinAlgError("composed posterior precision has invalid determinant")
        signs = _sign_matrix(self.groups)
        precision_means = np.einsum("gij,gcj->gci", precision, means)
        selected_precision_means = precision_means[np.arange(self.groups)[None, :], signs]
        b_values = np.sum(selected_precision_means, axis=1)
        quadratic_terms = np.einsum("gci,gci->gc", means, precision_means)
        selected_quadratic = quadratic_terms[np.arange(self.groups)[None, :], signs]
        log_weights = np.log(weights)[np.arange(self.groups)[None, :], signs]
        sign_logdet = 0.5 * np.linalg.slogdet(precision)[1]
        component_logs = (np.sum(log_weights - 0.5 * selected_quadratic, axis=1)
                          + np.sum(sign_logdet)
                          + 0.5 * np.einsum("ki,ij,kj->k", b_values, covariance, b_values))
        component_means = np.einsum("ij,kj->ki", covariance, b_values)
        log_evidence = logsumexp(component_logs) - 0.5 * logdet_precision
        return (np.exp(component_logs - logsumexp(component_logs)),
                np.asarray(component_means, dtype=np.float64), covariance, float(log_evidence))

    def exact_posterior(self, time=0.0):
        weights, means, covariance, _ = self._posterior_terms(time)
        return weights, means, covariance

    def log_composition_normalizer(self, time=0.0):
        return float(self._posterior_terms(time)[3])

    def log_evidence(self, time=0.0):
        time = _validate_time(time)
        if time != 0.0:
            raise ValueError("observed-data evidence is defined at time=0")
        variances = self.noise_std ** 2 + np.einsum("gi,gi->g", self.directions, self.directions)
        marginal_sum = np.sum(-0.5 * (np.log(2.0 * np.pi * variances)
                                      + self.observations ** 2 / variances))
        return float(self.log_composition_normalizer(0.0) + marginal_sum)

    def log_sensor_evidence(self):
        sign_bits = _sign_matrix(self.groups)
        values = np.empty(sign_bits.shape[0], dtype=np.float64)
        noise_covariance = np.diag(self.noise_std ** 2)
        for row, bits in enumerate(sign_bits):
            signs = 2.0 * bits - 1.0
            signed_directions = self.directions * signs[:, None]
            covariance = noise_covariance + signed_directions @ signed_directions.T
            values[row] = np.sum(np.log(np.where(bits, self.positive_probability,
                                                1.0 - self.positive_probability)))
            values[row] += _log_normal(self.observations, np.zeros(self.groups), covariance)
        return float(logsumexp(values))

    def _log_unnormalized_density(self, points, time=0.0):
        points = np.asarray(points, dtype=np.float64)
        original_shape = points.shape
        if points.ndim == 1:
            points = points[None, :]
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError("points must have shape (n, d) or (d,)")
        precision, means, weights = self.factor_parameters(time)
        values = np.zeros(points.shape[0], dtype=np.float64)
        prior_precision = np.eye(self.dimension, dtype=np.float64)
        for group in range(self.groups):
            covariance = np.linalg.inv(precision[group])
            terms = np.empty((points.shape[0], 2), dtype=np.float64)
            for component in range(2):
                delta = points - means[group, component]
                terms[:, component] = -0.5 * np.einsum("ni,ij,nj->n", delta, precision[group], delta)
                terms[:, component] += -0.5 * (self.dimension * np.log(2.0 * np.pi)
                                               - np.linalg.slogdet(precision[group])[1])
                terms[:, component] += np.log(weights[group, component])
            values += logsumexp(terms, axis=1)
        prior_terms = -0.5 * np.einsum("ni,ij,nj->n", points, prior_precision, points)
        prior_terms -= 0.5 * self.dimension * np.log(2.0 * np.pi)
        values -= (self.groups - 1) * prior_terms
        if original_shape == (self.dimension,):
            return values[0]
        return values

    def log_density(self, points, time=0.0):
        return self._log_unnormalized_density(points, time) - self.log_composition_normalizer(time)

    def log_sensor_posterior(self, points):
        points = np.asarray(points, dtype=np.float64)
        original_shape = points.shape
        if points.ndim == 1:
            points = points[None, :]
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError("points must have shape (n, d) or (d,)")
        prior = -0.5 * (self.dimension * np.log(2.0 * np.pi)
                        + np.einsum("ni,ni->n", points, points))
        sensor = np.zeros(points.shape[0], dtype=np.float64)
        for group in range(self.groups):
            projection = points @ self.directions[group]
            variance = self.noise_std[group] ** 2
            terms = np.stack((
                np.log1p(-self.positive_probability[group])
                - 0.5 * (np.log(2.0 * np.pi * variance)
                          + (self.observations[group] + projection) ** 2 / variance),
                np.log(self.positive_probability[group])
                - 0.5 * (np.log(2.0 * np.pi * variance)
                          + (self.observations[group] - projection) ** 2 / variance),
            ), axis=1)
            sensor += logsumexp(terms, axis=1)
        result = prior + sensor - self.log_sensor_evidence()
        if original_shape == (self.dimension,):
            return result[0]
        return result

    def sample(self, rng, n, time=0.0):
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")
        weights, means, covariance = self.exact_posterior(time)
        components = rng.choice(weights.size, size=int(n), p=weights)
        return means[components] + rng.multivariate_normal(np.zeros(self.dimension), covariance, size=int(n))

    def projected_cdf(self, direction, points, time=0.0):
        direction = np.asarray(direction, dtype=np.float64)
        points = np.asarray(points, dtype=np.float64)
        if direction.shape != (self.dimension,) or not np.all(np.isfinite(direction)):
            raise ValueError("direction must have shape (d,) and finite values")
        if np.linalg.norm(direction) == 0.0:
            raise ValueError("direction must be nonzero")
        if points.ndim == 0 or points.ndim > 1:
            raise ValueError("points must be a flat one-dimensional array")
        weights, means, covariance = self.exact_posterior(time)
        projected_means = means @ direction
        variance = direction @ covariance @ direction
        if variance <= 0.0:
            raise np.linalg.LinAlgError("projected posterior variance is not positive")
        return np.sum(weights[:, None] * ndtr((points[None, :] - projected_means[:, None]) / np.sqrt(variance)), axis=0)


def make_problem(seed, groups=12, dimension=2, regime="ambiguous"):
    if regime not in ("ambiguous", "regular"):
        raise ValueError("regime must be 'ambiguous' or 'regular'")
    if not isinstance(groups, (int, np.integer)) or groups < 1 or groups > 16:
        raise ValueError("groups must be in [1, 16]")
    if not isinstance(dimension, (int, np.integer)) or dimension < 1:
        raise ValueError("dimension must be positive")
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(groups, dimension))
    if regime == "regular" and dimension >= 2:
        angles = np.linspace(0.1, 2.7, groups) + 0.03 * rng.normal(size=groups)
        raw[:, 0] = np.cos(angles)
        raw[:, 1] = np.sin(angles)
    directions = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    truth = rng.normal(size=dimension)
    noise_std = (0.8 + 0.15 * rng.random(groups) if regime == "ambiguous"
                 else 0.35 + 0.08 * rng.random(groups))
    positive_probability = np.where(np.arange(groups) % 2 == 0, 0.65, 0.8)
    latent_positive = rng.random(groups) < positive_probability
    signs = np.where(latent_positive, 1.0, -1.0)
    observations = signs * (directions @ truth) + noise_std * rng.normal(size=groups)
    return SensorProblem(directions, noise_std, positive_probability, observations, truth)
