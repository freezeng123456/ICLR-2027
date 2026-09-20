from dataclasses import asdict, dataclass
import hashlib
import math
import time
from collections.abc import Mapping

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp
import torch


PARAMETER_NAMES = (
    "directions", "noise_std", "positive_probability", "observations"
)
LOG_2PI = math.log(2.0 * math.pi)


def _integer(name, value, minimum, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its supported range")
    return int(value)


def _finite(name, value):
    valid = torch.isfinite(value).all() if isinstance(value, torch.Tensor) else np.isfinite(value).all()
    if not bool(valid):
        raise FloatingPointError(f"nonfinite {name}")


def _device(device):
    result = torch.device(device)
    if result.type not in ("cpu", "cuda"):
        raise ValueError("device must be CPU or CUDA for float64")
    if result.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        result = torch.device("cuda", torch.cuda.current_device() if result.index is None else result.index)
    return result


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parameters(problem):
    values = []
    for name in PARAMETER_NAMES:
        raw = problem[name] if isinstance(problem, Mapping) else getattr(problem, name)
        if isinstance(raw, torch.Tensor):
            raw = raw.detach().cpu().numpy()
        value = np.array(raw, dtype=np.float64, copy=True)
        _finite(name, value)
        values.append(value)
    directions, noise, probability, observations = values
    if directions.ndim != 2 or min(directions.shape) < 1 or directions.shape[0] > 16:
        raise ValueError("directions must have shape (G,d), with 1 <= G <= 16 and d >= 1")
    if any(value.shape != (len(directions),) for value in values[1:]):
        raise ValueError("sensor parameters must have shape (G,)")
    if np.any(np.linalg.norm(directions, axis=1) == 0) or np.any(noise <= 0):
        raise ValueError("directions must be nonzero and noise_std must be positive")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("sign probabilities must be in [0,1]")
    return directions, noise, probability, observations


def _log_probabilities(probability):
    result = np.full((len(probability), 2), -np.inf, dtype=np.float64)
    weights = np.column_stack((1.0 - probability, probability))
    np.log(weights, out=result, where=weights > 0)
    return result


@dataclass
class EvaluationCounts:
    factor_preparation_calls: int = 0
    mode_factor_calls: int = 0
    particle_factor_calls: int = 0
    mode_point_evaluations: int = 0
    target_point_evaluations: int = 0
    reference_component_evaluations: int = 0
    reference_samples: int = 0
    mh_proposals: int = 0
    mh_accepted: int = 0
    sign_proposals: int = 0
    sign_accepted: int = 0
    sign_factor_calls: int = 0
    resampling_calls: int = 0

    def report(self):
        result = asdict(self)
        result["factor_calls"] = (
            self.factor_preparation_calls + self.mode_factor_calls + self.particle_factor_calls
        )
        return result


@dataclass
class ReferenceSpec:
    weights: np.ndarray
    means: np.ndarray
    covariance: np.ndarray
    tail_precision: np.ndarray
    search: dict
    evaluation_counts: dict


def _construct_reference(parameters, reference, components, mode_steps, covariance_scale, counts):
    if reference not in ("prior", "gaussian", "mixture"):
        raise ValueError("reference must be 'prior', 'gaussian', or 'mixture'")
    components = _integer("components", components, 1, 4)
    mode_steps = _integer("mode_steps", mode_steps, 1)
    if isinstance(covariance_scale, (bool, np.bool_)) or not np.isfinite(covariance_scale) or covariance_scale < 1:
        raise ValueError("covariance_scale must be finite and at least one")
    directions, noise, probability, observations = parameters
    groups, dimension = directions.shape
    counts.factor_preparation_calls += groups
    if reference == "prior":
        return ReferenceSpec(
            np.ones(1), np.zeros((1, dimension)), np.eye(dimension), np.eye(dimension),
            {"algorithm": "prior_identity", "mode_steps": 0, "starts": 0,
             "requested_components": 1, "actual_components": 1}, counts.report()
        )
    scaled = directions / noise[:, None]
    precision = np.eye(dimension) + scaled.T @ scaled
    tail_cholesky = cho_factor(precision, lower=True)
    covariance = cho_solve(tail_cholesky, np.eye(dimension))
    covariance = 0.5 * (covariance + covariance.T)
    linear = (observations / noise ** 2)[:, None] * directions
    log_weights = _log_probabilities(probability)

    affine = cho_solve(tail_cholesky, ((2.0 * probability - 1.0)[:, None] * linear).sum(0))
    all_positive = cho_solve(tail_cholesky, linear.sum(0))
    aligned_signs = np.where(directions @ directions.T >= 0.0, 1.0, -1.0)
    aligned = cho_solve(tail_cholesky, (aligned_signs @ linear).T).T
    starts = np.vstack((np.zeros(dimension), affine, -affine,
                        all_positive, -all_positive, aligned, -aligned))
    centers = starts.copy()

    # latent-sign EM 使用固定尾部精度；每次迭代同时评价全部候选中心。
    for _ in range(mode_steps):
        projection = centers @ linear.T
        terms = np.stack((-projection + log_weights[:, 0],
                          projection + log_weights[:, 1]), axis=-1)
        responsibilities = np.exp(terms[..., 1] - logsumexp(terms, axis=-1))
        centers = cho_solve(tail_cholesky, ((2.0 * responsibilities - 1.0) @ linear).T).T
        counts.mode_point_evaluations += len(centers)
        counts.mode_factor_calls += len(centers) * groups
        _finite("mode search", centers)

    projection = centers @ linear.T
    terms = np.stack((-projection + log_weights[:, 0],
                      projection + log_weights[:, 1]), axis=-1)
    log_terms = logsumexp(terms, axis=-1)
    constant = -0.5 * (dimension * LOG_2PI + np.sum(LOG_2PI + 2.0 * np.log(noise)
                                                   + (observations / noise) ** 2))
    scores = constant - 0.5 * np.einsum("ni,ij,nj->n", centers, precision, centers) + log_terms.sum(1)
    responsibility = np.exp(terms[..., 1] - log_terms)
    gradient = -centers @ precision + (2.0 * responsibility - 1.0) @ linear
    counts.mode_point_evaluations += len(centers)
    counts.mode_factor_calls += len(centers) * groups
    _finite("mode heights", scores)
    requested = 1 if reference == "gaussian" else components
    selected = []
    for index in np.argsort(-scores, kind="stable"):
        if selected:
            delta = centers[selected] - centers[index]
            distance2 = np.einsum("ni,ij,nj->n", delta, precision, delta)
            if np.min(distance2) <= 1e-8:
                continue
        selected.append(int(index))
        if len(selected) == requested:
            break
    selected_scores = scores[selected]
    weights = np.exp(selected_scores - logsumexp(selected_scores))
    if np.any(weights <= 0):
        raise FloatingPointError("reference component weight underflow")
    search = {
        "algorithm": "fixed_tail_precision_latent_sign_em",
        "mode_steps": mode_steps,
        "starts": len(starts),
        "requested_components": requested,
        "actual_components": len(selected),
        "merge_squared_mahalanobis_tolerance": 1e-8,
        "selected_start_indices": selected,
        "selected_log_joint": selected_scores.tolist(),
        "selected_score_norms": np.linalg.norm(gradient[selected], axis=1).tolist(),
        "all_terminal_log_joint": scores.tolist(),
        "all_terminal_score_norms": np.linalg.norm(gradient, axis=1).tolist(),
        "weight_rule": "normalized_target_heights_common_covariance",
    }
    return ReferenceSpec(weights, centers[selected].copy(), covariance_scale * covariance, precision,
                         search, counts.report())


def build_reference(problem, *, reference="mixture", components=2, mode_steps=32, covariance_scale=1.0):
    return _construct_reference(_parameters(problem), reference, components,
                                mode_steps, covariance_scale, EvaluationCounts())


class SensorTarget:
    def __init__(self, problem, device="cpu", counts=None):
        parameters = _parameters(problem)
        self.device = _device(device)
        self.counts = EvaluationCounts() if counts is None else counts
        self.groups, self.dimension = parameters[0].shape
        self.directions, self.noise, _, self.observations = [
            torch.as_tensor(value, dtype=torch.float64, device=self.device) for value in parameters
        ]
        self.log_weights = torch.as_tensor(_log_probabilities(parameters[2]),
                                           dtype=torch.float64, device=self.device)

    def log_prob(self, points):
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError("points must have shape (N,d)")
        if points.dtype != torch.float64 or points.device != self.directions.device:
            raise ValueError("points must be float64 on the target device")
        projection = points @ self.directions.T
        negative = self.log_weights[:, 0] - 0.5 * ((self.observations + projection) / self.noise).square()
        positive = self.log_weights[:, 1] - 0.5 * ((self.observations - projection) / self.noise).square()
        likelihood = (torch.logaddexp(negative, positive) - self.noise.log() - 0.5 * LOG_2PI).sum(1)
        self.counts.target_point_evaluations += len(points)
        self.counts.particle_factor_calls += len(points) * self.groups
        return likelihood - 0.5 * (points.square().sum(1) + self.dimension * LOG_2PI)


def _categorical(probabilities, generator):
    cdf = torch.cumsum(probabilities, dim=-1)
    cdf[..., -1] = 1.0
    draws = torch.rand((len(probabilities), 1), dtype=torch.float64,
                       device=probabilities.device, generator=generator)
    return torch.searchsorted(cdf.contiguous(), draws, right=True).squeeze(1)


class GaussianReference:
    def __init__(self, weights, means, covariance, device="cpu", counts=None):
        self.device = _device(device)
        self.counts = EvaluationCounts() if counts is None else counts
        self.weights, self.means, self.covariance = [
            torch.as_tensor(value, dtype=torch.float64, device=self.device).clone()
            for value in (weights, means, covariance)
        ]
        if self.means.ndim != 2 or min(self.means.shape) < 1:
            raise ValueError("reference means must have shape (C,d)")
        self.components, self.dimension = self.means.shape
        if self.weights.shape != (self.components,) or self.covariance.shape != (self.dimension, self.dimension):
            raise ValueError("invalid reference shapes")
        for name in ("weights", "means", "covariance"):
            _finite("reference " + name, getattr(self, name))
        if not bool((self.weights > 0).all()) or abs(float(self.weights.sum()) - 1.0) > 1e-12:
            raise ValueError("reference weights must be positive and sum to one")
        if not torch.allclose(self.covariance, self.covariance.T, atol=1e-12, rtol=0):
            raise ValueError("reference covariance must be symmetric")
        self.cholesky = torch.linalg.cholesky(self.covariance)
        self.log_determinant = 2.0 * self.cholesky.diag().log().sum()
        self.log_weights = self.weights.log()

    def _component_logs(self, points):
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError("reference points must have shape (N,d)")
        if points.dtype != torch.float64 or points.device != self.means.device:
            raise ValueError("reference points must be float64 on the reference device")
        delta = points[:, None] - self.means
        whitened = torch.linalg.solve_triangular(
            self.cholesky, delta.reshape(-1, self.dimension).T, upper=False
        ).T.reshape(delta.shape)
        self.counts.reference_component_evaluations += len(points) * self.components
        return self.log_weights - 0.5 * (
            whitened.square().sum(-1) + self.log_determinant + self.dimension * LOG_2PI
        )

    def log_prob(self, points):
        return torch.logsumexp(self._component_logs(points), dim=1)

    def responsibilities(self, points):
        logs = self._component_logs(points)
        return torch.exp(logs - torch.logsumexp(logs, 1, keepdim=True))

    def sample(self, n, motion, auxiliary):
        n = _integer("n", n, 1)
        components = _categorical(self.weights.expand(n, -1), auxiliary)
        noise = torch.randn((n, self.dimension), dtype=torch.float64,
                            device=self.device, generator=motion)
        self.counts.reference_samples += n
        return self.means[components] + noise @ self.cholesky.T

    def propose(self, points, scale, motion, auxiliary):
        if not np.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError("proposal scale must be in (0,1]")
        indices = _categorical(self.responsibilities(points), auxiliary)
        centers = self.means[indices]
        noise = torch.randn(points.shape, dtype=torch.float64,
                            device=self.device, generator=motion)
        return centers + math.sqrt(1.0 - scale * scale) * (points - centers) + scale * (noise @ self.cholesky.T)

    def transition_log_prob(self, origin, destination, scale):
        if origin.shape != destination.shape or not np.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError("invalid transition points or scale")
        logs = self._component_logs(origin)
        log_responsibility = logs - torch.logsumexp(logs, dim=1, keepdim=True)
        centers = self.means + math.sqrt(1.0 - scale * scale) * (origin[:, None] - self.means)
        delta = (destination[:, None] - centers) / scale
        whitened = torch.linalg.solve_triangular(
            self.cholesky, delta.reshape(-1, self.dimension).T, upper=False
        ).T.reshape(delta.shape)
        transition = -0.5 * (whitened.square().sum(-1) + self.log_determinant
                              + self.dimension * (LOG_2PI + 2.0 * math.log(scale)))
        return torch.logsumexp(log_responsibility + transition, dim=1)


def make_generators(seed, device="cpu"):
    seed = _integer("seed", seed, 0, 2 ** 63 - 1)
    device = _device(device)
    return tuple(torch.Generator(device=device).manual_seed((seed + offset) % (2 ** 63))
                 for offset in (0, 1000003, 2000003))


def _next_beta(residual, beta, ess_target):
    # 中心化消除残差常数；二分中的比较保持在设备上。
    centered = residual - residual.max()
    target_log_ess = math.log(len(residual) * ess_target)

    def log_ess(delta):
        log_increment = delta * centered
        return 2.0 * torch.logsumexp(log_increment, 0) - torch.logsumexp(2.0 * log_increment, 0)

    if float(log_ess(1.0 - beta)) >= target_log_ess:
        return 1.0
    lower = torch.zeros((), dtype=torch.float64, device=residual.device)
    upper = torch.full_like(lower, 1.0 - beta)
    for _ in range(48):
        middle = (lower + upper) / 2.0
        acceptable = log_ess(middle) >= target_log_ess
        lower = torch.where(acceptable, middle, lower)
        upper = torch.where(acceptable, upper, middle)
    result = beta + float(lower)
    if not beta < result < 1.0:
        raise FloatingPointError("adaptive temperature failed to make representable progress")
    return result


def _reweight(log_weights, residual, increment):
    raw = log_weights + increment * residual
    log_ratio = torch.logsumexp(raw, dim=0)
    _finite("incremental normalizer", log_ratio)
    updated = raw - log_ratio
    ess = torch.exp(-torch.logsumexp(2.0 * updated, dim=0))
    return updated, log_ratio, ess


def _resample(weights, generator):
    cdf = weights.cumsum(0)
    cdf[-1] = 1.0
    positions = (torch.arange(len(weights), dtype=torch.float64, device=weights.device)
                 + torch.rand((), dtype=torch.float64, device=weights.device, generator=generator)) / len(weights)
    return torch.searchsorted(cdf, positions, right=True)


def metropolis_step(target, reference, points, residual, beta, scale, motion, auxiliary):
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be in [0,1]")
    proposed = reference.propose(points, scale, motion, auxiliary)
    candidate_residual = target.log_prob(proposed) - reference.log_prob(proposed)
    _finite("MH proposed residual", candidate_residual)
    log_acceptance = torch.minimum(beta * (candidate_residual - residual), torch.zeros_like(residual))
    uniforms = torch.rand((len(points),), dtype=torch.float64,
                          device=points.device, generator=auxiliary)
    accepted = uniforms.log() < log_acceptance
    return (torch.where(accepted[:, None], proposed, points),
            torch.where(accepted, candidate_residual, residual), accepted)


def sign_flip_step(target, reference, points, residual, beta, auxiliary):
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be in [0,1]")
    current_log_q = reference.log_prob(points)
    proposed = -points
    proposed_log_q = reference.log_prob(proposed)
    proposed_residual = target.log_prob(proposed) - proposed_log_q
    _finite("sign proposal residual", proposed_residual)
    log_ratio = proposed_log_q - current_log_q + beta * (proposed_residual - residual)
    _finite("sign proposal log ratio", log_ratio)
    log_uniform = torch.rand((len(points),), dtype=torch.float64,
                             device=points.device, generator=auxiliary).log()
    accepted = log_uniform < torch.minimum(log_ratio, torch.zeros_like(log_ratio))
    return (torch.where(accepted[:, None], proposed, points),
            torch.where(accepted, proposed_residual, residual), accepted, log_ratio)


def _schedule(temperatures, max_stages):
    if temperatures is None:
        return None
    values = np.asarray(temperatures, dtype=np.float64)
    if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
            or values[0] != 0 or values[-1] != 1 or not (np.diff(values) > 0).all()
            or len(values) - 1 > max_stages):
        raise ValueError("temperatures must increase strictly from 0 to 1 within max_stages")
    return values


def _output_statistics(points, weights):
    mean = weights @ points
    centered = points - mean
    covariance = centered.T @ (centered * weights[:, None])
    quantiles = np.empty((points.shape[1], 4), dtype=np.float64)
    for index in range(points.shape[1]):
        order = np.argsort(points[:, index], kind="stable")
        cdf = weights[order].cumsum()
        cdf[-1] = 1.0
        quantiles[index] = points[order[np.searchsorted(cdf, [0.025, 0.05, 0.95, 0.975])], index]
    return {"weighted_mean": mean.tolist(), "weighted_covariance": covariance.tolist(),
            "quantile_levels": [0.025, 0.05, 0.95, 0.975],
            "weighted_quantiles": quantiles.tolist(),
            "ess": float(1.0 / np.sum(weights ** 2)),
            "weight_min": float(weights.min()), "weight_max": float(weights.max())}


@torch.no_grad()
def sample(problem, particles=8192, seed=0, *, reference="mixture", components=2,
           ess_target=0.8, moves=3, proposal_scale=0.5, mode_steps=32,
           covariance_scale=1.0, sign_flip=True, max_stages=256, temperatures=None,
           device="cpu", return_numpy=True):
    particles = _integer("particles", particles, 2)
    moves = _integer("moves", moves, 0)
    max_stages = _integer("max_stages", max_stages, 1)
    seed = _integer("seed", seed, 0, 2 ** 63 - 1)
    components = _integer("components", components, 1, 4)
    mode_steps = _integer("mode_steps", mode_steps, 1)
    if not np.isfinite(ess_target) or not 0 < ess_target < 1:
        raise ValueError("ess_target must be in (0,1)")
    if not np.isfinite(proposal_scale) or not 0 < proposal_scale <= 1:
        raise ValueError("proposal_scale must be in (0,1]")
    if not isinstance(return_numpy, bool):
        raise ValueError("return_numpy must be boolean")
    if not isinstance(sign_flip, bool):
        raise ValueError("sign_flip must be boolean")
    schedule = _schedule(temperatures, max_stages)
    device = _device(device)
    _synchronize(device)
    started = time.perf_counter()
    counts = EvaluationCounts()
    parameters = _parameters(problem)
    spec = _construct_reference(parameters, reference, components, mode_steps, covariance_scale, counts)
    target = SensorTarget(dict(zip(PARAMETER_NAMES, parameters)), device, counts)
    q = GaussianReference(spec.weights, spec.means, spec.covariance, device, counts)
    motion, auxiliary, resampling = make_generators(seed, device)
    _synchronize(device)
    prepared = time.perf_counter()

    points = q.sample(particles, motion, auxiliary)
    residual = target.log_prob(points) - q.log_prob(points)
    _finite("initial residual", residual)
    log_weights = torch.full((particles,), -math.log(particles), dtype=torch.float64, device=device)
    log_z = 0.0
    beta = 0.0
    records = []
    _synchronize(device)
    initialized = time.perf_counter()

    for stage in range(max_stages):
        next_beta = _next_beta(residual, beta, ess_target) if schedule is None else float(schedule[stage + 1])
        old_beta = beta
        log_weights, increment, ess = _reweight(log_weights, residual, next_beta - beta)
        log_z += float(increment)
        weights = log_weights.exp()
        resampled = next_beta < 1.0
        residual_mean = float(weights @ residual)
        residual_variance = float(weights @ (residual - residual_mean).square())
        if resampled:
            indices = _resample(weights, resampling)
            points, residual = points[indices], residual[indices]
            log_weights.fill_(-math.log(particles))
            counts.resampling_calls += 1
        beta = next_beta
        accepted = torch.zeros((), dtype=torch.int64, device=device)
        for _ in range(moves):
            points, residual, accept = metropolis_step(
                target, q, points, residual, beta, proposal_scale, motion, auxiliary
            )
            accepted += accept.sum()
        accepted_count = int(accepted)
        counts.mh_proposals += moves * particles
        counts.mh_accepted += accepted_count
        sign_accepted_count = 0
        if sign_flip:
            points, residual, sign_accept, _ = sign_flip_step(target, q, points, residual, beta, auxiliary)
            sign_accepted_count = int(sign_accept.sum())
            counts.sign_proposals += particles
            counts.sign_accepted += sign_accepted_count
            counts.sign_factor_calls += particles * target.groups
        _finite("stage particles", points)
        records.append({
            "stage": stage, "beta_previous": old_beta, "beta": beta,
            "delta_beta": beta - old_beta, "ess_fraction": float(ess) / particles,
            "resampled": resampled, "log_normalizer_increment": float(increment),
            "log_normalizer": log_z, "acceptance": accepted_count / (moves * particles) if moves else None,
            "mh_proposals": moves * particles, "mh_accepted": accepted_count,
            "sign_acceptance": sign_accepted_count / particles if sign_flip else None,
            "sign_proposals": particles if sign_flip else 0, "sign_accepted": sign_accepted_count,
            "sign_factor_calls": particles * target.groups if sign_flip else 0,
            "factor_calls": (moves + int(sign_flip)) * particles * target.groups,
            "weighted_residual_mean_before_move": residual_mean,
            "weighted_residual_variance_before_move": residual_variance,
        })
        if beta == 1.0:
            break
    if beta != 1.0:
        raise RuntimeError(f"max_stages={max_stages} reached at beta={beta}; no completed sample")
    weights = log_weights.exp()
    _finite("final weights", weights)
    if abs(float(weights.sum()) - 1.0) > 1e-10 or not math.isfinite(log_z):
        raise FloatingPointError("invalid final weight normalization or logZ")
    _synchronize(device)
    sampled = time.perf_counter()

    component_mass = weights @ q.responsibilities(points)
    cpu_points = points.cpu().numpy()
    cpu_weights = weights.cpu().numpy()
    parameter_hash = hashlib.sha256()
    for name, value in zip(PARAMETER_NAMES, parameters):
        parameter_hash.update(name.encode("ascii"))
        parameter_hash.update(np.asarray(value.shape, dtype="<i8").tobytes())
        parameter_hash.update(value.astype("<f8", copy=False).tobytes())
    metadata = {
        "status": "completed", "method": "tail_informed_annealed_smc",
        "target": "normalized_prior_times_normalized_sensor_likelihoods",
        "logZ_semantics": "estimated_log_joint_observed_data_evidence",
        "reference": reference, "components_requested": components,
        "components_actual": q.components, "particles": particles,
        "groups": target.groups, "dimension": target.dimension, "seed": seed,
        "device": str(device), "dtype": "float64", "return_numpy": return_numpy,
        "ess_target": ess_target, "moves": moves, "proposal_scale": proposal_scale,
        "covariance_scale": float(covariance_scale),
        "covariance_scale_applied": 1.0 if reference == "prior" else float(covariance_scale),
        "sign_flip": sign_flip,
        "mode_steps": mode_steps, "max_stages": max_stages,
        "temperature_strategy": "adaptive_ess" if schedule is None else "fixed",
        "temperatures": [0.0] + [item["beta"] for item in records],
        "stages": len(records), "records": records,
        "random_streams": {"motion_seed": seed, "auxiliary_seed": (seed + 1000003) % (2 ** 63),
                           "resampling_seed": (seed + 2000003) % (2 ** 63)},
        "parameter_sha256": parameter_hash.hexdigest(),
        "reference_parameters": {"weights": spec.weights.tolist(), "means": spec.means.tolist(),
                                 "covariance": spec.covariance.tolist(),
                                 "tail_precision": spec.tail_precision.tolist()},
        "mode_search": spec.search,
        "weighted_reference_responsibility_mass": component_mass.cpu().tolist(),
        "statistics": _output_statistics(cpu_points, cpu_weights),
        "normalizer_unbiasedness_claim": False,
    }
    _synchronize(device)
    finished = time.perf_counter()
    timing = {"total_seconds": finished - started, "preparation_seconds": prepared - started,
              "initialization_seconds": initialized - prepared, "annealing_seconds": sampled - initialized,
              "output_seconds": finished - sampled, "includes_cpu_output_and_statistics": True}
    result_points, result_weights = (cpu_points, cpu_weights) if return_numpy else (points, weights)
    return result_points, result_weights, log_z, timing, counts.report(), metadata
