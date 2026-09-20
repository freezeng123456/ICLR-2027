import hashlib
import math
import time

import numpy as np
from scipy.special import logsumexp
import torch

from tail_bridge_smc_20260921 import (EvaluationCounts, GaussianReference, _device, _finite,
    _next_beta, _resample, _reweight, _synchronize, make_generators, sign_flip_step)


FIELDS = ("directions", "variance", "means", "weights")


def parameters(payload):
    if any(np.iscomplexobj(payload[name]) for name in FIELDS):
        raise ValueError("learned factor parameters must be real")
    values = {name: np.array(payload[name], dtype=np.float64, copy=True) for name in FIELDS}
    a, v, m, w = [values[name] for name in FIELDS]
    if a.ndim != 2 or not 1 <= len(a) <= 16 or a.shape[1] < 1:
        raise ValueError("invalid projection directions")
    if v.shape != (len(a),) or m.shape != (len(a), 2) or w.shape != m.shape:
        raise ValueError("invalid learned factor shapes")
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError("nonfinite learned factor parameter")
    if not np.allclose(np.linalg.norm(a, axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError("projection directions must be unit vectors")
    if np.any((v <= 0) | (v >= 1)) or np.any(w <= 0) or not np.allclose(w.sum(1), 1, atol=1e-12, rtol=0):
        raise ValueError("variance must lie in (0,1); weights must be positive and normalized")
    return values


def parameter_hash(payload):
    digest = hashlib.sha256()
    for name, value in parameters(payload).items():
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.astype("<f8").tobytes())
    return digest.hexdigest()


def tail_precision(payload):
    p = parameters(payload)
    return np.eye(p["directions"].shape[1]) + p["directions"].T @ ((1 / p["variance"] - 1)[:, None] * p["directions"])


def log_target_numpy(payload, points):
    p = parameters(payload)
    x = np.asarray(points, dtype=np.float64)
    z = x @ p["directions"].T
    factor = logsumexp(np.log(p["weights"])[None] - .5 * (
        (z[:, :, None] - p["means"][None]) ** 2 / p["variance"][None, :, None]
        + np.log(2 * np.pi * p["variance"])[None, :, None]), axis=2)
    return -.5 * (np.sum(x * x, 1) + x.shape[1] * np.log(2 * np.pi)) + (
        factor + .5 * (z * z + np.log(2 * np.pi))).sum(1)


def construct_reference(payload, kind, counts):
    p = parameters(payload)
    a, v, m, w = [p[name] for name in FIELDS]
    groups, dimension = a.shape
    precision = tail_precision(p)
    counts.factor_preparation_calls += groups
    if kind == "prior":
        return np.ones(1), np.zeros((1, dimension)), np.eye(dimension), dict(algorithm="prior", starts=0, steps=0)
    if kind not in ("gaussian", "mixture"):
        raise ValueError("unknown reference")
    covariance = np.linalg.inv(precision)
    covariance = .5 * (covariance + covariance.T)
    centers = np.random.default_rng(20261020).normal(size=(16, dimension))
    for _ in range(50):
        z = centers @ a.T
        logits = np.log(w)[None] - .5 * (z[:, :, None] - m[None]) ** 2 / v[None, :, None]
        responsibilities = np.exp(logits - logsumexp(logits, axis=2, keepdims=True))
        centers = ((responsibilities * m[None]).sum(2) / v) @ a @ covariance
        counts.mode_point_evaluations += len(centers)
        counts.mode_factor_calls += len(centers) * groups
    heights = log_target_numpy(p, centers)
    counts.mode_point_evaluations += len(centers)
    counts.mode_factor_calls += len(centers) * groups
    z = centers @ a.T
    logits = np.log(w)[None] - .5 * (z[:, :, None] - m[None]) ** 2 / v[None, :, None]
    responsibility = np.exp(logits - logsumexp(logits, 2, keepdims=True))
    gradient = -centers @ precision + ((responsibility * m[None]).sum(2) / v) @ a
    counts.mode_point_evaluations += len(centers)
    counts.mode_factor_calls += len(centers) * groups
    selected = []
    for index in np.argsort(-heights, kind="stable"):
        if selected:
            delta = centers[selected] - centers[index]
            if np.min(np.einsum("ni,ij,nj->n", delta, precision, delta)) <= 1e-8:
                continue
        selected.append(int(index))
        if len(selected) == (1 if kind == "gaussian" else 4):
            break
    weights = np.maximum(np.exp(heights[selected] - logsumexp(heights[selected])), 1e-6)
    weights /= weights.sum()
    return weights, centers[selected], covariance, dict(algorithm="fixed_latent_component_em", starts=16, steps=50,
        initial_seed=20261020, selected_indices=selected, selected_log_target=heights[selected].tolist(),
        selected_gradient_norms=np.linalg.norm(gradient[selected], axis=1).tolist(),
        all_gradient_norms=np.linalg.norm(gradient, axis=1).tolist(), merge_distance_squared=1e-8,
        minimum_pre_normalization_weight=1e-6)


class LearnedTarget:
    def __init__(self, payload, device="cpu", counts=None):
        p = parameters(payload)
        self.device = _device(device)
        self.counts = EvaluationCounts() if counts is None else counts
        self.a, self.v, self.m, self.w = [torch.tensor(p[name], dtype=torch.float64, device=self.device) for name in FIELDS]
        self.groups, self.dimension = self.a.shape

    def log_prob(self, points):
        if points.ndim != 2 or points.shape[1] != self.dimension or points.dtype != torch.float64 or points.device != self.device:
            raise ValueError("invalid learned target points")
        z = points @ self.a.T
        logs = self.w.log() - .5 * ((z[:, :, None] - self.m) ** 2 / self.v[:, None]
                                  + (2 * math.pi * self.v[:, None]).log())
        result = -.5 * (points.square().sum(1) + self.dimension * math.log(2 * math.pi))
        result += (torch.logsumexp(logs, 2) + .5 * (z.square() + math.log(2 * math.pi))).sum(1)
        self.counts.target_point_evaluations += len(points)
        self.counts.particle_factor_calls += len(points) * self.groups
        return result


@torch.no_grad()
def sample(payload, seed, *, reference="mixture", particles=4096, proposal_scale=.5,
           global_probability=0., direct_is=False, device="cpu"):
    if (isinstance(particles, bool) or not isinstance(particles, int) or particles < 2
            or not 0 < proposal_scale <= 1 or not 0 <= global_probability <= 1):
        raise ValueError("invalid particle or proposal setting")
    device = _device(device)
    _synchronize(device)
    started = time.perf_counter()
    p = parameters(payload)
    counts = EvaluationCounts()
    alpha, centers, covariance, search = construct_reference(p, reference, counts)
    q = GaussianReference(alpha, centers, covariance, device, counts)
    target = LearnedTarget(p, device, counts)
    motion, auxiliary, resampling = make_generators(seed, device)
    _synchronize(device)
    prepared = time.perf_counter()
    points = q.sample(particles, motion, auxiliary)
    residual = target.log_prob(points) - q.log_prob(points)
    logweights = torch.full((particles,), -math.log(particles), dtype=torch.float64, device=device)
    logz, beta, records = 0., 0., []
    for stage in range(128):
        _finite("learned residual", residual)
        next_beta = 1. if direct_is else _next_beta(residual, beta, .8)
        logweights, increment, ess = _reweight(logweights, residual, next_beta - beta)
        logz += float(increment)
        old_beta, beta = beta, next_beta
        if beta < 1:
            indices = _resample(logweights.exp(), resampling)
            points, residual = points[indices], residual[indices]
            logweights.fill_(-math.log(particles))
            counts.resampling_calls += 1
        accepted_count, global_count, sign_count = 0, 0, 0
        if not direct_is:
            for _ in range(3):
                proposal = q.propose(points, proposal_scale, motion, auxiliary)
                if global_probability:
                    independent = q.sample(particles, motion, auxiliary)
                    global_mask = torch.rand(particles, device=device, generator=auxiliary) < global_probability
                    global_count += int(global_mask.sum())
                    proposal = torch.where(global_mask[:, None], independent, proposal)
                candidate = target.log_prob(proposal) - q.log_prob(proposal)
                log_uniform = torch.rand(particles, dtype=torch.float64, device=device, generator=auxiliary).log()
                accepted = log_uniform < torch.minimum(beta * (candidate - residual), torch.zeros_like(residual))
                accepted_count += int(accepted.sum())
                points = torch.where(accepted[:, None], proposal, points)
                residual = torch.where(accepted, candidate, residual)
            points, residual, accepted, _ = sign_flip_step(target, q, points, residual, beta, auxiliary)
            sign_count = int(accepted.sum())
            counts.mh_proposals += 3 * particles
            counts.mh_accepted += accepted_count
            counts.sign_proposals += particles
            counts.sign_accepted += sign_count
            counts.sign_factor_calls += particles * target.groups
        records.append(dict(stage=stage, beta_previous=old_beta, beta=beta, ess_fraction=float(ess) / particles,
            resampled=beta < 1, log_normalizer_increment=float(increment), log_normalizer=logz,
            mh_accepted=accepted_count, global_proposals=global_count, sign_accepted=sign_count))
        if beta == 1:
            break
    if beta != 1:
        raise RuntimeError("128 stages exhausted without completing the bridge")
    x, w = points.cpu().numpy(), logweights.exp().cpu().numpy()
    _finite("learned output", x)
    _finite("learned output weights", w)
    if abs(w.sum() - 1) > 1e-10 or not np.isfinite(logz):
        raise FloatingPointError("invalid learned weights or normalizer")
    _synchronize(device)
    elapsed = time.perf_counter() - started
    return x, w, logz, dict(seconds=elapsed, preparation_seconds=prepared - started, counts=counts.report(),
        reference=reference, direct_is=direct_is, proposal_scale=proposal_scale, global_probability=global_probability,
        target="prior_corrected_product_of_frozen_learned_projection_posteriors", logZ_semantics="composition_normalizer",
        parameter_sha256=parameter_hash(p), mode_search=search, reference_parameters=dict(weights=alpha.tolist(),
        means=centers.tolist(), covariance=covariance.tolist()), records=records, stages=len(records), device=str(device),
        normalizer_unbiasedness_claim=False)
