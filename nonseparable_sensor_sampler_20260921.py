import math
import time

import numpy as np
from scipy.special import expit
import torch

from run_anchored_confirmation_20260921 import _generators, _resample


class SensorScoreBank:
    def __init__(self, problem, grid, device="cpu", prepare_method="all"):
        self.groups, self.dimension = problem.groups, problem.dimension
        self.device = torch.device(device)
        self.grid = np.asarray(grid, dtype=np.float64)
        if self.grid.ndim != 1 or len(self.grid) < 2 or not np.isfinite(self.grid).all() or not (np.diff(self.grid) < 0).all() or self.grid[-1] < 0:
            raise ValueError("grid must be strictly decreasing, finite, and nonnegative")
        banks = {name: [] for name in ["A", "base", "delta", "intercept", "b_mean", "b_anchor", "anchor"]}
        for u in self.grid[:-1]:
            precision, means, weights = problem.factor_parameters(float(u))
            A = precision - np.eye(self.dimension)
            v = np.einsum("gij,gcj->gci", precision, means)
            base, delta = v[:, 0], v[:, 1] - v[:, 0]
            intercept = np.log(weights[:, 1] / weights[:, 0]) - .5 * np.sum(means[:, 1] * v[:, 1] - means[:, 0] * v[:, 0], axis=1)
            b_mean = np.einsum("gc,gci->gi", weights, v)
            backbone = np.eye(self.dimension) + A.sum(0)
            anchor = np.linalg.solve(backbone, b_mean.sum(0))
            for _ in range(6 if prepare_method in ["all", "tail_anchored"] else 0):
                probability = expit(delta @ anchor + intercept)
                score = -backbone @ anchor + (base + probability[:, None] * delta).sum(0)
                curvature = backbone - np.einsum("g,gi,gj->ij", probability * (1 - probability), delta, delta)
                eigenvalues, eigenvectors = np.linalg.eigh(curvature)
                move = eigenvectors @ ((eigenvectors.T @ score) / np.maximum(eigenvalues, .2))
                anchor += move / max(1., np.linalg.norm(move))
            b_anchor = base + expit(delta @ anchor + intercept)[:, None] * delta
            for name, value in dict(A=A, base=base, delta=delta, intercept=intercept, b_mean=b_mean,
                                    b_anchor=b_anchor, anchor=anchor).items():
                banks[name].append(value)
        self.numpy_banks = {name: np.stack(values) for name, values in banks.items()}
        self.banks = {name: torch.as_tensor(value, dtype=torch.float64, device=self.device) for name, value in self.numpy_banks.items()}
        self.controls = {}
        for method, key in [("tail_fixed", "b_mean"), ("tail_anchored", "b_anchor")]:
            if prepare_method not in ["all", method]:
                continue
            A, b = self.banks["A"], self.banks[key]
            self.controls[method] = dict(b=b, total_A=A.sum(1), total_b=b.sum(1),
                A2=torch.einsum("kgji,kgjl->kil", A, A), Ab=torch.einsum("kgji,kgj->ki", A, b), b2=b.square().sum((1, 2)))

    def residuals(self, x, step, indices=None):
        A, base, delta, intercept = [self.banks[name][step] for name in ["A", "base", "delta", "intercept"]]
        if indices is None:
            probability = torch.sigmoid(x @ delta.T + intercept)
            return -torch.einsum("gij,nj->ngi", A, x) + base + probability[:, :, None] * delta
        selected_A, selected_delta = A[indices], delta[indices]
        probability = torch.sigmoid((x[:, None] * selected_delta).sum(-1) + intercept[indices])
        return -torch.einsum("nmij,nj->nmi", selected_A, x) + base[indices] + probability[:, :, None] * selected_delta

    def estimate(self, x, step, method, indices=None):
        G = self.groups
        if method == "full":
            r = self.residuals(x, step)
            total = r.sum(1)
            return total, .5 * (total.square().sum(-1) - r.square().sum((1, 2)))
        if indices is None or indices.ndim != 2 or indices.shape[0] != len(x) or not 2 <= indices.shape[1] <= G:
            raise ValueError("a valid without-replacement index array is required")
        M = indices.shape[1]
        if method == "unbiased":
            r = self.residuals(x, step, indices)
            total = r.sum(1)
            potential = G * (G - 1) * (total.square().sum(-1) - r.square().sum((1, 2))) / (2 * M * (M - 1))
            return G * total / M, potential
        if method not in self.controls:
            raise ValueError("unknown method")
        entry = {name: value[step] for name, value in self.controls[method].items()}
        A = self.banks["A"][step]
        r0 = -torch.einsum("nmij,nj->nmi", A[indices], x) + entry["b"][indices]
        delta = self.banks["delta"][step][indices]
        probability = torch.sigmoid((x[:, None] * delta).sum(-1) + self.banks["intercept"][step][indices])
        e = self.banks["base"][step][indices] + probability[:, :, None] * delta - entry["b"][indices]
        R0 = -x @ entry["total_A"].T + entry["total_b"]
        q0 = torch.einsum("ni,ij,nj->n", x, entry["A2"], x) - 2 * (x * entry["Ab"]).sum(-1) + entry["b2"]
        E = e.sum(1)
        potential = .5 * (R0.square().sum(-1) - q0)
        potential += G / M * ((R0 * E).sum(-1) - (r0 * e).sum((1, 2)))
        potential += G * (G - 1) / (2 * M * (M - 1)) * (E.square().sum(-1) - e.square().sum((1, 2)))
        return R0 + G * E / M, potential


def diffusion_sample(problem, grid, particles, seed, method, batch=4, device="cpu", ess_fraction=.5):
    if particles < 2 or not 0 <= ess_fraction <= 1 or not 2 <= batch <= problem.groups:
        raise ValueError("invalid sampling configuration")
    bank = SensorScoreBank(problem, grid, device, prepare_method=method)
    motion, auxiliary, resampling = _generators(seed, device)
    x = torch.randn((particles, problem.dimension), device=device, dtype=torch.float64, generator=motion)
    log_weights = torch.full((particles,), -math.log(particles), device=device, dtype=torch.float64)
    logz = 0.
    records = []
    for step, (u, next_u) in enumerate(zip(grid[:-1], grid[1:])):
        h = float(u - next_u)
        indices = None
        if method != "full":
            indices = torch.rand((particles, problem.groups), device=device, dtype=torch.float64, generator=auxiliary).topk(batch, dim=1).indices
        R, potential = bank.estimate(x, step, method, indices)
        y = (1 - h / 2) * x + h * R + math.sqrt(h) * torch.randn(x.shape, device=device, dtype=x.dtype, generator=motion)
        log_weights += h * potential
        increment = torch.logsumexp(log_weights, 0)
        logz += float(increment)
        log_weights -= increment
        weights = log_weights.exp()
        if not torch.isfinite(y).all() or not torch.isfinite(log_weights).all():
            raise FloatingPointError(f"nonfinite diffusion output at step {step}")
        ess = 1 / weights.square().sum()
        resampled = bool(ess < particles * ess_fraction and step < len(grid) - 2)
        x = y
        if resampled:
            x = x[_resample(weights, resampling)]
            log_weights.fill_(-math.log(particles))
        records.append(dict(step=step, ess_fraction=float(ess / particles), resampled=resampled,
                            factor_calls=particles * (problem.groups if method == "full" else batch)))
    return x.cpu().numpy(), weights.cpu().numpy(), logz, records, bank.numpy_banks["A"]


def sensor_log_likelihood(x, directions, noise_std, probabilities, observations):
    projection = x @ directions.T
    positive = -.5 * ((observations - projection) / noise_std).square() + probabilities.log()
    negative = -.5 * ((observations + projection) / noise_std).square() + torch.log1p(-probabilities)
    return (torch.logaddexp(positive, negative) - noise_std.log() - .5 * math.log(2 * math.pi)).sum(-1)


def annealed_smc(problem, particles, seed, stages=64, proposal_scale=.3, moves=3, device="cpu"):
    if particles < 2 or stages < 2 or moves < 1 or not 0 < proposal_scale < 1:
        raise ValueError("invalid annealed SMC configuration")
    motion, auxiliary, resampling = _generators(seed, device)
    values = [torch.as_tensor(value, device=device, dtype=torch.float64) for value in
              [problem.directions, problem.noise_std, problem.positive_probability, problem.observations]]
    x = torch.randn((particles, problem.dimension), device=device, dtype=torch.float64, generator=motion)
    likelihood = sensor_log_likelihood(x, *values)
    logz, records = 0., []
    for step in range(stages):
        beta = (step + 1) / stages
        logw = likelihood / stages
        increment = torch.logsumexp(logw, 0) - math.log(particles)
        logz += float(increment)
        weights = torch.softmax(logw, 0)
        ess = 1 / weights.square().sum()
        indices = _resample(weights, resampling)
        x, likelihood = x[indices], likelihood[indices]
        accepted = []
        for _ in range(moves):
            proposal = math.sqrt(1 - proposal_scale ** 2) * x + proposal_scale * torch.randn(x.shape, device=device, dtype=x.dtype, generator=motion)
            proposal_likelihood = sensor_log_likelihood(proposal, *values)
            accept = torch.rand((particles,), device=device, dtype=x.dtype, generator=auxiliary).log() < beta * (proposal_likelihood - likelihood)
            x = torch.where(accept[:, None], proposal, x)
            likelihood = torch.where(accept, proposal_likelihood, likelihood)
            accepted.append(float(accept.double().mean()))
        # 全局符号翻转是对称提议，接受率仍然使用目标密度。
        flipped_likelihood = sensor_log_likelihood(-x, *values)
        accept = torch.rand((particles,), device=device, dtype=x.dtype, generator=auxiliary).log() < beta * (flipped_likelihood - likelihood)
        x = torch.where(accept[:, None], -x, x)
        likelihood = torch.where(accept, flipped_likelihood, likelihood)
        if not torch.isfinite(x).all() or not torch.isfinite(likelihood).all():
            raise FloatingPointError("nonfinite annealed SMC state")
        records.append(dict(step=step, beta=beta, ess_fraction=float(ess / particles),
                            acceptance=float(np.mean(accepted)), sign_acceptance=float(accept.double().mean()),
                            factor_calls=particles * problem.groups * (moves + 1)))
    return x.cpu().numpy(), np.full(particles, 1 / particles), logz, records


def timed_sample(function, *args, device="cpu", **kwargs):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = function(*args, device=device, **kwargs)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    return result, time.perf_counter() - start
