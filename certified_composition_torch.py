import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class FactorParameters:
    variance: object
    means: object
    weights: object

    def __post_init__(self):
        self.variance = torch.as_tensor(self.variance, dtype=torch.float64)
        self.means = torch.as_tensor(self.means, dtype=torch.float64, device=self.variance.device)
        self.weights = torch.as_tensor(self.weights, dtype=torch.float64, device=self.variance.device)
        if self.variance.ndim != 2 or self.means.shape != (*self.variance.shape, 2) or self.weights.shape != self.means.shape:
            raise ValueError("Expected diagonal two-component mixture factors")
        if not all(torch.isfinite(a).all() for a in (self.variance, self.means, self.weights)):
            raise ValueError("All parameters must be finite")
        if not ((self.variance > 0) & (self.variance <= 1)).all():
            raise ValueError("The certified tail class requires variance in (0, 1]")
        if not (self.weights > 0).all() or not torch.allclose(self.weights.sum(-1), torch.ones_like(self.weights[..., 0])):
            raise ValueError("Mixture weights must be positive and normalized")

    @property
    def groups(self):
        return self.variance.shape[0]

    @property
    def dimension(self):
        return self.variance.shape[1]


class TimeFactors:
    def __init__(self, parameters, u, table_nodes=0, table_radius=8.0):
        self.parameters = parameters
        self.device = parameters.variance.device
        self.alpha = math.exp(-float(u) / 2)
        v = 1 - self.alpha**2 + self.alpha**2 * parameters.variance
        mean = (parameters.weights * parameters.means).sum(-1)
        self.v = v
        self.a = 1 - 1 / v
        self.b = self.alpha * mean / v
        self.total_a = self.a.sum(0)
        self.total_b = self.b.sum(0)
        self.a2 = (self.a**2).sum(0)
        self.ab = (self.a * self.b).sum(0)
        self.b2 = (self.b**2).sum(0)
        self.mean = mean
        self.bound = self.alpha * (parameters.means - mean[..., None]).abs().amax(-1) / v
        delta = parameters.means[..., 1] - parameters.means[..., 0]
        self.logit_slope = self.alpha * delta / v
        self.logit_intercept = torch.log(parameters.weights[..., 1] / parameters.weights[..., 0]) - self.alpha**2 * (parameters.means[..., 1]**2 - parameters.means[..., 0]**2) / (2 * v)
        self.e0 = self.alpha * (parameters.means[..., 0] - mean) / v
        self.edelta = self.alpha * delta / v
        self.table = None
        self.preparation_calls = 0
        self.surrogate_bound = torch.zeros_like(self.bound)
        if table_nodes:
            if table_nodes < 3 or table_radius <= 0:
                raise ValueError("Interpolation requires at least three nodes and a positive radius")
            self.table_grid = torch.linspace(-table_radius, table_radius, table_nodes, dtype=torch.float64, device=self.device)
            self.table_spacing = self.table_grid[1] - self.table_grid[0]
            p = torch.sigmoid(self.logit_slope[..., None] * self.table_grid + self.logit_intercept[..., None])
            self.table = self.e0[..., None] + self.edelta[..., None] * p
            self.preparation_calls = parameters.groups * table_nodes
            self.surrogate_bound = self.table.abs().amax(-1)
            interior = self.edelta.abs() * self.logit_slope**2 * self.table_spacing**2 / (48 * math.sqrt(3))
            left = self.edelta.abs() * torch.where(self.logit_slope >= 0, p[..., 0], 1 - p[..., 0])
            right = self.edelta.abs() * torch.where(self.logit_slope >= 0, 1 - p[..., -1], p[..., -1])
            self.global_bound = torch.stack((interior, left, right)).amax(0)
            self.bound = interior
            self.table_sum = self.table.sum(0)
            self.table_a = (self.a[..., None] * self.table).sum(0)
            self.table_b = (self.b[..., None] * self.table).sum(0)
            self.table_square = (self.table**2).sum(0)
            self.table_cross = (self.table[..., :-1] * self.table[..., 1:]).sum(0)
        self.basis = torch.cat((self.bound, self.a.abs() * self.bound, (((self.b.abs() + self.surrogate_bound) * self.bound + self.bound**2 / 2).sum(-1, keepdim=True))), 1)
        self.basis_sum = self.basis.sum(0)
        self.basis_cdf = torch.zeros((self.basis.shape[1], self.basis.shape[0]), dtype=torch.float64, device=self.device)
        nz = self.basis_sum > 0
        self.basis_cdf[nz] = torch.cumsum((self.basis[:, nz] / self.basis_sum[nz]).T, 1)
        self.basis_cdf[nz, -1] = 1

    def affine(self, x):
        total = x * self.total_a + self.total_b
        q0 = x**2 * self.a2 + 2 * x * self.ab + self.b2
        return total, (total.square().sum(-1) - q0.sum(-1)) / 2

    def table_locations(self, x):
        position = ((x - self.table_grid[0]) / self.table_spacing).clamp(0, len(self.table_grid) - 1)
        left = position.to(torch.int64).clamp_max(len(self.table_grid) - 2)
        return left, position - left

    def tabulated_residual(self, x, indices=None):
        if self.table is None:
            return torch.zeros((len(x), self.parameters.dimension), dtype=torch.float64, device=self.device) if indices is not None else torch.zeros((len(x), 1, self.parameters.dimension), dtype=torch.float64, device=self.device)
        left, fraction = self.table_locations(x)
        if indices is not None:
            dims = torch.arange(self.parameters.dimension, device=self.device)
            return self.table[indices[:, None], dims, left] * (1 - fraction) + self.table[indices[:, None], dims, left + 1] * fraction
        dims = torch.arange(self.parameters.dimension, device=self.device)[None, None]
        groups = torch.arange(self.parameters.groups, device=self.device)[None, :, None]
        return self.table[groups, dims, left[:, None]] * (1 - fraction[:, None]) + self.table[groups, dims, left[:, None] + 1] * fraction[:, None]

    def residual(self, x, indices=None):
        if indices is None:
            return self.e0 + self.edelta * torch.sigmoid(x[:, None] * self.logit_slope + self.logit_intercept)
        return self.e0[indices] + self.edelta[indices] * torch.sigmoid(x * self.logit_slope[indices] + self.logit_intercept[indices])

    def proposal(self, x):
        if self.table is None:
            return self.affine(x)
        left, f = self.table_locations(x)
        dims = torch.arange(self.parameters.dimension, device=self.device)[None]
        interp = lambda z: z[dims, left] * (1 - f) + z[dims, left + 1] * f
        total = x * self.total_a + self.total_b + interp(self.table_sum)
        q0 = x**2 * self.a2 + 2 * x * self.ab + self.b2 + 2 * x * interp(self.table_a) + 2 * interp(self.table_b)
        q0 = q0 + (1 - f)**2 * self.table_square[dims, left] + f**2 * self.table_square[dims, left + 1] + 2 * f * (1 - f) * self.table_cross[dims, left]
        return total, (total.square().sum(-1) - q0.sum(-1)) / 2

    def full(self, x):
        r = x[:, None] * self.a + self.b + self.residual(x)
        total = r.sum(1)
        return total, (total.square().sum(-1) - r.square().sum((1, 2))) / 2

    def correction_terms(self, x, y, h):
        aa = y - (1 - h / 2) * x
        e = self.residual(x) - self.tabulated_residual(x)
        r0 = x[:, None] * self.a + self.b + self.tabulated_residual(x)
        return ((aa[:, None] - h * r0) * e - h * e.square() / 2).sum(-1)

    def random_correction(self, x, y, h, eta, generator, cost_fraction=0.25):
        if eta <= 0 or not 0 < cost_fraction <= 1 or h <= 0:
            raise ValueError("Positive time and variance budget, cost fraction in (0,1] required")
        aa = y - (1 - h / 2) * x
        coefficients = torch.cat((aa.abs(), h * x.abs(), torch.full((len(x), 1), h, dtype=torch.float64, device=self.device)), 1)
        envelope = coefficients @ self.basis_sum
        rate = torch.maximum(1.1 * envelope, envelope.square() / eta)
        full = rate >= cost_fraction * self.parameters.groups
        if self.table is not None:
            full |= ((x < self.table_grid[0]) | (x > self.table_grid[-1])).any(1)
        randomized = (~full) & (envelope > 0)
        result = torch.zeros(len(x), dtype=torch.float64, device=self.device)
        if full.any(): result[full] = self.correction_terms(x[full], y[full], h).sum(-1)
        rows = torch.nonzero(randomized).flatten()
        events = 0
        minimum_factor = 1.0
        if len(rows):
            counts = torch.poisson(rate[rows], generator=generator)
            parents = torch.repeat_interleave(rows, counts.to(torch.int64))
            events = len(parents)
            if events:
                probs = coefficients[parents] * self.basis_sum / envelope[parents, None]
                component_cdf = torch.cumsum(probs, 1)
                component_cdf[:, -1] = 1
                comps = (torch.rand((events, 1), dtype=torch.float64, device=self.device, generator=generator) >= component_cdf).sum(1)
                uniforms = torch.rand(events, dtype=torch.float64, device=self.device, generator=generator)
                indices = torch.searchsorted(self.basis_cdf[comps], uniforms[:, None]).flatten().clamp_max(self.parameters.groups - 1)
                points = x[parents]; e = self.residual(points, indices) - self.tabulated_residual(points, indices)
                r0 = points * self.a[indices] + self.b[indices] + self.tabulated_residual(points, indices)
                zz = ((aa[parents] - h * r0) * e - h * e.square() / 2).sum(1)
                bb = (coefficients[parents] * self.basis[indices]).sum(1)
                factor = 1 + zz / (rate[parents] * (bb / envelope[parents]))
                minimum_factor = float(factor.min())
                if minimum_factor <= 0 or not torch.isfinite(factor).all(): raise FloatingPointError("Positive Poisson factor contract violated")
                result.scatter_add_(0, parents, factor.log())
        inflation = torch.zeros_like(envelope); inflation[randomized] = envelope[randomized].square() / rate[randomized]
        if float(inflation.max()) > eta * (1 + 1e-12): raise ArithmeticError("Conditional second-moment budget violated")
        return result, {"factor_calls": int(full.sum()) * self.parameters.groups + events, "randomized_particles": int(randomized.sum()), "full_particles": int(full.sum()), "zero_residual_particles": int(((envelope == 0) & ~full).sum()), "poisson_events": events, "maximum_log_relative_second_moment_bound": float(inflation.max()), "minimum_poisson_factor": minimum_factor}


def moment_certificate(parameters, grid, power=1.0, margin=0.0):
    variance = torch.ones(parameters.dimension, dtype=torch.float64, device=parameters.variance.device)
    smallest = 1.0
    for index, (u, nxt) in enumerate(zip(grid[:-1], grid[1:])):
        h = float(u - nxt); alpha2 = math.exp(-float(u)); strengths = 1 / (1 - alpha2 + alpha2 * parameters.variance) - 1
        total = strengths.sum(0); curvature = (total.square() - strengths.square().sum(0)) / 2
        denominator = 1 - 2 * power * h * curvature * variance; smallest = min(smallest, float(denominator.min()))
        if float(denominator.min()) <= margin: return {"passed": False, "failure_step": index, "minimum_denominator": smallest, "power": power}
        variance = (1 - h * (total + 0.5)).square() * variance / denominator + h
    return {"passed": True, "minimum_denominator": smallest, "power": power, "terminal_tail_variance": variance.tolist()}


def prepare_grid(parameters, requested_steps=512, u_max=20.0, margin=0.02, maximum_steps=32768):
    steps = requested_steps
    while steps <= maximum_steps:
        grid = np.linspace(math.sqrt(u_max), 0, steps + 1)**2
        existence = moment_certificate(parameters, grid, 1, margin)
        if existence["passed"]:
            return grid, {"requested_steps": requested_steps, "actual_steps": steps, "existence": existence, "path_moment": moment_certificate(parameters, grid, 1, margin), "scope": "Gaussian quadratic-tail reference for fixed-grid importance paths; affine plus bounded proposal drift"}
        steps *= 2
    raise ArithmeticError("No certified full-factor grid within maximum_steps")


def sample(parameters, grid, particles, seed, method="certified", eta_total=4.0, cost_fraction=0.25, batch=4, ess_fraction=0.5, table_nodes=129, table_radius=8.0, device=None, return_type="numpy"):
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all() or not (np.diff(grid) < 0).all():
        raise ValueError("A finite strictly decreasing grid is required")
    if isinstance(parameters, dict):
        parameters = FactorParameters(parameters["variance"], parameters["means"], parameters["weights"])
    if device is not None:
        parameters = FactorParameters(parameters.variance.to(device), parameters.means.to(device), parameters.weights.to(device))
    if method not in ("certified", "affine_certified", "surrogate_full", "surrogate_only", "full", "affine_full", "tail_fixed"): raise ValueError(method)
    if particles < 2 or eta_total <= 0 or not 0 <= ess_fraction <= 1: raise ValueError("Invalid sampling controls")
    seeds = [int(seed) + 1000003 * i for i in range(3)]
    motion = torch.Generator(device=parameters.variance.device).manual_seed(seeds[0]); auxiliary = torch.Generator(device=parameters.variance.device).manual_seed(seeds[1]); resampling = torch.Generator(device=parameters.variance.device).manual_seed(seeds[2])
    x = torch.randn((particles, parameters.dimension), dtype=torch.float64, device=parameters.variance.device, generator=motion)
    log_weights = torch.full((particles,), -math.log(particles), dtype=torch.float64, device=x.device); log_normalizer = 0.0; records = []
    for step, (u, nxt) in enumerate(zip(grid[:-1], grid[1:])):
        h = float(u - nxt)
        table_count = table_nodes if method in ("certified", "surrogate_full", "surrogate_only") else 0
        f = TimeFactors(parameters, float(u), table_count, table_radius)
        noise = math.sqrt(h) * torch.randn(x.shape, dtype=torch.float64, device=x.device, generator=motion)
        stats = {"factor_calls": 0, "randomized_particles": 0, "full_particles": 0, "zero_residual_particles": 0, "poisson_events": 0, "maximum_log_relative_second_moment_bound": 0.0, "minimum_poisson_factor": 1.0}
        if method in ("certified", "affine_certified", "affine_full", "surrogate_full", "surrogate_only"):
            total, potential = f.proposal(x)
            y = (1 - h / 2) * x + h * total + noise
            if method in ("certified", "affine_certified"): correction, stats = f.random_correction(x, y, h, eta_total / (len(grid) - 1), auxiliary, cost_fraction)
            elif method == "surrogate_only": correction = torch.zeros(particles, dtype=torch.float64, device=x.device)
            else:
                correction = f.correction_terms(x, y, h).sum(-1)
                stats["factor_calls"] = particles * parameters.groups
                stats["full_particles"] = particles
            increment = h * potential + correction
        else:
            if method == "full":
                total, potential = f.full(x); stats["factor_calls"] = particles * parameters.groups
            else:
                if not 2 <= batch <= parameters.groups: raise ValueError("Invalid without-replacement batch size")
                indices = torch.rand((particles, parameters.groups), dtype=torch.float64, device=x.device, generator=auxiliary).topk(batch, dim=1).indices
                points = x[:, None, :].expand(-1, batch, -1); gathered_a = f.a[indices]; gathered_b = f.b[indices]
                e = f.residual(points.reshape(-1, parameters.dimension), indices.reshape(-1)).reshape(particles, batch, parameters.dimension)
                r0 = points * gathered_a + gathered_b; total0, potential0 = f.affine(x); etotal = e.sum(1)
                linear = parameters.groups * (total0 * e.mean(1)).sum(-1) - parameters.groups * (r0 * e).sum((1, 2)) / batch
                pair = (etotal.square().sum(-1) - e.square().sum((1, 2))) * parameters.groups * (parameters.groups - 1) / (2 * batch * (batch - 1))
                total = total0 + parameters.groups * etotal / batch
                potential = potential0 + linear + pair
                stats["factor_calls"] = particles * batch
            y = (1 - h / 2) * x + h * total + noise
            increment = h * potential
        if not torch.isfinite(y).all() or not torch.isfinite(increment).all(): raise FloatingPointError(f"Nonfinite update at step {step}")
        stats["preparation_calls"] = f.preparation_calls
        stats["factor_calls"] += f.preparation_calls
        log_weights += increment
        normalization = torch.logsumexp(log_weights, 0)
        log_normalizer += float(normalization)
        log_weights -= normalization
        weights = log_weights.exp()
        ess = 1 / weights.square().sum()
        should = bool(ess < particles * ess_fraction and step < len(grid) - 2)
        x = y
        if should:
            points = (torch.arange(particles, dtype=torch.float64, device=x.device) + torch.rand((), dtype=torch.float64, device=x.device, generator=resampling)) / particles
            cdf = weights.cumsum(0)
            cdf[-1] = 1
            x = x[torch.searchsorted(cdf, points)]
            log_weights.fill_(-math.log(particles))
        records.append(dict(stats, step=step, u=float(u), h=h, ess_fraction=float(ess / particles), resampled=should))
    if return_type == "torch": return x, weights, log_normalizer, records
    if return_type != "numpy": raise ValueError(return_type)
    return x.cpu().numpy(), weights.cpu().numpy(), log_normalizer, records
