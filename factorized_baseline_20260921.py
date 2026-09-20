import math
import time

import numpy as np
import torch

from certified_composition_torch import FactorParameters, TimeFactors


def product_coupling(values, generator):
    return torch.stack([values[torch.randperm(len(values), device=values.device, generator=generator), d]
                        for d in range(values.shape[1])], dim=1)


def factorized_sample(parameters, grid, particles, seed, device="cpu", ess_fraction=0.5):
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all() or not (np.diff(grid) < 0).all():
        raise ValueError("grid must be finite and strictly decreasing")
    if (grid < 0).any() or particles < 2 or not 0 <= ess_fraction <= 1:
        raise ValueError("invalid grid, particles, or ESS fraction")
    p = FactorParameters(parameters["variance"], parameters["means"], parameters["weights"])
    p = FactorParameters(p.variance.to(device), p.means.to(device), p.weights.to(device))
    motion = torch.Generator(device=device).manual_seed(seed)
    resampling = torch.Generator(device=device).manual_seed(seed + 2000006)
    x = torch.randn(particles, p.dimension, dtype=torch.float64, device=device, generator=motion)
    log_weights = torch.full_like(x, -math.log(particles))
    logz = torch.zeros(p.dimension, dtype=torch.float64, device=device)
    records = []
    for step, (u, next_u) in enumerate(zip(grid[:-1], grid[1:])):
        h = float(u - next_u)
        f = TimeFactors(p, float(u))
        residual = x[:, None] * f.a + f.b + f.residual(x)
        total = residual.sum(1)
        potential = (total.square() - residual.square().sum(1)) / 2
        noise = torch.randn(x.shape, dtype=x.dtype, device=device, generator=motion)
        x = (1 - h / 2) * x + h * total + math.sqrt(h) * noise
        log_weights += h * potential
        normalization = torch.logsumexp(log_weights, dim=0)
        logz += normalization
        log_weights -= normalization
        weights = log_weights.exp()
        ess = 1 / weights.square().sum(0)
        selected = ess < ess_fraction * particles
        if step == len(grid) - 2:
            selected = torch.ones_like(selected)
        if selected.any():
            positions = (torch.arange(particles, dtype=x.dtype, device=device)[None] +
                         torch.rand(p.dimension, 1, dtype=x.dtype, device=device, generator=resampling)) / particles
            cdf = weights.T.contiguous().cumsum(1)
            cdf[:, -1] = 1
            indices = torch.searchsorted(cdf, positions).T.contiguous()
            x = torch.where(selected[None], torch.gather(x, 0, indices), x)
            log_weights[:, selected] = -math.log(particles)
        if not torch.isfinite(x).all() or not torch.isfinite(log_weights).all():
            raise FloatingPointError(f"Nonfinite coordinate filter at step {step}")
        records.append(dict(step=step, minimum_coordinate_ess=float(ess.min()) / particles,
                            resampled_coordinates=int(selected.sum()), factor_calls=particles * p.groups))
    x = product_coupling(x, resampling)
    weights = np.full(particles, 1 / particles)
    return x.cpu().numpy(), weights, float(logz.sum()), records


def time_factorized(parameters, grid, particles, seed, device):
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = factorized_sample(parameters, grid, particles, seed, device)
    if device == "cuda":
        torch.cuda.synchronize()
    return result, time.perf_counter() - start
