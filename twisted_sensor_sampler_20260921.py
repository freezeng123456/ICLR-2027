import math

import numpy as np
from scipy.special import expit
import torch

from gaussian_twisted_path_20260921 import affine_factor_backbone, build_gaussian_twist
from nonseparable_sensor_sampler_20260921 import SensorScoreBank
from run_anchored_confirmation_20260921 import _generators


def sensor_references(bank, kind):
    A = bank.numpy_banks["A"]
    if kind == "mean":
        offsets = [bank.numpy_banks["b_mean"]]
    elif kind == "anchor":
        offsets = [bank.numpy_banks["b_anchor"]]
    elif kind == "mirror":
        base, delta, intercept, anchors = [bank.numpy_banks[name] for name in ["base", "delta", "intercept", "anchor"]]
        negative = base + expit(-np.einsum("kgd,kd->kg", delta, anchors) + intercept)[:, :, None] * delta
        offsets = [bank.numpy_banks["b_anchor"], negative]
    else:
        raise ValueError("unknown Gaussian reference")
    backbones = [affine_factor_backbone(A, b, bank.grid) for b in offsets]
    references = [build_gaussian_twist(np.zeros(bank.dimension), np.eye(bank.dimension), **args) for args in backbones]
    for other in references[1:]:
        np.testing.assert_allclose(other.matrices, references[0].matrices, atol=1e-12)
        np.testing.assert_allclose(other.covariances, references[0].covariances, atol=1e-12)
        np.testing.assert_allclose(other.initial_covariance, references[0].initial_covariance, atol=1e-12)
    return references, backbones, offsets


def twisted_sensor_sample(problem, grid, particles, seed, method="full", batch=4, reference="mirror",
                          device="cpu", return_paths=False):
    if particles < 2 or method not in ["full", "tail_anchored"] or not 2 <= batch <= problem.groups:
        raise ValueError("invalid twisted sampling configuration")
    bank = SensorScoreBank(problem, grid, device, prepare_method="all")
    references, backbones, offsets = sensor_references(bank, reference)
    motion, auxiliary, _ = _generators(seed, device)
    dtype = torch.float64
    tensor = lambda a: torch.as_tensor(a, dtype=dtype, device=device)
    initial_means = tensor(np.stack([r.initial_mean for r in references]))
    shifts = tensor(np.stack([r.shifts for r in references]))
    matrices = tensor(references[0].matrices)
    initial_covariance = references[0].initial_covariance
    initial_inverse = tensor(np.linalg.inv(initial_covariance))
    noise_cholesky = tensor(np.linalg.cholesky(references[0].covariances))
    noise_inverse = tensor(np.linalg.inv(references[0].covariances))
    component = torch.randint(len(references), (particles,), device=device, generator=auxiliary)
    x = initial_means[component] + torch.randn((particles, problem.dimension), dtype=dtype, device=device, generator=motion) @ tensor(np.linalg.cholesky(initial_covariance)).T
    differences = initial_means - initial_means[0]
    log_ratios = (x - initial_means[0]) @ initial_inverse @ differences.T - .5 * torch.einsum("jd,dl,jl->j", differences, initial_inverse, differences)
    logw = torch.full((particles,), references[0].log_normalizer, dtype=dtype, device=device)
    original_L, original_d, C, ell, constant = [tensor(backbones[0][key]) for key in ["matrices", "shifts", "quadratic", "linear", "constant"]]
    total_A, total_b = tensor(bank.numpy_banks["A"].sum(1)), tensor(offsets[0].sum(1))
    paths, batches = ([x.cpu().numpy().copy()] if return_paths else []), []
    for k, h in enumerate(-np.diff(bank.grid)):
        indices = None
        if method != "full":
            indices = torch.rand((particles, problem.groups), dtype=dtype, device=device, generator=auxiliary).topk(batch, dim=1).indices
        R, potential = bank.estimate(x, k, method, indices)
        base_mean = x @ matrices[k].T + shifts[0, k]
        y = base_mean + (shifts[:, k][component] - shifts[0, k]) + torch.randn(x.shape, dtype=dtype, device=device, generator=motion) @ noise_cholesky[k].T
        difference = shifts[:, k] - shifts[0, k]
        log_ratios += (y - base_mean) @ noise_inverse[k] @ difference.T - .5 * torch.einsum("jd,dl,jl->j", difference, noise_inverse[k], difference)
        residual_difference = R - (-x @ total_A[k].T + total_b[k])
        original_mean = x @ original_L[k].T + original_d[k]
        gaussian_potential = torch.einsum("ni,ij,nj->n", x, C[k], x) + x @ ell[k] + constant[k]
        logw += h * potential - gaussian_potential + (residual_difference * (y - original_mean)).sum(-1) - .5 * h * residual_difference.square().sum(-1)
        x = y
        if return_paths:
            paths.append(x.cpu().numpy().copy())
            batches.append(None if indices is None else indices.cpu().numpy().copy())
    # 每条路径使用完整混合提议密度，选中分量只用于抽样。
    logw -= torch.logsumexp(log_ratios, dim=1) - math.log(len(references))
    if not torch.isfinite(x).all() or not torch.isfinite(logw).all():
        raise FloatingPointError("nonfinite twisted path output")
    weights = torch.softmax(logw, dim=0)
    return dict(samples=x.cpu().numpy(), weights=weights.cpu().numpy(), log_weights=logw.cpu().numpy(),
                log_normalizer=float(torch.logsumexp(logw, 0) - math.log(particles)),
                ess_fraction=float(1 / weights.square().sum() / particles),
                reference_log_normalizers=[r.log_normalizer for r in references],
                paths=np.stack(paths, axis=1) if return_paths else None, batches=batches if return_paths else None)
