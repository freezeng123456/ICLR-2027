import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import socket
import time

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import logsumexp
import torch


DTYPE = torch.float64


class FactorModel:
    def __init__(self, groups, dimension, family, device):
        self.groups = groups
        self.dimension = dimension
        self.family = family
        self.device = torch.device(device)
        g = torch.arange(groups, device=device, dtype=DTYPE)[:, None]
        d = torch.arange(dimension, device=device, dtype=DTYPE)[None, :]
        phase = 2 * math.pi * (g + 0.37 * d) / groups
        if family == "gaussian":
            self.variance = 0.35 + 0.15 * (1 + torch.sin(phase))
            center = 0.35 + 0.3 * torch.cos(phase)
            self.means = torch.stack((center, center), dim=-1)
        elif family == "mixture":
            self.variance = 0.35 + 0.15 * (1 + torch.sin(phase))
            offset = 0.1 * torch.cos(phase)
            separation = 0.7 + 0.2 * torch.cos(phase * 2)
            self.means = torch.stack((offset - separation, offset + separation), dim=-1)
        elif family == "weak_mixture":
            self.variance = 1 / (1 + (4 + 2 * torch.sin(phase)) / groups)
            offset = 0.4 / groups * torch.cos(phase)
            separation = (0.6 + 0.1 * torch.cos(phase)) / math.sqrt(groups)
            self.means = torch.stack((offset - separation, offset + separation), dim=-1)
        else:
            raise ValueError(family)
        positive_weight = 0.5 + 0.1 * torch.sin(phase + 0.7)
        self.weights = torch.stack((1 - positive_weight, positive_weight), dim=-1)
        self.mean = (self.weights * self.means).sum(-1)
        self.marginal_variance = self.variance + (
            self.weights * (self.means - self.mean[..., None]).square()
        ).sum(-1)
        self.control_variance = self.marginal_variance

    def coefficients(self, u):
        alpha = math.exp(-0.5 * u)
        v = 1 - alpha * alpha + alpha * alpha * self.variance
        cv_v = 1 - alpha * alpha + alpha * alpha * self.control_variance
        a = 1 - 1 / cv_v
        b = alpha * self.mean / cv_v
        return alpha, v, a, b

    def residuals(self, x, u, indices=None):
        alpha, v, a, b = self.coefficients(u)
        if indices is None:
            means = self.means[None]
            weights = self.weights[None]
            v = v[None]
            a = a[None]
            b = b[None]
        else:
            means = self.means[indices]
            weights = self.weights[indices]
            v = v[indices]
            a = a[indices]
            b = b[indices]
        xx = x[:, None, :, None]
        difference = xx - alpha * means
        log_component = weights.log() - 0.5 * difference.square() / v[..., None]
        responsibilities = torch.softmax(log_component, dim=-1)
        score = -(responsibilities * difference).sum(-1) / v
        residual = score + x[:, None, :]
        control = a * x[:, None, :] + b
        return residual, control

    def exact(self, x, u):
        r, _ = self.residuals(x, u)
        total = r.sum(1)
        potential = 0.5 * (total.square().sum(-1) - r.square().sum((1, 2)))
        return 0.5 * total, potential

    def estimate(self, x, u, batch, generator, control=False, naive=False, indices=None):
        if batch < 2:
            raise ValueError("batch must be at least two")
        if indices is None:
            indices = torch.randint(self.groups, (len(x), batch), device=x.device, generator=generator)
        r, r0 = self.residuals(x, u, indices)
        if control:
            _, _, a, b = self.coefficients(u)
            r0_total = a.sum(0) * x + b.sum(0)
            q0 = (a.square().sum(0) * x.square() + 2 * (a * b).sum(0) * x + b.square().sum(0)).sum(-1)
            e = r - r0
            constant = 0.5 * (r0_total.square().sum(-1) - q0)
            linear = (r0_total * (self.groups * e.mean(1))).sum(-1)
            linear -= self.groups * (r0 * e).sum(-1).mean(1)
        else:
            e = r
            r0_total = torch.zeros_like(x)
            constant = torch.zeros(len(x), dtype=x.dtype, device=x.device)
            linear = constant
        e_sum = e.sum(1)
        e_squares = e.square().sum((1, 2))
        total = r0_total + self.groups * e_sum / batch
        if naive:
            norm_estimate = (self.groups * e_sum / batch).square().sum(-1)
        else:
            norm_estimate = self.groups ** 2 * (e_sum.square().sum(-1) - e_squares) / (batch * (batch - 1))
        potential = constant + linear + 0.5 * (norm_estimate - self.groups * e_squares / batch)
        return 0.5 * total, potential

    def reference(self, points=65537, bound=12.0):
        grid = np.linspace(-bound, bound, points)
        means = self.means.detach().cpu().numpy()
        weights = self.weights.detach().cpu().numpy()
        variance = self.variance.detach().cpu().numpy()
        densities = []
        cdfs = []
        for d in range(self.dimension):
            diff = grid[:, None, None] - means[None, :, d, :]
            log_components = np.log(weights[None, :, d, :]) - 0.5 * (
                diff ** 2 / variance[None, :, d, None] + np.log(2 * math.pi * variance[None, :, d, None])
            )
            log_target = logsumexp(log_components, axis=-1).sum(-1)
            log_target += (self.groups - 1) * 0.5 * (grid ** 2 + math.log(2 * math.pi))
            density = np.exp(log_target - log_target.max())
            density /= trapezoid(density, grid)
            cdf = cumulative_trapezoid(density, grid, initial=0)
            cdf /= cdf[-1]
            if max(density[0], density[-1]) > 1e-12:
                raise RuntimeError("reference domain is insufficient")
            densities.append(density)
            cdfs.append(cdf)
        return {"grid": grid, "density": np.array(densities), "cdf": np.array(cdfs)}


def systematic_resample(weights, generator):
    n = len(weights)
    positions = (torch.arange(n, device=weights.device, dtype=weights.dtype) + torch.rand((), device=weights.device, generator=generator, dtype=weights.dtype)) / n
    cumulative = weights.cumsum(0)
    cumulative[-1] = 1
    return torch.searchsorted(cumulative, positions).clamp_max(n - 1)


def measure(samples, weights, reference):
    grid = reference["grid"]
    means = []
    variances = []
    w1 = []
    ks = []
    sign_error = []
    for d in range(samples.shape[1]):
        density = reference["density"][d]
        cdf = reference["cdf"][d]
        target_mean = trapezoid(grid * density, grid)
        target_variance = trapezoid((grid - target_mean) ** 2 * density, grid)
        mean = np.sum(weights * samples[:, d])
        variance = np.sum(weights * (samples[:, d] - mean) ** 2)
        order = np.argsort(samples[:, d])
        sorted_samples = samples[order, d]
        cumulative = np.cumsum(weights[order])
        locations = np.searchsorted(sorted_samples, grid, side="right")
        empirical_cdf = np.concatenate(([0], cumulative))[locations]
        w1.append(trapezoid(abs(empirical_cdf - cdf), grid))
        target_at_sample = np.interp(sorted_samples, grid, cdf)
        ks.append(max(np.max(abs(cumulative - target_at_sample)), np.max(abs(cumulative - weights[order] - target_at_sample))))
        means.append(abs(mean - target_mean))
        variances.append(abs(variance / target_variance - 1))
        sign_error.append(abs(weights[samples[:, d] > 0].sum() - (1 - np.interp(0, grid, cdf))))
    return {"w1_mean": float(np.mean(w1)), "ks_mean": float(np.mean(ks)), "mean_absolute_error": float(np.mean(means)), "variance_relative_error": float(np.mean(variances)), "positive_mass_error": float(np.mean(sign_error))}


def run_cell(config, output, model=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    generator = torch.Generator(device=config["device"]).manual_seed(config["seed"])
    if model is None:
        model = FactorModel(config["groups"], config["dimension"], config["family"], config["device"])
    if config["method"].startswith("tail"):
        model.control_variance = model.variance
    reference = model.reference()
    n = config["particles"]
    x = torch.randn(n, config["dimension"], device=config["device"], dtype=DTYPE, generator=generator)
    log_weights = torch.zeros(n, device=x.device, dtype=x.dtype)
    grid = np.linspace(math.sqrt(config["u_max"]), 0, config["steps"] + 1) ** 2
    method = config["method"]
    resamples = 0
    min_ess = n
    score_evaluations = 0
    if x.is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with (output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "u", "ess", "resampled", "max_abs_state", "max_abs_log_increment"])
        writer.writeheader()
        for k, (u, u_next) in enumerate(zip(grid[:-1], grid[1:])):
            h = float(u - u_next)
            if method in ("full", "unweighted"):
                drift, potential = model.exact(x, float(u))
                score_evaluations += n * config["groups"]
            else:
                control = method.startswith(("cv", "tail"))
                drift, potential = model.estimate(x, float(u), config["batch"], generator, control=control, naive=method == "naive")
                score_evaluations += n * config["batch"]
            if method.endswith("cumulant"):
                drift2, potential2 = model.estimate(x, float(u), config["batch"], generator, control=method.startswith(("cv", "tail")))
                score_evaluations += n * config["batch"]
                drift = 0.5 * (drift + drift2)
                increment = 0.5 * h * (potential + potential2) - h * h * (potential - potential2).square() / 8
            else:
                increment = h * potential
            if config["diffusion"] > 0:
                diffusion = config["diffusion"]
                drift = (1 + diffusion) * drift - 0.5 * diffusion * x
                noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
                x = x + h * drift + math.sqrt(h * diffusion) * noise
            else:
                x = x + h * drift
            if method != "unweighted":
                log_weights += increment
            if not torch.isfinite(x).all() or not torch.isfinite(log_weights).all():
                raise FloatingPointError(f"nonfinite state at step {k}")
            log_weights -= torch.logsumexp(log_weights, dim=0)
            weights = log_weights.exp()
            ess = float(1 / weights.square().sum())
            min_ess = min(min_ess, ess)
            resampled = ess < config["ess_threshold"] * n and k < config["steps"] - 1
            if resampled:
                x = x[systematic_resample(weights, generator)]
                log_weights.fill_(-math.log(n))
                resamples += 1
            if k % max(1, config["steps"] // 32) == 0 or k == config["steps"] - 1 or resampled:
                writer.writerow({"step": k + 1, "u": u_next, "ess": ess, "resampled": resampled, "max_abs_state": float(x.abs().max()), "max_abs_log_increment": float(increment.abs().max())})
        stream.flush()
    if x.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    samples = x.cpu().numpy()
    weights = log_weights.exp().cpu().numpy()
    summary = dict(config)
    summary.update(measure(samples, weights, reference))
    summary.update({"seconds": elapsed, "score_evaluations": score_evaluations, "minimum_ess_fraction": min_ess / n, "final_ess_fraction": float(1 / np.sum(weights ** 2)) / n, "resampling_count": resamples, "peak_gpu_bytes": torch.cuda.max_memory_allocated() if x.is_cuda else 0, "initialization": "standard normal at finite u_max", "status": "completed"})
    np.savez_compressed(output / "samples.npz", samples=samples, weights=weights)
    np.savez_compressed(output / "reference.npz", **reference)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "run.log").write_text(json.dumps({"seconds": elapsed, "status": "completed", "steps": config["steps"]}) + "\n")
    (output / "done").write_text("completed\n")
    print(json.dumps(summary), flush=True)
    return summary


def runtime():
    return {"python": platform.python_version(), "python_executable": os.sys.executable, "torch": torch.__version__, "numpy": np.__version__, "host": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def self_check(output):
    results = {}
    x = torch.tensor([[-1.2], [0.3], [1.1]], dtype=DTYPE)
    generator = torch.Generator().manual_seed(42)
    model = FactorModel(4, 1, "mixture", "cpu")
    all_batches = torch.tensor(list(itertools.product(range(4), repeat=3)))
    for control in (False, True):
        errors = []
        for point in x:
            xx = point.repeat(len(all_batches), 1)
            bhat, ghat = model.estimate(xx, 0.7, 3, generator, control=control, indices=all_batches)
            b, g = model.exact(point[None], 0.7)
            errors.extend([float(abs(bhat.mean() - b.item())), float(abs(ghat.mean() - g.item()))])
        results[f"unbiased_control_{control}"] = max(errors)
        assert max(errors) < 2e-12
    gaussian = FactorModel(8, 1, "gaussian", "cpu")
    gaussian_error = []
    for u in (0.0, 0.3, 2.0, 10.0):
        b, g = gaussian.exact(x, u)
        bhat, ghat = gaussian.estimate(x, u, 2, generator, control=True)
        gaussian_error.extend([float((b - bhat).abs().max()), float((g - ghat).abs().max())])
    results["gaussian_control_exactness"] = max(gaussian_error)
    assert max(gaussian_error) < 1e-10
    model.control_variance = model.variance
    tail_errors = []
    for point in x:
        xx = point.repeat(len(all_batches), 1)
        bhat, ghat = model.estimate(xx, 0.7, 3, generator, control=True, indices=all_batches)
        b, g = model.exact(point[None], 0.7)
        tail_errors.extend([float(abs(bhat.mean() - b.item())), float(abs(ghat.mean() - g.item()))])
    results["tail_control_unbiasedness"] = max(tail_errors)
    assert max(tail_errors) < 1e-11
    extreme_states = torch.tensor([[-10000.0], [0.0], [10000.0]], dtype=DTYPE)
    for u in (0.0, 0.5, 3.0):
        r, r0 = model.residuals(extreme_states, u)
        alpha, variance, _, _ = model.coefficients(u)
        bound = alpha * (model.means - model.mean[..., None]).abs().max(-1).values / variance
        assert bool(((r - r0).abs() <= bound[None] + 1e-10).all())
    results["tail_residual_bound"] = "passed"
    reference = model.reference(points=32769)
    reference_fine = model.reference(points=65537)
    moments = []
    for power in (1, 2, 4):
        moments.append(abs(trapezoid(reference["grid"] ** power * reference["density"][0], reference["grid"]) - trapezoid(reference_fine["grid"] ** power * reference_fine["density"][0], reference_fine["grid"])))
    results["quadrature_moment_difference"] = max(moments)
    assert max(moments) < 1e-10
    # 直接微分归一化混合分布，独立检验 Feynman–Kac 势函数。
    xx = torch.tensor([[0.37]], dtype=DTYPE, requires_grad=True)
    uu = torch.tensor(0.8, dtype=DTYPE, requires_grad=True)
    alpha = torch.exp(-0.5 * uu)
    variance = 1 - alpha.square() + alpha.square() * model.variance
    diff = xx[:, None, :, None] - alpha * model.means[None]
    log_components = model.weights.log()[None] - 0.5 * (diff.square() / variance[None, ..., None] + torch.log(2 * math.pi * variance)[None, ..., None])
    logrho = torch.logsumexp(log_components, -1).sum() + 0.5 * (model.groups - 1) * xx.square().sum()
    s = torch.autograd.grad(logrho, xx, create_graph=True)[0]
    du = torch.autograd.grad(logrho, uu, retain_graph=True)[0]
    b = 0.5 * (xx + s)
    divb = torch.autograd.grad(b.sum(), xx)[0].sum()
    general_potential = -du + divb + (b * s).sum()
    exact_b, exact_g = model.exact(xx.detach(), float(uu.detach()))
    results["fk_identity_error"] = float(abs(general_potential.detach() - exact_g[0]))
    results["drift_identity_error"] = float(abs(b.detach() - exact_b).max())
    assert results["fk_identity_error"] < 1e-11
    assert results["drift_identity_error"] < 1e-11
    results["status"] = "passed"
    results["runtime"] = runtime()
    Path(output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--family", choices=["gaussian", "mixture", "weak_mixture"], default="mixture")
    parser.add_argument("--method", choices=["full", "unweighted", "naive", "unbiased", "cumulant", "cv", "cv_cumulant", "tail", "tail_cumulant"], default="full")
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--dimension", type=int, default=1)
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--u-max", type=float, default=20)
    parser.add_argument("--ess-threshold", type=float, default=0.5)
    parser.add_argument("--diffusion", type=float, default=1.0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.self_check:
        self_check(args.self_check)
    else:
        if args.output is None:
            parser.error("--output is required")
        config = vars(args).copy()
        del config["self_check"]
        del config["output"]
        config["runtime"] = runtime()
        run_cell(config, args.output)


if __name__ == "__main__":
    main()
