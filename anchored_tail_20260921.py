import math
import time

import numpy as np
from scipy.special import expit
import torch

from composition_extension import ExtensionModel


class AnchoredTailModel(ExtensionModel):

    def __init__(self, groups, dimension, family, device, sampling="without_replacement", parameters=None,
                 grid=None, newton_iterations=6, step_limit=1.0, curvature_floor=0.2):
        if grid is None:
            raise ValueError("A fixed decreasing grid is required")
        if newton_iterations != 6 or step_limit != 1.0 or curvature_floor != 0.2:
            raise ValueError("The anchored protocol fixes six iterations, step_limit=1, curvature_floor=0.2")
        super().__init__(groups, dimension, family, device, sampling, parameters)
        self.anchor_grid = np.asarray(grid, dtype=np.float64)
        if self.anchor_grid.ndim != 1 or len(self.anchor_grid) < 1 or not np.isfinite(self.anchor_grid).all():
            raise ValueError("grid must be a finite one-dimensional array")
        if len(self.anchor_grid) > 1 and not (np.diff(self.anchor_grid) < 0).all():
            raise ValueError("grid must be strictly decreasing")
        if (self.anchor_grid < 0).any():
            raise ValueError("grid must be nonnegative")
        self.newton_iterations = newton_iterations
        self.step_limit = step_limit
        self.curvature_floor = curvature_floor
        preparation_start = time.perf_counter()
        self._variance_np = self.variance.detach().cpu().numpy().copy()
        self._means_np = self.means.detach().cpu().numpy().copy()
        self._weights_np = self.weights.detach().cpu().numpy().copy()
        self._cache = self._prepare_cache()
        self._torch_cache = self._prepare_torch_cache()
        if self.variance.is_cuda:
            torch.cuda.synchronize(self.variance.device)
        self.preparation_seconds = time.perf_counter() - preparation_start
        self.preparation_calls = int(len(self.anchor_grid) * (7 * self.groups))

    def _numpy_coefficients(self, u):
        alpha = math.exp(-0.5 * float(u))
        alpha2 = alpha * alpha
        variance = self._variance_np
        means = self._means_np
        weights = self._weights_np
        v = 1 - alpha2 + alpha2 * variance
        a = 1 - 1 / v
        mean = (weights * means).sum(-1)
        b = alpha * mean / v
        delta = means[..., 1] - means[..., 0]
        slope = alpha * delta / v
        intercept = np.log(weights[..., 1] / weights[..., 0]) - alpha2 * (means[..., 1] ** 2 - means[..., 0] ** 2) / (2 * v)
        e0 = alpha * (means[..., 0] - mean) / v
        edelta = alpha * delta / v
        return a, b, e0, edelta, slope, intercept

    @staticmethod
    def _residual_numpy(x, e0, edelta, slope, intercept):
        if x.ndim == 1:
            probability = expit(x[None, :] * slope + intercept)
            return e0 + edelta * probability
        probability = expit(x[:, None, :] * slope[None, :, :] + intercept[None, :, :])
        return e0[None, :, :] + edelta[None, :, :] * probability

    def _prepare_cache(self):
        cache = {}
        for u in self.anchor_grid:
            a, b, e0, edelta, slope, intercept = self._numpy_coefficients(u)
            total_a = a.sum(0)
            total_b = b.sum(0)
            anchor = total_b / (1 - total_a)
            for _ in range(self.newton_iterations):
                residual = self._residual_numpy(anchor, e0, edelta, slope, intercept)
                total = (anchor[None, :] * a + b + residual).sum(0)
                probability = expit(anchor[None, :] * slope + intercept)
                derivative = (a + edelta * slope * probability * (1 - probability)).sum(0)
                denominator = np.maximum(1 - derivative, self.curvature_floor)
                step = np.clip((total - anchor) / denominator, -self.step_limit, self.step_limit)
                anchor = anchor + step
            residual = self._residual_numpy(anchor, e0, edelta, slope, intercept)
            anchored_b = b + residual
            cache[float(u)] = {
                "u": float(u),
                "a": a,
                "b_original": b,
                "b_anchored": anchored_b,
                "anchor": anchor,
            }
        return cache

    def _prepare_torch_cache(self):
        cache = {}
        for key, entry in self._cache.items():
            a, b = entry["a"], entry["b_anchored"]
            arrays = dict(a=a, b=b, total_a=a.sum(0), total_b=b.sum(0), a2=(a ** 2).sum(0),
                          ab=(a * b).sum(0), b2=(b ** 2).sum(0))
            cache[key] = {name: torch.as_tensor(value, dtype=self.variance.dtype, device=self.variance.device)
                          for name, value in arrays.items()}
        return cache

    def _entry(self, u):
        key = float(u)
        if key not in self._cache:
            raise KeyError("u must be one of the prepared grid points")
        return self._cache[key]

    def _torch_entry(self, u):
        key = float(u)
        if key not in self._torch_cache:
            raise KeyError("u must be one of the prepared grid points")
        return self._torch_cache[key]

    def anchor_parameters(self, u):
        entry = self._entry(u)
        return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in entry.items()}

    def estimate(self, x, u, batch, generator, control=False, naive=False, indices=None):
        if not control:
            return super().estimate(x, u, batch, generator, control=False, naive=naive, indices=indices)
        if self.sampling != "without_replacement":
            raise ValueError("Anchored control requires without_replacement sampling")
        if naive:
            raise ValueError("Anchored control requires the unbiased WOR potential estimator")
        if not 2 <= batch <= self.groups:
            raise ValueError("batch must satisfy 2 <= batch <= groups")
        entry = self._torch_entry(u)
        if x.device != self.variance.device:
            raise ValueError("x must use the model device")
        a = entry["a"]
        b = entry["b"]
        if indices is None:
            keys = torch.rand((len(x), self.groups), dtype=x.dtype, device=x.device, generator=generator)
            indices = keys.topk(batch, dim=1, largest=False).indices
        residual, _ = self.residuals(x, u, indices)
        r0 = a[indices] * x[:, None, :] + b[indices]
        e = residual - r0
        r0_total = entry["total_a"] * x + entry["total_b"]
        q0 = (entry["a2"] * x.square() + 2 * entry["ab"] * x + entry["b2"]).sum(-1)
        constant = 0.5 * (r0_total.square().sum(-1) - q0)
        linear = (r0_total * (self.groups * e.mean(1))).sum(-1)
        linear -= self.groups * (r0 * e).sum(-1).mean(1)
        e_sum = e.sum(1)
        pair_sum = e_sum.square().sum(-1) - e.square().sum((1, 2))
        potential = constant + linear + self.groups * (self.groups - 1) * pair_sum / (2 * batch * (batch - 1))
        return 0.5 * (r0_total + self.groups * e_sum / batch), potential

    def cost_report(self):
        return {"preparation_calls": self.preparation_calls, "preparation_seconds": self.preparation_seconds,
                "preparation_factor_calls_per_grid_point": 7 * self.groups,
                "newton_iterations": self.newton_iterations, "anchor_grid_points": len(self.anchor_grid),
                "anchor_rule": "parameter-only six-step clipped Newton fixed-point iteration"}


def make_anchored_tail_model(*args, **kwargs):
    return AnchoredTailModel(*args, **kwargs)
