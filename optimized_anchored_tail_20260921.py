import numpy as np
import torch

from anchored_tail_20260921 import AnchoredTailModel


class OptimizedAnchoredTailModel(AnchoredTailModel):
    """Equivalent anchored control with one device transfer per coefficient bank."""

    def _prepare_torch_cache(self):
        keys = [float(u) for u in self.anchor_grid]
        arrays = {name: [] for name in ["a", "b", "total_a", "total_b", "a2", "ab", "b2", "slope", "intercept", "edelta", "e0shift"]}
        for key in keys:
            entry = self._cache[key]
            a, b_original, e0, edelta, slope, intercept = self._numpy_coefficients(key)
            b = entry["b_anchored"]
            arrays["a"].append(a)
            arrays["b"].append(b)
            arrays["total_a"].append(a.sum(0))
            arrays["total_b"].append(b.sum(0))
            arrays["a2"].append((a ** 2).sum(0))
            arrays["ab"].append((a * b).sum(0))
            arrays["b2"].append((b ** 2).sum(0))
            arrays["slope"].append(slope)
            arrays["intercept"].append(intercept)
            arrays["edelta"].append(edelta)
            arrays["e0shift"].append(e0 + b_original - b)
        banks = {name: torch.as_tensor(np.stack(values, axis=0), dtype=self.variance.dtype, device=self.variance.device) for name, values in arrays.items()}
        self._torch_banks = banks
        self._time_index_map = {key: index for index, key in enumerate(keys)}
        return {key: {name: bank[index] for name, bank in banks.items()} for index, key in enumerate(keys)}

    def _time_index(self, u):
        key = float(u)
        try:
            return self._time_index_map[key]
        except KeyError as error:
            raise KeyError("u must be one of the prepared grid points") from error

    def estimate(self, x, u, batch, generator, control=False, naive=False, indices=None):
        if not control:
            return super().estimate(x, u, batch, generator, control=False, naive=naive, indices=indices)
        if self.sampling != "without_replacement":
            raise ValueError("Anchored control requires without_replacement sampling")
        if naive:
            raise ValueError("Anchored control requires the unbiased WOR potential estimator")
        if not 2 <= batch <= self.groups:
            raise ValueError("batch must satisfy 2 <= batch <= groups")
        if x.device != self.variance.device:
            raise ValueError("x must use the model device")
        entry = self._torch_cache[float(u)]
        if indices is None:
            keys = torch.rand((len(x), self.groups), dtype=x.dtype, device=x.device, generator=generator)
            indices = keys.topk(batch, dim=1, largest=False).indices
        if indices.shape != (len(x), batch) or indices.device != x.device:
            raise ValueError("indices must have shape (particles, batch) on the model device")
        slope = entry["slope"][indices]
        intercept = entry["intercept"][indices]
        edelta = entry["edelta"][indices]
        e0shift = entry["e0shift"][indices]
        e = e0shift + edelta * torch.sigmoid(x[:, None, :] * slope + intercept)
        a = entry["a"]
        b = entry["b"]
        r0 = a[indices] * x[:, None, :] + b[indices]
        r0_total = entry["total_a"] * x + entry["total_b"]
        q0 = (entry["a2"] * x.square() + 2 * entry["ab"] * x + entry["b2"]).sum(-1)
        constant = 0.5 * (r0_total.square().sum(-1) - q0)
        linear = (r0_total * (self.groups * e.mean(1))).sum(-1)
        linear -= self.groups * (r0 * e).sum(-1).mean(1)
        e_sum = e.sum(1)
        pair_sum = e_sum.square().sum(-1) - e.square().sum((1, 2))
        potential = constant + linear + self.groups * (self.groups - 1) * pair_sum / (2 * batch * (batch - 1))
        return 0.5 * (r0_total + self.groups * e_sum / batch), potential
