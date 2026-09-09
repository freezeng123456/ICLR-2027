import itertools
import math

import numpy as np
import torch

from composition_benchmark import DTYPE, FactorModel
from gaussian_integrability import audit_path


class ExtensionModel(FactorModel):
    def __init__(self, groups, dimension, family, device, sampling="with_replacement", parameters=None):
        super().__init__(groups, dimension, "mixture" if parameters is not None else family, device)
        self.family = family
        self.sampling = sampling
        if parameters is not None:
            self.variance = torch.as_tensor(parameters["variance"], device=device, dtype=DTYPE)
            self.means = torch.as_tensor(parameters["means"], device=device, dtype=DTYPE)
            self.weights = torch.as_tensor(parameters["weights"], device=device, dtype=DTYPE)
            assert self.variance.shape == (groups, dimension)
            assert self.means.shape == self.weights.shape == (groups, dimension, 2)
            assert bool(((self.variance > 0) & (self.variance < 1)).all())
            assert bool((self.weights > 0).all())
            assert bool(torch.isfinite(self.means).all())
            torch.testing.assert_close(self.weights.sum(-1), torch.ones_like(self.variance))
            self.mean = (self.weights * self.means).sum(-1)
            self.marginal_variance = self.variance + (self.weights * (self.means - self.mean[..., None]).square()).sum(-1)
            self.control_variance = self.marginal_variance

    def estimate(self, x, u, batch, generator, control=False, naive=False, indices=None):
        if self.sampling == "with_replacement":
            return super().estimate(x, u, batch, generator, control, naive, indices)
        assert self.sampling == "without_replacement" and 2 <= batch <= self.groups
        assert not naive
        if indices is None:
            # 独立连续随机键的前 M 项产生均匀子集；计时包含全部 G 个随机键。
            keys = torch.rand((len(x), self.groups), dtype=DTYPE, device=x.device, generator=generator)
            indices = keys.topk(batch, dim=1, largest=False).indices
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
        pair_sum = e_sum.square().sum(-1) - e.square().sum((1, 2))
        potential = constant + linear + self.groups * (self.groups - 1) * pair_sum / (2 * batch * (batch - 1))
        return 0.5 * (r0_total + self.groups * e_sum / batch), potential


def certificate(model, config):
    method = "full" if config["method"].startswith("tail") or config["method"] == "full" else (
        "without_replacement" if model.sampling == "without_replacement" else "unbiased"
    )
    strengths = 1 / model.variance.cpu().numpy() - 1
    coordinates = [audit_path(strengths[:, d], config["steps"], config["batch"], method,
                              diffusion=config["diffusion"], maximum_u=config["u_max"])
                   for d in range(model.dimension)]
    finite = all(row["finite_normalizer"] for row in coordinates)
    return {"finite_normalizer": finite, "coordinates": coordinates, "certificate_method": method,
            "scope": "Gaussian exact; equal-component-variance mixtures strict tail-envelope certificate"}


def verify_extension():
    generator = torch.Generator().manual_seed(42)
    errors = []
    for family, dimension, batch in itertools.product(["gaussian", "mixture", "weak_mixture"], [1, 3], [2, 3, 4]):
        model = ExtensionModel(4, dimension, family, "cpu", "without_replacement")
        indices = torch.tensor(list(itertools.combinations(range(4), batch)))
        for control, tail in [(False, False), (True, False), (True, True)]:
            model.control_variance = model.variance if tail else model.marginal_variance
            for u in [0, 0.7, 10]:
                point = torch.linspace(-1.2, 0.8, dimension, dtype=DTYPE)[None]
                drift, potential = model.estimate(point.repeat(len(indices), 1), u, batch, generator, control, indices=indices)
                exact_drift, exact_potential = model.exact(point, u)
                errors.extend([(drift.mean(0) - exact_drift[0]).abs().max().item(), abs(potential.mean().item() - exact_potential.item())])
    assert max(errors) < 2e-11
    rng = np.random.default_rng(317)
    comparisons = 0
    for groups in range(4, 11):
        for batch in range(2, groups + 1):
            for trial in range(6):
                strengths = np.exp(rng.uniform(-4, 3, groups))
                if trial == 0:
                    strengths[:] = strengths[0]
                steps = [16, 64, 128][trial % 3]
                kwargs = dict(diffusion=[0, 1, 2][trial % 3], maximum_u=[10, 20][trial % 2])
                reduced = audit_path(strengths, steps, batch, "without_replacement", **kwargs)
                complete = audit_path(strengths, steps, batch, "without_replacement", exhaustive=True, **kwargs)
                assert reduced["finite_normalizer"] == complete["finite_normalizer"]
                for key in ["failure_step", "variance_before", "maximum_terminal_component_variance", "smallest_denominator"]:
                    if key in reduced:
                        np.testing.assert_allclose(reduced[key], complete[key], rtol=1e-9, atol=1e-11)
                comparisons += 1
    return {"status": "passed", "wor_unbiased_max_error": max(errors), "prefix_suffix_vs_exhaustive_paths": comparisons}
