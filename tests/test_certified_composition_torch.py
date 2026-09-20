import numpy as np
import pytest
import torch

from certified_composition_torch import FactorParameters, TimeFactors, moment_certificate, prepare_grid, sample


def params(groups=4, dimension=2, gaussian=False):
    variance = np.full((groups, dimension), 0.7)
    if gaussian:
        means = np.zeros((groups, dimension, 2))
    else:
        means = np.stack((-np.ones((groups, dimension)), np.ones((groups, dimension))), -1)
    weights = np.full((groups, dimension, 2), 0.5)
    return FactorParameters(variance, means, weights)


def test_kernel_factorization_and_interpolation_exact_sum():
    p = params(); f = TimeFactors(p, 0.7, 65, 8)
    x = torch.tensor([[0.1, -0.4], [1.2, 0.3]], dtype=torch.float64)
    y = torch.tensor([[0.2, -0.1], [0.4, 0.8]], dtype=torch.float64); h = 0.03
    total, potential = f.full(x); total0, potential0 = f.proposal(x)
    assert torch.allclose(total0, (x[:, None] * f.a + f.b + f.tabulated_residual(x)).sum(1), atol=2e-14)
    direct = h * potential - (y - (1 - h / 2) * x - h * total).square().sum(-1) / (2 * h)
    decomposed = h * potential0 + f.correction_terms(x, y, h).sum(-1) - (y - (1 - h / 2) * x - h * total0).square().sum(-1) / (2 * h)
    assert torch.allclose(direct, decomposed, atol=2e-12, rtol=2e-12)


def test_bounded_residual_and_conditional_poisson_contract():
    p = params(); f = TimeFactors(p, 1.0, 65, 8); x = torch.tensor([[0.2, -0.3], [1.0, 0.4]], dtype=torch.float64)
    residual = f.residual(x) - f.tabulated_residual(x)
    assert torch.all(residual.abs() <= f.bound[None] + 1e-12)
    correction, report = f.random_correction(x, x + 0.1, 0.02, 0.2, torch.Generator().manual_seed(9), 1.0)
    assert torch.isfinite(correction).all() and report["factor_calls"] >= report["full_particles"] * p.groups
    assert report["maximum_log_relative_second_moment_bound"] <= 0.2 * (1 + 1e-12)


def test_gaussian_zero_residual_and_full_consistency():
    p = params(6, 2, gaussian=True); f = TimeFactors(p, 0.4, 65, 8); x = torch.randn((10, 2), dtype=torch.float64)
    assert torch.equal(f.residual(x), torch.zeros_like(f.residual(x)))
    grid = np.linspace(2.0, 0.0, 9); full = sample(p, grid, 32, 77, "full", device="cpu", return_type="torch"); cert = sample(p, grid, 32, 77, "certified", device="cpu", return_type="torch")
    for left, right in zip(full[:3], cert[:3]): assert np.allclose(np.asarray(left), np.asarray(right), atol=1e-12, rtol=1e-12)


def test_moment_certificate_does_not_claim_second_moment():
    p = params(3, 1); grid = np.linspace(2.0, 0.0, 12); cert = moment_certificate(p, grid, 1.0)
    assert cert["power"] == 1.0 and cert["passed"]
    assert prepare_grid(p, 8, 2.0)[1]["path_moment"]["power"] == 1.0


def test_independent_streams_and_numpy_dict_api():
    p = params(3, 1); d = {"variance": p.variance.numpy(), "means": p.means.numpy(), "weights": p.weights.numpy()}; grid = np.linspace(1.0, 0.0, 5)
    a = sample(d, grid, 16, 123, "surrogate_only", return_type="numpy"); b = sample(d, grid, 16, 123, "surrogate_only", return_type="numpy")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and len(a[3]) == len(grid) - 1


def test_tail_fixed_uses_fixed_batch_and_full_has_no_table_preparation():
    p = params(6, 1)
    grid = np.linspace(1.0, 0.0, 4)
    tail = sample(p, grid, 10, 12, "tail_fixed", batch=2, table_nodes=65, return_type="torch")
    full = sample(p, grid, 10, 12, "full", batch=2, table_nodes=65, return_type="torch")
    assert all(record["factor_calls"] == 20 and record["preparation_calls"] == 0 for record in tail[3])
    assert all(record["factor_calls"] == 60 and record["preparation_calls"] == 0 for record in full[3])


def test_event_index_shapes_and_cdf_terminal_guard():
    p = params(8, 2)
    f = TimeFactors(p, 1.0, 65, 8)
    assert f.basis_cdf.shape == (2 * p.dimension + 1, p.groups)
    assert torch.all(f.basis_cdf[:, -1][f.basis_sum > 0] == 1)
    x = torch.zeros((32, 2), dtype=torch.float64)
    y = x + 0.01
    correction, report = f.random_correction(x, y, 0.01, 1.0, torch.Generator().manual_seed(31), 1.0)
    assert correction.shape == (32,) and report["poisson_events"] >= 0


@pytest.mark.parametrize("method", ["full", "tail_fixed", "certified", "surrogate_full", "surrogate_only"])
def test_required_methods_return_records(method):
    p = params(4, 1); grid = np.linspace(1.0, 0.0, 4); out = sample(p, grid, 12, 4, method, batch=2, table_nodes=17, return_type="torch")
    assert out[0].dtype == torch.float64 and out[1].dtype == torch.float64
    assert all("preparation_calls" in record and "factor_calls" in record for record in out[3])
