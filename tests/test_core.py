from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import norm

from eval_decision import decision_curves, selected_risk
from exp_conditioning import PFN, load_pfn, train as train_gp
from exp_jump import train as train_jump
from identifiability import gauss_kl
import eval_conditioning
import eval_jump


torch.set_num_threads(1)


def test_gaussian_projection_excess_nll():
    weights = np.array([0.3, 0.7])
    means = np.array([-2.0, 1.0])
    variances = np.array([0.2, 0.4])
    mean = weights @ means
    variance = weights @ (variances + means ** 2) - mean ** 2
    m_net, v_net = 0.2, 0.8

    def integrand(y):
        density = weights @ norm.pdf(y, means, np.sqrt(variances))
        return density * (norm.logpdf(y, mean, np.sqrt(variance))
                          - norm.logpdf(y, m_net, np.sqrt(v_net)))

    expected, _ = quad(integrand, -np.inf, np.inf, epsabs=1e-10)
    assert gauss_kl(mean, variance, m_net, v_net) == pytest.approx(expected, abs=1e-10)


def test_query_isolation_and_context_permutation():
    torch.manual_seed(42)
    model = PFN(64, 2).eval()
    x, y = torch.randn(1, 12), torch.randn(1, 12)
    with torch.no_grad():
        mu, lv = model(x, y, 8)
        perm = torch.tensor([4, 7, 2, 1, 5, 0, 3, 6, 8, 9, 10, 11])
        perm_mu, perm_lv = model(x[:, perm], y[:, perm], 8)
        x_changed, y_changed = x.clone(), y.clone()
        x_changed[:, 9:] += 100
        y_changed[:, 8:] += 200
        changed_mu, changed_lv = model(x_changed, y_changed, 8)
    torch.testing.assert_close(mu, perm_mu, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(lv, perm_lv, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(mu[:, 0], changed_mu[:, 0], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(lv[:, 0], changed_lv[:, 0], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("train", [train_gp, train_jump])
def test_training_reproducibility_and_reload(train):
    root = Path("work/test-training") / train.__module__
    root.mkdir(parents=True, exist_ok=True)
    models = [train(4, bs=2, ckpt=root / f"seed_{seed}_{i}.pt", d_model=64, seed=seed)
              for i, seed in enumerate([13, 13, 17])]
    states = [m.state_dict() for m in models]
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
    assert any(not torch.equal(states[0][k], states[2][k]) for k in states[0])
    torch.manual_seed(13)
    initial = PFN(64, 2).state_dict()
    assert any(not torch.equal(initial[k], states[0][k]) for k in initial)
    restored, width = load_pfn(root / "seed_13_0.pt")
    assert width == 64
    assert all(torch.equal(restored.state_dict()[k], states[0][k]) for k in states[0])
    assert all(torch.isfinite(v).all() for v in states[0].values())


def test_fixed_predictor_oracle_separates_selection_from_mean_error():
    mu_e, var_e = np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4])
    mu_p, var_p = np.array([4.0, 0, 0, 0]), var_e.copy()
    curves = decision_curves(mu_p, var_p, mu_e, var_e, mu_e)
    assert curves["network"][0] > curves["same_mean_oracle"][0]
    assert curves["network"][-1] == pytest.approx(curves["same_mean_oracle"][-1])
    perfect = decision_curves(mu_e, var_e, mu_e, var_e, mu_e)
    np.testing.assert_allclose(perfect["network"], perfect["bayes"])
    np.testing.assert_allclose(selected_risk(var_p, var_e), selected_risk(10 * var_p, var_e))


@pytest.mark.parametrize("module,checkpoint,latent", [
    (eval_conditioning, "pfn_cond.pt", 2.0),
    (eval_jump, "pfn_jump.pt", 50.0),
])
def test_all_boundary_projection_retains_primary_metrics(module, checkpoint, latent):
    model, _ = load_pfn(checkpoint)
    grids = module.quad_grids(2, 2)
    grid_args = grids if module is eval_conditioning else (grids,)
    # GP 的特征向量符号可随 LAPACK 实现变化，固定真实任务集合验证边界分支。
    rows = [module.run_cell(model, latent, 0.5, 24, "paired", *grid_args, n_tasks=1, seed=seed)
            for seed in range(8)]
    assert any(row["n_edge"] == 1 for row in rows)
    for row in rows:
        assert np.isfinite(row["gap"])
        if row["n_edge"] == 1:
            assert row["z_net"] == row["z_exact"] == []
            assert row["shift_net"] is None and row["shift_exact"] is None
            if module is eval_conditioning:
                assert module.update_deficit([row])["failure_model"]["status"] == "unavailable_all_boundary"
        else:
            assert len(row["z_net"]) == len(row["z_exact"]) == 1
            assert np.isfinite(row["shift_net"]) and np.isfinite(row["shift_exact"])
