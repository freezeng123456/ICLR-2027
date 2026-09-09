import numpy as np
import torch

from exp_conditioning import load_pfn
from pilot_context import evaluate, fitted_scores, make_tasks


def test_risk_ranker_never_fits_test_labels():
    rng = np.random.default_rng(3)
    features = rng.standard_normal((8, 16, 4))
    labels = np.exp(rng.standard_normal((8, 16)))
    first, fit = fitted_scores(features, labels, 4)
    labels[4:] *= 1000
    second, fit_changed = fitted_scores(features, labels, 4)
    np.testing.assert_array_equal(first, second)
    assert fit == fit_changed


def test_real_checkpoint_batching_and_context_order():
    torch.set_num_threads(1)
    data = make_tasks("jump", 8, 8, 19)
    model, _ = load_pfn("pfn_jump_40k.pt")
    first = evaluate(model, data, "cpu", batch_size=8)
    data["xc"] = data["xc"][:, ::-1].copy()
    data["yc"] = data["yc"][:, ::-1].copy()
    second = evaluate(model, data, "cpu", batch_size=1)
    for key in first:
        np.testing.assert_allclose(first[key], second[key], atol=2e-5, rtol=2e-5)
    assert np.all(data["latent_var"] > 0)
