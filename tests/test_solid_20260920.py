import itertools

import numpy as np
import torch

from composition_benchmark import DTYPE
from composition_extension import ExtensionModel, certificate
from run_solid_20260920 import controlled_certificate, matrix


def test_complete_independent_matrix():
    cells = matrix()
    assert len(cells) == 630
    assert [sum(c["suite"] == name for c in cells) for name in ["heldout", "boundary", "tail_error"]] == [300, 240, 90]
    heldout = [c for c in cells if c["suite"] == "heldout"]
    assert set(c["dataset_seed"] for c in heldout) == set(range(100, 120))
    for training_seed, dataset_seed in itertools.product(range(5), range(100, 120)):
        group = [c for c in heldout if c["training_seed"] == training_seed and c["dataset_seed"] == dataset_seed]
        assert {c["method"] for c in group} == {"full", "unbiased", "tail"}
        assert len({c["seed"] for c in group}) == 1


def test_exact_tail_certificate_matches_full():
    model = ExtensionModel(8, 1, "weak_mixture", "cpu")
    model.control_variance = model.variance
    config = dict(steps=128, batch=4, diffusion=1.0, u_max=20.0, method="full")
    left, right = controlled_certificate(model, config), certificate(model, config)
    assert left["finite_normalizer"] == right["finite_normalizer"]
    for key in ["smallest_denominator", "maximum_terminal_component_variance"]:
        np.testing.assert_allclose(left["coordinates"][0][key], right["coordinates"][0][key], rtol=1e-12, atol=1e-12)


def test_misspecified_control_remains_unbiased():
    indices = torch.tensor(list(itertools.product(range(4), repeat=3)))
    generator = torch.Generator().manual_seed(20260920)
    for family, delta in itertools.product(["gaussian", "mixture", "weak_mixture"], [-0.1, 0, 0.1]):
        model = ExtensionModel(4, 2, family, "cpu")
        model.control_variance = model.variance * (1 + delta)
        point = torch.tensor([[0.31, -0.47]], dtype=DTYPE)
        drift, potential = model.estimate(point.repeat(len(indices), 1), 0.7, 3, generator, control=True, indices=indices)
        exact_drift, exact_potential = model.exact(point, 0.7)
        torch.testing.assert_close(drift.mean(0), exact_drift[0], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(potential.mean(), exact_potential[0], atol=1e-12, rtol=1e-12)
