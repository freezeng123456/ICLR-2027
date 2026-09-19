import itertools

import numpy as np

from analyze_solid_20260920 import independent_population
from composition_extension import ExtensionModel, certificate
from run_solid_20260920 import controlled_certificate


def test_independent_population_with_raw_and_controlled_updates():
    for family, steps, sampling in itertools.product(["gaussian", "weak_mixture"], [16, 128], ["with_replacement", "without_replacement"]):
        model = ExtensionModel(8, 3, family, "cpu", sampling)
        config = dict(suite="boundary", method="unbiased", sampling=sampling, batch=4, steps=steps,
                      diffusion=1.0, u_max=20.0)
        original = certificate(model, config)
        independent = independent_population(model.variance.numpy(), config)
        assert independent["finite_normalizer"] == original["finite_normalizer"]
        for left, right in zip(independent["coordinates"], original["coordinates"]):
            assert left["finite_normalizer"] == right["finite_normalizer"]
            if left["finite_normalizer"]:
                np.testing.assert_allclose(left["maximum_terminal_component_variance"], right["maximum_terminal_component_variance"], rtol=1e-10)
            else:
                assert left["failure_step"] == right["failure_step"]
    for delta in [-0.25, 0, 0.25]:
        model = ExtensionModel(8, 3, "weak_mixture", "cpu")
        model.control_variance = model.variance * (1 + delta)
        config = dict(suite="tail_error", method="cv", sampling="with_replacement", batch=4, steps=128,
                      diffusion=1.0, u_max=20.0, control_delta=delta)
        original = controlled_certificate(model, config)
        independent = independent_population(model.variance.numpy(), config)
        assert independent["finite_normalizer"] == original["finite_normalizer"]
        for left, right in zip(independent["coordinates"], original["coordinates"]):
            np.testing.assert_allclose(left["smallest_denominator"], right["smallest_denominator"], rtol=1e-10, atol=1e-12)
