import argparse
import itertools
import json
from fractions import Fraction
from pathlib import Path

import numpy as np


def coefficients(strengths, u, batch, method, exhaustive=False, control_variance=None):
    groups = len(strengths)
    variance = 1 / (1 + strengths)
    a = 1 / (1 - np.exp(-u) + np.exp(-u) * variance) - 1
    if method == "cv":
        assert control_variance is not None
        control_a = 1 / (1 - np.exp(-u) + np.exp(-u) * control_variance) - 1
        residual = a - control_a
        control_sum = control_a.sum()
        control_c = 0.5 * (control_sum ** 2 - (control_a * control_a).sum())
        if exhaustive:
            indices = np.array(list(itertools.product(range(groups), repeat=batch)))
            values = residual[indices]
            total = values.sum(1)
            squared = (values * values).sum(1)
            ahat = control_sum + groups * total / batch
            chat = control_c + control_sum * groups * total / batch
            chat -= groups * (control_a[indices] * values).mean(1)
            chat += 0.5 * (groups ** 2 * (total * total - squared) / (batch * (batch - 1)) - groups * squared / batch)
        else:
            ahat = control_sum + groups * residual
            chat = control_c + groups * (control_sum - control_a) * residual
            chat += 0.5 * groups * (groups - 1) * residual ** 2
    elif method == "full":
        ahat = np.array([a.sum()])
        chat = np.array([0.5 * (a.sum() ** 2 - (a * a).sum())])
    elif method == "without_replacement":
        assert 2 <= batch <= groups
        indices = np.array(list(itertools.combinations(range(groups), batch)))
        values = a[indices]
        total = values.sum(1)
        squared = (values * values).sum(1)
        ahat = groups * total / batch
        chat = groups * (groups - 1) * (total ** 2 - squared) / (2 * batch * (batch - 1))
    elif method not in ("naive", "unbiased"):
        raise ValueError(method)
    elif not exhaustive:
        extremes = np.array([a.min(), a.max()])
        ahat = groups * extremes
        chat = 0.5 * groups * (groups - 1) * extremes ** 2
    else:
        indices = np.array(list(itertools.product(range(groups), repeat=batch)))
        values = a[indices]
        total = values.sum(1)
        squared = (values * values).sum(1)
        ahat = groups * total / batch
        if method == "unbiased":
            chat = 0.5 * (groups ** 2 * (total * total - squared) / (batch * (batch - 1)) - groups * squared / batch)
        elif method == "naive":
            chat = 0.5 * (ahat * ahat - groups * squared / batch)
        else:
            raise ValueError(method)
    return ahat, chat


def audit_path(strengths, steps, batch, method, diffusion=1.0, maximum_u=20, exhaustive=False, control_variance=None):
    grid = np.linspace(np.sqrt(maximum_u), 0, steps + 1) ** 2
    largest_variance = 1.0
    smallest_denominator = 1.0
    rows = []
    for k, (u, unext) in enumerate(zip(grid[:-1], grid[1:])):
        h = u - unext
        a, c = coefficients(strengths, u, batch, method, exhaustive=exhaustive, control_variance=control_variance)
        denominator = 1 - 2 * h * c * largest_variance
        smallest_denominator = min(smallest_denominator, float(denominator.min()))
        if denominator.min() <= 0:
            return {"finite_normalizer": False, "failure_step": k + 1, "u": float(u), "h": float(h), "variance_before": largest_variance, "minimum_denominator": float(denominator.min()), "smallest_denominator": smallest_denominator, "coefficient_count": len(a)}
        multiplier = 1 - h * ((1 + diffusion) * a + diffusion) / 2
        next_variance = multiplier ** 2 * largest_variance / denominator + h * diffusion
        largest_variance = float(next_variance.max())
        rows.append({"step": k + 1, "u": float(u), "max_variance": largest_variance, "minimum_denominator": float(denominator.min())})
    return {"finite_normalizer": True, "maximum_terminal_component_variance": largest_variance, "smallest_denominator": smallest_denominator, "coefficient_count": len(a)}


def verify():
    a = [Fraction(1, 21), Fraction(1, 11), Fraction(1, 2), Fraction(2, 3)]
    total = sum(a)
    variance = 1 / (1 + total)
    full_c = (total ** 2 - sum(z ** 2 for z in a)) / 2
    max_c = Fraction(8, 3)
    assert 1 - full_c * variance == Fraction(2503, 3195)
    assert 1 - max_c * variance == -Fraction(167, 1065)
    for batch in (2, 3, 4):
        values = []
        for sample in itertools.product(a, repeat=batch):
            mean = sum(sample) / batch
            population_variance = sum((z - mean) ** 2 for z in sample) / batch
            q = sum(z ** 2 for z in sample)
            direct = (Fraction(16, batch * (batch - 1)) * (sum(sample) ** 2 - q) - Fraction(4, batch) * q) / 2
            reduced = (12 * mean ** 2 - (Fraction(16, batch - 1) + 4) * population_variance) / 2
            naive = ((4 * mean) ** 2 - Fraction(4, batch) * q) / 2
            assert direct == reduced
            assert naive == (12 * mean ** 2 - 4 * population_variance) / 2
            values.append(direct)
        assert sum(values) / len(values) == full_c
    comparisons = 0
    for strengths, steps, batch, method in itertools.product(
        [np.array([0.1, 0.2, 2, 4]), np.array([0.5, 0.7, 1, 1.5])],
        [16, 64, 128], [2, 3, 4], ["naive", "unbiased"],
    ):
        efficient = audit_path(strengths, steps, batch, method)
        exhaustive = audit_path(strengths, steps, batch, method, exhaustive=True)
        assert efficient["finite_normalizer"] == exhaustive["finite_normalizer"]
        for key in ("failure_step", "variance_before", "maximum_terminal_component_variance", "smallest_denominator"):
            if key in efficient:
                np.testing.assert_allclose(efficient[key], exhaustive[key], rtol=1e-10, atol=1e-12)
        comparisons += 1
    controlled_comparisons = 0
    for control_variance, steps, batch in itertools.product(
        [np.array([1.2, 0.7, 0.6, 0.8]), np.array([0.3, 0.4, 0.5, 0.6])],
        [16, 64, 128], [2, 3, 4],
    ):
        strengths = np.array([0.1, 0.2, 2, 4])
        efficient = audit_path(strengths, steps, batch, "cv", control_variance=control_variance)
        exhaustive = audit_path(strengths, steps, batch, "cv", control_variance=control_variance, exhaustive=True)
        assert efficient["finite_normalizer"] == exhaustive["finite_normalizer"]
        for key in ("failure_step", "variance_before", "maximum_terminal_component_variance", "smallest_denominator"):
            if key in efficient:
                np.testing.assert_allclose(efficient[key], exhaustive[key], rtol=1e-10, atol=1e-12)
        controlled_comparisons += 1
    without_replacement = []
    for batch in (2, 3, 4):
        values = []
        for sample in itertools.combinations(a, batch):
            c = Fraction(12, 2 * batch * (batch - 1)) * (sum(sample) ** 2 - sum(z ** 2 for z in sample))
            values.append(c)
        assert sum(values) / len(values) == full_c
        minimum_denominator = 1 - max(values) * variance
        assert minimum_denominator > 0
        without_replacement.append({"batch": batch, "maximum_quadratic_coefficient": str(max(values)), "minimum_denominator": str(minimum_denominator), "finite_one_step_normalizer": True})
    return {"rational_identities": "passed", "unbiased_batch_sizes_checked": [2, 3, 4], "full_denominator": "2503/3195", "bad_batch_denominator": "-167/1065", "extreme_vs_exhaustive_paths": comparisons, "controlled_vs_exhaustive_paths": controlled_comparisons, "without_replacement_one_step": without_replacement, "status": "passed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for strengths, steps, method in itertools.product(
        [[0.1, 0.2, 2, 4], [0.1, 0.1, 0.1, 10], [0.5, 0.7, 1, 1.5]],
        [16, 32, 64, 128, 256, 512], ["full", "naive", "unbiased"],
    ):
        result = audit_path(np.array(strengths), steps, 2, method)
        rows.append(dict(result, strengths=strengths, steps=steps, batch=2, method=method))
    report = {"verification": verify(), "rows": rows, "scope": "Exact integrability of the infinite-particle random-batch weighted Euler law for Gaussian factors. Resampling changes the finite-particle estimator, not this population operator.", "initial_variance": 1.0, "diffusion": 1.0, "maximum_u": 20.0}
    Path(args.output).write_text(json.dumps(report, indent=2))
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
