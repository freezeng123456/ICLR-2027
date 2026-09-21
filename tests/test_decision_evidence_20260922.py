import numpy as np
import pytest

from decision_evidence_20260922 import (
    BoundedStratumProblem,
    EvidenceState,
    choose_design,
    growth_optimal_mass,
    numerical_oracle_mass,
    oracle_bet,
    oracle_growth,
    run_bounded_trial,
)


@pytest.mark.parametrize("ratio", [0.01, 0.1, 0.5, 0.9, 1.001, 2, 9, 100, 10000])
def test_analytic_optimum_matches_numerical_optimization(ratio):
    analytic = float(growth_optimal_mass(ratio, 1))
    numeric = numerical_oracle_mass(ratio, 1)
    assert analytic == pytest.approx(numeric, abs=2e-5)
    assert oracle_growth(ratio, 1, analytic) >= oracle_growth(ratio, 1, ratio / (ratio + 1)) - 1e-13


def test_near_equal_and_scale_invariance():
    assert growth_optimal_mass(1, 1) == 0.5
    assert growth_optimal_mass(1 + 1e-12, 1) == pytest.approx(0.5, abs=1e-12)
    assert growth_optimal_mass(9e-100, 1e-100) == pytest.approx(growth_optimal_mass(9, 1))
    assert growth_optimal_mass(1, 9) == pytest.approx(1 - growth_optimal_mass(9, 1))


@pytest.mark.parametrize("ratio", [1.1, 2, 9, 1000])
def test_wealth_formula(ratio):
    mass = float(growth_optimal_mass(ratio, 1))
    bet = oracle_bet(ratio, 1, mass)
    growth = mass * np.log1p(bet * ratio / mass) + (1 - mass) * np.log1p(-bet / (1 - mass))
    assert growth == pytest.approx(oracle_growth(ratio, 1, mass))
    assert 0 < bet < 1 - mass


def test_null_conditional_expectation_with_adaptive_proposal():
    for mass in [0.02, 0.1, 0.5, 0.9, 0.98]:
        for fraction in [0.01, 0.5, 0.99]:
            bet = fraction * (1 - mass)
            multiplier = mass * (1 + bet / mass) + (1 - mass) * (1 - bet / (1 - mass))
            assert multiplier == pytest.approx(1)


@pytest.mark.parametrize("method", ["uniform", "variance", "growth"])
def test_predictable_design_respects_global_payoff_bounds(method):
    mass, plus, minus = choose_design(method, [0.1, 0.9, 0.3], [0.3, 0.7], 3, 2)
    assert 0 < mass < 1
    assert 0 <= plus < (1 - mass) / 2
    assert 0 <= minus < mass / 3


def test_evidence_rejects_invalid_bet():
    with pytest.raises(ValueError):
        EvidenceState().observe(-2, 0.5, 0)


def test_trial_reproducibility_and_accounting():
    problem = BoundedStratumProblem("check", 3, 1, (8, 2), (8, 2))
    first = run_bounded_trial(problem, "growth", 15, max_samples=100)
    second = run_bounded_trial(problem, "growth", 15, max_samples=100)
    assert first == second
    assert first["calls"] == len(first["trace"]) + 4
    assert first["calls"] <= 100
    assert all(1 + row[2] * row[4] > 0 and 1 - row[3] * row[4] > 0 for row in first["trace"])


def test_inputs_fail_fast():
    with pytest.raises(ValueError):
        growth_optimal_mass(0, 1)
    with pytest.raises(ValueError):
        BoundedStratumProblem("bad", 1, 0, (1, 1), (1, 1))
