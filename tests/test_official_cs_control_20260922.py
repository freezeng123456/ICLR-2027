import numpy as np
import pytest

from confseq.betting import betting_mart
from official_cs_control_20260922 import official_interval, run_control
from decision_evidence_20260922 import BoundedStratumProblem


@pytest.mark.parametrize("method", ["betting", "empbern"])
def test_official_intervals_and_replay(method):
    values = np.linspace(0.65, 0.95, 128)
    lower, upper = official_interval(values, method)
    assert 0 <= lower < upper <= 1
    assert lower < np.mean(values) < upper
    problem = BoundedStratumProblem("audit", 1, 2, (8, 2), (0.1, 0.9))
    result = run_control(problem, method, 89, budget=100)
    assert result["calls"] == len(result["observations"])
    assert result["calls"] == result["positive_queries"] + result["negative_queries"]


def test_official_bets_are_prefix_predictable():
    values = np.linspace(0.1, 0.7, 50)
    full = betting_mart(values, 0.4, alpha=0.0125, theta=1)
    prefix = betting_mart(values[:20], 0.4, alpha=0.0125, theta=1)
    assert np.allclose(full[:20], prefix)


def test_official_root_corresponds_to_threshold():
    values = np.repeat(0.001, 300)
    lower, upper = official_interval(values, "betting")
    assert lower == 0
    wealth = betting_mart(values, upper, alpha=0.0125, theta=0)[-1]
    assert wealth == pytest.approx(80, rel=1e-5)
