import numpy as np
import pytest

from decision_evidence_20260922 import BoundedStratumProblem
from run_matched_confirmation_20260922 import METHODS, trial


@pytest.mark.parametrize("method", METHODS)
def test_matched_query_and_checkpoint_contract(method):
    problem = BoundedStratumProblem("bounded", 1, 100, (8, 2), (0.002, 0.998))
    result = trial(problem, method, 123, budget=100)
    assert result["calls"] == len(result["observations"])
    assert result["calls"] == result["positive_queries"] + result["negative_queries"]
    assert result["certificate_checks"] == result["design_updates"] == len(result["checkpoints"])
    assert result["checkpoints"] == list(range(20, result["calls"] + 1, 16))
    assert result["checkpoints"][-1] == result["calls"]
    if method != "betting":
        trace = np.asarray(result["trace"])
        assert np.allclose(trace[:, 5], np.cumsum(np.log1p(trace[:, 2] * trace[:, 4])))
        assert np.allclose(trace[:, 6], np.cumsum(np.log1p(-trace[:, 3] * trace[:, 4])))


@pytest.mark.parametrize("method", METHODS)
def test_future_observations_cannot_change_first_block(method):
    problem = BoundedStratumProblem("prefix", 1, 600, (8, 2), (0.002, 0.998))
    short = trial(problem, method, 456, budget=20)
    long = trial(problem, method, 456, budget=52)
    assert short["observations"] == long["observations"][:20]
    assert short["trace"] == long["trace"][:len(short["trace"])]


@pytest.mark.parametrize("method", METHODS)
def test_terminal_partial_block_is_fully_charged(method):
    problem = BoundedStratumProblem("null", 1, 1, (8, 2), (8, 2))
    result = trial(problem, method, 891, budget=30)
    assert result["checkpoints"] == [20, 30]
    assert result["calls"] == 30
