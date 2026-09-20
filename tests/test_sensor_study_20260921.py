import numpy as np

from nonseparable_sensor_model_20260921 import make_problem
from nonseparable_sensor_sampler_20260921 import SensorScoreBank
from run_sensor_study_20260921 import certificate, problems, settings


def test_sensor_protocol_and_matrix_certificate():
    assert len(settings()) == 12
    assert len(problems("development")) == 8
    assert len(problems("confirmation")) == 80
    assert {p["dataset_seed"] for p in problems("development")}.isdisjoint(p["dataset_seed"] for p in problems("confirmation"))
    problem = make_problem(15, dimension=2)
    grid = np.linspace(np.sqrt(20), 0, 2049) ** 2
    bank = SensorScoreBank(problem, grid, prepare_method="full")
    result = certificate(bank.numpy_banks["A"], grid)
    assert result["status"] == "finite"
    assert np.asarray(result["final_tail_covariance"]).shape == (2, 2)
