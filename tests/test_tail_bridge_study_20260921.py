import json

import numpy as np
import pytest

from nonseparable_sensor_model_20260921 import SensorProblem
from run_sensor_study_20260921 import asset_name, prepare_one
from run_solid_20260920 import sha256, write_json
from run_tail_bridge_study_20260921 import development_settings, execute, finish_cell, problem_configs, select


def test_frozen_cohorts_and_equal_particle_settings():
    settings = development_settings()
    assert len(settings) == len({row["setting_id"] for row in settings}) == 26
    assert {row["particles"] for row in settings} == {16384}
    assert sum(row["role"] == "candidate" for row in settings) == 12
    assert sum(row["role"] == "baseline" for row in settings) == 12
    development, confirmation = problem_configs("development"), problem_configs("confirmation")
    assert len(development) == 16 and len(confirmation) == 80
    assert {row["dataset_seed"] for row in development}.isdisjoint({row["dataset_seed"] for row in confirmation})
    with pytest.raises(RuntimeError, match="complete frozen"):
        select([], settings)


def test_real_sampler_cell_receipt_and_metrics(tmp_path):
    problem_config = problem_configs("development")[0]
    prepare_one((tmp_path, problem_config))
    asset = tmp_path / "assets" / asset_name(problem_config)
    problem = SensorProblem.from_dict(json.loads((asset / "problem.json").read_text()))
    setting = dict(development_settings()[0], particles=256)
    cell = dict(problem_config, **setting, phase="development", repeat=0, cell_id=0)
    samples, weights, logz, diagnostics = execute(problem, setting, 921, "cpu")
    output = tmp_path / "development/cells/cell_0000"
    output.mkdir(parents=True)
    write_json(output / "config.json", cell)
    write_json(output / "diagnostics.json", diagnostics)
    np.savez_compressed(output / "samples.npz", samples=samples, weights=weights)
    row = finish_cell((tmp_path, cell, 921, logz, diagnostics))
    receipt = json.loads((output / "receipt.json").read_text())
    assert all(sha256(output / name) == digest for name, digest in receipt["files"].items())
    assert (output / "done").read_text().strip() == "completed"
    assert row["status"] == "completed" and row["sliced_w1_32"] >= 0
    assert row["seconds"] > 0 and np.isfinite(row["log_evidence_error"])
    assert row["particles"] == 256 and row["stages"] >= 1
