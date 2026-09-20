import json
import os
from pathlib import Path

import numpy as np
import pytest

from run_learned_factor_study_20260921 import (CHECKPOINT_SHA256, execute, finish_cell, load_bank,
    prepare_asset, select, settings)
from run_solid_20260920 import sha256, write_json


def test_frozen_settings_and_missing_selection_rejected():
    configurations = settings()
    assert len(configurations) == 8
    assert len({c["setting_id"] for c in configurations}) == 8
    assert len([c for c in configurations if c["role"] == "baseline"]) == 3
    assert len([c for c in configurations if c["role"] == "candidate"]) == 1
    with pytest.raises(ValueError, match="96"):
        select([])


@pytest.mark.skipif(not os.environ.get("ICLR_TRAINING_ROOT"), reason="requires original five trained checkpoints")
def test_original_checkpoint_bank_and_real_end_to_end_cell(tmp_path):
    models, bank = load_bank(Path(os.environ["ICLR_TRAINING_ROOT"]))
    assert [row["sha256"] for row in bank["checkpoints"]] == CHECKPOINT_SHA256
    root = tmp_path / "real-learned-task"
    parameters, prediction_seconds = prepare_asset(root, 1599, models)
    assert prediction_seconds > 0
    config = dict(settings()[4], dataset_seed=1599, repeat=0, cell_id=0, particles=512, phase="development")
    x, w, logz, diagnostic = execute(parameters, config, 444, "cpu", particles=512)
    folder = root / "development" / "cells" / "cell_0000"
    folder.mkdir(parents=True)
    write_json(folder / "config.json", dict(config, sampler_seed=444))
    np.savez_compressed(folder / "samples.npz", samples=x, weights=w)
    write_json(folder / "diagnostics.json", diagnostic)
    write_json(folder / "gpu_processes.json", dict(before=[], after=[]))
    row = finish_cell((root, "development", config, logz, diagnostic))
    assert np.isfinite(row["sliced_w1_32"]) and row["seconds"] > row["sampling_seconds"]
    assert (folder / "done").read_text().strip() == "completed"
    receipt = json.loads((folder / "receipt.json").read_text())
    assert all(sha256(folder / name) == digest for name, digest in receipt["files"].items())
