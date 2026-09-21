import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from audit_matched_confirmation_20260922 import audit_cell
from run_decision_evidence_20260922 import rare_problems
from run_matched_confirmation_20260922 import METHODS, run_one


@pytest.fixture
def real_cells():
    Path("work").mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="matched-audit-", dir="work") as temporary:
        root = Path(temporary)
        problem = rare_problems()[0]
        for method in METHODS:
            run_one((problem, method, 8000, 100, str(root)))
        yield {method: root / problem.name / method / "seed_08000" for method in METHODS}


@pytest.mark.parametrize("method", METHODS)
def test_actual_trials_pass_independent_replay(real_cells, method):
    summary, pilot, _ = audit_cell(real_cells[method], replay_official=True)
    assert summary["calls"] <= 100
    assert pilot.shape == (4, 2)


@pytest.mark.parametrize("field", ["calls", "decision", "certificate_checks", "seed"])
def test_inconsistent_summary_is_rejected(real_cells, field):
    directory = real_cells["growth"]
    summary = json.loads((directory / "summary.json").read_text())
    summary[field] += 1
    (directory / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        audit_cell(directory)


@pytest.mark.parametrize("field", ["trace", "observations", "checkpoints"])
def test_corrupted_numeric_record_is_rejected(real_cells, field):
    directory = real_cells["growth"]
    with np.load(directory / "trace.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if field == "trace":
        arrays[field][0, 4] += 1
    elif field == "observations":
        arrays[field][4, 1] = 1 - arrays[field][4, 1]
    else:
        arrays[field][0] -= 1
    np.savez_compressed(directory / "trace.npz", **arrays)
    with pytest.raises(ValueError):
        audit_cell(directory)
