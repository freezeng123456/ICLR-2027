import csv
import hashlib
import io
from pathlib import Path

import pytest

from scripts.package_probability_confirmation_20260922 import validate_resources


RESOURCE_RECORD = Path(__file__).resolve().parents[1] / "results/probability_confirmation_remote_review_20260922/workflow_evidence/slurm_accounting.psv"


def test_actual_scnet_slurm_record_passes_authorized_resource_checks():
    assert hashlib.sha256(RESOURCE_RECORD.read_bytes()).hexdigest() == "f6a3aaa1d7d400083eb3cdad3089c85b11ae025284d9a16470bf498008b14306"
    primary = validate_resources(RESOURCE_RECORD.read_text())
    assert primary["JobID"] == "24232628"
    assert primary["Elapsed"] == "01:18:58"
    assert primary["AllocCPUS"] == "4"
    assert primary["ReqMem"] == "8G"


@pytest.mark.parametrize("row_index,field,value", [
    (0, "Timelimit", "03:00:00"),
    (0, "ReqMem", "16G"),
    (0, "AllocCPUS", "8"),
    (0, "State", "RUNNING"),
    (0, "ExitCode", "1:0"),
    (1, "State", "FAILED"),
    (2, "ExitCode", "1:0"),
])
def test_changed_actual_resource_record_is_rejected(row_index, field, value):
    reader = csv.DictReader(io.StringIO(RESOURCE_RECORD.read_text()), delimiter="|")
    rows = list(reader)
    rows[row_index][field] = value
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames, delimiter="|")
    writer.writeheader()
    writer.writerows(rows)
    with pytest.raises(ValueError):
        validate_resources(output.getvalue())


@pytest.mark.parametrize("row_index", [1, 2])
def test_missing_step_from_actual_record_is_rejected(row_index):
    reader = csv.DictReader(io.StringIO(RESOURCE_RECORD.read_text()), delimiter="|")
    rows = list(reader)
    rows.pop(row_index)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames, delimiter="|")
    writer.writeheader()
    writer.writerows(rows)
    with pytest.raises(ValueError, match="both expected steps"):
        validate_resources(output.getvalue())
