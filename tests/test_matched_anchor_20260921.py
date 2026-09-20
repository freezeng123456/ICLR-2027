import json
from pathlib import Path

import pytest

from run_matched_anchor_20260921 import matched_cells


def test_real_saved_manifest_produces_only_equal_budget_comparisons():
    path = Path(__file__).resolve().parents[1] / "work/recovered-paper-20260921/new/run-anchor/confirmation/manifest.json"
    if not path.is_file():
        pytest.skip("recovered confirmation manifest is required")
    manifest = json.loads(path.read_text())
    cells, primary = matched_cells(manifest)
    assert len(cells) == 300 and len(primary) == 100
    assert {c["particles"] for c in cells} == {8192}
    assert {c["steps"] for c in cells} == {1024}
    assert {c["method"] for c in cells} == {"full", "factorized_full", "tail_anchored"}
    manifest["cells"] = [c for c in manifest["cells"] if not (c["training_seed"] == 4 and c["dataset_seed"] == 919)]
    with pytest.raises(ValueError):
        matched_cells(manifest)
