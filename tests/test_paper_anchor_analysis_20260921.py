import json
from pathlib import Path

import numpy as np
import pytest

from analyze_paper_anchor_20260921 import aggregate, bootstrap_indices, bootstrap_values, metric_names, rectangular_cells, validate_summary_rows


def anchored_confirmation_root():
    return Path(__file__).resolve().parent.parent / "work" / "recovered-dual-20260921" / "new" / "run-anchored-confirmation" / "confirmation"


def paper_anchor_development_root():
    return Path(__file__).resolve().parent.parent / "work" / "anchor-development-reports" / "development"


def real_rows_and_manifest():
    root = anchored_confirmation_root()
    manifest_path = root / "manifest.json"
    summary_paths = sorted((root / "cells").glob("*/summary.json"))
    if not manifest_path.is_file() or not summary_paths:
        pytest.skip("installed recovered anchored-confirmation assets are unavailable")
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(path.read_text()) for path in summary_paths]
    return manifest, rows


def test_real_saved_anchored_confirmation_is_5x10x3():
    manifest, rows = real_rows_and_manifest()
    models, datasets, settings = rectangular_cells(manifest, "confirmation", canonical=False)
    assert models == [0, 1, 2, 3, 4]
    assert datasets == list(range(700, 710))
    assert settings == ["full", "tail_anchored", "tail_fixed"]
    assert len(rows) == 150 == len(models) * len(datasets) * len(settings)


def test_real_saved_rows_have_dynamic_metric_selection():
    _, rows = real_rows_and_manifest()
    names = metric_names(rows)
    assert "w1_mean" in names
    assert "seconds_including_preparation" in names
    common_names = [name for name in names if all(name in row for row in rows)]
    assert all(np.isfinite(float(row[name])) for row in rows for name in common_names)


def test_aggregation_is_invariant_to_saved_row_order():
    manifest, rows = real_rows_and_manifest()
    baseline = aggregate(rows, "confirmation", manifest, canonical=False)
    shuffled = aggregate(list(reversed(rows)), "confirmation", manifest, canonical=False)
    assert baseline["settings"] == shuffled["settings"]
    assert baseline["training_seeds"] == shuffled["training_seeds"]
    assert baseline["dataset_seeds"] == shuffled["dataset_seeds"]
    for setting in baseline["settings"]:
        assert baseline["by_setting"][setting] == shuffled["by_setting"][setting]


def test_bootstrap_matrix_reversal_uses_reversed_indices():
    manifest, rows = real_rows_and_manifest()
    models, datasets, settings = rectangular_cells(manifest, "confirmation", canonical=False)
    setting = "full"
    lookup = {(int(row["training_seed"]), int(row["dataset_seed"])): row for row in rows if row["setting_id"] == setting}
    matrix = np.array([[lookup[(model, dataset)]["w1_mean"] for dataset in datasets] for model in models])
    indices = bootstrap_indices(len(models), len(datasets), 20260930)
    reversed_indices = (len(models) - 1 - indices[0], len(datasets) - 1 - indices[1])
    assert np.allclose(bootstrap_values(matrix, indices), bootstrap_values(matrix[::-1, ::-1], reversed_indices))


def test_fresh_development_manifest_and_72_saved_rows():
    root = paper_anchor_development_root()
    manifest_path = root / "manifest.json"
    summary_paths = sorted((root / "cells").glob("*/summary.json"))
    if not manifest_path.is_file() or not summary_paths:
        pytest.skip("installed fresh paper-anchor development assets are unavailable")
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(path.read_text()) for path in summary_paths]
    models, datasets, settings = rectangular_cells(manifest, "development")
    assert models == [0, 1]
    assert datasets == list(range(800, 804))
    assert len(settings) == 9
    assert len(rows) == 72
    validate_summary_rows(rows, manifest, root.parent, "development")


def test_fresh_development_tampered_summary_is_rejected():
    root = paper_anchor_development_root()
    manifest_path = root / "manifest.json"
    summary_paths = sorted((root / "cells").glob("*/summary.json"))
    if not manifest_path.is_file() or not summary_paths:
        pytest.skip("installed fresh paper-anchor development assets are unavailable")
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(path.read_text()) for path in summary_paths]
    rows[0]["cell_id"] = int(rows[0]["cell_id"]) + 100000
    with pytest.raises(RuntimeError):
        validate_summary_rows(rows, manifest, root.parent, "development")


def test_fresh_development_tampered_dataset_is_rejected():
    root = paper_anchor_development_root()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("installed fresh paper-anchor development assets are unavailable")
    manifest = json.loads(manifest_path.read_text())
    manifest["cells"][0]["dataset_seed"] = 999999
    with pytest.raises(RuntimeError):
        rectangular_cells(manifest, "development")
