import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from audit_learned_factor_study_20260921 import (ASSET_FILES, CELL_FILES, SOURCES, audit_cell, configurations,
    development_selection, expected_cells, gpu_pid, independent_metrics, input_hash, inspect_sampler_source, load_npz,
    prepare_asset, projection_grids, sha256, validate_manifest, validate_reference, validate_samples, verify_receipt)
from learned_factor_reference_20260921 import exact_mixture, metrics
from run_learned_factor_study_20260921 import (execute, finish_cell, load_bank,
    prepare_asset as prepare_runner_asset, settings)


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def renew_receipt(folder, files):
    write_json(folder / "receipt.json", dict(files={name: sha256(folder / name) for name in files}))


@pytest.fixture(scope="module")
def real_cell(tmp_path_factory):
    torch.set_num_threads(1)
    training_root = os.environ.get("ICLR_TRAINING_ROOT")
    asset_path = os.environ.get("ICLR_LEARNED_AUDIT_ASSET")
    if not training_root and not asset_path:
        pytest.skip("set ICLR_TRAINING_ROOT for the original bank or ICLR_LEARNED_AUDIT_ASSET for its seed1599 asset")
    root = tmp_path_factory.mktemp("learned-audit-real-cpu")
    target = root / "assets" / "seed_1599"
    if training_root:
        models, _ = load_bank(Path(training_root))
        prepare_runner_asset(root, 1599, models)
    else:
        source = Path(asset_path).resolve()
        verify_receipt(source, ASSET_FILES)
        target.parent.mkdir(parents=True)
        shutil.copytree(source, target)
    asset = prepare_asset(target, 1599)
    cell = dict(settings()[4], dataset_seed=1599, repeat=0, cell_id=0, particles=256, phase="development")
    seed = 9300000 + 10 * 1599
    samples, weights, logz, diagnostics = execute(asset["parameters"], cell, seed, "cpu", particles=256)
    folder = root / "development" / "cells" / "cell_0000"
    folder.mkdir(parents=True)
    write_json(folder / "config.json", dict(cell, sampler_seed=seed))
    np.savez_compressed(folder / "samples.npz", samples=samples, weights=weights)
    write_json(folder / "diagnostics.json", diagnostics)
    write_json(folder / "gpu_processes.json", dict(before=[], after=[]))
    finish_cell((root, "development", cell, logz, diagnostics))
    return root, folder, cell, asset


def test_actual_256_particle_cpu_cell_and_independent_all_projection_check(real_cell):
    _, folder, cell, asset = real_cell
    result = audit_cell(folder, cell, asset, require_gpu=False)
    assert result["w1_absolute_discrepancy"] <= result["w1_absolute_discrepancy_bound"]
    assert len(result["projected_grid_w1"]) == 32
    assert max(result["metric_absolute_errors"].values()) < 1e-12
    assert asset["report"]["references"]["learned"]["normalization_error"] < 1e-8
    assert asset["report"]["references"]["true"]["normalization_error"] < 1e-8
    assert result["gpu_pid"] is None


@pytest.mark.parametrize("field", ["sliced_w1_32", "mean_error", "covariance_error", "log_normalizer_error"])
def test_altered_real_metrics_fail_even_with_renewed_receipt(real_cell, tmp_path, field):
    _, source, cell, asset = real_cell
    folder = tmp_path / "cell_0000"
    shutil.copytree(source, folder)
    row = json.loads((folder / "summary.json").read_text())
    row[field] += .05
    write_json(folder / "summary.json", row)
    renew_receipt(folder, CELL_FILES)
    with pytest.raises(ValueError, match="mismatch|W1"):
        audit_cell(folder, cell, asset, require_gpu=False)


def test_altered_actual_file_rejected_by_receipt(real_cell, tmp_path):
    _, source, cell, asset = real_cell
    folder = tmp_path / "cell_0000"
    shutil.copytree(source, folder)
    with (folder / "summary.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_cell(folder, cell, asset, require_gpu=False)


@pytest.mark.parametrize("corruption", ["negative", "sum", "shape", "nonfinite"])
def test_corrupt_real_weights_fail_after_receipt_refresh(real_cell, tmp_path, corruption):
    _, source, cell, asset = real_cell
    folder = tmp_path / "cell_0000"
    shutil.copytree(source, folder)
    data = load_npz(folder / "samples.npz")
    if corruption == "negative":
        data["weights"][0] = -1.
    elif corruption == "sum":
        data["weights"] *= 2
    elif corruption == "shape":
        data["weights"] = data["weights"][:, None]
    else:
        data["weights"][0] = np.nan
    np.savez_compressed(folder / "samples.npz", **data)
    renew_receipt(folder, CELL_FILES)
    with pytest.raises(ValueError, match="weight|nonfinite"):
        audit_cell(folder, cell, asset, require_gpu=False)


def test_sampler_payload_hash_and_seed_tampering_rejected(real_cell, tmp_path):
    _, source, cell, asset = real_cell
    folder = tmp_path / "cell_0000"
    shutil.copytree(source, folder)
    diagnostic = json.loads((folder / "diagnostics.json").read_text())
    assert diagnostic["parameter_sha256"] == input_hash(asset["parameters"])
    diagnostic["parameter_sha256"] = "0" * 64
    write_json(folder / "diagnostics.json", diagnostic)
    renew_receipt(folder, CELL_FILES)
    with pytest.raises(ValueError, match="input hash"):
        audit_cell(folder, cell, asset, require_gpu=False)
    config = json.loads((folder / "config.json").read_text())
    config["sampler_seed"] += 1
    write_json(folder / "config.json", config)
    renew_receipt(folder, CELL_FILES)
    with pytest.raises(ValueError, match="paired sampler seed"):
        audit_cell(folder, cell, asset, require_gpu=False)


def test_actual_dataset_and_reference_tampering_rejected(real_cell, tmp_path):
    root, _, _, asset = real_cell
    target = tmp_path / "seed_1599"
    shutil.copytree(root / "assets" / "seed_1599", target)
    observed = load_npz(target / "observations.npz")
    observed["truth"][0] += .1
    np.savez_compressed(target / "observations.npz", **observed)
    renew_receipt(target, ASSET_FILES)
    with pytest.raises(ValueError, match="dataset reconstruction"):
        prepare_asset(target, 1599)
    bad = copy.deepcopy(asset["exact"])
    bad["log_normalizer"] += .01
    with pytest.raises(ValueError, match="normalization"):
        validate_reference(asset["parameters"], bad)


def test_normalized_real_gaussian_limit_grid_bound_and_zero_weight_points():
    angle = (np.arange(5) + .25) * np.pi / 5
    parameters = dict(directions=np.column_stack((np.cos(angle), np.sin(angle))), variance=np.full(5, .5),
                      means=np.zeros((5, 2)), weights=np.tile([.3, .7], (5, 1)))
    exact = exact_mixture(parameters)
    checked = validate_reference(parameters, exact)
    assert checked["normalization_error"] < 1e-12
    angle = np.arange(32) * np.pi / 32
    directions = np.column_stack((np.cos(angle), np.sin(angle)))
    grids = projection_grids(exact, directions)
    points, weights = np.array([[0., 0.], [25., -10.]]), np.array([1., 0.])
    computed = independent_metrics(points, weights, exact, directions, grids)
    analytic = float(np.mean(np.sqrt(np.einsum("ij,jk,ik->i", directions, exact["covariance"], directions))) * np.sqrt(2 / np.pi))
    assert abs(computed["grid_sliced_w1_32"] - analytic) <= computed["w1_absolute_discrepancy_bound"]
    for nodes, masses, bound, tail, spacing in grids:
        assert len(nodes) == 32769 and masses.sum() == pytest.approx(1)
        assert np.all(masses >= 0) and tail < 1e-20
        assert bound == pytest.approx(spacing / 2 + tail + 1e-9)
    production = metrics(points, weights, exact, directions)
    assert abs(production["sliced_w1_32"] - analytic) < 1e-12


def test_protocol_matrix_matches_actual_runner_and_frozen_counts():
    assert configurations() == settings()
    _, development = expected_cells("development")
    _, confirmation = expected_cells("confirmation", "prior_s0.35")
    assert len(development) == 96 and len(confirmation) == 216
    assert {r["dataset_seed"] for r in development} == set(range(1600, 1604))
    assert {r["dataset_seed"] for r in confirmation} == set(range(1700, 1712))
    assert {r["particles"] for r in development + confirmation} == {4096}
    assert len({(r["dataset_seed"], r["setting_id"], r["repeat"]) for r in development}) == 96
    with pytest.raises(ValueError):
        expected_cells("confirmation", "mixture")
    assert "payload hash" in inspect_sampler_source(ROOT)


def test_empty_gpu_receipt_cannot_certify_gpu_timing(real_cell):
    _, folder, cell, asset = real_cell
    with pytest.raises(ValueError, match="GPU process evidence"):
        audit_cell(folder, cell, asset, require_gpu=True)
    assert gpu_pid(dict(before=[], after=[]), required=False) is None


def test_cli_failure_report_is_exclusive_and_nonzero(real_cell, tmp_path):
    root, _, _, _ = real_cell
    output = tmp_path / "failed-audit"
    command = [sys.executable, "-B", str(ROOT / "audit_learned_factor_study_20260921.py"), "--root", str(root),
               "--source", str(ROOT), "--phase", "development", "--workers", "1", "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    report = json.loads((output / "audit.json").read_text())
    assert report["status"] == "failed" and report["error_type"] == "FileNotFoundError"
    frozen = (output / "audit.json").read_bytes()
    again = subprocess.run(command, capture_output=True, text=True, check=False)
    assert again.returncode != 0 and (output / "audit.json").read_bytes() == frozen


def test_finite_validation_includes_zero_weight_sample():
    with pytest.raises(ValueError, match="invalid samples"):
        validate_samples(np.array([[0., 0.], [np.nan, 0.]]), np.array([1., 0.]), 2)


@pytest.fixture
def recovered_run():
    root = Path(os.environ.get("ICLR_LEARNED_RUN_ROOT", ROOT / "work/recovered-learned-factor-20260921/run"))
    if not (root / "development/manifest.json").is_file():
        pytest.skip("completed original 96-cell run not installed; set ICLR_LEARNED_RUN_ROOT")
    source = Path(os.environ.get("ICLR_LEARNED_SOURCE", ROOT))
    return root, source


def test_actual_96_cell_manifest_receipts_and_selection(recovered_run):
    root, source = recovered_run
    manifest, rows = validate_manifest(root, source, "development")
    assert len(manifest["cells"]) == len(rows) == 96
    selected = development_selection(rows)
    saved = json.loads((root / "selection.json").read_text())
    assert selected["baseline"] == saved["baseline"]
    assert selected["expand_confirmation"] == saved["expand_confirmation"]
    for method, values in selected["means"].items():
        for key, value in values.items():
            assert value == pytest.approx(saved["means"][method][key], abs=1e-12)


@pytest.mark.parametrize("corruption", ["particles", "cohort", "duplicate", "order"])
def test_tampered_real_manifest_rejected(recovered_run, tmp_path, corruption):
    root, source = recovered_run
    manifest = json.loads((root / "development/manifest.json").read_text())
    if corruption == "particles":
        manifest["cells"][0]["particles"] = 256
    elif corruption == "cohort":
        manifest["cells"][0]["dataset_seed"] = 1599
    elif corruption == "duplicate":
        manifest["cells"][-1] = copy.deepcopy(manifest["cells"][0])
    else:
        manifest["order"][0] = manifest["order"][1]
    destination = tmp_path / "run" / "development"
    destination.mkdir(parents=True)
    write_json(destination / "manifest.json", manifest)
    with pytest.raises(ValueError, match="manifest mismatch|order mismatch"):
        validate_manifest(destination.parent, source, "development")


def test_tampered_actual_source_rejected(recovered_run, tmp_path):
    root, source = recovered_run
    destination = tmp_path / "source"
    for name in SOURCES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    with (destination / SOURCES[0]).open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="source hash mismatch"):
        validate_manifest(root, destination, "development")
