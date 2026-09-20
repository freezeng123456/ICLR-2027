import copy
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pytest
import torch

import audit_tail_bridge_study_20260921 as auditor
from nonseparable_sensor_model_20260921 import SensorProblem
from run_sensor_study_20260921 import prepare_one
from run_tail_bridge_study_20260921 import development_settings, execute, finish_cell


SOURCE = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def refresh_receipt(path, files=auditor.CELL_FILES):
    write_json(path / "receipt.json", {"files": {name: auditor._sha256(path / name) for name in files}})


@pytest.fixture(scope="module")
def fixture_directory():
    work = SOURCE / "work"
    work.mkdir(exist_ok=True)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="tail-bridge-audit-test-", dir=work) as name:
        yield Path(name)
    torch.set_num_threads(old_threads)


@pytest.fixture(scope="module")
def real_run(fixture_directory):
    # 只在测试准备阶段调用冻结生产代码，生成真实参数、精确参考和 CPU 粒子。
    root = fixture_directory / "real_run"
    root.mkdir()
    problem_config = dict(dimension=2, regime="ambiguous", dataset_seed=1400, groups=12)
    prepare_one((root, problem_config))
    asset = root / "assets" / auditor._asset_name(problem_config)
    checked, payload, reference = auditor.audit_asset(asset, problem_config)
    problem = SensorProblem.from_dict(payload)
    configs, rows = [], {}
    settings = auditor.frozen_settings()
    for index in (0, 7, 12, 15, 24, 25):
        config = dict(problem_config, **settings[index], phase="development", repeat=0, cell_id=index)
        config["particles"] = 128
        seed = 8100000 + 1000 * config["dimension"] + 10 * config["dataset_seed"]
        x, w, logz, diagnostics = execute(problem, config, seed, "cpu")
        cell = root / "development" / "cells" / f"cell_{index:04d}"
        cell.mkdir(parents=True)
        write_json(cell / "config.json", config)
        np.savez_compressed(cell / "samples.npz", samples=x, weights=w)
        write_json(cell / "diagnostics.json", diagnostics)
        write_json(cell / "gpu_processes.json", {"before": [], "after": []})
        rows[index] = finish_cell((root, config, seed, logz, diagnostics))
        configs.append(config)
    return dict(root=root, asset=asset, asset_checks=checked, payload=payload,
                reference=reference, evidence=checked["log_sensor_evidence"], configs=configs, rows=rows)


def checked_cell(real_run, path, config, expected_row=None):
    return auditor.audit_cell(path, config, real_run["payload"], real_run["reference"],
                              real_run["evidence"], "cpu", expected_row)


def cloned_cell(real_run, fixture_directory, index=0):
    config = next(c for c in real_run["configs"] if c["cell_id"] == index)
    source = real_run["root"] / "development" / "cells" / f"cell_{index:04d}"
    path = Path(tempfile.mkdtemp(prefix="mutated-cell-", dir=fixture_directory))
    shutil.copytree(source, path, dirs_exist_ok=True)
    return path, config


def test_frozen_matrix_matches_runner_without_calling_selection():
    assert auditor.frozen_settings() == development_settings()
    cells = auditor.expected_cells("development", auditor.frozen_settings())
    assert len(cells) == 416
    assert {c["dataset_seed"] for c in cells} == set(range(1400, 1404))
    assert {c["particles"] for c in cells} == {16384}
    assert len({auditor._asset_name(c) for c in cells}) == 16
    assert len(auditor._problems("confirmation")) == 80


def test_real_asset_exact_density_reference_and_rng(real_run):
    report = real_run["asset_checks"]
    assert report["passed"] and report["pass"]
    assert report["density_matches_saved_mixture"]
    assert report["receipt_strict"]["files_checked"] == 4
    assert all(v == 0 for v in report["strict_rng_max_errors"].values())
    assert set(report["reference_pair_metrics"]) == set(auditor.METRICS)


@pytest.mark.parametrize("index", [0, 7, 12, 15, 24, 25])
def test_real_cpu_cells_all_metrics_and_diagnostics(real_run, index):
    config = next(c for c in real_run["configs"] if c["cell_id"] == index)
    path = real_run["root"] / "development" / "cells" / f"cell_{index:04d}"
    report = checked_cell(real_run, path, config, real_run["rows"][index])
    assert report["passed"], report
    assert len(report["metrics"]) == 9
    assert report["diagnostics"]["passed"]
    assert report["gpu"]["gpu_verified"] is False
    if config["kind"] == "diffusion":
        assert "sensor evidence comparison not applicable" in report["normalizer"]["scope"]
    else:
        assert report["normalizer"]["accuracy_gate_applied"] is False


@pytest.mark.parametrize("metric", auditor.METRICS)
def test_every_metric_mutation_fails_even_with_fresh_hash(real_run, fixture_directory, metric):
    path, config = cloned_cell(real_run, fixture_directory)
    summary = auditor._json(path / "summary.json")
    summary[metric] += 0.1
    write_json(path / "summary.json", summary)
    refresh_receipt(path)
    report = checked_cell(real_run, path, config)
    assert not report["passed"] and f"independent metric {metric}" in report["error"]


@pytest.mark.parametrize("mutation", ["beta", "increment", "cost", "ess", "reference_pd", "oracle", "input_hash", "sign", "timing", "statistics"])
def test_bridge_diagnostic_mutations_fail(real_run, fixture_directory, mutation):
    path, config = cloned_cell(real_run, fixture_directory)
    d = auditor._json(path / "diagnostics.json")
    meta = d["metadata"]
    if mutation == "beta":
        meta["records"][-1]["beta"] = 0.99
    elif mutation == "increment":
        meta["records"][0]["log_normalizer_increment"] += 0.2
    elif mutation == "cost":
        d["counts"]["mode_factor_calls"] = 0
    elif mutation == "ess":
        meta["records"][0]["ess_fraction"] = 0.01
    elif mutation == "reference_pd":
        meta["reference_parameters"]["covariance"][0][0] = -1.0
    elif mutation == "oracle":
        meta["oracle"] = True
    elif mutation == "input_hash":
        meta["parameter_sha256"] = "0" * 64
    elif mutation == "sign":
        meta["records"][0]["sign_accepted"] += config["particles"] + 1
    elif mutation == "timing":
        d["timing"]["preparation_seconds"] = 0.0
    else:
        meta["statistics"]["weighted_mean"][0] += 0.1
    write_json(path / "diagnostics.json", d)
    refresh_receipt(path)
    report = checked_cell(real_run, path, config)
    assert not report["passed"], mutation


@pytest.mark.parametrize("mutation", ["negative", "mass", "nonfinite", "shape"])
def test_raw_weights_invalid_fails(real_run, fixture_directory, mutation):
    path, config = cloned_cell(real_run, fixture_directory)
    arrays = auditor._npz(path / "samples.npz", ("samples", "weights"))
    if mutation == "negative":
        arrays["weights"][0] = -0.1
    elif mutation == "mass":
        arrays["weights"] *= 2
    elif mutation == "nonfinite":
        arrays["samples"][0, 0] = np.nan
    else:
        arrays["weights"] = arrays["weights"][:, None]
    np.savez_compressed(path / "samples.npz", **arrays)
    refresh_receipt(path)
    assert not checked_cell(real_run, path, config)["passed"]


@pytest.mark.parametrize("mutation", ["empty", "missing", "digest", "done", "seed", "gpu", "rows", "evidence"])
def test_provenance_mutations_fail(real_run, fixture_directory, mutation):
    path, config = cloned_cell(real_run, fixture_directory)
    expected_row = None
    if mutation == "empty":
        write_json(path / "receipt.json", {})
    elif mutation == "missing":
        refresh_receipt(path, auditor.CELL_FILES - {"samples.npz"})
    elif mutation == "digest":
        with (path / "summary.json").open("a") as stream:
            stream.write(" ")
    elif mutation == "done":
        (path / "done").write_text("running\n")
    elif mutation in ("seed", "evidence"):
        summary = auditor._json(path / "summary.json")
        summary["seed" if mutation == "seed" else "log_evidence_error"] += 1
        write_json(path / "summary.json", summary)
        refresh_receipt(path)
    elif mutation == "gpu":
        write_json(path / "gpu_processes.json", {"before": ["12, python, 64 MiB"], "after": []})
        refresh_receipt(path)
    else:
        expected_row = dict(real_run["rows"][0], seconds=0.0)
    assert not checked_cell(real_run, path, config, expected_row)["passed"]


def test_fk_rejects_sensor_evidence_label(real_run, fixture_directory):
    path, config = cloned_cell(real_run, fixture_directory, index=24)
    summary = auditor._json(path / "summary.json")
    summary["log_evidence_error"] = 0.0
    write_json(path / "summary.json", summary)
    refresh_receipt(path)
    assert not checked_cell(real_run, path, config)["passed"]


def test_seed_identity_is_independent_and_strict(real_run):
    config = real_run["configs"][0]
    auditor._check_problem_seed(real_run["payload"], config)
    with pytest.raises(auditor.AuditViolation, match="seed-generated"):
        auditor._check_problem_seed(real_run["payload"], dict(config, dataset_seed=1401))


def test_real_sources_and_protocol_hashes():
    manifest = dict(source_commit="e9e35c3", sources={name: auditor._sha256(SOURCE / name) for name in auditor.SOURCE_FILES})
    assert auditor._sources(manifest, SOURCE) == manifest["sources"]
    manifest["sources"]["docs/TAIL_BRIDGE_STUDY_PROTOCOL_20260921.md"] = "0" * 64
    with pytest.raises(auditor.AuditViolation, match="source hash mismatch"):
        auditor._sources(manifest, SOURCE)


def test_incomplete_real_run_fails_closed_retains_report(real_run, fixture_directory):
    output = fixture_directory / "failed-report.json"
    args = ["--root", str(real_run["root"]), "--phase", "development", "--source", str(SOURCE),
            "--output", str(output), "--workers", "1"]
    assert auditor.main(args) == 1
    raw = output.read_bytes()
    report = auditor._json(output)
    assert report["status"] == "failed" and report["passed"] is False
    assert report["counts"]["cells_expected"] == 416
    assert report["counts"]["cells_passed"] == 0
    assert report["errors"]
    with pytest.raises(FileExistsError):
        auditor.main(args)
    assert output.read_bytes() == raw


@pytest.mark.parametrize("contents", ['{"value":NaN}', '{"value":1,"value":2}', '{"value":Infinity}'])
def test_json_nonfinite_and_duplicate_keys_fail(fixture_directory, contents):
    path = fixture_directory / "invalid.json"
    path.write_text(contents)
    with pytest.raises(auditor.AuditViolation):
        auditor._json(path)


def test_no_confirmation_when_no_candidate():
    with pytest.raises(auditor.AuditViolation, match="without eligible candidate"):
        auditor.confirmation_settings(dict(expand_confirmation=False, candidate=None))


def test_partial_rows_cannot_drive_selection(real_run):
    with pytest.raises(auditor.AuditViolation, match="rows count"):
        auditor.independent_selection(list(real_run["rows"].values()))


def test_worker_job_audits_real_asset_once_and_all_cells(real_run):
    job = (str(real_run["root"]), "development", real_run["configs"][0],
           real_run["configs"], real_run["rows"], "cpu")
    report = auditor._asset_job(job)
    assert report["passed"], report
    assert len(report["cells"]) == 6
