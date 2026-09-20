import hashlib
import json

import numpy as np

from audit_sensor_study_20260921 import audit
from nonseparable_sensor_model_20260921 import make_problem


def _write_asset(root):
    asset = root / "assets" / "d2_ambiguous_17"
    asset.mkdir(parents=True)
    problem = make_problem(17, groups=3, dimension=2, regime="ambiguous")
    asset.joinpath("problem.json").write_text(problem.to_json() + "\n")
    weights, means, covariance = problem.exact_posterior()
    np.savez(asset / "exact.npz", probabilities=weights, means=means, covariance=covariance)
    samples = problem.sample(np.random.default_rng(99), 512)
    directions = np.array([[1.0, 0.0], [0.0, 1.0]])
    mean = weights @ means
    total_covariance = covariance + (means - mean).T @ ((means - mean) * weights[:, None])
    np.savez(asset / "reference.npz", samples=samples, directions=directions,
             mean=mean, covariance=total_covariance)
    files = {}
    for path in sorted(asset.iterdir()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[path.name] = digest
    asset.joinpath("receipt.json").write_text(json.dumps({"files": files}) + "\n")
    return asset


def test_real_generated_asset_audit(tmp_path):
    root = tmp_path / "study"
    asset = _write_asset(root)
    report_path = tmp_path / "audit.json"
    report = audit(root, report_path, assets_only=True)
    assert report["status"] == "partial"
    assert report["counts"]["assets"] == 1
    assert report["counts"]["assets_exact_pass"] == 1
    assert report["assets"][0]["receipt"]["status"] == "ok"
    assert report["assets"][0]["density_finite"]
    assert report_path.exists()


def test_cell_counts_and_missing_output_are_explicit(tmp_path):
    root = tmp_path / "study"
    _write_asset(root)
    cell = root / "confirmation" / "cells" / "cell_0000"
    cell.mkdir(parents=True)
    report = audit(root, tmp_path / "audit.json")
    assert report["counts"]["cells"] == 1
    assert report["cells"][0]["status"] == "missing_outputs"


def test_empty_receipt_is_not_a_pass(tmp_path):
    root = tmp_path / "study"
    asset = _write_asset(root)
    asset.joinpath("receipt.json").write_text(json.dumps({"files": {}}) + "\n")
    report = audit(root, tmp_path / "audit.json", assets_only=True)
    assert report["assets"][0]["receipt"]["status"] == "empty"
    assert report["assets"][0]["pass"] is False
