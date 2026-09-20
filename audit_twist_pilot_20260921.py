import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from audit_sensor_study_20260921 import _audit_asset, _audit_cell, _matrix_certificate, _read_json, _sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    phase = args.root / "development"
    if (phase / "done").read_text().strip() != "completed":
        raise RuntimeError("pilot has not completed")
    manifest = _read_json(phase / "manifest.json")
    configurations = manifest["cells"]
    settings = {"twist_mean", "twist_anchor", "twist_mirror", "twist_mirror_fine", "twist_mirror_batch8", "full", "smc"}
    expected = set(itertools.product([2, 8], ["ambiguous", "regular"], [1300, 1301], settings))
    observed = {(c["dimension"], c["regime"], c["dataset_seed"], c["setting_id"]) for c in configurations}
    if observed != expected or len(configurations) != 56:
        raise ValueError("pilot cohort differs from frozen protocol")
    sources = {key: _sha256(args.source / key) == digest for key, digest in manifest["sources"].items()}
    if not sources or not all(sources.values()):
        raise RuntimeError("pilot source hash mismatch")
    cell_paths = [phase / "cells" / f"cell_{c['cell_id']:04d}" for c in configurations]
    if set(cell_paths) != set((phase / "cells").glob("cell_*")):
        raise RuntimeError("unexpected cell directories")
    for c, cell in zip(configurations, cell_paths):
        if _read_json(cell / "config.json") != c:
            raise RuntimeError("configuration mismatch")
    assets = sorted((args.root / "assets").glob("*"))
    with ProcessPoolExecutor(max_workers=4) as pool:
        asset_checks = list(pool.map(_audit_asset, assets))
        cell_checks = list(pool.map(_audit_cell, cell_paths))
    if len(assets) != 8 or not all(r["pass"] for r in asset_checks + cell_checks):
        raise RuntimeError("independent posterior or metric check failed")
    matrix_checks = []
    for asset in assets:
        for steps in [1024, 2048]:
            check = _matrix_certificate(_read_json(asset / "problem.json"), steps)
            if check["status"] != "finite":
                raise RuntimeError("Gaussian reference is not in the strict finite domain")
            matrix_checks.append(dict(asset=asset.name, **check))
    rows = []
    for cell in cell_paths:
        row = _read_json(cell / "summary.json")
        with np.load(cell / "samples.npz") as arrays:
            if row["kind"] == "twist":
                logs, weights = arrays["log_weights"], arrays["weights"]
                np.testing.assert_allclose(np.exp(logs - logsumexp(logs)), weights, atol=1e-12, rtol=1e-10)
                np.testing.assert_allclose(logsumexp(logs) - np.log(len(logs)), row["log_normalizer"], atol=1e-11)
            np.testing.assert_allclose(1 / np.square(arrays["weights"]).sum() / row["particles"], row["ess_fraction"], atol=1e-12)
        gpu = _read_json(cell / "gpu_processes.json")
        if manifest["device"] == "cuda" and (len(gpu["before"]) != 1 or len(gpu["after"]) != 1):
            raise RuntimeError("GPU timing was not isolated")
        rows.append(row)
    means = {name: {key: float(np.mean([r[key] for r in rows if r["setting_id"] == name])) for key in
        ["sliced_w1_32", "seconds", "ess_fraction"]} for name in settings}
    selection = _read_json(args.root / "selection.json")
    for name in settings:
        for key in means[name]:
            np.testing.assert_allclose(means[name][key], selection["means"][name][key], atol=1e-12, rtol=0)
    eligible = sorted(name for name in settings if name.startswith("twist_") and name != "twist_mirror_batch8"
                      and means[name]["sliced_w1_32"] <= means["full"]["sliced_w1_32"] + .005
                      and means[name]["seconds"] < .8 * means["full"]["seconds"])
    if eligible != sorted(selection["eligible"]) or bool(eligible) != selection["expand_confirmation"]:
        raise RuntimeError("pilot selection rule mismatch")
    result = dict(status="passed", source_commit=manifest["commit"], manifest_sha256=_sha256(phase / "manifest.json"),
        auditor_sha256=_sha256(Path(__file__)), helper_sha256=_sha256(Path(__file__).with_name("audit_sensor_study_20260921.py")),
        sources=sources, assets=asset_checks, cells=cell_checks, matrix_certificates=matrix_checks,
        means=means, eligible=eligible, scope="independent fixed-cohort development audit; no held-out confirmation claim")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status="passed", assets=8, cells=56, eligible=eligible)), flush=True)


if __name__ == "__main__":
    main()
