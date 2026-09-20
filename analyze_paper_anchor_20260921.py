import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path

import numpy as np
import torch

from analyze_dual_20260921 import audit_problem, validate_inputs
from run_dual_20260921 import asset_name, load_npz
from run_solid_20260920 import sha256, write_json


BOOTSTRAPS = 10000
W1_MARGIN = 0.002
TIME_RATIO_LIMIT = 0.8


def completed_phase(root, phase):
    marker = root / phase / "done"
    if not marker.is_file() or marker.read_text() != "completed\n":
        raise RuntimeError(f"invalid phase completion marker: {marker}")


def source_checks(source, manifest):
    checks = {name: {"expected": digest, "actual": sha256(source / name), "passed": sha256(source / name) == digest}
              for name, digest in manifest["sources"].items()}
    if not checks or not all(item["passed"] for item in checks.values()):
        raise RuntimeError("source archive hash validation failed")
    return checks


def rectangular_cells(manifest, phase, canonical=True):
    cells = manifest["cells"]
    keys = [(int(c["training_seed"]), int(c["dataset_seed"]), c["setting_id"]) for c in cells]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"duplicate {phase} cell key")
    models = sorted({key[0] for key in keys})
    datasets = sorted({key[1] for key in keys})
    settings = sorted({key[2] for key in keys})
    expected = {(model, dataset, setting) for model in models for dataset in datasets for setting in settings}
    if set(keys) != expected:
        raise RuntimeError(f"nonrectangular {phase} cell set")
    if not canonical:
        return models, datasets, settings
    expected_models = [0, 1] if phase == "development" else [0, 1, 2, 3, 4]
    expected_datasets = list(range(800, 804)) if phase == "development" else list(range(900, 920))
    expected_settings = (["full", "factorized_full"] + [f"anchor_n{particles}_k{steps}_m{batch}"
                       for particles, steps, batch in [(8192, 1024, 4), (8192, 1024, 8), (8192, 1024, 16),
                                                       (32768, 2048, 4), (32768, 2048, 8), (32768, 2048, 16),
                                                       (32768, 4096, 4)]]) if phase == "development" else None
    if models != expected_models or datasets != expected_datasets:
        raise RuntimeError(f"{phase} training or dataset seed cohort is not canonical")
    if expected_settings is not None and settings != sorted(expected_settings):
        raise RuntimeError("development settings are not the nine canonical settings")
    if phase == "confirmation" and len(settings) != 4 or phase == "confirmation" and not {"full", "factorized_full"}.issubset(settings):
        raise RuntimeError("confirmation settings are not the four frozen settings")
    return models, datasets, settings


def metric_names(rows):
    preferred = ["w1_mean", "true_w1_mean", "seconds_including_preparation", "sign_joint_tv",
                 "standardized_offdiagonal_second_rms", "sliced_w1_32", "resampling_count"]
    return [name for name in preferred if rows and name in rows[0]]


def bootstrap_indices(model_count, data_count, seed):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, model_count, size=(BOOTSTRAPS, model_count)),
            rng.integers(0, data_count, size=(BOOTSTRAPS, data_count)))


def bootstrap_values(matrix, indices):
    models, datasets = indices
    return matrix[models[:, :, None], datasets[:, None, :]].mean(axis=(1, 2))


def paired_summary(values, indices):
    bootstrap = bootstrap_values(values, indices)
    return {"mean": float(values.mean()), "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "per_training_seed": values.mean(axis=1).tolist(),
            "per_dataset_seed": values.mean(axis=0).tolist()}


def setting_matrices(rows, models, datasets, setting, metrics):
    selected = [row for row in rows if row["setting_id"] == setting]
    lookup = {(int(row["training_seed"]), int(row["dataset_seed"])): row for row in selected}
    return {metric: np.array([[lookup[(model, data)][metric] for data in datasets] for model in models], dtype=float)
            for metric in metrics}


def aggregate(rows, phase, manifest, canonical=True):
    models, datasets, settings = rectangular_cells(manifest, phase, canonical=canonical)
    metrics = metric_names(rows)
    indices = bootstrap_indices(len(models), len(datasets), 20260930)
    result = {"phase": phase, "training_seeds": models, "dataset_seeds": datasets, "settings": settings,
              "metrics": metrics, "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": 20260930, "by_setting": {}}
    matrices = {}
    for setting in settings:
        matrices[setting] = setting_matrices(rows, models, datasets, setting, metrics)
        result["by_setting"][setting] = {metric: paired_summary(matrices[setting][metric], indices) for metric in metrics}
    baselines = [setting for setting in ("full", "factorized_full") if setting in settings]
    anchors = [setting for setting in settings if setting.startswith("anchor_")]
    result["paired_anchor_comparisons"] = {}
    for anchor in anchors:
        result["paired_anchor_comparisons"][anchor] = {}
        for baseline in baselines:
            result["paired_anchor_comparisons"][anchor][baseline] = {}
            for metric in metrics:
                difference = matrices[anchor][metric] - matrices[baseline][metric]
                contrast_indices = bootstrap_indices(len(models), len(datasets), 20260930)
                difference_bootstrap = bootstrap_values(difference, contrast_indices)
                item = {"difference": float(difference.mean()),
                        "difference_ci95": np.quantile(difference_bootstrap, [0.025, 0.975]).tolist(),
                        "difference_upper95": float(np.quantile(difference_bootstrap, 0.95)),
                        "per_training_seed": difference.mean(axis=1).tolist(),
                        "per_dataset_seed": difference.mean(axis=0).tolist()}
                if metric == "seconds_including_preparation":
                    anchor_bootstrap = bootstrap_values(matrices[anchor][metric], contrast_indices)
                    baseline_bootstrap = bootstrap_values(matrices[baseline][metric], contrast_indices)
                    ratio_bootstrap = anchor_bootstrap / baseline_bootstrap
                    item.update(time_ratio=float(matrices[anchor][metric].mean() / matrices[baseline][metric].mean()),
                                time_ratio_ci95=np.quantile(ratio_bootstrap, [0.025, 0.975]).tolist(),
                                time_ratio_upper95=float(np.quantile(ratio_bootstrap, 0.95)),
                                per_cell_time_ratio_mean=float(np.mean(matrices[anchor][metric] / matrices[baseline][metric])))
                result["paired_anchor_comparisons"][anchor][baseline][metric] = item
            w1 = result["paired_anchor_comparisons"][anchor][baseline].get("w1_mean", {})
            time = result["paired_anchor_comparisons"][anchor][baseline].get("seconds_including_preparation", {})
            result["paired_anchor_comparisons"][anchor][baseline]["accuracy_cost_gate"] = bool(
                w1.get("difference_upper95", np.inf) <= W1_MARGIN and time.get("time_ratio_upper95", np.inf) < TIME_RATIO_LIMIT)
    return result


def write_cells(rows, output):
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (output / "cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_summary_rows(rows, manifest, root, phase):
    if len(rows) != len(manifest["cells"]):
        raise RuntimeError("summary and manifest cell count differ")
    config_fields = ["method", "particles", "steps", "batch", "table_nodes", "eta_total", "setting_id",
                     "family", "training_seed", "dataset_seed", "repeat", "line", "phase", "seed", "cell_id"]
    for row, config in zip(rows, sorted(manifest["cells"], key=lambda item: item["cell_id"])):
        if any(row.get(field) != config.get(field) for field in config_fields):
            raise RuntimeError(f"summary/config mismatch for cell {config['cell_id']}")
        cell_path = root / phase / "cells" / f"cell_{config['cell_id']:04d}"
        if not (cell_path / "samples.npz").is_file():
            if phase == "development":
                continue
            raise RuntimeError(f"sample file is missing for cell {config['cell_id']}")
        arrays = load_npz(cell_path / "samples.npz")
        if arrays["samples"].shape != (config["particles"], 8) or arrays["weights"].shape != (config["particles"],):
            raise RuntimeError(f"sample or weight shape mismatch for cell {config['cell_id']}")
        if not np.isfinite(arrays["samples"]).all() or not np.isfinite(arrays["weights"]).all() or (arrays["weights"] < 0).any():
            raise RuntimeError(f"nonfinite or negative samples/weights for cell {config['cell_id']}")
        np.testing.assert_allclose(arrays["weights"].sum(), 1.0, atol=1e-10, rtol=1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--phase", choices=["development", "confirmation"], required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    torch.set_num_threads(1)
    completed_phase(args.root, args.phase)
    manifest = json.loads((args.root / args.phase / "manifest.json").read_text())
    source = source_checks(args.source, manifest)
    rectangular_cells(manifest, args.phase)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(path.read_text()) for path in sorted((args.root / args.phase / "cells").glob("*/summary.json"))]
    validate_summary_rows(rows, manifest, args.root, args.phase)
    statistics = aggregate(rows, args.phase, manifest)
    write_json(args.output / "statistics.json", statistics)
    write_cells(rows, args.output)
    inputs = validate_inputs(args.root, args.phase, manifest, args.training_root)
    references = args.output / "references"
    references.mkdir(exist_ok=True)
    jobs = []
    for name in sorted({asset_name(cell) for cell in manifest["cells"]}):
        cells = [cell for cell in manifest["cells"] if asset_name(cell) == name]
        jobs.append((str(args.root), args.phase, name, cells, str(references)))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        checks = list(executor.map(audit_problem, jobs))
    if sum(check["cells"] for check in checks) != len(manifest["cells"]):
        raise RuntimeError("reference audit did not cover every cell")
    maxima = {key: max(check["errors"].get(key, 0.0) for check in checks)
              for key in set().union(*(check["errors"].keys() for check in checks))}
    write_json(args.output / "audit.json", {"status": "passed", "phase": args.phase, "source_commit": manifest["commit"], "source_hashes": source,
        "inputs": inputs, "manifest_sha256": sha256(args.root / args.phase / "manifest.json"),
        "source_archive_runner_sha256": sha256(args.source / "run_paper_anchor_20260921.py"),
        "reference_grid": "131073 points, independently reconstructed with SciPy", "maximum_metric_discrepancies": maxima,
        "checks": checks, "scope": "local source, receipt, checkpoint, dataset, reference, and metric audit"})
    if args.phase == "development":
        selection_path = args.root / "selection.json"
        if not selection_path.is_file():
            raise RuntimeError("development selection is missing")
        selection = json.loads(selection_path.read_text())
        observed = statistics["by_setting"]
        selected_id = selection.get("selected", {}).get("setting_id")
        if selected_id not in observed:
            raise RuntimeError("development selection does not match manifest observations")
        for setting_id, values in selection.get("setting_means", {}).items():
            if setting_id not in observed:
                raise RuntimeError("development selection contains an unknown setting")
            for metric_name, observed_name in [("w1", "w1_mean"), ("seconds", "seconds_including_preparation")]:
                if not np.isclose(values[metric_name], observed[setting_id][observed_name]["mean"], atol=1e-12, rtol=0):
                    raise RuntimeError("development selection observations do not match saved summaries")
        write_json(args.output / "statistics.json", {**statistics, "selection_check": "passed", "observed_selection": selection})
        print(json.dumps({"status": "passed", "phase": args.phase, "selection": selection}, sort_keys=True))
    else:
        selection_path = args.root / "selection.json"
        if not selection_path.is_file():
            raise RuntimeError("frozen development selection is missing for confirmation audit")
        primary = json.loads(selection_path.read_text())
        if primary.get("selected", {}).get("setting_id") not in statistics["settings"]:
            raise RuntimeError("frozen primary selection is absent from confirmation settings")
        if "full" not in statistics["settings"] or "factorized_full" not in statistics["settings"]:
            raise RuntimeError("confirmation lacks required full baselines")
        statistics["frozen_primary_selection"] = primary
        statistics["frozen_primary_gate_status"] = {anchor: {base: values["accuracy_cost_gate"]
            for base, values in baselines.items()} for anchor, baselines in statistics["paired_anchor_comparisons"].items()}
        write_json(args.output / "statistics.json", statistics)
        print(json.dumps({"status": "passed", "phase": args.phase, "primary_selection": primary}, sort_keys=True))


if __name__ == "__main__":
    main()
