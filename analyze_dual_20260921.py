import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import simpson
import torch

from analyze_solid_20260920 import independent_population
from audit_extension_results import compare_reference, independent_metrics, reference_arrays, true_parameters
from learned_sbi import PosteriorMDN, dataset
from run_dual_20260921 import asset_name, load_npz
from run_solid_20260920 import sha256, write_json


def independent_joint(samples, weights, reference):
    grid, density, cdf = [reference[key] for key in ["grid", "density", "cdf"]]
    d = samples.shape[1]
    mean = simpson(density * grid, x=grid, axis=1)
    variance = simpson(density * (grid - mean[:, None]) ** 2, x=grid, axis=1)
    cross = []
    for i in range(d):
        for j in range(d):
            if i != j:
                value = np.dot(weights, (samples[:, i] - mean[i]) * (samples[:, j] - mean[j]))
                cross.append(value / np.sqrt(variance[i] * variance[j]))
    sign_prob = [1 - np.interp(0, grid, cdf[i]) for i in range(d)]
    tv = 0.0
    for code in range(2 ** d):
        included = np.ones(len(samples), dtype=bool)
        target = 1.0
        for i in range(d):
            positive = bool((code >> i) & 1)
            included &= ((samples[:, i] > 0) == positive)
            target *= sign_prob[i] if positive else 1 - sign_prob[i]
        tv += abs(weights[included].sum() - target)
    return dict(standardized_offdiagonal_second_rms=float(np.sqrt(np.mean(np.square(cross)))), sign_joint_tv=tv / 2)


def audit_problem(job):
    root, phase, name, cells, output = job
    root, output = Path(root), Path(output)
    asset = root / "assets" / name
    params = load_npz(asset / "parameters.npz")
    reference = reference_arrays(**params)
    ref_checks = compare_reference(asset / "reference.npz", reference)
    np.savez_compressed(output / f"{name}.npz", **reference)
    truth = None
    if (asset / "data.npz").exists():
        data = load_npz(asset / "data.npz")
        actual = load_npz(asset / "true_parameters.npz")
        independent_params = true_parameters(data["context"])
        for key, expected in zip(["variance", "means", "weights"], independent_params):
            np.testing.assert_allclose(actual[key], expected, atol=2e-14, rtol=2e-13)
        truth = reference_arrays(*independent_params)
        compare_reference(asset / "true_reference.npz", truth)
    errors = {}
    population_cache = {}
    for config in cells:
        cell = root / phase / "cells" / f"cell_{config['cell_id']:04d}"
        summary = json.loads((cell / "summary.json").read_text())
        arrays = load_npz(cell / "samples.npz")
        assert arrays["samples"].shape == (config["particles"], 8)
        actual = independent_metrics(arrays["samples"], arrays["weights"], reference)
        actual.update(independent_joint(arrays["samples"], arrays["weights"], reference))
        if truth is not None:
            actual.update({"true_" + key: value for key, value in independent_metrics(arrays["samples"], arrays["weights"], truth).items()})
        for key, value in actual.items():
            delta = abs(value - summary[key])
            tolerance = 2e-5 if "w1" in key else 4e-6
            assert delta < tolerance, (name, config["cell_id"], key, delta)
            errors[key] = max(errors.get(key, 0), delta)
        if config["steps"] not in population_cache:
            check = dict(config, method="full", groups=64, dimension=8, u_max=20.0, diffusion=1.0, sampling="without_replacement")
            population_cache[config["steps"]] = independent_population(params["variance"], check)
        saved = json.loads((cell / "certificate.json").read_text())
        independent = population_cache[config["steps"]]
        assert saved["finite_normalizer"] == independent["finite_normalizer"]
        for left, right in zip(saved["coordinates"], independent["coordinates"]):
            assert left["finite_normalizer"] == right["finite_normalizer"]
            np.testing.assert_allclose(left["smallest_denominator"], right["smallest_denominator"], rtol=1e-8, atol=1e-10)
    return dict(name=name, cells=len(cells), errors=errors, reference=ref_checks, population_configs=len(population_cache))


def validate_inputs(root, phase, manifest, training_root):
    cells = manifest["cells"]
    seen, assets, models = set(), set(), {}
    file_count = 0
    for config in cells:
        cell = root / phase / "cells" / f"cell_{config['cell_id']:04d}"
        assert (cell / "done").read_text().strip() == "completed"
        actual_config = json.loads((cell / "config.json").read_text())
        assert all(actual_config[key] == value for key, value in config.items())
        receipt = json.loads((cell / "receipt.json").read_text())
        assert receipt["status"] == "completed"
        assert all(sha256(cell / name) == value for name, value in receipt["files"].items())
        file_count += len(receipt["files"])
        summary = json.loads((cell / "summary.json").read_text())
        assert summary["status"] == "completed" and summary["seconds_including_preparation"] > 0
        seen.add(config["cell_id"])
        name = asset_name(config)
        if name in assets:
            continue
        assets.add(name)
        asset = root / "assets" / name
        receipt = json.loads((asset / "receipt.json").read_text())
        assert all(sha256(asset / key) == value for key, value in receipt["files"].items())
        file_count += len(receipt["files"])
        if config["family"] == "learned":
            t, d = config["training_seed"], config["dataset_seed"]
            checkpoint = training_root / f"training_{t}" / "final.pt"
            assert sha256(checkpoint) == manifest["checkpoints"][str(t)]
            if t not in models:
                model = PosteriorMDN()
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
                models[t] = model.eval()
            params = load_npz(asset / "parameters.npz")
            data = load_npz(asset / "data.npz")
            theta, context = dataset(64, 8, d)
            np.testing.assert_allclose(theta, data["theta"], atol=1e-14, rtol=0)
            np.testing.assert_allclose(context, data["context"], atol=1e-14, rtol=0)
            with torch.no_grad():
                v, m, logw = models[t](torch.as_tensor(context, dtype=torch.float32))
            for key, expected in [("variance", v.numpy()), ("means", m.numpy()), ("weights", logw.double().softmax(-1).numpy())]:
                np.testing.assert_allclose(params[key], expected, atol=2e-5, rtol=2e-5)
    assert len(seen) == len(cells) == len(list((root / phase / "cells").glob("*/done")))
    return dict(cells=len(cells), hashed_files=file_count, unique_parameters=len(assets), checkpoints_reloaded=len(models))


def aggregate(root, phase):
    rows = [json.loads(p.read_text()) for p in sorted((root / phase / "cells").glob("*/summary.json"))]
    settings = sorted({r["setting_id"] for r in rows})
    metrics = ["w1_mean", "true_w1_mean", "seconds_including_preparation", "sign_joint_tv",
               "standardized_offdiagonal_second_rms", "sliced_w1_32", "resampling_count"]
    grouped = {}
    for family in sorted({r["family"] for r in rows}):
        grouped[family] = {}
        for setting in settings:
            subset = [r for r in rows if r["family"] == family and r["setting_id"] == setting]
            grouped[family][setting] = dict(n=len(subset), **{key: float(np.mean([r[key] for r in subset])) for key in metrics if key in subset[0]})
    result = dict(phase=phase, settings=grouped)
    if phase != "confirmation":
        return result, rows
    trained = sorted({r["training_seed"] for r in rows})
    data = sorted({r["dataset_seed"] for r in rows})
    lookup = {(r["training_seed"], r["dataset_seed"], r["setting_id"]): r for r in rows}
    rng = np.random.default_rng(20260921)
    ii = rng.integers(0, len(trained), size=(10000, len(trained)))
    jj = rng.integers(0, len(data), size=(10000, len(data)))
    arrays, boots = {}, {}
    intervals = {}
    for setting in settings:
        intervals[setting] = {}
        for key in metrics:
            values = np.array([[lookup[t, d, setting][key] for d in data] for t in trained])
            boot = values[ii[:, :, None], jj[:, None, :]].mean((1, 2))
            arrays[setting, key], boots[setting, key] = values, boot
            intervals[setting][key] = dict(mean=float(values.mean()), ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
                                           per_training_seed=values.mean(1).tolist(),
                                           conditional_data_ci95=np.quantile(values.mean(0)[jj].mean(1), [0.025, 0.975]).tolist())
    baseline = "baseline" if "baseline" in settings else "full"
    contrasts = {}
    for setting in settings:
        if setting == baseline:
            continue
        error_boot = boots[setting, "w1_mean"] - boots[baseline, "w1_mean"]
        time_boot = boots[setting, "seconds_including_preparation"] / boots[baseline, "seconds_including_preparation"]
        error_mean = float((arrays[setting, "w1_mean"] - arrays[baseline, "w1_mean"]).mean())
        ratio_mean = float(arrays[setting, "seconds_including_preparation"].mean() / arrays[baseline, "seconds_including_preparation"].mean())
        contrasts[setting] = dict(w1_difference=error_mean, w1_difference_ci95=np.quantile(error_boot, [0.025, 0.975]).tolist(),
                                  w1_difference_upper95=float(np.quantile(error_boot, 0.95)),
                                  time_ratio=ratio_mean, time_ratio_ci95=np.quantile(time_boot, [0.025, 0.975]).tolist(),
                                  time_ratio_upper95=float(np.quantile(time_boot, 0.95)),
                                  declared_accuracy_cost_gate=bool(np.quantile(error_boot, 0.95) <= 0.002 and np.quantile(time_boot, 0.95) < 0.8))
    result.update(intervals=intervals, paired_contrasts=contrasts, bootstrap_replicates=10000,
                  unit="5 training seeds crossed with held-out datasets; paired methods")
    return result, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--phase", choices=["development", "confirmation"], required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert (args.root / args.phase / "done").exists()
    args.output.mkdir(parents=True, exist_ok=True)
    stats, rows = aggregate(args.root, args.phase)
    write_json(args.output / "statistics.json", stats)
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with (args.output / "cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if args.summary_only:
        return
    manifest = json.loads((args.root / args.phase / "manifest.json").read_text())
    inputs = validate_inputs(args.root, args.phase, manifest, args.training_root)
    references = args.output / "references"
    references.mkdir(exist_ok=True)
    jobs = []
    for name in sorted({asset_name(c) for c in manifest["cells"]}):
        cells = [c for c in manifest["cells"] if asset_name(c) == name]
        jobs.append((str(args.root), args.phase, name, cells, str(references)))
    assert 1 <= args.workers <= 4
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        checks = list(executor.map(audit_problem, jobs))
    assert sum(r["cells"] for r in checks) == len(manifest["cells"])
    maxima = {key: max(r["errors"].get(key, 0) for r in checks) for key in set().union(*(r["errors"].keys() for r in checks))}
    write_json(args.output / "audit.json", dict(status="passed", inputs=inputs, source_commit=manifest["commit"],
                                               reference_grid="131073 points on [-16,16], scipy norm.logpdf/simpson",
                                               maximum_metric_discrepancies=maxima, checks=checks,
                                               scope="independent metric/reference/population recomputation, input/checkpoint and receipt validation; no independent human review"))


if __name__ == "__main__":
    main()
