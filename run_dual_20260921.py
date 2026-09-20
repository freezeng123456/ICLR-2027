import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import wasserstein_distance
import torch

from composition_benchmark import measure, run_cell, runtime
from composition_extension import ExtensionModel, certificate
from learned_sbi import dataset, exact_parameters, load_model, predict
from run_solid_20260920 import sha256, write_json


def settings(line):
    rows = []
    if line == "current":
        for n, k in itertools.product([8192, 32768], [512, 2048]):
            rows.append(dict(method="full", particles=n, steps=k, batch=4))
        rows += [dict(method="tail", particles=8192, steps=2048, batch=m) for m in [4, 8, 16, 32]]
        rows += [dict(method="tail", particles=8192, steps=8192, batch=4),
                 dict(method="tail", particles=32768, steps=2048, batch=4)]
    else:
        base = dict(particles=8192, steps=512, batch=4)
        rows = [dict(base, method=method, table_nodes=129, eta_total=1.0)
                for method in ["full", "tail_fixed"]]
        for nodes in [65, 129]:
            rows += [dict(base, method=method, table_nodes=nodes, eta_total=1.0)
                     for method in ["surrogate_only", "surrogate_full"]]
            rows += [dict(base, method="certified", table_nodes=nodes, eta_total=eta) for eta in [1.0, 4.0]]
    for i, row in enumerate(rows):
        row["setting_id"] = f"{line}_{i:02d}"
    return rows


def problems(line, phase):
    if phase == "development":
        rows = [dict(family="learned", training_seed=t, dataset_seed=d, repeat=0)
                for t, d in itertools.product(range(2), range(200, 204))]
        if line == "explore":
            rows += [dict(family=f, training_seed=-1, dataset_seed=-1, repeat=s)
                     for f, s in itertools.product(["gaussian", "mixture", "weak_mixture"], [40, 41])]
        return rows
    data_seeds = range(300, 320) if line == "current" else range(400, 410)
    return [dict(family="learned", training_seed=t, dataset_seed=d, repeat=0)
            for t, d in itertools.product(range(5), data_seeds)]


def asset_name(problem):
    if problem["family"] == "learned":
        return f"learned_{problem['training_seed']}_{problem['dataset_seed']}"
    return problem["family"]


def paired_seed(problem):
    return 21000000 + 10000 * (problem["training_seed"] + 1) + 10 * (problem["dataset_seed"] + 1) + problem["repeat"]


def prepare_asset(root, problem, training_root, models):
    asset = root / "assets" / asset_name(problem)
    if (asset / "receipt.json").exists():
        receipt = json.loads((asset / "receipt.json").read_text())
        assert all(sha256(asset / name) == digest for name, digest in receipt["files"].items())
        return asset
    asset.mkdir(parents=True, exist_ok=False)
    if problem["family"] == "learned":
        t = problem["training_seed"]
        if t not in models:
            models[t] = load_model(training_root / f"training_{t}" / "final.pt", "cpu")
        theta, context = dataset(64, 8, problem["dataset_seed"])
        params = predict(models[t], context)
        model = ExtensionModel(64, 8, "learned", "cpu", parameters=params)
        np.savez_compressed(asset / "data.npz", theta=theta, context=context)
        truth = exact_parameters(context)
        true_model = ExtensionModel(64, 8, "true", "cpu", parameters=truth)
        np.savez_compressed(asset / "true_parameters.npz", **truth)
        np.savez_compressed(asset / "true_reference.npz", **true_model.reference())
    else:
        model = ExtensionModel(64, 8, problem["family"], "cpu")
        params = {key: getattr(model, key).numpy() for key in ["variance", "means", "weights"]}
    np.savez_compressed(asset / "parameters.npz", **params)
    np.savez_compressed(asset / "reference.npz", **model.reference())
    write_json(asset / "receipt.json", dict(files={path.name: sha256(path) for path in sorted(asset.iterdir())}))
    return asset


def load_npz(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def joint_metrics(samples, weights, reference):
    grid, density, cdf = [reference[key] for key in ["grid", "density", "cdf"]]
    dimension = samples.shape[1]
    mean = trapezoid(density * grid, grid, axis=1)
    variance = trapezoid(density * (grid[None] - mean[:, None]) ** 2, grid, axis=1)
    centered = (samples - mean) / np.sqrt(variance)
    second = (centered * weights[:, None]).T @ centered
    mask = ~np.eye(dimension, dtype=bool)
    sign_probability = np.array([1 - np.interp(0, grid, cc) for cc in cdf])
    codes = np.arange(2 ** dimension)
    bits = ((codes[:, None] >> np.arange(dimension)) & 1).astype(bool)
    exact_sign = np.prod(np.where(bits, sign_probability, 1 - sign_probability), axis=1)
    sample_codes = ((samples > 0) * (2 ** np.arange(dimension))).sum(axis=1)
    empirical_sign = np.bincount(sample_codes, weights=weights, minlength=2 ** dimension)
    rng = np.random.default_rng(20260921)
    reference_samples = np.column_stack([np.interp(rng.random(32768), cc, grid) for cc in cdf])
    directions = rng.normal(size=(32, dimension))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    projected_w1 = [wasserstein_distance(samples @ direction, reference_samples @ direction, u_weights=weights)
                    for direction in directions]
    return dict(standardized_offdiagonal_second_rms=float(np.sqrt(np.mean(second[mask] ** 2))),
                sign_joint_tv=float(np.sum(abs(empirical_sign - exact_sign)) / 2),
                sliced_w1_32=float(np.mean(projected_w1)),
                sliced_reference_samples=32768, projection_seed=20260921)


class AssetModel(ExtensionModel):
    def __init__(self, *args, reference, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved_reference = reference

    def reference(self, points=65537, bound=12.0):
        return self.saved_reference


def source_hashes(line):
    files = ["run_dual_20260921.py", "composition_benchmark.py", "composition_extension.py", "learned_sbi.py",
             "run_solid_20260920.py", "gaussian_integrability.py", "docs/DUAL_PROTOCOL_20260921.md"]
    if line == "explore":
        files += ["certified_composition_torch.py"]
    return {name: sha256(name) for name in files}


def run_one(config, output, asset, device):
    params = load_npz(asset / "parameters.npz")
    reference = load_npz(asset / "reference.npz")
    model = AssetModel(64, 8, config["family"], device, "without_replacement", params, reference=reference)
    full_config = dict(config, groups=64, dimension=8, u_max=20.0, diffusion=1.0, ess_threshold=0.5,
                       sampling="without_replacement", device=device)
    cert_config = dict(full_config, method="full")
    population = certificate(model, cert_config)
    assert population["finite_normalizer"], population
    if config["line"] == "current":
        report = run_cell(full_config, output, model)
        arrays = load_npz(output / "samples.npz")
        report["seconds_including_preparation"] = report["seconds"]
    else:
        from certified_composition_torch import FactorParameters, sample

        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "config.json", full_config)
        parameters = FactorParameters(**params)
        grid = np.linspace(math.sqrt(20), 0, config["steps"] + 1) ** 2
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        samples, weights, logz, records = sample(parameters, grid, config["particles"], config["seed"],
                                                method=config["method"], eta_total=config["eta_total"],
                                                cost_fraction=0.25, batch=config["batch"],
                                                table_nodes=config["table_nodes"], table_radius=8.0, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        samples = samples.detach().cpu().numpy() if torch.is_tensor(samples) else np.asarray(samples)
        weights = weights.detach().cpu().numpy() if torch.is_tensor(weights) else np.asarray(weights)
        arrays = dict(samples=samples, weights=weights)
        np.savez_compressed(output / "samples.npz", **arrays)
        write_json(output / "steps.json", records)
        report = dict(config, seconds=seconds, seconds_including_preparation=seconds,
                      log_normalizer=float(logz), **measure(samples, weights, reference),
                      factor_calls=sum(row["factor_calls"] for row in records),
                      preparation_calls=sum(row["preparation_calls"] for row in records),
                      resampling_count=sum(row["resampled"] for row in records),
                      poisson_events=sum(row["poisson_events"] for row in records),
                      full_particles=sum(row["full_particles"] for row in records),
                      randomized_particles=sum(row["randomized_particles"] for row in records),
                      minimum_ess_fraction=min(row["ess_fraction"] for row in records), status="completed")
        np.savez_compressed(output / "reference.npz", **reference)
    samples, weights = arrays["samples"], arrays["weights"]
    assert np.isfinite(samples).all() and np.isfinite(weights).all() and (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-10)
    report.update(joint_metrics(samples, weights, reference))
    if (asset / "true_reference.npz").exists():
        truth = measure(samples, weights, load_npz(asset / "true_reference.npz"))
        report.update({"true_" + key: value for key, value in truth.items()})
    write_json(output / "certificate.json", population)
    write_json(output / "summary.json", report)
    write_json(output / "receipt.json", dict(status="completed", files={p.name: sha256(p) for p in sorted(output.iterdir()) if p.name != "done"}))
    (output / "done").write_text("completed\n")
    return report


def choose(root, line):
    rows = [json.loads(p.read_text()) for p in sorted((root / "development" / "cells").glob("*/summary.json"))]
    rows = [r for r in rows if r["family"] == "learned"]
    configurations = settings(line)
    baseline = next(c for c in configurations if c["method"] == "full" and
                    (line == "explore" or (c["particles"] == 32768 and c["steps"] == 2048)))
    means = {}
    for config in configurations:
        subset = [r for r in rows if r["setting_id"] == config["setting_id"]]
        assert len(subset) == 8
        means[config["setting_id"]] = dict(w1=float(np.mean([r["w1_mean"] for r in subset])),
                                           seconds=float(np.mean([r["seconds_including_preparation"] for r in subset])))
    baseline_error = means[baseline["setting_id"]]["w1"]
    candidates = [c for c in configurations if c["method"] == ("tail" if line == "current" else "certified")]
    eligible = [c for c in candidates if means[c["setting_id"]]["w1"] <= baseline_error + 0.002]
    chosen = min(eligible, key=lambda c: means[c["setting_id"]]["seconds"]) if eligible else min(candidates, key=lambda c: means[c["setting_id"]]["w1"])
    return dict(selected=chosen, development_eligible=bool(eligible), baseline=baseline, setting_means=means,
                selection_data="training seeds 0,1; dataset seeds 200..203", criterion="mean W1 <= baseline+0.002, then minimum mean seconds")


def confirmation_settings(line, selection):
    selected = dict(selection["selected"])
    if line == "current":
        rows = [dict(method="full", particles=32768, steps=2048, batch=4, setting_id="baseline"),
                dict(method="tail", particles=32768, steps=2048, batch=4, setting_id="original_tail"),
                dict(selected, setting_id="selected")]
    else:
        rows = [dict(selected, method=m, particles=32768, steps=2048, setting_id=m)
                for m in ["full", "tail_fixed", "surrogate_only", "surrogate_full", "certified"]]
    return rows


def execute_phase(args, phase, configs):
    phase_root = args.root / phase
    if (phase_root / "done").exists():
        return
    phase_root.mkdir(parents=True, exist_ok=True)
    manifest_path = phase_root / "manifest.json"
    cells = [dict(config, **problem, line=args.line, phase=phase, seed=paired_seed(problem))
             for problem, config in itertools.product(problems(args.line, phase), configs)]
    for i, config in enumerate(cells):
        config["cell_id"] = i
    manifest = dict(commit=args.commit, sources=source_hashes(args.line), cells=cells,
                    order=np.random.default_rng(20260921).permutation(len(cells)).tolist(),
                    checkpoints={str(t): sha256(args.training_root / f"training_{t}" / "final.pt") for t in range(5)})
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        write_json(manifest_path, manifest)
    models = {}
    for problem in problems(args.line, phase):
        prepare_asset(args.root, problem, args.training_root, models)
    if args.mode == "prepare":
        return
    del models
    write_json(phase_root / "runtime.json", dict(runtime(), pid=os.getpid()))
    if args.line == "explore":
        from certified_composition_torch import FactorParameters, sample

        params = load_npz(args.root / "assets" / asset_name(cells[0]) / "parameters.npz")
        for method in sorted({c["method"] for c in cells}):
            sample(FactorParameters(**params), np.array([20.0, 19.99]), 128, 19, method=method, device=args.device)
    else:
        params = load_npz(args.root / "assets" / asset_name(cells[0]) / "parameters.npz")
        model = ExtensionModel(64, 8, "learned", args.device, "without_replacement", params)
        model.control_variance = model.variance
        generator = torch.Generator(device=args.device).manual_seed(9182)
        x = torch.randn(256, 8, dtype=torch.float64, device=args.device, generator=generator)
        for u in [20.0, 2.0, 0.0]:
            model.exact(x, u)
            for batch in [4, 8, 16, 32]:
                model.estimate(x, u, batch, generator, control=True)
    for cell_id in manifest["order"]:
        config = cells[cell_id]
        output = phase_root / "cells" / f"cell_{cell_id:04d}"
        if (output / "done").exists():
            receipt = json.loads((output / "receipt.json").read_text())
            assert all(sha256(output / name) == digest for name, digest in receipt["files"].items())
            continue
        if time.perf_counter() - args.started > args.hours * 3600:
            write_json(args.root / "state.json", dict(status="time_limit", phase=phase, next_cell=cell_id))
            raise SystemExit(2)
        write_json(args.root / "state.json", dict(status="running", phase=phase, current_cell=cell_id,
                                                  expected=len(cells), completed=len(list((phase_root / "cells").glob("*/done")))))
        report = run_one(config, output, args.root / "assets" / asset_name(config), args.device)
        print(json.dumps({key: report[key] for key in ["cell_id", "phase", "setting_id", "w1_mean", "seconds_including_preparation"]}), flush=True)
    count = len(list((phase_root / "cells").glob("*/done")))
    assert count == len(cells)
    write_json(phase_root / "state.json", dict(status="completed", completed=count, expected=count))
    (phase_root / "done").write_text("completed\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--line", choices=["current", "explore"], required=True)
    parser.add_argument("--mode", choices=["prepare", "run"], default="run")
    parser.add_argument("--phase", choices=["development", "confirmation", "all"], default="all")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hours", type=float, default=4)
    args = parser.parse_args()
    args.started = time.perf_counter()
    torch.set_num_threads(1)
    args.root.mkdir(parents=True, exist_ok=True)
    if args.phase in ["development", "all"]:
        execute_phase(args, "development", settings(args.line))
    if args.mode == "prepare":
        return
    selection_path = args.root / "selection.json"
    if args.phase in ["confirmation", "all"]:
        assert (args.root / "development" / "done").exists()
        selection = choose(args.root, args.line)
        if selection_path.exists():
            assert json.loads(selection_path.read_text()) == selection
        else:
            write_json(selection_path, selection)
        execute_phase(args, "confirmation", confirmation_settings(args.line, selection))
    write_json(args.root / "state.json", dict(status="completed", phase=args.phase,
                                              seconds=time.perf_counter() - args.started))


if __name__ == "__main__":
    main()
