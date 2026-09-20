import argparse
import hashlib
import itertools
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from anchored_tail_20260921 import AnchoredTailModel
from certified_composition_torch import FactorParameters, sample as torch_sample
from composition_benchmark import measure, runtime
from composition_extension import ExtensionModel, certificate
from run_dual_20260921 import asset_name, joint_metrics, load_npz, paired_seed, prepare_asset
from run_solid_20260920 import sha256, write_json


GRID = np.linspace(math.sqrt(20.0), 0.0, 2049, dtype=np.float64) ** 2
TRAINING_SEEDS = range(5)
DATASET_SEEDS = range(700, 710)
METHODS = ("full", "tail_fixed", "tail_anchored")
PARTICLES = 32768
STEPS = 2048
BATCH = 4
GROUPS = 64
DIMENSION = 8
DIFFUSION = 1.0
ESS_FRACTION = 0.5
RANDOM_ORDER_SEED = 20260924


def confirmation_problems():
    return [dict(family="learned", training_seed=t, dataset_seed=d, repeat=0)
            for t, d in itertools.product(TRAINING_SEEDS, DATASET_SEEDS)]


def confirmation_cells():
    cells = []
    for problem, method in itertools.product(confirmation_problems(), METHODS):
        cells.append(dict(problem, method=method, particles=PARTICLES, steps=STEPS, batch=BATCH,
                          groups=GROUPS, dimension=DIMENSION,
                          u_max=20.0, diffusion=DIFFUSION, ess_fraction=ESS_FRACTION,
                          setting_id=method, line="anchored_confirmation", phase="confirmation",
                          seed=paired_seed(problem)))
    for cell_id, cell in enumerate(cells):
        cell["cell_id"] = cell_id
    return cells


def _generators(seed, device):
    return [torch.Generator(device=device).manual_seed(int(seed) + 1000003 * index) for index in range(3)]


def _resample(weights, generator):
    n = len(weights)
    points = (torch.arange(n, dtype=torch.float64, device=weights.device) + torch.rand((), dtype=torch.float64, device=weights.device, generator=generator)) / n
    cdf = weights.cumsum(0)
    cdf[-1] = 1
    return torch.searchsorted(cdf, points)


def anchored_sample(parameters, model, grid, particles, seed, batch, ess_fraction, device):
    requested_device = parameters.variance.device if device is None else torch.device(device)
    device = parameters.variance.device
    if requested_device.type != device.type or (requested_device.index is not None and requested_device.index != device.index):
        raise ValueError("parameter and requested sampling devices must match")
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all() or not (np.diff(grid) < 0).all() or (grid < 0).any():
        raise ValueError("grid must be finite, strictly decreasing, and nonnegative")
    if particles < 2 or not 2 <= batch <= parameters.groups or not 0 <= ess_fraction <= 1:
        raise ValueError("invalid particles, batch, or ESS fraction")
    if torch.device(parameters.variance.device) != device or torch.device(model.variance.device) != device:
        raise ValueError("parameter and sampling devices must match")
    motion, auxiliary, resampling = _generators(seed, device)
    x = torch.randn((particles, parameters.dimension), dtype=torch.float64, device=device, generator=motion)
    log_weights = torch.full((particles,), -math.log(particles), dtype=torch.float64, device=device)
    log_normalizer = 0.0
    records = []
    for step, (u, next_u) in enumerate(zip(grid[:-1], grid[1:])):
        h = float(u - next_u)
        noise = math.sqrt(h * DIFFUSION) * torch.randn(x.shape, dtype=torch.float64, device=device, generator=motion)
        keys = torch.rand((particles, parameters.groups), dtype=torch.float64, device=device, generator=auxiliary)
        indices = keys.topk(batch, dim=1, largest=True).indices
        drift, potential = model.estimate(x, float(u), batch, auxiliary, control=True, indices=indices)
        y = (1 - h / 2) * x + 2 * h * drift + noise
        increment = h * potential
        if not torch.isfinite(y).all() or not torch.isfinite(increment).all():
            raise FloatingPointError(f"nonfinite anchored update at step {step}")
        log_weights += increment
        normalization = torch.logsumexp(log_weights, 0)
        log_normalizer += float(normalization)
        log_weights -= normalization
        weights = log_weights.exp()
        ess = 1 / weights.square().sum()
        should_resample = bool(ess < particles * ess_fraction and step < len(grid) - 2)
        x = y
        if should_resample:
            x = x[_resample(weights, resampling)]
            log_weights.fill_(-math.log(particles))
        records.append(dict(step=step, u=float(u), h=h, ess_fraction=float(ess / particles),
                            resampled=should_resample, factor_calls=particles * batch + 7 * parameters.groups,
                            preparation_calls=7 * parameters.groups, preparation_derivative_calls=6 * parameters.groups, poisson_events=0,
                            full_particles=0, randomized_particles=particles))
    return x, weights, log_normalizer, records


def run_sampling_cell(config, asset, output, device):
    params = load_npz(asset / "parameters.npz")
    reference = load_npz(asset / "reference.npz")
    cert_model = ExtensionModel(GROUPS, DIMENSION, "learned", "cpu", "without_replacement", params)
    cert_model.control_variance = cert_model.variance
    population = certificate(cert_model, config)
    if not population["finite_normalizer"]:
        raise RuntimeError("full finite-normalizer certificate failed")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", dict(config, groups=GROUPS, dimension=DIMENSION, device=device, grid=GRID.tolist(), rng_schema="three independent generators for Torch sample; anchored uses motion/auxiliary/resampling with auxiliary also receiving fixed WOR indices"))
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))
    started = time.perf_counter()
    parameters = FactorParameters(**params)
    if config["method"] == "tail_anchored":
        model = AnchoredTailModel(GROUPS, DIMENSION, "learned", device, "without_replacement", params, GRID[:-1])
        parameters = FactorParameters(parameters.variance.to(device), parameters.means.to(device), parameters.weights.to(device))
        samples, weights, logz, records = anchored_sample(parameters, model, GRID, config["particles"], config["seed"], config["batch"], config["ess_fraction"], device)
    else:
        samples, weights, logz, records = torch_sample(parameters, GRID, config["particles"], config["seed"], method=config["method"], batch=config["batch"], device=device, return_type="torch")
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))
    samples_cpu = samples.detach().cpu().numpy()
    weights_cpu = weights.detach().cpu().numpy()
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))
    seconds = time.perf_counter() - started
    arrays = dict(samples=samples_cpu, weights=weights_cpu)
    np.savez_compressed(output / "samples.npz", **arrays)
    write_json(output / "steps.json", records)
    report = dict(config, seconds=seconds, seconds_including_preparation=seconds,
                  log_normalizer=float(logz), factor_calls=sum(row["factor_calls"] for row in records),
                  preparation_calls=sum(row["preparation_calls"] for row in records),
                  preparation_derivative_calls=sum(row.get("preparation_derivative_calls", 0) for row in records),
                  resampling_count=sum(row["resampled"] for row in records),
                  status="completed",
                  **measure(samples_cpu, weights_cpu, reference),
                  **joint_metrics(samples_cpu, weights_cpu, reference))
    if (asset / "true_reference.npz").exists():
        report.update({"true_" + key: value for key, value in measure(samples_cpu, weights_cpu, load_npz(asset / "true_reference.npz")).items()})
    if config["method"] == "tail_anchored":
        report["anchor_preparation"] = model.cost_report()
    write_json(output / "certificate.json", population)
    write_json(output / "summary.json", report)
    receipt_files = {path.name: sha256(path) for path in sorted(output.iterdir())}
    write_json(output / "receipt.json", dict(status="completed", files=receipt_files))
    (output / "done").write_text("completed\n")
    return report


def manifest(root, cells, commit, training_root):
    source_names = ["run_anchored_confirmation_20260921.py", "run_dual_20260921.py", "gaussian_integrability.py", "anchored_tail_20260921.py", "certified_composition_torch.py", "composition_benchmark.py", "composition_extension.py", "learned_sbi.py", "run_solid_20260920.py", "docs/ANCHORED_CONFIRMATION_PROTOCOL_20260921.md"]
    return dict(commit=commit, phase="confirmation", random_order_seed=RANDOM_ORDER_SEED,
                order=np.random.default_rng(RANDOM_ORDER_SEED).permutation(len(cells)).tolist(), cells=cells,
                sources={name: sha256(name) for name in source_names},
                checkpoints={str(t): sha256(training_root / f"training_{t}" / "final.pt") for t in TRAINING_SEEDS})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hours-bound", type=float, default=4.0)
    args = parser.parse_args()
    if args.hours_bound <= 0 or args.hours_bound > 4:
        raise ValueError("hours-bound must be in (0, 4]")
    torch.set_num_threads(1)
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("confirmation requires CUDA")
    started = time.perf_counter()
    args.root.mkdir(parents=True, exist_ok=False)
    phase = args.root / "confirmation"
    phase.mkdir()
    cells = confirmation_cells()
    write_json(phase / "manifest.json", manifest(args.root, cells, args.commit, args.training_root))
    write_json(args.root / "runtime.json", dict(runtime(), pid=os.getpid(), phase="confirmation", started=started))
    models = {}
    for problem in confirmation_problems():
        prepare_asset(args.root, problem, args.training_root, models)
    del models
    warm_asset = args.root / "assets" / asset_name(cells[0])
    for method in METHODS:
        warm_config = dict(cells[0], method=method, particles=128, steps=2, seed=12345)
        warm_output = args.root / "warmup" / method
        if method == "tail_anchored":
            warm_output.mkdir(parents=True, exist_ok=False)
            params = load_npz(warm_asset / "parameters.npz")
            model = AnchoredTailModel(GROUPS, DIMENSION, "learned", args.device, "without_replacement", params, GRID[:-1])
            model.estimate(torch.zeros((128, DIMENSION), dtype=torch.float64, device=args.device), float(GRID[0]), BATCH, torch.Generator(device=args.device).manual_seed(7), control=True)
        else:
            from certified_composition_torch import sample
            sample(FactorParameters(**load_npz(warm_asset / "parameters.npz")), np.array([20.0, 19.99]), 128, 12345, method=method, batch=BATCH, device=args.device, return_type="torch")
    order = manifest(args.root, cells, args.commit, args.training_root)["order"]
    for cell_id in order:
        if time.perf_counter() - started > args.hours_bound * 3600:
            write_json(args.root / "state.json", dict(status="time_limit", phase="confirmation", current_cell=cell_id, seconds=time.perf_counter() - started))
            raise SystemExit(2)
        config = cells[cell_id]
        output = phase / "cells" / f"cell_{cell_id:04d}"
        asset = args.root / "assets" / asset_name(config)
        report = run_sampling_cell(config, asset, output, args.device)
        write_json(args.root / "state.json", dict(status="running", phase="confirmation", completed=len(list((phase / "cells").glob("*/done"))), expected=len(cells)))
        print(json.dumps({"cell_id": cell_id, "method": config["method"], "w1_mean": report["w1_mean"], "seconds_including_preparation": report["seconds_including_preparation"]}), flush=True)
    if len(list((phase / "cells").glob("*/done"))) != len(cells):
        raise RuntimeError("confirmation cell count mismatch")
    (phase / "done").write_text("completed\n")
    write_json(args.root / "state.json", dict(status="completed", phase="confirmation", completed=len(cells), expected=len(cells), seconds=time.perf_counter() - started))


if __name__ == "__main__":
    main()
