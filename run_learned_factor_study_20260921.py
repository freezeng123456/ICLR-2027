import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from learned_factor_bridge_20260921 import sample
from learned_factor_reference_20260921 import exact_mixture, metrics, model_error
from learned_sbi import PosteriorMDN, exact_parameters, predict
from run_solid_20260920 import sha256, write_json
from run_tail_bridge_study_20260921 import gpu_processes


SOURCES = ["learned_factor_bridge_20260921.py", "learned_factor_reference_20260921.py",
           "run_learned_factor_study_20260921.py", "tail_bridge_smc_20260921.py", "learned_sbi.py",
           "docs/LEARNED_FACTOR_STUDY_PROTOCOL_20260921.md"]
CHECKPOINT_SHA256 = [
    "ce9344dce5f56b4cda176bcdb36371f48911618aa600da1465617419b4995b69",
    "642c9ac8b37cc879a46c5dc332c9d99eb4ae9d0bd2c4b38a1ce3bbe1af22951a",
    "4ad56f1f2de8ac7e99a997dc798fcc9a82d5e41bcaa01483fa8161ba24fa2737",
    "40d2f60f7b01b721909df18cdd26cc0bda1e28ef62dd47bb3bc9b7acfaa1fbc4",
    "4f8f7d024dee2ba1d205fcf59f39c4f32d87524ab703a9051d4ad4e7ca38f59f",
]


def settings():
    rows = [dict(setting_id=f"prior_s{scale:g}", reference="prior", proposal_scale=scale,
                 global_probability=0., direct_is=False, role="baseline") for scale in [.15, .35, .65]]
    rows += [dict(setting_id=name, reference=ref, proposal_scale=.5, global_probability=refresh,
                  direct_is=direct, role="candidate" if name == "mixture" else "diagnostic")
             for name, ref, refresh, direct in [("gaussian", "gaussian", 0., False),
                 ("mixture", "mixture", 0., False), ("mixture_direct_is", "mixture", 0., True),
                 ("mixture_global", "mixture", .1, False)]]
    rows.append(dict(setting_id="exact_enumeration", reference="exact", proposal_scale=.5,
                     global_probability=0., direct_is=False, role="diagnostic"))
    return rows


def load_bank(training_root):
    models, receipt = [], []
    started = time.perf_counter()
    for index in range(5):
        folder = training_root / f"training_{index}"
        expected = json.loads((folder / "summary.json").read_text())["final_sha256"]
        if (expected != CHECKPOINT_SHA256[index] or (folder / "done").read_text().strip() != "completed"
                or sha256(folder / "final.pt") != expected):
            raise RuntimeError("training completion or checkpoint hash mismatch")
        model = PosteriorMDN()
        state = torch.load(folder / "final.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        if not all(torch.isfinite(value).all() for value in state.values()):
            raise FloatingPointError("nonfinite checkpoint parameter")
        models.append(model.eval())
        receipt.append(dict(training_seed=index, sha256=expected, strict_reload=True))
    return models, dict(checkpoints=receipt, load_seconds=time.perf_counter() - started)


def prepare_asset(root, seed, models):
    folder = root / "assets" / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    g = np.arange(5)
    angles = (g + .25) * np.pi / 5
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    sigma, offset = .55 + .1 * g, .4 + .15 * g
    rng = np.random.default_rng(920000 + seed)
    truth = rng.normal(size=2)
    sign = 2 * rng.integers(0, 2, size=5) - 1
    y = directions @ truth + sign * offset + sigma * rng.normal(size=5)
    context = np.column_stack((y, sigma, offset))
    started = time.perf_counter()
    outputs = [predict(models[index], context[index:index + 1]) for index in range(5)]
    prediction_seconds = time.perf_counter() - started
    parameters = {key: np.concatenate([item[key] for item in outputs], axis=0) for key in outputs[0]}
    parameters["directions"] = directions
    true_parameters = dict(exact_parameters(context), directions=directions)
    np.savez_compressed(folder / "parameters.npz", **parameters)
    np.savez_compressed(folder / "true_parameters.npz", **true_parameters)
    np.savez_compressed(folder / "observations.npz", truth=truth, context=context, generated_sign=sign)
    learned, true = exact_mixture(parameters), exact_mixture(true_parameters)
    np.savez_compressed(folder / "learned_reference.npz", **learned)
    np.savez_compressed(folder / "true_reference.npz", **true)
    projections = np.column_stack((np.cos(np.arange(32) * np.pi / 32), np.sin(np.arange(32) * np.pi / 32)))
    np.savez_compressed(folder / "projections.npz", directions=projections)
    error = model_error(learned, true, projections)
    write_json(folder / "summary.json", dict(seed=seed, prediction_seconds=prediction_seconds,
        network_calls=5, learned_to_true_projected_w1=error, exact_components=len(learned["probabilities"])))
    write_json(folder / "receipt.json", dict(files={p.name: sha256(p) for p in folder.iterdir()}))
    return parameters, prediction_seconds


def load_npz(path):
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def execute(payload, setting, seed, device, particles=4096):
    if setting["reference"] == "exact":
        started = time.perf_counter()
        mixture = exact_mixture(payload)
        rng = np.random.default_rng(seed)
        index = rng.choice(len(mixture["probabilities"]), size=particles, p=mixture["probabilities"])
        x = mixture["means"][index] + rng.multivariate_normal(np.zeros(2), mixture["covariance"], size=particles)
        return x, np.full(particles, 1 / particles), float(mixture["log_normalizer"]), dict(
            seconds=time.perf_counter() - started, preparation_seconds=None,
            counts=dict(factor_calls=len(mixture["probabilities"]) * 5),
            exact_components=len(mixture["probabilities"]), stages=0, device="cpu")
    kwargs = {key: setting[key] for key in ["reference", "proposal_scale", "global_probability", "direct_is"]}
    return sample(payload, seed, particles=particles, device=device, **kwargs)


def finish_cell(item):
    root, phase, cell, logz, diagnostic = item
    folder = root / phase / "cells" / f"cell_{cell['cell_id']:04d}"
    asset = root / "assets" / f"seed_{cell['dataset_seed']}"
    saved = load_npz(folder / "samples.npz")
    reference = load_npz(asset / "learned_reference.npz")
    projections = load_npz(asset / "projections.npz")["directions"]
    measured = metrics(saved["samples"], saved["weights"], reference, projections)
    common = json.loads((asset / "summary.json").read_text())
    row = dict(cell, **measured, status="completed", sampling_seconds=diagnostic["seconds"],
        prediction_seconds=common["prediction_seconds"], seconds=diagnostic["seconds"] + common["prediction_seconds"],
        log_normalizer=logz, log_normalizer_error=logz - float(reference["log_normalizer"]),
        ess_fraction=float(1 / np.square(saved["weights"]).sum() / len(saved["weights"])),
        factor_calls=diagnostic["counts"]["factor_calls"], stages=diagnostic["stages"])
    write_json(folder / "summary.json", row)
    write_json(folder / "receipt.json", dict(files={p.name: sha256(p) for p in folder.iterdir()}))
    (folder / "done").write_text("completed\n")
    return row


def select(rows):
    expected = {(s, c["setting_id"], r) for s, c, r in itertools.product(range(1600, 1604), settings(), range(3))}
    if len(rows) != len(expected) or {(r["dataset_seed"], r["setting_id"], r["repeat"]) for r in rows} != expected:
        raise ValueError("selection requires all 96 frozen cells")
    means = {c["setting_id"]: {key: float(np.mean([r[key] for r in rows if r["setting_id"] == c["setting_id"]]))
             for key in ["sliced_w1_32", "seconds"]} for c in settings()}
    if not all(np.isfinite(value) for row in means.values() for value in row.values()):
        raise FloatingPointError("nonfinite development result")
    baselines = [c["setting_id"] for c in settings() if c["role"] == "baseline"]
    best = min(means[name]["sliced_w1_32"] for name in baselines)
    eligible = [name for name in baselines if means[name]["sliced_w1_32"] <= best + .001]
    baseline = min(eligible, key=lambda name: (means[name]["seconds"], name))
    passed = (means["mixture"]["sliced_w1_32"] <= means[baseline]["sliced_w1_32"] + .002
              and means["mixture"]["seconds"] < .8 * means[baseline]["seconds"])
    return dict(baseline=baseline, candidate="mixture", means=means, expand_confirmation=passed,
                development_only=True, comparison_scope="against prior-SMC; exact enumeration also reported")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--phase", choices=["development", "confirmation"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.device == "cuda" and (not torch.cuda.is_available() or gpu_processes()):
        raise RuntimeError("an initially idle GPU is required")
    folder = args.root / args.phase
    folder.mkdir(parents=True, exist_ok=False)
    configurations = settings()
    if args.phase == "confirmation":
        selection = json.loads((args.root / "selection.json").read_text())
        if not selection["expand_confirmation"]:
            raise RuntimeError("development gate did not pass")
        configurations = [c for c in configurations if c["role"] != "baseline" or c["setting_id"] == selection["baseline"]]
    seeds = range(1600, 1604) if args.phase == "development" else range(1700, 1712)
    cells = [dict(setting, dataset_seed=s, repeat=r, cell_id=i, particles=4096, phase=args.phase)
             for i, (s, setting, r) in enumerate(itertools.product(seeds, configurations, range(3)))]
    order = np.random.default_rng(20261022 if args.phase == "development" else 20261023).permutation(len(cells))
    models, bank_receipt = load_bank(args.training_root)
    write_json(folder / "manifest.json", dict(cells=cells, settings=configurations, order=order.tolist(),
        source_commit=args.source_commit, sources={p: sha256(p) for p in SOURCES}, factor_bank=bank_receipt))
    assets = {s: prepare_asset(args.root, s, models) for s in seeds}
    execute(assets[min(seeds)][0], configurations[0], 441, args.device, particles=256)
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
        futures = []
        for index in order:
            if time.perf_counter() - started > 3600:
                raise TimeoutError("learned-factor phase exceeded one-hour wall limit")
            cell = cells[index]
            before = gpu_processes() if args.device == "cuda" else []
            if args.device == "cuda" and len(before) != 1:
                raise RuntimeError("external GPU work invalidates timing")
            seed = 9300000 + 10 * cell["dataset_seed"] + cell["repeat"]
            x, w, logz, diagnostic = execute(assets[cell["dataset_seed"]][0], cell, seed, args.device)
            after = gpu_processes() if args.device == "cuda" else []
            if args.device == "cuda" and len(after) != 1:
                raise RuntimeError("external GPU work invalidates timing")
            output = folder / "cells" / f"cell_{cell['cell_id']:04d}"
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "config.json", dict(cell, sampler_seed=seed))
            np.savez_compressed(output / "samples.npz", samples=x, weights=w)
            write_json(output / "diagnostics.json", diagnostic)
            write_json(output / "gpu_processes.json", dict(before=before, after=after))
            futures.append(pool.submit(finish_cell, (args.root, args.phase, cell, logz, diagnostic)))
            print(json.dumps(dict(cell_id=cell["cell_id"], setting_id=cell["setting_id"], sampled=True)), flush=True)
        rows = [future.result() for future in futures]
    write_json(folder / "rows.json", rows)
    if args.phase == "development":
        write_json(args.root / "selection.json", select(rows))
    write_json(folder / "state.json", dict(status="completed", cells=len(rows), seconds=time.perf_counter() - started,
        sampler_seconds=sum(r["sampling_seconds"] for r in rows)))
    (folder / "done").write_text("completed\n")


if __name__ == "__main__":
    main()
