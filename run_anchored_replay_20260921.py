import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from certified_composition_torch import FactorParameters, sample
from composition_benchmark import measure, runtime
from optimized_anchored_tail_20260921 import OptimizedAnchoredTailModel
from run_anchored_confirmation_20260921 import GRID, anchored_sample
from run_dual_20260921 import asset_name, joint_metrics, load_npz
from run_solid_20260920 import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--hours", type=float, required=True)
    args = parser.parse_args()
    if not 0 < args.hours <= 4 or not torch.cuda.is_available():
        raise ValueError("a CUDA device and a positive remaining budget <=4h are required")
    assert args.root.parent.resolve() == args.original_root.parent.resolve()
    assert (args.original_root / "confirmation" / "done").read_text().strip() == "completed"
    torch.set_num_threads(1)
    started = time.perf_counter()
    original_manifest_path = args.original_root / "confirmation" / "manifest.json"
    original_manifest = json.loads(original_manifest_path.read_text())
    assert original_manifest["commit"] == "39814c1192ff5ad02744d3d332c4088ffe2663f7"
    assert len(original_manifest["cells"]) == 150
    for name, digest in original_manifest["sources"].items():
        assert sha256(args.root.parent / "anchored-confirmation-code" / name) == digest
    for t, digest in original_manifest["checkpoints"].items():
        assert sha256(args.training_root / f"training_{t}" / "final.pt") == digest
    args.root.mkdir(parents=True, exist_ok=False)
    (args.root / "assets").symlink_to(Path("..") / args.original_root.name / "assets", target_is_directory=True)
    phase = args.root / "confirmation"
    phase.mkdir()
    configs = [dict(c, line="anchored_equivalent_replay") for c in original_manifest["cells"]]
    order = np.random.default_rng(20260925).permutation(150).tolist()
    sources = ["run_anchored_replay_20260921.py", "optimized_anchored_tail_20260921.py", "anchored_tail_20260921.py",
               "run_anchored_confirmation_20260921.py", "certified_composition_torch.py", "composition_benchmark.py",
               "composition_extension.py", "run_dual_20260921.py", "docs/ANCHORED_EQUIVALENT_REPLAY_20260921.md"]
    write_json(phase / "manifest.json", dict(commit=args.commit, cells=configs, order=order,
               checkpoints=original_manifest["checkpoints"], sources={name: sha256(name) for name in sources},
               original_manifest_sha256=sha256(original_manifest_path),
               original_commit=original_manifest["commit"], scope="same-cohort algebraic-equivalence and interleaved timing replay; not new independent cases"))
    write_json(phase / "runtime.json", runtime())
    assets = {asset_name(c) for c in configs}
    for name in assets:
        asset = args.root / "assets" / name
        receipt = json.loads((asset / "receipt.json").read_text())
        assert all(sha256(asset / p) == digest for p, digest in receipt["files"].items())
    warm_params = load_npz(args.root / "assets" / asset_name(configs[0]) / "parameters.npz")
    for method in ["full", "tail_fixed"]:
        sample(warm_params, [20, 19.99], 128, 143, method=method, device="cuda")
    warm_model = OptimizedAnchoredTailModel(64, 8, "learned", "cuda", "without_replacement", warm_params, [20, 19.99])
    warm_p = FactorParameters(**warm_params)
    warm_p = FactorParameters(warm_p.variance.cuda(), warm_p.means.cuda(), warm_p.weights.cuda())
    anchored_sample(warm_p, warm_model, [20, 19.99], 128, 143, 4, 0.5, "cuda")
    equivalences = []
    for cell_id in order:
        if time.perf_counter() - started > args.hours * 3600:
            write_json(args.root / "state.json", dict(status="time_limit", next_cell=cell_id))
            raise SystemExit(2)
        config = configs[cell_id]
        original = args.original_root / "confirmation" / "cells" / f"cell_{cell_id:04d}"
        assert (original / "done").exists()
        receipt = json.loads((original / "receipt.json").read_text())
        assert all(sha256(original / p) == digest for p, digest in receipt["files"].items())
        previous = json.loads((original / "summary.json").read_text())
        original_arrays = load_npz(original / "samples.npz")
        asset = args.root / "assets" / asset_name(config)
        params, reference = load_npz(asset / "parameters.npz"), load_npz(asset / "reference.npz")
        population = json.loads((original / "certificate.json").read_text())
        assert population["finite_normalizer"]
        output = phase / "cells" / f"cell_{cell_id:04d}"
        output.mkdir(parents=True)
        write_json(output / "config.json", config)
        torch.cuda.synchronize()
        cell_start = time.perf_counter()
        p = FactorParameters(**params)
        if config["method"] == "tail_anchored":
            model = OptimizedAnchoredTailModel(64, 8, "learned", "cuda", "without_replacement", params, GRID[:-1])
            p = FactorParameters(p.variance.cuda(), p.means.cuda(), p.weights.cuda())
            x, w, logz, records = anchored_sample(p, model, GRID, config["particles"], config["seed"], 4, 0.5, "cuda")
        else:
            x, w, logz, records = sample(p, GRID, config["particles"], config["seed"], method=config["method"], device="cuda", return_type="torch")
        torch.cuda.synchronize()
        samples, weights = x.cpu().numpy(), w.cpu().numpy()
        torch.cuda.synchronize()
        seconds = time.perf_counter()-cell_start
        np.savez_compressed(output / "samples.npz", samples=samples, weights=weights)
        equivalence = dict(cell_id=cell_id, method=config["method"],
                           sample_max_abs=float(np.max(np.abs(samples-original_arrays["samples"]))),
                           weight_max_abs=float(np.max(np.abs(weights-original_arrays["weights"]))),
                           logz_abs=abs(float(logz)-previous["log_normalizer"]))
        equivalence["passed"] = (equivalence["sample_max_abs"] <= 1e-8 and equivalence["weight_max_abs"] <= 1e-10 and equivalence["logz_abs"] <= 1e-8)
        write_json(output / "equivalence.json", equivalence)
        if not equivalence["passed"]:
            write_json(args.root / "state.json", dict(status="equivalence_failed", **equivalence))
            raise ArithmeticError(f"Equivalence gate failed at cell {cell_id}")
        equivalences.append(equivalence)
        report = dict(config, status="completed", seconds=seconds, seconds_including_preparation=seconds,
                      log_normalizer=float(logz), factor_calls=sum(r["factor_calls"] for r in records),
                      preparation_calls=sum(r["preparation_calls"] for r in records),
                      preparation_derivative_calls=sum(r.get("preparation_derivative_calls", 0) for r in records),
                      resampling_count=sum(r["resampled"] for r in records),
                      **measure(samples, weights, reference), **joint_metrics(samples, weights, reference))
        report.update({"true_"+k: v for k, v in measure(samples, weights, load_npz(asset / "true_reference.npz")).items()})
        if config["method"] == "tail_anchored":
            report["anchor_preparation"] = model.cost_report()
        write_json(output / "summary.json", report)
        write_json(output / "steps.json", records)
        write_json(output / "certificate.json", population)
        write_json(output / "receipt.json", dict(status="completed", files={p.name: sha256(p) for p in output.iterdir()}))
        (output / "done").write_text("completed\n")
        write_json(args.root / "state.json", dict(status="running", completed=len(equivalences), expected=150))
        print(json.dumps(dict(cell_id=cell_id, method=config["method"], seconds=seconds, equivalence=equivalence)), flush=True)
    assert len(equivalences) == len(list((phase / "cells").glob("*/done"))) == 150
    write_json(args.root / "equivalence.json", dict(status="passed", cells=equivalences,
               maxima={key:max(r[key] for r in equivalences) for key in ["sample_max_abs", "weight_max_abs", "logz_abs"]}))
    (phase / "done").write_text("completed\n")
    write_json(args.root / "state.json", dict(status="completed", completed=150, expected=150, seconds=time.perf_counter()-started))


if __name__ == "__main__":
    main()
