import argparse
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from anchored_tail_20260921 import AnchoredTailModel
from composition_benchmark import measure, run_cell, runtime
from composition_extension import ExtensionModel, certificate
from run_dual_20260921 import asset_name, joint_metrics, load_npz, paired_seed, prepare_asset
from run_solid_20260920 import sha256, write_json


class CachedOriginal(ExtensionModel):
    def reference(self, *args, **kwargs):
        return self.saved_reference


class CachedAnchored(AnchoredTailModel):
    def reference(self, *args, **kwargs):
        return self.saved_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert not torch.cuda.is_available()
    started = time.perf_counter()
    args.root.mkdir(parents=True, exist_ok=False)
    phase = args.root / "development"
    phase.mkdir()
    problems = [dict(family="learned", training_seed=t, dataset_seed=d, repeat=0)
                for t, d in itertools.product([0, 1], range(600, 604))]
    configs = [dict(problem, method=method, setting_id=method, groups=64, dimension=8, particles=4096,
                    steps=1024, batch=4, u_max=20.0, diffusion=1.0, ess_threshold=0.5, device="cpu",
                    seed=paired_seed(problem), phase="development", line="anchored_pilot")
               for problem, method in itertools.product(problems, ["full", "tail", "tail_anchored"])]
    for i, config in enumerate(configs):
        config["cell_id"] = i
    sources = ["run_anchored_pilot_20260921.py", "anchored_tail_20260921.py", "composition_benchmark.py",
               "composition_extension.py", "run_dual_20260921.py", "learned_sbi.py", "docs/ANCHORED_PILOT_PROTOCOL_20260921.md"]
    order = np.random.default_rng(20260923).permutation(len(configs)).tolist()
    write_json(phase / "manifest.json", dict(commit=args.commit, cells=configs, order=order,
                sources={name: sha256(name) for name in sources},
                checkpoints={str(t): sha256(args.training_root / f"training_{t}" / "final.pt") for t in range(5)}))
    models = {}
    for problem in problems:
        prepare_asset(args.root, problem, args.training_root, models)
    del models
    write_json(phase / "runtime.json", runtime())
    grid = np.linspace(math.sqrt(20), 0, 1025) ** 2
    for cell_id in order:
        if time.perf_counter() - started > 3600:
            write_json(args.root / "state.json", dict(status="time_limit", next_cell=cell_id))
            raise SystemExit(2)
        config = configs[cell_id]
        output = phase / "cells" / f"cell_{cell_id:04d}"
        asset = args.root / "assets" / asset_name(config)
        params, reference = load_npz(asset / "parameters.npz"), load_npz(asset / "reference.npz")
        preparation_start = time.perf_counter()
        if config["method"] == "tail_anchored":
            model = CachedAnchored(64, 8, "learned", "cpu", "without_replacement", params, grid[:-1])
        else:
            model = CachedOriginal(64, 8, "learned", "cpu", "without_replacement", params)
        model.control_variance = model.variance
        model.saved_reference = reference
        preparation_seconds = time.perf_counter() - preparation_start
        population = certificate(model, config)
        assert population["finite_normalizer"]
        report = run_cell(config, output, model)
        arrays = load_npz(output / "samples.npz")
        report.update(joint_metrics(arrays["samples"], arrays["weights"], reference))
        report.update({"true_" + key: value for key, value in measure(arrays["samples"], arrays["weights"], load_npz(asset / "true_reference.npz")).items()})
        report.update(preparation_seconds=preparation_seconds, seconds_including_preparation=report["seconds"]+preparation_seconds)
        if config["method"] == "tail_anchored":
            report["anchor_preparation"] = model.cost_report()
        write_json(output / "summary.json", report)
        write_json(output / "certificate.json", population)
        write_json(output / "receipt.json", dict(status="completed", files={p.name: sha256(p) for p in sorted(output.iterdir()) if p.name != "done"}))
        count = len(list((phase / "cells").glob("*/done")))
        write_json(args.root / "state.json", dict(status="running", completed=count, expected=24))
        print(json.dumps(dict(cell_id=cell_id, method=config["method"], w1=report["w1_mean"])), flush=True)
    assert len(list((phase / "cells").glob("*/done"))) == 24
    (phase / "done").write_text("completed\n")
    write_json(args.root / "state.json", dict(status="completed", completed=24, expected=24,
                                             seconds=time.perf_counter()-started))


if __name__ == "__main__":
    main()
