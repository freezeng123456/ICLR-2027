import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from certified_composition_torch import FactorParameters, sample
from factorized_baseline_20260921 import time_factorized
from optimized_anchored_tail_20260921 import OptimizedAnchoredTailModel
from run_anchored_confirmation_20260921 import anchored_sample
from run_dual_20260921 import asset_name, load_npz
from run_paper_anchor_20260921 import run_cell
from run_solid_20260920 import sha256, write_json
from run_twist_pilot_20260921 import gpu_processes


def matched_cells(base_manifest):
    primary = [c for c in base_manifest["cells"] if c["setting_id"] == "anchor_n8192_k1024_m8"]
    if len(primary) != 100 or {(c["training_seed"], c["dataset_seed"]) for c in primary} != {(t, s) for t in range(5) for s in range(900, 920)}:
        raise ValueError("the complete fixed 100-target cohort is required")
    result = []
    for original in primary:
        for method, name in [("full", "full"), ("factorized_full", "factorized_full"), ("tail_anchored", "anchor_n8192_k1024_m8")]:
            result.append(dict(original, method=method, setting_id=name, line="matched_anchor", cell_id=len(result)))
    return result, {(c["training_seed"], c["dataset_seed"]): c for c in primary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or gpu_processes():
        raise RuntimeError("CUDA must be available and idle")
    torch.set_num_threads(1)
    base = json.loads((args.base / "confirmation" / "manifest.json").read_text())
    cells, primary = matched_cells(base)
    args.root.mkdir(parents=True, exist_ok=False)
    if args.root.parent.resolve() != args.base.parent.resolve():
        raise ValueError("base and replay must be sibling roots for recoverable relative assets")
    (args.root / "assets").symlink_to(Path("..") / args.base.name / "assets", target_is_directory=True)
    phase_root = args.root / "confirmation"
    phase_root.mkdir()
    sources = {name: sha256(name) for name in base["sources"]}
    if sources != base["sources"]:
        raise RuntimeError("original sampling implementation changed")
    sources.update({name: sha256(name) for name in ["run_matched_anchor_20260921.py", "run_twist_pilot_20260921.py", "docs/MATCHED_BUDGET_PROTOCOL_20260921.md"]})
    order = np.random.default_rng(20261003).permutation(len(cells))
    write_json(phase_root / "manifest.json", dict(commit=args.commit, cells=cells, sources=sources, checkpoints=base["checkpoints"], order=order.tolist(),
        parent_manifest_sha256=sha256(args.base / "confirmation" / "manifest.json"), evidence_status="post-hoc matched-budget ablation on existing targets"))
    params = load_npz(args.base / "assets" / asset_name(cells[0]) / "parameters.npz")
    grid = np.array([20., 19.99])
    sample(params, grid, 128, 1, method="full", device="cuda")
    time_factorized(params, grid, 128, 1, "cuda")
    model = OptimizedAnchoredTailModel(64, 8, "learned", "cuda", "without_replacement", params, grid[:-1])
    tensors = FactorParameters(**{key: torch.as_tensor(value, device="cuda") for key, value in params.items()})
    anchored_sample(tensors, model, grid, 128, 1, 8, .5, "cuda")
    start, equivalence = time.perf_counter(), []
    for index in order:
        if time.perf_counter() - start > 1200:
            raise TimeoutError("matched-budget study exceeded twenty minutes")
        before = gpu_processes()
        if len(before) != 1:
            raise RuntimeError("external GPU process detected")
        c = cells[index]
        cell_root = phase_root / "cells" / f"cell_{c['cell_id']:04d}"
        row = run_cell(c, args.root / "assets" / asset_name(c), cell_root, "cuda")
        after = gpu_processes()
        if len(after) != 1:
            raise RuntimeError("external GPU process detected; timing invalid")
        write_json(args.root / f"gpu_{c['cell_id']:04d}.json", dict(before=before, after=after))
        if c["method"] == "tail_anchored":
            original = primary[c["training_seed"], c["dataset_seed"]]
            original_root = args.base / "confirmation" / "cells" / f"cell_{original['cell_id']:04d}"
            a, b = load_npz(original_root / "samples.npz"), load_npz(cell_root / "samples.npz")
            errors = {key: float(np.max(np.abs(a[key] - b[key]))) for key in a}
            for key in a:
                np.testing.assert_allclose(a[key], b[key], atol=1e-12, rtol=1e-12)
            saved = json.loads((original_root / "summary.json").read_text())
            np.testing.assert_allclose(row["log_normalizer"], saved["log_normalizer"], atol=1e-10, rtol=1e-12)
            equivalence.append(dict(cell_id=c["cell_id"], original_cell_id=original["cell_id"], errors=errors,
                                    logz_difference=abs(row["log_normalizer"] - saved["log_normalizer"])))
        print(json.dumps({k: row[k] for k in ["cell_id", "setting_id", "w1_mean", "seconds"]}), flush=True)
    write_json(args.root / "equivalence.json", dict(status="passed", cells=len(equivalence), checks=equivalence))
    write_json(args.root / "state.json", dict(status="completed", seconds=time.perf_counter() - start, cells=len(cells)))
    (phase_root / "done").write_text("completed\n")


if __name__ == "__main__":
    main()
