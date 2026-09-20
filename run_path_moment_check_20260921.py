import argparse
import json
from pathlib import Path

import numpy as np
import torch

from composition_benchmark import FactorModel
from factorized_baseline_20260921 import factorized_sample
from gaussian_path_moments_20260921 import path_moment
from run_solid_20260920 import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model = FactorModel(4, 3, "gaussian", "cpu")
    parameters = {key: getattr(model, key).numpy() for key in ["variance", "means", "weights"]}
    grid = np.linspace(1.0, 0.0, 17)
    report = dict(scope="follow-up CPU diagnostic after an observed finite-sample deviation; no performance selection or held-out claim",
                  particles=65536, seed=759, grid=grid.tolist(), cases={},
                  sources={name: sha256(name) for name in ["run_path_moment_check_20260921.py", "gaussian_path_moments_20260921.py", "factorized_baseline_20260921.py"]})
    for family in ["original_gaussian", "weak_gaussian"]:
        params = {key: value.copy() for key, value in parameters.items()}
        if family == "weak_gaussian":
            params["variance"] = 0.8 + 0.2 * params["variance"]
        first, second = path_moment(params, grid, 1), path_moment(params, grid, 2)
        row = dict(first_moment=first, second_moment=second, runs=[])
        row["relative_variance"] = float(np.expm1(second["log_moment"] - 2 * first["log_moment"])) if second["finite"] else None
        for ess in [0.0, 0.95]:
            samples, weights, logz, records = factorized_sample(params, grid, 65536, 759, "cpu", ess)
            name = f"{family}_ess{ess:.2f}.npz"
            np.savez_compressed(args.output / name, samples=samples, particle_weights=weights, **params)
            row["runs"].append(dict(ess_fraction=ess, log_normalizer=logz,
                                   log_normalizer_difference=logz-first["log_moment"],
                                   sample_mean=samples.mean(0).tolist(), sample_variance=samples.var(0).tolist(),
                                   minimum_coordinate_ess_fraction=min(r["minimum_coordinate_ess"] for r in records),
                                   resampled_coordinates=sum(r["resampled_coordinates"] for r in records),
                                   raw_file=name, sha256=sha256(args.output / name)))
        report["cases"][family] = row
    write_json(args.output / "report.json", report)
    (args.output / "done").write_text("completed\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
