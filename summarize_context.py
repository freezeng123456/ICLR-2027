import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    cells = sorted(p.parent for p in args.root.glob("*/done"))
    if len(cells) != 12:
        raise ValueError(f"预期 12 个完整单元，实际 {len(cells)}")
    summaries, rows, data = {}, [], {}
    for cell in cells:
        config = json.loads((cell / "config.json").read_text())
        summary = json.loads((cell / "summary.json").read_text())
        prior, n, seed = config["prior"], config["n_context"], config["seed"]
        summaries[cell.name] = summary
        data[(prior, n, seed)] = np.load(cell / "records.npz")
        for width in [64, 128]:
            record = summary[f"{prior}{width}"]
            rows.append({"prior": prior, "n_context": n, "seed": seed, "width": width,
                         "improvement": record["relative_improvement"], "ci_low": record["ci95"][0], "ci_high": record["ci95"][1],
                         **{key: np.mean(value) for key, value in record["risk_curves"].items()}})
    with (args.root / "stability_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    routing = {}
    for n in [8, 24]:
        for seed in [20260910, 20260911, 20260912]:
            risks, accuracy = {}, {}
            for prior in ["gp", "jump"]:
                record = data[(prior, n, seed)]
                split = len(record["xc"]) // 2
                chosen_gp = record["gp128_loo_nll"] < record["jump128_loo_nll"]
                mean = np.where(chosen_gp[:, None], record["gp128_mean"], record["jump128_mean"])
                risks[prior] = {key: (record["latent_var"] + (value - record["latent_mean"]) ** 2)[split:].mean(axis=1)
                                for key, value in {"gp_fixed": record["gp128_mean"], "jump_fixed": record["jump128_mean"], "routed": mean}.items()}
                accuracy[prior] = float(np.mean(chosen_gp[split:] == (prior == "gp")))
            point = {key: float(np.mean([risks[prior][key].mean() for prior in risks])) for key in risks["gp"]}
            rng = np.random.default_rng(seed)
            draws = []
            for _ in range(1000):
                indices = {p: rng.integers(0, len(risks[p]["routed"]), len(risks[p]["routed"])) for p in risks}
                draw = {key: np.mean([risks[p][key][indices[p]].mean() for p in risks]) for key in point}
                best = min(draw["gp_fixed"], draw["jump_fixed"])
                draws.append((best - draw["routed"]) / best)
            best = min(point["gp_fixed"], point["jump_fixed"])
            routing[f"n{n}_s{seed}"] = {"risks": point, "accuracy": accuracy,
                                        "improvement_vs_best_fixed": (best - point["routed"]) / best,
                                        "ci95": np.quantile(draws, [0.025, 0.975]).tolist()}
    output = {"cells": len(cells), "test_tasks": 12 * 256, "routing": routing,
              "stability_gate_pass": all(row["improvement"] >= 0.15 for row in rows),
              "routing_gate_pass": all(value["improvement_vs_best_fixed"] >= 0.10 for value in routing.values()),
              "stability_rows": rows}
    (args.root / "aggregate.json").write_text(json.dumps(output, indent=2, allow_nan=False))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
