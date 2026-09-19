import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

from audit_extension_results import compare_reference, independent_metrics, reference_arrays, true_parameters
from composition_extension import ExtensionModel
from run_solid_20260920 import sha256, write_json


def reference_job(job):
    name, cell, output = job
    config = json.loads((cell / "config.json").read_text())
    if config["suite"] == "heldout":
        with np.load(cell / "learned_parameters.npz") as archive:
            variance, means, weights = [archive[key] for key in ["variance", "means", "weights"]]
            true_variance, true_means, true_weights = true_parameters(archive["context"])
        truth = reference_arrays(true_variance, true_means, true_weights)
        true_checks = compare_reference(cell / "true_reference.npz", truth)
        np.savez_compressed(output / f"{name}_true.npz", **truth)
    else:
        model = ExtensionModel(config["groups"], config["dimension"], config["family"], "cpu")
        variance, means, weights = [value.numpy() for value in [model.variance, model.means, model.weights]]
        true_checks = []
    reference = reference_arrays(variance, means, weights)
    checks = compare_reference(cell / "reference.npz", reference)
    np.savez_compressed(output / f"{name}.npz", **reference)
    np.savez_compressed(output / f"{name}_parameters.npz", variance=variance, means=means, weights=weights)
    return name, dict(reference=checks, true_reference=true_checks)


def independent_population(variance, config):
    # 同时递推全部坐标，使用独立的累计和计算 WOR 子集二次项。
    groups, dimensions = variance.shape
    batch = config["batch"]
    current = np.ones(dimensions)
    minima = np.ones(dimensions)
    alive = np.ones(dimensions, dtype=bool)
    failed_step = np.zeros(dimensions, dtype=int)
    failing_denominator = np.zeros(dimensions)
    levels = np.linspace(math.sqrt(config["u_max"]), 0, config["steps"] + 1) ** 2
    for k, u in enumerate(levels[:-1]):
        alpha2 = math.exp(-u)
        h = u - levels[k + 1]
        strengths = 1 / (1 - alpha2 + alpha2 * variance) - 1
        if config["method"] in ["full", "tail"]:
            total = strengths.sum(0, keepdims=True)
            quadratic = (total ** 2 - (strengths ** 2).sum(0, keepdims=True)) / 2
        elif config["suite"] == "tail_error":
            control_variance = variance * (1 + config["control_delta"])
            a0 = 1 / (1 - alpha2 + alpha2 * control_variance) - 1
            residual = strengths - a0
            total0 = a0.sum(0)
            quadratic0 = (total0 ** 2 - (a0 ** 2).sum(0)) / 2
            total = total0 + groups * residual
            quadratic = quadratic0 + groups * (total0 - a0) * residual + groups * (groups - 1) * residual ** 2 / 2
        elif config["sampling"] == "without_replacement":
            ordered = np.sort(strengths, axis=0)
            prefix = np.concatenate([np.zeros((1, dimensions)), np.cumsum(ordered, axis=0)])
            prefix2 = np.concatenate([np.zeros((1, dimensions)), np.cumsum(ordered ** 2, axis=0)])
            j = np.arange(batch + 1)
            sums = prefix[j] + prefix[-1] - prefix[groups - batch + j]
            squares = prefix2[j] + prefix2[-1] - prefix2[groups - batch + j]
            total = groups * sums / batch
            quadratic = groups * (groups - 1) * (sums ** 2 - squares) / (2 * batch * (batch - 1))
        else:
            extrema = np.stack([strengths.min(0), strengths.max(0)])
            total = groups * extrema
            quadratic = groups * (groups - 1) * extrema ** 2 / 2
        denominator = 1 - 2 * h * quadratic * current
        smallest = denominator.min(0)
        minima[alive] = np.minimum(minima[alive], smallest[alive])
        failed = alive & (smallest <= 0)
        assert np.all(smallest[failed] < -1e-10)
        failed_step[failed] = k + 1
        failing_denominator[failed] = smallest[failed]
        alive[failed] = False
        if not alive.any():
            break
        multiplier = 1 - h * ((1 + config["diffusion"]) * total + config["diffusion"]) / 2
        current[alive] = (multiplier[:, alive] ** 2 * current[alive] / denominator[:, alive] + h * config["diffusion"]).max(0)
    rows = []
    for d in range(dimensions):
        row = dict(finite_normalizer=bool(alive[d]), smallest_denominator=float(minima[d]))
        if alive[d]:
            row["maximum_terminal_component_variance"] = float(current[d])
        else:
            row.update(failure_step=int(failed_step[d]), minimum_denominator=float(failing_denominator[d]))
        rows.append(row)
    return dict(finite_normalizer=bool(alive.all()), coordinates=rows)


def asset_name(config):
    return f"learned_{config['training_seed']}_{config['dataset_seed']}" if config["suite"] == "heldout" else config["family"]


def intervals(values, crossed=False):
    values = np.asarray(values)
    rng = np.random.default_rng(20260920)
    if crossed:
        i = rng.integers(0, values.shape[0], size=(10000, values.shape[0]))
        j = rng.integers(0, values.shape[1], size=(10000, values.shape[1]))
        boot = values[i[:, :, None], j[:, None, :]].mean((1, 2))
        conditional = values.mean(0)[j].mean(1)
    else:
        index = rng.integers(0, len(values), size=(10000, len(values)))
        boot = values[index].mean(1)
        conditional = boot
    return dict(mean=float(values.mean()), ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
                conditional_data_ci95=np.quantile(conditional, [0.025, 0.975]).tolist(),
                per_training_seed=values.mean(1).tolist() if crossed else None)


def aggregate(rows):
    learned = {}
    pairs = {}
    for condition in ["full", "unbiased", "tail"]:
        selected = [r for r in rows if r["suite"] == "heldout" and r["condition"] == condition]
        arrays = {}
        for metric in ["w1_mean", "true_w1_mean", "seconds"]:
            arrays[metric] = np.array([[next(r[metric] for r in selected if r["training_seed"] == t and r["dataset_seed"] == d)
                                        for d in range(100, 120)] for t in range(5)])
        learned[condition] = {metric: intervals(value, crossed=True) for metric, value in arrays.items()}
        pairs[condition] = arrays
    differences = {}
    for condition in ["unbiased", "tail"]:
        differences[condition] = {metric: intervals(pairs[condition][metric] - pairs["full"][metric], crossed=True)
                                  for metric in ["w1_mean", "true_w1_mean", "seconds"]}
    boundary, tail_error = [], []
    for family in ["gaussian", "mixture", "weak_mixture"]:
        for steps in [512, 2048]:
            group = [r for r in rows if r["suite"] == "boundary" and r["family"] == family and r["steps"] == steps]
            means = {}
            vectors = {}
            for condition in ["full", "below", "at", "tail"]:
                cells = sorted([r for r in group if r["condition"] == condition], key=lambda r: r["seed"])
                assert len(cells) == 10
                vectors[condition] = np.array([r["w1_mean"] for r in cells])
                means[condition] = dict(w1=intervals(vectors[condition]), seconds=float(np.mean([r["seconds"] for r in cells])),
                                        finite=all(r["finite_normalizer"] for r in cells), batch=cells[0]["batch"],
                                        final_ess=float(np.mean([r["final_ess_fraction"] for r in cells])))
            boundary.append(dict(family=family, steps=steps, conditions=means,
                                 at_minus_below=intervals(vectors["at"] - vectors["below"]),
                                 tail_minus_full=intervals(vectors["tail"] - vectors["full"])))
        vectors = {}
        conditions = {}
        for delta in [-0.1, 0, 0.1]:
            cells = sorted([r for r in rows if r["suite"] == "tail_error" and r["family"] == family and r["control_delta"] == delta], key=lambda r: r["seed"])
            assert len(cells) == 10
            vectors[str(delta)] = np.array([r["w1_mean"] for r in cells])
            conditions[str(delta)] = dict(w1=intervals(vectors[str(delta)]), finite=all(r["finite_normalizer"] for r in cells))
        tail_error.append(dict(family=family, conditions=conditions,
                               minus=intervals(vectors["-0.1"] - vectors["0"]),
                               plus=intervals(vectors["0.1"] - vectors["0"])))
    return dict(heldout=learned, heldout_paired_differences=differences, boundary=boundary, tail_error=tail_error)


def figures(summary, sensitivity, output):
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    names = ["full", "unbiased", "tail"]
    for i, name in enumerate(names):
        result = summary["heldout"][name]["w1_mean"]
        lower, upper = result["ci95"]
        ax.errorbar(i, result["mean"], yerr=[[result["mean"] - lower], [upper - result["mean"]]], fmt="o", capsize=5)
        ax.scatter(np.full(5, i) + np.linspace(-0.12, 0.12, 5), result["per_training_seed"], s=16, alpha=0.5)
    ax.set_xticks(range(3), ["Full", "Raw WOR", "Tail WOR"])
    ax.set_ylabel("W1 to learned composition")
    ax.set_title("20 new datasets × 5 frozen training seeds")
    fig.tight_layout()
    fig.savefig(output / "heldout.pdf")
    fig.savefig(output / "heldout.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6))
    for row in summary["boundary"]:
        i = [512, 2048].index(row["steps"])
        j = ["gaussian", "mixture", "weak_mixture"].index(row["family"])
        ax = axes[i, j]
        for k, condition in enumerate(["full", "below", "at", "tail"]):
            result = row["conditions"][condition]
            ax.scatter(k, result["w1"]["mean"], marker="o" if result["finite"] else "x", s=45)
            lo, hi = result["w1"]["ci95"]
            ax.vlines(k, lo, hi, linewidth=1)
        ax.set_xticks(range(4), ["Full", "Below", "At", "Tail"], rotation=20)
        ax.set_title(f"{row['family']}, K={row['steps']}")
        ax.set_ylabel("W1")
    fig.tight_layout()
    fig.savefig(output / "boundary.pdf")
    fig.savefig(output / "boundary.png", dpi=180)
    plt.close(fig)
    families = ["gaussian", "mixture", "weak_mixture"]
    deltas = sorted({r["delta"] for r in sensitivity})
    grid = np.array([[int(next(r["finite_normalizer"] for r in sensitivity if r["family"] == f and r["steps"] == k and r["delta"] == delta))
                      for delta in deltas] for f in families for k in [512, 2048]])
    fig, ax = plt.subplots(figsize=(9, 3.7))
    ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(deltas)), [f"{d:+.1%}" for d in deltas], rotation=45)
    ax.set_yticks(range(6), [f"{f}, K={k}" for f in families for k in [512, 2048]])
    for i, j in np.ndindex(grid.shape):
        ax.text(j, i, "finite" if grid[i, j] else "infinite", ha="center", va="center", fontsize=7)
    ax.set_xlabel("Relative error in control component variance")
    fig.tight_layout()
    fig.savefig(output / "tail_sensitivity.pdf")
    fig.savefig(output / "tail_sensitivity.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    references_root = args.output / "references"
    references_root.mkdir()
    manifest = json.loads((args.root / "manifest.json").read_text())
    assert all(sha256(Path(__file__).parent / name) == value for name, value in manifest["sources"].items())
    cells = sorted(path.parent for path in (args.root / "cells").glob("*/solid_done"))
    if not args.partial:
        assert len(cells) == manifest["expected_cells"] == 630
        assert (args.root / "launcher.exit").read_text().strip() == "0"
    assets = {}
    for cell in cells:
        config = json.loads((cell / "config.json").read_text())
        assets.setdefault(asset_name(config), cell)
    jobs = [(name, cell, references_root) for name, cell in assets.items()]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        reference_checks = dict(executor.map(reference_job, jobs))
    write_json(args.output / "reference_checks.json", reference_checks)
    cache, certs = {}, {}
    rows, differences = [], []
    for cell in cells:
        config = json.loads((cell / "config.json").read_text())
        expected = manifest["cells"][config["cell_id"]]
        assert all(config[key] == value for key, value in expected.items())
        assert config["commit"] == manifest["commit"]
        receipt = json.loads((cell / "receipt.json").read_text())
        assert all(sha256(cell / name) == value for name, value in receipt["files"].items())
        summary = json.loads((cell / "summary.json").read_text())
        assert summary["status"] == "completed"
        name = asset_name(config)
        if name not in cache:
            with np.load(references_root / f"{name}.npz") as archive:
                reference = {key: archive[key] for key in archive.files}
            with np.load(references_root / f"{name}_parameters.npz") as archive:
                variance = archive["variance"]
            cache[name] = reference, variance
        reference, variance = cache[name]
        with np.load(cell / "samples.npz") as archive:
            samples, weights = archive["samples"], archive["weights"]
        assert samples.shape == (config["particles"], config["dimension"])
        metrics = independent_metrics(samples, weights, reference)
        difference = abs(metrics["w1_mean"] - summary["w1_mean"])
        assert difference < 2 * 24 / 65536, (cell, difference)
        differences.append(difference)
        cert_key = (name, config["method"], config["sampling"], config["batch"], config["steps"], config.get("control_delta"))
        if cert_key not in certs:
            certs[cert_key] = independent_population(variance, config)
        independent = certs[cert_key]
        recorded = json.loads((cell / "certificate.json").read_text())
        assert independent["finite_normalizer"] == recorded["finite_normalizer"]
        for left, right in zip(independent["coordinates"], recorded["coordinates"]):
            assert left["finite_normalizer"] == right["finite_normalizer"]
            if not left["finite_normalizer"]:
                assert left["failure_step"] == right["failure_step"]
            else:
                np.testing.assert_allclose(left["maximum_terminal_component_variance"], right["maximum_terminal_component_variance"], rtol=1e-7, atol=1e-9)
        row = {key: value for key, value in summary.items() if key != "runtime"}
        row.update(metrics, finite_normalizer=independent["finite_normalizer"])
        if config["suite"] == "heldout":
            with np.load(references_root / f"{name}_true.npz") as archive:
                truth = {key: archive[key] for key in archive.files}
            row.update({f"true_{key}": value for key, value in independent_metrics(samples, weights, truth).items()})
            row["learning_w1"] = float(trapezoid(abs(reference["cdf"] - truth["cdf"]), reference["grid"], axis=1).mean())
        rows.append(row)
    names = sorted(set().union(*(row.keys() for row in rows)))
    with (args.output / "cells.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    report = dict(status="partial_audit_passed" if args.partial else "passed", audited_cells=len(rows), expected=630,
                  independent_references=len(assets), independent_certificates=len(certs),
                  max_w1_recomputation_difference=max(differences), analysis_sha256=sha256(__file__))
    if not args.partial:
        sensitivity = json.loads((args.root / "tail_sensitivity.json").read_text())
        assert len(sensitivity) == 66
        for row in sensitivity:
            model = ExtensionModel(64, 8, row["family"], "cpu")
            config = dict(suite="tail_error", method="cv", sampling="with_replacement", batch=4,
                          steps=row["steps"], u_max=20.0, diffusion=1.0, control_delta=row["delta"])
            independent = independent_population(model.variance.numpy(), config)
            assert independent["finite_normalizer"] == row["finite_normalizer"]
        summary = aggregate(rows)
        write_json(args.output / "statistics.json", summary)
        figures(summary, sensitivity, args.output)
    write_json(args.output / "audit.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
