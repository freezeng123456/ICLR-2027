import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from scipy.integrate import trapezoid
import torch

from audit_composition_references import parameters as oracle_parameters
from audit_extension_results import compare_reference, independent_metrics, reference_arrays, true_parameters
from learned_sbi import PosteriorMDN, dataset
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
        variance, means, weights = oracle_parameters(config["groups"], config["dimension"], config["family"])
        true_checks = []
    reference = reference_arrays(variance, means, weights)
    checks = compare_reference(cell / "reference.npz", reference)
    np.savez_compressed(output / f"{name}.npz", **reference)
    np.savez_compressed(output / f"{name}_parameters.npz", variance=variance, means=means, weights=weights)
    return name, dict(reference=checks, true_reference=true_checks)


def validate_inputs(cells, manifest, training_root, context_atol=0.0):
    networks, parameter_cache, hashes, contexts = {}, {}, {}, {}
    identifiers = []
    context_error = 0.0
    for cell in cells:
        config = json.loads((cell / "config.json").read_text())
        identifiers.append(config["cell_id"])
        expected = manifest["cells"][config["cell_id"]]
        assert all(config[key] == value for key, value in expected.items())
        assert config["commit"] == manifest["commit"]
        assert config["device"] == "cuda" and config["runtime"]["gpu"] == "NVIDIA H20"
        receipt = json.loads((cell / "receipt.json").read_text())
        required = {"config.json", "summary.json", "samples.npz", "reference.npz", "certificate.json", "metrics.csv", "run.log", "done"}
        if config["suite"] == "heldout":
            required |= {"learned_parameters.npz", "true_reference.npz", "true_metrics.json"}
        assert receipt["status"] == "completed" and set(receipt["files"]) == required
        assert all(sha256(cell / name) == value for name, value in receipt["files"].items())
        name = asset_name(config)
        reference_hash = sha256(cell / "reference.npz")
        assert reference_hash == hashes.setdefault(name, reference_hash)
        if config["suite"] != "heldout":
            continue
        training_seed = config["training_seed"]
        checkpoint = training_root / f"training_{training_seed}" / "final.pt"
        assert sha256(checkpoint) == config["checkpoint_sha256"] == manifest["checkpoints"][str(training_seed)]
        if training_seed not in networks:
            network = PosteriorMDN()
            network.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
            networks[training_seed] = network.eval()
        with np.load(cell / "learned_parameters.npz") as archive:
            parameters = {key: archive[key] for key in archive.files}
        assert set(parameters) == {"theta", "context", "variance", "means", "weights"}
        data_seed = config["dataset_seed"]
        if data_seed not in contexts:
            contexts[data_seed] = dataset(config["groups"], config["dimension"], data_seed)
        theta, context = contexts[data_seed]
        np.testing.assert_array_equal(parameters["theta"], theta)
        context_error = max(context_error, float(np.max(abs(parameters["context"] - context))))
        np.testing.assert_allclose(parameters["context"], context, rtol=0, atol=context_atol)
        true_hash = sha256(cell / "true_reference.npz")
        assert true_hash == hashes.setdefault(f"true_{data_seed}", true_hash)
        if name not in parameter_cache:
            with torch.no_grad():
                variance, means, log_weights = networks[training_seed](torch.as_tensor(context, dtype=torch.float32))
            for key, predicted in [("variance", variance.numpy()), ("means", means.numpy()), ("weights", log_weights.double().softmax(-1).numpy())]:
                np.testing.assert_allclose(parameters[key], predicted, atol=2e-5, rtol=2e-5)
            parameter_cache[name] = parameters
        else:
            for key, value in parameters.items():
                np.testing.assert_array_equal(value, parameter_cache[name][key])
    assert len(identifiers) == len(set(identifiers))
    return dict(checkpoints_reloaded=len(networks), data_seeds_reconstructed=len(contexts),
                learned_parameter_sets_checked=len(parameter_cache), unique_reference_hashes=len(hashes),
                maximum_context_reconstruction_error=context_error, context_absolute_tolerance=context_atol)


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
        title = {"gaussian": "Gaussian", "mixture": "Mixture", "weak_mixture": "Weak mixture"}[row["family"]]
        ax.set_title(f"{title}, K={row['steps']}")
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
    ax.imshow(grid, vmin=0, vmax=1, cmap=ListedColormap(["#f5cdd0", "#d2e9e3"]), aspect="auto")
    ax.set_xticks(range(len(deltas)), [f"{d:+.1%}" for d in deltas], rotation=45)
    ax.set_yticks(range(6), [f"{f}, K={k}" for f in ["Gaussian", "Mixture", "Weak mixture"] for k in [512, 2048]])
    for i, j in np.ndindex(grid.shape):
        ax.text(j, i, "F" if grid[i, j] else "∞", ha="center", va="center", fontsize=10)
    ax.set_xlabel("Relative error in control component variance")
    fig.tight_layout()
    fig.savefig(output / "tail_sensitivity.pdf")
    fig.savefig(output / "tail_sensitivity.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    references_root = args.output / "references"
    references_root.mkdir()
    manifest = json.loads((args.root / "manifest.json").read_text())
    assert all(sha256(Path(__file__).parent / name) == value for name, value in manifest["sources"].items())
    cells = sorted(path.parent for path in (args.root / "cells").glob("*/solid_done"))
    if not args.partial:
        assert len(cells) == manifest["expected_cells"] == 630
        assert (args.root / "launcher.exit").read_text().strip() == "0"
    input_report = validate_inputs(cells, manifest, args.training_root)
    write_json(args.output / "input_checks.json", input_report)
    assets = {}
    for cell in cells:
        config = json.loads((cell / "config.json").read_text())
        assets.setdefault(asset_name(config), cell)
    jobs = [(name, cell, references_root) for name, cell in assets.items()]
    reference_checks = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for name, checks in executor.map(reference_job, jobs):
            reference_checks[name] = checks
            print(json.dumps(dict(references_completed=len(reference_checks), expected=len(jobs))), flush=True)
    write_json(args.output / "reference_checks.json", reference_checks)
    cache, certs = {}, {}
    rows, differences, metric_differences = [], [], {}
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
        for key, value in metrics.items():
            error = abs(value - summary[key])
            assert error < (2 * 24 / 65536 if key == "w1_mean" else 1e-5), (cell, key, error)
            metric_differences[key] = max(metric_differences.get(key, 0), error)
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
            truth_metrics = independent_metrics(samples, weights, truth)
            recorded_truth = json.loads((cell / "true_metrics.json").read_text())
            for key, value in truth_metrics.items():
                error = abs(value - recorded_truth[key])
                assert error < (2 * 24 / 65536 if key == "w1_mean" else 1e-5), (cell, key, error)
                metric_differences[f"true_{key}"] = max(metric_differences.get(f"true_{key}", 0), error)
            row.update({f"true_{key}": value for key, value in truth_metrics.items()})
            row["learning_w1"] = float(trapezoid(abs(reference["cdf"] - truth["cdf"]), reference["grid"], axis=1).mean())
        rows.append(row)
        if len(rows) % 25 == 0:
            print(json.dumps(dict(cells_audited=len(rows), expected=len(cells))), flush=True)
    names = sorted(set().union(*(row.keys() for row in rows)))
    with (args.output / "cells.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    report = dict(status="partial_audit_passed" if args.partial else "passed", audited_cells=len(rows), expected=630,
                  independent_references=len(assets), independent_certificates=len(certs),
                  max_w1_recomputation_difference=max(differences), metric_maximum_discrepancies=metric_differences,
                  input_checks=input_report, analysis_sha256=sha256(__file__))
    if not args.partial:
        sensitivity = json.loads((args.root / "tail_sensitivity.json").read_text())
        assert len(sensitivity) == 66
        for row in sensitivity:
            variance, _, _ = oracle_parameters(64, 8, row["family"])
            config = dict(suite="tail_error", method="cv", sampling="with_replacement", batch=4,
                          steps=row["steps"], u_max=20.0, diffusion=1.0, control_delta=row["delta"])
            independent = independent_population(variance, config)
            assert independent["finite_normalizer"] == row["finite_normalizer"]
            for left, right in zip(independent["coordinates"], row["coordinates"]):
                assert left["finite_normalizer"] == right["finite_normalizer"]
                if not left["finite_normalizer"]:
                    assert left["failure_step"] == right["failure_step"]
                np.testing.assert_allclose(left["smallest_denominator"], right["smallest_denominator"], rtol=1e-7, atol=1e-9)
        report["sensitivity_configurations_checked"] = len(sensitivity)
        summary = aggregate(rows)
        write_json(args.output / "statistics.json", summary)
        figures(summary, sensitivity, args.output)
    write_json(args.output / "audit.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
