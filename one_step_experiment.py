import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from composition_benchmark import DTYPE, runtime


def run_one_step(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    device = config["device"]
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    n, batch = config["particles"], config["batch"]
    a = torch.tensor([1 / 21, 1 / 11, 1 / 2, 2 / 3], dtype=DTYPE, device=device)
    v = 154 / 355
    h = 0.5
    full_a = 201 / 154
    full_c = 346 / 693
    full_denominator = 2503 / 3195
    exact_log_normalizer = -0.5 * math.log(full_denominator)
    full_multiplier = 1 - h * (full_a + 0.5)
    exact_variance = full_multiplier ** 2 * v / full_denominator + h
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    states, propagated, log_weights = [], [], []
    sum_score_mse = 0.0
    sum_potential_mse = 0.0
    bad_count = 0
    all_max_count = 0
    for start in range(0, n, 65536):
        count = min(n - start, 65536)
        x = math.sqrt(v) * torch.randn(count, device=device, dtype=DTYPE, generator=generator)
        if config["method"] == "full":
            ahat = torch.full_like(x, full_a)
            chat = torch.full_like(x, full_c)
        else:
            indices = torch.randint(4, (count, batch), device=device, generator=generator)
            sample_a = a[indices]
            total = sample_a.sum(1)
            squared = sample_a.square().sum(1)
            ahat = 4 * total / batch
            chat = 0.5 * (16 * (total.square() - squared) / (batch * (batch - 1)) - 4 * squared / batch)
            all_max_count += int((indices == 3).all(1).sum())
        sum_score_mse += float(((ahat - full_a) * x).square().sum())
        sum_potential_mse += float(((chat - full_c) * x.square()).square().sum())
        bad_count += int((1 - 2 * h * chat * v <= 0).sum())
        y = (1 - h * (ahat + 0.5)) * x + math.sqrt(h) * torch.randn(count, device=device, dtype=DTYPE, generator=generator)
        states.append(x)
        propagated.append(y)
        log_weights.append(h * chat * x.square())
    x = torch.cat(states)
    y = torch.cat(propagated)
    lw = torch.cat(log_weights)
    log_total = torch.logsumexp(lw, 0)
    weights = torch.exp(lw - log_total)
    mean = (weights * y).sum()
    variance = (weights * (y - mean).square()).sum()
    log_normalizer = log_total - math.log(n)
    ess = 1 / weights.square().sum()
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    top = torch.topk(weights, min(1024, n)).indices
    sample_x = x[top].cpu().numpy()
    sample_y = y[top].cpu().numpy()
    sample_w = weights[top].cpu().numpy()
    yy = y.cpu().numpy()
    ww = weights.cpu().numpy()
    edges = np.linspace(-12, 12, 2049)
    mass, _ = np.histogram(yy, bins=edges, weights=ww)
    np.savez_compressed(output / "diagnostics.npz", histogram_edges=edges, histogram_mass=mass, largest_weight_x=sample_x, largest_weight_y=sample_y, largest_weights=sample_w)
    summary = dict(config, status="completed", seconds=seconds, log_normalizer=float(log_normalizer), full_exact_log_normalizer=exact_log_normalizer, weighted_mean=float(mean), weighted_variance=float(variance), full_exact_variance=exact_variance, ess_fraction=float(ess) / n, largest_normalized_weight=float(weights.max()), score_mse=sum_score_mse / n, potential_mse=sum_potential_mse / n, nonintegrable_batch_count=bad_count, all_strongest_batch_count=all_max_count, bad_event_lower_probability=4.0 ** -batch, finite_population_normalizer=config["method"] == "full", peak_gpu_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else 0)
    assert np.isfinite([summary["log_normalizer"], summary["weighted_variance"], summary["ess_fraction"]]).all()
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    with (output / "metrics.csv").open("w") as stream:
        row = {k: summary[k] for k in ["particles", "batch", "log_normalizer", "weighted_variance", "ess_fraction", "score_mse", "potential_mse"]}
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (output / "run.log").write_text(json.dumps({"status": "completed", "seconds": seconds, "particles": n}) + "\n")
    (output / "done").write_text("completed\n")
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--method", choices=["full", "unbiased"], default="unbiased")
    parser.add_argument("--particles", type=int, default=65536)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = dict(vars(args), runtime=runtime())
    del config["output"]
    run_one_step(config, args.output)
