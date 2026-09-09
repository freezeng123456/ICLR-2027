import argparse
import inspect
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import exp_conditioning as gp
import exp_jump as jump
from identifiability import gp_posterior
from prior_jump import LEVELS, predict_single, sample_task
from train_repro import sha256


def load_checkpoint(path):
    # 仓库内已核验哈希的检查点允许使用 SCNet 的 PyTorch 1.12 读取。
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = True
    state = torch.load(path, **options)
    width = state["x_enc.weight"].shape[0]
    model = gp.PFN(width, max(1, width // 32))
    model.load_state_dict(state)
    return model.eval()


def predict(model, xc, yc, xq, device):
    x = torch.as_tensor(np.concatenate([xc, xq], axis=1), dtype=torch.float32, device=device)
    y = torch.as_tensor(np.concatenate([yc, np.zeros_like(xq)], axis=1), dtype=torch.float32, device=device)
    with torch.no_grad():
        mean, logv = model(x, y, xc.shape[1])
    return mean.cpu().numpy(), logv.exp().cpu().numpy()


def make_tasks(prior, n, count, seed):
    rng = np.random.default_rng(seed)
    family = gp if prior == "gp" else jump
    rows = []
    for _ in range(count):
        latent, sigma = family.draw_latent(rng)
        design = family.DESIGNS[int(rng.integers(2))]
        xc, xq = family.draw_design(rng, n, design), rng.uniform(-1, 1, 16)
        x = np.concatenate([xc, xq])
        y = gp.sample_gp(rng, x, latent) + sigma * rng.standard_normal(len(x)) if prior == "gp" else sample_task(rng, x, latent, sigma)
        if prior == "gp":
            mu, var = gp_posterior(xc, y[:n], xq, latent, sigma)
        else:
            weights, _ = predict_single(xc, y[:n], xq, latent, sigma)
            mu = weights @ LEVELS
            var = weights @ (LEVELS ** 2) + sigma ** 2 - mu ** 2
        rows.append((xc, y[:n], xq, y[n:], mu, var, latent, sigma, design))
    fields = ["xc", "yc", "xq", "yq", "latent_mean", "latent_var", "latent", "sigma", "design"]
    return {key: np.asarray([row[i] for row in rows]) for i, key in enumerate(fields)}


def evaluate(model, data, device, batch_size=32):
    outputs = {key: [] for key in ["mean", "variance", "stability", "loo_nll"]}
    n = data["xc"].shape[1]
    for start in range(0, len(data["xc"]), batch_size):
        xc, yc, xq = [data[key][start:start + batch_size] for key in ["xc", "yc", "xq"]]
        mean, variance = predict(model, xc, yc, xq, device)
        shifts, losses = [], []
        for removed in range(n):
            keep = np.arange(n) != removed
            queries = np.concatenate([xq, xc[:, removed:removed + 1]], axis=1)
            reduced_mean, reduced_var = predict(model, xc[:, keep], yc[:, keep], queries, device)
            shifts.append((reduced_mean[:, :-1] - mean) ** 2)
            losses.append(0.5 * (np.log(reduced_var[:, -1]) + (yc[:, removed] - reduced_mean[:, -1]) ** 2 / reduced_var[:, -1]))
        for key, value in zip(outputs, [mean, variance, np.mean(shifts, axis=0), np.mean(losses, axis=0)]):
            outputs[key].append(value)
    return {key: np.concatenate(values) for key, values in outputs.items()}


def curve(score, risk):
    order = np.argsort(score.ravel(), kind="stable")
    coverage = np.arange(1, 11) / 10
    counts = np.ceil(coverage * len(order)).astype(int)
    return np.cumsum(risk.ravel()[order])[counts - 1] / counts


def fitted_scores(features, labels, split):
    flat = features[:split].reshape(-1, features.shape[-1])
    center, scale = flat.mean(axis=0), np.maximum(flat.std(axis=0), 1e-8)
    transformed = (features - center) / scale
    train = transformed[:split].reshape(-1, features.shape[-1])
    train = np.column_stack([np.ones(len(train)), train])
    target = np.log(np.maximum(labels[:split].ravel(), 1e-8))
    penalty = np.eye(train.shape[1]) * 1.0
    penalty[0, 0] = 0
    weights = np.linalg.solve(train.T @ train + penalty, train.T @ target)
    score = weights[0] + transformed @ weights[1:]
    return score, {"center": center.tolist(), "scale": scale.tolist(), "weights": weights.tolist()}


def analyze(data, predictions, prior, split, bootstrap, seed):
    result = {}
    distance = np.min(np.abs(data["xq"][:, :, None] - data["xc"][:, None, :]), axis=2)
    large_gp, large_jump = predictions["gp128"], predictions["jump128"]
    chosen_gp = large_gp["loo_nll"] < large_jump["loo_nll"]
    routed = np.where(chosen_gp[:, None], large_gp["mean"], large_jump["mean"])
    routed_risk = data["latent_var"] + (routed - data["latent_mean"]) ** 2
    result["routing"] = {"accuracy": float(np.mean(chosen_gp[split:] == (prior == "gp"))),
                         "routed_risk": float(routed_risk[split:].mean()),
                         "family_risks": {name: float((data["latent_var"] + (p["mean"] - data["latent_mean"]) ** 2)[split:].mean()) for name, p in predictions.items() if name.endswith("128")}}
    for name in [prior + "64", prior + "128"]:
        p = predictions[name]
        risk = data["latent_var"] + (p["mean"] - data["latent_mean"]) ** 2
        empirical = (data["yq"] - p["mean"]) ** 2
        baseline = np.stack([np.log(p["variance"]), np.log(distance + 1e-5), np.abs(p["mean"])], axis=-1)
        augmented = np.concatenate([baseline, np.log(p["stability"][:, :, None] + 1e-8)], axis=-1)
        base_score, base_fit = fitted_scores(baseline, empirical, split)
        aug_score, aug_fit = fitted_scores(augmented, empirical, split)
        scores = {"variance": p["variance"], "stability": p["stability"], "distance": distance,
                  "calibrated_baseline": base_score, "calibrated_stability": aug_score}
        curves = {key: curve(value[split:], risk[split:]).tolist() for key, value in scores.items()}
        empirical_curves = {key: curve(value[split:], empirical[split:]).tolist() for key, value in scores.items()}
        rng = np.random.default_rng(seed)
        differences = []
        for _ in range(bootstrap):
            idx = rng.integers(split, len(risk), len(risk) - split)
            a, b = curve(base_score[idx], risk[idx]).mean(), curve(aug_score[idx], risk[idx]).mean()
            differences.append((a - b) / a)
        base_auc, aug_auc = np.mean(curves["calibrated_baseline"]), np.mean(curves["calibrated_stability"])
        result[name] = {"risk_curves": curves, "empirical_curves": empirical_curves,
                        "relative_improvement": float((base_auc - aug_auc) / base_auc),
                        "ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
                        "fits": {"baseline": base_fit, "augmented": aug_fit}}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", choices=["gp", "jump"], required=True)
    parser.add_argument("--n-context", type=int, choices=[8, 24], required=True)
    parser.add_argument("--tasks", type=int, default=512)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tasks < 8 or args.tasks % 2:
        parser.error("tasks 必须是至少 8 的偶数")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    started = time.time()
    repo = Path(__file__).resolve().parent
    files = {"gp64": "pfn_cond_w64.pt", "gp128": "pfn_cond_40k.pt", "jump64": "pfn_jump_w64.pt", "jump128": "pfn_jump_40k.pt"}
    config = {**vars(args), "output": str(args.output), "python": sys.executable, "torch": torch.__version__,
              "hostname": socket.gethostname(), "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "source_sha256": {p: sha256(repo / p) for p in ["pilot_context.py", "train_repro.py", "exp_conditioning.py", "exp_jump.py", "prior_jump.py", "identifiability.py"]},
              "checkpoint_sha256": {key: sha256(repo / path) for key, path in files.items()},
              "risk_reference": "exact conditional on sampled latent parameters; no hyperparameter quadrature",
              "fit_labels": "held-out validation squared observation error", "validation_tasks": args.tasks // 2,
              "coverage_grid": (np.arange(1, 11) / 10).tolist(), "bootstrap_unit": "test task"}
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    data = make_tasks(args.prior, args.n_context, args.tasks, args.seed)
    predictions = {}
    for name, filename in files.items():
        if name.endswith("64") and not name.startswith(args.prior):
            continue
        model = load_checkpoint(repo / filename).to(args.device)
        predictions[name] = evaluate(model, data, args.device)
        print(json.dumps({"model": name, "tasks": args.tasks, "elapsed": time.time() - started}), flush=True)
    arrays = {**data, **{f"{name}_{key}": value for name, pred in predictions.items() for key, value in pred.items()}}
    if not all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind != "U"):
        raise ValueError("实验记录存在非有限数值")
    np.savez_compressed(args.output / "records.npz", **arrays)
    summary = analyze(data, predictions, args.prior, args.tasks // 2, args.bootstrap, args.seed)
    summary["elapsed_seconds"] = time.time() - started
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (args.output / "done").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
