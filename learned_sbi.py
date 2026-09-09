import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import logsumexp
import torch
from torch import nn

from composition_benchmark import runtime
from composition_extension import ExtensionModel


class PosteriorMDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                                     nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 4))

    def forward(self, context):
        features = torch.stack((context[..., 0] / 3, context[..., 1].log(), context[..., 2]), dim=-1)
        values = self.network(features)
        center = values[..., 0]
        separation = nn.functional.softplus(values[..., 1])
        variance = 0.02 + 0.96 * values[..., 2].sigmoid()
        means = torch.stack((center - separation, center + separation), dim=-1)
        log_weights = torch.stack((nn.functional.logsigmoid(-values[..., 3]), nn.functional.logsigmoid(values[..., 3])), dim=-1)
        return variance, means, log_weights


def simulate(size, generator, device):
    theta = torch.randn(size, device=device, generator=generator)
    sigma = 0.45 + 0.75 * torch.rand(size, device=device, generator=generator)
    offset = 0.3 + torch.rand(size, device=device, generator=generator)
    sign = 2 * torch.randint(2, (size,), device=device, generator=generator) - 1
    y = theta + sign * offset + sigma * torch.randn(size, device=device, generator=generator)
    return theta, torch.stack((y, sigma, offset), dim=-1)


def negative_log_likelihood(model, theta, context):
    variance, means, log_weights = model(context)
    log_components = log_weights - 0.5 * ((theta[..., None] - means).square() / variance[..., None] + (2 * math.pi * variance[..., None]).log())
    return -torch.logsumexp(log_components, dim=-1).mean()


def exact_parameters(context):
    context = np.asarray(context, dtype=np.float64)
    y, sigma, offset = [context[..., i] for i in range(3)]
    denominator = 1 + sigma ** 2
    means = np.stack(((y - offset) / denominator, (y + offset) / denominator), axis=-1)
    log_weights = -0.5 * np.stack(((y - offset) ** 2, (y + offset) ** 2), axis=-1) / denominator[..., None]
    weights = np.exp(log_weights - logsumexp(log_weights, axis=-1, keepdims=True))
    return {"variance": sigma ** 2 / denominator, "means": means, "weights": weights}


def dataset(groups, dimension, seed):
    rng = np.random.default_rng(1000 + seed)
    theta = rng.normal(size=dimension)
    phase = 2 * math.pi * (np.arange(groups)[:, None] + 0.37 * np.arange(dimension)[None]) / groups
    sigma = 0.45 + 0.75 * (0.5 + 0.5 * np.sin(phase))
    offset = 0.3 + 0.5 + 0.5 * np.cos(2 * phase)
    sign = 2 * rng.integers(0, 2, size=(groups, dimension)) - 1
    y = theta[None] + sign * offset + sigma * rng.normal(size=(groups, dimension))
    return theta, np.stack((y, sigma, offset), axis=-1)


def predict(model, context):
    device = next(model.parameters()).device
    with torch.no_grad():
        variance, means, log_weights = model(torch.as_tensor(context, dtype=torch.float32, device=device))
    weights = log_weights.double().softmax(-1).cpu().numpy()
    return {"variance": variance.cpu().double().numpy(), "means": means.cpu().double().numpy(), "weights": weights}


def heldout_kl(model, context):
    grid = np.linspace(-12, 12, 8193)
    true = exact_parameters(context)
    learned = predict(model, context)
    values = []
    for start in range(0, len(context), 32):
        logs = []
        for params in [true, learned]:
            v = params["variance"][start:start + 32]
            m = params["means"][start:start + 32]
            w = params["weights"][start:start + 32]
            log_density = logsumexp(np.log(w)[None] - 0.5 * ((grid[:, None, None] - m[None]) ** 2 / v[None, :, None] + np.log(2 * math.pi * v)[None, :, None]), axis=-1)
            logs.append(log_density)
        density = np.exp(logs[0])
        np.testing.assert_allclose(trapezoid(density, grid, axis=0), 1, atol=1e-10)
        values.extend(trapezoid(density * (logs[0] - logs[1]), grid, axis=0).tolist())
    assert min(values) > -1e-9 and np.isfinite(values).all()
    return {"mean": float(np.mean(values)), "median": float(np.median(values)), "maximum": max(values), "contexts": len(context), "values": values}


def train(output, seed, device, updates=40000, batch=2048):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(10000 + seed)
    validation_generator = torch.Generator(device=device).manual_seed(800000)
    validation_theta, validation_context = simulate(100000, validation_generator, device)
    model = PosteriorMDN().to(device)
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.save(initial, output / "initial.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, updates, eta_min=3e-5)
    config = {"seed": seed, "updates": updates, "batch": batch, "architecture": "3-128-128-128-4 SiLU", "variance": "0.02+0.96*sigmoid", "learning_rate": 3e-4, "learning_rate_final": 3e-5, "checkpoint_selection": "fixed final update", "runtime": runtime()}
    (output / "config.json").write_text(json.dumps(config, indent=2))
    started = time.perf_counter()
    records = []
    for step in range(updates + 1):
        if step % 2000 == 0 or step == updates:
            with torch.no_grad():
                validation_loss = sum(negative_log_likelihood(model, validation_theta[i:i + 2000], validation_context[i:i + 2000]).item() for i in range(0, 100000, 2000)) / 50
            record = {"step": step, "validation_nll": validation_loss, "seconds": time.perf_counter() - started}
            records.append(record)
            print(json.dumps(record), flush=True)
        if step == updates:
            break
        theta, context = simulate(batch, generator, device)
        loss = negative_log_likelihood(model, theta, context)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
    final = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    assert all(torch.isfinite(value).all() for value in final.values())
    changed = sum((final[name] - initial[name]).double().square().sum().item() for name in initial)
    assert changed > 0
    torch.save(final, output / "final.pt")
    kl = heldout_kl(model, validation_context[:1024].cpu().numpy())
    (output / "heldout_kl.json").write_text(json.dumps(kl, indent=2))
    (output / "training.json").write_text(json.dumps(records, indent=2))
    summary = {"status": "completed", "updates": updates, "simulated_training_pairs": updates * batch, "parameter_delta_l2": math.sqrt(changed), "finite_parameters": True, "initial_sha256": hashlib.sha256((output / "initial.pt").read_bytes()).hexdigest(), "final_sha256": hashlib.sha256((output / "final.pt").read_bytes()).hexdigest(), "heldout_kl_mean": kl["mean"], "seconds": time.perf_counter() - started}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "done").write_text("completed\n")


def load_model(path, device="cpu"):
    model = PosteriorMDN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()


def verify_posterior():
    _, contexts = dataset(16, 3, 0)
    params = exact_parameters(contexts)
    grid = np.linspace(-8, 8, 4097)
    errors = []
    for g in [0, 3, 15]:
        for d in range(3):
            y, sigma, offset = contexts[g, d]
            log_likelihood = logsumexp(-0.5 * ((y - grid[:, None] - np.array([-offset, offset])[None]) / sigma) ** 2, axis=-1) - math.log(2 * math.sqrt(2 * math.pi) * sigma)
            direct = log_likelihood - 0.5 * grid ** 2
            v, m, w = params["variance"][g, d], params["means"][g, d], params["weights"][g, d]
            posterior = logsumexp(np.log(w)[None] - 0.5 * (grid[:, None] - m[None]) ** 2 / v, axis=-1) - 0.5 * math.log(2 * math.pi * v)
            difference = direct - posterior
            errors.append(float(np.ptp(difference)))
    assert max(errors) < 1e-10
    return {"status": "passed", "bayes_identity_max_nonconstant_error": max(errors)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=40000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    train(Path(args.root) / f"training_{args.seed}", args.seed, args.device, args.updates)
