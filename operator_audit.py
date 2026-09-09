import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
import torch

from composition_benchmark import DTYPE, FactorModel


def observables(x):
    return np.stack((x, x ** 2, np.sin(x)), axis=-1)


def derivatives(x):
    first = np.stack((np.ones_like(x), 2 * x, np.cos(x)), axis=-1)
    second = np.stack((np.zeros_like(x), np.full_like(x, 2), -np.sin(x)), axis=-1)
    return first, second


def paired_values(b, g, coupling):
    if coupling == "shared":
        return b, g
    if coupling == "independent":
        return np.repeat(b, len(g)), np.tile(g, len(b))
    order = np.argsort(b)
    if coupling == "positive":
        return b[order], np.sort(g)
    if coupling == "negative":
        return b[order], np.sort(g)[::-1]
    raise ValueError(coupling)


def audit(output):
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    model = FactorModel(4, 1, "mixture", "cpu")
    batch = torch.tensor(list(itertools.product(range(4), repeat=2)))
    states = np.array([-1.5, -0.5, 0.5, 1.5])
    population = np.array([0.1, 0.2, 0.3, 0.4])
    f = observables(states)
    f1, f2 = derivatives(states)
    eta_f = population @ f
    steps = 2.0 ** -np.arange(4, 16)
    rows = []
    checks = []
    exact_b, exact_g = model.exact(torch.tensor(states[:, None], dtype=DTYPE), 0.3)
    exact_b = exact_b[:, 0].numpy()
    exact_g = exact_g.numpy()
    estimates = []
    for x in states:
        bhat, ghat = model.estimate(torch.full((len(batch), 1), x, dtype=DTYPE), 0.3, 2, None, indices=batch)
        estimates.append((bhat[:, 0].numpy(), ghat.numpy()))
    for coupling in ("shared", "independent", "positive", "negative"):
        samples = [paired_values(b, g, coupling) for b, g in estimates]
        vb = np.array([np.var(b) for b, g in samples])
        vg = np.array([np.var(g) for b, g in samples])
        cbg = np.array([np.mean((b - b.mean()) * (g - g.mean())) for b, g in samples])
        np.testing.assert_allclose([b.mean() for b, g in samples], exact_b, atol=1e-13)
        np.testing.assert_allclose([g.mean() for b, g in samples], exact_g, atol=1e-13)
        for correction in ("none", "exact_variance"):
            drift_term = population @ (0.5 * vb[:, None] * f2)
            cross_term = population @ (cbg[:, None] * f1)
            weight_term = population @ (0.5 * vg[:, None] * (f - eta_f))
            if correction == "exact_variance":
                weight_term[:] = 0
            predicted = drift_term + cross_term + weight_term
            errors = []
            for h in steps:
                exact_weights = population * np.exp(h * exact_g)
                deterministic = exact_weights @ observables(states + h * exact_b) / exact_weights.sum()
                numerator = np.zeros(3)
                denominator = 0.0
                for index, ((b, g), x, probability) in enumerate(zip(samples, states, population)):
                    exponent = h * g
                    if correction == "exact_variance":
                        exponent -= 0.5 * h * h * vg[index]
                    weight = np.exp(exponent)
                    numerator += probability * (weight[:, None] * observables(x + h * b)).mean(0)
                    denominator += probability * weight.mean()
                delta = numerator / denominator - deterministic
                residual = delta - h * h * predicted
                errors.append(float(np.linalg.norm(residual)))
                for i, name in enumerate(("x", "x2", "sin")):
                    rows.append({"coupling": coupling, "correction": correction, "observable": name, "h": h, "difference": delta[i], "predicted_h2_coefficient": predicted[i], "drift_coefficient": drift_term[i], "cross_coefficient": cross_term[i], "weight_coefficient": weight_term[i], "residual": residual[i], "score_mse_proxy": float(population @ vb), "potential_variance": float(population @ vg)})
            # 中等步长避开浮点减法误差，同时检验三阶余项。
            slope = float(np.polyfit(np.log(steps[2:7]), np.log(errors[2:7]), 1)[0])
            assert 2.85 < slope < 3.15, (coupling, correction, slope)
            checks.append({"coupling": coupling, "correction": correction, "remainder_slope": slope, "coefficient": predicted.tolist(), "drift": drift_term.tolist(), "cross": cross_term.tolist(), "weight": weight_term.tolist(), "score_mse_proxy": float(population @ vb), "potential_variance": float(population @ vg)})
    # 改变 coupling 保持两个边际估计量的全部取值与概率不变。
    proxies = [r["score_mse_proxy"] for r in checks]
    variances = [r["potential_variance"] for r in checks]
    assert np.ptp(proxies) < 1e-14
    assert np.ptp(variances) < 1e-14
    with (output / "operator_errors.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"status": "passed", "distribution": "four-point population; exact enumeration of four-group batches", "states": states.tolist(), "population": population.tolist(), "steps": steps.tolist(), "batch_size": 2, "batch_count": len(batch), "checks": checks, "note": "Local operator identity; this is not a particle convergence theorem. Exact-variance correction is a diagnostic with oracle cost."}
    (output / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    audit(args.output)
