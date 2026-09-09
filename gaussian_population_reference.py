import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def normal_wasserstein(mean_difference, std_difference):
    scale = np.abs(std_difference)
    output = np.abs(mean_difference).copy()
    selected = scale > 1e-14
    z = mean_difference[selected] / scale[selected]
    output[selected] = 2 * scale[selected] * norm.pdf(z) + mean_difference[selected] * (2 * norm.cdf(z) - 1)
    return output


def population(groups, dimension, steps, maximum_u=20, diffusion=1):
    phase = 2 * np.pi * (np.arange(groups)[:, None] + 0.37 * np.arange(dimension)[None, :]) / groups
    factor_variance = 0.5 + 0.15 * np.sin(phase)
    factor_mean = 0.35 + 0.3 * np.cos(phase)
    target_variance = 1 / (1 + (1 / factor_variance - 1).sum(0))
    target_mean = target_variance * (factor_mean / factor_variance).sum(0)
    state_mean, state_variance = np.zeros(dimension), np.ones(dimension)
    grid = np.linspace(np.sqrt(maximum_u), 0, steps + 1) ** 2
    for k, (u, unext) in enumerate(zip(grid[:-1], grid[1:])):
        h = u - unext
        alpha = np.exp(-u / 2)
        noised_variance = 1 - alpha ** 2 + alpha ** 2 * factor_variance
        a = 1 / noised_variance - 1
        d = alpha * factor_mean / noised_variance
        precision, intercept = a.sum(0), d.sum(0)
        quadratic = (precision ** 2 - (a * a).sum(0)) / 2
        linear = -precision * intercept + (a * d).sum(0)
        denominator = 1 - 2 * h * quadratic * state_variance
        if np.min(denominator) <= 0:
            return {"groups": groups, "dimension": dimension, "steps": steps, "status": "infinite", "failure_step": k + 1, "failure_denominator": float(denominator.min())}
        tilted_mean = (state_mean + h * linear * state_variance) / denominator
        tilted_variance = state_variance / denominator
        multiplier = 1 - h * ((1 + diffusion) * precision + diffusion) / 2
        state_mean = multiplier * tilted_mean + h * (1 + diffusion) * intercept / 2
        state_variance = multiplier ** 2 * tilted_variance + h * diffusion
    w1 = normal_wasserstein(state_mean - target_mean, np.sqrt(state_variance) - np.sqrt(target_variance))
    return {"groups": groups, "dimension": dimension, "steps": steps, "status": "finite", "population_w1_mean": float(w1.mean()), "population_mean_error": float(np.abs(state_mean - target_mean).mean()), "population_variance_relative_error": float(np.abs(state_variance / target_variance - 1).mean()), "target_mean": target_mean.tolist(), "target_variance": target_variance.tolist(), "euler_mean": state_mean.tolist(), "euler_variance": state_variance.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    rows = [population(groups, dimension, steps) for groups, dimension, steps in itertools.product([16, 64], [1, 8], [128, 512, 2048])]
    # 单因子没有 Feynman--Kac potential；解析递推应给出有限 Gaussian law。
    single = population(1, 1, 32768)
    assert single["status"] == "finite" and single["population_w1_mean"] < 1e-4
    report = {"initial_mean": 0, "initial_variance": 1, "maximum_u": 20, "diffusion": 1, "scope": "Exact full-factor Gaussian weighted Euler population moments, including finite-noise initialization and time-discretization error", "single_factor_high_resolution_w1": single["population_w1_mean"], "rows": rows}
    (arguments.output / "gaussian_population.json").write_text(json.dumps(report, indent=2))
    frame = pd.DataFrame([{key: value for key, value in row.items() if not isinstance(value, list)} for row in rows])
    frame.to_csv(arguments.output / "gaussian_population.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
