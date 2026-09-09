import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def grouped_posterior(values, sizes, prior_mean, prior_var, between_var, noise_var):
    means = np.array([part.mean() for part in np.split(values, np.cumsum(sizes)[:-1])])
    information = 1.0 / (between_var + noise_var / np.asarray(sizes))
    precision = 1.0 / prior_var + information.sum()
    mean = (prior_mean / prior_var + information @ means) / precision
    return mean, 1.0 / precision, information


def joint_gaussian_posterior(values, sizes, prior_mean, prior_var, between_var, noise_var):
    # 独立计算完整观测协方差，包含总体参数的先验随机性。
    labels = np.repeat(np.arange(len(sizes)), sizes)
    same_group = labels[:, None] == labels[None, :]
    covariance = prior_var + between_var * same_group + noise_var * np.eye(len(values))
    cross = np.full(len(values), prior_var)
    solved = np.linalg.solve(covariance, np.column_stack([values - prior_mean, cross]))
    return prior_mean + cross @ solved[:, 0], prior_var - cross @ solved[:, 1]


def run(seed):
    rng = np.random.default_rng(seed)
    posterior_checks = []
    partitions = [[1] * 64, [8] * 8, [1, 1, 2, 4, 8, 16, 32], [1, 3, 7, 13]]
    for sizes in partitions:
        for between_var in [0.0, 0.01, 1.0, 100.0]:
            prior_mean, prior_var, noise_var = 0.7, 9.0, 0.4
            theta = rng.normal(prior_mean, np.sqrt(prior_var))
            offsets = rng.normal(0, np.sqrt(between_var), len(sizes))
            values = theta + np.repeat(offsets, sizes) + rng.normal(0, np.sqrt(noise_var), sum(sizes))
            grouped_mean, grouped_var, _ = grouped_posterior(
                values, sizes, prior_mean, prior_var, between_var, noise_var
            )
            joint_mean, joint_var = joint_gaussian_posterior(
                values, sizes, prior_mean, prior_var, between_var, noise_var
            )
            np.testing.assert_allclose([grouped_mean, grouped_var], [joint_mean, joint_var], rtol=1e-10, atol=1e-10)
            posterior_checks.append({
                "group_sizes": sizes, "between_var": between_var,
                "mean_absolute_difference": float(abs(grouped_mean - joint_mean)),
                "variance_absolute_difference": float(abs(grouped_var - joint_var)),
            })

    composition_checks = []
    for groups in [1, 4, 16, 64, 256]:
        per_group_information = 8.0 / 9.0
        information = groups * per_group_information
        precision = 1.0 / 9.0 + information
        bias, error_var = 0.03, 0.01
        shift = bias * information / precision
        posterior_sd = precision ** -0.5
        expected_kl = bias ** 2 * information ** 2 / (2 * precision)

        def integrand(z):
            value = posterior_sd * z
            return norm.pdf(z) * (
                norm.logpdf(value, scale=posterior_sd)
                - norm.logpdf(value, loc=shift, scale=posterior_sd)
            )

        numerical_kl, integration_error = quad(integrand, -12, 12, epsabs=1e-11)
        np.testing.assert_allclose(numerical_kl, expected_kl, rtol=1e-9, atol=1e-10)
        trials = 20000
        errors = rng.normal(bias, np.sqrt(error_var), size=(trials, groups))
        sample_kl = (per_group_information * errors.sum(axis=1)) ** 2 / (2 * precision)
        mean_kl = float(sample_kl.mean())
        standard_error = float(sample_kl.std(ddof=1) / np.sqrt(trials))
        expected_random_kl = expected_kl + groups * per_group_information ** 2 * error_var / (2 * precision)
        assert abs(mean_kl - expected_random_kl) < 6 * standard_error
        composition_checks.append({
            "groups": groups, "measurements_per_group": 8,
            "constructed_center_bias": bias, "constructed_center_error_variance": error_var,
            "information": information, "systematic_kl_formula": expected_kl,
            "systematic_kl_quadrature": numerical_kl, "quadrature_error_estimate": integration_error,
            "random_error_expected_kl": expected_random_kl, "random_error_monte_carlo_kl": mean_kl,
            "monte_carlo_standard_error": standard_error, "monte_carlo_trials": trials,
        })
    return {
        "status": "PASS", "evidence_type": "analytic_constructed_example_not_trained_model",
        "seed": seed, "python": platform.python_version(), "numpy": np.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "posterior_checks": posterior_checks, "composition_checks": composition_checks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = run(args.seed)
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "posterior_cases": len(report["posterior_checks"]),
                      "composition_cases": len(report["composition_checks"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
