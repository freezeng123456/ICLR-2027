import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad, simpson
from scipy.special import logsumexp
from scipy.stats import norm


def parameters(groups, dimension, family):
    phase = 2 * np.pi * (np.arange(groups)[:, None] + 0.37 * np.arange(dimension)[None, :]) / groups
    variance = 0.5 + 0.15 * np.sin(phase)
    positive_weight = 0.5 + 0.1 * np.sin(phase + 0.7)
    if family == "gaussian":
        center = 0.35 + 0.3 * np.cos(phase)
        minus, plus = center, center
    elif family == "mixture":
        center = 0.1 * np.cos(phase)
        separation = 0.7 + 0.2 * np.cos(2 * phase)
        minus, plus = center - separation, center + separation
    elif family == "weak_mixture":
        variance = 1 / (1 + (4 + 2 * np.sin(phase)) / groups)
        center = 0.4 * np.cos(phase) / groups
        separation = (0.6 + 0.1 * np.cos(phase)) / np.sqrt(groups)
        minus, plus = center - separation, center + separation
    else:
        raise ValueError(family)
    return variance, np.stack((minus, plus), axis=-1), np.stack((1 - positive_weight, positive_weight), axis=-1)


def audit(root, output):
    representatives = {}
    for row in csv.DictReader((root / "summary.csv").open()):
        key = (row["family"], int(row["groups"]), int(row["dimension"]))
        representatives.setdefault(key, root / row["relative_path"])
    rows = []
    for (family, groups, dimension), path in representatives.items():
        variance, means, weights = parameters(groups, dimension, family)
        with np.load(path / "reference.npz") as archive:
            grid, stored_density, stored_cdf = archive["grid"], archive["density"], archive["cdf"]
        for j in range(dimension):
            def log_density(x):
                component = norm.logpdf(x, loc=means[:, j], scale=np.sqrt(variance[:, j, None])) + np.log(weights[:, j])
                return float(logsumexp(component, axis=-1).sum() - (groups - 1) * norm.logpdf(x))

            shift = max(log_density(x) for x in np.linspace(-3, 3, 129))

            def density(x):
                return np.exp(log_density(x) - shift)

            normalization, normalization_error = quad(density, -np.inf, np.inf, epsabs=1e-10, epsrel=1e-10, limit=300)
            assert normalization > 0 and normalization_error / normalization < 1e-8
            mean = quad(lambda x: x * density(x), -np.inf, np.inf, epsabs=1e-10, epsrel=1e-10, limit=300)[0] / normalization
            target_variance = quad(lambda x: (x - mean) ** 2 * density(x), -np.inf, np.inf, epsabs=1e-10, epsrel=1e-10, limit=300)[0] / normalization
            grid_mean = simpson(grid * stored_density[j], x=grid)
            grid_variance = simpson((grid - grid_mean) ** 2 * stored_density[j], x=grid)
            cdf_errors = []
            for probability in [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]:
                location = np.interp(probability, stored_cdf[j], grid)
                adaptive_cdf = quad(density, -np.inf, location, epsabs=1e-10, epsrel=1e-10, limit=300)[0] / normalization
                cdf_errors.append(abs(adaptive_cdf - probability))
            row = {"family": family, "groups": groups, "dimension": dimension, "coordinate": j, "mean_discrepancy": abs(mean - grid_mean), "variance_discrepancy": abs(target_variance - grid_variance), "maximum_cdf_discrepancy": max(cdf_errors), "relative_normalization_error_bound": normalization_error / normalization}
            assert row["mean_discrepancy"] < 1e-7 and row["variance_discrepancy"] < 1e-7 and row["maximum_cdf_discrepancy"] < 2e-6, row
            if family == "gaussian":
                exact_variance = 1 / (1 + (1 / variance[:, j] - 1).sum())
                exact_mean = exact_variance * (means[:, j, 0] / variance[:, j]).sum()
                assert abs(mean - exact_mean) < 1e-10 and abs(target_variance - exact_variance) < 1e-10
            rows.append(row)
    report = {"status": "passed", "families_verified": len(representatives), "coordinates_verified": len(rows), "scope": "Adaptive quadrature over the whole real line using independently constructed SciPy Gaussian-mixture densities; comparison with saved fixed-grid references", "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    audit(arguments.root, arguments.output)
