import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_composition_references import parameters
from gaussian_integrability import audit_path


def scan(variance, steps, maximum_u=20, diffusion=1):
    groups, dimension = variance.shape
    batches = np.arange(2, groups + 1)
    prefix_length = np.arange(groups + 1)[None, :]
    suffix_length = batches[:, None] - prefix_length
    valid = (suffix_length >= 0) & (suffix_length <= groups)
    suffix_length = np.clip(suffix_length, 0, groups)
    variance_state = np.ones((len(batches), dimension))
    finite = np.ones_like(variance_state, dtype=bool)
    failure_step = np.zeros_like(variance_state, dtype=int)
    grid = np.linspace(np.sqrt(maximum_u), 0, steps + 1) ** 2
    for step, u in enumerate(grid[:-1]):
        h = u - grid[step + 1]
        a = np.sort(1 / (1 - np.exp(-u) + np.exp(-u) * variance) - 1, axis=0)
        sums = np.concatenate((np.zeros((1, dimension)), np.cumsum(a, axis=0)))
        squares = np.concatenate((np.zeros((1, dimension)), np.cumsum(a ** 2, axis=0)))
        total = sums[prefix_length] + sums[-1] - sums[groups - suffix_length]
        squared = squares[prefix_length] + squares[-1] - squares[groups - suffix_length]
        ahat = groups * total / batches[:, None, None]
        chat = groups * (groups - 1) * (total ** 2 - squared) / (2 * batches[:, None, None] * (batches[:, None, None] - 1))
        denominator = 1 - 2 * h * chat * variance_state[:, None, :]
        denominator = np.where(valid[..., None], denominator, np.inf)
        failed_now = finite & (denominator.min(axis=1) <= 0)
        failure_step[failed_now] = step + 1
        finite &= ~failed_now
        multiplier = 1 - h * ((1 + diffusion) * ahat + diffusion) / 2
        safe_denominator = np.where((denominator > 0) & finite[:, None, :], denominator, np.inf)
        proposed = multiplier ** 2 * variance_state[:, None, :] / safe_denominator + diffusion * h
        proposed = np.where(valid[..., None], proposed, -np.inf)
        variance_state = np.where(finite, proposed.max(axis=1), 1)
    return [{"batch": int(batch), "finite_normalizer": bool(finite[i].all()), "failure_step": int(failure_step[i][failure_step[i] > 0].min()) if np.any(failure_step[i] > 0) else 0,
             "coordinate_terminal_variance": variance_state[i].tolist() if finite[i].all() else None}
            for i, batch in enumerate(batches)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    comparisons = 0
    for steps in [16, 64, 128]:
        variance = np.array([[0.25, 0.35], [0.4, 0.5], [0.6, 0.7], [0.85, 0.9]])
        for row in scan(variance, steps):
            direct = [audit_path(1 / variance[:, d] - 1, steps, row["batch"], "without_replacement", exhaustive=True) for d in range(2)]
            assert row["finite_normalizer"] == all(item["finite_normalizer"] for item in direct)
            if row["finite_normalizer"]:
                np.testing.assert_allclose(row["coordinate_terminal_variance"], [item["maximum_terminal_component_variance"] for item in direct], rtol=1e-9)
            else:
                assert row["failure_step"] == min(item["failure_step"] for item in direct if not item["finite_normalizer"])
            comparisons += 1
    rows = []
    for family in ["gaussian", "weak_mixture"]:
        for groups in [16, 64]:
            for dimension in [1, 8]:
                variance, _, _ = parameters(groups, dimension, family)
                for steps in [128, 512, 2048]:
                    for item in scan(variance, steps):
                        rows.append(dict(family=family, groups=groups, dimension=dimension, steps=steps, **item))
    with (args.output / "without_replacement_batch_scan.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"status": "passed", "exhaustive_checks": comparisons, "configurations": len(rows), "scope": "Deterministic certificate scan; ordinary mixture shares Gaussian component variances and certificate. All batch sizes 2..G, U=20, diffusion=1, initial variance=1."}
    (args.output / "without_replacement_batch_scan.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
