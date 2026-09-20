import json

import numpy as np
from scipy.stats import norm

from analyze_dual_20260921 import aggregate, independent_joint
from run_dual_20260921 import joint_metrics


def test_independent_joint_agrees_with_direct_diagnostics():
    rng = np.random.default_rng(5381)
    samples = rng.normal(size=(5000, 3))
    samples[:, 1] += 0.5 * samples[:, 0]
    weights = rng.uniform(size=len(samples))
    weights /= weights.sum()
    grid = np.linspace(-8, 8, 8193)
    reference = dict(grid=grid, density=np.tile(norm.pdf(grid), (3, 1)), cdf=np.tile(norm.cdf(grid), (3, 1)))
    actual = independent_joint(samples, weights, reference)
    expected = joint_metrics(samples, weights, reference)
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-12, rtol=1e-12)


def test_crossed_bootstrap_gate_has_known_exact_contrast(tmp_path):
    # 确定性构造验证统计公式，不代表任何实验性能。
    root = tmp_path / "confirmation" / "cells"
    cell_id = 0
    for t in range(5):
        for d in range(10):
            for method in ["full", "certified"]:
                cell = root / f"cell_{cell_id:04d}"
                cell.mkdir(parents=True)
                value = 0.02 + t * 0.001 + d * 0.0001
                row = dict(training_seed=t, dataset_seed=d, setting_id=method, family="learned",
                           w1_mean=value, true_w1_mean=value + 0.001,
                           seconds_including_preparation=10.0 if method == "full" else 5.0,
                           sign_joint_tv=0.02, standardized_offdiagonal_second_rms=0.03,
                           sliced_w1_32=0.01, resampling_count=100)
                (cell / "summary.json").write_text(json.dumps(row))
                cell_id += 1
    result, rows = aggregate(tmp_path, "confirmation")
    comparison = result["paired_contrasts"]["certified"]
    assert len(rows) == 100
    assert comparison["declared_accuracy_cost_gate"]
    assert comparison["w1_difference_upper95"] == 0
    assert comparison["time_ratio_ci95"] == [0.5, 0.5]
