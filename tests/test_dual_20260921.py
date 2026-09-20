from pathlib import Path

import numpy as np
from scipy.stats import norm

from run_dual_20260921 import confirmation_settings, joint_metrics, problems, settings


def test_development_and_confirmation_are_disjoint():
    for line, expected in [("current", 8), ("explore", 14)]:
        assert len(settings(line)) == 10
        assert len(problems(line, "development")) == expected
        development = {p["dataset_seed"] for p in problems(line, "development")}
        confirmation = {p["dataset_seed"] for p in problems(line, "confirmation")}
        assert not development & confirmation
        assert not confirmation & set(range(100, 120))
    assert len(problems("current", "confirmation")) == 100
    assert len(problems("explore", "confirmation")) == 50


def test_confirmation_settings_preserve_selected_parameters():
    selected = settings("current")[6]
    assert confirmation_settings("current", {"selected": selected})[-1]["batch"] == selected["batch"]
    selected = settings("explore")[-1]
    rows = confirmation_settings("explore", {"selected": selected})
    assert len(rows) == 5
    assert all(r["table_nodes"] == selected["table_nodes"] for r in rows)
    assert all(r["eta_total"] == selected["eta_total"] for r in rows)


def test_joint_diagnostics_detect_dependence_with_correct_marginals():
    rng = np.random.default_rng(7621)
    x = rng.normal(size=(8192, 2))
    dependent = np.column_stack((x[:, 0], x[:, 0]))
    weights = np.full(len(x), 1 / len(x))
    grid = np.linspace(-8, 8, 8193)
    reference = dict(grid=grid, density=np.tile(norm.pdf(grid), (2, 1)), cdf=np.tile(norm.cdf(grid), (2, 1)))
    independent = joint_metrics(x, weights, reference)
    coupled = joint_metrics(dependent, weights, reference)
    assert independent["sign_joint_tv"] < 0.02
    assert coupled["sign_joint_tv"] > 0.49
    assert independent["standardized_offdiagonal_second_rms"] < 0.03
    assert coupled["standardized_offdiagonal_second_rms"] > 0.95
    assert coupled["sliced_w1_32"] > independent["sliced_w1_32"] * 4


def test_user_secrets_not_in_new_sources():
    files = [Path("run_dual_20260921.py"), Path("docs/DUAL_PROTOCOL_20260921.md")]
    for file in files:
        text = file.read_text()
        assert "Password:" not in text and "sshpass" not in text
