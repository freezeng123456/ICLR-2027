import pytest

from run_paper_anchor_20260921 import problems, select, settings


def test_protocol_split_and_matrix():
    assert len(settings()) == 9
    assert len({s["setting_id"] for s in settings()}) == 9
    dev, confirm = problems("development"), problems("confirmation")
    assert len(dev) == 8 and len(confirm) == 100
    assert {p["dataset_seed"] for p in dev}.isdisjoint(p["dataset_seed"] for p in confirm)


def test_selection_requires_observed_complete_development():
    with pytest.raises(ValueError, match="eight development"):
        select([])
