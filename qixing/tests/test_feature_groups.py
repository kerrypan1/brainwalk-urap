"""Ablation groups must exactly partition the model feature pool."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from qixing_fga.features.columns import MODEL_FEATURE_COLUMNS  # noqa: E402
from utils.feature_groups import (  # noqa: E402
    FEATURE_GROUPS,
    GROUP_LABELS,
    groups_present,
    validate_groups,
)


def test_groups_partition_feature_pool():
    validate_groups()


def test_no_feature_in_two_groups():
    assigned = [f for feats in FEATURE_GROUPS.values() for f in feats]
    assert len(assigned) == len(set(assigned))


def test_every_group_has_a_label():
    assert set(GROUP_LABELS) == set(FEATURE_GROUPS)


def test_groups_present_filters_to_available_columns():
    subset = MODEL_FEATURE_COLUMNS[:10]
    present = groups_present(subset)
    flat = [f for feats in present.values() for f in feats]
    assert set(flat).issubset(set(subset))
    assert flat, "expected at least one group to survive the filter"
