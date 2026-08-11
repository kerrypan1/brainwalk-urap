"""Participant-level CV must not leak the same person into train and val."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from qixing_fga.cv.splits import get_cv_splits
from qixing_fga.features.columns import LEAKY_METADATA_COLUMNS, reject_leaky_columns


def _toy_frame(n_participants: int = 20, rows_per: int = 2, n_classes: int = 3):
    rows = []
    rng = np.random.default_rng(42)
    for i in range(n_participants):
        label = i % n_classes
        for j in range(rows_per):
            rows.append(
                {
                    "participant_id": f"P{i:03d}",
                    "sample_id": f"P{i:03d}_{j}",
                    "y": label,
                    "f0": float(rng.normal()),
                    "f1": float(rng.normal()),
                }
            )
    df = pd.DataFrame(rows)
    X = df[["f0", "f1"]]
    y = df["y"]
    groups = df["participant_id"]
    return X, y, groups


def test_participant_not_split_across_train_val():
    X, y, groups = _toy_frame()
    splits = get_cv_splits(
        X, y, groups, n_splits=5, random_state=42, fold_indices=None, task="classification"
    )
    assert len(splits) == 5
    for train_idx, val_idx in splits:
        train_pids = set(groups.iloc[train_idx])
        val_pids = set(groups.iloc[val_idx])
        assert train_pids.isdisjoint(val_pids)


def test_continuous_groupkfold_no_leakage():
    X, y, groups = _toy_frame()
    y = y.astype(float) + 0.1
    splits = get_cv_splits(
        X,
        y,
        groups,
        n_splits=5,
        random_state=42,
        fold_indices=None,
        task="regression_continuous",
    )
    for train_idx, val_idx in splits:
        assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[val_idx]))


def test_single_visit_view_is_one_row_per_participant():
    """The finalised data definition: FW only, one visit, one sample per person."""
    from qixing_fga.data.loading import select_visit

    df = pd.DataFrame(
        {
            "participant_id": ["A", "A", "B", "B", "C", "C"],
            "video_index": [1, 2, 1, 2, 1, 2],
            "y": [1, 2, 3, 3, 0, 1],
        }
    )
    for visit in (1, 2):
        view = select_visit(df, visit)
        assert len(view) == df["participant_id"].nunique()
        assert view["participant_id"].is_unique
        assert (view["video_index"] == visit).all()
        # Label is that visit's own score, never merged across visits.
        expected = df[df.video_index == visit]["y"].tolist()
        assert view["y"].tolist() == expected

    # "all" must stay a no-op so existing multi-visit runs are unchanged.
    assert select_visit(df, "all").equals(df)
    assert select_visit(df, None).equals(df)


def test_single_visit_view_rejects_unknown_visit():
    from qixing_fga.data.loading import select_visit

    df = pd.DataFrame({"participant_id": ["A"], "video_index": [1], "y": [1]})
    with pytest.raises(ValueError):
        select_visit(df, 3)


def test_leaky_columns_rejected():
    with pytest.raises(ValueError):
        reject_leaky_columns(
            ["f0", "fga_estimate_score", "f1"], context="unit_test"
        )
    assert "fga_estimate_score" in LEAKY_METADATA_COLUMNS
    assert "landmark_file_path" in LEAKY_METADATA_COLUMNS