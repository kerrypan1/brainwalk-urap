"""ΔMAE paired bootstrap: the reported statistic vs the naive baseline.

Guards the properties the reported statistic depends on: tied predictions must
still count, resampling must be by participant, and feature variants must be
scored separately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from qixing_fga.evaluation.metrics import paired_bootstrap_delta_mae  # noqa: E402


def _predictions(model_preds: dict[str, list[float]], y_true: list[float],
                 participants: list[str]) -> pd.DataFrame:
    rows = []
    for model, preds in model_preds.items():
        for i, (yt, yp, pid) in enumerate(zip(y_true, preds, participants)):
            rows.append({
                "model": model, "fold": 1, "sample_id": f"s{i}",
                "participant_id": pid, "y_true": yt, "y_pred": yp,
            })
    return pd.DataFrame(rows)


def test_delta_mae_matches_direct_mae_difference():
    y = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0]
    parts = ["p1", "p1", "p2", "p2", "p3", "p3"]
    preds = {
        "baseline_median": [2.0] * 6,
        "good": [0.0, 1.0, 2.0, 3.0, 2.0, 1.0],  # perfect
    }
    out = paired_bootstrap_delta_mae(_predictions(preds, y, parts), n_boot=200)
    row = out[out.model == "good"].iloc[0]
    base_mae = float(np.mean(np.abs(np.array(y) - 2.0)))
    assert row["mae"] == 0.0
    assert row["baseline_mae"] == base_mae
    assert row["delta_mae"] == -base_mae
    assert row["delta_mae_ci_upper"] < 0  # a perfect model must clear zero
    assert bool(row["beats_baseline_ci"]) is True


def test_tied_predictions_are_kept_not_discarded():
    """A model identical to the baseline has ΔMAE exactly 0 with a zero-width CI.

    scipy's signed-rank test drops tied pairs, which is what made it incomparable
    across discrete and continuous estimators; the bootstrap must keep them.
    """
    y = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0]
    parts = ["p1", "p1", "p2", "p2", "p3", "p3"]
    preds = {"baseline_median": [2.0] * 6, "same": [2.0] * 6}
    out = paired_bootstrap_delta_mae(_predictions(preds, y, parts), n_boot=200)
    row = out[out.model == "same"].iloc[0]
    assert row["n_samples"] == 6  # nothing dropped
    assert row["delta_mae"] == 0.0
    assert row["delta_mae_ci_lower"] == 0.0
    assert row["delta_mae_ci_upper"] == 0.0
    assert bool(row["beats_baseline_ci"]) is False


def test_resamples_participants_not_rows():
    """Both visits of a participant move together, so the CI reflects 2 units, not 4.

    Rows within a participant are perfectly correlated here: resampling rows would
    sometimes split them and produce intermediate ΔMAE values, while resampling
    participants can only ever yield the two per-participant values or their mean.
    """
    y = [0.0, 0.0, 3.0, 3.0]
    parts = ["p1", "p1", "p2", "p2"]
    # Model is perfect on p1's rows and off by 3 on p2's rows.
    preds = {"baseline_median": [1.0] * 4, "m": [0.0, 0.0, 0.0, 0.0]}
    out = paired_bootstrap_delta_mae(
        _predictions(preds, y, parts), n_boot=4000, random_state=0
    )
    row = out[out.model == "m"].iloc[0]
    assert row["n_participants"] == 2
    assert row["n_samples"] == 4
    # Per-participant ΔMAE: p1 = 0-1 = -1, p2 = 3-2 = +1. Participant resampling
    # can only draw {p1,p1}, {p1,p2}, {p2,p2} -> means of exactly -1, 0, +1.
    assert set(np.round([row["delta_mae_ci_lower"], row["delta_mae_ci_upper"]], 6)) <= {-1.0, 0.0, 1.0}


def test_feature_variants_are_scored_separately():
    """Nested-RFECV runs must not pool variants whose errors differ.

    Pooling all-gait with RFECV-selected reports a MAE that matches neither
    variant, which is what the real nested run exposed (0.659 / 0.582 -> 0.621).
    """
    y = [0.0, 3.0, 0.0, 3.0]
    parts = ["p1", "p1", "p2", "p2"]
    rows = []
    # variant A: model is perfect. variant B: model always predicts the baseline.
    for variant, preds in [("A", [0.0, 3.0, 0.0, 3.0]), ("B", [1.5, 1.5, 1.5, 1.5])]:
        for model, yp in [("baseline_median", [1.5] * 4), ("m", preds)]:
            for i, (yt, p_, pid) in enumerate(zip(y, yp, parts)):
                rows.append({
                    "model": model, "feature_variant": variant, "fold": 1,
                    "sample_id": f"s{i}", "participant_id": pid,
                    "y_true": yt, "y_pred": p_,
                })
    out = paired_bootstrap_delta_mae(pd.DataFrame(rows), n_boot=200)

    assert "feature_variant" in out.columns
    assert set(out["feature_variant"]) == {"A", "B"}
    a = out[(out.feature_variant == "A") & (out.model == "m")].iloc[0]
    b = out[(out.feature_variant == "B") & (out.model == "m")].iloc[0]
    assert a["mae"] == 0.0 and a["delta_mae"] == -1.5   # perfect in A
    assert b["mae"] == 1.5 and b["delta_mae"] == 0.0    # ties the baseline in B


def test_single_variant_output_has_no_variant_column():
    """Runs with one variant keep the original schema, so existing readers work."""
    y = [0.0, 1.0, 2.0, 3.0]
    parts = ["p1", "p1", "p2", "p2"]
    df = _predictions({"baseline_median": [2.0] * 4, "m": [0.0, 1.0, 2.0, 3.0]}, y, parts)
    df["feature_variant"] = "full"
    out = paired_bootstrap_delta_mae(df, n_boot=200)
    assert "feature_variant" not in out.columns
    assert len(out) == 1


def _folded_predictions(rows: list[tuple[str, str, float, float, float]]) -> pd.DataFrame:
    """Rows of (fold, participant_id, y_true, baseline_pred, model_pred)."""
    out = []
    for i, (fold, pid, yt, bp, mp) in enumerate(rows):
        for model, yp in [("baseline_median", bp), ("m", mp)]:
            out.append({
                "model": model, "fold": fold, "sample_id": f"s{i}",
                "participant_id": pid, "y_true": yt, "y_pred": yp,
            })
    return pd.DataFrame(out)


def test_delta_mae_is_the_fold_mean_not_the_pooled_mean():
    """Unequal fold sizes must be weighted equally, matching mae_mean in evaluate.py.

    Fold 1 (2 samples) has Δ = -1 throughout, fold 2 (4 samples) has Δ = 0. The
    fold mean is -0.5; pooling all six samples would give -1/3. Grouping by
    participant makes unequal folds the normal case, so this is not a corner case.
    """
    rows = [
        ("f1", "p1", 0.0, 1.0, 0.0),
        ("f1", "p2", 0.0, 1.0, 0.0),
        ("f2", "p3", 0.0, 1.0, 1.0),
        ("f2", "p4", 0.0, 1.0, 1.0),
        ("f2", "p5", 0.0, 1.0, 1.0),
        ("f2", "p6", 0.0, 1.0, 1.0),
    ]
    row = paired_bootstrap_delta_mae(_folded_predictions(rows), n_boot=200).iloc[0]
    assert row["n_folds"] == 2
    assert row["delta_mae"] == -0.5           # fold mean, not the pooled -1/3
    assert row["mae"] == 0.5                  # fold mean of 0.0 and 1.0, not 4/6
    assert row["baseline_mae"] == 1.0
    assert row["delta_mae_fold_sd"] == pytest.approx(np.std([-1.0, 0.0], ddof=1))


def test_bootstrap_resamples_within_folds_not_across_them():
    """Participants must never be drawn across fold boundaries.

    Every participant in fold 1 has Δ = -1 and every one in fold 2 has Δ = +1, so
    a fold-stratified resample always yields exactly mean(-1, +1) = 0 and the CI
    has zero width. Pooling the four participants into one draw would instead
    range over -1..+1, so a nonzero CI here means the strata leaked.
    """
    rows = [
        ("f1", "p1", 0.0, 1.0, 0.0),  # baseline off by 1, model perfect -> Δ = -1
        ("f1", "p2", 0.0, 1.0, 0.0),
        ("f2", "p3", 0.0, 0.0, 1.0),  # baseline perfect, model off by 1 -> Δ = +1
        ("f2", "p4", 0.0, 0.0, 1.0),
    ]
    row = paired_bootstrap_delta_mae(
        _folded_predictions(rows), n_boot=2000, random_state=0
    ).iloc[0]
    assert row["delta_mae"] == 0.0
    assert row["delta_mae_ci_lower"] == 0.0
    assert row["delta_mae_ci_upper"] == 0.0


def test_missing_fold_column_falls_back_to_a_single_stratum():
    """Ad-hoc prediction tables without a fold column still score, as pooled."""
    y = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0]
    parts = ["p1", "p1", "p2", "p2", "p3", "p3"]
    df = _predictions({"baseline_median": [2.0] * 6, "m": [0.0] * 6}, y, parts)
    df = df.drop(columns=["fold"])
    out = paired_bootstrap_delta_mae(df, n_boot=200)
    assert len(out) == 1
    assert out.iloc[0]["n_folds"] == 1
    assert np.isnan(out.iloc[0]["delta_mae_fold_sd"])


def test_returns_empty_without_baseline():
    y = [0.0, 1.0, 2.0]
    parts = ["p1", "p2", "p3"]
    out = paired_bootstrap_delta_mae(_predictions({"only": [1.0, 1.0, 1.0]}, y, parts))
    assert out.empty


def test_is_deterministic_for_a_fixed_seed():
    y = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0]
    parts = ["p1", "p1", "p2", "p2", "p3", "p3"]
    preds = {"baseline_median": [2.0] * 6, "m": [0.5, 1.2, 2.1, 2.7, 1.9, 1.4]}
    df = _predictions(preds, y, parts)
    a = paired_bootstrap_delta_mae(df, n_boot=500, random_state=7)
    b = paired_bootstrap_delta_mae(df, n_boot=500, random_state=7)
    pd.testing.assert_frame_equal(a, b)
