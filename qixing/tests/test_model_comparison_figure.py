"""The model-vs-baseline comparison figure and the ΔMAE CI it annotates."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from qixing_fga.evaluation.metrics import paired_bootstrap_delta_mae  # noqa: E402
from qixing_fga.reporting.plots import save_model_comparison_figure  # noqa: E402


def _predictions(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, size=n).astype(float)
    rows = []
    for i in range(n):
        rows.append({"model": "baseline_median", "fold": 1, "feature_variant": "full",
                     "sample_id": f"s{i}", "participant_id": f"p{i}",
                     "y_true": y[i], "y_pred": 2.0})
        rows.append({"model": "good_model", "fold": 1, "feature_variant": "full",
                     "sample_id": f"s{i}", "participant_id": f"p{i}",
                     "y_true": y[i], "y_pred": y[i] + 0.1})
    return pd.DataFrame(rows)


def test_delta_mae_detects_a_better_model():
    tests = paired_bootstrap_delta_mae(_predictions())
    assert len(tests) == 1
    row = tests.iloc[0]
    assert row["model"] == "good_model"
    assert row["delta_mae"] < 0                 # better than baseline
    assert row["delta_mae_ci_upper"] < 0        # and the interval clears zero
    assert bool(row["beats_baseline_ci"]) is True


def test_delta_mae_returns_empty_without_baseline():
    preds = _predictions()
    preds = preds[preds.model != "baseline_median"]
    assert paired_bootstrap_delta_mae(preds).empty


def test_identical_predictor_gives_zero_delta_not_a_crash():
    preds = _predictions()
    same = preds[preds.model == "baseline_median"].copy()
    same["model"] = "clone"
    tests = paired_bootstrap_delta_mae(pd.concat([preds, same], ignore_index=True))
    clone = tests[tests.model == "clone"].iloc[0]
    assert clone["delta_mae"] == 0.0
    assert clone["delta_mae_ci_lower"] == clone["delta_mae_ci_upper"] == 0.0
    assert bool(clone["beats_baseline_ci"]) is False


def test_figure_is_written(tmp_path: Path):
    summary = pd.DataFrame(
        [
            {"model": "baseline_median", "mae_mean": 0.9, "mae_std": 0.2},
            {"model": "good_model", "mae_mean": 0.4, "mae_std": 0.1},
        ]
    )
    per_fold = pd.DataFrame(
        [{"model": "baseline_median", "fold": f, "mae": 0.9} for f in range(1, 6)]
        + [{"model": "good_model", "fold": f, "mae": 0.4} for f in range(1, 6)]
    )
    out = save_model_comparison_figure(
        summary, per_fold, tmp_path, target_name="fga_estimate_score",
        delta_mae=paired_bootstrap_delta_mae(_predictions()),
    )
    assert out is not None and out.is_file() and out.stat().st_size > 0


def test_figure_handles_multiple_feature_variants(tmp_path: Path):
    """Nested-RFECV summaries carry several variants; one chart must still emerge."""
    summary = pd.DataFrame(
        [
            {"model": "baseline_median", "feature_variant": "nested_all_gait",
             "mae_mean": 0.9, "mae_std": 0.2},
            {"model": "m", "feature_variant": "nested_all_gait", "mae_mean": 0.6, "mae_std": 0.1},
            {"model": "baseline_median", "feature_variant": "nested_rfecv",
             "mae_mean": 0.9, "mae_std": 0.2},
            {"model": "m", "feature_variant": "nested_rfecv", "mae_mean": 0.5, "mae_std": 0.1},
        ]
    )
    out = save_model_comparison_figure(
        summary, pd.DataFrame(), tmp_path, target_name="fga_estimate_score"
    )
    assert out is not None and out.is_file()


def test_no_figure_for_empty_summary(tmp_path: Path):
    assert save_model_comparison_figure(
        pd.DataFrame(), pd.DataFrame(), tmp_path, target_name="x"
    ) is None
