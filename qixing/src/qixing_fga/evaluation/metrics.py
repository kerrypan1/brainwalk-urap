"""Metric helpers and random baselines (names preserved from batch_training)."""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from ..models.registry import TaskType


def _compute_metrics(
    y_true,
    y_pred,
    *,
    task: TaskType = "classification",
    n_classes: Optional[int] = None,
) -> dict[str, float]:
    """Compute MAE/MSE/RMSE plus auxiliary classification metrics.

    In ``regression`` mode MAE/MSE/RMSE are computed on the raw continuous
    predictions (aligned with the training objective), while accuracy/f1/
    balanced_accuracy are auxiliary only: predictions are clipped to
    ``[0, n_classes-1]`` and rounded before discrete scoring. In
    ``classification`` mode predictions are simply rounded.
    AUROC is reported when class set and prediction structure allow it;
    otherwise NaN (e.g. continuous tasks, single-class folds).
    """
    mse = mean_squared_error(y_true, y_pred)
    if task == "regression_continuous":
        return {
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "f1_weighted": np.nan,
            "auroc": np.nan,
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mse,
            "rmse": float(mse**0.5),
        }
    if task == "regression":
        if n_classes is None:
            n_classes = int(np.max(y_true)) + 1
        y_pred_class = np.clip(np.rint(y_pred), 0, n_classes - 1).astype(int)
    else:
        y_pred_class = np.rint(y_pred).astype(int)
        if n_classes is None:
            n_classes = int(max(np.max(y_true), np.max(y_pred_class))) + 1

    y_true_int = np.asarray(y_true, dtype=int).ravel()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="y_pred contains classes not in y_true",
            category=UserWarning,
        )
        bal_acc = float(balanced_accuracy_score(y_true_int, y_pred_class))
        f1w = float(f1_score(y_true_int, y_pred_class, average="weighted", zero_division=0))

    return {
        "accuracy": float(accuracy_score(y_true_int, y_pred_class)),
        "balanced_accuracy": bal_acc,
        "f1_weighted": f1w,
        "auroc": _safe_auroc(y_true_int, y_pred_class, n_classes),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(mse**0.5),
    }


def _safe_auroc(y_true: np.ndarray, y_pred_class: np.ndarray, n_classes: int) -> float:
    """One-vs-rest AUROC on discrete predictions; NaN if undefined."""
    unique = np.unique(y_true)
    if unique.size < 2:
        return float("nan")
    try:
        if unique.size == 2:
            # Binary: use predicted class as a crude score (0/1).
            return float(roc_auc_score(y_true, y_pred_class))
        # Multiclass: one-vs-rest on one-hot of predicted class.
        labels = list(range(n_classes))
        y_bin = np.eye(n_classes)[np.clip(y_true, 0, n_classes - 1)]
        y_score = np.eye(n_classes)[np.clip(y_pred_class, 0, n_classes - 1)]
        # Drop columns with no positive in y_true to avoid roc_auc errors.
        present = [i for i in labels if (y_true == i).any()]
        if len(present) < 2:
            return float("nan")
        return float(
            roc_auc_score(
                y_bin[:, present],
                y_score[:, present],
                average="weighted",
                multi_class="ovr",
            )
        )
    except ValueError:
        return float("nan")


def _random_class_predictions(
    size: int, n_classes: int, random_state: int
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    return rng.integers(0, n_classes, size=size)


def expected_random_baseline_accuracy(n_classes: int) -> float:
    return 1.0 / n_classes


def expected_random_baseline_mae(y: pd.Series, n_classes: int) -> float:
    """Expected MAE for uniform random predictions over classes 0..K-1."""
    if n_classes < 2:
        return 0.0
    class_vals = np.arange(n_classes, dtype=float)
    total = 0.0
    for label in range(n_classes):
        count = int((y == label).sum())
        if count == 0:
            continue
        total += count * float(np.mean(np.abs(label - class_vals)))
    return total / len(y)


def random_baseline_fold_metrics(
    y_train: pd.Series,
    y_test: pd.Series,
    n_classes: int,
    random_state: int,
    task: TaskType = "classification",
) -> tuple[dict[str, float], dict[str, float], np.ndarray, np.ndarray]:
    val_pred = _random_class_predictions(len(y_test), n_classes, random_state)
    train_pred = _random_class_predictions(
        len(y_train), n_classes, random_state + 10_000
    )
    return (
        _compute_metrics(y_test, val_pred, task=task, n_classes=n_classes),
        _compute_metrics(y_train, train_pred, task=task, n_classes=n_classes),
        val_pred,
        train_pred,
    )


def paired_bootstrap_delta_mae(
    predictions: pd.DataFrame,
    baseline: str = "baseline_median",
    *,
    n_boot: int = 10_000,
    ci_alpha: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Paired bootstrap CI on ΔMAE (model − baseline), aggregated as a fold mean.

    This is the reported inferential statistic. It stays on an interpretable scale
    (levels of FGA), keeps every sample — a tied prediction contributes 0 rather
    than being dropped — and is comparable across models, including between an
    estimator that emits integers (mord) and one that emits continuous values.

    Aggregation is the equal-weight mean over outer folds, matching the project's
    primary metric (``mae_mean`` in evaluate.py) so the table and this file report
    the same number. Pooling every out-of-fold sample instead would differ
    whenever folds are unequal in size — grouping by participant makes that the
    normal case (91 samples split 18/20/17/18/18) — and on the FGA run the two
    conventions disagree by ~0.008 MAE, which previously showed up as the table
    saying 0.597 while this file said 0.604.

    The bootstrap is fold-stratified: participants are resampled *within* each
    fold, each fold's mean Δ is recomputed, and those are averaged with equal
    weight. Participants are the resampling unit (a participant's visits are not
    independent) and GroupKFold puts each participant in exactly one validation
    fold, so the strata are disjoint. Predictions are resampled; models are never
    refit. Because a fold's mean over resampled rows is ``sum(selected participant
    sums) / sum(selected participant counts)``, each fold costs a pair of matrix
    products rather than a Python loop.

    Do NOT replace this with a paired t-test over the per-fold Δ values, which is
    the obvious-looking way to get a CI once the point estimate is a fold mean.
    The folds share ~80% of their training data, so the five Δ values are not
    independent, and with n=5 the test has almost no power: on the FGA run it
    returns p=0.208 where the fold-stratified bootstrap returns p=0.037 for the
    same effect (Dietterich 1998; Nadeau & Bengio 2003).

    Nested-RFECV runs carry several ``feature_variant`` blocks whose errors differ
    (e.g. all-gait 0.659 vs RFECV-selected 0.582). Those are compared separately,
    each against the baseline of its own variant — pooling them would report a
    number that describes neither.

    Returns one row per model (per feature variant, when present) with delta_mae,
    its CI, and ``p_one_sided`` — the fraction of resamples in which the model
    does not beat the baseline. Note that p_one_sided is one-sided while the CI is
    two-sided, so a model can show p_one_sided < 0.05 and still have a 95% CI that
    crosses zero.
    """
    if predictions is None or predictions.empty:
        return pd.DataFrame()
    needed = {"model", "sample_id", "participant_id", "y_true", "y_pred"}
    if not needed.issubset(predictions.columns):
        return pd.DataFrame()
    if baseline not in set(predictions["model"]):
        return pd.DataFrame()

    rng = np.random.default_rng(random_state)

    has_variants = (
        "feature_variant" in predictions.columns
        and predictions["feature_variant"].nunique(dropna=True) > 1
    )
    if has_variants:
        blocks = [
            (str(v), grp) for v, grp in predictions.groupby("feature_variant", sort=True)
        ]
    else:
        blocks = [(None, predictions)]

    rows: list[dict] = []
    for variant, block in blocks:
        base = block[block["model"] == baseline]
        if base.empty:
            continue
        base_err = (
            (base["y_true"] - base["y_pred"]).abs().groupby(base["sample_id"]).mean()
        )
        base_part = base.groupby("sample_id")["participant_id"].first()
        if "fold" in base.columns:
            base_fold = base.groupby("sample_id")["fold"].first()
        else:
            # No fold column (ad-hoc prediction tables): one stratum, so the fold
            # mean collapses to the pooled mean rather than failing.
            base_fold = pd.Series(0, index=base_err.index, name="fold")

        for name, grp in block.groupby("model"):
            if str(name) == baseline:
                continue
            err = (grp["y_true"] - grp["y_pred"]).abs().groupby(grp["sample_id"]).mean()
            common = base_err.index.intersection(err.index)
            if len(common) < 3:
                continue

            diff = (err.loc[common] - base_err.loc[common]).to_numpy(dtype=float)
            model_err = err.loc[common].to_numpy(dtype=float)
            base_e = base_err.loc[common].to_numpy(dtype=float)
            parts = base_part.loc[common].to_numpy()
            folds = base_fold.loc[common].to_numpy()

            fold_boot: list[np.ndarray] = []
            fold_delta: list[float] = []
            fold_mae: list[float] = []
            fold_base_mae: list[float] = []
            for f in np.unique(folds):
                m = folds == f
                uniq_f, inv_f = np.unique(parts[m], return_inverse=True)
                # Per-participant sums/counts score a resample without regrouping.
                sums = np.bincount(inv_f, weights=diff[m], minlength=len(uniq_f))
                counts = np.bincount(inv_f, minlength=len(uniq_f)).astype(float)
                draws = rng.integers(0, len(uniq_f), size=(n_boot, len(uniq_f)))
                fold_boot.append(sums[draws].sum(axis=1) / counts[draws].sum(axis=1))
                fold_delta.append(float(diff[m].mean()))
                fold_mae.append(float(model_err[m].mean()))
                fold_base_mae.append(float(base_e[m].mean()))

            # Equal weight per fold, matching mae_mean in evaluate.py.
            boot = np.mean(fold_boot, axis=0)
            n_folds = len(fold_delta)
            fold_sd = float(np.std(fold_delta, ddof=1)) if n_folds > 1 else float("nan")

            lo, hi = np.percentile(boot, [100 * ci_alpha / 2, 100 * (1 - ci_alpha / 2)])
            row = {
                "model": str(name),
                "baseline": baseline,
                "n_samples": int(len(common)),
                "n_participants": int(len(np.unique(parts))),
                "n_folds": int(n_folds),
                "mae": float(np.mean(fold_mae)),
                "baseline_mae": float(np.mean(fold_base_mae)),
                "delta_mae": float(np.mean(fold_delta)),
                "delta_mae_fold_sd": fold_sd,
                "delta_mae_ci_lower": float(lo),
                "delta_mae_ci_upper": float(hi),
                "beats_baseline_ci": bool(hi < 0),
                "p_one_sided": float((boot >= 0).mean()),
                "n_bootstrap": int(n_boot),
                "ci_alpha": float(ci_alpha),
            }
            if variant is not None:
                row = {"feature_variant": variant, **row}
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    sort_cols = (["feature_variant"] if has_variants else []) + ["delta_mae"]
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)


def participant_level_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-model participant-level MAE/RMSE from out-of-fold predictions.

    OPT-IN SECONDARY DIAGNOSTIC ONLY — not the reported evaluation metric.
    Predictions (and labels) are averaged within each participant before scoring.
    When a participant's visits carry different FGA labels this averages distinct
    targets, so it must not be used as the headline metric; the reported metric is
    sample-level (see evaluate_models / the ``summary`` block). Kept for the
    single-visit case (where participant == sample) and manual inspection.
    """
    if predictions is None or predictions.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for model_name, grp in predictions.groupby("model"):
        agg = grp.groupby("participant_id").agg(
            y_true=("y_true", "mean"), y_pred=("y_pred", "mean")
        )
        err = agg["y_true"] - agg["y_pred"]
        rows.append(
            {
                "model": str(model_name),
                "n_participants": int(len(agg)),
                "participant_mae": float(err.abs().mean()),
                "participant_rmse": float((err**2).mean() ** 0.5),
            }
        )
    return pd.DataFrame(rows).sort_values("participant_mae").reset_index(drop=True)
