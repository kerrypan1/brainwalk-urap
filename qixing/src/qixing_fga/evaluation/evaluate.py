"""Cross-validated model evaluation and console summaries."""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ..cv.splits import get_cv_splits
from ..features.columns import reject_leaky_columns
from ..models.registry import TaskType, maybe_tune
from .bootstrap import per_fold_bootstrap_mae_rmse
from .metrics import (
    _compute_metrics,
    expected_random_baseline_accuracy,
    expected_random_baseline_mae,
    random_baseline_fold_metrics,
)


def _prediction_rows(
    model_name: str,
    fold_idx: int,
    feature_variant: str,
    y_true,
    y_pred,
    test_idx: list[int],
    groups: pd.Series,
    ids: Optional[pd.Series],
) -> list[dict]:
    """Per-sample out-of-fold prediction records for predictions.csv / plots."""
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()
    pids = groups.iloc[test_idx].to_numpy()
    sids = ids.iloc[test_idx].to_numpy() if ids is not None else pids
    rows: list[dict] = []
    for i in range(len(y_true_arr)):
        rows.append(
            {
                "model": model_name,
                "fold": fold_idx,
                "feature_variant": feature_variant,
                "sample_id": sids[i],
                "participant_id": pids[i],
                "y_true": float(y_true_arr[i]),
                "y_pred": float(y_pred_arr[i]),
            }
        )
    return rows


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    preprocessor: ColumnTransformer,
    models: dict[str, object],
    n_splits: int,
    random_state: int,
    fold_indices: Optional[list[tuple[list[int], list[int]]]],
    bootstrap_by_fold: Optional[dict[str, object]],
    ci_alpha: float,
    task: TaskType = "classification",
    tune: str = "none",
    tune_splits: int = 3,
    ids: Optional[pd.Series] = None,
    feature_variant: str = "full",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(groups) != len(y):
        raise ValueError("groups length must match y length.")

    reject_leaky_columns(list(X.columns), context="evaluate_models")
    splits = get_cv_splits(X, y, groups, n_splits, random_state, fold_indices, task)
    rows = []
    pred_rows: list[dict] = []
    has_bootstrap = bootstrap_by_fold is not None
    is_continuous = task == "regression_continuous"
    n_classes = 0 if is_continuous else int(y.max()) + 1

    def _build_row(
        model_name: str,
        fold_idx: int,
        val_metrics: dict[str, float],
        train_metrics: dict[str, float],
        val_y_true,
        val_y_pred,
        test_idx: list[int],
    ):
        row = {
            "model": model_name,
            "fold": fold_idx,
            **val_metrics,
            "train_accuracy": train_metrics["accuracy"],
            "train_f1_weighted": train_metrics["f1_weighted"],
            "train_mae": train_metrics["mae"],
            "train_mse": train_metrics["mse"],
            "train_rmse": train_metrics["rmse"],
            "mae_gap": val_metrics["mae"] - train_metrics["mae"],
            "mae_gap_ratio": (
                (val_metrics["mae"] - train_metrics["mae"]) / val_metrics["mae"]
                if val_metrics["mae"] > 0
                else np.nan
            ),
        }

        if has_bootstrap:
            fold_key = f"fold_{fold_idx - 1}"
            if fold_key not in bootstrap_by_fold:
                raise KeyError(f"Bootstrap npz missing key: {fold_key}")
            boot_stats = per_fold_bootstrap_mae_rmse(
                y_val=val_y_true,
                y_pred=val_y_pred,
                val_idx=test_idx,
                groups=groups.to_numpy(),
                boot_matrix=bootstrap_by_fold[fold_key],
                ci_alpha=ci_alpha,
            )
            row.update(
                {
                    "bootstrap_n": boot_stats["n_bootstrap"],
                    "bootstrap_sample_size_mean": boot_stats[
                        "bootstrap_sample_size_mean"
                    ],
                    "bootstrap_sample_size_min": boot_stats[
                        "bootstrap_sample_size_min"
                    ],
                    "bootstrap_sample_size_max": boot_stats[
                        "bootstrap_sample_size_max"
                    ],
                    "mae_boot_mean": boot_stats["mae_boot_mean"],
                    "mae_boot_std": boot_stats["mae_boot_std"],
                    "mae_ci_lower": boot_stats["mae_ci_lower"],
                    "mae_ci_upper": boot_stats["mae_ci_upper"],
                    "mse_boot_mean": boot_stats["mse_boot_mean"],
                    "mse_boot_std": boot_stats["mse_boot_std"],
                    "mse_ci_lower": boot_stats["mse_ci_lower"],
                    "mse_ci_upper": boot_stats["mse_ci_upper"],
                    "rmse_boot_mean": boot_stats["rmse_boot_mean"],
                    "rmse_boot_std": boot_stats["rmse_boot_std"],
                    "rmse_ci_lower": boot_stats["rmse_ci_lower"],
                    "rmse_ci_upper": boot_stats["rmse_ci_upper"],
                    "ci_alpha": boot_stats["ci_alpha"],
                }
            )

        return row

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        median_value = float(y_train.median())
        val_pred = np.full(len(y_test), median_value)
        train_pred = np.full(len(y_train), median_value)
        val_metrics = _compute_metrics(y_test, val_pred, task=task, n_classes=n_classes)
        train_metrics = _compute_metrics(y_train, train_pred, task=task, n_classes=n_classes)
        rows.append(
            _build_row(
                "baseline_median",
                fold_idx,
                val_metrics,
                train_metrics,
                y_test,
                val_pred,
                test_idx,
            )
        )
        pred_rows.extend(
            _prediction_rows(
                "baseline_median", fold_idx, feature_variant,
                y_test, val_pred, test_idx, groups, ids,
            )
        )

        if not is_continuous:
            # Uniform-random-over-classes baseline is only meaningful for
            # discrete targets; skip it for continuous regression.
            val_metrics_r, train_metrics_r, val_pred_r, train_pred_r = (
                random_baseline_fold_metrics(
                    y_train,
                    y_test,
                    int(y.nunique()),
                    random_state + fold_idx + 20_000,
                    task=task,
                )
            )
            rows.append(
                _build_row(
                    "baseline_random",
                    fold_idx,
                    val_metrics_r,
                    train_metrics_r,
                    y_test,
                    val_pred_r,
                    test_idx,
                )
            )

        groups_train = groups.iloc[train_idx]
        for model_name, model in models.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", model),
                ]
            )

            estimator = maybe_tune(
                pipeline,
                model_name,
                task,
                y_train,
                groups_train,
                tune=tune,
                tune_splits=tune_splits,
                random_state=random_state + fold_idx,
            )
            try:
                estimator.fit(X_train, y_train)
                val_pred = estimator.predict(X_test)
                train_pred = estimator.predict(X_train)
            except Exception as exc:
                # Some estimators (ordinal mord, XGBoost) require every class to
                # be present and contiguous in the training fold; with rare
                # classes a fold may drop one. Skip this model for this fold
                # rather than aborting the whole outcome (safety net for the
                # heterogeneous multi-outcome sweep; no effect when models fit).
                warnings.warn(
                    f"Model '{model_name}' failed on fold {fold_idx}; skipping "
                    f"this fold. ({type(exc).__name__}: {exc})"
                )
                continue
            val_metrics = _compute_metrics(y_test, val_pred, task=task, n_classes=n_classes)
            train_metrics = _compute_metrics(y_train, train_pred, task=task, n_classes=n_classes)

            rows.append(
                _build_row(
                    model_name,
                    fold_idx,
                    val_metrics,
                    train_metrics,
                    y_test,
                    val_pred,
                    test_idx,
                )
            )
            pred_rows.extend(
                _prediction_rows(
                    model_name, fold_idx, feature_variant,
                    y_test, val_pred, test_idx, groups, ids,
                )
            )

    results = pd.DataFrame(rows)
    summary = (
        results.groupby("model")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            f1_weighted_mean=("f1_weighted", "mean"),
            f1_weighted_std=("f1_weighted", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            train_mae_mean=("train_mae", "mean"),
            train_mae_std=("train_mae", "std"),
            mae_gap_mean=("mae_gap", "mean"),
            mae_gap_std=("mae_gap", "std"),
            mae_gap_ratio_mean=("mae_gap_ratio", "mean"),
            mae_gap_ratio_std=("mae_gap_ratio", "std"),
        )
        .reset_index()
    )

    predictions = pd.DataFrame(pred_rows)
    return results, summary, predictions


def print_overfitting_diagnosis(
    summary: pd.DataFrame, gap_ratio_warn: float = 0.25
) -> None:
    print("\n=== Overfitting diagnosis ===")
    if "mae_gap_ratio_mean" not in summary.columns:
        print("Train/validation gap metrics not available.")
        return

    baseline_rows = summary.loc[summary["model"] == "baseline_median", "mae_mean"]
    baseline_mae = float(baseline_rows.iloc[0]) if len(baseline_rows) else np.nan

    diag = summary[
        ~summary["model"].isin(["baseline_median", "baseline_random"])
    ].copy()
    diag = diag.sort_values("mae_gap_ratio_mean", ascending=False, na_position="last")

    for _, row in diag.iterrows():
        warn = ""
        ratio = row["mae_gap_ratio_mean"]
        if pd.notna(ratio) and ratio > gap_ratio_warn:
            warn = " WARNING: possible overfitting"
        print(
            f"[Overfitting] model={row['model']} "
            f"mae_val={row['mae_mean']:.3f} "
            f"mae_train={row['train_mae_mean']:.3f} "
            f"mae_gap={row['mae_gap_mean']:.3f} "
            f"ratio={ratio:.3f}{warn}"
        )

    if pd.notna(baseline_mae):
        print(f"\nBaseline median mae_val={baseline_mae:.3f}")
        for _, row in diag.iterrows():
            delta = row["mae_mean"] - baseline_mae
            print(
                f"  vs baseline: {row['model']} mae_val delta={delta:+.3f} "
                f"(negative is better than median)"
            )


def print_nested_compare(
    summary: pd.DataFrame,
    y: pd.Series,
    n_classes: int,
) -> pd.DataFrame:
    """Compare nested all-gait vs nested RFECV; add random-baseline reference columns."""
    print("\n=== Nested CV: all gait vs RFECV-selected (outer test folds) ===")
    compare_cols = [
        "mae_mean",
        "mae_std",
        "accuracy_mean",
        "accuracy_std",
        "mae_gap_mean",
        "mae_gap_std",
        "mae_gap_ratio_mean",
        "mae_gap_ratio_std",
    ]
    all_gait = summary[summary["feature_variant"] == "nested_all_gait"]
    selected = summary[summary["feature_variant"] == "nested_rfecv"]

    random_rows = all_gait[all_gait["model"] == "baseline_random"]
    if len(random_rows):
        random_mae = float(random_rows["mae_mean"].iloc[0])
        random_mae_std = float(random_rows["mae_std"].iloc[0])
        random_acc = float(random_rows["accuracy_mean"].iloc[0])
        random_acc_std = float(random_rows["accuracy_std"].iloc[0])
    else:
        random_mae = expected_random_baseline_mae(y, n_classes)
        random_mae_std = np.nan
        random_acc = expected_random_baseline_accuracy(n_classes)
        random_acc_std = np.nan

    compare_models = all_gait[~all_gait["model"].isin(["baseline_random"])]
    merged = compare_models[["model"] + compare_cols].merge(
        selected[~selected["model"].isin(["baseline_random"])][
            ["model"] + compare_cols
        ],
        on="model",
        suffixes=("_all_gait", "_rfecv"),
    )
    merged["mae_mean_random"] = random_mae
    merged["mae_std_random"] = random_mae_std
    merged["accuracy_mean_random"] = random_acc
    merged["accuracy_std_random"] = random_acc_std

    if len(random_rows):
        rr = random_rows.iloc[0]
        random_row: dict[str, object] = {"model": "baseline_random"}
        for col in compare_cols:
            random_row[f"{col}_all_gait"] = rr[col]
            random_row[f"{col}_rfecv"] = rr[col]
        random_row["mae_mean_random"] = random_mae
        random_row["mae_std_random"] = random_mae_std
        random_row["accuracy_mean_random"] = random_acc
        random_row["accuracy_std_random"] = random_acc_std
        merged = pd.concat([merged, pd.DataFrame([random_row])], ignore_index=True)

    for _, row in merged.iterrows():
        print(
            f"[Compare] {row['model']}: "
            f"mae {row['mae_mean_all_gait']:.3f}±{row['mae_std_all_gait']:.3f} "
            f"-> {row['mae_mean_rfecv']:.3f}±{row['mae_std_rfecv']:.3f} | "
            f"acc {row['accuracy_mean_all_gait']:.3f}±{row['accuracy_std_all_gait']:.3f} "
            f"-> {row['accuracy_mean_rfecv']:.3f}±{row['accuracy_std_rfecv']:.3f} | "
            f"random ref mae={row['mae_mean_random']:.3f} acc={row['accuracy_mean_random']:.3f}"
        )
    print(
        f"\nRandom baseline (uniform over {n_classes} classes): "
        f"MAE={random_mae:.3f}"
        f"{f'±{random_mae_std:.3f}' if pd.notna(random_mae_std) else ''}, "
        f"accuracy={random_acc:.3f}"
        f"{f'±{random_acc_std:.3f}' if pd.notna(random_acc_std) else ''} "
        f"(expected acc={expected_random_baseline_accuracy(n_classes):.3f})"
    )
    print(merged.to_string(index=False))
    return merged


def _summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["feature_variant", "model"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            f1_weighted_mean=("f1_weighted", "mean"),
            f1_weighted_std=("f1_weighted", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            train_mae_mean=("train_mae", "mean"),
            train_mae_std=("train_mae", "std"),
            mae_gap_mean=("mae_gap", "mean"),
            mae_gap_std=("mae_gap", "std"),
            mae_gap_ratio_mean=("mae_gap_ratio", "mean"),
            mae_gap_ratio_std=("mae_gap_ratio", "std"),
            n_selected_features_mean=("n_selected_features", "mean"),
        )
        .reset_index()
    )
