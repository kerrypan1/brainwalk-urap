"""Inner-loop RFECV and nested outer-fold evaluation."""

from __future__ import annotations

import json
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline

from ..cv.splits import compute_inner_cv_splits
from ..evaluation.bootstrap import per_fold_bootstrap_mae_rmse
from ..evaluation.evaluate import _prediction_rows, _summarize_results
from ..evaluation.metrics import _compute_metrics, random_baseline_fold_metrics
from ..features.columns import reject_leaky_columns
from ..models.registry import TaskType, maybe_tune
from ..preprocessing import build_numeric_preprocessor


def _progress(msg: str) -> None:
    """Print progress line immediately (for long-running nested CV)."""
    print(msg, flush=True)


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def _pipeline_importance_getter(estimator: Pipeline) -> np.ndarray:
    return estimator.named_steps["model"].feature_importances_


def rfecv_select_features(
    X: pd.DataFrame,
    y: pd.Series,
    inner_cv_splits: list[tuple[list[int], list[int]]],
    min_features_to_select: int,
    step: int,
    random_state: int,
    *,
    task: TaskType = "classification",
    verbose: bool = True,
) -> tuple[list[str], pd.DataFrame]:
    """
    Inner-loop RFECV on outer-train only. ``inner_cv_splits`` are indices into X/y.
    """
    if len(X.columns) < min_features_to_select:
        raise ValueError(
            f"RFECV min_features_to_select ({min_features_to_select}) exceeds "
            f"available columns ({len(X.columns)})."
        )

    n_candidates = len(X.columns)
    n_elim_steps = max(
        1, (n_candidates - min_features_to_select + step - 1) // step
    )
    if verbose:
        _progress(
            f"    RFECV fit: {n_candidates} features, "
            f"~{n_elim_steps} elimination steps (step={step}), "
            f"{len(inner_cv_splits)} inner CV folds — running..."
        )

    if task == "regression":
        # Align the selection estimator with the MAE scoring: use a regressor
        # so feature importances reflect regression error, not class impurity.
        selection_model: object = RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            random_state=random_state,
        )
    else:
        selection_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            class_weight="balanced",
            random_state=random_state,
        )
    estimator = Pipeline(
        steps=[
            ("preprocess", build_numeric_preprocessor()),
            ("model", selection_model),
        ]
    )

    selector = RFECV(
        estimator=estimator,
        step=step,
        min_features_to_select=min_features_to_select,
        cv=inner_cv_splits,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        importance_getter=_pipeline_importance_getter,
    )
    t_rfecv = time.perf_counter()
    selector.fit(X, y)
    rfecv_elapsed = time.perf_counter() - t_rfecv

    selected = X.columns[selector.support_].tolist()
    if verbose:
        _progress(
            f"    RFECV fit finished in {_format_elapsed(rfecv_elapsed)} "
            f"→ {len(selected)} features retained"
        )
    ranking = pd.DataFrame(
        {
            "n_features": selector.cv_results_["n_features"],
            "mean_test_score": selector.cv_results_["mean_test_score"],
            "std_test_score": selector.cv_results_["std_test_score"],
        }
    )
    return selected, ranking


def _evaluate_outer_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: dict[str, object],
    fold_idx: int,
    *,
    feature_variant: str,
    n_selected: int,
    selected_features: list[str],
    test_idx: list[int],
    groups: pd.Series,
    bootstrap_by_fold: Optional[dict[str, object]],
    ci_alpha: float,
    n_classes: int,
    random_state: int,
    task: TaskType = "classification",
    tune: str = "none",
    tune_splits: int = 3,
    groups_train: Optional[pd.Series] = None,
    ids: Optional[pd.Series] = None,
    pred_sink: Optional[list[dict]] = None,
    verbose: bool = True,
) -> list[dict]:
    """Fit on outer-train, predict outer-test (one nested-CV fold)."""
    rows: list[dict] = []
    preprocessor = build_numeric_preprocessor()
    has_bootstrap = bootstrap_by_fold is not None
    model_names = list(models.keys())

    median_value = float(y_train.median())
    val_pred = np.full(len(y_test), median_value)
    train_pred = np.full(len(y_train), median_value)
    val_metrics = _compute_metrics(y_test, val_pred, task=task, n_classes=n_classes)
    train_metrics = _compute_metrics(y_train, train_pred, task=task, n_classes=n_classes)

    def _append_row(model_name: str, val_m: dict, train_m: dict, y_v, y_p) -> None:
        row = {
            "model": model_name,
            "fold": fold_idx,
            "feature_variant": feature_variant,
            "n_selected_features": n_selected,
            "selected_features": json.dumps(selected_features),
            **val_m,
            "train_accuracy": train_m["accuracy"],
            "train_f1_weighted": train_m["f1_weighted"],
            "train_mae": train_m["mae"],
            "train_mse": train_m["mse"],
            "train_rmse": train_m["rmse"],
            "mae_gap": val_m["mae"] - train_m["mae"],
            "mae_gap_ratio": (
                (val_m["mae"] - train_m["mae"]) / val_m["mae"]
                if val_m["mae"] > 0
                else np.nan
            ),
        }
        if has_bootstrap:
            fold_key = f"fold_{fold_idx - 1}"
            boot_stats = per_fold_bootstrap_mae_rmse(
                y_val=y_v,
                y_pred=y_p,
                val_idx=test_idx,
                groups=groups.to_numpy(),
                boot_matrix=bootstrap_by_fold[fold_key],
                ci_alpha=ci_alpha,
            )
            row.update(
                {
                    "bootstrap_n": boot_stats["n_bootstrap"],
                    "mae_boot_mean": boot_stats["mae_boot_mean"],
                    "mae_ci_lower": boot_stats["mae_ci_lower"],
                    "mae_ci_upper": boot_stats["mae_ci_upper"],
                    "rmse_boot_mean": boot_stats["rmse_boot_mean"],
                    "ci_alpha": boot_stats["ci_alpha"],
                }
            )
        rows.append(row)
        if pred_sink is not None:
            pred_sink.extend(
                _prediction_rows(
                    model_name, fold_idx, feature_variant,
                    y_v, y_p, test_idx, groups, ids,
                )
            )

    if verbose:
        _progress(
            f"      [{feature_variant}] baseline_median "
            f"(train n={len(y_train)}, test n={len(y_test)})..."
        )
    _append_row(
        "baseline_median", val_metrics, train_metrics, y_test, val_pred
    )

    if feature_variant == "nested_all_gait":
        val_metrics_r, train_metrics_r, val_pred_r, train_pred_r = (
            random_baseline_fold_metrics(
                y_train,
                y_test,
                n_classes,
                random_state + fold_idx + 20_000,
                task=task,
            )
        )
        if verbose:
            _progress(
                f"      [{feature_variant}] baseline_random "
                f"(uniform 0..{n_classes - 1})..."
            )
        _append_row(
            "baseline_random",
            val_metrics_r,
            train_metrics_r,
            y_test,
            val_pred_r,
        )

    for model_i, model_name in enumerate(model_names, start=1):
        if verbose:
            _progress(
                f"      [{feature_variant}] model {model_i}/{len(model_names)}: "
                f"{model_name}..."
            )
        model = models[model_name]
        pipeline = Pipeline(
            steps=[("preprocess", preprocessor), ("model", model)]
        )
        if tune == "grid" and groups_train is not None:
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
        else:
            estimator = pipeline
        estimator.fit(X_train, y_train)
        val_pred = estimator.predict(X_test)
        train_pred = estimator.predict(X_train)
        val_metrics = _compute_metrics(y_test, val_pred, task=task, n_classes=n_classes)
        train_metrics = _compute_metrics(y_train, train_pred, task=task, n_classes=n_classes)
        _append_row(
            model_name, val_metrics, train_metrics, y_test, val_pred
        )
        if verbose:
            _progress(
                f"        → val mae={val_metrics['mae']:.3f}, "
                f"acc={val_metrics['accuracy']:.3f}"
            )

    return rows


def run_nested_cv_rfecv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    models: dict[str, object],
    outer_splits: list[tuple[list[int], list[int]]],
    inner_splits_requested: int,
    random_state: int,
    min_features_to_select: int,
    rfecv_step: int,
    bootstrap_by_fold: Optional[dict[str, object]],
    ci_alpha: float,
    task: TaskType = "classification",
    tune: str = "none",
    tune_splits: int = 3,
    ids: Optional[pd.Series] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, list[str]], pd.DataFrame]:
    """
    Nested CV: outer folds from teammate bootstrap; inner RFECV on outer-train only;
    final models fit on outer-train selected features, evaluated on outer-test.
    """
    reject_leaky_columns(list(X.columns), context="nested CV input")
    n_classes = int(y.nunique())

    all_rows: list[dict] = []
    pred_rows: list[dict] = []
    inner_rankings: list[dict] = []
    fold_selected: dict[int, list[str]] = {}

    n_outer = len(outer_splits)
    n_models = len(models)
    t_nested = time.perf_counter()
    _progress(
        f"\n[Nested CV] Starting {n_outer} outer folds × "
        f"(inner RFECV + {n_models} models × 2 variants)..."
    )

    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        t_fold = time.perf_counter()
        train_idx = list(train_idx)
        test_idx = list(test_idx)

        X_train_outer = X.iloc[train_idx]
        y_train_outer = y.iloc[train_idx]
        groups_train = groups.iloc[train_idx]
        X_test_outer = X.iloc[test_idx]
        y_test_outer = y.iloc[test_idx]
        n_train_pid = groups_train.nunique()
        n_test_pid = groups.iloc[test_idx].nunique()

        _progress(
            f"\n[Outer {fold_idx}/{n_outer}] "
            f"train: {len(train_idx)} samples / {n_train_pid} participants | "
            f"test: {len(test_idx)} samples / {n_test_pid} participants"
        )

        _progress(f"  [Outer {fold_idx}] Building inner {inner_splits_requested}-fold SGKF...")
        inner_cv = compute_inner_cv_splits(
            y_train_outer,
            groups_train,
            inner_splits_requested,
            random_state + fold_idx,
        )
        _progress(
            f"  [Outer {fold_idx}] Inner CV ready ({len(inner_cv)} folds on outer-train)."
        )

        _progress(f"  [Outer {fold_idx}] Inner RFECV feature selection...")
        selected, inner_ranking = rfecv_select_features(
            X_train_outer,
            y_train_outer,
            inner_cv,
            min_features_to_select,
            rfecv_step,
            random_state + fold_idx,
            task=task,
        )
        fold_selected[fold_idx] = selected

        best_idx = int(np.argmax(inner_ranking["mean_test_score"]))
        _progress(
            f"  [Outer {fold_idx}] RFECV result: {len(selected)} features "
            f"(inner best n={int(inner_ranking['n_features'].iloc[best_idx])}, "
            f"neg_MAE={float(inner_ranking['mean_test_score'].iloc[best_idx]):.4f})"
        )
        _progress(f"  [Outer {fold_idx}] Selected: {', '.join(selected)}")

        for _, rank_row in inner_ranking.iterrows():
            inner_rankings.append(
                {
                    "outer_fold": fold_idx,
                    "n_features": rank_row["n_features"],
                    "mean_test_score": rank_row["mean_test_score"],
                    "std_test_score": rank_row["std_test_score"],
                }
            )

        X_train_sel = X_train_outer[selected]
        X_test_sel = X_test_outer[selected]
        _progress(
            f"  [Outer {fold_idx}] Outer-test evaluation (RFECV subset, "
            f"{len(selected)} features)..."
        )
        all_rows.extend(
            _evaluate_outer_fold(
                X_train_sel,
                y_train_outer,
                X_test_sel,
                y_test_outer,
                models,
                fold_idx,
                feature_variant="nested_rfecv",
                n_selected=len(selected),
                selected_features=selected,
                test_idx=test_idx,
                groups=groups,
                bootstrap_by_fold=bootstrap_by_fold,
                ci_alpha=ci_alpha,
                n_classes=n_classes,
                random_state=random_state,
                task=task,
                tune=tune,
                tune_splits=tune_splits,
                groups_train=groups_train,
                ids=ids,
                pred_sink=pred_rows,
            )
        )

        _progress(
            f"  [Outer {fold_idx}] Outer-test evaluation (all gait, "
            f"{X.shape[1]} features)..."
        )
        all_rows.extend(
            _evaluate_outer_fold(
                X_train_outer,
                y_train_outer,
                X_test_outer,
                y_test_outer,
                models,
                fold_idx,
                feature_variant="nested_all_gait",
                n_selected=X.shape[1],
                selected_features=list(X.columns),
                test_idx=test_idx,
                groups=groups,
                bootstrap_by_fold=bootstrap_by_fold,
                ci_alpha=ci_alpha,
                n_classes=n_classes,
                random_state=random_state,
                task=task,
                tune=tune,
                tune_splits=tune_splits,
                groups_train=groups_train,
                ids=ids,
                pred_sink=pred_rows,
            )
        )

        _progress(
            f"[Outer {fold_idx}/{n_outer}] Done in "
            f"{_format_elapsed(time.perf_counter() - t_fold)} "
            f"(total elapsed {_format_elapsed(time.perf_counter() - t_nested)})"
        )

    _progress(
        f"\n[Nested CV] All outer folds finished in "
        f"{_format_elapsed(time.perf_counter() - t_nested)}. Aggregating..."
    )
    results = pd.DataFrame(all_rows)
    summary = _summarize_results(results)
    inner_ranking_df = pd.DataFrame(inner_rankings)
    predictions = pd.DataFrame(pred_rows)
    return results, summary, inner_ranking_df, fold_selected, predictions
