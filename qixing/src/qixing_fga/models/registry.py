"""Model constructors, optional-deps flags, and nested hyperparameter tuning."""

from __future__ import annotations

import warnings
from typing import Literal, Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import mord

    HAS_MORD = True
except ImportError:
    HAS_MORD = False

TaskType = Literal["classification", "regression", "regression_continuous"]


def build_models(num_classes: int, random_state: int) -> dict[str, object]:
    models: dict[str, object] = {}

    if HAS_MORD:
        models["ordinal_logistic"] = mord.LogisticAT(alpha=1.0)
    else:
        warnings.warn("mord not available; falling back to non-ordinal logistic regression.")
        models["logistic_regression"] = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
        )

    models["random_forest"] = RandomForestClassifier(
        n_estimators=400,
        random_state=random_state,
        class_weight="balanced",
    )
    models["svm"] = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
    )

    if HAS_XGBOOST:
        if num_classes <= 2:
            models["xgboost"] = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                tree_method="hist",
            )
        else:
            models["xgboost"] = XGBClassifier(
                objective="multi:softprob",
                num_class=num_classes,
                eval_metric="mlogloss",
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                tree_method="hist",
            )

    return models


def build_regression_models(random_state: int) -> dict[str, object]:
    """Regressors whose training objective aligns with MAE evaluation.

    The ordinal_logistic entry (mord.LogisticAT) predicts integer ordinal
    labels; it is included as an ordinal-aware estimator that respects the
    0..K-1 ordering rather than treating classes as nominal.
    """
    models: dict[str, object] = {}

    if HAS_MORD:
        models["ordinal_logistic"] = mord.LogisticAT(alpha=1.0)
    else:
        warnings.warn("mord not available; ordinal_logistic skipped in regression task.")

    models["ridge"] = Ridge(alpha=1.0, random_state=random_state)
    models["random_forest"] = RandomForestRegressor(
        n_estimators=400,
        random_state=random_state,
    )
    models["svr"] = SVR(kernel="rbf", C=1.0, gamma="scale")

    if HAS_XGBOOST:
        models["xgboost"] = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            tree_method="hist",
        )

    return models


def build_models_for_task(
    task: TaskType, num_classes: int, random_state: int
) -> dict[str, object]:
    if task in ("regression", "regression_continuous"):
        models = build_regression_models(random_state)
        if task == "regression_continuous":
            # Ordinal logistic (mord) is a classifier over integer levels; it
            # cannot fit continuous float targets, so drop it here.
            models.pop("ordinal_logistic", None)
        return models
    return build_models(num_classes, random_state)


# Regularisation-focused grids (pipeline step name is "model"). Kept small so
# nested tuning stays tractable; all values bias toward simpler / less-overfit
# configurations, and the inner CV (neg-MAE) selects among them.
_REGRESSION_PARAM_GRIDS: dict[str, dict[str, list]] = {
    "ridge": {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
    "random_forest": {
        "model__max_depth": [2, 3, 5],
        "model__min_samples_leaf": [1, 3, 5],
    },
    "svr": {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale"]},
    "xgboost": {
        "model__max_depth": [2, 3],
        "model__n_estimators": [100, 300],
        "model__min_child_weight": [1, 5],
        "model__reg_lambda": [1.0, 5.0],
        "model__subsample": [0.8],
    },
    "ordinal_logistic": {"model__alpha": [0.1, 1.0, 10.0]},
}

_CLASSIFICATION_PARAM_GRIDS: dict[str, dict[str, list]] = {
    "random_forest": {
        "model__max_depth": [2, 3, 5],
        "model__min_samples_leaf": [1, 3, 5],
    },
    "svm": {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale"]},
    "xgboost": {
        "model__max_depth": [2, 3],
        "model__n_estimators": [100, 300],
        "model__min_child_weight": [1, 5],
        "model__reg_lambda": [1.0, 5.0],
        "model__subsample": [0.8],
    },
    "ordinal_logistic": {"model__alpha": [0.1, 1.0, 10.0]},
    "logistic_regression": {"model__C": [0.1, 1.0, 10.0]},
}


def get_param_grid(task: TaskType, model_name: str) -> Optional[dict[str, list]]:
    grids = (
        _REGRESSION_PARAM_GRIDS
        if task.startswith("regression")
        else _CLASSIFICATION_PARAM_GRIDS
    )
    return grids.get(model_name)


def maybe_tune(
    pipeline: Pipeline,
    model_name: str,
    task: TaskType,
    y_train: pd.Series,
    groups_train: pd.Series,
    *,
    tune: str,
    tune_splits: int,
    random_state: int,
) -> object:
    """Wrap ``pipeline`` in a group-aware GridSearchCV when tuning is enabled.

    Inner CV splits by ``participant_id`` (no leakage) and selects the
    hyperparameters with the best held-out MAE, so regularisation strength is
    chosen on validation error rather than fixed a priori. Falls back to the
    plain pipeline when tuning is off, no grid exists, or the inner-train fold
    is too small to split.
    """
    # Lazy import avoids circular dependency with cv.splits (TaskType lives here).
    from ..cv.splits import compute_continuous_cv_splits, compute_inner_cv_splits

    if tune != "grid":
        return pipeline
    grid = get_param_grid(task, model_name)
    if not grid:
        return pipeline
    try:
        if task == "regression_continuous":
            inner_splits = compute_continuous_cv_splits(
                groups_train, tune_splits
            )
        else:
            inner_splits = compute_inner_cv_splits(
                y_train, groups_train, tune_splits, random_state
            )
    except (ValueError, ImportError):
        return pipeline
    return GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring="neg_mean_absolute_error",
        cv=inner_splits,
        n_jobs=-1,
        refit=True,
    )
