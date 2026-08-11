"""
Evaluation metrics for ordinal FGA prediction (classification & regression).

Provides computation of AUROC, AUPRC, F1, Accuracy, Balanced Accuracy, 
MAE, RMSE, and auxiliary metrics with consistent numerical stability.
"""

import warnings
from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    auc,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.utils.class_weight import compute_class_weight


# =====================================================================
# Primary metric configuration
# =====================================================================

# Registry of metrics that can serve as the primary (headline) metric.
# Maps user-facing name → key used in the metrics dict returned by
# compute_classification_metrics.
SUPPORTED_PRIMARY_METRICS = {
    "mae":               "mae",
    "rmse":              "rmse",
    "accuracy":          "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1":                "f1",
    "auroc":             "auroc",
}

DEFAULT_PRIMARY_METRIC = "mae"


def get_primary_metric_value(
    metrics: Dict[str, float],
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
) -> float:
    """
    Extract the primary metric value from a metrics dict.

    Parameters
    ----------
    metrics : dict
        Metrics dict as returned by compute_classification_metrics.
    primary_metric : str
        Name of the primary metric (key in SUPPORTED_PRIMARY_METRICS).

    Returns
    -------
    float
        The value of the primary metric, or NaN if not found.
    """
    key = SUPPORTED_PRIMARY_METRICS.get(primary_metric, primary_metric)
    val = metrics.get(key, np.nan)
    # Fallback: balanced_accuracy ↔ balanced_acc alias
    if isinstance(val, float) and np.isnan(val) and key == "balanced_accuracy":
        val = metrics.get("balanced_acc", np.nan)
    return float(val)


# =====================================================================
# Class-balanced sample weights (for models that lack class_weight=)
# =====================================================================

def make_balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """
    Compute per-sample weights equivalent to sklearn's class_weight='balanced'.

    Formula per class *c*:
        w_c = n_samples / (n_classes * n_samples_of_class_c)

    Parameters
    ----------
    y : array, shape (n_samples,)
        Integer class labels (e.g. 0, 1, 2, 3).

    Returns
    -------
    sample_weight : array, shape (n_samples,)
        Weight for every training sample.  Samples from rarer classes
        receive higher weights so that each class contributes equally
        to the loss.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    cw = compute_class_weight("balanced", classes=classes, y=y)
    weight_map = dict(zip(classes, cw))
    return np.array([weight_map[label] for label in y], dtype=float)


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_label: np.ndarray = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute comprehensive binary classification metrics.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True binary labels (0 or 1).
    y_pred_proba : array-like, shape (n_samples,)
        Predicted probabilities for the positive class.
    y_pred_label : array-like, shape (n_samples,), optional
        Predicted binary labels (0 or 1). If None, derived from y_pred_proba 
        using threshold.
    threshold : float, default=0.5
        Threshold for converting probabilities to binary labels.

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'auroc': Area Under ROC Curve
        - 'auprc': Area Under Precision-Recall Curve
        - 'f1': F1 score
        - 'accuracy': Accuracy
        - 'balanced_accuracy': Balanced accuracy
        - 'sensitivity': True Positive Rate (Recall for positive class)
        - 'specificity': True Negative Rate
        - 'mae': Mean Absolute Error (ordinal-aware)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)

    # Generate binary predictions if not provided
    if y_pred_label is None:
        y_pred_label = (y_pred_proba >= threshold).astype(int)
    else:
        y_pred_label = np.asarray(y_pred_label, dtype=int)

    metrics = {}

    # AUROC
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_pred_proba))
    except Exception:
        metrics["auroc"] = np.nan

    # AUPRC
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        metrics["auprc"] = float(auc(recall, precision))
    except Exception:
        metrics["auprc"] = np.nan

    # F1 Score
    try:
        metrics["f1"] = float(f1_score(y_true, y_pred_label, zero_division=0))
    except Exception:
        metrics["f1"] = np.nan

    # Accuracy
    try:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred_label))
    except Exception:
        metrics["accuracy"] = np.nan

    # Balanced Accuracy
    try:
        metrics["balanced_accuracy"] = float(
            balanced_accuracy_score(y_true, y_pred_label)
        )
    except Exception:
        metrics["balanced_accuracy"] = np.nan

    # MAE (ordinal-aware: measures average distance between predicted and true labels)
    metrics["mae"] = float(mean_absolute_error(y_true, y_pred_label))

    # Sensitivity (TPR for positive class) and Specificity (TNR)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_label).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["sensitivity"] = float(sensitivity)
        metrics["specificity"] = float(specificity)
    except (ValueError, Exception):
        # ValueError raised if only one class in y_pred_label
        metrics["sensitivity"] = np.nan
        metrics["specificity"] = np.nan

    # Backward-compatible alias used by batch training / ablation scripts
    metrics["balanced_acc"] = metrics["balanced_accuracy"]

    return metrics


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute classification metrics, auto-detecting binary vs multiclass.

    Binary (≤2 labels): delegates to compute_binary_classification_metrics.
    Multiclass (>2 labels): accuracy, balanced_accuracy, macro-F1, weighted-F1,
    multiclass AUROC (one-vs-rest, macro).  Binary-only metrics set to NaN.

    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True labels.
    y_pred : array, shape (n_samples,)
        Predicted labels.
    y_pred_proba : array, optional
        Probabilities — (n_samples,) for binary, (n_samples, n_classes) multiclass.

    Returns
    -------
    metrics : dict
        Always contains: auroc, f1, accuracy, balanced_accuracy.
        Binary adds: auprc, sensitivity, specificity.
        Multiclass adds: f1_macro, f1_weighted.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Detect n_classes from the probability matrix shape when available,
    # because np.unique(y_true) may miss classes in small folds.
    if y_pred_proba is not None and hasattr(y_pred_proba, 'ndim') and y_pred_proba.ndim == 2:
        n_classes = y_pred_proba.shape[1]
    else:
        n_classes = len(np.unique(y_true))
    is_binary = n_classes <= 2

    # --- Binary path: delegate to existing function ---
    if is_binary:
        if y_pred_proba is not None:
            proba_1d = (y_pred_proba[:, 1] if y_pred_proba.ndim == 2
                        else np.asarray(y_pred_proba, dtype=float))
        else:
            proba_1d = np.zeros(len(y_true), dtype=float)
        return compute_binary_classification_metrics(y_true, proba_1d, y_pred)

    # --- Multiclass path ---
    # Explicitly declare the full label space so sklearn doesn't warn
    # when a validation fold is missing a class that the model predicts.
    all_labels = list(range(n_classes))

    # Suppress "y_pred contains classes not in y_true" — expected when
    # small CV folds lack an entire class (e.g. Class 0 with only 4 PIDs).
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="y_pred contains classes not in y_true",
            category=UserWarning,
        )

        metrics = {}
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        metrics["f1_macro"] = float(
            f1_score(y_true, y_pred, average="macro",
                     labels=all_labels, zero_division=0)
        )
        metrics["f1_weighted"] = float(
            f1_score(y_true, y_pred, average="weighted",
                     labels=all_labels, zero_division=0)
        )
        metrics["f1"] = metrics["f1_macro"]  # backward-compat alias

        # Multiclass AUROC via one-vs-rest (requires full probability matrix)
        if y_pred_proba is not None and y_pred_proba.ndim == 2:
            try:
                metrics["auroc"] = float(roc_auc_score(
                    y_true, y_pred_proba,
                    multi_class="ovr", average="macro",
                    labels=all_labels,
                ))
            except Exception:
                metrics["auroc"] = np.nan
        else:
            metrics["auroc"] = np.nan

    # MAE (ordinal-aware: measures average distance between predicted and true labels)
    metrics["mae"] = float(mean_absolute_error(y_true, y_pred))

    # Binary-only metrics: not applicable for multiclass
    metrics["auprc"] = np.nan
    metrics["sensitivity"] = np.nan
    metrics["specificity"] = np.nan

    # Backward-compatible alias used by batch training / ablation scripts
    metrics["balanced_acc"] = metrics["balanced_accuracy"]

    return metrics


# =====================================================================
# Participant-level aggregation for multi-view evaluation
# =====================================================================
#
# Why this is needed:
#   In a multi-view setup each participant contributes multiple samples
#   (e.g. FW + PWS walking conditions).  Computing metrics on raw
#   samples inflates the effective N and introduces within-participant
#   correlation, which can overstate classifier performance.
#
#   Participant-level aggregation averages the predicted probabilities
#   across a participant's samples, produces a single participant-level
#   prediction, and then scores against the (unique) ground-truth label.
#   This yields an honest per-participant accuracy and is the correct
#   evaluation for multi-view data.
#
#   Aggregation is applied ONLY at evaluation time; training always
#   uses all multi-view samples as-is.
# =====================================================================


def aggregate_predictions_by_participant(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    groups: np.ndarray,
    method: str = "mean",
) -> dict:
    """
    Aggregate per-sample predictions to participant level.

    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True labels (all samples for one participant must share the same label).
    y_pred : array, shape (n_samples,)
        Predicted labels (used only for sanity-check; the returned participant
        predictions are derived from the aggregated probabilities).
    y_pred_proba : array, shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities.  For binary data with 1-D proba, the function
        works with that single column.  For multiclass, a 2-D matrix is expected.
    groups : array, shape (n_samples,)
        Participant / group identifiers (same length as y_true).
    method : str, default "mean"
        Aggregation function applied to the probability vectors:
        ``"mean"`` (default), ``"median"``, or ``"max"``.

    Returns
    -------
    result : dict with keys
        ``"y_true"``            – (n_participants,) ground-truth labels
        ``"y_pred"``            – (n_participants,) argmax of aggregated probabilities
        ``"y_pred_proba"``      – (n_participants, n_classes) aggregated probabilities
        ``"participant_ids"``   – (n_participants,) unique group labels, sorted
        ``"n_participants"``    – int
        ``"aggregation_method"``– str
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    groups = np.asarray(groups)

    agg_fn = {"mean": np.mean, "median": np.median, "max": np.max}
    if method not in agg_fn:
        raise ValueError(f"Unknown aggregation method: {method!r}. Choose from {list(agg_fn)}")

    unique_ids = np.unique(groups)
    n_participants = len(unique_ids)

    is_1d = y_pred_proba.ndim == 1
    if is_1d:
        n_classes = 2
        # expand to 2-column matrix for uniform handling
        y_pred_proba = np.column_stack([1.0 - y_pred_proba, y_pred_proba])
    else:
        n_classes = y_pred_proba.shape[1]

    agg_y_true = np.empty(n_participants, dtype=int)
    agg_proba = np.empty((n_participants, n_classes), dtype=float)

    for i, pid in enumerate(unique_ids):
        mask = groups == pid
        labels_for_pid = np.unique(y_true[mask])
        assert len(labels_for_pid) == 1, (
            f"Participant {pid} has inconsistent labels: {labels_for_pid}. "
            "All samples for one participant must share the same ground-truth label."
        )
        agg_y_true[i] = labels_for_pid[0]
        agg_proba[i] = agg_fn[method](y_pred_proba[mask], axis=0)

    agg_y_pred = np.argmax(agg_proba, axis=1)

    assert agg_y_true.shape[0] == n_participants

    return {
        "y_true": agg_y_true,
        "y_pred": agg_y_pred,
        "y_pred_proba": agg_proba,
        "participant_ids": unique_ids,
        "n_participants": n_participants,
        "aggregation_method": method,
    }


def compute_participant_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    groups: np.ndarray,
    method: str = "mean",
) -> dict:
    """
    Aggregate sample predictions to participant level, then compute metrics.

    This is the main entry-point for participant-level evaluation.
    It first calls :func:`aggregate_predictions_by_participant`, then
    :func:`compute_classification_metrics` on the aggregated arrays.

    Parameters
    ----------
    y_true, y_pred, y_pred_proba, groups, method
        See :func:`aggregate_predictions_by_participant`.

    Returns
    -------
    result : dict with keys
        ``"metrics"``           – dict of classification metrics at participant level
        ``"n_participants"``    – int
        ``"aggregation_method"``– str
        ``"agg_y_true"``        – participant-level ground truth
        ``"agg_y_pred"``        – participant-level predictions
        ``"agg_y_pred_proba"``  – participant-level aggregated probabilities
    """
    agg = aggregate_predictions_by_participant(
        y_true, y_pred, y_pred_proba, groups, method=method,
    )
    metrics = compute_classification_metrics(
        agg["y_true"], agg["y_pred"], agg["y_pred_proba"],
    )
    return {
        "metrics": metrics,
        "n_participants": agg["n_participants"],
        "aggregation_method": agg["aggregation_method"],
        "agg_y_true": agg["y_true"],
        "agg_y_pred": agg["y_pred"],
        "agg_y_pred_proba": agg["y_pred_proba"],
    }


def compute_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve coordinates and thresholds.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred_proba : array-like
        Predicted probabilities.

    Returns
    -------
    fpr : array
        False Positive Rates.
    tpr : array
        True Positive Rates.
    thresholds : array
        Thresholds for each (fpr, tpr) point.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    return fpr, tpr, thresholds


def compute_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Precision-Recall curve.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred_proba : array-like
        Predicted probabilities.

    Returns
    -------
    precision : array
        Precision values.
    recall : array
        Recall values.
    thresholds : array
        Thresholds.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    return precision, recall, thresholds


def bootstrap_uncertainty_estimation(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_iterations: int = 1000,
    sample_size: float = 0.8,
    threshold: float = 0.5,
    random_state: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, float]]:
    """
    Estimate uncertainty of evaluation metrics using bootstrap resampling.
    
    Randomly samples 80% of data points from provided predictions (no retraining)
    and computes metrics at each iteration. Returns full distributions and summary
    statistics (mean ± std).

    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True binary labels from held-out test set.
    y_pred_proba : array, shape (n_samples,)
        Predicted probabilities for positive class (no retraining).
    n_iterations : int, default=1000
        Number of bootstrap iterations.
    sample_size : float, default=0.8
        Fraction of data to sample at each iteration.
    threshold : float, default=0.5
        Threshold for converting probabilities to binary labels.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    distributions : dict
        Full distributions for each metric:
        - 'auroc': array of AUROC values (n_iterations,)
        - 'auprc': array of AUPRC values
        - 'balanced_accuracy': array of balanced accuracy values
        - 'sensitivity': array of sensitivity values
        - 'specificity': array of specificity values

    mean_metrics : dict
        Mean value of each metric across iterations.

    std_metrics : dict
        Standard deviation of each metric across iterations.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    y_true = np.asarray(y_true, dtype=int)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    
    n_samples = len(y_true)
    n_bootstrap_samples = int(np.ceil(n_samples * sample_size))
    
    # Initialize storage for distributions
    distributions = {
        'auroc': [],
        'auprc': [],
        'balanced_accuracy': [],
        'sensitivity': [],
        'specificity': [],
    }
    
    # Bootstrap iterations
    for iteration in range(n_iterations):
        # Stratified random sampling (maintain label distribution)
        # Sample indices with replacement
        indices = np.random.choice(n_samples, size=n_bootstrap_samples, replace=True)
        
        y_true_boot = y_true[indices]
        y_pred_proba_boot = y_pred_proba[indices]
        y_pred_label_boot = (y_pred_proba_boot >= threshold).astype(int)
        
        # Compute metrics for this bootstrap sample
        metrics = compute_binary_classification_metrics(
            y_true_boot,
            y_pred_proba_boot,
            y_pred_label_boot,
            threshold=threshold
        )
        
        # Store metrics
        distributions['auroc'].append(metrics['auroc'])
        distributions['auprc'].append(metrics['auprc'])
        distributions['balanced_accuracy'].append(metrics['balanced_accuracy'])
        distributions['sensitivity'].append(metrics['sensitivity'])
        distributions['specificity'].append(metrics['specificity'])
    
    # Convert to arrays
    for key in distributions:
        distributions[key] = np.array(distributions[key])
    
    # Compute mean and std
    mean_metrics = {}
    std_metrics = {}
    
    for key in distributions:
        valid_values = distributions[key][~np.isnan(distributions[key])]
        if len(valid_values) > 0:
            mean_metrics[key] = float(np.mean(valid_values))
            std_metrics[key] = float(np.std(valid_values))
        else:
            mean_metrics[key] = np.nan
            std_metrics[key] = np.nan
    
    return distributions, mean_metrics, std_metrics


def bootstrap_ordinal_regression_mae_rmse(
    y_true: np.ndarray,
    y_pred_continuous: np.ndarray,
    n_iterations: int = 1000,
    sample_frac: float = 0.8,
    stratified: bool = True,
    random_state: Optional[int] = None,
    ci_alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Bootstrap uncertainty for ordinal regression (MAE / RMSE on continuous preds).

    Resamples rows (with replacement) without retraining — same spirit as
    ``bootstrap_uncertainty_estimation`` for binary metrics.

    When ``stratified`` is True, each bootstrap draw of size ``ceil(n * sample_frac)``
    preserves the empirical class proportions of ``y_true`` via a multinomial
    allocation per class (recommended for imbalanced ordinal labels).

    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True ordinal labels (integer-valued; coerced to int).
    y_pred_continuous : array, shape (n_samples,)
        Model continuous predictions (e.g. E[y]).
    n_iterations : int, default 1000
        Number of bootstrap replicates.
    sample_frac : float, default 0.8
        Fraction of sample size per bootstrap draw (rounded up to an integer count).
    stratified : bool, default True
        If True, stratify resampling by discrete ``y_true`` class.
    random_state : int, optional
        RNG seed for reproducibility.
    ci_alpha : float, default 0.05
        Two-sided interval uses ``[ci_alpha/2, 1-ci_alpha/2]`` percentiles.

    Returns
    -------
    dict
        Summary stats including mean/std and percentile CIs for MAE, MSE, and RMSE.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred_continuous = np.asarray(y_pred_continuous, dtype=float).ravel()
    if len(y_true) != len(y_pred_continuous):
        raise ValueError("y_true and y_pred_continuous must have the same length")
    n = int(len(y_true))
    if n == 0:
        return {
            "n_samples": 0,
            "n_iterations": int(n_iterations),
            "sample_frac": float(sample_frac),
            "stratified": bool(stratified),
            "mae_mean": float("nan"),
            "mae_std": float("nan"),
            "mae_ci_lower": float("nan"),
            "mae_ci_upper": float("nan"),
            "rmse_mean": float("nan"),
            "rmse_std": float("nan"),
            "rmse_ci_lower": float("nan"),
            "rmse_ci_upper": float("nan"),
        }

    y_true_int = np.rint(y_true).astype(np.int64)
    n_b = max(1, int(np.ceil(n * float(sample_frac))))
    rng = np.random.RandomState(random_state)

    mae_boot: list[float] = []
    mse_boot: list[float] = []
    rmse_boot: list[float] = []

    classes = np.unique(y_true_int)
    counts = np.array([(y_true_int == c).sum() for c in classes], dtype=float)
    props = counts / float(n)

    for _ in range(int(n_iterations)):
        if stratified and len(classes) > 1:
            # Multinomial allocation of n_b draws across classes
            alloc = rng.multinomial(n_b, props)
            idx_parts: list[np.ndarray] = []
            for k, c in enumerate(classes):
                pool = np.where(y_true_int == c)[0]
                if alloc[k] == 0:
                    continue
                idx_parts.append(rng.choice(pool, size=int(alloc[k]), replace=True))
            idx = np.concatenate(idx_parts) if idx_parts else rng.choice(n, size=n_b, replace=True)
        else:
            idx = rng.choice(n, size=n_b, replace=True)

        yt = y_true[idx]
        yp = y_pred_continuous[idx]
        mae_boot.append(float(mean_absolute_error(yt, yp)))
        mse_b = float(mean_squared_error(yt, yp))
        mse_boot.append(mse_b)
        rmse_boot.append(float(np.sqrt(mse_b)))

    mae_arr = np.asarray(mae_boot, dtype=float)
    mse_arr = np.asarray(mse_boot, dtype=float)
    rmse_arr = np.asarray(rmse_boot, dtype=float)
    lo = 100.0 * (ci_alpha / 2.0)
    hi = 100.0 * (1.0 - ci_alpha / 2.0)

    return {
        "n_samples": n,
        "bootstrap_draw_size": n_b,
        "n_iterations": int(n_iterations),
        "sample_frac": float(sample_frac),
        "stratified": bool(stratified),
        "ci_alpha": float(ci_alpha),
        "mae_mean": float(np.mean(mae_arr)),
        "mae_std": float(np.std(mae_arr)),
        "mae_ci_lower": float(np.percentile(mae_arr, lo)),
        "mae_ci_upper": float(np.percentile(mae_arr, hi)),
        "mse_mean": float(np.mean(mse_arr)),
        "mse_std": float(np.std(mse_arr)),
        "mse_ci_lower": float(np.percentile(mse_arr, lo)),
        "mse_ci_upper": float(np.percentile(mse_arr, hi)),
        "rmse_mean": float(np.mean(rmse_arr)),
        "rmse_std": float(np.std(rmse_arr)),
        "rmse_ci_lower": float(np.percentile(rmse_arr, lo)),
        "rmse_ci_upper": float(np.percentile(rmse_arr, hi)),
    }


# =====================================================================
# Regression metrics for ordinal task_type="regression_ordinal"
# =====================================================================

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred_continuous: np.ndarray,
    n_classes: int = 4,
    decision_delta: Optional[float] = None,
    y_pred_discrete_override: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for ordinal regression.

    The model itself is fit by a cumulative-link ordinal surrogate objective
    (K-1 binary heads); MAE and RMSE are evaluation metrics computed on the
    continuous ordinal score. Auxiliary classification metrics
    (accuracy, balanced_accuracy, f1) are computed on clip-then-round
    discretised predictions, purely as a sanity check — they must NOT be used
    for model selection.

    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True ordinal labels (integer 0 … n_classes-1).
    y_pred_continuous : array, shape (n_samples,)
        Continuous model output (e.g. from a Ridge / SVR / RF regressor).
    n_classes : int, default 4
        Number of ordinal bins (used for clipping range 0 … n_classes-1).

    Returns
    -------
    metrics : dict
        Keys: mae, mse, rmse, accuracy_aux, balanced_accuracy_aux,
        f1_macro_aux, y_pred_discrete (the rounded predictions).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_continuous = np.asarray(y_pred_continuous, dtype=float)

    metrics: Dict[str, float] = {}

    # ── Primary regression metrics ──
    metrics["mae"] = float(mean_absolute_error(y_true, y_pred_continuous))
    mse = float(mean_squared_error(y_true, y_pred_continuous))
    metrics["mse"] = mse
    metrics["rmse"] = float(np.sqrt(mse))

    # ── Auxiliary: discretise predictions → classification metrics ──
    if y_pred_discrete_override is not None:
        y_pred_discrete = np.asarray(y_pred_discrete_override, dtype=int).ravel()
        if len(y_pred_discrete) != len(y_true):
            raise ValueError(
                "y_pred_discrete_override length must match y_true: "
                f"{len(y_pred_discrete)} vs {len(y_true)}"
            )
        y_pred_discrete = np.clip(y_pred_discrete, 0, n_classes - 1)
    elif decision_delta is None:
        y_pred_discrete = np.clip(np.round(y_pred_continuous), 0, n_classes - 1).astype(int)
    else:
        # Legacy alternative ordinal decision rule:
        #   y_pred_discrete = floor(E[y] + delta)
        y_pred_discrete = np.clip(np.floor(y_pred_continuous + decision_delta), 0, n_classes - 1).astype(int)
    y_true_int = y_true.astype(int)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="y_pred contains classes not in y_true",
            category=UserWarning,
        )
        metrics["accuracy_aux"] = float(accuracy_score(y_true_int, y_pred_discrete))
        metrics["balanced_accuracy_aux"] = float(
            balanced_accuracy_score(y_true_int, y_pred_discrete)
        )
        all_labels = list(range(n_classes))
        metrics["f1_macro_aux"] = float(
            f1_score(y_true_int, y_pred_discrete, average="macro",
                     labels=all_labels, zero_division=0)
        )

    # Backward-compat aliases so summary tables can use a uniform key set
    metrics["accuracy"] = metrics["accuracy_aux"]
    metrics["balanced_acc"] = metrics["balanced_accuracy_aux"]
    metrics["f1"] = metrics["f1_macro_aux"]
    # Not meaningful for regression, but keeps JSON shape consistent
    metrics["auroc"] = np.nan
    metrics["auprc"] = np.nan
    metrics["sensitivity"] = np.nan
    metrics["specificity"] = np.nan

    return metrics


def aggregate_regression_by_participant(
    y_true: np.ndarray,
    y_pred_continuous: np.ndarray,
    groups: np.ndarray,
    method: str = "mean",
    *,
    continuous_targets: bool = False,
    strict_single_label: bool = True,
) -> dict:
    """
    Aggregate continuous predictions to participant level.

    Parameters
    ----------
    y_true : array, shape (n_samples,)
    y_pred_continuous : array, shape (n_samples,)
    groups : array, shape (n_samples,)
    method : str
        'mean', 'median', or 'max'.
    continuous_targets
        If True (Zeno float labels, e.g. velocity), ``y_true`` is aggregated with
        the same ``method`` as predictions (typically mean). If False (ordinal
        integer labels), each participant must have a single integer label when
        ``strict_single_label=True``. If set to False, participant labels are
        aggregated with ``method`` when multiple visit labels exist.

    Returns
    -------
    dict  with keys y_true, y_pred, participant_ids, n_participants.
    """
    agg_fn = {"mean": np.mean, "median": np.median, "max": np.max}
    if method not in agg_fn:
        raise ValueError(f"Unknown aggregation method: {method!r}")

    y_true = np.asarray(y_true, dtype=float)
    y_pred_continuous = np.asarray(y_pred_continuous, dtype=float)
    groups = np.asarray(groups)

    unique_ids = np.unique(groups)
    n = len(unique_ids)
    agg_y_true = np.empty(n, dtype=float)
    agg_y_pred = np.empty(n, dtype=float)
    n_inconsistent_participants = 0

    for i, pid in enumerate(unique_ids):
        mask = groups == pid
        if continuous_targets:
            agg_y_true[i] = float(agg_fn[method](y_true[mask]))
            agg_y_pred[i] = float(agg_fn[method](y_pred_continuous[mask]))
        else:
            labels = np.unique(y_true[mask].astype(int))
            if len(labels) == 1:
                agg_y_true[i] = float(labels[0])
            elif strict_single_label:
                raise AssertionError(
                    f"Participant {pid} has inconsistent labels: {labels}"
                )
            else:
                n_inconsistent_participants += 1
                agg_y_true[i] = float(agg_fn[method](y_true[mask]))
            agg_y_pred[i] = agg_fn[method](y_pred_continuous[mask])

    if n_inconsistent_participants > 0 and not strict_single_label:
        warnings.warn(
            "Detected participants with multiple ordinal labels across samples; "
            f"aggregated y_true by '{method}' for {n_inconsistent_participants} participants.",
            RuntimeWarning,
        )

    return {
        "y_true": agg_y_true,
        "y_pred": agg_y_pred,
        "participant_ids": unique_ids,
        "n_participants": n,
        "aggregation_method": method,
        "n_inconsistent_participants": n_inconsistent_participants,
    }


def compute_regression_participant_level_metrics(
    y_true: np.ndarray,
    y_pred_continuous: np.ndarray,
    groups: np.ndarray,
    n_classes: int = 4,
    method: str = "mean",
    decision_delta: Optional[float] = None,
    *,
    continuous_targets: bool = False,
    strict_single_label: bool = True,
) -> dict:
    """
    Participant-level regression evaluation.

    Aggregates continuous predictions per participant, then computes
    regression metrics on the de-duplicated set.

    Returns
    -------
    dict with keys: metrics, n_participants, aggregation_method,
    agg_y_true, agg_y_pred.
    """
    agg = aggregate_regression_by_participant(
        y_true,
        y_pred_continuous,
        groups,
        method=method,
        continuous_targets=continuous_targets,
        strict_single_label=strict_single_label,
    )
    if continuous_targets:
        mse_c = float(mean_squared_error(agg["y_true"], agg["y_pred"]))
        metrics = {
            "mae": float(mean_absolute_error(agg["y_true"], agg["y_pred"])),
            "mse": mse_c,
            "rmse": float(np.sqrt(mse_c)),
            "accuracy": float("nan"),
            "balanced_acc": float("nan"),
            "f1": float("nan"),
            "auroc": float("nan"),
        }
    else:
        metrics = compute_regression_metrics(
            agg["y_true"],
            agg["y_pred"],
            n_classes=n_classes,
            decision_delta=decision_delta,
        )
    return {
        "metrics": metrics,
        "n_participants": agg["n_participants"],
        "aggregation_method": agg["aggregation_method"],
        "agg_y_true": agg["y_true"],
        "agg_y_pred": agg["y_pred"],
        "n_inconsistent_participants": int(agg.get("n_inconsistent_participants", 0)),
    }
