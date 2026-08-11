"""
Plotting utilities for visualizing model results and interpretability.

Handles ROC curves, PR curves, confusion matrices, feature importance, etc.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix as sklearn_confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
)


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auroc: float,
    output_path: str,
) -> None:
    """
    Plot ROC curve and save to file.

    Parameters
    ----------
    fpr : array
        False Positive Rates.
    tpr : array
        True Positive Rates.
    auroc : float
        Area Under ROC Curve value.
    output_path : str
        Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    auprc: float,
    output_path: str,
) -> None:
    """
    Plot Precision-Recall curve and save to file.

    Parameters
    ----------
    precision : array
        Precision values.
    recall : array
        Recall values.
    auprc : float
        Area Under Precision-Recall Curve value.
    output_path : str
        Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="darkgreen", lw=2, label=f"PR curve (AUPRC = {auprc:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    labels: Optional[list] = None,
) -> None:
    """
    Plot confusion matrix and save to file.

    Parameters
    ----------
    y_true : array
        True labels.
    y_pred : array
        Predicted labels.
    output_path : str
        Path to save the figure.
    labels : list, optional
        Class labels (e.g., ['NonMS', 'MS']).
    """
    if labels is None:
        # Auto-detect labels from data (supports binary and multiclass)
        unique_labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
        labels = [str(lbl) for lbl in unique_labels]

    cm = sklearn_confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    importances: Dict[str, float],
    output_path: str,
    top_n: int = 20,
) -> None:
    """
    Plot feature importance bar chart and save to file.

    Parameters
    ----------
    importances : dict
        Feature names to importance values mapping.
    output_path : str
        Path to save the figure.
    top_n : int, default=20
        Number of top features to display.
    """
    # Sort and select top N
    sorted_features = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_features[:top_n]

    names = [f[0] for f in top_features]
    values = [f[1] for f in top_features]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.35)))
    colors = ["green" if v > 0 else "red" for v in values]
    ax.barh(names, values, color=colors, alpha=0.7)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    metrics: Optional[list] = None,
) -> None:
    """
    Plot metrics comparison across multiple experiments.

    Parameters
    ----------
    results : dict
        Mapping of experiment name to metrics dict.
    output_path : str
        Path to save the figure.
    metrics : list, optional
        Which metrics to plot. If None, uses all available.
    """
    if not results:
        return

    if metrics is None:
        # Get all metrics from first result
        metrics = list(next(iter(results.values())).keys())

    # Create bar plot
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        names = list(results.keys())
        values = [results[name].get(metric, 0) for name in names]
        ax.bar(names, values, alpha=0.7, color="steelblue")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_ylim([0, 1.0])
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticklabels(names, rotation=45, ha="right")

    axes[0].set_title("Metrics Comparison Across Experiments")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fold_diagnostics(diagnostics: Dict, fold_idx: int, output_dir: str) -> None:
    """
    Visualize fold diagnostics: confusion matrix, label distribution, and probability distribution.

    Parameters
    ----------
    diagnostics : dict
        Diagnostic information from a fold.
    fold_idx : int
        Fold index.
    output_dir : str
        Directory to save plots.
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Fold {fold_idx} Diagnostics", fontsize=14, fontweight="bold")
    
    # 1. Confusion Matrix
    cm = np.array(diagnostics["confusion_matrix"])
    im = axes[0].imshow(cm, cmap="Blues", aspect="auto")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Pred 0", "Pred 1"])
    axes[0].set_yticklabels(["True 0", "True 1"])
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].set_title("Confusion Matrix")
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = axes[0].text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=14, fontweight="bold")
    
    plt.colorbar(im, ax=axes[0])
    
    # 2. Predicted Label Distribution
    true_dist = diagnostics["true_label_distribution"]
    pred_dist = diagnostics["predicted_label_distribution"]
    
    labels = ["Class 0", "Class 1"]
    true_counts = [true_dist.get(0, 0), true_dist.get(1, 0)]
    pred_counts = [pred_dist.get(0, 0), pred_dist.get(1, 0)]
    
    x = np.arange(len(labels))
    width = 0.35
    
    axes[1].bar(x - width/2, true_counts, width, label="True", alpha=0.8, color="skyblue")
    axes[1].bar(x + width/2, pred_counts, width, label="Predicted", alpha=0.8, color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Count")
    axes[1].set_title("Label Distribution")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    
    # 3. Predicted Probability Distribution
    proba_dist = diagnostics["predicted_proba_distribution"]
    proba_stats = diagnostics["predicted_proba_stats"]
    
    axes[2].bar(range(len(proba_dist["counts"])), proba_dist["counts"], alpha=0.8, color="lightgreen")
    axes[2].set_xticks(range(len(proba_dist["bins"])))
    axes[2].set_xticklabels(proba_dist["bins"], rotation=45, ha="right", fontsize=8)
    axes[2].set_xlabel("Predicted Probability")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Predicted Probability Distribution")
    axes[2].grid(axis="y", alpha=0.3)
    
    # Add warning if single class
    if diagnostics["single_class_warning"]:
        axes[2].text(0.5, 0.95, "⚠️ WARNING: >95% single class", 
                    transform=axes[2].transAxes, ha="center", va="top",
                    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
                    fontweight="bold", fontsize=10)
    
    fig.tight_layout()
    output_path = os.path.join(output_dir, f"fold_{fold_idx}_diagnostics.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_tradeoff(
    threshold_df,
    output_path: str,
) -> None:
    """
    Plot sensitivity-specificity tradeoff curve across decision thresholds.
    
    Shows how sensitivity and specificity change with decision threshold,
    highlighting the default threshold (0.5) for reference.

    Parameters
    ----------
    threshold_df : DataFrame
        Result from analyze_decision_threshold containing:
        - 'threshold': decision threshold
        - 'sensitivity': True Positive Rate
        - 'specificity': True Negative Rate
    output_path : str
        Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot sensitivity and specificity
    ax.plot(
        threshold_df["threshold"],
        threshold_df["sensitivity"],
        marker="o",
        linewidth=2.5,
        markersize=6,
        label="Sensitivity (TPR)",
        color="steelblue"
    )
    
    ax.plot(
        threshold_df["threshold"],
        threshold_df["specificity"],
        marker="s",
        linewidth=2.5,
        markersize=6,
        label="Specificity (TNR)",
        color="darkorange"
    )
    
    # Highlight default threshold (0.5) using tolerance-based comparison
    # This handles floating point precision issues
    default_idx = np.argmin(np.abs(threshold_df["threshold"].values - 0.5))
    if abs(threshold_df.iloc[default_idx]["threshold"] - 0.5) < 0.01:
        ax.axvline(x=0.5, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Default threshold (0.5)")
        # Mark the points at default threshold
        sens_at_default = threshold_df.iloc[default_idx]["sensitivity"]
        spec_at_default = threshold_df.iloc[default_idx]["specificity"]
        ax.plot(0.5, sens_at_default, "o", markersize=12, markerfacecolor="none", 
               markeredgecolor="steelblue", markeredgewidth=2.5)
        ax.plot(0.5, spec_at_default, "s", markersize=12, markerfacecolor="none",
               markeredgecolor="darkorange", markeredgewidth=2.5)
    
    ax.set_xlabel("Decision Threshold", fontsize=12, fontweight="bold")
    ax.set_ylabel("Rate", fontsize=12, fontweight="bold")
    ax.set_title("Sensitivity-Specificity Tradeoff Curve", fontsize=14, fontweight="bold")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=10, framealpha=0.95)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bootstrap_distributions(
    distributions: Dict[str, np.ndarray],
    mean_metrics: Dict[str, float],
    std_metrics: Dict[str, float],
    output_path: str,
) -> None:
    """
    Plot bootstrap distributions for evaluation metrics.
    
    Creates a multi-panel figure showing histograms of metric distributions
    with mean and 95% confidence interval marked.

    Parameters
    ----------
    distributions : dict
        Bootstrap distributions from estimate_bootstrap_uncertainty.
        Keys: 'auroc', 'auprc', 'balanced_accuracy', 'sensitivity', 'specificity'
        Values: arrays of metric values (n_iterations,)
    mean_metrics : dict
        Mean values for each metric.
    std_metrics : dict
        Standard deviation for each metric.
    output_path : str
        Path to save the figure.
    """
    metric_names = ['auroc', 'auprc', 'balanced_accuracy', 'sensitivity', 'specificity']
    metric_labels = ['AUROC', 'AUPRC', 'Balanced Accuracy', 'Sensitivity', 'Specificity']
    colors = ['steelblue', 'darkorange', 'seagreen', 'crimson', 'purple']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (metric_name, metric_label, color) in enumerate(zip(metric_names, metric_labels, colors)):
        ax = axes[idx]
        
        if metric_name not in distributions:
            ax.text(0.5, 0.5, f'No data for {metric_label}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
        
        values = distributions[metric_name]
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) == 0:
            ax.text(0.5, 0.5, f'No valid data for {metric_label}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
        
        # Plot histogram
        ax.hist(valid_values, bins=30, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
        
        # Mark mean
        mean_val = mean_metrics.get(metric_name, np.nan)
        std_val = std_metrics.get(metric_name, np.nan)
        
        if not np.isnan(mean_val):
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2.5, label=f'Mean')
            
            # Mark ±1 std
            ax.axvline(mean_val - std_val, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Mean ±SD')
            ax.axvline(mean_val + std_val, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        
        # Mark 95% CI
        ci_lower = np.percentile(valid_values, 2.5)
        ci_upper = np.percentile(valid_values, 97.5)
        
        ax.axvline(ci_lower, color='green', linestyle='-', linewidth=1.5, alpha=0.6, label='95% CI')
        ax.axvline(ci_upper, color='green', linestyle='-', linewidth=1.5, alpha=0.6)
        
        ax.set_xlabel('Metric Value', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{metric_label}\n{mean_val:.4f} ± {std_val:.4f}', fontweight='bold', fontsize=11)
        ax.grid(alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=9)
    
    # Hide the 6th subplot
    axes[5].axis('off')
    
    fig.suptitle('Bootstrap Uncertainty Estimation\n(1000 iterations, 80% resampling)', 
                fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Prediction-vs-true distribution / calibration (ordinal regression)
# =====================================================================

def _discretise(values: np.ndarray, n_classes: int) -> np.ndarray:
    """Clip continuous predictions to [0, n_classes-1] and round to integers."""
    return np.clip(np.rint(values), 0, n_classes - 1).astype(int)


def _headline_mae(data: Dict[str, np.ndarray]) -> float:
    """MAE on the project's reported convention: the equal-weight fold mean.

    Folds are unequal in size once samples are grouped by participant, so a
    pooled mean over all out-of-fold rows would not match the number reported in
    the results table (0.604 vs 0.597 on the FGA run). Callers that have no fold
    information fall back to the pooled mean.
    """
    y_true = np.asarray(data["y_true"], dtype=float)
    y_pred = np.asarray(data["y_pred"], dtype=float)
    folds = data.get("fold")
    if folds is None:
        return float(mean_absolute_error(y_true, y_pred))
    folds = np.asarray(folds)
    err = np.abs(y_true - y_pred)
    return float(np.mean([err[folds == f].mean() for f in np.unique(folds)]))


def plot_prediction_distribution_grid(
    per_model: Dict[str, Dict[str, np.ndarray]],
    output_path: str,
    *,
    n_classes: int,
    target_name: str = "FGA",
) -> None:
    """Grouped-bar comparison of true vs predicted class distributions per model.

    Parameters
    ----------
    per_model : dict
        ``{model_name: {"y_true": array, "y_pred": array}}`` where ``y_pred`` is
        the continuous (or label) out-of-fold prediction. An optional ``"fold"``
        array puts the title's MAE on the reported fold-mean convention; without
        it the title falls back to a pooled mean.
    output_path : str
        File to save the figure.
    n_classes : int
        Number of ordinal classes (bars span 0..n_classes-1).
    target_name : str
        Label used in titles/axes (e.g. "FGA").
    """
    if not per_model:
        return
    model_names = list(per_model.keys())
    n = len(model_names)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    classes = np.arange(n_classes)
    width = 0.4

    for idx, model_name in enumerate(model_names):
        ax = axes[idx // ncols][idx % ncols]
        y_true = np.asarray(per_model[model_name]["y_true"], dtype=float)
        y_pred = np.asarray(per_model[model_name]["y_pred"], dtype=float)
        true_int = _discretise(y_true, n_classes)
        pred_int = _discretise(y_pred, n_classes)
        true_counts = [int((true_int == c).sum()) for c in classes]
        pred_counts = [int((pred_int == c).sum()) for c in classes]
        ax.bar(classes - width / 2, true_counts, width, label="True",
               color="steelblue", alpha=0.85)
        ax.bar(classes + width / 2, pred_counts, width, label="Predicted",
               color="coral", alpha=0.85)
        ax.set_xticks(list(classes))
        ax.set_xlabel(f"{target_name} score")
        ax.set_ylabel("Count")
        mae = _headline_mae(per_model[model_name])
        ax.set_title(f"{model_name} (MAE={mae:.3f})")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"{target_name}: true vs predicted distribution (out-of-fold)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_scatter_grid(
    per_model: Dict[str, Dict[str, np.ndarray]],
    output_path: str,
    *,
    n_classes: int,
    target_name: str = "FGA",
    random_state: int = 0,
    continuous: bool = False,
) -> None:
    """Predicted-vs-true scatter (true jittered) with y=x reference, per model.

    When ``continuous`` is True, axis limits are derived from the data and no
    jitter is applied (used for continuous targets like Zeno walkway measures).
    """
    if not per_model:
        return
    rng = np.random.default_rng(random_state)
    model_names = list(per_model.keys())
    n = len(model_names)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    if continuous:
        all_vals = np.concatenate(
            [
                np.concatenate(
                    [
                        np.asarray(v["y_true"], dtype=float),
                        np.asarray(v["y_pred"], dtype=float),
                    ]
                )
                for v in per_model.values()
            ]
        )
        finite = all_vals[np.isfinite(all_vals)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        pad = 0.05 * (hi - lo or 1.0)
        lim = (lo - pad, hi + pad)
    else:
        lim = (-0.5, n_classes - 0.5)

    for idx, model_name in enumerate(model_names):
        ax = axes[idx // ncols][idx % ncols]
        y_true = np.asarray(per_model[model_name]["y_true"], dtype=float)
        y_pred = np.asarray(per_model[model_name]["y_pred"], dtype=float)
        jitter = (
            np.zeros(len(y_true))
            if continuous
            else rng.uniform(-0.12, 0.12, size=len(y_true))
        )
        ax.scatter(y_true + jitter, y_pred, alpha=0.5, s=25,
                   color="darkorange", edgecolor="none")
        ax.plot(lim, lim, color="navy", lw=1.5, linestyle="--", label="y = x")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"True {target_name}")
        ax.set_ylabel(f"Predicted {target_name}")
        mae = _headline_mae(per_model[model_name])
        ax.set_title(f"{model_name} (MAE={mae:.3f})")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"{target_name}: predicted vs true (out-of-fold, true jittered)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


