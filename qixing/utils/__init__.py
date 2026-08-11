"""Utility modules for ML pipeline."""

from .metrics import compute_binary_classification_metrics, compute_roc_curve, compute_precision_recall_curve
from .logging_utils import setup_logging, log_dict
from .io import ensure_dir, save_json, load_json, save_pickle, load_pickle, save_numpy, load_numpy
from .plotting import (
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_metrics_comparison,
)

__all__ = [
    "compute_binary_classification_metrics",
    "compute_roc_curve",
    "compute_precision_recall_curve",
    "setup_logging",
    "log_dict",
    "ensure_dir",
    "save_json",
    "load_json",
    "save_pickle",
    "load_pickle",
    "save_numpy",
    "load_numpy",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_metrics_comparison",
]
