"""Evaluation metrics, model evaluation loops, and bootstrap re-exports.

Package init stays light (bootstrap only) to avoid circular imports with
``cv.splits``, which needs ``sorted_participant_ids`` at import time.
"""

from .bootstrap import (
    load_teammate_bootstrap_npz,
    per_fold_bootstrap_mae_rmse,
    sorted_participant_ids,
)

__all__ = [
    "load_teammate_bootstrap_npz",
    "per_fold_bootstrap_mae_rmse",
    "sorted_participant_ids",
]


def __getattr__(name: str):
    if name in {
        "evaluate_models",
        "print_nested_compare",
        "print_overfitting_diagnosis",
    }:
        from . import evaluate as _evaluate

        return getattr(_evaluate, name)
    if name in {
        "expected_random_baseline_accuracy",
        "expected_random_baseline_mae",
        "participant_level_summary",
        "random_baseline_fold_metrics",
    }:
        from . import metrics as _metrics

        return getattr(_metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
