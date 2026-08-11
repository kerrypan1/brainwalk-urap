"""Compatibility shim for the former monolithic ``batch_training`` module.

Re-exports public symbols from ``qixing_fga`` and delegates ``__main__`` to
``qixing_fga.cli.train.main``. Prefer importing from ``qixing_fga`` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qixing_fga.cli.train import (  # noqa: E402
    DEFAULT_BOOTSTRAP_NPZ,
    DEFAULT_FEATURE_PATH,
    DEFAULT_ID_COL,
    DEFAULT_LABEL_PATH,
    DEFAULT_SPLITS,
    DEFAULT_TARGET_COL,
    _format_elapsed,
    _progress,
    main,
)
# Keep shim PROJECT_ROOT (src/batch_training.py -> parents[1]); train.py uses parents[3].
from qixing_fga.cv.splits import (  # noqa: E402
    HAS_SGKF,
    build_fold_indices_from_bootstrap,
    compute_continuous_cv_splits,
    compute_inner_cv_splits,
    continuous_effective_n_splits,
    effective_n_splits,
    get_cv_splits,
)
from qixing_fga.data.loading import (  # noqa: E402
    add_sample_id,
    load_merged_data,
    load_single_file,
    prepare_features,
)
from qixing_fga.evaluation.bootstrap import (  # noqa: E402
    load_teammate_bootstrap_npz,
    per_fold_bootstrap_mae_rmse,
    sorted_participant_ids,
)
from qixing_fga.evaluation.evaluate import (  # noqa: E402
    _prediction_rows,
    _summarize_results,
    evaluate_models,
    print_nested_compare,
    print_overfitting_diagnosis,
)
from qixing_fga.evaluation.metrics import (  # noqa: E402
    _compute_metrics,
    expected_random_baseline_accuracy,
    expected_random_baseline_mae,
    participant_level_summary,
    random_baseline_fold_metrics,
)
from qixing_fga.features.columns import (  # noqa: E402
    FEATURE_COLUMNS,
    FeaturePoolMode,
    GAIT_NUMERIC_COLUMNS,
    LEAKY_METADATA_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    aggregate_top_features_by_frequency,
    load_features_from_json,
    reject_leaky_columns,
    resolve_feature_columns,
)
from qixing_fga.models.registry import (  # noqa: E402
    HAS_MORD,
    HAS_XGBOOST,
    TaskType,
    build_models,
    build_models_for_task,
    build_regression_models,
    get_param_grid,
    maybe_tune,
)
from qixing_fga.preprocessing import (  # noqa: E402
    build_model_preprocessor,
    build_numeric_preprocessor,
)
from qixing_fga.reporting.artifacts import save_results_to_dir  # noqa: E402
from qixing_fga.reporting.plots import save_prediction_figures  # noqa: E402
from qixing_fga.selection.rfecv import (  # noqa: E402
    _evaluate_outer_fold,
    _pipeline_importance_getter,
    rfecv_select_features,
    run_nested_cv_rfecv,
)

__all__ = [
    "DEFAULT_BOOTSTRAP_NPZ",
    "DEFAULT_FEATURE_PATH",
    "DEFAULT_ID_COL",
    "DEFAULT_LABEL_PATH",
    "DEFAULT_SPLITS",
    "DEFAULT_TARGET_COL",
    "FEATURE_COLUMNS",
    "FeaturePoolMode",
    "GAIT_NUMERIC_COLUMNS",
    "HAS_MORD",
    "HAS_SGKF",
    "HAS_XGBOOST",
    "LEAKY_METADATA_COLUMNS",
    "MODEL_FEATURE_COLUMNS",
    "PROJECT_ROOT",
    "TaskType",
    "add_sample_id",
    "aggregate_top_features_by_frequency",
    "build_fold_indices_from_bootstrap",
    "build_model_preprocessor",
    "build_models",
    "build_models_for_task",
    "build_numeric_preprocessor",
    "build_regression_models",
    "compute_continuous_cv_splits",
    "compute_inner_cv_splits",
    "continuous_effective_n_splits",
    "effective_n_splits",
    "evaluate_models",
    "expected_random_baseline_accuracy",
    "expected_random_baseline_mae",
    "get_cv_splits",
    "get_param_grid",
    "load_features_from_json",
    "load_merged_data",
    "load_single_file",
    "load_teammate_bootstrap_npz",
    "main",
    "maybe_tune",
    "participant_level_summary",
    "per_fold_bootstrap_mae_rmse",
    "prepare_features",
    "print_nested_compare",
    "print_overfitting_diagnosis",
    "random_baseline_fold_metrics",
    "reject_leaky_columns",
    "resolve_feature_columns",
    "rfecv_select_features",
    "run_nested_cv_rfecv",
    "save_prediction_figures",
    "save_results_to_dir",
    "sorted_participant_ids",
]


if __name__ == "__main__":
    main()
