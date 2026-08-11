"""Compatibility shim: re-export from ``qixing_fga.reporting.artifacts``."""

from .reporting.artifacts import (  # noqa: F401
    file_sha256,
    git_commit_short,
    make_run_dir,
    save_results_to_dir,
    write_config,
    write_environment,
    write_feature_importance,
    write_metrics,
    write_predictions,
    write_provenance,
)

__all__ = [
    "file_sha256",
    "git_commit_short",
    "make_run_dir",
    "save_results_to_dir",
    "write_config",
    "write_environment",
    "write_feature_importance",
    "write_metrics",
    "write_predictions",
    "write_provenance",
]
