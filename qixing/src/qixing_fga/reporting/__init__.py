"""Reporting: run artifacts, CSV dumps, and prediction figures.

Keep package init light; import submodules directly when needed.
"""

__all__ = [
    "file_sha256",
    "git_commit_short",
    "make_run_dir",
    "save_prediction_figures",
    "save_results_to_dir",
    "write_config",
    "write_environment",
    "write_feature_importance",
    "write_metrics",
    "write_predictions",
    "write_provenance",
]


def __getattr__(name: str):
    if name in {
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
    }:
        from . import artifacts as _artifacts

        return getattr(_artifacts, name)
    if name == "save_prediction_figures":
        from . import plots as _plots

        return getattr(_plots, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
