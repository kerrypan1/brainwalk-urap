"""Cross-validation split construction."""

__all__ = [
    "HAS_SGKF",
    "build_fold_indices_from_bootstrap",
    "compute_continuous_cv_splits",
    "compute_inner_cv_splits",
    "continuous_effective_n_splits",
    "effective_n_splits",
    "get_cv_splits",
]


def __getattr__(name: str):
    from . import splits as _splits

    if name in __all__:
        return getattr(_splits, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
