"""Nested RFECV feature selection."""

__all__ = [
    "rfecv_select_features",
    "run_nested_cv_rfecv",
]


def __getattr__(name: str):
    from . import rfecv as _rfecv

    if name in __all__:
        return getattr(_rfecv, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
