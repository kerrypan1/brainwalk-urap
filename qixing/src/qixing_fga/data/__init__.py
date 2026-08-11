"""Data loading and feature preparation."""

__all__ = [
    "add_sample_id",
    "load_merged_data",
    "load_single_file",
    "prepare_features",
]


def __getattr__(name: str):
    from . import loading as _loading

    if name in __all__:
        return getattr(_loading, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
