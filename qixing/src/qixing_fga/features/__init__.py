"""Feature column pools and helpers."""

__all__ = [
    "FEATURE_COLUMNS",
    "FeaturePoolMode",
    "GAIT_NUMERIC_COLUMNS",
    "LEAKY_METADATA_COLUMNS",
    "MODEL_FEATURE_COLUMNS",
    "aggregate_top_features_by_frequency",
    "load_features_from_json",
    "reject_leaky_columns",
    "resolve_feature_columns",
]


def __getattr__(name: str):
    from . import columns as _columns

    if name in __all__:
        return getattr(_columns, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
