"""Model builders and registries."""

__all__ = [
    "HAS_MORD",
    "HAS_XGBOOST",
    "TaskType",
    "build_models",
    "build_models_for_task",
    "build_regression_models",
    "get_param_grid",
    "maybe_tune",
]


def __getattr__(name: str):
    from . import registry as _registry

    if name in __all__:
        return getattr(_registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
