"""Command-line entry points."""

__all__ = ["main"]


def __getattr__(name: str):
    if name == "main":
        from .train import main as _main

        return _main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
