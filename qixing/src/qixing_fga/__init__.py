"""Qixing FGA prediction pipeline (publication-quality, modular).

Import preferred entry points from submodules::

    from qixing_fga.cli.train import main
    from qixing_fga.config import load_config
    from qixing_fga.reporting.artifacts import make_run_dir
"""

from . import artifacts, config

__all__ = ["config", "artifacts"]
