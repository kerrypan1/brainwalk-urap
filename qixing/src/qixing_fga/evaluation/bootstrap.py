"""Thin re-export / fallback loader for utils.per_fold_bootstrap."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# src/qixing_fga/evaluation/bootstrap.py -> parents[3] = project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from utils.per_fold_bootstrap import (
        load_teammate_bootstrap_npz,
        per_fold_bootstrap_mae_rmse,
        sorted_participant_ids,
    )
except ModuleNotFoundError:
    bootstrap_path = _PROJECT_ROOT / "utils" / "per_fold_bootstrap.py"
    spec = importlib.util.spec_from_file_location(
        "per_fold_bootstrap", bootstrap_path
    )
    if spec is None or spec.loader is None:
        raise
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    load_teammate_bootstrap_npz = module.load_teammate_bootstrap_npz
    per_fold_bootstrap_mae_rmse = module.per_fold_bootstrap_mae_rmse
    sorted_participant_ids = module.sorted_participant_ids

__all__ = [
    "load_teammate_bootstrap_npz",
    "per_fold_bootstrap_mae_rmse",
    "sorted_participant_ids",
]
