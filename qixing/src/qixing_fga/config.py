"""Experiment configuration: YAML load + deep-merge over defaults + overrides.

A config is a plain nested dict. `load_config` merges a YAML file over
`DEFAULT_CONFIG`; `apply_overrides` layers non-None CLI values on top so the
resolved config can be saved verbatim with each run (reproducibility).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import yaml

# Visits excluded from the study by agreement with the team (2026-08-06).
# BW-0271 visit 2 (2025-04-21) is the only recording whose camera angle differs
# from every other video in the set, so its pose features are not comparable;
# BW-0271 visit 1 (2024-06-03) is retained. This takes the double-visit view to
# n = 46 x 2 - 1 = 91. Listed as sample_id (participant_id + "_" + video_index).
STUDY_EXCLUDED_SAMPLE_IDS: list[str] = ["BW-0271_2"]

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_name": "fga",
    "random_seed": 42,
    "task": "classification",  # classification | regression | regression_continuous
    "data": {
        "features": "data/2026_05_17_FWOnly_2visits_features.csv",
        "labels": "data/2026_05_17_FWOnly_2visits_labels.csv",
        "data_file": None,
        "id_col": "sample_id",
        "target": "fga_estimate_score",
        # Row view: "all" keeps every visit; an integer keeps that video_index
        # only (one sample per participant — the finalised FW-only definition).
        "visit": "all",
        "exclude_sample_ids": list(STUDY_EXCLUDED_SAMPLE_IDS),
    },
    "cv": {
        "strategy": "stratified_group",  # stratified_group | group | bootstrap_aligned
        "n_splits": 5,
        "bootstrap_npz": "data/bootstrap_indices.npz",
        "use_bootstrap_folds": False,
    },
    "model": {"tune": "none", "tune_splits": 3},
    "feature_selection": {
        "method": "none",  # none | rfecv
        "rfecv_min_features": 5,
        "rfecv_step": 1,
        "features_json": None,
    },
    # participant_level is an opt-in SECONDARY diagnostic only. The reported
    # evaluation metric is sample-level (each visit scored against its own label);
    # averaging a participant's visits conflates their distinct FGA labels.
    "reporting": {"ci_alpha": 0.05, "participant_level": False},
}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Optional[str]) -> dict[str, Any]:
    """Load a YAML config merged over `DEFAULT_CONFIG` (defaults if path is None)."""
    if not path:
        return copy.deepcopy(DEFAULT_CONFIG)
    text = Path(path).read_text(encoding="utf-8")
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file {path} must contain a mapping at top level.")
    return deep_merge(DEFAULT_CONFIG, loaded)


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Layer non-None dotted overrides (e.g. {'data.target': 'speed'}) onto config."""
    result = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        keys = dotted_key.split(".")
        node = result
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return result
