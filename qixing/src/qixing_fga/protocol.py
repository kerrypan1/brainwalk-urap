"""Canonical data view + fold indices for the finalised FW single-visit protocol.

Every downstream analysis (baselines, ablation, SHAP, multi-outcome) must see
exactly the same rows, folds and bootstrap resamples as the main training run,
otherwise their numbers cannot be compared with each other. Loading that view
lives here once rather than being re-derived per script.

See ``docs/CV_PROTOCOL.md`` for why the teammate npz needs no regeneration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .config import STUDY_EXCLUDED_SAMPLE_IDS
from .cv.splits import build_fold_indices_from_bootstrap
from .data.loading import (
    exclude_samples,
    load_merged_data,
    prepare_features,
    select_visit,
)
from .evaluation.bootstrap import load_teammate_bootstrap_npz
from .features.columns import resolve_feature_columns

DEFAULT_FEATURES = "data/2026_05_17_FWOnly_2visits_features.csv"
DEFAULT_LABELS = "data/2026_05_17_FWOnly_2visits_labels.csv"
DEFAULT_NPZ = "data/bootstrap_indices.npz"
DEFAULT_TARGET = "fga_estimate_score"


@dataclass
class ProtocolData:
    """One immutable view of the study data plus its fixed folds."""

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    ids: pd.Series
    fold_indices: list[tuple[list[int], list[int]]]
    bootstrap_by_fold: dict[str, np.ndarray]
    feature_names: list[str]
    visit: object
    target: str

    @property
    def n_samples(self) -> int:
        return len(self.y)

    @property
    def n_participants(self) -> int:
        return int(self.groups.nunique())

    def describe(self) -> str:
        return (
            f"n={self.n_samples} samples / {self.n_participants} participants, "
            f"{len(self.feature_names)} features, visit={self.visit}, "
            f"target={self.target}, folds={len(self.fold_indices)}"
        )


def load_protocol_data(
    *,
    features: str = DEFAULT_FEATURES,
    labels: str = DEFAULT_LABELS,
    npz: Optional[str] = DEFAULT_NPZ,
    target: str = DEFAULT_TARGET,
    visit: object = 1,
    n_splits: int = 5,
    task: str = "regression",
    root: Optional[Path] = None,
    exclude: Optional[Sequence[str]] = None,
) -> ProtocolData:
    """Load the FW study view with the teammate's fold assignment.

    ``exclude`` defaults to the study exclusion list so every downstream script
    (baselines, ablation, SHAP, diagnostics) drops the same unusable recordings
    as the main training run; pass an explicit sequence to override.
    """
    base = Path(root) if root else Path.cwd()

    def _p(rel: str) -> str:
        p = Path(rel)
        return str(p if p.is_absolute() else base / p)

    merged = load_merged_data(_p(features), _p(labels), "sample_id")
    # Exclude before the visit filter so the ids validate against every row.
    merged = exclude_samples(
        merged,
        STUDY_EXCLUDED_SAMPLE_IDS if exclude is None else exclude,
    )
    merged = select_visit(merged, visit)

    feature_cols = resolve_feature_columns("gait_numeric")
    X, y, _missing, valid_mask = prepare_features(
        merged, feature_cols, [], target, task=task
    )
    groups = merged.loc[valid_mask, "participant_id"].astype(str).reset_index(drop=True)
    ids = (
        merged.loc[valid_mask, "sample_id"].astype(str).reset_index(drop=True)
        if "sample_id" in merged.columns
        else groups
    )

    boot = load_teammate_bootstrap_npz(_p(npz), n_splits=n_splits) if npz else {}
    fold_indices = (
        build_fold_indices_from_bootstrap(groups, boot, n_splits) if boot else []
    )
    return ProtocolData(
        X=X,
        y=y,
        groups=groups,
        ids=ids,
        fold_indices=fold_indices,
        bootstrap_by_fold=boot,
        feature_names=list(X.columns),
        visit=visit,
        target=target,
    )
