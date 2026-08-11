"""Participant-grouped CV split helpers (StratifiedGroupKFold / GroupKFold)."""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGKF = True
except ImportError:
    HAS_SGKF = False

from ..evaluation.bootstrap import sorted_participant_ids
from ..models.registry import TaskType


def build_fold_indices_from_bootstrap(
    groups: pd.Series,
    bootstrap_by_fold: dict[str, object],
    n_splits: int,
) -> list[tuple[list[int], list[int]]]:
    group_array = groups.to_numpy()
    sorted_ids = sorted_participant_ids(group_array)
    folds: list[tuple[list[int], list[int]]] = []

    for fold_idx in range(n_splits):
        fold_key = f"fold_{fold_idx}"
        if fold_key not in bootstrap_by_fold:
            raise KeyError(f"Bootstrap npz missing key: {fold_key}")
        boot_matrix = bootstrap_by_fold[fold_key]
        unique_indices = sorted(set(int(i) for i in boot_matrix.ravel()))
        if len(unique_indices) != boot_matrix.shape[1]:
            warnings.warn(
                f"Fold {fold_idx}: inferred {len(unique_indices)} participants from bootstrap, "
                f"expected {boot_matrix.shape[1]}."
            )
        val_participants = [sorted_ids[i] for i in unique_indices]
        val_mask = np.isin(group_array, val_participants)
        test_idx = list(np.flatnonzero(val_mask))
        train_idx = list(np.flatnonzero(~val_mask))
        folds.append((train_idx, test_idx))

    return folds


def effective_n_splits(y: pd.Series, requested: int) -> int:
    min_count = int(y.value_counts().min())
    n_splits = min(requested, min_count)
    if n_splits < 2:
        raise ValueError("Not enough samples per class for stratified CV.")
    if n_splits != requested:
        warnings.warn(
            f"Reducing CV splits to {n_splits} due to small class counts."
        )
    return n_splits


def compute_inner_cv_splits(
    y: pd.Series,
    groups: pd.Series,
    n_splits: int,
    random_state: int,
) -> list[tuple[list[int], list[int]]]:
    """Group-stratified CV splits on a subset (outer-train only)."""
    if not HAS_SGKF:
        raise ImportError(
            "StratifiedGroupKFold is required for inner RFECV CV."
        )
    n_splits = effective_n_splits(y, n_splits)
    X_dummy = np.zeros((len(y), 1))
    skf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    return list(skf.split(X_dummy, y, groups))


def continuous_effective_n_splits(groups: pd.Series, requested: int) -> int:
    """Group-count-limited split count for continuous (unstratified) CV."""
    n_groups = int(groups.nunique())
    n_splits = min(requested, n_groups)
    if n_splits < 2:
        raise ValueError("Not enough participants for GroupKFold CV.")
    if n_splits != requested:
        warnings.warn(
            f"Reducing CV splits to {n_splits} due to participant count."
        )
    return n_splits


def compute_continuous_cv_splits(
    groups: pd.Series,
    n_splits: int,
) -> list[tuple[list[int], list[int]]]:
    """GroupKFold splits for continuous targets (no class stratification)."""
    n_splits = continuous_effective_n_splits(groups, n_splits)
    X_dummy = np.zeros((len(groups), 1))
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(X_dummy, groups=groups))


def get_cv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int,
    random_state: int,
    fold_indices: Optional[list[tuple[list[int], list[int]]]],
    task: TaskType = "classification",
) -> list[tuple[list[int], list[int]]]:
    if fold_indices is not None:
        if len(fold_indices) != n_splits:
            raise ValueError("Fold indices count must match n_splits.")
        return fold_indices

    if task == "regression_continuous":
        # Continuous targets cannot be class-stratified; group by participant.
        gkf = GroupKFold(n_splits=n_splits)
        return list(gkf.split(X, groups=groups))

    if not HAS_SGKF:
        raise ImportError(
            "StratifiedGroupKFold is unavailable in this scikit-learn version. "
            "Please upgrade scikit-learn to use group-stratified CV."
        )

    skf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    try:
        return list(skf.split(X, y, groups))
    except ValueError:
        # A class may be too small (or too concentrated within participants) to
        # stratify; degrade gracefully to participant-grouped folds (no leakage).
        warnings.warn(
            "StratifiedGroupKFold infeasible (small/rare class); "
            "falling back to GroupKFold."
        )
        return list(GroupKFold(n_splits=n_splits).split(X, groups=groups))
