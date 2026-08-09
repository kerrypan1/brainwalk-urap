"""Shared CV helpers for §26 experiments (seed-42 patient 5-fold default)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.paths import ARTIFACTS_DIR

FGA_4CLASS = (0, 1, 2, 3)
FGA_3CLASS = (0, 1, 2)  # collapsed: severe+moderate -> 0, mild -> 1, normal -> 2


def collapse_fga_3(y: np.ndarray) -> np.ndarray:
    """Merge FGA classes 0+1 -> 0, 2 -> 1, 3 -> 2."""
    y = np.asarray(y, dtype=int)
    out = np.empty_like(y)
    out[y <= 1] = 0
    out[y == 2] = 1
    out[y == 3] = 2
    return out


def load_labeled_features(features_path: str, target_col: str = "fga_score",
                          collapse_3: bool = False, dropna_target: bool = True):
    """Load mean-pooled per-frame features + labels for one target column.

    Uses the existing seed-42 **5-fold** patient split in `labeled_fw.csv` (`fold_0`..`fold_4`).
    """
    z = np.load(features_path, allow_pickle=True)
    ids = list(z["ids"])
    feats = z["feats"].astype(np.float32)
    id_to_row = {s: i for i, s in enumerate(ids)}

    df = pd.read_csv(ARTIFACTS_DIR / "labeled_fw.csv")
    if target_col not in df.columns:
        raise KeyError(f"{target_col} not in labeled_fw.csv — rebuild via data.labeled_table")

    if dropna_target:
        df = df[df[target_col].notna()].copy()
    df = df[df["stem"].isin(id_to_row) & df["fold"].notna()].reset_index(drop=True)

    X = np.stack([feats[id_to_row[s]].mean(axis=0) for s in df["stem"]]).astype(np.float32)
    y = df[target_col].astype(int).to_numpy()
    if collapse_3 and target_col == "fga_score":
        y = collapse_fga_3(y)
    folds = df["fold"].to_numpy()
    stems = df["stem"].to_numpy()
    return X, y, folds, stems, df
