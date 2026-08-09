"""Zeno metric feature preparation for the contrastive branch.

Given the paired corpus (pairs.csv) and a set of TRAIN row indices, standardize
the selected gait metrics using TRAIN statistics only, median-impute missing
values (train medians), and append a missingness mask. No test/val leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.paths import ARTIFACTS_DIR

KEY_COLS = ["patient_id", "date", "protocol", "trial", "batch", "filename",
            "rel_path", "video_id", "source", "visit_index", "zeno_n_trials", "split"]


def metric_columns(pairs: pd.DataFrame) -> list[str]:
    # metric columns = numeric columns that are not keys/meta
    cols = []
    for c in pairs.columns:
        if c in KEY_COLS:
            continue
        if pd.api.types.is_numeric_dtype(pairs[c]):
            cols.append(c)
    return cols


def build_zeno_matrix(pairs: pd.DataFrame, train_mask: np.ndarray):
    """Return (Z, meta) where Z = [N, 2M] standardized metrics + mask.

    Stats computed on train_mask rows only.
    """
    mcols = metric_columns(pairs)
    raw = pairs[mcols].to_numpy(dtype=np.float64)  # [N, M]
    miss = np.isnan(raw).astype(np.float32)         # 1 = missing

    tr = raw[train_mask]
    mean = np.nanmean(tr, axis=0)
    std = np.nanstd(tr, axis=0)
    std = np.where((std == 0) | ~np.isfinite(std), 1.0, std)
    median = np.nanmedian(tr, axis=0)
    median = np.where(~np.isfinite(median), 0.0, median)

    filled = np.where(np.isnan(raw), median, raw)
    z = (filled - mean) / std
    z = np.clip(z, -5, 5).astype(np.float32)         # guard against outliers

    Z = np.concatenate([z, miss], axis=1).astype(np.float32)  # [N, 2M]
    stats = {"mcols": mcols, "mean": mean, "std": std, "median": median}
    return Z, stats


def load_pairs_with_split() -> pd.DataFrame:
    pairs = pd.read_csv(ARTIFACTS_DIR / "pairs.csv")
    split = pd.read_csv(ARTIFACTS_DIR / "corpus_split.csv")[["video_id", "split"]]
    return pairs.merge(split, on="video_id", how="left")
