"""Load and merge feature/label tables; prepare model matrices."""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..features.columns import LEAKY_METADATA_COLUMNS, reject_leaky_columns
from ..models.registry import TaskType


def add_sample_id(df: pd.DataFrame) -> pd.DataFrame:
    if "participant_id" in df.columns and "video_index" in df.columns:
        df = df.copy()
        df["sample_id"] = df["participant_id"].astype(str) + "_" + df["video_index"].astype(str)
    return df


def load_merged_data(feature_path: str, label_path: str, id_col: str) -> pd.DataFrame:
    features = pd.read_csv(feature_path)
    labels = pd.read_csv(label_path)
    features = add_sample_id(features)
    labels = add_sample_id(labels)

    if id_col in features.columns and id_col in labels.columns:
        merged = pd.merge(features, labels, on=id_col, how="inner")
    elif "participant_id" in features.columns and "video_index" in features.columns:
        merged = pd.merge(
            features,
            labels,
            on=["participant_id", "video_index"],
            how="inner",
        )
    else:
        raise ValueError("No common id columns found to merge features and labels.")

    merged = _coalesce_merge_columns(merged, "participant_id")
    merged = _coalesce_merge_columns(merged, "video_index")

    return merged


def exclude_samples(
    df: pd.DataFrame, exclude: Optional[Sequence[str]], *, strict: bool = True
) -> pd.DataFrame:
    """Drop named ``sample_id`` rows (unusable recordings) from the study view.

    Applied to the full merged table *before* any visit filter, so the ids are
    validated against every row that exists rather than whichever subset the
    current run happens to look at. ``strict`` raises when a listed id is absent:
    a typo in the exclusion list would otherwise silently leave the row in the
    study, which is the failure mode this list exists to prevent.
    """
    if not exclude:
        return df
    if "sample_id" not in df.columns:
        raise ValueError("exclude_sample_ids requires a 'sample_id' column.")

    wanted = list(dict.fromkeys(str(s) for s in exclude))
    present = set(df["sample_id"].astype(str))
    missing = [s for s in wanted if s not in present]
    if missing and strict:
        raise ValueError(
            f"exclude_sample_ids not found in the data: {missing}. "
            "Check the id spelling (expected participant_id + '_' + video_index)."
        )
    if missing:
        warnings.warn(f"exclude_sample_ids not found (ignored): {missing}")

    keep = ~df["sample_id"].astype(str).isin(wanted)
    return df.loc[keep].reset_index(drop=True)


def select_visit(df: pd.DataFrame, visit: object) -> pd.DataFrame:
    """Keep a single visit per participant (the finalised one-sample-per-person view).

    ``visit="all"`` (default) is a no-op, so existing multi-visit runs are
    unaffected. An integer keeps only rows with that ``video_index``; the label
    is then that visit's own score, never merged across visits. Participants
    without the requested visit are dropped, which is reported by the caller
    via the row/participant counts.
    """
    if visit is None or (isinstance(visit, str) and str(visit).lower() == "all"):
        return df
    if "video_index" not in df.columns:
        raise ValueError(
            "data.visit filtering requires a 'video_index' column in the merged table."
        )
    wanted = int(visit)
    filtered = df.loc[pd.to_numeric(df["video_index"], errors="coerce") == wanted]
    if filtered.empty:
        raise ValueError(
            f"data.visit={wanted} matched no rows; available: "
            f"{sorted(pd.to_numeric(df['video_index'], errors='coerce').dropna().unique().tolist())}"
        )
    dup = filtered["participant_id"].duplicated().sum() if "participant_id" in filtered else 0
    if dup:
        warnings.warn(
            f"data.visit={wanted} still leaves {dup} duplicated participant row(s); "
            "the one-sample-per-participant assumption does not hold for this table."
        )
    return filtered.reset_index(drop=True)


def load_single_file(path: str) -> pd.DataFrame:
    """Load a single wide table containing features + targets + participant_id.

    Skips the features/labels merge. A ``sample_id`` is derived from
    ``participant_id`` (plus ``video_index`` when present, else a
    within-participant row counter) so per-sample prediction export works and
    participant grouping stays leakage-free.
    """
    df = pd.read_csv(path)
    df = add_sample_id(df)
    if "sample_id" not in df.columns:
        if "participant_id" not in df.columns:
            raise ValueError(
                "Single-file mode requires a 'participant_id' column."
            )
        parts = df["participant_id"].astype(str)
        if "walking_condition" in df.columns:
            parts = parts + "_" + df["walking_condition"].astype(str)
        parts = parts + "_" + df.groupby("participant_id").cumcount().astype(str)
        df = df.assign(sample_id=parts)
    return df


def _coalesce_merge_columns(df: pd.DataFrame, base: str) -> pd.DataFrame:
    if base in df.columns:
        return df

    col_x = f"{base}_x"
    col_y = f"{base}_y"
    if col_x in df.columns and col_y in df.columns:
        if not df[col_x].equals(df[col_y]):
            warnings.warn(
                f"Columns {col_x} and {col_y} differ; using {col_x} as {base}."
            )
        df = df.copy()
        df[base] = df[col_x]
        return df
    if col_x in df.columns:
        df = df.copy()
        df[base] = df[col_x]
        return df
    if col_y in df.columns:
        df = df.copy()
        df[base] = df[col_y]
        return df
    return df


def _parse_numeric_target(raw: pd.Series) -> pd.Series:
    """Coerce a target column to numeric, tolerating text-prefixed labels.

    Values may be plain numbers (e.g. ``2`` or ``1.34``) or text-prefixed
    ordinal labels (e.g. ``"2 normal"``). Direct numeric coercion is tried
    first; for cells that fail but are non-empty, the leading numeric prefix
    is extracted (e.g. ``"2 normal" -> 2``).
    """
    numeric = pd.to_numeric(raw, errors="coerce")
    non_empty = raw.notna() & (raw.astype(str).str.strip() != "")
    needs_prefix = numeric.isna() & non_empty
    if needs_prefix.any():
        extracted = (
            raw[needs_prefix].astype(str).str.extract(r"(-?\d+\.?\d*)", expand=False)
        )
        numeric.loc[needs_prefix] = pd.to_numeric(extracted, errors="coerce")
    return numeric


def prepare_features(
    merged: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    task: TaskType = "classification",
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.Series]:
    if target_col not in merged.columns:
        raise ValueError(f"Target column '{target_col}' not found in merged data.")

    feature_cols = reject_leaky_columns(feature_cols, context="prepare_features")
    if categorical_cols:
        cat_leaky = [c for c in categorical_cols if c in LEAKY_METADATA_COLUMNS]
        if cat_leaky:
            raise ValueError(
                f"Categorical metadata columns are not allowed: {cat_leaky}"
            )

    available_cols = [col for col in feature_cols if col in merged.columns]
    missing_cols = [col for col in feature_cols if col not in merged.columns]

    X = merged[available_cols].copy()
    for col in available_cols:
        if col in categorical_cols:
            X[col] = X[col].astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    y = _parse_numeric_target(merged[target_col])
    valid_mask = y.notna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    if task == "regression_continuous":
        # Keep float targets as-is; no rounding for continuous outcomes.
        y = y.astype(float)
        return X, y, missing_cols, valid_mask

    y_non_null = y.dropna()
    if not (y_non_null % 1 == 0).all():
        warnings.warn("Target has non-integer values; rounding for classification.")
        y = y.round()

    y = y.astype(int)

    if task == "classification":
        # Some estimators (mord, XGBoost) require contiguous 0..K-1 labels.
        # Nominal targets like assistive_device may be {0,1,2,4}; remap to
        # contiguous integers. This is identity for already-contiguous targets
        # (e.g. fga_estimate_score), so existing runs are unaffected. MAE on
        # nominal labels is not meaningful; accuracy/F1 are the relevant metrics.
        uniq = np.sort(y.unique())
        if not np.array_equal(uniq, np.arange(len(uniq))):
            mapping = {int(v): i for i, v in enumerate(uniq)}
            warnings.warn(
                f"Re-encoded non-contiguous classification labels to contiguous: {mapping}"
            )
            y = y.map(mapping).astype(int)

    return X, y, missing_cols, valid_mask
