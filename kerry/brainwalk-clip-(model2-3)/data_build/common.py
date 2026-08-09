"""Shared paths and parsing helpers for Model 2/3 data builders.

All builders resolve paths relative to the repo root (the `ucsf/` folder that
contains both `data/` and `brainwalk-clip-(model2-3)/`), so scripts run
correctly from any working directory.
"""
from __future__ import annotations

import re
from pathlib import Path

# brainwalk-clip-(model2-3)/data_build/common.py -> repo root is two parents up
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # brainwalk-clip-(model2-3)/

DATA_DIR = REPO_ROOT / "data"
BATH_FW_DIR = DATA_DIR / "bath_fw"
BATH_PWS_DIR = DATA_DIR / "bath_pws"
RAW_VIDEO_DIR = DATA_DIR / "raw" / "bw_gait_videos"
ZENO_XLSX = DATA_DIR / "raw" / "zeno" / "2025_12_03_BW_MS_ZenoData.xlsx"
REVIEW_XLSX = DATA_DIR / "raw" / "zeno" / "BW_gait_videos_DPT_review.xlsx"
SPLIT_CSV = DATA_DIR / "participant_stratified_groupkfold_split_seed42.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

WALKING_PROTOCOLS = ["FW", "PWS", "DTW"]
ALL_PROTOCOLS = ["FW", "PWS", "DTW", "TUG", "QSLOS"]

# Supervised FGA fields (from the DPT review sheet)
FGA_FIELDS = [
    "speed",
    "assistive_device",
    "imbalance",
    "gait_deviation",
    "deviation_outside_walkway",
    "fga_score",
]

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
_ID_RE = re.compile(r"(\d+)")


def ensure_artifacts() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def participant_num(x) -> int | None:
    """Normalize an integer-like participant identifier to its numeric form."""
    if x is None:
        return None
    m = _ID_RE.search(str(x))
    return int(m.group(1)) if m else None


def participant_id(x) -> str | None:
    """Return the canonical zero-padded ``BW-####`` participant identifier."""
    n = participant_num(x)
    return f"BW-{n:04d}" if n is not None else None


def leading_number(x):
    """Extract first numeric token from a messy cell like '2 normal' -> 2.0.

    Returns float, or None if no number present.
    """
    import pandas as pd

    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = _NUM_RE.search(str(x))
    return float(m.group(0)) if m else None


def norm_date(x) -> str | None:
    """Normalize any date-ish value to 'YYYY-MM-DD' string, else None."""
    import pandas as pd

    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    # Folder-style '2022_09_12'
    if isinstance(x, str):
        s = x.strip().replace("_", "-").replace("/", "-")
        dt = pd.to_datetime(s, errors="coerce")
    else:
        dt = pd.to_datetime(x, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.strftime("%Y-%m-%d")
