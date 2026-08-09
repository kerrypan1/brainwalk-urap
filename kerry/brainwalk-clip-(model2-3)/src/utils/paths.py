"""Path constants for the modeling code (mirrors data_build/common paths)."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # brainwalk-clip-(model2-3)/
REPO_ROOT = PROJECT_ROOT.parent                       # ucsf/
DATA_DIR = REPO_ROOT / "data"
BATH_FW_DIR = DATA_DIR / "bath_fw"
SPLIT_CSV = DATA_DIR / "participant_stratified_groupkfold_split_seed42.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for _d in (ARTIFACTS_DIR, CACHE_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
