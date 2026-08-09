"""Path constants for the modeling code (mirrors data_build/common paths)."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # brainwalk-clip-(model2-3)/
REPO_ROOT = PROJECT_ROOT.parent                       # kerry/ package root (contains data/)
DATA_DIR = REPO_ROOT / "data"
BATH_FW_DIR = DATA_DIR / "bath_fw"
SPLIT_CSV = DATA_DIR / "participant_stratified_groupkfold_split_seed42.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for _d in (ARTIFACTS_DIR, CACHE_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def resolve_video_path(path: str | Path, stem: str | None = None) -> Path:
    """Resolve a labeled video path for the current checkout.

    Prefers repo-relative paths written by ``data.labeled_table`` (e.g.
    ``data/bath_fw/0002_1.mp4``). Absolute paths that still exist are kept.
    Stale absolute paths from a moved checkout fall back to
    ``data/bath_fw/{stem}.mp4``.
    """
    p = Path(path)
    if p.is_file():
        return p.resolve()
    if not p.is_absolute():
        cand = (REPO_ROOT / p).resolve()
        if cand.is_file():
            return cand
    if stem:
        cand = BATH_FW_DIR / f"{stem}.mp4"
        if cand.is_file():
            return cand.resolve()
    cand = BATH_FW_DIR / p.name
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError(
        f"video not found for path={path!r} stem={stem!r}; "
        f"expected under {BATH_FW_DIR}"
    )
