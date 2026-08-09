"""Build a flat index of every raw gait video.

Walks `data/raw/bw_gait_videos/{batch}/BW-XXXX/YYYY_MM_DD/gait_vertical_{PROTOCOL}_{n}.mp4`
and writes `artifacts/video_index.csv` with one row per video.
"""
from __future__ import annotations

import os
import re

import pandas as pd

from common import RAW_VIDEO_DIR, BATH_FW_DIR, BATH_PWS_DIR, ensure_artifacts, participant_id, norm_date

FNAME_RE = re.compile(r"gait_vertical_(?P<protocol>[A-Za-z]+)_(?P<trial>\d+)\.mp4$", re.IGNORECASE)


def build_raw_index() -> pd.DataFrame:
    rows = []
    for dp, _dn, fns in os.walk(RAW_VIDEO_DIR):
        for fn in fns:
            if not fn.lower().endswith(".mp4"):
                continue
            m = FNAME_RE.search(fn)
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, RAW_VIDEO_DIR)
            parts = rel.split(os.sep)
            # parts: [batch, BW-XXXX, YYYY_MM_DD, filename]
            batch = parts[0] if len(parts) >= 1 else None
            pid = participant_id(parts[1]) if len(parts) >= 2 else None
            date = norm_date(parts[2]) if len(parts) >= 3 else None
            protocol = m.group("protocol").upper() if m else None
            trial = int(m.group("trial")) if m else None
            rows.append(
                {
                    "source": "raw",
                    "batch": batch,
                    "patient_id": pid,
                    "date": date,
                    "protocol": protocol,
                    "trial": trial,
                    "filename": fn,
                    "rel_path": rel.replace(os.sep, "/"),
                }
            )
    df = pd.DataFrame(rows)
    df["video_id"] = (
        df["patient_id"].astype(str)
        + "|" + df["date"].astype(str)
        + "|" + df["protocol"].astype(str)
        + "|" + df["trial"].astype(str)
    )
    return df


def build_bath_index() -> pd.DataFrame:
    """Index the pre-extracted labeled clips (bath_fw / bath_pws)."""
    rows = []
    for d, proto in [(BATH_FW_DIR, "FW"), (BATH_PWS_DIR, "PWS")]:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.mp4")):
            stem = p.stem  # {PPPP}_{visit}
            m = re.match(r"(\d+)_(\d+)$", stem)
            pid = participant_id(m.group(1)) if m else None
            visit = int(m.group(2)) if m else None
            rows.append(
                {
                    "source": f"bath_{proto.lower()}",
                    "patient_id": pid,
                    "protocol": proto,
                    "visit_index": visit,
                    "filename": p.name,
                    "rel_path": str(p.relative_to(d.parents[1])).replace(os.sep, "/"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = ensure_artifacts()

    raw = build_raw_index()
    raw_path = out_dir / "video_index.csv"
    raw.to_csv(raw_path, index=False)

    bath = build_bath_index()
    bath_path = out_dir / "bath_index.csv"
    bath.to_csv(bath_path, index=False)

    print(f"[video_index] {len(raw)} raw videos -> {raw_path}")
    print("  by protocol:")
    print(raw["protocol"].value_counts(dropna=False).to_string())
    print(f"  unique patients: {raw['patient_id'].nunique()}  unique dates: {raw['date'].nunique()}")
    print(f"  rows missing date: {raw['date'].isna().sum()}  missing protocol: {raw['protocol'].isna().sum()}")
    print(f"[bath_index] {len(bath)} labeled clips -> {bath_path}")
    if len(bath):
        print(bath.groupby(['source']).size().to_string())


if __name__ == "__main__":
    main()
