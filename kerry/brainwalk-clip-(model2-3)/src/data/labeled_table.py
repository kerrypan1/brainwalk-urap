"""Build the labeled FW table: one row per curated bath_fw clip with its FGA
class and CV fold. This is the supervised dataset for Phase 1.

Output: artifacts/labeled_fw.csv
"""
from __future__ import annotations

import re

import pandas as pd

from utils.paths import ARTIFACTS_DIR, BATH_FW_DIR, SPLIT_CSV


FGA_ITEM_FIELDS = [
    "speed",
    "assistive_device",
    "imbalance",
    "gait_deviation",
    "deviation_outside_walkway",
]


def build() -> pd.DataFrame:
    fga = pd.read_csv(ARTIFACTS_DIR / "fga_labels.csv")
    split = pd.read_csv(SPLIT_CSV)

    def pid_norm(x):
        m = re.search(r"(\d+)", str(x))
        return f"BW-{int(m.group(1)):04d}" if m else None

    split = split.assign(patient_id=split["participant_id"].map(pid_norm))
    fold_of = dict(zip(split["patient_id"], split["split"]))

    rows = []
    for p in sorted(BATH_FW_DIR.glob("*.mp4")):
        m = re.match(r"(\d+)_(\d+)$", p.stem)
        if not m:
            continue
        pid = f"BW-{int(m.group(1)):04d}"
        visit = int(m.group(2))
        rows.append({"stem": p.stem, "path": str(p), "patient_id": pid, "visit_index": visit})
    clips = pd.DataFrame(rows)

    fga_cols = ["patient_id", "visit_index", "fga_score", *FGA_ITEM_FIELDS]
    merged = clips.merge(fga[fga_cols], on=["patient_id", "visit_index"], how="left")
    merged["fold"] = merged["patient_id"].map(fold_of)
    return merged


def main() -> None:
    df = build()
    path = ARTIFACTS_DIR / "labeled_fw.csv"
    df.to_csv(path, index=False)
    n_missing_label = df["fga_score"].isna().sum()
    n_missing_fold = df["fold"].isna().sum()
    print(f"[labeled_fw] {len(df)} clips -> {path}")
    print(f"  missing fga_score: {n_missing_label}  missing fold: {n_missing_fold}")
    print("  class distribution:")
    print(df["fga_score"].value_counts(dropna=False).sort_index().to_string())
    print("  clips per fold:")
    print(df["fold"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
