"""Build FGA labels from the DPT review workbook.

The review sheet has one row per participant with two timepoints (video1/video2).
We reshape to long format: one row per (patient, visit_index) with the six
ordinal fields. Labels were assigned from FW videos, so downstream these attach
to the FW clips only.

Output: artifacts/fga_labels.csv
"""
from __future__ import annotations

import pandas as pd

from common import REVIEW_XLSX, FGA_FIELDS, ensure_artifacts, participant_id, leading_number

# Map canonical field -> base column name in the review sheet (pre-normalization)
FIELD_TO_BASE = {
    "speed": "speed",
    "assistive_device": "assistive_device",
    "imbalance": "imbalance",
    "gait_deviation": "gait_deviation",
    "deviation_outside_walkway": "deviation_outside_walkway",
    "fga_score": "fga_estimate_score",
}


def _norm(colname: str) -> str:
    return str(colname).strip().lower().replace(" ", "").replace("\n", "")


def find_col(df: pd.DataFrame, base: str, suffix: str) -> str | None:
    target = f"{base}{suffix}"  # e.g. 'speed1', 'fga_estimate_score1'
    for c in df.columns:
        if _norm(c).startswith(target):
            return c
    return None


def main() -> None:
    out_dir = ensure_artifacts()
    df = pd.read_excel(REVIEW_XLSX, sheet_name=0, engine="openpyxl")

    date_col = {1: "visit_date_video1", 2: "visit_date_video2"}
    rows = []
    for _, r in df.iterrows():
        pid = participant_id(r.get("BW-ID"))
        if pid is None:
            continue
        for visit in (1, 2):
            rec = {"patient_id": pid, "visit_index": visit}
            dcol = date_col[visit]
            rec["visit_date"] = None
            if dcol in df.columns:
                from common import norm_date

                rec["visit_date"] = norm_date(r.get(dcol))
            n_present = 0
            for field, base in FIELD_TO_BASE.items():
                col = find_col(df, base, str(visit))
                val = leading_number(r.get(col)) if col else None
                rec[field] = val
                if field != "fga_score" and val is not None:
                    n_present += 1
            rows.append(rec)

    out = pd.DataFrame(rows, columns=["patient_id", "visit_index", "visit_date"] + FGA_FIELDS)
    # Keep only rows that actually have an FGA score
    out_labeled = out[out["fga_score"].notna()].copy()
    out_labeled["fga_score"] = out_labeled["fga_score"].astype(int)

    path = out_dir / "fga_labels.csv"
    out_labeled.to_csv(path, index=False)

    print(f"[fga_labels] {len(out_labeled)} labeled (patient,visit) rows -> {path}")
    print(f"  unique patients: {out_labeled['patient_id'].nunique()}")
    print("  FGA class distribution (fga_score):")
    print(out_labeled["fga_score"].value_counts().sort_index().to_string())
    print("  per-field non-null counts:")
    print(out_labeled[FGA_FIELDS].notna().sum().to_string())
    miss_date = out_labeled["visit_date"].isna().sum()
    print(f"  rows missing visit_date: {miss_date}")


if __name__ == "__main__":
    main()
