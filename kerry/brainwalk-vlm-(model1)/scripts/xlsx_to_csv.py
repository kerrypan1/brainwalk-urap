# clean_bw_xlsx_to_csv.py

# File Summary
# ------------
# Converts the raw review XLSX into a normalized `gt.csv` used by inference and
# evaluation scripts. The script:
# - cleans ID format (e.g., BW-XXXX -> numeric),
# - extracts leading numeric values from mixed text cells,
# - expands each subject into two rows (suffix _1 and _2),
# - outputs canonical columns expected by downstream scripts

import re
import numpy as np
import pandas as pd

# ===== HARD-CODE PATHS =====
INPUT_XLSX = "../data/raw/zeno/BW_gait_videos_DPT_review.xlsx"
OUTPUT_CSV = "gt.csv"
SHEET = 0
# ===========================

NUM_RE = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def extract_leading_number(x):
    """Extract first numeric token from a cell, else NaN"""
    if pd.isna(x):
        return np.nan
    m = NUM_RE.match(str(x).strip())
    if not m:
        return np.nan
    v = float(m.group(1))
    return int(v) if abs(v - int(v)) < 1e-12 else v


def bw_id_to_int(x):
    """Extract integer portion of a BW-style identifier"""
    if pd.isna(x):
        return np.nan
    m = re.search(r"(\d+)", str(x))
    return int(m.group(1)) if m else np.nan


# ---------- read ----------
# Load source workbook from configured sheet index
df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET, engine="openpyxl")

# ---------- drop unwanted columns ----------
# Drop common index or date columns not used for model targets
first_col = str(df.columns[0]).strip()
if first_col in {"0", "Unnamed: 0"}:
    df = df.drop(columns=[df.columns[0]])

if "visit_date_video1" in df.columns:
    df = df.drop(columns=["visit_date_video1"])

# ---------- id cleanup ----------
# Standardize "BW-ID" to canonical "id" numeric base
df["BW-ID"] = df["BW-ID"].apply(bw_id_to_int)
df = df.rename(columns={"BW-ID": "id"})

# ---------- numeric cleanup ----------
# Normalize all metric columns to numeric where possible
for c in df.columns:
    if c != "id":
        df[c] = df[c].apply(extract_leading_number)

# ---------- build long-format rows for *_1 and *_2 ----------
def find_col(base_name: str, suffix: str) -> str:
    """Find exact column name case-insensitively for metric suffix 1 or 2"""
    target = f"{base_name}{suffix}"
    for c in df.columns:
        if str(c).strip().lower() == target.lower():
            return c
    return ""


def find_speed_col(suffix: str) -> str:
    """Find speed column for a given suffix with tolerant naming match"""
    for c in df.columns:
        if f"speed{suffix}" in str(c).strip().lower():
            return c
    return ""


metric_templates = {
    "assistive_device": "assistive_device",
    "imbalance": "imbalance",
    "gait_deviation": "gait_deviation",
    "deviation_outside_walkway": "deviation_outside_walkway",
    "fga_score": "FGA_estimate_score",
}

rows = []
for _, r in df.iterrows():
    base_id = r["id"]
    if pd.isna(base_id):
        continue

    base_id_int = int(base_id)
    for suffix in ("1", "2"):
        # Resolve source columns for this suffix and emit one long-format row
        speed_col = find_speed_col(suffix)
        metric_cols = {k: find_col(v, suffix) for k, v in metric_templates.items()}

        row = {
            "id": f"{base_id_int}_{suffix}",
            "speed": r[speed_col] if speed_col else np.nan,
            "assistive_device": r[metric_cols["assistive_device"]] if metric_cols["assistive_device"] else np.nan,
            "imbalance": r[metric_cols["imbalance"]] if metric_cols["imbalance"] else np.nan,
            "gait_deviation": r[metric_cols["gait_deviation"]] if metric_cols["gait_deviation"] else np.nan,
            "deviation_outside_walkway": r[metric_cols["deviation_outside_walkway"]] if metric_cols["deviation_outside_walkway"] else np.nan,
            "fga_score": r[metric_cols["fga_score"]] if metric_cols["fga_score"] else np.nan,
        }
        rows.append(row)

out_df = pd.DataFrame(rows, columns=[
    "id",
    "speed",
    "assistive_device",
    "imbalance",
    "gait_deviation",
    "deviation_outside_walkway",
    "fga_score",
])

# ---------- save ----------
# Persist canonical GT used by evaluate.py and notebook analyses
out_df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)
