"""Build a curated Zeno gait-metric table from the raw 813-column workbook.

Selects spatiotemporal gait families (mean / asymmetry / variability), drops
posturography (quiet-standing COP) columns, and keeps join keys. Assigns a
within-group trial order for (patient, date, protocol) so videos can be matched.

Output: artifacts/zeno_metrics.csv  (+ artifacts/zeno_columns.txt for review)
"""
from __future__ import annotations

import pandas as pd

from common import ZENO_XLSX, ensure_artifacts, participant_id, norm_date

# Spatiotemporal gait metric family stems (column-name prefixes, lowercased)
FAMILY_STEMS = [
    "velocitycmsec",
    "stridevelocitycmsec",
    "cadencestepsmin",
    "steplengthcm",
    "absolutesteplengthcm",
    "stridelengthcm",
    "stepwidthcm",
    "steptimesec",
    "stridetimesec",
    "stancetimesec",
    "swingtimesec",
    "singlesupport",
    "doublesupport",
    "initialdoublesupport",
    "terminaldoublesupport",
    "ambulationtimesec",
]

# Suffixes to keep per family (mean + side means, asymmetry, variability, L/R ratio)
KEEP_SUFFIXES = ("mean", "meanleft", "meanright", "asi", "cv", "cvleft", "cvright", "ratiolr")

# Any column containing these markers is posturography / non-gait -> excluded
EXCLUDE_MARKERS = ("quietstanding", "cop", "cisp")

KEY_COLS = ["bw_id", "trialdate_zeno", "gaitProtocol"]


def select_metric_columns(cols: list[str]) -> list[str]:
    selected = []
    for c in cols:
        lc = str(c).lower()
        if any(mk in lc for mk in EXCLUDE_MARKERS):
            continue
        if not any(lc.startswith(stem) for stem in FAMILY_STEMS):
            continue
        if not lc.endswith(KEEP_SUFFIXES):
            continue
        selected.append(c)
    return selected


def main() -> None:
    out_dir = ensure_artifacts()
    df = pd.read_excel(ZENO_XLSX, engine="openpyxl")

    metric_cols = select_metric_columns(list(df.columns))

    out = pd.DataFrame()
    out["patient_id"] = df["bw_id"].map(participant_id)
    out["date"] = df["trialdate_zeno"].map(norm_date)
    out["protocol"] = df["gaitProtocol"].astype(str).str.upper()
    for c in metric_cols:
        out[c] = pd.to_numeric(df[c], errors="coerce")

    # Within-group trial order (1-based) for joining to videos
    out = out.sort_values(["patient_id", "date", "protocol"]).reset_index(drop=True)
    out["trial"] = out.groupby(["patient_id", "date", "protocol"]).cumcount() + 1

    cols = ["patient_id", "date", "protocol", "trial"] + metric_cols
    out = out[cols]

    path = out_dir / "zeno_metrics.csv"
    out.to_csv(path, index=False)
    (out_dir / "zeno_columns.txt").write_text("\n".join(metric_cols), encoding="utf-8")

    print(f"[zeno_metrics] {len(out)} trials x {len(metric_cols)} metrics -> {path}")
    print(f"  selected metric columns: {len(metric_cols)} (list in zeno_columns.txt)")
    print(f"  unique patients: {out['patient_id'].nunique()}  unique dates: {out['date'].nunique()}")
    print("  trials by protocol:")
    print(out["protocol"].value_counts(dropna=False).to_string())
    miss = out[metric_cols].isna().mean().mean()
    print(f"  overall metric missingness: {miss:.1%}")
    print("  sample selected columns:", metric_cols[:12])


if __name__ == "__main__":
    main()
