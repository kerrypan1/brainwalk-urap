"""Verify how FGA-labeled visits map to raw FW videos, with caution around the
manually curated bath_fw set.

Goal: decide the labeled-clip source without re-introducing camera-angle /
corruption ambiguity that the manual bath_fw curation already resolved.

Reports:
1. For every labeled visit, whether a bath_fw clip exists and how many raw FW
   candidates exist on the visit_date (0 / 1 / multiple = angle or trial ambiguity).
2. Deep dive on the 3 visits missing a bath_fw clip.
This script is read-only.
"""
from __future__ import annotations

import pandas as pd

from common import ARTIFACTS_DIR, BATH_FW_DIR, participant_num


def main() -> None:
    fga = pd.read_csv(ARTIFACTS_DIR / "fga_labels.csv")
    vid = pd.read_csv(ARTIFACTS_DIR / "video_index.csv")
    fw = vid[vid["protocol"] == "FW"].copy()

    fw_stems = {p.stem for p in BATH_FW_DIR.glob("*.mp4")}

    def stem(r):
        return f"{participant_num(r['patient_id']):04d}_{int(r['visit_index'])}"

    fga = fga.assign(fw_stem=fga.apply(stem, axis=1))
    fga["has_bath_clip"] = fga["fw_stem"].isin(fw_stems)

    # raw FW candidates on the exact visit_date
    def candidates(r):
        c = fw[(fw["patient_id"] == r["patient_id"]) & (fw["date"] == r["visit_date"])]
        return list(c["filename"])

    fga["raw_fw_on_date"] = fga.apply(candidates, axis=1)
    fga["n_raw_fw_on_date"] = fga["raw_fw_on_date"].map(len)

    n = len(fga)
    print("=== Labeled visit -> raw FW mapping (all 92) ===")
    print(f"visit_date has raw FW video: {(fga['n_raw_fw_on_date']>0).sum()}/{n}")
    print("raw FW candidate count per visit_date:")
    print(fga["n_raw_fw_on_date"].value_counts().sort_index().to_string())
    print(f"\nof visits WITH a bath_fw clip ({fga['has_bath_clip'].sum()}):")
    sub = fga[fga["has_bath_clip"]]
    print(f"  also have >=1 raw FW on date: {(sub['n_raw_fw_on_date']>0).sum()}/{len(sub)}")
    print(f"  raw FW candidates = exactly 1 (unambiguous): {(sub['n_raw_fw_on_date']==1).sum()}")
    print(f"  raw FW candidates > 1 (angle/trial ambiguity): {(sub['n_raw_fw_on_date']>1).sum()}")

    print("\n=== Deep dive: 3 visits missing a bath_fw clip ===")
    miss = fga[~fga["has_bath_clip"]]
    for _, r in miss.iterrows():
        print(f"\n- {r['fw_stem']}  patient={r['patient_id']} visit={r['visit_index']} visit_date={r['visit_date']} fga={r['fga_score']}")
        print(f"    raw FW on visit_date: {r['raw_fw_on_date']}")
        allp = vid[vid["patient_id"] == r["patient_id"]]
        print(f"    all raw dates for patient: {sorted(allp['date'].unique())}")
        fwp = allp[allp["protocol"] == "FW"]
        print(f"    all raw FW for patient (date/file): {list(zip(fwp['date'], fwp['filename']))}")


if __name__ == "__main__":
    main()
