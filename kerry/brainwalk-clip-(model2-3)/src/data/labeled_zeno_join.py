"""Join FGA-labeled clips to their Zeno gait-parameter trials (session-level).

For NTE (numerical text embedding, arXiv:2403.13756 Sec 2.3) we need gait
parameters *with a class label* -- the paper notes "each set of gait parameters
is linked to a video [that] is assigned a class label", i.e. the numeric-text
training signal only needs the parameters + the label of the video they came
from, not a frame-exact pairing to the current training clip.

Join key: patient_id + visit_index, where visit_index = dense rank of a
patient's FW-protocol session dates in pairs.csv (mirrors labeled_table.py's
visit numbering, which is the rank of `bath_fw` filenames per patient). All
FW trials (usually 1-2) for the matched session are kept, giving more numeric
samples per labeled clip.

Output: artifacts/labeled_zeno.csv (stem, patient_id, visit_index, fga_score,
fold, video_id, trial, + all Zeno metric columns).
"""
from __future__ import annotations

import pandas as pd

from utils.paths import ARTIFACTS_DIR


def build() -> pd.DataFrame:
    pairs = pd.read_csv(ARTIFACTS_DIR / "pairs.csv")
    labeled = pd.read_csv(ARTIFACTS_DIR / "labeled_fw.csv")
    labeled = labeled[labeled["fga_score"].notna()].copy()

    fw = pairs[pairs["protocol"] == "FW"].copy()
    fw["date"] = pd.to_datetime(fw["date"])
    fw = fw.sort_values(["patient_id", "date"])
    fw["visit_index"] = fw.groupby("patient_id")["date"].rank(method="dense").astype(int)

    merged = labeled.merge(fw, on=["patient_id", "visit_index"], how="inner", suffixes=("", "_z"))
    return merged


def main() -> None:
    df = build()
    labeled = pd.read_csv(ARTIFACTS_DIR / "labeled_fw.csv")
    n_labeled = int(labeled["fga_score"].notna().sum())
    path = ARTIFACTS_DIR / "labeled_zeno.csv"
    df.to_csv(path, index=False)
    print(
        f"[labeled_zeno] {len(df)} Zeno-trial rows from "
        f"{df['stem'].nunique()}/{n_labeled} labeled clips -> {path}"
    )
    print("  rows per fga class:")
    print(df.groupby("fga_score")["stem"].nunique().to_string())


if __name__ == "__main__":
    main()
