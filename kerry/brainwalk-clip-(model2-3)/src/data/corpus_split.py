"""Patient-independent train/val split over the paired video<->Zeno corpus.

Deterministic hash of patient_id -> ~85/15 train/val. Labeled FGA patients are
forced into TRAIN so the small labeled set is never spent on contrastive val.

Output: artifacts/corpus_split.csv (video_id, patient_id, protocol, split)
"""
from __future__ import annotations

import hashlib

import pandas as pd

from utils.paths import ARTIFACTS_DIR


def _bucket(pid: str, val_frac: float = 0.15) -> str:
    h = int(hashlib.md5(pid.encode()).hexdigest(), 16) % 1000
    return "val" if h < val_frac * 1000 else "train"


def build() -> pd.DataFrame:
    pairs = pd.read_csv(ARTIFACTS_DIR / "pairs.csv")
    labeled = pd.read_csv(ARTIFACTS_DIR / "labeled_fw.csv")
    labeled_pids = set(labeled["patient_id"])

    split = []
    for pid in pairs["patient_id"]:
        if pid in labeled_pids:
            split.append("train")
        else:
            split.append(_bucket(pid))
    pairs = pairs.assign(split=split)
    return pairs[["video_id", "patient_id", "protocol", "split"]]


def main() -> None:
    df = build()
    path = ARTIFACTS_DIR / "corpus_split.csv"
    df.to_csv(path, index=False)
    print(f"[corpus_split] {len(df)} videos -> {path}")
    print(df["split"].value_counts().to_string())
    print("patients:", df.groupby("split")["patient_id"].nunique().to_dict())


if __name__ == "__main__":
    main()
