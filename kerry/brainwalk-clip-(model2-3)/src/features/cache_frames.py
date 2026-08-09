"""Cache person-cropped RAW frames (uint8, 224x224 RGB) for end-to-end training.

Phase 8 (VitaCLIP) makes the CLIP image tower part of the trainable graph, so we
can no longer feed pre-computed features: we need the actual pixels. Decoding +
YOLO cropping is the slow part, so we do it ONCE here and cache a dense contiguous
frame stack per clip. Training then slices contiguous windows and only pays a
cheap uint8->normalize cost per batch.

We store [T_dense, 224, 224, 3] uint8 per clip (person-cropped square, resized to
CLIP's 224). Atomic + resumable, mirroring features/extract_frames.py.

Usage:
  python -m features.cache_frames --num_frames 64
  python -m features.cache_frames --num_frames 64 --limit 30
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

from data.person_crop import PersonCropper
from data.video_io import sample_frames_bgr
from utils.paths import ARTIFACTS_DIR, CACHE_DIR


def load_rows():
    from data.labeled_table import build as build_labeled

    df = build_labeled()
    df = df[df["fga_score"].notna()].reset_index(drop=True)
    df.to_csv(ARTIFACTS_DIR / "labeled_fw.csv", index=False)
    return [{"key": r["stem"], "path": r["path"]} for _, r in df.iterrows()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--margin", type=float, default=1.3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tag = f"frames_labeled_crop_T{args.num_frames}_{args.size}"
    per_dir = CACHE_DIR / tag
    per_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows()

    todo = [r for r in rows if not (per_dir / f"{r['key']}.npy").exists()]
    print(f"labeled frames: total={len(rows)} cached={len(rows) - len(todo)} "
          f"processing_now={min(len(todo), args.limit) if args.limit else len(todo)}")
    if args.limit > 0:
        todo = todo[: args.limit]
    if not todo:
        print("[cache_frames] nothing to do")
        return

    cropper = PersonCropper(device=0, conf=args.conf, margin=args.margin)
    det_rates = []
    for r in tqdm(todo, desc="labeled-frames"):
        try:
            frames = sample_frames_bgr(r["path"], num_frames=args.num_frames)
        except Exception as e:
            print(f"skip {r['key']}: {e}")
            continue
        cropped, det_rate = cropper.crop_clip(frames)
        det_rates.append(det_rate)
        stack = np.empty((len(cropped), args.size, args.size, 3), dtype=np.uint8)
        for i, fr in enumerate(cropped):
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            stack[i] = cv2.resize(rgb, (args.size, args.size), interpolation=cv2.INTER_AREA)
        final = per_dir / f"{r['key']}.npy"
        tmp = per_dir / f"{r['key']}.tmp.npy"
        with open(tmp, "wb") as fh:
            np.save(fh, stack)
        os.replace(tmp, final)
    if det_rates:
        print(f"[cache_frames] detection rate this run: mean={np.mean(det_rates):.2f} "
              f"min={np.min(det_rates):.2f}")
    print(f"[cache_frames] cached to {per_dir}")


if __name__ == "__main__":
    main()
