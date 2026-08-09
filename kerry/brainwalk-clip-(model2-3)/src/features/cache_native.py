"""Data job for the faithful reproduction (arXiv 2403.13756 preprocessing).

Paper preprocessing: "crop the original videos based on bounding boxes, and employ
a sliding window scheme (window size: 70 frames) with a stride of 25 for training
and 0 for validation" at 30 fps. To support that we must keep **every native-fps
frame**, person-cropped, so the trainer can slice contiguous 70-frame windows.

This caches, per clip, the full cropped frame stack `[Nf, 224, 224, 3]` uint8
(RGB, CLIP input size). Memory-bounded via two streaming passes (detect boxes,
then crop) so a 2000+ frame 720p clip never materializes in RAM. Atomic + resumable.

Usage:
  python -m features.cache_native
  python -m features.cache_native --limit 20
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

from data.person_crop import _interp_nan, _median_smooth, _square_boxes
from utils.paths import ARTIFACTS_DIR, CACHE_DIR, resolve_video_path


def load_rows():
    from data.labeled_table import build as build_labeled

    df = build_labeled()
    df = df[df["fga_score"].notna()].reset_index(drop=True)
    df.to_csv(ARTIFACTS_DIR / "labeled_fw.csv", index=False)
    return [
        {"key": r["stem"], "path": str(resolve_video_path(r["path"], r["stem"]))}
        for _, r in df.iterrows()
    ]


def detect_boxes(path, model, conf, chunk=64):
    """Streaming YOLO person detection over every frame. Returns [Nf,4] xyxy (NaN=miss)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    boxes = []
    buf = []

    def flush():
        if not buf:
            return
        results = model.predict(buf, classes=[0], conf=conf, verbose=False, device=model.device)
        for r in results:
            if len(r.boxes) == 0:
                boxes.append([np.nan, np.nan, np.nan, np.nan])
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            c = r.boxes.conf.cpu().numpy()
            area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            k = (c * np.sqrt(area)).argmax()
            boxes.append(xyxy[k].tolist())
        buf.clear()

    while True:
        ok, fr = cap.read()
        if not ok:
            break
        buf.append(fr)
        if len(buf) == chunk:
            flush()
    flush()
    cap.release()
    return np.array(boxes, dtype=np.float32)


def crop_stack(path, sq, size):
    """Second pass: crop each frame to its square box and resize to `size`. -> [Nf,size,size,3] uint8."""
    cap = cv2.VideoCapture(path)
    out = np.empty((len(sq), size, size, 3), dtype=np.uint8)
    i = 0
    while i < len(sq):
        ok, fr = cap.read()
        if not ok:
            break
        x1, y1, x2, y2 = sq[i]
        crop = fr[y1:y2, x1:x2]
        if crop.size == 0:
            crop = fr
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        out[i] = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        i += 1
    cap.release()
    return out[:i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--margin", type=float, default=1.3)
    ap.add_argument("--smooth_k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tag = f"frames_labeled_native_{args.size}"
    per_dir = CACHE_DIR / tag
    per_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    todo = [r for r in rows if not (per_dir / f"{r['key']}.npy").exists()]
    print(f"native frames: total={len(rows)} cached={len(rows) - len(todo)} "
          f"processing_now={min(len(todo), args.limit) if args.limit else len(todo)}")
    if args.limit > 0:
        todo = todo[: args.limit]
    if not todo:
        print("[cache_native] nothing to do")
        return

    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.to(0)
    det_rates = []
    for r in tqdm(todo, desc="native-frames"):
        raw = detect_boxes(r["path"], model, args.conf)
        det = float(np.isfinite(raw[:, 0]).mean()) if len(raw) else 0.0
        det_rates.append(det)
        if len(raw) == 0:
            print(f"skip {r['key']}: no frames")
            continue
        # first frame gives W,H for square clipping
        cap = cv2.VideoCapture(r["path"])
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        if det == 0.0:
            # no detections anywhere: full-frame square center crop fallback
            sq = _square_boxes(np.tile([0, 0, W, H], (len(raw), 1)).astype(np.float32),
                               W, H, margin=1.0)
        else:
            boxes = _median_smooth(_interp_nan(raw), args.smooth_k)
            sq = _square_boxes(boxes, W, H, args.margin)
        stack = crop_stack(r["path"], sq, args.size)
        final = per_dir / f"{r['key']}.npy"
        tmp = per_dir / f"{r['key']}.tmp.npy"
        with open(tmp, "wb") as fh:
            np.save(fh, stack)
        os.replace(tmp, final)
    if det_rates:
        print(f"[cache_native] detection rate: mean={np.mean(det_rates):.2f} min={np.min(det_rates):.2f}")
    print(f"[cache_native] cached to {per_dir}")


if __name__ == "__main__":
    main()
