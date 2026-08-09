"""Mid-walk turn detection + straight-leg segments (§26.4).

Round-trip FW clips mix outbound gait, a turn, and return gait in one file. Blind
sliding windows straddle the turn and hurt temporal models. We detect the turn as
the frame where the YOLO person centroid is farthest from its start position
(along the walkway), then expose outbound / return frame ranges for windowing.

Cached to `artifacts/turn_segments.json` (resumable). Uses subsampled YOLO
(every `stride` frames) + interpolation for speed (~5–10 s/clip vs ~50 s full).

Usage:
  python -m data.turn_segment
  python -m data.turn_segment --limit 5
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd

from data.person_crop import _interp_nan, _median_smooth
from utils.paths import ARTIFACTS_DIR, CACHE_DIR


def _load_rows():
    df = pd.read_csv(ARTIFACTS_DIR / "labeled_fw.csv")
    return df[df["fga_score"].notna()].reset_index(drop=True)


def detect_centroid_x(path: str, model, conf: float = 0.25, stride: int = 10) -> tuple[np.ndarray, int]:
    """Subsampled person centroid x trajectory; returns (cx [Nf], n_frames)."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n <= 0:
        cap.release()
        return np.array([]), 0

    sample_idx = list(range(0, n, stride))
    if sample_idx[-1] != n - 1:
        sample_idx.append(n - 1)

    xs = np.full(len(sample_idx), np.nan, dtype=np.float32)
    for j, fi in enumerate(sample_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        res = model.predict(fr, classes=[0], conf=conf, verbose=False, device=model.device)
        if len(res[0].boxes) == 0:
            continue
        xyxy = res[0].boxes.xyxy.cpu().numpy()
        c = res[0].boxes.conf.cpu().numpy()
        area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        k = (c * np.sqrt(area)).argmax()
        xs[j] = (xyxy[k, 0] + xyxy[k, 2]) * 0.5
    cap.release()

    if not np.isfinite(xs).any():
        return np.full(n, np.nan, dtype=np.float32), n

    src = np.array(sample_idx, dtype=np.float32)
    out = np.interp(np.arange(n), src, _interp_nan(np.column_stack([xs, xs]))[:, 0])
    out = _median_smooth(out.reshape(-1, 1), k=5).ravel()
    return out.astype(np.float32), n


def turn_index(cx: np.ndarray) -> int:
    """Frame farthest from start centroid — turn apex on a round-trip walk."""
    if len(cx) == 0 or not np.isfinite(cx).any():
        return 0
    cx = np.where(np.isfinite(cx), cx, np.nanmedian(cx))
    return int(np.argmax(np.abs(cx - cx[0])))


def segment_bounds(n_frames: int, turn_idx: int, margin_frac: float = 0.05, min_margin: int = 15) -> dict:
    m = max(min_margin, int(n_frames * margin_frac))
    ob_end = max(0, turn_idx - m)
    ret_start = min(n_frames, turn_idx + m)
    return {
        "n_frames": int(n_frames),
        "turn_idx": int(turn_idx),
        "margin": int(m),
        "outbound": [0, ob_end],
        "turn": [ob_end, ret_start],
        "return": [ret_start, n_frames],
    }


def native_to_feat(frame: int, n_native: int, T: int) -> int:
    if n_native <= 1:
        return 0
    return int(round(frame * (T - 1) / (n_native - 1)))


def feat_segments(seg: dict, T: int) -> dict:
    """Map native-frame segment bounds to uniform-subsample feature indices."""
    n = seg["n_frames"]
    out = {}
    for name in ("outbound", "turn", "return"):
        a, b = seg[name]
        fa, fb = native_to_feat(a, n, T), native_to_feat(b, n, T)
        if fb <= fa:
            out[name] = None
        else:
            out[name] = [fa, fb]
    out["turn_feat"] = native_to_feat(seg["turn_idx"], n, T)
    return out


def build_segments(stride: int = 10, conf: float = 0.25, limit: int = 0, force: bool = False) -> dict:
    out_path = ARTIFACTS_DIR / "turn_segments.json"
    existing = {}
    if out_path.exists() and not force:
        existing = json.load(open(out_path))

    df = _load_rows()
    todo = [r for r in df.itertuples() if force or r.stem not in existing]
    if limit > 0:
        todo = todo[:limit]
    print(f"turn_segment: total={len(df)} cached={len(existing)} processing={len(todo)}")

    if not todo:
        return existing

    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.to(0)

    for r in todo:
        cx, n = detect_centroid_x(r.path, model, conf=conf, stride=stride)
        if n == 0:
            print(f"  skip {r.stem}: no frames")
            continue
        ti = turn_index(cx)
        seg = segment_bounds(n, ti)
        seg["turn_frac"] = round(ti / n, 4)
        existing[r.stem] = seg
        # atomic write per clip
        tmp = out_path.with_suffix(".tmp.json")
        with open(tmp, "w") as f:
            json.dump(existing, f, indent=2)
        os.replace(tmp, out_path)
        print(f"  {r.stem}: n={n} turn={ti} ({seg['turn_frac']:.2f}) "
              f"out={seg['outbound']} ret={seg['return']}")

    print(f"[written] {out_path} ({len(existing)} clips)")
    return existing


def load_segments() -> dict:
    path = ARTIFACTS_DIR / "turn_segments.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run: python -m data.turn_segment")
    return json.load(open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    build_segments(stride=args.stride, conf=args.conf, limit=args.limit, force=args.force)


if __name__ == "__main__":
    main()
