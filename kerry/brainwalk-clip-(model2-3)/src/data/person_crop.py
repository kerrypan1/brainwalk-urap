"""Person detection and temporally tracked per-frame cropping.

Our gait clips are 1280x720 with the patient walking toward/away from the camera:
the person stays horizontally centered but their pixel footprint swings from ~4%
of frame width (far) to ~47% (near). CLIP's default Resize(224)+CenterCrop(224)
therefore feeds a tiny person on a large hallway background. The original paper
crops to the person's bounding box; we do the same, per frame, so the patient
fills the frame throughout the clip.

Strategy: detect persons per frame (YOLOv8n), pick the main track (conf*area with
temporal continuity), median-smooth the box center/size, interpolate gaps, expand
to a square box with margin, crop. Frames with no detection fall back to the
smoothed box (or center crop if the whole clip is empty).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_MODEL = None


def _get_model(weights: str, device):
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO

        _MODEL = YOLO(weights)
        _MODEL.to(device)
    return _MODEL


def _pick_boxes(frames_bgr, model, conf, batch=32):
    """Return [T,4] array of chosen person boxes (xyxy) with NaN where none found."""
    boxes = np.full((len(frames_bgr), 4), np.nan, dtype=np.float32)
    for i in range(0, len(frames_bgr), batch):
        chunk = frames_bgr[i : i + batch]
        results = model.predict(chunk, classes=[0], conf=conf, verbose=False, device=model.device)
        for j, r in enumerate(results):
            if len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            c = r.boxes.conf.cpu().numpy()
            area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            k = (c * np.sqrt(area)).argmax()  # prefer confident AND large (the near patient)
            boxes[i + j] = xyxy[k]
    return boxes


def _interp_nan(a: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs along axis 0 (per column); carry ends."""
    a = a.copy()
    n = a.shape[0]
    x = np.arange(n)
    for c in range(a.shape[1]):
        col = a[:, c]
        good = ~np.isnan(col)
        if good.sum() == 0:
            continue
        if good.sum() == 1:
            a[:, c] = col[good][0]
        else:
            a[:, c] = np.interp(x, x[good], col[good])
    return a


def _median_smooth(a: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1 or a.shape[0] < k:
        return a
    pad = k // 2
    out = np.empty_like(a)
    ap = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    for i in range(a.shape[0]):
        out[i] = np.median(ap[i : i + k], axis=0)
    return out


def _square_boxes(boxes, W, H, margin=1.3):
    """boxes [T,4] xyxy -> [T,4] square, margin-expanded, clipped to frame."""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    side = np.maximum(w, h) * margin
    side = np.clip(side, 16, max(W, H))
    x1 = cx - side / 2
    y1 = cy - side / 2
    x2 = cx + side / 2
    y2 = cy + side / 2
    # shift boxes fully inside the frame where possible
    x1 = np.clip(x1, 0, W - 1)
    y1 = np.clip(y1, 0, H - 1)
    x2 = np.clip(x2, 1, W)
    y2 = np.clip(y2, 1, H)
    return np.stack([x1, y1, x2, y2], axis=1).astype(int)


class PersonCropper:
    def __init__(self, weights="yolov8n.pt", device=0, conf=0.25, margin=1.3, smooth_k=5):
        self.model = _get_model(weights, device)
        self.conf = conf
        self.margin = margin
        self.smooth_k = smooth_k

    def crop_clip(self, frames_bgr):
        """frames_bgr: list of HxWx3 BGR uint8. Returns list of cropped BGR frames.

        Returns (cropped_frames, det_rate) where det_rate is the fraction of frames
        with a real detection (diagnostic for how well cropping worked).
        """
        if not frames_bgr:
            return frames_bgr, 0.0
        H, W = frames_bgr[0].shape[:2]
        raw = _pick_boxes(frames_bgr, self.model, self.conf)
        det_rate = float(np.isfinite(raw[:, 0]).mean())
        if det_rate == 0.0:
            return frames_bgr, 0.0  # nothing detected: leave to center-crop downstream
        boxes = _interp_nan(raw)
        boxes = _median_smooth(boxes, self.smooth_k)
        sq = _square_boxes(boxes, W, H, self.margin)
        out = []
        for fr, (x1, y1, x2, y2) in zip(frames_bgr, sq):
            crop = fr[y1:y2, x1:x2]
            if crop.size == 0:
                crop = fr
            out.append(crop)
        return out, det_rate
