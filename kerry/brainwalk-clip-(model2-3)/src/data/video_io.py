"""Frame sampling from mp4 clips using OpenCV.

`sample_frames_bgr` returns raw BGR frames (so we can person-crop before CLIP
preprocess); `to_pil` converts a BGR frame to an RGB PIL image.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def sample_frames_bgr(path: str | Path, num_frames: int = 32) -> list[np.ndarray]:
    """Uniformly sample `num_frames` raw BGR uint8 frames across the whole clip."""
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if not frames:
            raise IOError(f"No frames decoded: {path}")
        idxs = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
        return [frames[i] for i in idxs]

    idxs = np.linspace(0, total - 1, num_frames).round().astype(int)
    grabbed: dict[int, np.ndarray] = {}
    for target in sorted(set(int(i) for i in idxs)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
        ok, fr = cap.read()
        if ok:
            grabbed[int(target)] = fr
    cap.release()
    if not grabbed:
        raise IOError(f"No frames decoded: {path}")
    fallback = next(iter(grabbed.values()))
    return [grabbed.get(int(i), fallback) for i in idxs]


def to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
