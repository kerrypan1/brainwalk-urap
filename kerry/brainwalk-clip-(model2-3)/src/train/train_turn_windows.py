"""§26.4: turn-aware sliding windows on frozen per-frame features.

Builds windows only within outbound and return straight segments (skips the turn
zone), then trains a lightweight attention head with focal loss. Uses the seed-42
**5-fold** patient split (same as Phase 5 / §26 Tier-1).

Usage:
  python -m data.turn_segment
  python -m train.train_turn_windows \\
    --features ../cache/framefeat_labeled_crop_ViT-B-32-quickgelu_openai_T96.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from data.turn_segment import feat_segments, load_segments
from eval.metrics import (bootstrap_ci, classification_report_dict,
                          constant_baselines, save_confusion_png)
from losses.focal import FocalLoss
from models.temporal_head import AttnPoolHead
from train.train_windows import build_window_index, focal_alpha, make_windows, train_eval_fold
from utils.paths import ARTIFACTS_DIR, OUTPUTS_DIR
from utils.seed import set_seed

CLASSES = (0, 1, 2, 3)


def load(features_path: str):
    z = np.load(features_path, allow_pickle=True)
    ids = list(z["ids"])
    feats = z["feats"].astype(np.float32)
    id_to_row = {s: i for i, s in enumerate(ids)}
    df = pd.read_csv(ARTIFACTS_DIR / "labeled_fw.csv")
    df = df[df["fga_score"].notna() & df["fold"].notna()].copy()
    df = df[df["stem"].isin(id_to_row)].reset_index(drop=True)
    X = np.stack([feats[id_to_row[s]] for s in df["stem"]]).astype(np.float32)
    y = df["fga_score"].astype(int).to_numpy()
    folds = df["fold"].to_numpy()
    stems = df["stem"].to_numpy()
    return X, y, folds, stems, df


def straight_segment_windows(T: int, seg_feats: dict, W: int, S: int) -> list[tuple[int, int]]:
    """Windows [feat_start, feat_end) fully inside outbound or return legs."""
    wins = []
    for leg in ("outbound", "return"):
        bounds = seg_feats.get(leg)
        if bounds is None:
            continue
        a, b = bounds
        if b - a < 2:
            continue
        for (ws, we) in make_windows(b - a, W, S):
            fa, fb = a + ws, a + we
            if fb - fa >= max(W // 4, 4):  # keep short legs; pad to W later
                wins.append((fa, fb))
    if not wins:
        wins = make_windows(T, W, S)
    return wins


def pad_window(X: np.ndarray, c: int, a: int, b: int, W: int) -> np.ndarray:
    """Slice [a:b) and pad/repeat last frame to fixed length W."""
    chunk = X[c, a:b]
    if chunk.shape[0] == 0:
        chunk = X[c, max(0, a - 1) : a + 1]
    if chunk.shape[0] >= W:
        return chunk[:W]
    if chunk.shape[0] == 1:
        return np.repeat(chunk, W, axis=0)
    pad = np.repeat(chunk[-1:], W - chunk.shape[0], axis=0)
    return np.concatenate([chunk, pad], axis=0)


def build_turn_window_index(N, T, stems, segments, W, S):
    clip_of, spans = [], []
    per_clip = []
    for c, stem in enumerate(stems):
        seg = segments.get(stem)
        if seg is None:
            leg_wins = make_windows(T, W, S)
        else:
            leg_wins = straight_segment_windows(T, feat_segments(seg, T), W, S)
        per_clip.append(len(leg_wins))
        for (a, b) in leg_wins:
            clip_of.append(c)
            spans.append((a, b))
    return np.array(clip_of), spans, per_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str,
                    default="../cache/framefeat_labeled_crop_ViT-B-32-quickgelu_openai_T96.npz")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    segments = load_segments()
    X, y, folds, stems, df = load(args.features)
    N, T, D = X.shape
    groups = df["patient_id"].to_numpy()

    clip_of, spans, wins_per = build_turn_window_index(N, T, stems, segments, args.window, args.stride)
    Xw = np.stack([pad_window(X, c, a, b, args.window) for c, (a, b) in zip(clip_of, spans)]).astype(np.float32)
    print(f"turn-windows: n={N} T={T} W={args.window} S={args.stride} folds=5")
    print(f"  windows/clip: mean={np.mean(wins_per):.1f} min={min(wins_per)} max={max(wins_per)} "
          f"total={len(Xw)}")

    oof = np.full(N, -1, dtype=int)
    args.head = "attn"
    for fold in sorted(pd.unique(folds)):
        te = np.where(folds == fold)[0]
        tr = np.where(folds != fold)[0]
        preds = train_eval_fold(Xw, clip_of, y, groups, tr, te, args, device)
        for c, p in preds.items():
            oof[c] = p
    assert (oof >= 0).all()

    print("\n--- constant baselines ---")
    for name, m in constant_baselines(y, CLASSES).items():
        print(f"  {name:16s} acc={m['accuracy']:.3f} macro_f1={m['macro_f1']:.3f} "
              f"MAE={m['mae']:.3f} QWK={m['qwk']:.3f}")

    rep = classification_report_dict(y, oof, classes=CLASSES)
    f1c = bootstrap_ci(y, oof, "macro_f1", n_boot=args.n_boot)
    accc = bootstrap_ci(y, oof, "accuracy", n_boot=args.n_boot)
    print(f"\n[turn_windows attn] acc={rep['accuracy']:.3f} [{accc['lo']:.3f},{accc['hi']:.3f}]  "
          f"macro_f1={rep['macro_f1']:.3f} [{f1c['lo']:.3f},{f1c['hi']:.3f}]  "
          f"MAE={rep['mae']:.3f}  QWK={rep['qwk']:.3f}")
    print(f"  per_class_f1={ {k: round(v,2) for k,v in rep['per_class_f1'].items()} }")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS_DIR / "phase26_turn_windows_metrics.json"
    with open(out, "w") as f:
        json.dump({"args": vars(args), "n": N, "folds": 5, "windows_per_clip": wins_per,
                   "report": rep, "acc_ci": accc, "macro_f1_ci": f1c,
                   "constant": constant_baselines(y, CLASSES)}, f, indent=2)
    save_confusion_png(y, oof, OUTPUTS_DIR / "phase26_turn_windows_confusion.png",
                       classes=CLASSES, title="Turn-aware windows OOF (5-fold)")
    print(f"[written] {out}")


if __name__ == "__main__":
    main()
