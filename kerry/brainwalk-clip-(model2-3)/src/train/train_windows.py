"""Phase 7: sliding-window clips + focal loss (the paper's data/loss regime).

The paper turns ~100 videos into ~900 sliding-window clips per fold and trains a
temporal video model with multi-class focal loss, averaging window predictions at
test time. We mirror that on cached dense per-frame features:
  - dense features [N, T_dense, D] (person-cropped) -> contiguous windows of length W
    (stride S) => many training clips/video (fixes the n=89 overfit).
  - temporal head (attention-pool or small Transformer) trained with focal loss.
  - patient-grouped K-fold; at test, AVERAGE softmax over a clip's windows -> label.

Usage:
  python -m train.train_windows \
    --features ../cache/framefeat_labeled_crop_ViT-B-32-quickgelu_openai_T96.npz \
    --head transformer --window 32 --stride 8 --folds 10
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold

from eval.metrics import (bootstrap_ci, classification_report_dict,
                          constant_baselines, save_confusion_png)
from losses.focal import FocalLoss
from models.temporal_head import AttnPoolHead, TemporalTransformer
from utils.paths import ARTIFACTS_DIR, OUTPUTS_DIR
from utils.seed import set_seed

CLASSES = (0, 1, 2, 3)


def load(features_path):
    z = np.load(features_path, allow_pickle=True)
    ids = list(z["ids"])
    feats = z["feats"].astype(np.float32)          # [N, T, D]
    id_to_row = {s: i for i, s in enumerate(ids)}
    df = pd.read_csv(ARTIFACTS_DIR / "labeled_fw.csv")
    df = df[df["fga_score"].notna()].copy()
    df = df[df["stem"].isin(id_to_row)].reset_index(drop=True)
    X = np.stack([feats[id_to_row[s]] for s in df["stem"]]).astype(np.float32)
    y = df["fga_score"].astype(int).to_numpy()
    groups = df["patient_id"].to_numpy()
    return X, y, groups


def make_windows(n_frames, W, S):
    if n_frames <= W:
        return [(0, n_frames)]
    starts = list(range(0, n_frames - W + 1, S))
    if starts[-1] != n_frames - W:
        starts.append(n_frames - W)
    return [(s, s + W) for s in starts]


def build_window_index(N, T, W, S):
    """Return arrays clip_of_window[M], and slices list for each window."""
    wins = make_windows(T, W, S)
    clip_of = []
    spans = []
    for c in range(N):
        for (a, b) in wins:
            clip_of.append(c)
            spans.append((a, b))
    return np.array(clip_of), spans, len(wins)


def focal_alpha(y_tr):
    cnt = np.bincount(y_tr, minlength=len(CLASSES)).astype(np.float32)
    inv = cnt.sum() / np.maximum(cnt, 1)
    inv = inv / inv.mean()                         # normalized inverse-freq, mean 1
    return torch.tensor(inv, dtype=torch.float32)


def train_eval_fold(Xw, clip_of, y, groups, tr_clip, te_clip, args, device):
    """Xw: [M, W, D] all windows. Returns per-test-clip predicted labels dict."""
    tr_w = np.isin(clip_of, tr_clip)
    Xtr = torch.from_numpy(Xw[tr_w]).to(device)
    ytr = torch.from_numpy(y[clip_of[tr_w]]).long().to(device)

    D = Xw.shape[-1]
    if args.head == "transformer":
        model = TemporalTransformer(in_dim=D, n_classes=len(CLASSES), dropout=args.dropout,
                                    max_len=args.window).to(device)
    else:
        model = AttnPoolHead(in_dim=D, n_classes=len(CLASSES), dropout=args.dropout).to(device)

    lossf = FocalLoss(gamma=args.gamma, alpha=focal_alpha(y[tr_clip]))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n = Xtr.shape[0]
    rng = np.random.default_rng(args.seed)
    model.train()
    for _ in range(args.epochs):
        perm = rng.permutation(n)
        for i in range(0, n, args.batch_size):
            b = perm[i : i + args.batch_size]
            opt.zero_grad()
            loss = lossf(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()

    model.eval()
    preds = {}
    with torch.no_grad():
        for c in te_clip:
            w = clip_of == c
            logits = model(torch.from_numpy(Xw[w]).to(device))
            prob = torch.softmax(logits, dim=-1).mean(dim=0)   # average over windows
            preds[c] = int(prob.argmax().item())
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, required=True)
    ap.add_argument("--head", choices=["attn", "transformer"], default="transformer")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, y, groups = load(args.features)     # X: [N, T, D]
    N, T, D = X.shape
    clip_of, spans, wins_per_clip = build_window_index(N, T, args.window, args.stride)
    Xw = np.stack([X[c, a:b] for c, (a, b) in zip(clip_of, spans)]).astype(np.float32)
    print(f"n_clips={N} T={T} D={D}  window={args.window} stride={args.stride} "
          f"-> {wins_per_clip} windows/clip, {Xw.shape[0]} total windows")
    print(f"n_by_class={ {int(c): int((y==c).sum()) for c in CLASSES} }  folds={args.folds} head={args.head}")

    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.full(N, -1, dtype=int)
    for tr_clip, te_clip in sgkf.split(np.arange(N), y, groups):
        preds = train_eval_fold(Xw, clip_of, y, groups, tr_clip, te_clip, args, device)
        for c, p in preds.items():
            oof[c] = p
    assert (oof >= 0).all()

    print("\n--- constant-predictor references ---")
    for name, m in constant_baselines(y, CLASSES).items():
        print(f"  {name:16s} acc={m['accuracy']:.3f} macro_f1={m['macro_f1']:.3f} "
              f"MAE={m['mae']:.3f} QWK={m['qwk']:.3f}")

    rep = classification_report_dict(y, oof, classes=CLASSES)
    f1c = bootstrap_ci(y, oof, "macro_f1")
    accc = bootstrap_ci(y, oof, "accuracy")
    print(f"\n[Phase7 {args.head}] acc={rep['accuracy']:.3f} [{accc['lo']:.3f},{accc['hi']:.3f}]  "
          f"macro_f1={rep['macro_f1']:.3f} [{f1c['lo']:.3f},{f1c['hi']:.3f}]  "
          f"bal_acc={rep['balanced_accuracy']:.3f}  MAE={rep['mae']:.3f}  QWK={rep['qwk']:.3f}")
    print(f"  per_class_f1={ {k: round(v,2) for k,v in rep['per_class_f1'].items()} }")
    print(f"  confusion={np.array(rep['confusion']).tolist()}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "phase7_windows_metrics.json", "w") as f:
        json.dump({"args": vars(args), "n": N, "windows_per_clip": wins_per_clip,
                   "report": rep, "acc_ci": accc, "macro_f1_ci": f1c,
                   "constant": constant_baselines(y, CLASSES)}, f, indent=2)
    save_confusion_png(y, oof, OUTPUTS_DIR / "phase7_confusion.png", classes=CLASSES,
                       title=f"Phase 7 OOF ({args.head}, windows+focal)")
    print(f"[written] {OUTPUTS_DIR / 'phase7_windows_metrics.json'}")


if __name__ == "__main__":
    main()
