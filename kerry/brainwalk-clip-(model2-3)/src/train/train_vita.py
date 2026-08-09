"""Phase 8: end-to-end training of VitaCLIP (prompt-tuned CLIP + temporal head).

Trains on contiguous sliding-window clips of cached person-cropped frames, with
multi-class focal loss and patient-grouped K-fold. Test prediction averages
softmax over a clip's windows. FOLD-RESUMABLE: each fold's OOF predictions are
written to outputs/phase8_oof/fold{k}.json and skipped on re-run (GPU jobs can be
reaped when the shell's blocking window closes), so re-invoking continues.

Usage:
  python -m train.train_vita --frames_dir ../cache/frames_labeled_crop_T64_224 \
    --window 16 --stride 12 --folds 10 --epochs 12 --batch 4
  python -m train.train_vita ... --aggregate_only      # build metrics from saved folds
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from eval.metrics import (bootstrap_ci, classification_report_dict,
                          constant_baselines, save_confusion_png)
from losses.focal import FocalLoss
from models.fga_prompts import classnames_in_order
from models.vita_clip import VitaCLIP, normalize_frames
from utils.paths import ARTIFACTS_DIR, OUTPUTS_DIR
from utils.seed import set_seed

CLASSES = (0, 1, 2, 3)


def load_clips(frames_dir: Path):
    """Return stems, uint8 frame stacks [Td,H,W,3], labels y, patient groups."""
    df = pd.read_csv(ARTIFACTS_DIR / "labeled_fw.csv")
    df = df[df["fga_score"].notna()].copy()
    stems, stacks, y, groups = [], [], [], []
    for _, r in df.iterrows():
        fp = frames_dir / f"{r['stem']}.npy"
        if not fp.exists():
            continue
        stems.append(r["stem"])
        stacks.append(np.load(fp))                 # [Td, H, W, 3] uint8
        y.append(int(r["fga_score"]))
        groups.append(r["patient_id"])
    return stems, stacks, np.array(y), np.array(groups)


def make_windows(n_frames, W, S):
    if n_frames <= W:
        return [(0, n_frames)]
    starts = list(range(0, n_frames - W + 1, S))
    if starts[-1] != n_frames - W:
        starts.append(n_frames - W)
    return [(s, s + W) for s in starts]


def focal_alpha(y_tr):
    cnt = np.bincount(y_tr, minlength=len(CLASSES)).astype(np.float32)
    inv = cnt.sum() / np.maximum(cnt, 1)
    return torch.tensor(inv / inv.mean(), dtype=torch.float32)


def to_batch(stacks, items, device):
    """items: list of (clip_idx, a, b). -> normalized [B, W, 3, H, W] on device."""
    arrs = [stacks[c][a:b] for (c, a, b) in items]           # each [W,H,W,3] uint8
    x = torch.from_numpy(np.stack(arrs)).to(device).float().div_(255.0)
    x = x.permute(0, 1, 4, 2, 3).contiguous()                # [B, W, 3, H, W]
    return normalize_frames(x)


def train_eval_fold(stacks, spans_per_clip, y, groups, tr_clip, te_clip, args, device):
    model = VitaCLIP(classnames_in_order(), n_ctx=args.n_ctx, n_prompt=args.n_prompt,
                     deep=not args.shallow, dropout=args.dropout, grad_ckpt=True,
                     head_type=args.head_type).to(device)
    lossf = FocalLoss(gamma=args.gamma, alpha=focal_alpha(y[tr_clip]).to(device))
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.wd)

    train_items = [(c, a, b) for c in tr_clip for (a, b) in spans_per_clip[c]]
    rng = np.random.default_rng(args.seed)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model.train()
    for ep in range(args.epochs):
        rng.shuffle(train_items)
        ep_loss = 0.0
        for i in range(0, len(train_items), args.batch):
            batch = train_items[i : i + args.batch]
            x = to_batch(stacks, batch, device)
            yb = torch.tensor([y[c] for (c, _, _) in batch], device=device).long()
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(x)
            loss = lossf(logits.float(), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(batch)
        if args.verbose:
            print(f"    epoch {ep+1}/{args.epochs} loss={ep_loss/len(train_items):.4f}")

    model.eval()
    preds = {}
    with torch.no_grad():
        for c in te_clip:
            spans = spans_per_clip[c]
            probs = []
            for i in range(0, len(spans), args.batch):
                items = [(c, a, b) for (a, b) in spans[i : i + args.batch]]
                x = to_batch(stacks, items, device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(x)
                probs.append(torch.softmax(logits.float(), dim=-1).cpu())
            prob = torch.cat(probs, 0).mean(0)
            preds[int(c)] = int(prob.argmax().item())
    del model
    torch.cuda.empty_cache()
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", type=str, required=True)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--n_ctx", type=int, default=8)
    ap.add_argument("--n_prompt", type=int, default=8)
    ap.add_argument("--head_type", choices=["linear", "text"], default="linear")
    ap.add_argument("--shallow", action="store_true", help="VPT-shallow (prompts only at input)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--aggregate_only", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames_dir = Path(args.frames_dir)
    stems, stacks, y, groups = load_clips(frames_dir)
    N = len(stems)
    Td = stacks[0].shape[0]
    print(f"n_clips={N} Td={Td} frame={stacks[0].shape[1:]}  "
          f"n_by_class={ {int(c): int((y==c).sum()) for c in CLASSES} }")
    spans_per_clip = [make_windows(s.shape[0], args.window, args.stride) for s in stacks]
    print(f"window={args.window} stride={args.stride} -> "
          f"{[len(sp) for sp in spans_per_clip[:1]][0]} windows/clip (first), "
          f"{sum(len(sp) for sp in spans_per_clip)} total")

    fold_dir = OUTPUTS_DIR / f"phase8_oof_{args.head_type}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(sgkf.split(np.arange(N), y, groups))

    if not args.aggregate_only:
        for k, (tr_clip, te_clip) in enumerate(splits):
            fp = fold_dir / f"fold{k}.json"
            if fp.exists():
                print(f"[fold {k}] cached, skip")
                continue
            print(f"[fold {k}] train_clips={len(tr_clip)} test_clips={len(te_clip)}")
            preds = train_eval_fold(stacks, spans_per_clip, y, groups, tr_clip, te_clip, args, device)
            tmp = fold_dir / f"fold{k}.tmp.json"
            with open(tmp, "w") as f:
                json.dump({str(c): p for c, p in preds.items()}, f)
            tmp.replace(fp)
            print(f"[fold {k}] done, wrote {fp.name}")

    oof = np.full(N, -1, dtype=int)
    have = 0
    for k in range(args.folds):
        fp = fold_dir / f"fold{k}.json"
        if not fp.exists():
            continue
        with open(fp) as f:
            for c, p in json.load(f).items():
                oof[int(c)] = int(p)
                have += 1
    print(f"\nOOF coverage: {have}/{N}")
    if have < N:
        print("Not all folds complete yet; re-run to continue. Skipping final metrics.")
        return

    print("\n--- constant-predictor references ---")
    for name, m in constant_baselines(y, CLASSES).items():
        print(f"  {name:16s} acc={m['accuracy']:.3f} macro_f1={m['macro_f1']:.3f} "
              f"MAE={m['mae']:.3f} QWK={m['qwk']:.3f}")

    rep = classification_report_dict(y, oof, classes=CLASSES)
    f1c = bootstrap_ci(y, oof, "macro_f1")
    accc = bootstrap_ci(y, oof, "accuracy")
    print(f"\n[Phase8 VitaCLIP] acc={rep['accuracy']:.3f} [{accc['lo']:.3f},{accc['hi']:.3f}]  "
          f"macro_f1={rep['macro_f1']:.3f} [{f1c['lo']:.3f},{f1c['hi']:.3f}]  "
          f"bal_acc={rep['balanced_accuracy']:.3f}  MAE={rep['mae']:.3f}  QWK={rep['qwk']:.3f}")
    print(f"  per_class_f1={ {k: round(v,2) for k,v in rep['per_class_f1'].items()} }")
    print(f"  confusion={np.array(rep['confusion']).tolist()}")

    mfile = OUTPUTS_DIR / f"phase8_vita_{args.head_type}_metrics.json"
    with open(mfile, "w") as f:
        json.dump({"args": vars(args), "n": N, "report": rep, "acc_ci": accc,
                   "macro_f1_ci": f1c, "constant": constant_baselines(y, CLASSES)}, f, indent=2)
    save_confusion_png(y, oof, OUTPUTS_DIR / f"phase8_{args.head_type}_confusion.png", classes=CLASSES,
                       title=f"Phase 8 OOF (VitaCLIP {args.head_type}, VPT+temporal+focal)")
    print(f"[written] {mfile}")


if __name__ == "__main__":
    main()
