"""Faithful reproduction training loop for arXiv:2403.13756 (gait scoring, Baseline).

Preprocessing (paper): bbox-cropped, native-fps, sliding 70-frame windows, stride 25
train / non-overlapping val. Model: frozen CLIP + Vita-CLIP video prompts + CoOp text
context. Loss: multi-class focal (alpha=0.25, gamma=2, tau=0.01) on cosine logits.
Inference: video-only, average softmax over a clip's (non-overlapping) windows -> label.
By default, evaluation uses the fixed seed-42 patient-grouped five-fold assignments
from ``artifacts/labeled_fw.csv``. FOLD-RESUMABLE (atomic per-fold OOF json).

--kapt swaps the short class labels for the KAPT clinical descriptions (Sec 2.2 knob).

--nte adds the Numerical Text Embedding branch (Sec 2.3): at each training step, in
addition to the video<->text focal loss L_k, a small batch of Zeno gait-parameter
sentences (from the 174 FGA-labeled Zeno trials, `data/labeled_zeno_join.py`) is
encoded and aligned (CE) to the same class-text prototypes; total loss = L_k + omega*L_gp.
Both --kapt and --nte can be combined (paper's best "Ours" config).

Efficiency knob vs paper: `--k_windows` caps windows sampled per clip per epoch
(paper uses all ~8-9/clip; end-to-end 70-frame backprop is heavy on 8 GB). Documented.

Usage:
  python -m train.train_vita_faithful --frames_dir ../cache/frames_labeled_native_224 \
    --folds 5 --epochs 8 --k_windows 8 --batch 1
  python -m train.train_vita_faithful --frames_dir ../cache/frames_labeled_native_224 \
    --folds 5 --epochs 8 --k_windows 8 --batch 4 --nte --kapt \
    --run_name kapt_nte_e8k8_n91_5fold_seed42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from data.zeno_features import metric_columns
from eval.metrics import (
    fold_classification_report,
    fold_constant_baselines,
    save_confusion_png,
)
from losses.focal import FocalLoss
from models.fga_prompts import baseline_labels_in_order, classnames_in_order
from models.nte import NumericTextEncoder, describe_metric, select_low_corr_combos
from models.vita_clip_faithful import VitaCLIPFaithful, normalize_frames
from utils.paths import ARTIFACTS_DIR, OUTPUTS_DIR
from utils.seed import set_seed

CLASSES = (0, 1, 2, 3)
WINDOW = 70


def load_clips(frames_dir: Path):
    df = pd.read_csv(ARTIFACTS_DIR / "labeled_fw.csv")
    df = df[df["fga_score"].notna() & df["fold"].notna()].copy()
    stems, paths, y, groups, nfr, fold_labels = [], [], [], [], [], []
    missing = []
    for _, r in df.iterrows():
        fp = frames_dir / f"{r['stem']}.npy"
        if not fp.exists():
            missing.append(r["stem"])
            continue
        arr = np.load(fp, mmap_mode="r")
        stems.append(r["stem"])
        paths.append(fp)
        y.append(int(r["fga_score"]))
        groups.append(r["patient_id"])
        nfr.append(arr.shape[0])
        fold_labels.append(r["fold"])
    return (stems, paths, np.array(y), np.array(groups), np.array(nfr),
            np.array(fold_labels), missing)


def make_splits(y, groups, fold_labels, args):
    """Return train/test clip indices for the requested patient-grouped protocol."""
    if args.split == "fixed":
        labels = sorted(pd.unique(fold_labels))
        if len(labels) != args.folds:
            raise ValueError(
                f"Expected {args.folds} fixed folds, found {len(labels)}: {labels}"
            )
        splits = []
        for label in labels:
            te = np.flatnonzero(fold_labels == label)
            tr = np.flatnonzero(fold_labels != label)
            if set(groups[tr]) & set(groups[te]):
                raise AssertionError(f"Patient leakage in fixed fold {label}")
            splits.append((tr, te))
        return splits

    sgkf = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    return list(sgkf.split(np.arange(len(y)), y, groups))


def window_starts(nf, stride):
    if nf <= WINDOW:
        return [0]
    s = list(range(0, nf - WINDOW + 1, stride))
    if s[-1] != nf - WINDOW:
        s.append(nf - WINDOW)
    return s


def load_window(path, start):
    arr = np.load(path, mmap_mode="r")
    start = min(start, max(0, arr.shape[0] - WINDOW))
    w = np.asarray(arr[start:start + WINDOW])            # [<=70,224,224,3] uint8
    if w.shape[0] < WINDOW:                              # pad short clips by repeating last
        pad = np.repeat(w[-1:], WINDOW - w.shape[0], axis=0)
        w = np.concatenate([w, pad], 0)
    return w


def to_batch(paths, items, device):
    """items: list of (clip_idx, start). -> normalized [B, 70, 3, 224, 224]."""
    arrs = [load_window(paths[c], s) for (c, s) in items]
    x = torch.from_numpy(np.stack(arrs)).to(device).float().div_(255.0)
    x = x.permute(0, 1, 4, 2, 3).contiguous()
    return normalize_frames(x)


def build_nte_globals(args):
    """Combo selection is label-independent (structural), safe to do once globally."""
    pairs = pd.read_csv(ARTIFACTS_DIR / "pairs.csv")
    mcols = metric_columns(pairs)
    combos = select_low_corr_combos(pairs, mcols, k=4, thresh=0.4,
                                    max_combos=args.n_combos, seed=args.seed)
    zeno_df = pd.read_csv(ARTIFACTS_DIR / "labeled_zeno.csv")
    return combos, zeno_df, mcols


def fold_nte_stats(zeno_df, mcols, tr_patients):
    """Per-column zero-ref (class-3/'normal' mean) + std, computed on TRAIN patients only."""
    tr = zeno_df[zeno_df["patient_id"].isin(tr_patients)]
    normal = tr[tr["fga_score"] == 3]
    stats = {}
    for c in mcols:
        col = tr[c].astype(float)
        std = float(np.nanstd(col)) if np.isfinite(np.nanstd(col)) and np.nanstd(col) > 1e-6 else 1.0
        ref_col = normal[c].astype(float) if len(normal) else col
        zero_ref = float(np.nanmean(ref_col)) if len(ref_col) and np.isfinite(np.nanmean(ref_col)) else 0.0
        median = float(np.nanmedian(col)) if len(col) and np.isfinite(np.nanmedian(col)) else 0.0
        stats[c] = (zero_ref, std, median)
    return tr, stats


def sample_gp_batch(tr_zeno, stats, combos, k, rng, device):
    """Sample k numeric-text rows -> (desc_batch, values[k,4], labels[k])."""
    rows = tr_zeno.sample(n=k, replace=True, random_state=int(rng.integers(0, 2**31)))
    desc_batch, values, labels = [], [], []
    for _, r in rows.iterrows():
        combo = combos[int(rng.integers(0, len(combos)))]
        vals = []
        for c in combo:
            raw = r[c]
            zero_ref, std, median = stats[c]
            raw = median if not np.isfinite(raw) else float(raw)
            vals.append(float(np.clip((raw - zero_ref) / std, -2.5, 2.5)))
        desc_batch.append([describe_metric(c) for c in combo])
        values.append(vals)
        labels.append(int(r["fga_score"]))
    values_t = torch.tensor(values, device=device, dtype=torch.float32)
    labels_t = torch.tensor(labels, device=device, dtype=torch.long)
    return desc_batch, values_t, labels_t


def train_eval_fold(paths, y, nfr, groups, tr_clip, te_clip, args, device, nte_globals=None):
    classnames = classnames_in_order() if args.kapt else baseline_labels_in_order()
    model = VitaCLIPFaithful(classnames, n_frames=WINDOW, n_ctx=args.n_ctx,
                             n_global=args.n_global, tau=args.tau, grad_ckpt=True).to(device)
    lossf = FocalLoss(gamma=args.gamma, alpha=args.alpha)                 # constant alpha (paper)
    trainable = list(model.trainable_parameters())

    nte = None
    if args.nte:
        combos, zeno_df, mcols = nte_globals
        tr_patients = set(groups[tr_clip])
        tr_zeno, nte_stats = fold_nte_stats(zeno_df, mcols, tr_patients)
        nte = NumericTextEncoder(model.clip_model, model.tokenizer, tau_gp=args.tau_gp).to(device)
        trainable += list(nte.parameters())

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tr_starts = {c: window_starts(nfr[c], args.train_stride) for c in tr_clip}
    rng = np.random.default_rng(args.seed)

    model.train()
    for ep in range(args.epochs):
        items = []
        for c in tr_clip:
            starts = tr_starts[c]
            pick = starts if len(starts) <= args.k_windows else \
                list(rng.choice(starts, args.k_windows, replace=False))
            items += [(c, s) for s in pick]
        rng.shuffle(items)
        ep_loss, ep_gp_loss = 0.0, 0.0
        for i in range(0, len(items), args.batch):
            batch = items[i:i + args.batch]
            x = to_batch(paths, batch, device)
            yb = torch.tensor([y[c] for (c, _) in batch], device=device).long()
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(x)
            loss = lossf(logits.float(), yb)
            if nte is not None:
                desc_batch, gp_vals, gp_labels = sample_gp_batch(
                    tr_zeno, nte_stats, combos, args.gp_batch, rng, device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    f_num = nte(desc_batch, gp_vals)
                    text_feats = model.text_features()
                loss_gp = nte.align_loss(f_num.float(), text_feats.float(), gp_labels)
                loss = loss + args.omega * loss_gp
                ep_gp_loss += loss_gp.item() * len(batch)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(batch)
        if args.verbose:
            msg = f"    epoch {ep+1}/{args.epochs} loss={ep_loss/max(1,len(items)):.4f}"
            if nte is not None:
                msg += f" gp_loss={ep_gp_loss/max(1,len(items)):.4f}"
            print(msg)

    model.eval()
    preds = {}
    with torch.no_grad():
        for c in te_clip:
            starts = window_starts(nfr[c], WINDOW)      # non-overlapping (val stride 0)
            probs = []
            for i in range(0, len(starts), args.batch):
                items = [(c, s) for s in starts[i:i + args.batch]]
                x = to_batch(paths, items, device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(x)
                probs.append(torch.softmax(logits.float(), dim=-1).cpu())
            preds[int(c)] = int(torch.cat(probs, 0).mean(0).argmax().item())
    del model
    torch.cuda.empty_cache()
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", type=str, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument(
        "--split",
        choices=["fixed", "sgkf"],
        default="fixed",
        help="fixed uses labeled_fw.csv fold assignments; sgkf reproduces legacy generated folds",
    )
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--k_windows", type=int, default=8)
    ap.add_argument("--train_stride", type=int, default=25)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--n_ctx", type=int, default=8)
    ap.add_argument("--n_global", type=int, default=8)
    ap.add_argument("--kapt", action="store_true")
    ap.add_argument("--nte", action="store_true", help="add the Numerical Text Embedding branch (Sec 2.3)")
    ap.add_argument("--omega", type=float, default=0.05, help="weight of L_gp (paper default 0.05)")
    ap.add_argument("--tau_gp", type=float, default=0.01, help="temperature for the NTE alignment CE")
    ap.add_argument("--n_combos", type=int, default=150, help="number of 4-param low-corr combos")
    ap.add_argument("--gp_batch", type=int, default=8, help="numeric-text rows sampled per training step")
    ap.add_argument("--run_name", type=str, default="", help="output dir suffix (defaults to baseline/kapt)")
    ap.add_argument("--seed", type=int, default=0,
                    help="training/NTE RNG seed; fixed folds come from the seed-42 split CSV")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--aggregate_only", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames_dir = Path(args.frames_dir)
    stems, paths, y, groups, nfr, fold_labels, missing = load_clips(frames_dir)
    if args.split == "fixed" and missing:
        raise FileNotFoundError(
            f"Fixed-fold run requires complete frame-cache coverage; missing {len(missing)}: {missing}"
        )
    N = len(stems)
    tot_win = int(sum(len(window_starts(n, args.train_stride)) for n in nfr))
    print(f"n_clips={N} frames[min/med/max]={nfr.min()}/{int(np.median(nfr))}/{nfr.max()}  "
          f"train windows(stride{args.train_stride})={tot_win}  "
          f"n_by_class={ {int(c): int((y==c).sum()) for c in CLASSES} }")
    tags = []
    if args.kapt:
        tags.append("KAPT")
    if args.nte:
        tags.append("NTE")
    print(f"tag={'+'.join(tags) if tags else 'Baseline'} epochs={args.epochs} k_windows={args.k_windows} "
          f"batch={args.batch} tau={args.tau}"
          + (f" omega={args.omega} gp_batch={args.gp_batch}" if args.nte else ""))

    nte_globals = None
    if args.nte and not args.aggregate_only:
        nte_globals = build_nte_globals(args)
        print(f"[nte] {len(nte_globals[0])} low-corr 4-param combos, "
              f"{len(nte_globals[1])} labeled Zeno-trial rows")

    default_tag = "_".join((["kapt"] if args.kapt else []) + (["nte"] if args.nte else []) or ["baseline"])
    tag = args.run_name if args.run_name else default_tag
    fold_dir = OUTPUTS_DIR / f"vita_faithful_oof_{tag}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    splits = make_splits(y, groups, fold_labels, args)
    eval_folds = np.full(N, -1, dtype=int)
    for k, (_, te_clip) in enumerate(splits):
        eval_folds[te_clip] = k
    if np.any(eval_folds < 0):
        raise AssertionError("Every clip must belong to exactly one held-out fold")

    if not args.aggregate_only:
        for k, (tr_clip, te_clip) in enumerate(splits):
            fp = fold_dir / f"fold{k}.json"
            if fp.exists():
                print(f"[fold {k}] cached, skip")
                continue
            print(f"[fold {k}] train={len(tr_clip)} test={len(te_clip)}")
            preds = train_eval_fold(paths, y, nfr, groups, tr_clip, te_clip, args, device, nte_globals)
            tmp = fold_dir / f"fold{k}.tmp.json"
            with open(tmp, "w") as f:
                json.dump({str(c): p for c, p in preds.items()}, f)
            tmp.replace(fp)
            print(f"[fold {k}] wrote {fp.name}")

    oof = np.full(N, -1, dtype=int)
    have = 0
    for k in range(args.folds):
        fp = fold_dir / f"fold{k}.json"
        if fp.exists():
            for c, p in json.load(open(fp)).items():
                oof[int(c)] = int(p)
                have += 1
    print(f"\nOOF coverage: {have}/{N}")
    if have < N:
        print("Not all folds complete; re-run to continue.")
        return

    print("\n--- constant-predictor references ---")
    baselines = fold_constant_baselines(y, eval_folds, CLASSES)
    for name, m in baselines.items():
        print(
            f"  {name:8s} acc={m['accuracy']:.3f}±{m['sd']['accuracy']:.3f} "
            f"MAE={m['mae']:.3f}±{m['sd']['mae']:.3f}"
        )

    rep = fold_classification_report(y, oof, eval_folds, classes=CLASSES)
    sd = rep["sd"]
    print(
        f"\n[Vita-faithful {tag}] "
        f"acc={rep['accuracy']:.3f}±{sd['accuracy']:.3f} "
        f"macro_f1={rep['macro_f1']:.3f}±{sd['macro_f1']:.3f} "
        f"MAE={rep['mae']:.3f}±{sd['mae']:.3f} "
        f"QWK={rep['qwk']:.3f}±{sd['qwk']:.3f}"
    )
    print(f"  pooled_confusion={rep['pooled']['confusion']}")

    mfile = OUTPUTS_DIR / f"vita_faithful_{tag}_metrics.json"
    with open(mfile, "w") as f:
        json.dump({"args": vars(args), "n": N,
                   "split_protocol": (
                       "fixed seed-42 patient folds from labeled_fw.csv"
                       if args.split == "fixed" else "generated StratifiedGroupKFold"
                   ),
                   "aggregation": "equal-weight mean and sample SD across held-out folds",
                   "mae_prediction": "native discrete class output; no continuous score cached",
                   "classification_prediction": "rounded and clipped to class range",
                   "report": rep, "baselines": baselines}, f, indent=2)
    save_confusion_png(y, oof, OUTPUTS_DIR / f"vita_faithful_{tag}_confusion.png", classes=CLASSES,
                       title=f"Vita-CLIP faithful {tag} OOF")
    print(f"[written] {mfile}")


if __name__ == "__main__":
    main()
