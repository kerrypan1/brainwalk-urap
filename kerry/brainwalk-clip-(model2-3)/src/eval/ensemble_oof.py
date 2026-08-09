"""§26.6: stack uncorrelated frozen-feature OOF predictors (5-fold patient split).

Combines Phase-5 mean-pool logreg, Phase-7 window-attn, and turn-aware OOF via within-fold
meta-logreg. Also reports majority vote.

Usage:
  python -m eval.ensemble_oof
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data.cv_utils import FGA_4CLASS, load_labeled_features
from eval.metrics import bootstrap_ci, classification_report_dict, constant_baselines
from train.train_turn_windows import build_turn_window_index, load as load_feats, pad_window
from train.train_windows import build_window_index, train_eval_fold
from data.turn_segment import load_segments
from train.train_item_heads import logreg_oof
from utils.paths import ARTIFACTS_DIR, OUTPUTS_DIR
from utils.seed import set_seed

CLASSES = FGA_4CLASS


def phase7_oof(X, y, folds, window=32, stride=8, seed=0):
    """Phase-7 style blind windows with 5-fold split."""
    import torch
    from argparse import Namespace

    N, T, D = X.shape
    clip_of, spans, _ = build_window_index(N, T, window, stride)
    Xw = np.stack([pad_window(X, c, a, b, window) for c, (a, b) in zip(clip_of, spans)]).astype(np.float32)
    groups = pd.read_csv(ARTIFACTS_DIR / "labeled_fw.csv")
    groups = groups[groups["fga_score"].notna() & groups["fold"].notna()].reset_index(drop=True)
    patient = groups["patient_id"].to_numpy()

    args = Namespace(head="attn", window=window, stride=stride, epochs=60, batch_size=128,
                     lr=1e-3, wd=1e-2, gamma=2.0, dropout=0.3, seed=seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    oof = np.full(N, -1, dtype=int)
    for fold in sorted(pd.unique(folds)):
        te = np.where(folds == fold)[0]
        tr = np.where(folds != fold)[0]
        preds = train_eval_fold(Xw, clip_of, y, patient, tr, te, args, device)
        for c, p in preds.items():
            oof[c] = p
    return oof


def turn_oof(X, y, folds, stems, segments, window=32, stride=8, seed=0):
    import torch
    from argparse import Namespace

    N, T, D = X.shape
    clip_of, spans, _ = build_turn_window_index(N, T, stems, segments, window, stride)
    Xw = np.stack([pad_window(X, c, a, b, window) for c, (a, b) in zip(clip_of, spans)]).astype(np.float32)
    groups = pd.read_csv(ARTIFACTS_DIR / "labeled_fw.csv")
    groups = groups[groups["fga_score"].notna() & groups["fold"].notna()].reset_index(drop=True)
    patient = groups["patient_id"].to_numpy()

    args = Namespace(head="attn", window=window, stride=stride, epochs=60, batch_size=128,
                     lr=1e-3, wd=1e-2, gamma=2.0, dropout=0.3, seed=seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    oof = np.full(N, -1, dtype=int)
    for fold in sorted(pd.unique(folds)):
        te = np.where(folds == fold)[0]
        tr = np.where(folds != fold)[0]
        preds = train_eval_fold(Xw, clip_of, y, patient, tr, te, args, device)
        for c, p in preds.items():
            oof[c] = p
    return oof


def stack_oof(y, folds, preds_dict: dict):
    """Within-fold meta-logreg on integer class predictions."""
    names = list(preds_dict.keys())
    oof = np.full(len(y), -1, dtype=int)
    for fold in sorted(pd.unique(folds)):
        te = folds == fold
        tr = ~te
        Xtr = np.column_stack([preds_dict[n][tr] for n in names]).astype(np.float32)
        Xte = np.column_stack([preds_dict[n][te] for n in names]).astype(np.float32)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
        clf.fit(sc.transform(Xtr), y[tr])
        oof[te] = clf.predict(sc.transform(Xte))
    return oof, names


def majority_oof(preds_dict: dict):
    stacks = np.column_stack(list(preds_dict.values()))
    # mode per row
    out = np.zeros(len(stacks), dtype=int)
    for i in range(len(stacks)):
        vals, cnts = np.unique(stacks[i], return_counts=True)
        out[i] = vals[cnts.argmax()]
    return out


def report(name, y, oof, n_boot=500):
    r = classification_report_dict(y, oof, classes=CLASSES)
    f1c = bootstrap_ci(y, oof, "macro_f1", classes=CLASSES, n_boot=n_boot)
    print(f"[{name:18s}] acc={r['accuracy']:.3f} macro_f1={r['macro_f1']:.3f} "
          f"MAE={r['mae']:.3f} QWK={r['qwk']:.3f}  F1 CI[{f1c['lo']:.3f},{f1c['hi']:.3f}]")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-t32", type=str,
                    default="../cache/framefeat_labeled_crop_ViT-B-32-quickgelu_openai_T32.npz")
    ap.add_argument("--features-t96", type=str,
                    default="../cache/framefeat_labeled_crop_ViT-B-32-quickgelu_openai_T96.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--skip-turn", action="store_true", help="skip turn-windows (slow)")
    ap.add_argument("--skip-phase7", action="store_true", help="skip blind phase7 windows (slow)")
    args = ap.parse_args()
    set_seed(args.seed)

    Xmean, y, folds, stems, _ = load_labeled_features(args.features_t32, "fga_score")
    # load_labeled_features mean-pools per clip -> [N, D]

    print(f"ensemble OOF  n={len(y)}  folds=5")
    print("\n--- baselines ---")
    for name, m in constant_baselines(y, CLASSES).items():
        print(f"  {name:16s} acc={m['accuracy']:.3f} macro_f1={m['macro_f1']:.3f}")

    preds = {}
    preds["phase5_mean"] = logreg_oof(Xmean, y, folds)
    report("phase5_mean", y, preds["phase5_mean"], args.n_boot)

    if not args.skip_phase7:
        X96, y96, folds96, stems96, _ = load_feats(args.features_t96)
        assert (y96 == y).all()
        preds["phase7_blind"] = phase7_oof(X96, y, folds, seed=args.seed)
        report("phase7_blind", y, preds["phase7_blind"], args.n_boot)

    if not args.skip_turn:
        segments = load_segments()
        X96, _, _, stems96, _ = load_feats(args.features_t96)
        preds["turn_windows"] = turn_oof(X96, y, folds, stems96, segments, seed=args.seed)
        report("turn_windows", y, preds["turn_windows"], args.n_boot)

    maj = majority_oof(preds)
    report("majority_vote", y, maj, args.n_boot)

    stack, used = stack_oof(y, folds, preds)
    results = {k: classification_report_dict(y, v, classes=CLASSES) for k, v in preds.items()}
    results["majority_vote"] = classification_report_dict(y, maj, classes=CLASSES)
    results["meta_logreg"] = report("meta_logreg", y, stack, args.n_boot)

    out = OUTPUTS_DIR / "phase26_ensemble_metrics.json"
    with open(out, "w") as f:
        json.dump({"folds": 5, "n": len(y), "stack_inputs": used,
                   "results": results, "constant": constant_baselines(y, CLASSES)}, f, indent=2)
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
