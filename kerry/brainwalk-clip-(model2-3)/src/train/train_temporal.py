"""Phase 5: FGA on person-cropped, per-frame features (Issues 1 & 2 addressed).

Compares, on the SAME patient-grouped seed-42 5-fold, out-of-fold:
  - crop_mean   : mean-pool cropped per-frame features + logreg   (isolates the CROP effect)
  - crop_tstats : [mean||std||max] over time + logreg             (cheap temporal, no training)
  - crop_attn   : learned attention-pool head (torch)             (learned temporal)
Reports constant-predictor references and bootstrap 95% CIs (n=89 is noisy).

Usage:
  python -m train.train_temporal --features ../cache/framefeat_labeled_crop_ViT-B-32-quickgelu_openai_T32.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from eval.metrics import (bootstrap_ci, classification_report_dict,
                          constant_baselines, save_confusion_png)
from models.temporal_head import AttnPoolHead, time_stats
from utils.paths import ARTIFACTS_DIR, OUTPUTS_DIR
from utils.seed import set_seed

CLASSES = (0, 1, 2, 3)


def load(features_path):
    z = np.load(features_path, allow_pickle=True)
    ids = list(z["ids"])
    feats = z["feats"].astype(np.float32)          # [N, T, D]
    id_to_row = {s: i for i, s in enumerate(ids)}
    df = pd.read_csv(ARTIFACTS_DIR / "labeled_fw.csv")
    df = df[df["fga_score"].notna() & df["fold"].notna()].copy()
    df = df[df["stem"].isin(id_to_row)].reset_index(drop=True)
    X = np.stack([feats[id_to_row[s]] for s in df["stem"]]).astype(np.float32)
    y = df["fga_score"].astype(int).to_numpy()
    folds = df["fold"].to_numpy()
    return X, y, folds


def logreg_oof(Xflat, y, folds, C=1.0):
    oof = np.full(len(y), -1, dtype=int)
    for fold in sorted(pd.unique(folds)):
        te = folds == fold
        tr = ~te
        sc = StandardScaler().fit(Xflat[tr])
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
        clf.fit(sc.transform(Xflat[tr]), y[tr])
        oof[te] = clf.predict(sc.transform(Xflat[te]))
    return oof


def attn_oof(X, y, folds, epochs=200, lr=1e-3, wd=1e-3, seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    oof = np.full(len(y), -1, dtype=int)
    D = X.shape[-1]
    for fold in sorted(pd.unique(folds)):
        set_seed(seed)
        te = folds == fold
        tr = ~te
        Xtr = torch.from_numpy(X[tr]).to(device)
        ytr = torch.from_numpy(y[tr]).long().to(device)
        Xte = torch.from_numpy(X[te]).to(device)
        cls_count = np.bincount(y[tr], minlength=len(CLASSES)).astype(np.float32)
        w = torch.from_numpy((cls_count.sum() / np.maximum(cls_count, 1))).to(device)
        w = w / w.mean()
        model = AttnPoolHead(in_dim=D, n_classes=len(CLASSES)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        lossf = nn.CrossEntropyLoss(weight=w)
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = lossf(model(Xtr), ytr)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            oof[te] = model(Xte).argmax(dim=-1).cpu().numpy()
    return oof


def report(name, y, oof, ci=True):
    r = classification_report_dict(y, oof, classes=CLASSES)
    line = (f"[{name:12s}] macro_f1={r['macro_f1']:.3f} bal_acc={r['balanced_accuracy']:.3f} "
            f"acc={r['accuracy']:.3f} MAE={r['mae']:.3f} QWK={r['qwk']:.3f}")
    if ci:
        f1c = bootstrap_ci(y, oof, "macro_f1")
        maec = bootstrap_ci(y, oof, "mae")
        line += f"  | macroF1 95%CI[{f1c['lo']:.3f},{f1c['hi']:.3f}] MAE 95%CI[{maec['lo']:.3f},{maec['hi']:.3f}]"
    print(line)
    print(f"               per_class_f1={ {k: round(v,2) for k,v in r['per_class_f1'].items()} }")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)

    X, y, folds = load(args.features)  # X: [N,T,D]
    N, T, D = X.shape
    print(f"n={N} T={T} D={D}  n_by_class={ {int(c): int((y==c).sum()) for c in CLASSES} }")

    print("\n--- constant-predictor references ---")
    for name, m in constant_baselines(y, CLASSES).items():
        print(f"  {name:16s} acc={m['accuracy']:.3f} macro_f1={m['macro_f1']:.3f} "
              f"MAE={m['mae']:.3f} QWK={m['qwk']:.3f}")

    Xmean = X.mean(axis=1)
    Xstats = np.concatenate([X.mean(1), X.std(1), X.max(1)], axis=1)

    print("\n--- models (out-of-fold) ---")
    results = {}
    results["crop_mean"] = report("crop_mean", y, logreg_oof(Xmean, y, folds))
    results["crop_tstats"] = report("crop_tstats", y, logreg_oof(Xstats, y, folds))
    attn = attn_oof(X, y, folds, seed=args.seed)
    results["crop_attn"] = report("crop_attn", y, attn)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "phase5_temporal_metrics.json", "w") as f:
        json.dump({"n": N, "T": T, "D": D,
                   "n_by_class": {int(c): int((y == c).sum()) for c in CLASSES},
                   "constant": constant_baselines(y, CLASSES),
                   "results": results, "features": args.features}, f, indent=2)
    save_confusion_png(y, attn, OUTPUTS_DIR / "phase5_confusion.png", classes=CLASSES,
                       title="Phase 5 OOF (cropped + attention temporal)")
    print(f"\n[written] {OUTPUTS_DIR / 'phase5_temporal_metrics.json'}")


if __name__ == "__main__":
    main()
