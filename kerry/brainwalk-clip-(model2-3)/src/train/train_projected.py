"""Phase 4: does the Zeno-guided (gait-aware) projection help FGA classification?

Apply the Phase-3 VideoProjection (trained by video<->Zeno contrastive) to the 89
labeled clips' frozen features, then re-run the same patient-grouped 5-fold
linear probe used in Phase 1. Compare three feature sets to isolate the effect
of the projection:
  - raw16      : frozen CLIP (16 frames), the fair control for Phase 1 (which used 32f)
  - projected  : VideoProjection(raw16)  -- gait-aware
  - concat     : [raw16 || projected]    -- does gait-aware add info on top of raw?

Usage:
  python -m train.train_projected \
    --features ../cache/clip_feats_ViT-B-32-quickgelu_openai_T16.npz \
    --ckpt ../outputs/phase3_contrastive.pt
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from eval.metrics import classification_report_dict, save_confusion_png
from models.zeno_encoder import VideoProjection
from utils.paths import ARTIFACTS_DIR, OUTPUTS_DIR
from utils.seed import set_seed


def load_labeled(features_path):
    z = np.load(features_path, allow_pickle=True)
    ids = list(z["ids"])
    feats = z["feats"].astype(np.float32)
    if feats.ndim == 3:  # per-frame [N,T,D] (cropped) -> mean-pool + L2
        feats = feats.mean(axis=1)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    id_to_row = {sid: i for i, sid in enumerate(ids)}

    df = pd.read_csv(ARTIFACTS_DIR / "labeled_fw.csv")
    df = df[df["fga_score"].notna() & df["fold"].notna()].copy()
    df = df[df["stem"].isin(id_to_row)].reset_index(drop=True)
    X = np.stack([feats[id_to_row[s]] for s in df["stem"]]).astype(np.float32)
    y = df["fga_score"].astype(int).to_numpy()
    folds = df["fold"].to_numpy()
    return df, X, y, folds


def apply_projection(ckpt_path, X):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    embed_dim = ckpt["args"]["embed_dim"]
    vproj = VideoProjection(in_dim=X.shape[1], embed_dim=embed_dim)
    vproj.load_state_dict(ckpt["vproj"])
    vproj.eval()
    with torch.no_grad():
        P = vproj(torch.from_numpy(X)).numpy().astype(np.float32)
    return P


def cv_probe(X, y, folds, C=1.0):
    classes = (0, 1, 2, 3)
    oof = np.full(len(y), -1, dtype=int)
    for fold in sorted(pd.unique(folds)):
        te = folds == fold
        tr = ~te
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
        clf.fit(scaler.transform(X[tr]), y[tr])
        oof[te] = clf.predict(scaler.transform(X[te]))
    assert (oof >= 0).all()
    rep = classification_report_dict(y, oof, classes=classes)
    return rep, oof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)

    df, Xraw, y, folds = load_labeled(args.features)
    Xproj = apply_projection(args.ckpt, Xraw)
    Xcat = np.concatenate([Xraw, Xproj], axis=1)

    feature_sets = {"raw16": Xraw, "projected": Xproj, "concat": Xcat}
    classes = (0, 1, 2, 3)
    n_by_class = {int(c): int((y == c).sum()) for c in classes}

    results = {}
    best_oof = None
    print(f"n={len(y)}  n_by_class={n_by_class}  proj_dim={Xproj.shape[1]}")
    for name, X in feature_sets.items():
        rep, oof = cv_probe(X, y, folds, C=args.C)
        results[name] = rep
        print(f"\n[{name}] dim={X.shape[1]}")
        print(f"  macro_f1={rep['macro_f1']:.3f}  balanced_acc={rep['balanced_accuracy']:.3f}  "
              f"acc={rep['accuracy']:.3f}  MAE={rep['mae']:.3f}  QWK={rep['qwk']:.3f}")
        print(f"  per_class_f1={rep['per_class_f1']}")
        if name == "projected":
            best_oof = oof

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "phase4_projected_metrics.json", "w") as f:
        json.dump({"n": len(y), "n_by_class": n_by_class,
                   "results": {k: v for k, v in results.items()},
                   "features": args.features, "ckpt": args.ckpt}, f, indent=2)
    if best_oof is not None:
        save_confusion_png(y, best_oof, OUTPUTS_DIR / "phase4_confusion.png",
                           classes=classes, title="Phase 4 OOF (gait-aware projection + logreg)")

    print("\n=== Phase 4 summary (out-of-fold) vs Phase 1 ref (macro-F1 0.237, QWK 0.286) ===")
    for name in feature_sets:
        r = results[name]
        print(f"{name:10s} macro_f1={r['macro_f1']:.3f}  QWK={r['qwk']:.3f}  MAE={r['mae']:.3f}")
    print(f"[written] {OUTPUTS_DIR / 'phase4_projected_metrics.json'}")


if __name__ == "__main__":
    main()
