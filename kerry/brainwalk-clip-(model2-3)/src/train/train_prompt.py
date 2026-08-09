"""Phase 2: CoOp-style FGA prompt learning on frozen CLIP features.

Both CLIP towers are frozen; only the learnable context vectors train. Classify
cached (frozen) video features by cosine similarity to learned FGA text prompts.
Patient-grouped 5-fold CV with out-of-fold aggregation, mirroring Phase 1.

Also reports a zero-shot CLIP reference (hand-written prompts, no training).

Usage:
  python -m train.train_prompt --features ../cache/clip_feats_ViT-B-32-quickgelu_openai_T32.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from eval.metrics import classification_report_dict, save_confusion_png
from losses.focal import FocalLoss
from models.fga_prompts import classnames_in_order, zeroshot_in_order
from models.prompt_learner import CoOpFGA
from utils.paths import ARTIFACTS_DIR, OUTPUTS_DIR
from utils.seed import set_seed

CLASSES = (0, 1, 2, 3)


def load_data(features_path):
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


def zeroshot_predict(clip_model, tokenizer, X, device):
    prompts = zeroshot_in_order()
    toks = torch.cat([tokenizer(p) for p in prompts]).to(device)
    with torch.no_grad():
        tf = clip_model.encode_text(toks).float()
        tf = F.normalize(tf, dim=-1)
        xf = F.normalize(torch.from_numpy(X).to(device), dim=-1)
        logits = xf @ tf.t()
        return logits.argmax(dim=-1).cpu().numpy()


def train_fold(clip_model, tokenizer, Xtr, ytr, Xte, args, device):
    model = CoOpFGA(clip_model, tokenizer, classnames_in_order(),
                    n_ctx=args.n_ctx, class_specific=args.class_specific).to(device)
    # only prompt context is trainable
    params = [model.prompt_learner.ctx]
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    counts = np.bincount(ytr, minlength=len(CLASSES)).astype(np.float32)
    w = np.where(counts > 0, counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), 0.0)
    class_w = torch.tensor(w, dtype=torch.float32, device=device)
    if args.loss == "focal":
        alpha = class_w / class_w.mean()
        lossf = FocalLoss(gamma=args.gamma, alpha=alpha)
    else:
        lossf = lambda lo, ta: F.cross_entropy(lo, ta, weight=class_w)

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)

    model.train()
    for _ in range(args.epochs):
        opt.zero_grad()
        logits = model(Xtr_t)
        loss = lossf(logits, ytr_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte).to(device)).argmax(dim=-1).cpu().numpy()
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, required=True)
    ap.add_argument("--model", type=str, default="ViT-B-32-quickgelu")
    ap.add_argument("--pretrained", type=str, default="openai")
    ap.add_argument("--n_ctx", type=int, default=8)
    ap.add_argument("--class_specific", action="store_true")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--loss", choices=["focal", "ce"], default="focal")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)

    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(args.model)

    df, X, y, folds = load_data(args.features)
    fold_names = sorted(pd.unique(folds))

    # --- zero-shot reference ---
    zs_pred = zeroshot_predict(clip_model, tokenizer, X, device)
    zs = classification_report_dict(y, zs_pred, classes=CLASSES)

    # --- CoOp OOF ---
    oof = np.full(len(y), -1, dtype=int)
    for fold in fold_names:
        te = folds == fold
        tr = ~te
        oof[te] = train_fold(clip_model, tokenizer, X[tr], y[tr], X[te], args, device)
    assert (oof >= 0).all()

    rep = classification_report_dict(y, oof, classes=CLASSES)
    rep["model"] = f"coop_nctx{args.n_ctx}_{'csc' if args.class_specific else 'unified'}"
    rep["features"] = args.features
    rep["n_by_class"] = {int(c): int((y == c).sum()) for c in CLASSES}
    rep["zeroshot"] = {k: zs[k] for k in ["macro_f1", "balanced_accuracy", "accuracy", "mae", "qwk"]}

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "phase2_prompt_metrics.json", "w") as f:
        json.dump(rep, f, indent=2)
    save_confusion_png(y, oof, OUTPUTS_DIR / "phase2_confusion.png",
                       classes=CLASSES, title="Phase 2 OOF (CoOp FGA prompts)")

    print("=== Phase 2: CoOp FGA prompt learning (out-of-fold) ===")
    print(f"n={rep['n']}  n_by_class={rep['n_by_class']}  config={rep['model']}")
    print(f"macro_f1={rep['macro_f1']:.3f}  balanced_acc={rep['balanced_accuracy']:.3f}  "
          f"acc={rep['accuracy']:.3f}  MAE={rep['mae']:.3f}  QWK={rep['qwk']:.3f}")
    print(f"per_class_f1={rep['per_class_f1']}")
    print(f"confusion (rows=true 0..3):\n{np.array(rep['confusion'])}")
    print(f"[ref] zero-shot CLIP: macro_f1={zs['macro_f1']:.3f}  balanced_acc={zs['balanced_accuracy']:.3f}  "
          f"acc={zs['accuracy']:.3f}  MAE={zs['mae']:.3f}  QWK={zs['qwk']:.3f}")
    print("[ref] Phase 1 logreg (quickgelu): macro_f1=0.237  balanced_acc=0.238  MAE=0.865  QWK=0.286")
    print(f"[written] {OUTPUTS_DIR / 'phase2_prompt_metrics.json'}")


if __name__ == "__main__":
    main()
