"""Phase 3: video<->Zeno contrastive pretraining on the paired corpus.

Frozen CLIP video features -> VideoProjection; session-mean Zeno metrics ->
ZenoEncoder MLP. Symmetric InfoNCE aligns paired video/Zeno embeddings.
Patient-independent train/val; reports video<->Zeno retrieval on val.

Usage:
  python -m train.train_contrastive --features ../cache/corpus_feats_ViT-B-32-quickgelu_openai_T16.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from data.zeno_features import build_zeno_matrix, load_pairs_with_split
from losses.contrastive import retrieval_metrics, symmetric_infonce
from models.zeno_encoder import VideoProjection, ZenoEncoder
from utils.paths import CACHE_DIR, OUTPUTS_DIR
from utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, required=True)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- load corpus features and align to pairs rows ---
    z = np.load(args.features, allow_pickle=True)
    feat_ids = list(z["ids"])
    feats = z["feats"].astype(np.float32)
    if feats.ndim == 3:  # per-frame [N,T,D] (cropped) -> mean-pool + L2
        feats = feats.mean(axis=1)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        print(f"[features] mean-pooled per-frame features -> {feats.shape}")
    id_to_row = {vid: i for i, vid in enumerate(feat_ids)}

    pairs = load_pairs_with_split()
    pairs = pairs[pairs["video_id"].isin(id_to_row)].reset_index(drop=True)
    V = np.stack([feats[id_to_row[v]] for v in pairs["video_id"]]).astype(np.float32)

    train_mask = (pairs["split"] == "train").to_numpy()
    val_mask = ~train_mask
    Z, stats = build_zeno_matrix(pairs, train_mask)

    print(f"corpus aligned: {len(pairs)}  train={train_mask.sum()}  val={val_mask.sum()}  "
          f"zeno_dim={Z.shape[1]} (={len(stats['mcols'])} metrics x2)")

    Vt = torch.from_numpy(V).to(device)
    Zt = torch.from_numpy(Z).to(device)
    tr_idx = np.where(train_mask)[0]
    va_idx = np.where(val_mask)[0]

    # session-deduped val: one video per (patient,date,protocol) so retrieval
    # targets are unique (trials in the same session share an identical Zeno vector).
    sess = (pairs["patient_id"].astype(str) + "|" + pairs["date"].astype(str)
            + "|" + pairs["protocol"].astype(str))
    seen, va_dedup = set(), []
    for i in va_idx:
        k = sess.iloc[i]
        if k not in seen:
            seen.add(k)
            va_dedup.append(i)
    va_dedup = np.array(va_dedup)
    print(f"val: {len(va_idx)} videos -> {len(va_dedup)} unique sessions (deduped retrieval)")

    vproj = VideoProjection(in_dim=V.shape[1], embed_dim=args.embed_dim).to(device)
    zenc = ZenoEncoder(in_dim=Z.shape[1], embed_dim=args.embed_dim).to(device)
    opt = torch.optim.AdamW(list(vproj.parameters()) + list(zenc.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)

    rng = np.random.default_rng(args.seed)
    best = {"recall@5": -1.0}
    for ep in range(1, args.epochs + 1):
        vproj.train(); zenc.train()
        perm = rng.permutation(tr_idx)
        losses = []
        for i in range(0, len(perm), args.batch_size):
            bidx = perm[i : i + args.batch_size]
            if len(bidx) < 8:
                continue
            vb = vproj(Vt[bidx])
            zb = zenc(Zt[bidx])
            loss = symmetric_infonce(vb, zb, args.temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        if ep % 10 == 0 or ep == args.epochs:
            vproj.eval(); zenc.eval()
            with torch.no_grad():
                vv = vproj(Vt[va_dedup]); zv = zenc(Zt[va_dedup])
                m = retrieval_metrics(vv, zv)
            print(f"ep{ep:3d} loss={np.mean(losses):.4f}  val R@1={m['recall@1']:.3f} "
                  f"R@5={m['recall@5']:.3f} R@10={m['recall@10']:.3f} medrank={m['median_rank']:.1f} "
                  f"(rand R@1={m['random_recall@1']:.4f})")
            # select on R@5 (more stable than R@1 on small val)
            if m["recall@5"] > best.get("recall@5", -1.0):
                best = m
                torch.save({"vproj": vproj.state_dict(), "zenc": zenc.state_dict(),
                            "stats_mcols": stats["mcols"], "args": vars(args)},
                           OUTPUTS_DIR / "phase3_contrastive.pt")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "phase3_retrieval_metrics.json", "w") as f:
        json.dump({"best_val_retrieval": best, "n_train": int(train_mask.sum()),
                   "n_val_videos": int(val_mask.sum()), "n_val_sessions": int(len(va_dedup))}, f, indent=2)
    print("=== Phase 3 best val retrieval ===")
    print(best)
    print(f"[written] {OUTPUTS_DIR / 'phase3_retrieval_metrics.json'}")
    print(f"[written] {OUTPUTS_DIR / 'phase3_contrastive.pt'}")


if __name__ == "__main__":
    main()
