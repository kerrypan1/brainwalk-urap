"""Per-frame CLIP feature extraction, with person cropping by default.

Difference vs the older extractors:
  - crops each frame to the tracked person bbox before CLIP preprocess (Issue 1)
  - caches PER-FRAME features [T, D] instead of a mean-pooled vector, so a temporal
    head can model gait dynamics (Issue 2). Mean-pool baselines just average axis 0.

Use ``--no_crop`` for the matched uncropped control.

Resumable: one atomic .npy ([T, D]) per clip; re-run to continue; --aggregate_only
to build the [N, T, D] npz. --limit bounds new clips per run (GPU jobs get reaped
when the shell's blocking window closes).

Usage:
  python -m features.extract_frames --source labeled --num_frames 32
  python -m features.extract_frames --source corpus  --num_frames 32 --limit 400
  python -m features.extract_frames --source corpus  --num_frames 32 --aggregate_only
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data.video_io import sample_frames_bgr, to_pil
from utils.paths import ARTIFACTS_DIR, CACHE_DIR, DATA_DIR, resolve_video_path

RAW_VIDEO_DIR = DATA_DIR / "raw" / "bw_gait_videos"


def safe_key(s: str) -> str:
    return s.replace("|", "__").replace(":", "-").replace("/", "-")


def load_rows(source: str):
    """Return list of dicts with keys: key (cache filename), id (logical id for npz), path."""
    if source == "labeled":
        from data.labeled_table import build as build_labeled

        df = build_labeled()
        df = df[df["fga_score"].notna()].reset_index(drop=True)
        df.to_csv(ARTIFACTS_DIR / "labeled_fw.csv", index=False)
        return [
            {
                "key": r["stem"],
                "id": r["stem"],
                "path": str(resolve_video_path(r["path"], r["stem"])),
            }
            for _, r in df.iterrows()
        ]
    if source == "corpus":
        pairs = pd.read_csv(ARTIFACTS_DIR / "pairs.csv")
        return [{"key": safe_key(r["video_id"]), "id": r["video_id"],
                 "path": str(RAW_VIDEO_DIR / r["rel_path"])}
                for _, r in pairs.iterrows()]
    raise ValueError(source)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["labeled", "corpus"], required=True)
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--model", type=str, default="ViT-B-32-quickgelu")
    ap.add_argument("--pretrained", type=str, default="openai")
    ap.add_argument("--batch_frames", type=int, default=64)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--margin", type=float, default=1.3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_crop", action="store_true",
                    help="encode uniformly sampled full frames instead of person crops")
    ap.add_argument("--aggregate_only", action="store_true")
    args = ap.parse_args()

    crop_tag = "nocrop" if args.no_crop else "crop"
    tag = f"{args.source}_{crop_tag}_{args.model}_{args.pretrained}_T{args.num_frames}".replace("/", "-")
    per_dir = CACHE_DIR / f"framefeat_{tag}"
    per_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.source)

    def aggregate():
        ids, feats = [], []
        for r in rows:
            fp = per_dir / f"{r['key']}.npy"
            if fp.exists():
                arr = np.load(fp)
                ids.append(r["id"])
                feats.append(arr)
        if not feats:
            print("[frames] nothing cached yet")
            return
        feats = np.stack(feats).astype(np.float32)  # [N, T, D]
        out = CACHE_DIR / f"framefeat_{tag}.npz"
        np.savez_compressed(out, ids=np.array(ids), feats=feats)
        print(f"[frames] aggregated {feats.shape} ({len(ids)}/{len(rows)}) -> {out}")

    if args.aggregate_only:
        aggregate()
        return

    todo = [r for r in rows if not (per_dir / f"{r['key']}.npy").exists()]
    print(f"{args.source}: total={len(rows)} cached={len(rows) - len(todo)} processing_now="
          f"{min(len(todo), args.limit) if args.limit else len(todo)}")
    if args.limit > 0:
        todo = todo[: args.limit]

    if todo:
        import open_clip

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        cropper = None
        if not args.no_crop:
            from data.person_crop import PersonCropper

            cropper = PersonCropper(device=0 if device == "cuda" else "cpu",
                                    conf=args.conf, margin=args.margin)

        det_rates = []
        for r in tqdm(todo, desc=f"{args.source}-{crop_tag}"):
            try:
                frames = sample_frames_bgr(r["path"], num_frames=args.num_frames)
            except Exception as e:
                print(f"skip {r['key']}: {e}")
                continue
            if cropper is not None:
                frames, det_rate = cropper.crop_clip(frames)
                det_rates.append(det_rate)
            pil = [to_pil(fr) for fr in frames]
            x = torch.stack([preprocess(im) for im in pil]).to(device)
            with torch.no_grad():
                chunks = []
                for i in range(0, x.shape[0], args.batch_frames):
                    chunks.append(model.encode_image(x[i : i + args.batch_frames]).float())
                emb = torch.cat(chunks, dim=0)                       # [T, D]
                emb = torch.nn.functional.normalize(emb, dim=-1)
            final = per_dir / f"{r['key']}.npy"
            tmp = per_dir / f"{r['key']}.tmp.npy"
            with open(tmp, "wb") as fh:
                np.save(fh, emb.cpu().numpy().astype(np.float32))
            os.replace(tmp, final)
        if det_rates:
            print(f"[frames] detection rate this run: mean={np.mean(det_rates):.2f} "
                  f"min={np.min(det_rates):.2f}  (frac frames with a person box)")

    aggregate()


if __name__ == "__main__":
    main()
