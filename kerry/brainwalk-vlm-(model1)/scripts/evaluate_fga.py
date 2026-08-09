"""Evaluate clip-averaged Model-1 FGA predictions on the curated FW cohort.

Unlike the legacy ``evaluate.py``, this preserves the visit suffix (``_1``/``_2``),
restricts evaluation to videos currently present in ``data/bath_fw``, and reports
both raw clip-average MAE and rounded-class accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


PAIR_RE = re.compile(r"\bfga_score\b\s*:\s*([-+]?\d+(?:\.\d+)?)")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def norm_sample_id(raw: str) -> str:
    left, right = str(raw).strip().split("_", 1)
    return f"{int(left)}_{int(right)}"


def load_gt(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as fh:
        return {
            norm_sample_id(row["id"]): float(row["fga_score"])
            for row in csv.DictReader(fh)
        }


def load_predictions(root: Path) -> dict[str, float]:
    predictions = {}
    for sample_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        scores = []
        for txt in sorted(sample_dir.glob("clip_*.txt"))[:3]:
            match = PAIR_RE.search(txt.read_text(encoding="utf-8", errors="ignore"))
            if match:
                scores.append(float(match.group(1)))
        if scores:
            predictions[norm_sample_id(sample_dir.name)] = float(np.mean(scores))
    return predictions


def metrics(y_true: np.ndarray, y_raw: np.ndarray) -> dict:
    y_round = np.clip(np.rint(y_raw), 0, 3).astype(int)
    return {
        "n": int(len(y_true)),
        "accuracy": float(np.mean(y_round == y_true)),
        "mae": float(np.mean(np.abs(y_raw - y_true))),
        "rounded_prediction_counts": {
            str(k): int(v) for k, v in sorted(Counter(y_round).items())
        },
    }


def fold_metrics(
    y_true: np.ndarray, y_raw: np.ndarray, folds: np.ndarray
) -> dict:
    rows = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        row = metrics(y_true[mask], y_raw[mask])
        row["fold"] = str(fold)
        rows.append(row)
    result = {
        "n": int(len(y_true)),
        "n_folds": len(rows),
        "accuracy": float(np.mean([row["accuracy"] for row in rows])),
        "mae": float(np.mean([row["mae"] for row in rows])),
        "sd": {
            "accuracy": float(np.std([row["accuracy"] for row in rows], ddof=1)),
            "mae": float(np.std([row["mae"] for row in rows], ddof=1)),
        },
        "fold_metrics": rows,
        "pooled": metrics(y_true, y_raw),
    }
    return result


def load_fold_map(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return {
            str(row["participant_id"]).strip(): str(row["split"]).strip()
            for row in csv.DictReader(fh)
        }


def sample_to_participant(sample_id: str) -> str:
    participant = int(sample_id.split("_", 1)[0])
    return f"BW-{participant:04d}"


def fold_baselines(
    y_true: np.ndarray, folds: np.ndarray
) -> dict[str, dict]:
    predictions = {
        "mean": np.full(len(y_true), np.nan),
        "median": np.full(len(y_true), np.nan),
        "mode": np.full(len(y_true), np.nan),
    }
    fitted_values = {name: {} for name in predictions}
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        train_y = y_true[~test]
        values = {
            "mean": float(np.mean(train_y)),
            "median": float(np.median(train_y)),
            "mode": float(np.bincount(train_y.astype(int)).argmax()),
        }
        for name, value in values.items():
            predictions[name][test] = value
            fitted_values[name][str(fold)] = value
    return {
        name: {
            **fold_metrics(y_true, pred, folds),
            "fitted_values": fitted_values[name],
        }
        for name, pred in predictions.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", type=Path, required=True)
    ap.add_argument("--gt", type=Path, default=PROJECT_ROOT / "gt.csv")
    ap.add_argument("--video_dir", type=Path, default=WORKSPACE_ROOT / "data" / "bath_fw")
    ap.add_argument(
        "--split_csv",
        type=Path,
        default=WORKSPACE_ROOT / "data" / "participant_stratified_groupkfold_split_seed42.csv",
    )
    ap.add_argument("--output", type=Path)
    ap.add_argument("--expect_n", type=int, default=91)
    args = ap.parse_args()

    gt = load_gt(args.gt)
    predictions = load_predictions(args.pred_root)
    cohort = {norm_sample_id(p.stem) for p in args.video_dir.glob("*.mp4")}
    ids = sorted(cohort & gt.keys() & predictions.keys())
    missing = sorted((cohort & gt.keys()) - predictions.keys())
    if len(ids) != args.expect_n:
        raise RuntimeError(
            f"Expected {args.expect_n} scored samples, found {len(ids)}; "
            f"missing predictions: {missing}"
        )

    y_true = np.array([gt[sid] for sid in ids], dtype=int)
    y_raw = np.array([predictions[sid] for sid in ids], dtype=float)
    fold_map = load_fold_map(args.split_csv)
    missing_folds = sorted(
        {sample_to_participant(sid) for sid in ids} - fold_map.keys()
    )
    if missing_folds:
        raise RuntimeError(f"Participants missing from split CSV: {missing_folds}")
    folds = np.array([fold_map[sample_to_participant(sid)] for sid in ids])
    baselines = fold_baselines(y_true, folds)
    report = {
        "prediction_root": str(args.pred_root.resolve()),
        "cohort_video_dir": str(args.video_dir.resolve()),
        "split_csv": str(args.split_csv.resolve()),
        "aggregation": "equal-weight mean and sample SD across held-out folds",
        "mae_prediction": "raw continuous clip-average",
        "classification_prediction": "rounded and clipped to [0, 3]",
        "ids": ids,
        "model": fold_metrics(y_true, y_raw, folds),
        "baselines": baselines,
    }

    print(f"n={len(ids)}")
    for name, row in [*baselines.items(), ("model", report["model"])]:
        print(
            f"{name:8s} accuracy={row['accuracy']:.3f}±{row['sd']['accuracy']:.3f} "
            f"MAE={row['mae']:.3f}±{row['sd']['mae']:.3f}"
        )
    print(
        "rounded model predictions="
        f"{report['model']['pooled']['rounded_prediction_counts']}"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[written] {args.output}")


if __name__ == "__main__":
    main()
