"""§26 Tier-1: FGA item-level heads, ordinal regression, 3-class collapse.

Uses person-cropped frozen CLIP mean-pooled features (Phase 5 cache) and the
existing **seed-42 5-fold** patient split (`fold_0`..`fold_4`). 5-fold is preferred
over 10-fold here: more train patients per fold (~36 vs ~80) and less sparse
minority-class test sets on n=89.

Targets:
  - fga_score (4-class and 3-class collapsed)
  - FGA item fields: speed, imbalance, gait_deviation, deviation_outside_walkway
    (assistive_device 1–4 also available)

Heads per target: class-balanced logreg, CORAL ordinal (K>=3), regress-and-round.

Also fits a simple **item-stack → fga_score** combiner within each CV fold
(train item heads on train, combiner on train preds, eval on test).

Usage:
  python -m data.labeled_table   # rebuild labeled_fw.csv with item columns
  python -m train.train_item_heads \\
    --features ../cache/framefeat_labeled_crop_ViT-B-32-quickgelu_openai_T32.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data.cv_utils import FGA_3CLASS, FGA_4CLASS, collapse_fga_3, load_labeled_features
from eval.metrics import (
    fold_classification_report,
    fold_constant_baselines,
    save_confusion_png,
)
from losses.ordinal import coral_oof_continuous, regress_oof
from utils.paths import ARTIFACTS_DIR, OUTPUTS_DIR
from utils.seed import set_seed

# item targets: column -> class tuple (None = infer from data)
TARGETS = {
    "fga_score": FGA_4CLASS,
    "fga_score_3class": FGA_3CLASS,
    "speed": (0, 1, 2),
    "imbalance": (0, 1, 2),
    "gait_deviation": FGA_4CLASS,
    "deviation_outside_walkway": FGA_4CLASS,
    "assistive_device": (1, 2, 3, 4),
}


def logreg_oof(X, y, folds, C=1.0):
    oof = np.full(len(y), np.nan, dtype=float)
    for fold in sorted(pd.unique(folds)):
        te = folds == fold
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
        clf.fit(sc.transform(X[tr]), y[tr])
        probabilities = clf.predict_proba(sc.transform(X[te]))
        oof[te] = probabilities @ clf.classes_.astype(float)
    return oof


def report(name, y, oof, folds, classes):
    r = fold_classification_report(y, oof, folds, classes=classes)
    sd = r["sd"]
    print(
        f"[{name:22s}] "
        f"acc={r['accuracy']:.3f}±{sd['accuracy']:.3f} "
        f"macro_f1={r['macro_f1']:.3f}±{sd['macro_f1']:.3f} "
        f"MAE={r['mae']:.3f}±{sd['mae']:.3f} "
        f"QWK={r['qwk']:.3f}±{sd['qwk']:.3f}"
    )
    return r


def eval_target(X, y, folds, col, classes):
    results = {}
    results["logreg"] = report(
        f"{col}/logreg", y, logreg_oof(X, y, folds), folds, classes
    )
    if len(classes) >= 3:
        try:
            coral = coral_oof_continuous(X, y, folds)
            results["coral"] = report(
                f"{col}/coral", y, coral, folds, classes
            )
        except ValueError as e:
            print(f"[{col}/coral] skipped: {e}")
        regression = regress_oof(X, y, folds)
        results["regress_round"] = report(
            f"{col}/regress_round", y, regression, folds, classes
        )
    return results


def item_stack_oof(df, X, folds, item_cols, fga_classes=FGA_4CLASS):
    """Per-fold: train item logregs on train, combiner on train OOF preds, test on held-out."""
    y_fga = df["fga_score"].astype(int).to_numpy()
    oof = np.full(len(y_fga), -1, dtype=int)
    valid_items = [c for c in item_cols if df[c].notna().all()]

    for fold in sorted(pd.unique(folds)):
        te = folds == fold
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        Xtr_s, Xte_s = sc.transform(X[tr]), sc.transform(X[te])

        item_tr = np.zeros((tr.sum(), len(valid_items)), dtype=np.float32)
        item_te = np.zeros((te.sum(), len(valid_items)), dtype=np.float32)
        for j, col in enumerate(valid_items):
            y_item = df[col].astype(int).to_numpy()
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
            clf.fit(Xtr_s, y_item[tr])
            item_tr[:, j] = clf.predict(Xtr_s)
            item_te[:, j] = clf.predict(Xte_s)

        comb = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
        comb.fit(item_tr, y_fga[tr])
        oof[te] = comb.predict(item_te)

    return oof, valid_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str,
                    default="../cache/framefeat_labeled_crop_ViT-B-32-quickgelu_openai_T32.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_stack", action="store_true")
    ap.add_argument("--run_name", type=str, default="",
                    help="optional output suffix, preserving prior result files")
    args = ap.parse_args()
    set_seed(args.seed)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- fga_score 4-class ---
    X, y, folds, stems, df = load_labeled_features(args.features, "fga_score")
    print(f"n={len(y)} folds=5 (seed-42 patient split)  fga_score dist="
          f"{ {int(c): int((y==c).sum()) for c in FGA_4CLASS} }")

    print("\n--- constant baselines (fga_score 4-class) ---")
    baselines4 = fold_constant_baselines(y, folds, FGA_4CLASS)
    for name, m in baselines4.items():
        print(
            f"  {name:8s} acc={m['accuracy']:.3f}±{m['sd']['accuracy']:.3f} "
            f"MAE={m['mae']:.3f}±{m['sd']['mae']:.3f}"
        )

    all_results = {}
    print("\n=== fga_score (4-class) ===")
    all_results["fga_score"] = eval_target(
        X, y, folds, "fga_score", FGA_4CLASS
    )

    # --- fga_score 3-class collapsed ---
    y3 = collapse_fga_3(y)
    print("\n=== fga_score (3-class collapsed 0+1) ===")
    print(f"  collapsed dist={ {int(c): int((y3==c).sum()) for c in FGA_3CLASS} }")
    baselines3 = fold_constant_baselines(y3, folds, FGA_3CLASS)
    all_results["fga_score_3class"] = eval_target(
        X, y3, folds, "fga_score_3class", FGA_3CLASS
    )

    # --- per-item targets ---
    item_cols = ["speed", "imbalance", "gait_deviation", "deviation_outside_walkway"]
    for col in item_cols:
        sub = df[df[col].notna()].copy()
        if len(sub) < 20:
            print(f"\n[{col}] skipped — too few labels ({len(sub)})")
            continue
        idx = sub.index.to_numpy()
        X_sub, y_sub, folds_sub = X[idx], sub[col].astype(int).to_numpy(), folds[idx]
        classes = tuple(range(int(y_sub.min()), int(y_sub.max()) + 1))
        print(f"\n=== {col} (n={len(y_sub)}, classes={classes}) ===")
        all_results[col] = eval_target(X_sub, y_sub, folds_sub, col, classes)

    # --- item stack -> fga_score ---
    if not args.skip_stack:
        stack_items = [c for c in item_cols if df[c].notna().all()]
        if len(stack_items) >= 2:
            print(f"\n=== item_stack -> fga_score (items={stack_items}) ===")
            oof_stack, used = item_stack_oof(df, X, folds, stack_items)
            all_results["item_stack_fga"] = {
                "stack": report(
                    "item_stack/logreg", y, oof_stack, folds, FGA_4CLASS
                ),
                "items_used": used,
            }
            suffix = f"_{args.run_name}" if args.run_name else ""
            save_confusion_png(y, oof_stack, OUTPUTS_DIR / f"item_stack_fga_confusion{suffix}.png",
                               classes=FGA_4CLASS, title="Item stack -> FGA OOF (5-fold)")

    suffix = f"_{args.run_name}" if args.run_name else ""
    out_path = OUTPUTS_DIR / f"phase26_item_heads_metrics{suffix}.json"
    with open(out_path, "w") as f:
        json.dump({"features": args.features, "run_name": args.run_name,
                   "folds": 5, "split": "seed-42 patient",
                   "aggregation": "equal-weight mean and sample SD across held-out folds",
                   "mae_prediction": "raw continuous",
                   "classification_prediction": "rounded and clipped to class range",
                   "n": len(y), "results": all_results,
                   "baselines_4class": baselines4,
                   "baselines_3class": baselines3}, f, indent=2)
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    main()
