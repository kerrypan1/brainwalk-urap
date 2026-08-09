"""§26.5 reference: Zeno kinematics -> FGA ceiling (5-fold patient split).

Another student got okay Zeno->FGA results. This script quantifies the upper bound
on the labeled clips using true mat metrics — not a product path (VLM cannot predict
Zeno reliably), but tells us whether the FGA label is learnable from gait.

Aggregates dual FW trials to one row per clip (mean of metrics). Uses the same
seed-42 5-fold as video models.

Usage:
  python -m eval.zeno_to_fga
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

from data.cv_utils import FGA_3CLASS, FGA_4CLASS, collapse_fga_3
from data.zeno_features import metric_columns
from eval.metrics import classification_report_dict, constant_baselines
from utils.paths import ARTIFACTS_DIR, OUTPUTS_DIR


def clip_level_zeno():
    """One row per labeled clip: mean Zeno metrics across FW trials in session."""
    zeno = pd.read_csv(ARTIFACTS_DIR / "labeled_zeno.csv")
    labeled = pd.read_csv(ARTIFACTS_DIR / "labeled_fw.csv")
    labeled = labeled[labeled["fga_score"].notna() & labeled["fold"].notna()].copy()

    mcols = [c for c in metric_columns(zeno) if c != "fga_score"]
    per_clip = zeno.groupby("stem", as_index=False)[mcols].mean()
    return labeled.merge(per_clip, on="stem", how="inner")


def fold_design(raw, tr, te):
    """Impute and normalize from training rows only, then append missingness."""
    train = raw[tr]
    test = raw[te]
    median = np.nanmedian(train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    train_filled = np.where(np.isnan(train), median, train)
    test_filled = np.where(np.isnan(test), median, test)
    mean = train_filled.mean(axis=0)
    std = train_filled.std(axis=0)
    std = np.where((std == 0) | ~np.isfinite(std), 1.0, std)
    xtr = np.clip((train_filled - mean) / std, -5, 5)
    xte = np.clip((test_filled - mean) / std, -5, 5)
    xtr = np.concatenate([xtr, np.isnan(train).astype(np.float32)], axis=1)
    xte = np.concatenate([xte, np.isnan(test).astype(np.float32)], axis=1)
    return xtr.astype(np.float32), xte.astype(np.float32)


def logreg_oof(raw, y, folds):
    oof = np.full(len(y), -1, dtype=int)
    for fold in sorted(pd.unique(folds)):
        te = folds == fold
        tr = ~te
        xtr, xte = fold_design(raw, tr, te)
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
        clf.fit(xtr, y[tr])
        oof[te] = clf.predict(xte)
    return oof


def coral_oof(raw, y, folds):
    y = np.asarray(y, dtype=int)
    n_classes = int(y.max()) + 1
    oof = np.full(len(y), -1, dtype=int)
    for fold in sorted(pd.unique(folds)):
        te = folds == fold
        tr = ~te
        xtr, xte = fold_design(raw, tr, te)
        probas = []
        for threshold in range(n_classes - 1):
            y_bin = (y[tr] > threshold).astype(int)
            if y_bin.min() == y_bin.max():
                probas.append(np.full(te.sum(), float(y_bin[0])))
                continue
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
            clf.fit(xtr, y_bin)
            probas.append(clf.predict_proba(xte)[:, 1])
        oof[te] = (np.stack(probas, axis=1) > 0.5).sum(axis=1)
    return oof


def regress_round_oof(raw, y, folds):
    y = np.asarray(y, dtype=int)
    oof = np.full(len(y), -1, dtype=int)
    for fold in sorted(pd.unique(folds)):
        te = folds == fold
        tr = ~te
        xtr, xte = fold_design(raw, tr, te)
        reg = Ridge(alpha=1.0)
        reg.fit(xtr, y[tr])
        oof[te] = np.clip(
            np.rint(reg.predict(xte)), int(y.min()), int(y.max())
        ).astype(int)
    return oof


def main():
    df = clip_level_zeno()
    y = df["fga_score"].astype(int).to_numpy()
    folds = df["fold"].to_numpy()

    pairs = pd.read_csv(ARTIFACTS_DIR / "pairs.csv")
    mcols = metric_columns(pairs)
    raw = df[mcols].to_numpy(dtype=np.float64)

    print(f"Zeno->FGA ceiling diagnostic  n_clips={len(df)}  folds=5")
    print(f"  fga dist={ {int(c): int((y==c).sum()) for c in FGA_4CLASS} }")

    results = {}
    for name, yy, classes in [
        ("fga_4class", y, FGA_4CLASS),
        ("fga_3class", collapse_fga_3(y), FGA_3CLASS),
    ]:
        print(f"\n--- {name} ---")
        for bname, m in constant_baselines(yy, classes).items():
            print(f"  baseline {bname:16s} acc={m['accuracy']:.3f} MAE={m['mae']:.3f} QWK={m['qwk']:.3f}")
        oof_lr = logreg_oof(raw, yy, folds)
        r = classification_report_dict(yy, oof_lr, classes=classes)
        print(f"  logreg           acc={r['accuracy']:.3f} macro_f1={r['macro_f1']:.3f} "
              f"MAE={r['mae']:.3f} QWK={r['qwk']:.3f}")
        results[name] = {"logreg": r}
        if len(classes) >= 3:
            oof_c = coral_oof(raw, yy, folds)
            rc = classification_report_dict(yy, oof_c, classes=classes)
            print(f"  coral            acc={rc['accuracy']:.3f} macro_f1={rc['macro_f1']:.3f} "
                  f"MAE={rc['mae']:.3f} QWK={rc['qwk']:.3f}")
            results[name]["coral"] = rc
            oof_rr = regress_round_oof(raw, yy, folds)
            rr = classification_report_dict(yy, oof_rr, classes=classes)
            print(f"  regress_round    acc={rr['accuracy']:.3f} MAE={rr['mae']:.3f} QWK={rr['qwk']:.3f}")
            results[name]["regress_round"] = rr

    out = OUTPUTS_DIR / "phase26_zeno_to_fga_ceiling_n91_5fold_seed42.json"
    with open(out, "w") as f:
        json.dump({
            "n_clips": len(df),
            "folds": 5,
            "split": "fixed seed-42 patient folds",
            "preprocessing": "fold-safe train-only median imputation and normalization",
            "results": results,
        }, f, indent=2)
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
