#!/usr/bin/env python3
"""Standardised coefficients of the main-table model (mord.LogisticAT), per outer fold.

This is the interpretability artefact for the reported model. mord.LogisticAT is
linear with a single coefficient vector shared across all thresholds, so its SHAP
values have a closed form, ``phi_j = coef_j * (x_j - E[x_j])``. Features are
standardised inside the fold pipeline, so ``|coef|`` already *is* that ranking —
exactly, for free, and with a sign that an ``|SHAP|`` share cannot give. Running a
model-agnostic explainer here would only reproduce these numbers with sampling
noise.

The protocol mirrors the main run: same participant-grouped outer folds, same
fold-internal preprocessing, same inner grid search. Coefficients are read from
each outer-train fit, so nothing here is fitted on a test fold.

Outputs -> results/mord_coefficients_2visit_n91/
  coefficients.csv   one row per feature (share, mean/SD across folds, sign flag)
  fold_coefficients.csv  the raw per-fold matrix
  meta.json          protocol provenance
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qixing_fga.models.registry import build_models_for_task, maybe_tune  # noqa: E402
from qixing_fga.preprocessing import build_model_preprocessor  # noqa: E402
from qixing_fga.protocol import load_protocol_data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visit", default="all")
    ap.add_argument("--model", default="ordinal_logistic",
                    help="must be a linear model exposing coef_")
    ap.add_argument("--tune", default="grid", choices=["none", "grid"])
    ap.add_argument("--tune-splits", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--output-dir", default="results/mord_coefficients_2visit_n91")
    args = ap.parse_args()

    visit = args.visit if args.visit == "all" else int(args.visit)
    data = load_protocol_data(visit=visit, root=PROJECT_ROOT)
    print(f"[Protocol] {data.describe()}")

    features = list(data.feature_names)
    n_classes = int(data.y.max()) + 1
    rows = []

    for fold_idx, (train_idx, _test_idx) in enumerate(data.fold_indices):
        X_tr, y_tr = data.X.iloc[train_idx], data.y.iloc[train_idx]
        model = build_models_for_task("regression", n_classes, args.random_state)[args.model]
        pipe = Pipeline([("preprocess", build_model_preprocessor(X_tr)),
                         ("model", model)])
        estimator = maybe_tune(
            pipe, args.model, "regression", y_tr, data.groups.iloc[train_idx],
            tune=args.tune, tune_splits=args.tune_splits,
            random_state=args.random_state + fold_idx,
        )
        estimator.fit(X_tr, y_tr)
        fitted = getattr(estimator, "best_estimator_", estimator)
        fitted_model = fitted.named_steps["model"]
        if not hasattr(fitted_model, "coef_"):
            raise SystemExit(
                f"{args.model} has no coef_; this script only describes linear models."
            )
        rows.append(np.asarray(fitted_model.coef_, dtype=float).ravel())

    C = np.vstack(rows)                       # (n_folds, n_features)
    mean_abs = np.abs(C).mean(axis=0)
    signs = np.sign(C)

    table = pd.DataFrame({
        "feature": features,
        "share_pct": 100 * mean_abs / mean_abs.sum(),
        "mean_abs_coef": mean_abs,
        "mean_coef": C.mean(axis=0),
        "sd_coef": C.std(axis=0, ddof=1),
        # A coefficient whose sign flips between folds must not be read
        # directionally, however large its magnitude.
        "sign_consistent": (signs == signs[0]).all(axis=0),
    }).sort_values("share_pct", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))

    out = PROJECT_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "coefficients.csv", index=False)
    pd.DataFrame(C, columns=features,
                 index=[f"fold_{i + 1}" for i in range(len(C))]).to_csv(
        out / "fold_coefficients.csv")
    (out / "meta.json").write_text(json.dumps({
        "model": args.model,
        "visit": str(visit),
        "n_samples": int(len(data.y)),
        "n_features": len(features),
        "n_folds": int(len(C)),
        "tune": args.tune,
        "random_state": args.random_state,
        "note": (
            "Standardised coefficients from each outer-train fit. For a linear "
            "ordinal model these are the exact SHAP ranking; no explainer needed."
        ),
    }, indent=2), encoding="utf-8")

    print(table.head(15).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n[OK] Written to {out}")


if __name__ == "__main__":
    main()
