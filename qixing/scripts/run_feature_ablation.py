#!/usr/bin/env python3
"""Feature-group ablation (Project_discription.txt S4).

Two complementary views, both on the full protocol (teammate folds, inner grid
search, per-fold bootstrap CI):

  leave-one-group-out : drop one group, keep the rest -> "is this group needed
                        given everything else?" (answers redundancy)
  group-only          : keep one group, drop the rest -> "how far does this
                        group get on its own?" (answers sufficiency)

Both are reported because they disagree whenever groups carry overlapping
information, which is the normal case for gait measures — a group can be
individually predictive yet removable without loss.

Outputs -> results/ablation_2visit_n91/ (or --output-dir).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qixing_fga.evaluation.bootstrap import per_fold_bootstrap_mae_rmse  # noqa: E402
from qixing_fga.evaluation.metrics import _compute_metrics  # noqa: E402
from qixing_fga.models.registry import build_models_for_task, maybe_tune  # noqa: E402
from qixing_fga.preprocessing import build_model_preprocessor  # noqa: E402
from qixing_fga.protocol import load_protocol_data  # noqa: E402
from utils.feature_groups import (  # noqa: E402
    GROUP_LABELS,
    groups_present,
    validate_groups,
)


def evaluate_variant(
    data,
    cols: list[str],
    variant: str,
    model_name: str,
    random_state: int,
    tune: str,
    tune_splits: int,
    ci_alpha: float,
) -> tuple[list[dict], pd.DataFrame]:
    """Full nested-CV evaluation of one feature subset."""
    n_classes = int(data.y.max()) + 1
    rows: list[dict] = []
    preds: list[pd.DataFrame] = []

    for fold_idx, (train_idx, test_idx) in enumerate(data.fold_indices):
        X_tr, X_te = data.X.iloc[train_idx][cols], data.X.iloc[test_idx][cols]
        y_tr, y_te = data.y.iloc[train_idx], data.y.iloc[test_idx]

        model = build_models_for_task("regression", n_classes, random_state)[model_name]
        pipe = Pipeline(
            [("preprocess", build_model_preprocessor(X_tr)), ("model", model)]
        )
        estimator = maybe_tune(
            pipe,
            model_name,
            "regression",
            y_tr,
            data.groups.iloc[train_idx],
            tune=tune,
            tune_splits=tune_splits,
            random_state=random_state + fold_idx,
        )
        estimator.fit(X_tr, y_tr)
        y_hat = estimator.predict(X_te)

        m = _compute_metrics(y_te, y_hat, task="regression", n_classes=n_classes)
        boot = per_fold_bootstrap_mae_rmse(
            y_val=y_te.to_numpy(),
            y_pred=np.asarray(y_hat, dtype=float),
            val_idx=np.asarray(test_idx, dtype=int),
            groups=data.groups.to_numpy(),
            boot_matrix=data.bootstrap_by_fold[f"fold_{fold_idx}"],
            ci_alpha=ci_alpha,
        )
        rows.append(
            {
                "variant": variant,
                "model": model_name,
                "fold": fold_idx + 1,
                "n_features": len(cols),
                **m,
                "train_mae": float(np.mean(np.abs(estimator.predict(X_tr) - y_tr))),
                "mae_ci_lower": boot["mae_ci_lower"],
                "mae_ci_upper": boot["mae_ci_upper"],
            }
        )
        preds.append(
            pd.DataFrame(
                {
                    "variant": variant,
                    "model": model_name,
                    "fold": fold_idx + 1,
                    "sample_id": data.ids.iloc[test_idx].to_numpy(),
                    "participant_id": data.groups.iloc[test_idx].to_numpy(),
                    "y_true": y_te.to_numpy(dtype=float),
                    "y_pred": np.asarray(y_hat, dtype=float),
                }
            )
        )
    return rows, pd.concat(preds, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visit", default="all")
    ap.add_argument("--model", default="ordinal_logistic")
    ap.add_argument("--tune", default="grid", choices=["none", "grid"])
    ap.add_argument("--tune-splits", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--ci-alpha", type=float, default=0.05)
    ap.add_argument("--output-dir", default="results/ablation_2visit_n91")
    args = ap.parse_args()

    validate_groups()
    visit = args.visit if args.visit == "all" else int(args.visit)
    data = load_protocol_data(visit=visit, root=PROJECT_ROOT)
    print(f"[Protocol] {data.describe()}")

    groups = groups_present(data.feature_names)
    print(f"[Groups] {len(groups)}: " + ", ".join(f"{k}({len(v)})" for k, v in groups.items()))

    out = Path(args.output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # Zero-variance columns carry no information, so any group made only of them
    # is indistinguishable from an intercept-only model and "dropping" it cannot
    # change a single prediction. Recording them keeps that from being read as a
    # substantive finding about the model.
    constant = [c for c in data.feature_names if data.X[c].nunique(dropna=True) <= 1]
    if constant:
        const_df = pd.DataFrame(
            {
                "feature": constant,
                "value": [data.X[c].iloc[0] for c in constant],
                "group": [
                    next((g for g, f in groups.items() if c in f), "unassigned")
                    for c in constant
                ],
            }
        )
        const_df.to_csv(out / "constant_features.csv", index=False)
        print(f"[Note] {len(constant)} zero-variance feature(s) in this view: {constant}")
        for g, feats in groups.items():
            if feats and all(f in constant for f in feats):
                print(
                    f"[Note] group '{g}' is entirely constant — its drop/only results are "
                    "degenerate by construction, not evidence about the model."
                )

    variants: list[tuple[str, list[str]]] = [("full", list(data.feature_names))]
    for name, feats in groups.items():
        rest = [c for c in data.feature_names if c not in feats]
        if rest:
            variants.append((f"drop::{name}", rest))
    for name, feats in groups.items():
        variants.append((f"only::{name}", list(feats)))

    all_rows: list[dict] = []
    all_preds: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    for i, (variant, cols) in enumerate(variants, 1):
        print(f"  [{i}/{len(variants)}] {variant} ({len(cols)} features)...", flush=True)
        rows, preds = evaluate_variant(
            data, cols, variant, args.model, args.random_state,
            args.tune, args.tune_splits, args.ci_alpha,
        )
        all_rows += rows
        all_preds.append(preds)
    print(f"[Time] {time.perf_counter() - t0:.1f}s")

    per_fold = pd.DataFrame(all_rows)
    per_fold.to_csv(out / "cv_results.csv", index=False)
    pd.concat(all_preds, ignore_index=True).to_csv(out / "predictions.csv", index=False)

    summary = (
        per_fold.groupby("variant")
        .agg(
            n_features=("n_features", "first"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            accuracy_mean=("accuracy", "mean"),
            train_mae_mean=("train_mae", "mean"),
            mae_ci_lower_mean=("mae_ci_lower", "mean"),
            mae_ci_upper_mean=("mae_ci_upper", "mean"),
        )
        .reset_index()
    )
    full_mae = float(summary.loc[summary.variant == "full", "mae_mean"].iloc[0])
    summary["delta_mae_vs_full"] = (summary["mae_mean"] - full_mae).round(4)

    # Uncertainty of a *difference* must be computed on the difference itself.
    # Each variant is scored on the same folds as `full`, so pair by fold and
    # take the spread of those paired deltas; the per-variant bootstrap CI
    # describes that variant's own MAE and would badly overstate the delta's
    # uncertainty if drawn on a delta axis.
    full_by_fold = (
        per_fold[per_fold.variant == "full"].set_index("fold")["mae"]
    )
    paired = []
    for variant, grp in per_fold.groupby("variant"):
        d = grp.set_index("fold")["mae"] - full_by_fold
        paired.append(
            {
                "variant": variant,
                "delta_mean": float(d.mean()),
                "delta_sd": float(d.std()),
                "delta_min": float(d.min()),
                "delta_max": float(d.max()),
                "n_folds_worse": int((d > 0).sum()),
            }
        )
    summary = summary.merge(pd.DataFrame(paired), on="variant", how="left")
    summary["group"] = summary["variant"].str.split("::").str[-1]
    summary["group_label"] = summary["group"].map(GROUP_LABELS).fillna("All features")
    summary = summary.sort_values("mae_mean").reset_index(drop=True)
    # utf-8-sig: this table carries Chinese group labels and gets opened in Excel,
    # which assumes the local codepage for BOM-less UTF-8 and mangles them.
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    print("\n" + summary.round(4).to_string(index=False))

    # --- figures ------------------------------------------------------------
    drop = summary[summary.variant.str.startswith("drop::")].sort_values("delta_mae_vs_full")
    only = summary[summary.variant.str.startswith("only::")].sort_values("mae_mean")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    colors = ["#C44E52" if d > 0 else "#55A868" for d in drop.delta_mae_vs_full]
    axes[0].barh(drop.group, drop.delta_mae_vs_full, color=colors,
                 xerr=drop.delta_sd, capsize=3, error_kw={"alpha": 0.55})
    axes[0].axvline(0, color="black", lw=1)
    axes[0].set_xlabel("ΔMAE vs full model  (>0 = removing the group hurts)\n"
                       "error bars: SD of the fold-paired difference")
    axes[0].set_title("Leave-one-group-out")
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh(only.group, only.mae_mean, color="#4C72B0",
                 xerr=[(only.mae_mean - only.mae_ci_lower_mean).clip(lower=0),
                       (only.mae_ci_upper_mean - only.mae_mean).clip(lower=0)],
                 capsize=3, error_kw={"alpha": 0.45})
    axes[1].axvline(full_mae, ls="--", color="#4C72B0", lw=1.2,
                    label=f"full model ({full_mae:.3f})")
    axes[1].set_xlabel("MAE using only this group")
    axes[1].set_title("Group-only")
    axes[1].invert_yaxis()
    axes[1].legend()
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"Feature-group ablation — {args.model}, visit {visit}, n={data.n_samples}",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out / "figures" / "ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    (out / "meta.json").write_text(
        json.dumps(
            {
                "visit": visit,
                "model": args.model,
                "tune": args.tune,
                "n_samples": data.n_samples,
                "full_mae": full_mae,
                "groups": {k: len(v) for k, v in groups.items()},
                "random_state": args.random_state,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[OK] Ablation written to {out}")


if __name__ == "__main__":
    main()
