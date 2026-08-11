#!/usr/bin/env python3
"""Reference baselines before any complex model (Project_discription.txt S2).

Three tiers, all on the identical protocol (FW single-visit view, teammate folds,
per-fold bootstrap CI), so that later ablation / SHAP claims can be expressed as
gains over a stated reference rather than in the abstract:

  B1 naive       - train-fold median / mean / majority class
  B2 simple      - Ridge on a handful of interpretable gait features
  B3 one feature - Ridge on each single feature, ranked

Note on inputs: the clinician sub-scores (speed, imbalance, ...) are excluded
from B2 on purpose. They are the rater's judgement of the same video that
produced the FGA estimate, so predicting FGA from them would be circular. They
appear only as prediction *targets* in the multi-outcome sweep.

Outputs -> results/baselines_2visit_n91/ (or --output-dir).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qixing_fga.evaluation.bootstrap import per_fold_bootstrap_mae_rmse  # noqa: E402
from qixing_fga.evaluation.metrics import _compute_metrics  # noqa: E402
from qixing_fga.preprocessing import build_model_preprocessor  # noqa: E402
from qixing_fga.protocol import load_protocol_data  # noqa: E402

# Interpretable, clinically legible gait quantities (B2). Chosen a priori from
# the gait literature (pace / rhythm / stride / double-support), not by peeking
# at CV performance, so this stays an honest reference rather than a tuned model.
SIMPLE_FEATURES = [
    "mean_gait_speed",
    "cadence_spm",
    "step_length_mean",
    "double_support_time_fraction",
]


def _fold_eval(
    data,
    predict_fn,
    name: str,
    ci_alpha: float,
    n_classes: int,
    *,
    bootstrap: bool = True,
) -> tuple[list[dict], pd.DataFrame]:
    """Run one predictor over the fixed folds; return per-fold rows + OOF preds.

    ``bootstrap=False`` skips the 10,000-replicate CI. The single-feature scan
    only needs a ranking, and paying for 77 x 5 x 10,000 resamples to rank them
    would dominate the runtime; the feature that wins is then re-run with CIs.
    """
    rows: list[dict] = []
    oof: list[pd.DataFrame] = []
    for fold_idx, (train_idx, test_idx) in enumerate(data.fold_indices):
        y_tr = data.y.iloc[train_idx]
        y_te = data.y.iloc[test_idx]
        y_hat = predict_fn(train_idx, test_idx)

        m = _compute_metrics(y_te, y_hat, task="regression", n_classes=n_classes)
        row = {"model": name, "fold": fold_idx + 1, **m}

        if bootstrap:
            boot = per_fold_bootstrap_mae_rmse(
                y_val=y_te.to_numpy(),
                y_pred=np.asarray(y_hat, dtype=float),
                val_idx=np.asarray(test_idx, dtype=int),
                groups=data.groups.to_numpy(),
                boot_matrix=data.bootstrap_by_fold[f"fold_{fold_idx}"],
                ci_alpha=ci_alpha,
            )
            row.update(
                {
                    "mae_ci_lower": boot["mae_ci_lower"],
                    "mae_ci_upper": boot["mae_ci_upper"],
                    "rmse_ci_lower": boot["rmse_ci_lower"],
                    "rmse_ci_upper": boot["rmse_ci_upper"],
                }
            )
        rows.append(row)
        oof.append(
            pd.DataFrame(
                {
                    "model": name,
                    "fold": fold_idx + 1,
                    "sample_id": data.ids.iloc[test_idx].to_numpy(),
                    "participant_id": data.groups.iloc[test_idx].to_numpy(),
                    "y_true": y_te.to_numpy(dtype=float),
                    "y_pred": np.asarray(y_hat, dtype=float),
                }
            )
        )
        _ = y_tr  # kept for readability of the fold contract
    return rows, pd.concat(oof, ignore_index=True)


def _ridge_predictor(data, cols: list[str], alpha: float):
    def _predict(train_idx, test_idx):
        X_tr = data.X.iloc[train_idx][cols]
        X_te = data.X.iloc[test_idx][cols]
        pipe = Pipeline(
            [
                ("preprocess", build_model_preprocessor(X_tr)),
                ("model", Ridge(alpha=alpha)),
            ]
        )
        pipe.fit(X_tr, data.y.iloc[train_idx])
        return pipe.predict(X_te)

    return _predict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visit", default="all")
    ap.add_argument("--alpha", type=float, default=1.0, help="Ridge regularisation")
    ap.add_argument("--ci-alpha", type=float, default=0.05)
    ap.add_argument("--output-dir", default="results/baselines_2visit_n91")
    args = ap.parse_args()

    visit = args.visit if args.visit == "all" else int(args.visit)
    data = load_protocol_data(visit=visit, root=PROJECT_ROOT)
    n_classes = int(data.y.max()) + 1
    print(f"[Protocol] {data.describe()}")

    out = Path(args.output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_oof: list[pd.DataFrame] = []

    # --- B1: naive ----------------------------------------------------------
    naive = {
        "baseline_median": lambda tr, te: np.full(len(te), float(data.y.iloc[tr].median())),
        "baseline_mean": lambda tr, te: np.full(len(te), float(data.y.iloc[tr].mean())),
        "baseline_majority": lambda tr, te: np.full(
            len(te), float(data.y.iloc[tr].mode().iloc[0])
        ),
    }
    for name, fn in naive.items():
        rows, oof = _fold_eval(data, fn, name, args.ci_alpha, n_classes)
        all_rows += rows
        all_oof.append(oof)

    # --- B2: small interpretable feature set --------------------------------
    simple_cols = [c for c in SIMPLE_FEATURES if c in data.X.columns]
    missing = [c for c in SIMPLE_FEATURES if c not in data.X.columns]
    if missing:
        print(f"[WARN] simple-baseline features absent from the table: {missing}")
    rows, oof = _fold_eval(
        data,
        _ridge_predictor(data, simple_cols, args.alpha),
        f"simple_ridge_{len(simple_cols)}feat",
        args.ci_alpha,
        n_classes,
    )
    all_rows += rows
    all_oof.append(oof)

    # --- B3: single feature -------------------------------------------------
    single_rows: list[dict] = []
    for col in data.feature_names:
        rows_c, _ = _fold_eval(
            data, _ridge_predictor(data, [col], args.alpha), f"single::{col}",
            args.ci_alpha, n_classes, bootstrap=False,
        )
        df_c = pd.DataFrame(rows_c)
        single_rows.append(
            {
                "feature": col,
                "mae_mean": float(df_c["mae"].mean()),
                "mae_std": float(df_c["mae"].std()),
                "rmse_mean": float(df_c["rmse"].mean()),
                "accuracy_mean": float(df_c["accuracy"].mean()),
            }
        )
    single = pd.DataFrame(single_rows).sort_values("mae_mean").reset_index(drop=True)
    single.to_csv(out / "single_feature_ranking.csv", index=False)

    # Re-run the best single feature so it lands in the main comparison table.
    best_feat = single.iloc[0]["feature"]
    rows, oof = _fold_eval(
        data,
        _ridge_predictor(data, [best_feat], args.alpha),
        f"single_best::{best_feat}",
        args.ci_alpha,
        n_classes,
    )
    all_rows += rows
    all_oof.append(oof)

    # --- summarise ----------------------------------------------------------
    per_fold = pd.DataFrame(all_rows)
    per_fold.to_csv(out / "cv_results.csv", index=False)
    oof_all = pd.concat(all_oof, ignore_index=True)
    oof_all.to_csv(out / "predictions.csv", index=False)

    summary = (
        per_fold.groupby("model")
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            accuracy_mean=("accuracy", "mean"),
            mae_ci_lower_mean=("mae_ci_lower", "mean"),
            mae_ci_upper_mean=("mae_ci_upper", "mean"),
        )
        .reset_index()
        .sort_values("mae_mean")
    )
    ref = float(summary.loc[summary.model == "baseline_median", "mae_mean"].iloc[0])
    summary["pct_improvement_vs_median"] = (
        100.0 * (ref - summary["mae_mean"]) / ref
    ).round(2)
    summary.to_csv(out / "summary.csv", index=False)
    print("\n" + summary.round(4).to_string(index=False))

    # --- figures ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.2))
    s = summary.sort_values("mae_mean")
    colors = ["#C44E52" if m.startswith("baseline") else "#4C72B0" for m in s.model]
    ax.barh(s.model, s.mae_mean, xerr=s.mae_std, color=colors, capsize=4)
    ax.axvline(ref, ls="--", color="#C44E52", lw=1, label=f"median baseline ({ref:.3f})")
    ax.set_xlabel("MAE (mean ± SD across 5 outer folds)")
    ax.set_title(f"Baseline tiers, FGA 0-3, visit {visit} (n={data.n_samples})")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "baseline_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    top = single.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top.feature, top.mae_mean, xerr=top.mae_std, color="#55A868", capsize=3)
    ax.axvline(ref, ls="--", color="#C44E52", lw=1, label=f"median baseline ({ref:.3f})")
    ax.set_xlabel("MAE (mean ± SD across 5 outer folds)")
    ax.set_title("Best 15 single-feature Ridge models")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "single_feature_top15.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    (out / "meta.json").write_text(
        json.dumps(
            {
                "visit": visit,
                "n_samples": data.n_samples,
                "n_participants": data.n_participants,
                "ridge_alpha": args.alpha,
                "simple_features": simple_cols,
                "best_single_feature": best_feat,
                "median_baseline_mae": ref,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[OK] Baselines written to {out}")


if __name__ == "__main__":
    main()
