#!/usr/bin/env python3
"""SHAP on out-of-fold rows, with cross-fold rank stability (S5).

Two rules this script exists to enforce:

  1. Explain held-out predictions, never training rows. For each outer fold the
     model is fitted on that fold's training participants and SHAP is computed
     only on its validation rows, then the per-fold blocks are stacked. SHAP on
     a model that has memorised its training data describes the memorisation.

  2. At n=46 a single global ranking is not trustworthy on its own. Feature
     importance is therefore also computed per fold and compared across folds
     (pairwise Spearman + Kendall's W). A high mean |SHAP| with low cross-fold
     agreement is a coin flip, not a finding, and the report says so.

Model choice: a tree model is used for SHAP because TreeExplainer is exact and
fast. The main-table winner (ordinal_logistic / mord) is not supported by any
SHAP explainer directly, so its ranking is not interchangeable with this one —
stated in the output rather than glossed over.

Outputs -> results/shap_2visit_n91/ (or --output-dir).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import shap  # noqa: E402

from qixing_fga.models.registry import build_models_for_task, maybe_tune  # noqa: E402
from qixing_fga.preprocessing import build_model_preprocessor  # noqa: E402
from qixing_fga.protocol import load_protocol_data  # noqa: E402


def kendalls_w(rank_matrix: np.ndarray) -> float:
    """Kendall's W over (n_raters x n_items) ranks; 0 = no agreement, 1 = identical."""
    m, n = rank_matrix.shape
    if m < 2:
        return float("nan")
    col_sums = rank_matrix.sum(axis=0)
    s = float(((col_sums - col_sums.mean()) ** 2).sum())
    denom = m**2 * (n**3 - n) / 12.0
    return float(s / denom) if denom > 0 else float("nan")


def _safe(name: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", name)[:80]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visit", default="all")
    ap.add_argument("--model", default="random_forest",
                    choices=["random_forest", "xgboost"])
    ap.add_argument("--tune", default="grid", choices=["none", "grid"])
    ap.add_argument("--tune-splits", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--output-dir", default="results/shap_2visit_n91")
    args = ap.parse_args()

    visit = args.visit if args.visit == "all" else int(args.visit)
    data = load_protocol_data(visit=visit, root=PROJECT_ROOT)
    print(f"[Protocol] {data.describe()}")
    print(f"[Model] {args.model} (TreeExplainer, exact)")

    out = Path(args.output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    features = list(data.feature_names)
    n_classes = int(data.y.max()) + 1
    shap_blocks: list[np.ndarray] = []
    x_blocks: list[np.ndarray] = []
    fold_importance: dict[str, np.ndarray] = {}
    oof_rows: list[pd.DataFrame] = []

    for fold_idx, (train_idx, test_idx) in enumerate(data.fold_indices):
        X_tr, X_te = data.X.iloc[train_idx], data.X.iloc[test_idx]
        y_tr, y_te = data.y.iloc[train_idx], data.y.iloc[test_idx]

        model = build_models_for_task("regression", n_classes, args.random_state)[args.model]
        pipe = Pipeline([("preprocess", build_model_preprocessor(X_tr)), ("model", model)])
        estimator = maybe_tune(
            pipe, args.model, "regression", y_tr, data.groups.iloc[train_idx],
            tune=args.tune, tune_splits=args.tune_splits,
            random_state=args.random_state + fold_idx,
        )
        estimator.fit(X_tr, y_tr)
        fitted: Pipeline = estimator.best_estimator_ if hasattr(estimator, "best_estimator_") else estimator

        # Explain in the model's own (preprocessed) space, on validation rows only.
        pre = fitted.named_steps["preprocess"]
        X_te_t = np.asarray(pre.transform(X_te), dtype=float)
        explainer = shap.TreeExplainer(fitted.named_steps["model"])
        sv = explainer.shap_values(X_te_t, check_additivity=False)
        sv = np.asarray(sv, dtype=float)
        if sv.ndim == 3:  # (n, f, outputs) -> average over outputs
            sv = sv.mean(axis=2)

        shap_blocks.append(sv)
        x_blocks.append(X_te_t)
        fold_importance[f"fold_{fold_idx + 1}"] = np.abs(sv).mean(axis=0)

        y_hat = fitted.predict(X_te)
        oof_rows.append(
            pd.DataFrame(
                {
                    "fold": fold_idx + 1,
                    "sample_id": data.ids.iloc[test_idx].to_numpy(),
                    "participant_id": data.groups.iloc[test_idx].to_numpy(),
                    "y_true": y_te.to_numpy(dtype=float),
                    "y_pred": np.asarray(y_hat, dtype=float),
                }
            )
        )
        print(f"  fold {fold_idx + 1}: SHAP on {sv.shape[0]} held-out rows")

    shap_all = np.vstack(shap_blocks)
    x_all = np.vstack(x_blocks)
    pd.concat(oof_rows, ignore_index=True).to_csv(out / "predictions.csv", index=False)

    # --- global importance --------------------------------------------------
    mean_abs = np.abs(shap_all).mean(axis=0)
    mean_signed = shap_all.mean(axis=0)
    imp = (
        pd.DataFrame(
            {
                "feature": features,
                "mean_abs_shap": mean_abs,
                "mean_signed_shap": mean_signed,
                "share_pct": 100.0 * mean_abs / mean_abs.sum(),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    imp.insert(0, "rank", np.arange(1, len(imp) + 1))
    imp.to_csv(out / "feature_importance.csv", index=False)
    print("\nTop features by mean |SHAP| (out-of-fold):")
    print(imp.head(args.top_k).round(4).to_string(index=False))

    # --- cross-fold stability ----------------------------------------------
    fold_imp = pd.DataFrame(fold_importance, index=features)
    fold_imp.to_csv(out / "fold_importance.csv")

    # Rank 1 = most important, within each fold.
    ranks = np.vstack([rankdata(-fold_imp[c].to_numpy()) for c in fold_imp.columns])
    w = kendalls_w(ranks)

    pairs = []
    for a, b in combinations(range(ranks.shape[0]), 2):
        rho, p = spearmanr(ranks[a], ranks[b])
        pairs.append(
            {
                "fold_a": fold_imp.columns[a],
                "fold_b": fold_imp.columns[b],
                "spearman_rho": float(rho),
                "p_value": float(p),
            }
        )
    pair_df = pd.DataFrame(pairs)
    pair_df.to_csv(out / "rank_stability_pairs.csv", index=False)

    # Does the top-k set survive across folds?
    topk_sets = [
        set(fold_imp[c].sort_values(ascending=False).head(args.top_k).index)
        for c in fold_imp.columns
    ]
    jaccard = [
        len(a & b) / len(a | b) for a, b in combinations(topk_sets, 2)
    ]
    counts = pd.Series(
        [f for s in topk_sets for f in s]
    ).value_counts().rename("n_folds_in_topk")
    counts.to_frame().to_csv(out / "topk_frequency.csv")

    stability = {
        "kendalls_w": w,
        "mean_pairwise_spearman": float(pair_df.spearman_rho.mean()),
        "min_pairwise_spearman": float(pair_df.spearman_rho.min()),
        "mean_topk_jaccard": float(np.mean(jaccard)),
        "top_k": args.top_k,
        "n_features_in_any_topk": int(len(counts)),
        "n_features_in_all_folds_topk": int((counts == len(topk_sets)).sum()),
    }
    print("\nCross-fold rank stability:")
    for k, v in stability.items():
        print(f"  {k}: {v}")
    verdict = (
        "high" if w >= 0.7 else "moderate" if w >= 0.5 else "low"
    )
    print(
        f"  -> agreement is {verdict}. "
        + (
            "Global ranking can be reported as-is."
            if verdict == "high"
            else "Report the ranking as indicative only; individual positions are "
            "not resolvable at this sample size."
        )
    )

    # --- figures ------------------------------------------------------------
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_all, x_all, feature_names=features, show=False, max_display=20)
    plt.title(f"SHAP (out-of-fold) — {args.model}, visit {visit}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out / "figures" / "shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    top = imp.head(args.top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in top.mean_signed_shap]
    ax.barh(top.feature, top.mean_abs_shap, color=colors)
    ax.set_xlabel("mean |SHAP| (out-of-fold rows)")
    ax.set_title(f"Top {args.top_k} features — colour = sign of mean signed SHAP")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "shap_top_features.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Stability heat map: per-fold rank of the global top-k.
    topk_feats = imp.head(args.top_k).feature.tolist()
    rank_df = pd.DataFrame(
        {c: rankdata(-fold_imp[c].to_numpy()) for c in fold_imp.columns}, index=features
    ).loc[topk_feats]
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(topk_feats) + 2))
    im = ax.imshow(rank_df.to_numpy(), cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(rank_df.columns)), rank_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(topk_feats)), topk_feats)
    for i in range(rank_df.shape[0]):
        for j in range(rank_df.shape[1]):
            ax.text(j, i, int(rank_df.iat[i, j]), ha="center", va="center",
                    color="white", fontsize=8)
    ax.set_title(f"Per-fold importance rank of the global top {args.top_k}\n"
                 f"Kendall's W = {w:.3f} ({verdict} agreement)")
    fig.colorbar(im, ax=ax, label="rank (1 = most important)")
    fig.tight_layout()
    fig.savefig(out / "figures" / "rank_stability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Dependence plots for the global top 3.
    for idx, feat in enumerate(imp.head(3).feature):
        j = features.index(feat)
        plt.figure(figsize=(6.5, 4.5))
        shap.dependence_plot(j, shap_all, x_all, feature_names=features,
                             show=False, interaction_index=None)
        plt.tight_layout()
        plt.savefig(out / "figures" / f"dependence_{idx + 1}_{_safe(feat)}.png",
                    dpi=200, bbox_inches="tight")
        plt.close()

    (out / "stability.json").write_text(json.dumps(stability, indent=2), encoding="utf-8")
    (out / "meta.json").write_text(
        json.dumps(
            {
                "visit": visit,
                "model": args.model,
                "explainer": "TreeExplainer",
                "shap_rows": int(shap_all.shape[0]),
                "n_features": len(features),
                "note": (
                    "SHAP computed on out-of-fold validation rows only. The "
                    "main-table best model (ordinal_logistic, mord.LogisticAT) has "
                    "no SHAP explainer, so this ranking describes the tree model's "
                    "behaviour and is not a direct explanation of that model."
                ),
                "random_state": args.random_state,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[OK] SHAP written to {out}")


if __name__ == "__main__":
    main()
