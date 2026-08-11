#!/usr/bin/env python3
"""Zeno multi-outcome prediction with correlation structure and FDR control (S6).

The brief asks for 13-14 Zeno measures to be predicted. Running 16 separate
models and reporting the ones that worked would be selective reporting, and the
measures are not independent tests anyway — stride length, velocity and cadence
are kinematically coupled (velocity = cadence x stride length / 2). So this
script does four things in order:

  1. reconcile the target list (the brief lists 13 and says 14; the repo config
     has 16) and write the discrepancy out rather than silently picking one;
  2. measure how coupled the targets actually are (Spearman + hierarchical
     clustering + PCA), giving an empirical count of independent dimensions;
  3. predict every target under one protocol, testing each against its own naive
     baseline with a paired Wilcoxon on absolute errors;
  4. control the false discovery rate across those tests (Benjamini-Hochberg),
     and report significance before and after correction.

Data view: FW only, one row per participant — the same definition as the FGA
experiments. Note this is a larger, different cohort (199 participants) than the
46-participant FGA set, so numbers are not comparable across the two.

Outputs -> results/multi_outcome_fdr/ (or --output-dir).
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
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from scipy.stats import false_discovery_control, wilcoxon  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qixing_fga.features.columns import resolve_feature_columns  # noqa: E402
from qixing_fga.models.registry import build_models_for_task  # noqa: E402
from qixing_fga.preprocessing import build_model_preprocessor  # noqa: E402

# The brief's list (13 named entries, described there as 14).
BRIEF_TARGETS = [
    "singlesupportratiolr", "walkratiocmstepsminmean", "stridewidthcmsd",
    "stridelengthcmcv", "stridetimeseccv", "meanegvimean", "stridewidthcmmean",
    "cadencestepsminmean", "absolutesteplengthcmmean", "singlesupportmean",
    "stridelengthcmmean", "velocitycmsecmean", "stridevelocitycmsecmean",
]
# The repo config's list (configs/multi_outcome.yaml).
CONFIG_TARGETS = BRIEF_TARGETS + [
    "totaldsupportmean", "totaldsupportratiolr", "absolutesteplengthcmratiolr",
]


def build_view(path: Path, targets: list[str]) -> tuple[pd.DataFrame, dict]:
    """FW rows, one per participant (first occurrence), targets coerced to float."""
    df = pd.read_csv(path)
    info = {"raw_rows": int(len(df)), "raw_participants": int(df.participant_id.nunique())}

    if "walking_condition" in df.columns:
        df = df[df.walking_condition == "FW"]
    info["fw_rows"] = int(len(df))

    if "feature_status" in df.columns:
        df = df[df.feature_status.astype(str).str.upper().str.strip() == "OK"]
    info["fw_ok_rows"] = int(len(df))

    # Deterministic one-row-per-participant rule: earliest labelled video date
    # when available, else first row in file order. Recorded so it is auditable.
    if "label_video_date" in df.columns:
        df = df.assign(_d=pd.to_datetime(df.label_video_date, errors="coerce"))
        df = df.sort_values(["participant_id", "_d"], na_position="last").drop(columns="_d")
        info["dedup_rule"] = "earliest label_video_date, ties by file order"
    else:
        info["dedup_rule"] = "first row in file order"
    df = df.drop_duplicates(subset="participant_id", keep="first").reset_index(drop=True)
    info["final_rows"] = int(len(df))
    info["final_participants"] = int(df.participant_id.nunique())

    coerced = {}
    for t in targets:
        if t in df.columns:
            before = df[t].notna().sum()
            df[t] = pd.to_numeric(df[t], errors="coerce")
            after = df[t].notna().sum()
            if after < before:
                coerced[t] = int(before - after)
    info["non_numeric_cells_dropped"] = coerced
    return df, info


def evaluate_target(df, feature_cols, target, models, n_splits, random_state):
    """GroupKFold CV for one continuous target; returns per-fold rows + OOF preds."""
    sub = df[df[target].notna()]
    if len(sub) < 20 or sub.participant_id.nunique() < n_splits:
        return None, None

    X = sub[feature_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    y = sub[target].astype(float).reset_index(drop=True)
    groups = sub.participant_id.astype(str).reset_index(drop=True)

    splits = list(GroupKFold(n_splits=n_splits).split(X, groups=groups))
    rows, preds = [], []
    for fold, (tr, te) in enumerate(splits, 1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        base = np.full(len(te), float(y_tr.median()))
        rows.append({"target": target, "model": "baseline_median", "fold": fold,
                     "mae": float(np.mean(np.abs(y_te - base))),
                     "rmse": float(np.sqrt(np.mean((y_te - base) ** 2)))})
        preds.append(pd.DataFrame({"target": target, "model": "baseline_median",
                                   "fold": fold, "participant_id": groups.iloc[te].to_numpy(),
                                   "y_true": y_te.to_numpy(), "y_pred": base}))

        for name in models:
            model = build_models_for_task("regression_continuous", 0, random_state)[name]
            pipe = Pipeline([("preprocess", build_model_preprocessor(X_tr)), ("model", model)])
            try:
                pipe.fit(X_tr, y_tr)
                yh = pipe.predict(X_te)
            except Exception as exc:
                print(f"    [skip] {target}/{name} fold {fold}: {type(exc).__name__}")
                continue
            rows.append({"target": target, "model": name, "fold": fold,
                         "mae": float(np.mean(np.abs(y_te - yh))),
                         "rmse": float(np.sqrt(np.mean((y_te - yh) ** 2)))})
            preds.append(pd.DataFrame({"target": target, "model": name, "fold": fold,
                                       "participant_id": groups.iloc[te].to_numpy(),
                                       "y_true": y_te.to_numpy(), "y_pred": yh}))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-file", default="data/training_features_zeno_mentor_outcomes_wide.csv")
    ap.add_argument("--models", nargs="+", default=["ridge", "random_forest"])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--output-dir", default="results/multi_outcome_fdr")
    args = ap.parse_args()

    out = Path(args.output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # --- 1. reconcile the target list --------------------------------------
    df_raw = pd.read_csv(PROJECT_ROOT / args.data_file)
    manifest = pd.DataFrame(
        {
            "target": sorted(set(CONFIG_TARGETS) | set(BRIEF_TARGETS)),
        }
    )
    manifest["in_brief"] = manifest.target.isin(BRIEF_TARGETS)
    manifest["in_repo_config"] = manifest.target.isin(CONFIG_TARGETS)
    manifest["in_data_file"] = manifest.target.isin(df_raw.columns)
    manifest.to_csv(out / "target_manifest.csv", index=False)
    print("=" * 72)
    print("TARGET LIST RECONCILIATION")
    print(f"  brief lists {len(BRIEF_TARGETS)} names (text claims 14 -> off by one)")
    print(f"  repo config lists {len(CONFIG_TARGETS)}")
    print(f"  present in data file: {int(manifest.in_data_file.sum())}")
    extra = manifest[manifest.in_repo_config & ~manifest.in_brief].target.tolist()
    print(f"  in config but not in the brief: {extra}")
    missing = manifest[~manifest.in_data_file].target.tolist()
    print(f"  requested but absent from data: {missing if missing else 'none'}")

    targets = [t for t in CONFIG_TARGETS if t in df_raw.columns]

    # --- 2. build the view --------------------------------------------------
    df, info = build_view(PROJECT_ROOT / args.data_file, targets)
    print("\n" + "=" * 72)
    print("DATA VIEW (FW, one row per participant)")
    for k, v in info.items():
        print(f"  {k}: {v}")

    feature_cols = [c for c in resolve_feature_columns("gait_numeric") if c in df.columns]
    print(f"  features available: {len(feature_cols)}")

    # --- 3. correlation structure ------------------------------------------
    T = df[targets].apply(pd.to_numeric, errors="coerce")
    corr = T.corr(method="spearman")
    corr.to_csv(out / "target_correlation_spearman.csv", encoding="utf-8-sig")

    # pandas 3 hands back read-only arrays, so take an explicit writable copy.
    dist = (1.0 - corr.abs()).to_numpy(dtype=float, copy=True)
    dist = np.nan_to_num(dist, nan=1.0)  # a target with no valid pairs is "far"
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # enforce exact symmetry for squareform
    Z = linkage(squareform(dist, checks=False), method="average")
    for thresh in (0.3, 0.5):
        df_cl = pd.DataFrame(
            {"target": corr.columns,
             "cluster": fcluster(Z, t=thresh, criterion="distance")}
        )
        df_cl.to_csv(out / f"target_clusters_dist{thresh}.csv", index=False)
        print(f"\n  |rho|>{1 - thresh:.1f} clusters: "
              f"{df_cl.cluster.nunique()} groups from {len(targets)} targets")

    complete = T.dropna()
    pca_info = {}
    if len(complete) >= 10:
        Zs = (complete - complete.mean()) / complete.std(ddof=0)
        Zs = Zs.dropna(axis=1, how="any")
        pca = PCA().fit(Zs.to_numpy())
        evr = pca.explained_variance_ratio_
        cum = np.cumsum(evr)
        pca_info = {
            "n_complete_rows": int(len(complete)),
            "n_targets_in_pca": int(Zs.shape[1]),
            "explained_variance_ratio": [round(float(v), 4) for v in evr[:10]],
            "n_pcs_for_80pct": int(np.searchsorted(cum, 0.80) + 1),
            "n_pcs_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
        }
        print(f"\n  PCA on {Zs.shape[1]} targets ({len(complete)} complete rows): "
              f"{pca_info['n_pcs_for_80pct']} PCs explain 80%, "
              f"{pca_info['n_pcs_for_90pct']} explain 90%")
        print("  -> effective number of independent outcomes is far below "
              f"{len(targets)}; treat the sweep accordingly.")

    # --- 4. predict every target -------------------------------------------
    print("\n" + "=" * 72)
    print("PREDICTION SWEEP")
    all_rows, all_preds = [], []
    for i, t in enumerate(targets, 1):
        rows, preds = evaluate_target(df, feature_cols, t, args.models,
                                      args.n_splits, args.random_state)
        if rows is None:
            print(f"  [{i}/{len(targets)}] {t}: skipped (too few labelled rows)")
            continue
        all_rows.append(rows)
        all_preds.append(preds)
        best = rows[rows.model != "baseline_median"].groupby("model").mae.mean().min()
        base = rows[rows.model == "baseline_median"].mae.mean()
        print(f"  [{i}/{len(targets)}] {t}: best MAE {best:.4f} vs baseline {base:.4f}")

    per_fold = pd.concat(all_rows, ignore_index=True)
    per_fold.to_csv(out / "cv_results.csv", index=False)
    preds_all = pd.concat(all_preds, ignore_index=True)
    preds_all.to_csv(out / "predictions.csv", index=False)

    # --- 5. paired tests + FDR ---------------------------------------------
    tests = []
    for t in preds_all.target.unique():
        sub = preds_all[preds_all.target == t]
        base = sub[sub.model == "baseline_median"]
        base_err = np.abs(base.y_true.to_numpy() - base.y_pred.to_numpy())
        for m in args.models:
            mm = sub[sub.model == m]
            if mm.empty or len(mm) != len(base):
                continue
            err = np.abs(mm.y_true.to_numpy() - mm.y_pred.to_numpy())
            if np.allclose(err, base_err):
                stat, p = np.nan, 1.0
            else:
                stat, p = wilcoxon(err, base_err)
            tests.append({
                "target": t, "model": m, "n": int(len(err)),
                "mae": float(err.mean()), "baseline_mae": float(base_err.mean()),
                "mae_delta": float(err.mean() - base_err.mean()),
                "pct_improvement": float(100 * (base_err.mean() - err.mean()) / base_err.mean()),
                "p_value": float(p),
            })
    tests = pd.DataFrame(tests)

    # BH across every target x model test in the family.
    tests["p_fdr"] = false_discovery_control(tests.p_value.to_numpy(), method="bh")
    tests["sig_raw"] = tests.p_value < args.alpha
    tests["sig_fdr"] = tests.p_fdr < args.alpha
    tests["beats_baseline"] = tests.mae_delta < 0
    tests["sig_fdr_and_better"] = tests.sig_fdr & tests.beats_baseline
    tests = tests.sort_values("p_fdr").reset_index(drop=True)
    tests.to_csv(out / "fdr_tests.csv", index=False)

    print("\n" + "=" * 72)
    print("FDR-CONTROLLED COMPARISON vs naive median baseline")
    print(tests[["target", "model", "mae", "baseline_mae", "pct_improvement",
                 "p_value", "p_fdr", "sig_raw", "sig_fdr_and_better"]]
          .round(4).to_string(index=False))
    print(f"\n  significant before correction: {int(tests.sig_raw.sum())}/{len(tests)}")
    print(f"  significant after BH-FDR:      {int(tests.sig_fdr.sum())}/{len(tests)}")
    print(f"  ...and actually better than baseline: {int(tests.sig_fdr_and_better.sum())}")

    # --- 6. figures ---------------------------------------------------------
    order = dendrogram(Z, no_plot=True, labels=list(corr.columns))["ivl"]
    cm = corr.loc[order, order]
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=8)
    ax.set_yticks(range(len(order)), order, fontsize=8)
    for i in range(len(order)):
        for j in range(len(order)):
            v = cm.iat[i, j]
            if abs(v) >= 0.7 and i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("Zeno outcome Spearman correlation (clustered)\n"
                 "labels shown where |rho| >= 0.7", fontsize=11)
    fig.colorbar(im, ax=ax, label="Spearman rho", shrink=0.7)
    fig.tight_layout()
    fig.savefig(out / "figures" / "target_correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    dendrogram(Z, labels=list(corr.columns), ax=ax, leaf_rotation=90, leaf_font_size=8)
    ax.set_ylabel("1 - |Spearman rho|")
    ax.set_title("Hierarchical clustering of Zeno outcomes")
    fig.tight_layout()
    fig.savefig(out / "figures" / "target_dendrogram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if pca_info:
        evr = np.array(pca.explained_variance_ratio_)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.bar(range(1, len(evr) + 1), evr, color="#4C72B0", label="individual")
        ax.plot(range(1, len(evr) + 1), np.cumsum(evr), "o-", color="#C44E52",
                label="cumulative")
        ax.axhline(0.8, ls="--", color="grey", lw=1)
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Explained variance ratio")
        ax.set_title(f"PCA of Zeno outcomes — {pca_info['n_pcs_for_80pct']} PCs reach 80%")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "figures" / "target_pca_scree.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    best = (tests.sort_values("mae_delta").groupby("target", as_index=False).first()
            .sort_values("pct_improvement", ascending=False))
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(best) + 2))
    colors = ["#55A868" if s else "#BBBBBB" for s in best.sig_fdr_and_better]
    ax.barh(best.target, best.pct_improvement, color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("% MAE improvement over naive median baseline")
    ax.set_title("Per-outcome improvement (green = survives BH-FDR at "
                 f"alpha={args.alpha})")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "outcome_performance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    (out / "summary.json").write_text(
        json.dumps(
            {
                "data_view": info,
                "n_targets_evaluated": int(preds_all.target.nunique()),
                "brief_target_count": len(BRIEF_TARGETS),
                "config_target_count": len(CONFIG_TARGETS),
                "pca": pca_info,
                "n_tests": int(len(tests)),
                "n_sig_raw": int(tests.sig_raw.sum()),
                "n_sig_fdr": int(tests.sig_fdr.sum()),
                "n_sig_fdr_and_better": int(tests.sig_fdr_and_better.sum()),
                "models": args.models,
                "n_splits": args.n_splits,
                "random_state": args.random_state,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[OK] Multi-outcome + FDR written to {out}")


if __name__ == "__main__":
    main()
