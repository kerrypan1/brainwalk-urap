#!/usr/bin/env python3
"""Main-model diagnostics: learning curves + significance vs baseline (S3).

Two questions a reviewer will ask about an n=46 study:

  1. "Is the model saturated, or would more participants help?"  -> learning curve
     over training-set size, drawn per outer fold with the validation fold held
     fixed so the curve is not confounded by which people are being scored.
  2. "Is the improvement over the naive baseline real?"  -> paired bootstrap CI on
     delta-MAE (same samples, two predictors; participants are the resampling
     unit). No claim of an improvement without this interval.

Outputs -> results/model_diagnostics_2visit_n91/ (or --output-dir).
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
from sklearn.pipeline import Pipeline  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qixing_fga.evaluation.metrics import paired_bootstrap_delta_mae  # noqa: E402
from qixing_fga.models.registry import build_models_for_task  # noqa: E402
from qixing_fga.preprocessing import build_model_preprocessor  # noqa: E402
from qixing_fga.protocol import load_protocol_data  # noqa: E402

FRACTIONS = [0.25, 0.4, 0.55, 0.7, 0.85, 1.0]


def learning_curve(data, model_name: str, random_state: int, n_repeats: int) -> pd.DataFrame:
    """Train/val MAE as the training fold grows, subsampling by participant."""
    rows: list[dict] = []
    for fold_idx, (train_idx, test_idx) in enumerate(data.fold_indices):
        train_pids = data.groups.iloc[train_idx].unique()
        X_te, y_te = data.X.iloc[test_idx], data.y.iloc[test_idx]

        for frac in FRACTIONS:
            n_take = max(4, int(round(frac * len(train_pids))))
            repeats = 1 if n_take >= len(train_pids) else n_repeats
            for rep in range(repeats):
                rng = np.random.default_rng(random_state + 1000 * fold_idx + rep)
                pids = rng.choice(train_pids, size=n_take, replace=False)
                mask = data.groups.iloc[train_idx].isin(pids).to_numpy()
                sub = np.asarray(train_idx)[mask]
                X_tr, y_tr = data.X.iloc[sub], data.y.iloc[sub]
                if y_tr.nunique() < 2:
                    continue

                model = build_models_for_task("regression", int(data.y.max()) + 1, random_state)[
                    model_name
                ]
                pipe = Pipeline(
                    [("preprocess", build_model_preprocessor(X_tr)), ("model", model)]
                )
                try:
                    pipe.fit(X_tr, y_tr)
                except Exception as exc:  # rare: a level missing from a small draw
                    print(f"  [skip] fold{fold_idx} frac{frac} rep{rep}: {type(exc).__name__}")
                    continue
                rows.append(
                    {
                        "fold": fold_idx + 1,
                        "fraction": frac,
                        "n_train_participants": int(n_take),
                        "rep": rep,
                        "train_mae": float(np.mean(np.abs(pipe.predict(X_tr) - y_tr))),
                        "val_mae": float(np.mean(np.abs(pipe.predict(X_te) - y_te))),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visit", default="all")
    ap.add_argument("--model", default="ordinal_logistic")
    ap.add_argument("--predictions", default="results/fga_fw_2visit_n91/predictions.csv")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--output-dir", default="results/model_diagnostics_2visit_n91")
    args = ap.parse_args()

    visit = args.visit if args.visit == "all" else int(args.visit)
    data = load_protocol_data(visit=visit, root=PROJECT_ROOT)
    print(f"[Protocol] {data.describe()}")

    out = Path(args.output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    print(f"[1/2] Learning curve for {args.model}...")
    lc = learning_curve(data, args.model, args.random_state, args.n_repeats)
    lc.to_csv(out / "learning_curve_raw.csv", index=False)
    # Aggregate by fraction, not by raw participant count: outer folds differ in
    # size, so equal fractions land on different absolute counts and grouping by
    # count would split each curve point into several thinly-sampled bins.
    agg = (
        lc.groupby("fraction")
        .agg(
            n_train_participants=("n_train_participants", "mean"),
            train_mae_mean=("train_mae", "mean"),
            train_mae_std=("train_mae", "std"),
            val_mae_mean=("val_mae", "mean"),
            val_mae_std=("val_mae", "std"),
            n_runs=("val_mae", "size"),
        )
        .reset_index()
    )
    agg["n_train_participants"] = agg["n_train_participants"].round(1)
    agg.to_csv(out / "learning_curve.csv", index=False)
    print(agg.round(3).to_string(index=False))

    print(f"\n[2/2] Evidence vs baseline_median ({args.predictions})...")
    preds = pd.read_csv(PROJECT_ROOT / args.predictions)
    deltas = paired_bootstrap_delta_mae(preds, random_state=args.random_state)
    if not deltas.empty:
        deltas.to_csv(out / "delta_mae_ci.csv", index=False)
        print("Delta-MAE vs baseline, paired bootstrap 95% CI (reported statistic):")
        print(deltas.round(4).to_string(index=False))

    # Learning curve figure
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(agg.n_train_participants, agg.val_mae_mean, "o-", color="#4C72B0",
            label="Validation MAE (outer test fold)")
    ax.fill_between(
        agg.n_train_participants,
        agg.val_mae_mean - agg.val_mae_std,
        agg.val_mae_mean + agg.val_mae_std,
        alpha=0.2, color="#4C72B0",
    )
    ax.plot(agg.n_train_participants, agg.train_mae_mean, "s--", color="#DD8452",
            label="Training MAE")
    ax.fill_between(
        agg.n_train_participants,
        agg.train_mae_mean - agg.train_mae_std,
        agg.train_mae_mean + agg.train_mae_std,
        alpha=0.2, color="#DD8452",
    )
    ax.set_xlabel("Training participants")
    ax.set_ylabel("MAE (FGA 0-3)")
    ax.set_title(f"Learning curve — {args.model}, visit {visit}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "learning_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Train/val gap figure
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    gap = agg.val_mae_mean - agg.train_mae_mean
    ax.bar(agg.n_train_participants.astype(str), gap, color="#C44E52")
    ax.set_xlabel("Training participants")
    ax.set_ylabel("val MAE - train MAE")
    ax.set_title(f"Generalisation gap — {args.model}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "generalisation_gap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    (out / "meta.json").write_text(
        json.dumps(
            {
                "visit": visit,
                "model": args.model,
                "n_samples": data.n_samples,
                "fractions": FRACTIONS,
                "n_repeats": args.n_repeats,
                "random_state": args.random_state,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[OK] Diagnostics written to {out}")


if __name__ == "__main__":
    main()
