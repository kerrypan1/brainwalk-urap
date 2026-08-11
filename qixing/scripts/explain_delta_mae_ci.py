#!/usr/bin/env python3
"""Regenerate the worked example that explains the reported 95% CI column.

The main results table carries a CI on ΔMAE (model − baseline), not on MAE. That
distinction is the single most misread thing in the table: two MAE CIs can
overlap heavily while the paired difference is consistently negative, because
both MAEs rise and fall together with whichever participants a resample happens
to draw. Rather than assert that, this prints the actual resamples so a reader
can watch it happen.

The resampling is fold-stratified and the per-fold MAEs are averaged with equal
weight, matching evaluation/metrics.py::paired_bootstrap_delta_mae — so the delta
CI printed here is the one the main table reports, not a second convention.

Outputs -> results/delta_mae_ci_explainer/ (table + the summary numbers quoted in
docs/METRICS_RATIONALE.md). Nothing here feeds a model; it exists so the numbers
in the docs can be re-derived instead of trusted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--predictions", default="results/fga_fw_2visit_n91/predictions.csv"
    )
    ap.add_argument("--model", default="ordinal_logistic")
    ap.add_argument("--baseline", default="baseline_median")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--n-show", type=int, default=8, help="resamples to tabulate")
    # Seed 7 (not the project's 42) only so the illustrated draws are a plain
    # sample rather than the ones the reported CI happens to start with.
    ap.add_argument("--random-state", type=int, default=7)
    ap.add_argument("--output-dir", default="results/delta_mae_ci_explainer")
    args = ap.parse_args()

    preds = pd.read_csv(PROJECT_ROOT / args.predictions)
    base = preds[preds.model == args.baseline].set_index("sample_id")
    if base.empty:
        raise SystemExit(f"{args.baseline} not found in {args.predictions}")
    model = preds[preds.model == args.model].set_index("sample_id")
    if model.empty:
        raise SystemExit(f"{args.model} not found in {args.predictions}")

    base_err = (base.y_true - base.y_pred).abs()
    model_err = (model.y_true - model.y_pred).abs().loc[base_err.index]
    pid = base.participant_id

    # Fold-stratified, matching paired_bootstrap_delta_mae: participants are
    # resampled within their own fold and the fold MAEs are averaged with equal
    # weight, so the delta CI printed here is the one the main table reports.
    fold_of = base.fold.values
    folds = sorted(pd.unique(fold_of))
    parts = np.array(sorted(pid.unique()))
    fold_parts: dict[object, np.ndarray] = {}
    rows_of: dict[tuple, np.ndarray] = {}
    counts_of: dict[object, np.ndarray] = {}
    for f in folds:
        in_fold = fold_of == f
        qs = np.array(sorted(pd.unique(pid.values[in_fold])))
        fold_parts[f] = qs
        for q in qs:
            rows_of[(f, q)] = np.where(in_fold & (pid.values == q))[0]
        counts_of[f] = np.array([len(rows_of[(f, q)]) for q in qs], dtype=float)

    rng = np.random.default_rng(args.random_state)
    # Drawn once and reused for both models so every resample stays paired.
    draws = {
        f: rng.integers(0, len(fold_parts[f]), size=(args.n_boot, len(fold_parts[f])))
        for f in folds
    }

    def bootstrap(err: pd.Series) -> np.ndarray:
        """Fold-mean MAE per resample; participants resampled inside each fold."""
        v = err.values
        total = np.zeros(args.n_boot)
        for f in folds:
            sums = np.array([v[rows_of[(f, q)]].sum() for q in fold_parts[f]])
            d = draws[f]
            total += sums[d].sum(axis=1) / counts_of[f][d].sum(axis=1)
        return total / len(folds)

    def fold_mean(err: pd.Series) -> float:
        return float(np.mean([err.values[fold_of == f].mean() for f in folds]))

    boot_model = bootstrap(model_err)
    boot_base = bootstrap(base_err)
    boot_delta = boot_model - boot_base

    shown = pd.DataFrame(
        {
            "resample": np.arange(1, args.n_show + 1),
            "model_mae": boot_model[: args.n_show].round(3),
            "baseline_mae": boot_base[: args.n_show].round(3),
            "delta_mae": boot_delta[: args.n_show].round(3),
        }
    )

    def ci(x: np.ndarray) -> tuple[float, float]:
        lo, hi = np.percentile(x, [2.5, 97.5])
        return float(lo), float(hi)

    m_lo, m_hi = ci(boot_model)
    b_lo, b_hi = ci(boot_base)
    d_lo, d_hi = ci(boot_delta)
    overlap = max(0.0, min(m_hi, b_hi) - max(m_lo, b_lo))

    summary = {
        "model": args.model,
        "baseline": args.baseline,
        "n_samples": int(len(base_err)),
        "n_participants": int(len(parts)),
        "n_folds": int(len(folds)),
        "aggregation": "fold_mean",
        "n_bootstrap": int(args.n_boot),
        "random_state": int(args.random_state),
        "model_mae": fold_mean(model_err),
        "baseline_mae": fold_mean(base_err),
        "delta_mae": fold_mean(model_err) - fold_mean(base_err),
        "model_mae_ci": [m_lo, m_hi],
        "baseline_mae_ci": [b_lo, b_hi],
        "delta_mae_ci": [d_lo, d_hi],
        "mae_ci_overlap_width": float(overlap),
        "mae_ci_overlap_pct_of_model_ci": float(100 * overlap / (m_hi - m_lo)),
        "sd_model_mae": float(boot_model.std()),
        "sd_baseline_mae": float(boot_base.std()),
        "sd_delta_mae": float(boot_delta.std()),
        "corr_model_baseline": float(np.corrcoef(boot_model, boot_base)[0, 1]),
        "pct_resamples_model_better": float(100 * (boot_delta < 0).mean()),
        "delta_ci_width": float(d_hi - d_lo),
        "delta_ci_width_if_independent": float(
            np.hypot(m_hi - m_lo, b_hi - b_lo)
        ),
    }

    out = PROJECT_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    shown.to_csv(out / "resample_examples.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(
        f"First {args.n_show} resamples "
        f"(participants redrawn within each of the {len(folds)} folds, "
        f"fold MAEs then averaged with equal weight):"
    )
    print(shown.to_string(index=False))
    print(
        f"\nMAE 95% CI      model    [{m_lo:.3f}, {m_hi:.3f}]"
        f"\n                baseline [{b_lo:.3f}, {b_hi:.3f}]"
        f"\n                overlap  {summary['mae_ci_overlap_pct_of_model_ci']:.0f}%"
        f" of the model's interval"
        f"\ndelta-MAE 95% CI         [{d_lo:+.3f}, {d_hi:+.3f}]"
        f"\n\nSD  model {summary['sd_model_mae']:.3f} | baseline"
        f" {summary['sd_baseline_mae']:.3f} | delta {summary['sd_delta_mae']:.3f}"
        f"\ncorr(model, baseline) across resamples ="
        f" {summary['corr_model_baseline']:.3f}"
        f"\nresamples where the model wins ="
        f" {summary['pct_resamples_model_better']:.1f}%"
    )
    print(f"\n[OK] Explainer written to {out}")


if __name__ == "__main__":
    main()
