#!/usr/bin/env python3
"""Re-draw the model-vs-baseline MAE comparison for an existing run directory.

`batch_training.py` now writes this figure automatically on every run, so this
script is only needed to regenerate it for runs produced before that, or to
re-render with a different title. It reads the saved tables and does not retrain.

The plotting itself lives in ``qixing_fga.reporting.plots`` so the standalone
path and the training path cannot drift apart.

Usage:
  python scripts/plot_model_comparison.py --run-dir results/fga_fw_visit1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qixing_fga.evaluation.metrics import paired_bootstrap_delta_mae  # noqa: E402
from qixing_fga.reporting.plots import save_model_comparison_figure  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="results/fga_fw_2visit_n91")
    ap.add_argument("--target-name", default="fga_estimate_score")
    ap.add_argument("--title", default=None)
    ap.add_argument("--filename", default="model_mae_comparison.png")
    args = ap.parse_args()

    run = PROJECT_ROOT / args.run_dir
    summary_path = run / "nested_summary.csv"
    if not summary_path.is_file():
        raise SystemExit(f"No nested_summary.csv in {run}")
    summary = pd.read_csv(summary_path)

    per_fold_path = run / "cv_results_nested.csv"
    per_fold = pd.read_csv(per_fold_path) if per_fold_path.is_file() else pd.DataFrame()

    # Prefer the run's own saved CIs; otherwise recompute from its predictions.
    tests = pd.DataFrame()
    saved = run / "delta_mae_ci.csv"
    preds = run / "predictions.csv"
    if saved.is_file():
        tests = pd.read_csv(saved)
    elif preds.is_file():
        tests = paired_bootstrap_delta_mae(pd.read_csv(preds))
        if not tests.empty:
            tests.to_csv(saved, index=False)
            print(f"[OK] recomputed delta-MAE CIs → {saved}")

    out = save_model_comparison_figure(
        summary,
        per_fold,
        run,
        target_name=args.target_name,
        delta_mae=tests if not tests.empty else None,
        title=args.title or f"{args.target_name} — {run.name}",
        filename=args.filename,
    )
    if out is None:
        raise SystemExit("Figure not produced (empty summary or matplotlib missing).")

    cols = [c for c in ("model", "mae_mean", "mae_std", "accuracy_mean") if c in summary]
    tbl = summary[cols].sort_values("mae_mean")
    if not tests.empty:
        tbl = tbl.merge(tests[["model", "p_value"]], on="model", how="left")
    print(tbl.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
