"""Multi-outcome sweep: FGA clinical sub-scores + Zeno walkway measures.

Runs ``src/batch_training.py`` once per outcome via subprocess (no refactor of
its ``main``), then aggregates each run's ``nested_summary.csv`` into a single
``comparison.csv`` and a MAE bar chart.

Outcomes
--------
- FGA sub-scores (ordinal, text-prefixed labels) -> ``--task regression``:
  speed, imbalance, gait_deviation, deviation_outside_walkway.
- FGA sub-score (nominal, imbalanced) -> ``--task classification``:
  assistive_device.
- Zeno walkway measures (continuous) -> ``--task regression_continuous``
  with ``--data-file`` on the wide table (16 mentor targets).

The feature whitelist (MODEL_FEATURE_COLUMNS) intersected with the wide table
keeps only mediapipe features and drops every Zeno target column, so no target
leaks into X. Grouping is by participant_id in every run.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

try:  # allow both `python scripts/run_multi_outcome.py` and module import
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:  # pragma: no cover - plotting is optional
    HAS_MPL = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BATCH_SCRIPT = PROJECT_ROOT / "src" / "batch_training.py"
ZENO_DATA_FILE = "data/training_features_zeno_mentor_outcomes_wide.csv"

SUBSCORE_ORDINAL = ["speed", "imbalance", "gait_deviation", "deviation_outside_walkway"]
SUBSCORE_NOMINAL = ["assistive_device"]
ZENO_TARGETS = [
    "cadencestepsminmean",
    "velocitycmsecmean",
    "totaldsupportmean",
    "totaldsupportratiolr",
    "singlesupportratiolr",
    "singlesupportmean",
    "stridevelocitycmsecmean",
    "stridewidthcmmean",
    "absolutesteplengthcmratiolr",
    "absolutesteplengthcmmean",
    "meanegvimean",
    "walkratiocmstepsminmean",
    "stridetimeseccv",
    "stridelengthcmmean",
    "stridelengthcmcv",
    "stridewidthcmsd",
]


def build_plan(zeno_targets: list[str]) -> list[dict]:
    """Ordered list of {outcome, task, kind, data_file} to run."""
    plan: list[dict] = []
    for name in SUBSCORE_ORDINAL:
        plan.append({"outcome": name, "task": "regression", "kind": "subscore"})
    for name in SUBSCORE_NOMINAL:
        plan.append({"outcome": name, "task": "classification", "kind": "subscore"})
    for name in zeno_targets:
        plan.append(
            {
                "outcome": name,
                "task": "regression_continuous",
                "kind": "zeno",
                "data_file": ZENO_DATA_FILE,
            }
        )
    return plan


def run_one(
    item: dict,
    out_dir: Path,
    *,
    splits: int,
    random_state: int,
    tune: str,
    python_exe: str,
) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        str(BATCH_SCRIPT),
        "--target",
        item["outcome"],
        "--task",
        item["task"],
        "--splits",
        str(splits),
        "--random-state",
        str(random_state),
        "--tune",
        tune,
        "--no-participant-aggregation",
        "--output-dir",
        str(out_dir),
    ]
    if item.get("data_file"):
        cmd += ["--data-file", item["data_file"]]

    print(f"\n=== [{item['kind']}] {item['outcome']} ({item['task']}) ===", flush=True)
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        print(f"  ! run failed (exit {proc.returncode}); skipping in aggregation.")
        return False
    return True


def _full_coverage_models(out_dir: Path) -> set:
    """Models whose predictions cover every sample the baseline did (all folds).

    A model that fails on some folds (e.g. mord/XGBoost when a fold lacks a rare
    class) predicts on only a subset of samples, so its MAE is computed on fewer
    rows than the baseline and is NOT comparable. Such models are excluded from
    the best-model choice.
    """
    pred_path = out_dir / "predictions.csv"
    if not pred_path.exists():
        return set()
    preds = pd.read_csv(pred_path)
    if preds.empty or "model" not in preds.columns:
        return set()
    counts = preds.groupby("model").size()
    base = int(counts.get("baseline_median", counts.max()))
    return {str(m) for m, c in counts.items() if int(c) == base}


def _n_samples_participants(out_dir: Path) -> tuple[int, int]:
    pred_path = out_dir / "predictions.csv"
    if not pred_path.exists():
        return (0, 0)
    preds = pd.read_csv(pred_path)
    if preds.empty:
        return (0, 0)
    one_model = preds[preds["model"] == "baseline_median"]
    if one_model.empty:
        one_model = preds[preds["model"] == preds["model"].iloc[0]]
    return (int(len(one_model)), int(one_model["participant_id"].nunique()))


def summarise_outcome(item: dict, out_dir: Path) -> dict:
    """Read one run's nested_summary.csv into a single comparison row."""
    row: dict = {
        "outcome": item["outcome"],
        "kind": item["kind"],
        "task": item["task"],
    }
    n_samples, n_participants = _n_samples_participants(out_dir)
    row["n_samples"] = n_samples
    row["n_participants"] = n_participants

    summary_path = out_dir / "nested_summary.csv"
    if not summary_path.exists():
        row["status"] = "missing_summary"
        return row
    summary = pd.read_csv(summary_path)
    if "model" not in summary.columns or "mae_mean" not in summary.columns:
        row["status"] = "bad_summary"
        return row

    baseline = summary[summary["model"] == "baseline_median"]
    baseline_mae = float(baseline["mae_mean"].iloc[0]) if len(baseline) else float("nan")

    models = summary[~summary["model"].str.startswith("baseline_")]
    if models.empty:
        row["status"] = "no_models"
        row["baseline_median_mae"] = baseline_mae
        return row
    # Only compare models that predicted on every fold; a partial-coverage model's
    # MAE is on a subset of samples and not comparable to the baseline.
    full = _full_coverage_models(out_dir)
    full_models = models[models["model"].isin(full)] if full else models
    dropped = int(len(models) - len(full_models))
    if full_models.empty:  # nothing covered all folds; report all but flag it
        full_models = models
        dropped = 0
    row["n_models_dropped_partial_coverage"] = dropped
    best = full_models.loc[full_models["mae_mean"].idxmin()]
    row["baseline_median_mae"] = baseline_mae
    row["best_model"] = str(best["model"])
    row["best_mae"] = float(best["mae_mean"])
    row["best_mae_std"] = float(best.get("mae_std", float("nan")))
    row["delta_vs_median"] = baseline_mae - float(best["mae_mean"])

    # Participant-level MAE (continuous targets only).
    part_path = out_dir / "participant_metrics.csv"
    if part_path.exists():
        part = pd.read_csv(part_path)
        if not part.empty and "participant_mae" in part.columns:
            row["best_participant_mae"] = float(part["participant_mae"].min())
    row["status"] = "ok"
    return row


def plot_mae_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    if not HAS_MPL:
        return
    df = comparison[comparison["status"] == "ok"].copy()
    if df.empty:
        return
    df = df.sort_values(["kind", "best_mae"])  # group sub-scores then zeno
    labels = df["outcome"].tolist()
    x = range(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 6))
    ax.bar(
        [i - width / 2 for i in x],
        df["baseline_median_mae"],
        width,
        label="baseline_median",
        color="lightgray",
    )
    ax.bar(
        [i + width / 2 for i in x],
        df["best_mae"],
        width,
        label="best model",
        color="steelblue",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAE (mean across folds)")
    ax.set_title("Multi-outcome: best model vs median baseline (MAE, lower is better)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="results/multi_outcome",
        help="Root directory for per-outcome runs and the aggregated comparison.",
    )
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tune", choices=["none", "grid"], default="none")
    parser.add_argument(
        "--only",
        choices=["all", "subscores", "zeno"],
        default="all",
        help="Restrict the sweep to a subset of outcomes.",
    )
    parser.add_argument(
        "--zeno-targets",
        nargs="*",
        default=None,
        help="Override the Zeno target list (default: all 16 mentor measures).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for subprocess runs (default: current).",
    )
    args = parser.parse_args()

    zeno_targets = args.zeno_targets if args.zeno_targets is not None else ZENO_TARGETS
    plan = build_plan(zeno_targets)
    if args.only == "subscores":
        plan = [p for p in plan if p["kind"] == "subscore"]
    elif args.only == "zeno":
        plan = [p for p in plan if p["kind"] == "zeno"]

    output_root = (PROJECT_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for item in plan:
        out_dir = output_root / item["outcome"]
        ok = run_one(
            item,
            out_dir,
            splits=args.splits,
            random_state=args.random_state,
            tune=args.tune,
            python_exe=args.python,
        )
        row = summarise_outcome(item, out_dir)
        if not ok and row.get("status") == "ok":
            row["status"] = "run_failed"
        rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison_path = output_root / "comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\n[Save] Comparison table -> {comparison_path}")
    print(comparison.to_string(index=False))

    plot_path = output_root / "mae_comparison.png"
    plot_mae_comparison(comparison, plot_path)
    if plot_path.exists():
        print(f"[Save] MAE comparison chart -> {plot_path}")


if __name__ == "__main__":
    main()
