"""CLI entry point for batch / nested-CV training."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

# src/qixing_fga/cli/train.py -> parents[3] = project root (parent of src)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ..config import apply_overrides, load_config
from ..cv.splits import (
    build_fold_indices_from_bootstrap,
    continuous_effective_n_splits,
    effective_n_splits,
)
from ..data.loading import (
    exclude_samples,
    load_merged_data,
    load_single_file,
    prepare_features,
    select_visit,
)
from ..evaluation.bootstrap import load_teammate_bootstrap_npz
from ..evaluation.evaluate import (
    evaluate_models,
    print_nested_compare,
    print_overfitting_diagnosis,
)
from ..evaluation.metrics import (
    paired_bootstrap_delta_mae,
    participant_level_summary,
)
from ..features.columns import (
    LEAKY_METADATA_COLUMNS,
    load_features_from_json,
    resolve_feature_columns,
)
from ..models.registry import HAS_XGBOOST, build_models_for_task
from ..preprocessing import build_model_preprocessor
from ..reporting.artifacts import (
    make_run_dir,
    save_results_to_dir,
    write_config,
    write_environment,
    write_feature_importance,
    write_metrics,
    write_predictions,
    write_provenance,
)
from ..reporting.plots import save_model_comparison_figure, save_prediction_figures
from ..selection.rfecv import run_nested_cv_rfecv


def _progress(msg: str) -> None:
    """Print progress line immediately (for long-running nested CV)."""
    print(msg, flush=True)


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


DEFAULT_FEATURE_PATH = "data/2026_05_17_FWOnly_2visits_features.csv"
DEFAULT_LABEL_PATH = "data/2026_05_17_FWOnly_2visits_labels.csv"
DEFAULT_TARGET_COL = "fga_estimate_score"
DEFAULT_ID_COL = "sample_id"
DEFAULT_SPLITS = 5
DEFAULT_BOOTSTRAP_NPZ = "data/bootstrap_indices.npz"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch training pipeline for FGA estimate score classification."
    )
    parser.add_argument("--features", default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--labels", default=DEFAULT_LABEL_PATH)
    parser.add_argument(
        "--data-file",
        default=None,
        help=(
            "Single wide table (features + target + participant_id). Skips the "
            "features/labels merge; a sample_id is synthesised per row. Use for "
            "continuous outcomes (e.g. Zeno walkway) with --task regression_continuous."
        ),
    )
    parser.add_argument(
        "--visit",
        default=None,
        help=(
            "Keep only this video_index (one sample per participant). Default "
            "'all' keeps every visit. Use 1 for the finalised FW-only, "
            "single-visit data definition."
        ),
    )
    parser.add_argument("--target", default=DEFAULT_TARGET_COL)
    parser.add_argument("--id-col", default=DEFAULT_ID_COL)
    parser.add_argument("--splits", type=int, default=DEFAULT_SPLITS)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "regression_continuous"],
        default=None,
        help=(
            "classification: label models + rounded predictions. "
            "regression: regression/ordinal models with MAE computed on continuous "
            "predictions and integer (ordinal) targets; StratifiedGroupKFold. "
            "regression_continuous: float targets (no rounding), GroupKFold CV, "
            "MAE/MSE/RMSE only (accuracy/f1 = NaN), plus participant-level metrics. "
            "Default: the config's task (DEFAULT_CONFIG uses 'classification'). "
            "Must stay None here so a --config's task is not silently overridden."
        ),
    )
    parser.add_argument(
        "--tune",
        choices=["none", "grid"],
        default="none",
        help=(
            "grid: inner group-aware GridSearchCV (neg-MAE) selects regularisation "
            "hyperparameters per fold on outer/train data only. Default none keeps "
            "fixed hyperparameters (reproducible baseline)."
        ),
    )
    parser.add_argument(
        "--tune-splits",
        type=int,
        default=3,
        help="Inner CV folds used by --tune grid (participant-grouped).",
    )
    parser.add_argument("--per-fold-bootstrap-npz", default=DEFAULT_BOOTSTRAP_NPZ)
    parser.add_argument("--ci-alpha", type=float, default=0.05)
    parser.add_argument(
        "--participant-aggregation",
        action="store_true",
        help=(
            "Also emit an optional participant-level MAE/RMSE diagnostic (average "
            "predictions per participant, then score). OFF by default: the reported "
            "evaluation metric is sample-level (each visit scored against its own "
            "label). Participant averaging conflates visits with different labels."
        ),
    )
    # Deprecated: sample-level is now the default, so this flag is a no-op.
    # Kept so existing callers (run_top*/run_multi_outcome) don't break.
    parser.add_argument(
        "--no-participant-aggregation",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config (configs/*.yaml). CLI flags override config values.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment name for experiments/<name>/<run_id>/ artifact directory.",
    )
    parser.add_argument("--force-team-alignment", action="store_true")
    parser.add_argument(
        "--use-bootstrap-folds",
        action="store_true",
        help="Infer fold splits from the bootstrap npz participant indices.",
    )
    parser.add_argument(
        "--diagnose-overfitting",
        action="store_true",
        help="Report train vs validation gap metrics (auto-enabled with --feature-selection rfecv).",
    )
    parser.add_argument(
        "--feature-selection",
        choices=["none", "rfecv"],
        default="none",
        help=(
            "rfecv: nested CV (outer=bootstrap folds, inner=RFECV on outer-train only). "
            "Requires --use-bootstrap-folds."
        ),
    )
    parser.add_argument(
        "--inner-splits",
        type=int,
        default=None,
        help="Inner CV folds for RFECV within each outer fold (default: same as --splits).",
    )
    parser.add_argument(
        "--rfecv-min-features",
        type=int,
        default=5,
        help="Minimum features retained by RFECV.",
    )
    parser.add_argument(
        "--rfecv-step",
        type=int,
        default=1,
        help="Number of features removed per RFECV iteration.",
    )
    parser.add_argument(
        "--gap-ratio-warn",
        type=float,
        default=0.25,
        help="Warn when mean mae_gap_ratio exceeds this threshold.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save CSV/JSON outputs.",
    )
    parser.add_argument(
        "--top-k-by-frequency",
        type=int,
        default=20,
        help="After nested RFECV, keep top-K features by outer-fold selection frequency.",
    )
    parser.add_argument(
        "--features-json",
        default=None,
        help=(
            "Use a fixed feature list from JSON (e.g. "
            "selected_features_top20_by_frequency.json). "
            "Skips RFECV; use with --feature-selection none."
        ),
    )
    args = parser.parse_args()

    # YAML config (optional) + CLI overrides. Resolved config is saved with the run.
    cfg = load_config(args.config)
    cfg = apply_overrides(
        cfg,
        {
            "experiment_name": args.experiment_name,
            "random_seed": args.random_state,
            "task": args.task,
            "data.features": args.features if args.features != DEFAULT_FEATURE_PATH else None,
            "data.labels": args.labels if args.labels != DEFAULT_LABEL_PATH else None,
            "data.data_file": args.data_file,
            "data.visit": args.visit,
            "data.target": args.target if args.target != DEFAULT_TARGET_COL else None,
            "data.id_col": args.id_col if args.id_col != DEFAULT_ID_COL else None,
            "cv.n_splits": args.splits if args.splits != DEFAULT_SPLITS else None,
            "cv.bootstrap_npz": args.per_fold_bootstrap_npz,
            "cv.use_bootstrap_folds": args.use_bootstrap_folds or None,
            "model.tune": args.tune if args.tune != "none" else None,
            "model.tune_splits": args.tune_splits if args.tune_splits != 3 else None,
            "feature_selection.method": args.feature_selection
            if args.feature_selection != "none"
            else None,
            "feature_selection.rfecv_min_features": args.rfecv_min_features,
            "feature_selection.rfecv_step": args.rfecv_step,
            "feature_selection.features_json": args.features_json,
            "reporting.ci_alpha": args.ci_alpha,
            "reporting.participant_level": args.participant_aggregation or None,
        },
    )
    # Propagate resolved config back onto args for the rest of main().
    args.random_state = int(cfg["random_seed"])
    args.task = cfg["task"]
    args.features = cfg["data"]["features"]
    args.labels = cfg["data"]["labels"]
    if cfg["data"].get("data_file"):
        args.data_file = cfg["data"]["data_file"]
    args.target = cfg["data"]["target"]
    args.id_col = cfg["data"]["id_col"]
    args.visit = cfg["data"].get("visit", "all")
    excluded_sample_ids = list(cfg["data"].get("exclude_sample_ids") or [])
    args.splits = int(cfg["cv"]["n_splits"])
    args.per_fold_bootstrap_npz = cfg["cv"]["bootstrap_npz"]
    if cfg["cv"].get("use_bootstrap_folds"):
        args.use_bootstrap_folds = True
    args.tune = cfg["model"]["tune"]
    args.tune_splits = int(cfg["model"]["tune_splits"])
    if cfg["feature_selection"]["method"] != "none" and args.feature_selection == "none":
        args.feature_selection = cfg["feature_selection"]["method"]
    args.rfecv_min_features = int(cfg["feature_selection"]["rfecv_min_features"])
    args.rfecv_step = int(cfg["feature_selection"]["rfecv_step"])
    if cfg["feature_selection"].get("features_json") and not args.features_json:
        args.features_json = cfg["feature_selection"]["features_json"]
    args.ci_alpha = float(cfg["reporting"]["ci_alpha"])
    report_participant_level = bool(cfg["reporting"].get("participant_level", False))
    if args.participant_aggregation:
        report_participant_level = True
    experiment_name = cfg.get("experiment_name") or "fga"

    if args.features_json and args.feature_selection == "rfecv":
        raise ValueError(
            "Use either --features-json (fixed list) or --feature-selection rfecv, not both."
        )

    diagnose_overfitting = (
        args.diagnose_overfitting or args.feature_selection == "rfecv"
    )
    inner_splits = args.inner_splits if args.inner_splits is not None else args.splits

    if args.force_team_alignment:
        default_npz = Path(DEFAULT_BOOTSTRAP_NPZ).resolve()
        requested_npz = Path(args.per_fold_bootstrap_npz).resolve()
        if args.splits != DEFAULT_SPLITS:
            raise ValueError(
                "--force-team-alignment disallows custom --splits; use default."
            )
        if requested_npz != default_npz:
            raise ValueError(
                "--force-team-alignment disallows custom --per-fold-bootstrap-npz; use default."
            )
        args.use_bootstrap_folds = True
        cfg["cv"]["use_bootstrap_folds"] = True
        cfg["cv"]["strategy"] = "bootstrap_aligned"

    if report_participant_level:
        _progress(
            "[Config] Reporting sample-level metrics (primary); participant-level "
            "MAE/RMSE also emitted as an opt-in secondary diagnostic."
        )
    else:
        _progress(
            "[Config] Reporting sample-level metrics: per-fold MAE (mean +/- SD across "
            "folds), each visit scored against its own label. Pass "
            "--participant-aggregation to also emit the participant-mean diagnostic."
        )

    is_continuous = args.task == "regression_continuous"
    if args.data_file and args.feature_selection == "rfecv":
        raise ValueError(
            "--data-file single-file mode does not support --feature-selection rfecv."
        )
    if is_continuous and args.feature_selection == "rfecv":
        raise ValueError(
            "--task regression_continuous does not support nested RFECV; "
            "use --feature-selection none."
        )

    t_run = time.perf_counter()
    if args.data_file:
        _progress("[Stage 1/5] Loading single wide table (features + target)...")
        merged = load_single_file(args.data_file)
    else:
        _progress("[Stage 1/5] Loading and merging feature/label CSVs...")
        merged = load_merged_data(args.features, args.labels, args.id_col)
        # Drop unusable recordings before the visit filter, so the ids are
        # validated against every row rather than the current visit's subset.
        n_all = len(merged)
        merged = exclude_samples(merged, excluded_sample_ids)
        if len(merged) != n_all:
            _progress(
                f"  [Excluded] {excluded_sample_ids}: {n_all} → {len(merged)} rows"
            )
    n_before = len(merged)
    merged = select_visit(merged, args.visit)
    if len(merged) != n_before:
        _progress(
            f"  [Visit filter] video_index=={args.visit}: {n_before} → {len(merged)} rows"
        )
    _progress(
        f"  → {len(merged)} rows "
        f"({merged['participant_id'].nunique() if 'participant_id' in merged.columns else '?'} participants)"
    )

    if "participant_id" not in merged.columns:
        raise ValueError("participant_id column is required for group CV.")

    output_dir = Path(args.output_dir) if args.output_dir else None

    _progress("[Stage 2/5] Preparing model features (metadata columns excluded)...")
    if args.features_json:
        feature_cols = load_features_from_json(args.features_json)
        _progress(
            f"  Using {len(feature_cols)} features from {args.features_json}"
        )
    else:
        feature_cols = resolve_feature_columns("gait_numeric")
    X, y, missing_cols, valid_mask = prepare_features(
        merged, feature_cols, [], args.target, task=args.task
    )
    groups = merged.loc[valid_mask, "participant_id"].astype(str).reset_index(drop=True)
    if "sample_id" in merged.columns:
        ids = merged.loc[valid_mask, "sample_id"].astype(str).reset_index(drop=True)
    else:
        ids = groups
    _progress(
        f"  → {len(X)} labeled samples, {len(X.columns)} features, "
        f"{groups.nunique()} participants, "
        + ("continuous target" if is_continuous else f"{y.nunique()} classes")
    )

    if is_continuous:
        n_splits = continuous_effective_n_splits(groups, args.splits)
        _progress(f"  → GroupKFold with {n_splits} folds (continuous target)")
    elif args.use_bootstrap_folds:
        # Folds are predefined by the teammate npz, so class-count feasibility
        # for StratifiedGroupKFold does not apply: no stratified split is drawn.
        # Requiring it would block the one-sample-per-participant view (rarest
        # FGA class has 4 members at n=46) even though the folds already exist.
        n_splits = args.splits
        _progress(f"  → {n_splits} predefined folds from bootstrap npz (no stratification needed)")
    else:
        try:
            n_splits = effective_n_splits(y, args.splits)
        except ValueError:
            # A class is too small to stratify; fall back to participant-grouped
            # folds sized by participant count (get_cv_splits also degrades).
            n_splits = continuous_effective_n_splits(groups, args.splits)
            warnings.warn(
                "A class is too small to stratify; using GroupKFold with "
                f"{n_splits} folds."
            )
        if (
            n_splits != args.splits
            and args.use_bootstrap_folds
            and args.per_fold_bootstrap_npz
        ):
            raise ValueError(
                "Bootstrap npz requires the original split count; "
                "reduce splits or regenerate the npz."
            )

    if args.feature_selection == "rfecv" and not args.use_bootstrap_folds:
        raise ValueError(
            "Nested RFECV requires --use-bootstrap-folds (outer CV aligned with "
            "teammate bootstrap_indices.npz). Use --force-team-alignment."
        )

    bootstrap_by_fold = None
    fold_indices = None
    if args.use_bootstrap_folds:
        _progress(
            f"[Stage 3/5] Loading teammate bootstrap folds from "
            f"{args.per_fold_bootstrap_npz}..."
        )
        if not args.per_fold_bootstrap_npz:
            raise ValueError("--use-bootstrap-folds requires --per-fold-bootstrap-npz")
        bootstrap_by_fold = load_teammate_bootstrap_npz(
            args.per_fold_bootstrap_npz, n_splits=n_splits
        )
        fold_indices = build_fold_indices_from_bootstrap(
            groups, bootstrap_by_fold, n_splits
        )
        _progress(f"  → {n_splits} outer folds aligned with bootstrap npz")
    elif args.feature_selection != "rfecv" and not is_continuous:
        warnings.warn(
            "Without --use-bootstrap-folds, outer CV uses StratifiedGroupKFold "
            "(not teammate-aligned)."
        )

    num_classes = 0 if is_continuous else int(y.nunique())
    models = build_models_for_task(args.task, num_classes, args.random_state)
    if not models:
        raise ValueError("No models available to train.")
    tune_note = (
        f"grid search (inner {args.tune_splits}-fold, neg-MAE)"
        if args.tune == "grid"
        else "fixed hyperparameters"
    )
    _progress(
        f"[Stage 4/5] Models ready ({args.task}, {tune_note}): "
        f"{', '.join(models.keys())}"
    )

    nested_results: Optional[pd.DataFrame] = None
    nested_summary: Optional[pd.DataFrame] = None
    inner_ranking_df: Optional[pd.DataFrame] = None
    fold_selected: Optional[dict[int, list[str]]] = None
    nested_compare_df: Optional[pd.DataFrame] = None
    predictions_df: Optional[pd.DataFrame] = None

    if args.feature_selection == "rfecv":
        if fold_indices is None:
            raise ValueError("Nested RFECV requires bootstrap-aligned outer folds.")

        _progress(
            f"\n=== Nested CV (outer={n_splits} bootstrap folds, "
            f"inner={inner_splits} SGKF + RFECV) ==="
        )
        _progress(
            f"  Models: {', '.join(models.keys())} | "
            f"RFECV step={args.rfecv_step}, min_features={args.rfecv_min_features}"
        )
        _progress(
            f"  Feature pool: {len(X.columns)} gait/pose columns "
            f"(metadata blocked: {len(LEAKY_METADATA_COLUMNS)} cols)"
        )

        nested_results, nested_summary, inner_ranking_df, fold_selected, predictions_df = (
            run_nested_cv_rfecv(
                X,
                y,
                groups,
                models,
                fold_indices,
                inner_splits,
                args.random_state,
                args.rfecv_min_features,
                args.rfecv_step,
                bootstrap_by_fold,
                args.ci_alpha,
                task=args.task,
                tune=args.tune,
                tune_splits=args.tune_splits,
                ids=ids,
            )
        )

        _progress("\n[Stage 5/5] Nested CV complete — printing and saving results...")
        print("\nPer-fold nested results:")
        print(nested_results.to_string(index=False))
        print("\nNested summary (mean +/- std across outer folds):")
        print(nested_summary.to_string(index=False))

        if diagnose_overfitting:
            for variant in ("nested_rfecv", "nested_all_gait"):
                sub = nested_summary[nested_summary["feature_variant"] == variant]
                if len(sub):
                    print(f"\n--- Overfitting diagnosis ({variant}) ---")
                    print_overfitting_diagnosis(
                        sub.rename(columns={"model": "model"}),
                        args.gap_ratio_warn,
                    )

        nested_compare_df = print_nested_compare(nested_summary, y, y.nunique())

    else:
        preprocessor = build_model_preprocessor(X)
        results, summary, predictions_df = evaluate_models(
            X,
            y,
            groups,
            preprocessor,
            models,
            n_splits,
            args.random_state,
            fold_indices,
            bootstrap_by_fold,
            args.ci_alpha,
            task=args.task,
            tune=args.tune,
            tune_splits=args.tune_splits,
            ids=ids,
            feature_variant="selected" if args.features_json else "full",
        )

        if missing_cols:
            print("Missing feature columns:")
            for col in missing_cols:
                print(f"- {col}")

        print(f"\nModel features: {len(X.columns)} (no metadata columns)")
        print("\nPer-fold results:")
        print(results.to_string(index=False))
        print("\nSummary (mean +/- std):")
        print(summary.to_string(index=False))

        if diagnose_overfitting:
            print_overfitting_diagnosis(summary, args.gap_ratio_warn)

        nested_results = results
        nested_summary = summary

    if missing_cols:
        print("\nMissing feature columns:")
        for col in missing_cols:
            print(f"- {col}")

    # Evidence vs the naive baseline, computed on this run's own OOF predictions
    # so no result leaves the pipeline without the numbers behind it.
    delta_mae_df: Optional[pd.DataFrame] = None
    if predictions_df is not None and not predictions_df.empty:
        delta_mae_df = paired_bootstrap_delta_mae(
            predictions_df, ci_alpha=args.ci_alpha, random_state=args.random_state
        )
        if delta_mae_df is not None and not delta_mae_df.empty:
            print(
                f"\nDelta-MAE vs baseline_median, paired bootstrap "
                f"{int(100 * (1 - args.ci_alpha))}% CI (participants resampled):"
            )
            print(delta_mae_df.to_string(index=False))

    participant_summary_df: Optional[pd.DataFrame] = None
    if report_participant_level and predictions_df is not None:
        model_preds = predictions_df[
            ~predictions_df["model"].str.startswith("baseline_")
        ]
        participant_summary_df = participant_level_summary(model_preds)
        if not participant_summary_df.empty:
            print("\nParticipant-level metrics (out-of-fold, averaged per participant):")
            print(participant_summary_df.to_string(index=False))

    # Resolve output directory: explicit --output-dir or experiments/<name>/<run_id>/.
    if output_dir is None:
        output_dir = make_run_dir(experiment_name, base=str(PROJECT_ROOT / "experiments"))
        _progress(f"[Save] No --output-dir; writing artifacts to {output_dir}")

    if output_dir is not None:
        _progress(f"[Save] Writing outputs to {output_dir}...")
        save_results_to_dir(
            output_dir,
            results=nested_results,
            summary=nested_summary,
            inner_rfecv_ranking=inner_ranking_df,
            fold_selected_features=fold_selected,
            top_k_by_frequency=args.top_k_by_frequency,
            nested_compare=nested_compare_df,
            predictions=predictions_df,
        )
        # Publication artifacts (config / metrics / provenance / environment).
        write_config(Path(output_dir), cfg)
        metrics_payload = {
            "summary": nested_summary.to_dict(orient="records")
            if nested_summary is not None
            else [],
            "participant_level": participant_summary_df.to_dict(orient="records")
            if participant_summary_df is not None and not participant_summary_df.empty
            else [],
            "delta_mae_vs_baseline": delta_mae_df.to_dict(orient="records")
            if delta_mae_df is not None and not delta_mae_df.empty
            else [],
        }
        write_metrics(Path(output_dir), metrics_payload)
        write_predictions(Path(output_dir), predictions_df)
        if fold_selected is not None:
            # Frequency table as feature_importance.csv
            from collections import Counter

            counts: Counter[str] = Counter()
            for feats in fold_selected.values():
                counts.update(feats)
            n_folds = max(len(fold_selected), 1)
            fi = pd.DataFrame(
                [
                    {
                        "feature": name,
                        "n_folds_selected": freq,
                        "frequency": freq / n_folds,
                    }
                    for name, freq in counts.most_common()
                ]
            )
            write_feature_importance(Path(output_dir), fi)
        if participant_summary_df is not None and not participant_summary_df.empty:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            participant_summary_df.to_csv(
                Path(output_dir) / "participant_metrics.csv", index=False
            )
        dataset_paths = []
        if args.data_file:
            dataset_paths.append(str(args.data_file))
        else:
            dataset_paths.extend([str(args.features), str(args.labels)])
        write_provenance(
            Path(output_dir),
            random_seed=args.random_state,
            dataset_paths=dataset_paths,
            repo_dir=str(PROJECT_ROOT),
        )
        write_environment(Path(output_dir))
        if delta_mae_df is not None and not delta_mae_df.empty:
            delta_mae_df.to_csv(Path(output_dir) / "delta_mae_ci.csv", index=False)
        save_prediction_figures(
            predictions_df,
            output_dir,
            n_classes=1 if is_continuous else int(y.max()) + 1,
            target_name=args.target,
            random_state=args.random_state,
            continuous=is_continuous,
        )
        save_model_comparison_figure(
            nested_summary,
            nested_results,
            Path(output_dir),
            target_name=args.target,
            delta_mae=delta_mae_df,
            title=f"{args.target} — {experiment_name}",
        )

    _progress(
        f"\n[Done] Total runtime {_format_elapsed(time.perf_counter() - t_run)}"
    )

    if not HAS_XGBOOST:
        print("\nXGBoost not available. Install with: pip install xgboost")


if __name__ == "__main__":
    main()
