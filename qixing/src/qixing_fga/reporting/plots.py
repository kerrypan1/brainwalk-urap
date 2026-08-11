"""Prediction distribution / scatter figure export."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _progress(msg: str) -> None:
    print(msg, flush=True)


BASELINE_MODEL = "baseline_median"


def save_model_comparison_figure(
    summary: pd.DataFrame,
    per_fold: pd.DataFrame,
    output_dir: Path,
    *,
    target_name: str,
    delta_mae: pd.DataFrame | None = None,
    title: str | None = None,
    filename: str = "model_mae_comparison.png",
) -> Path | None:
    """Every model against the naive baseline, on one chart.

    The per-model prediction grids deliberately exclude baselines, so nothing in
    a run showed the comparison the results table is actually built on. Two
    panels: mean MAE with cross-fold SD, and the individual fold values — the
    second exists because at n=46 a mean can rest on one lucky fold, and an
    error bar alone hides that.

    Returns the written path, or None when the inputs cannot support the figure.
    """
    if summary is None or summary.empty or "mae_mean" not in summary.columns:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # matplotlib missing
        warnings.warn(f"Skipping model comparison figure (matplotlib: {exc})")
        return None

    # Nested-RFECV runs carry several feature variants; chart the primary one.
    df = summary
    if "feature_variant" in df.columns and df["feature_variant"].notna().any():
        variants = list(df["feature_variant"].dropna().unique())
        preferred = next(
            (v for v in ("nested_rfecv", "full", "selected") if v in variants),
            variants[0],
        )
        df = df[df["feature_variant"] == preferred]
        if title is None:
            title = f"{target_name} — {preferred}"
    if df.empty:
        return None

    order = df.sort_values("mae_mean", ascending=False)
    models = [str(m) for m in order["model"]]
    means = order["mae_mean"].to_numpy(dtype=float)
    sds = order["mae_std"].fillna(0).to_numpy(dtype=float) if "mae_std" in order else np.zeros(len(models))

    # Annotate with the reported statistic: ΔMAE and its bootstrap CI.
    deltas: dict[str, tuple[float, float, float]] = {}
    if delta_mae is not None and not delta_mae.empty:
        if {"delta_mae", "delta_mae_ci_lower", "delta_mae_ci_upper"} <= set(
            delta_mae.columns
        ):
            deltas = {
                str(r["model"]): (
                    float(r["delta_mae"]),
                    float(r["delta_mae_ci_lower"]),
                    float(r["delta_mae_ci_upper"]),
                )
                for _, r in delta_mae.iterrows()
            }

    base_rows = order.loc[order["model"] == BASELINE_MODEL, "mae_mean"]
    base_mae = float(base_rows.iloc[0]) if len(base_rows) else float("nan")

    labels = []
    for m in models:
        if m in deltas:
            d, lo, hi = deltas[m]
            star = "*" if hi < 0 else ""
            labels.append(f"{m}\n(Δ={d:+.3f} [{lo:+.3f},{hi:+.3f}]{star})")
        else:
            labels.append(m)
    colors = ["#C44E52" if m.startswith("baseline_") else "#4C72B0" for m in models]

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    has_folds = per_fold is not None and not per_fold.empty and "mae" in per_fold.columns
    ncols = 2 if has_folds else 1
    fig, axes = plt.subplots(
        1, ncols,
        figsize=(14 if has_folds else 8.5, 5.4),
        gridspec_kw={"width_ratios": [1.25, 1]} if has_folds else None,
        squeeze=False,
    )
    ax0 = axes[0][0]

    y = np.arange(len(models))
    ax0.barh(y, means, xerr=sds, color=colors, capsize=4, error_kw={"alpha": 0.55})
    ax0.set_yticks(y, labels, fontsize=9)
    if np.isfinite(base_mae):
        ax0.axvline(base_mae, ls="--", color="#C44E52", lw=1.2,
                    label=f"median baseline ({base_mae:.3f})")
        ax0.legend(loc="upper right", fontsize=9)
    for i, (m, v) in enumerate(zip(models, means)):
        x = v + sds[i] + 0.03 * max(means.max(), 1e-9)
        if np.isfinite(base_mae) and base_mae > 0 and not m.startswith("baseline_"):
            pct = 100.0 * (base_mae - v) / base_mae
            ax0.text(x, i, f"{v:.3f}  ({pct:+.0f}%)", va="center", fontsize=8.5)
        else:
            ax0.text(x, i, f"{v:.3f}", va="center", fontsize=8.5)
    ax0.set_xlim(0, float((means + sds).max()) * 1.45)
    ax0.set_xlabel("MAE (mean ± SD across outer folds)")
    ax0.set_title("Model comparison — lower is better\n"
                  "* = 95% CI on ΔMAE excludes zero", fontsize=10)
    ax0.grid(axis="x", alpha=0.3)

    if has_folds:
        ax1 = axes[0][1]
        pf = per_fold
        if "feature_variant" in pf.columns and "feature_variant" in summary.columns:
            keep = set(df["feature_variant"].dropna().unique())
            if keep:
                pf = pf[pf["feature_variant"].isin(keep)]
        rng = np.random.default_rng(42)
        for i, m in enumerate(models):
            vals = pf.loc[pf["model"].astype(str) == m, "mae"].to_numpy(dtype=float)
            if not len(vals):
                continue
            jitter = rng.uniform(-0.13, 0.13, size=len(vals))
            ax1.scatter(vals, np.full(len(vals), i) + jitter,
                        color=colors[i], alpha=0.85, s=42, zorder=3)
            ax1.plot([vals.mean()], [i], marker="|", color="black",
                     markersize=18, markeredgewidth=2, zorder=4)
        ax1.set_yticks(y, models, fontsize=9)
        if np.isfinite(base_mae):
            ax1.axvline(base_mae, ls="--", color="#C44E52", lw=1.2)
        ax1.set_xlabel("MAE on each outer fold")
        ax1.set_title("Per-fold spread (| = mean)", fontsize=10)
        ax1.grid(axis="x", alpha=0.3)

    fig.suptitle(title or target_name, y=1.02, fontsize=12)
    fig.tight_layout()
    out_path = figures_dir / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _progress(f"[Save] Model comparison figure → {out_path.resolve()}")
    return out_path


def save_prediction_figures(
    predictions: pd.DataFrame,
    output_dir: Path,
    *,
    n_classes: int,
    target_name: str,
    random_state: int = 42,
    continuous: bool = False,
) -> None:
    """Save true-vs-predicted distribution and scatter figures per model.

    Uses out-of-fold predictions. Baselines are excluded from the plots (they
    are trivial references). Silently skips if matplotlib is unavailable. For
    continuous targets the class-distribution grid is skipped and the scatter
    uses data-driven axis limits.
    """
    if predictions is None or predictions.empty:
        return
    try:
        from utils.plotting import (
            plot_prediction_distribution_grid,
            plot_prediction_scatter_grid,
        )
    except Exception as exc:  # matplotlib missing or import error
        warnings.warn(f"Skipping figures (plotting import failed): {exc}")
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = predictions[~predictions["model"].str.startswith("baseline_")]
    variants = list(df["feature_variant"].dropna().unique()) or ["full"]
    for variant in variants:
        sub = df[df["feature_variant"] == variant]
        per_model: dict[str, dict[str, np.ndarray]] = {}
        for model_name, grp in sub.groupby("model"):
            entry = {
                "y_true": grp["y_true"].to_numpy(),
                "y_pred": grp["y_pred"].to_numpy(),
            }
            # Carried so the figure titles can report the fold mean, matching the
            # results table rather than a pooled mean over all out-of-fold rows.
            if "fold" in grp.columns:
                entry["fold"] = grp["fold"].to_numpy()
            per_model[str(model_name)] = entry
        if not per_model:
            continue
        suffix = "" if variant in ("full", None) else f"_{variant}"
        if not continuous:
            plot_prediction_distribution_grid(
                per_model,
                str(figures_dir / f"{target_name}_pred_true_distribution{suffix}.png"),
                n_classes=n_classes,
                target_name=target_name,
            )
            # Detailed per-model figure: one true-vs-predicted distribution chart
            # per model (the combined grid above crams them together). Reuses the
            # same plotting fn with a single-model dict.
            per_model_dir = figures_dir / "per_model"
            per_model_dir.mkdir(parents=True, exist_ok=True)
            for model_name, data in per_model.items():
                safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(model_name))
                plot_prediction_distribution_grid(
                    {model_name: data},
                    str(
                        per_model_dir
                        / f"{target_name}_{safe}_pred_true_distribution{suffix}.png"
                    ),
                    n_classes=n_classes,
                    target_name=target_name,
                )
        plot_prediction_scatter_grid(
            per_model,
            str(figures_dir / f"{target_name}_pred_vs_true_scatter{suffix}.png"),
            n_classes=n_classes,
            target_name=target_name,
            random_state=random_state,
            continuous=continuous,
        )
    _progress(f"[Save] Prediction figures → {figures_dir.resolve()}")
