#!/usr/bin/env python3
"""Dataset description for the paper (Project_discription.txt S0 + S7-1).

Answers the open question "what is n after the FW-only / one-visit-per-participant
definition?" and produces the distribution figures a methods section needs.
Writes to ``results/dataset_description/`` by default.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qixing_fga.config import STUDY_EXCLUDED_SAMPLE_IDS  # noqa: E402
from qixing_fga.data.loading import exclude_samples, load_merged_data  # noqa: E402
from qixing_fga.features.columns import MODEL_FEATURE_COLUMNS  # noqa: E402

TARGET = "fga_estimate_score"
SUBSCORES = [
    "speed",
    "imbalance",
    "gait_deviation",
    "deviation_outside_walkway",
    "assistive_device",
]


def _fga_counts(series: pd.Series, levels: list[int]) -> dict[str, int]:
    counts = series.value_counts().to_dict()
    return {str(lv): int(counts.get(lv, 0)) for lv in levels}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="data/2026_05_17_FWOnly_2visits_features.csv")
    ap.add_argument("--labels", default="data/2026_05_17_FWOnly_2visits_labels.csv")
    ap.add_argument("--output-dir", default="results/dataset_description")
    args = ap.parse_args()

    out = Path(args.output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    merged = load_merged_data(args.features, args.labels, "sample_id")
    # Same exclusion as every experiment, so the described cohort is the
    # analysed cohort (see qixing_fga.config.STUDY_EXCLUDED_SAMPLE_IDS).
    merged = exclude_samples(merged, STUDY_EXCLUDED_SAMPLE_IDS)
    levels = sorted(int(v) for v in merged[TARGET].dropna().unique())

    rows = []
    for name, df in [
        ("all_visits", merged),
        ("visit_1", merged[merged.video_index == 1]),
        ("visit_2", merged[merged.video_index == 2]),
    ]:
        row = {
            "view": name,
            "n_samples": len(df),
            "n_participants": int(df.participant_id.nunique()),
            "rows_per_participant": (
                "1" if df.participant_id.is_unique
                else str(sorted(df.groupby("participant_id").size().unique().tolist()))
            ),
            "fga_mean": round(float(df[TARGET].mean()), 3),
            "fga_sd": round(float(df[TARGET].std()), 3),
            "fga_median": float(df[TARGET].median()),
            "fga_missing": int(df[TARGET].isna().sum()),
        }
        row.update({f"fga_{k}": v for k, v in _fga_counts(df[TARGET], levels).items()})
        rows.append(row)
    desc = pd.DataFrame(rows)
    desc.to_csv(out / "dataset_description.csv", index=False)
    print(desc.to_string(index=False))

    # Missingness: features and label sub-scores.
    feat_cols = [c for c in MODEL_FEATURE_COLUMNS if c in merged.columns]
    miss = (
        merged[feat_cols + [c for c in SUBSCORES if c in merged.columns]]
        .isna()
        .sum()
        .rename("n_missing")
        .to_frame()
    )
    miss["pct_missing"] = (100.0 * miss.n_missing / len(merged)).round(2)
    miss = miss[miss.n_missing > 0].sort_values("n_missing", ascending=False)
    miss.to_csv(out / "missingness.csv")
    print(f"\nColumns with any missing value: {len(miss)} (of {len(feat_cols)} features "
          f"+ {len(SUBSCORES)} sub-scores)")
    if len(miss):
        print(miss.to_string())

    # Within-participant FGA change across visits (justifies not merging visits).
    piv = merged.pivot_table(index="participant_id", columns="video_index", values=TARGET)
    change = None
    if {1, 2}.issubset(set(piv.columns)):
        change = (piv[2] - piv[1]).dropna()
        change.rename("fga_visit2_minus_visit1").to_frame().to_csv(out / "visit_change.csv")
        print(f"\nFGA change visit2-visit1 over {len(change)} participants: "
              f"{change.value_counts().sort_index().to_dict()}")

    # --- Figures -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.38
    x = np.arange(len(levels))
    v1 = [int((merged[merged.video_index == 1][TARGET] == lv).sum()) for lv in levels]
    v2 = [int((merged[merged.video_index == 2][TARGET] == lv).sum()) for lv in levels]
    axes[0].bar(x - width / 2, v1, width, label=f"Visit 1 (n={sum(v1)})", color="#4C72B0")
    axes[0].bar(x + width / 2, v2, width, label=f"Visit 2 (n={sum(v2)})", color="#DD8452")
    axes[0].set_xticks(x, [str(lv) for lv in levels])
    axes[0].set_xlabel("FGA estimate score (0-3 ordinal)")
    axes[0].set_ylabel("Number of participants")
    axes[0].set_title("FGA distribution by visit (FW only, 1 sample/participant)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    if change is not None:
        vals = sorted(change.unique())
        cnt = [int((change == v).sum()) for v in vals]
        axes[1].bar([str(int(v)) for v in vals], cnt, color="#55A868")
        axes[1].set_xlabel("FGA(visit 2) - FGA(visit 1)")
        axes[1].set_ylabel("Number of participants")
        axes[1].set_title("Within-participant FGA change across visits")
        axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "fga_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Sub-score distributions (each is an ordinal FGA item).
    present = [c for c in SUBSCORES if c in merged.columns]
    if present:
        fig, axes = plt.subplots(1, len(present), figsize=(3.1 * len(present), 3.4))
        axes = np.atleast_1d(axes)
        v1df = merged[merged.video_index == 1]
        for ax, col in zip(axes, present):
            vc = v1df[col].astype(str).str.extract(r"^(\d+)", expand=False)
            counts = vc.value_counts().sort_index()
            ax.bar(counts.index, counts.to_numpy(), color="#4C72B0")
            ax.set_title(col, fontsize=9)
            ax.set_xlabel("level")
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel("Participants (visit 1)")
        fig.suptitle("FGA sub-score distributions (visit 1, n=%d)" % len(v1df), y=1.02)
        fig.tight_layout()
        fig.savefig(out / "figures" / "subscore_distributions.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    summary = {
        "n_participants_fw": int(merged.participant_id.nunique()),
        "n_rows_fw_all_visits": int(len(merged)),
        "n_rows_single_visit": int(merged.participant_id.nunique()),
        "fga_levels_observed": levels,
        "fga_scale_note": (
            "fga_estimate_score is a 0-3 ordinal clinician estimate, NOT the 0-30 "
            "FGA total. Verified against data/BW_gait_videos_DPT_review.xlsx "
            "('FGA score estimate', range 0-3)."
        ),
        "n_model_features": len(feat_cols),
        "n_feature_columns_with_missing": int((merged[feat_cols].isna().sum() > 0).sum()),
        "demographics_available": False,
        "demographics_note": (
            "Age/sex are not present in the released feature or label tables; "
            "they must be supplied from the clinical source to complete the "
            "dataset description."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[OK] Wrote dataset description to {out}")


if __name__ == "__main__":
    main()
