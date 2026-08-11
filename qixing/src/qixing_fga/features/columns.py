"""Feature column definitions and pool resolution (no leaky metadata)."""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Literal, Optional

# Never use in ML (paths, dates, QC flags, outcome labels — high leakage risk).
LEAKY_METADATA_COLUMNS = [
    "walking_condition",
    "dpt_date_folder",
    "landmark_date_folder",
    "date_match_strict",
    "landmark_file_path",
    "feature_status",
    "fga_estimate_score",
    "FGA_score",
    "mediapipe_world_file_path",
    "matched_subset",
    "label_video_date",
    "video_date",
    "source_excel_row",
    # Source-video frame rate: a property of the recording session, not of the
    # patient. It is 30 for 89 of the 91 FW samples and 26 for the remaining
    # two, so after standardisation those two rows sit at z = -6.6 and a linear
    # model can use the column as a near-identifier for them. It also only
    # became non-constant when the study moved to the double-visit definition,
    # so it was never a deliberate predictor.
    "fps",
]

# Model inputs only (gait / pose numerics).
MODEL_FEATURE_COLUMNS = [
    "arm_swing_amplitude_asymmetry",
    "arm_swing_peak_to_peak_asymmetry",
    "arm_swing_peak_to_peak_left",
    "arm_swing_peak_to_peak_right",
    "cadence_hz",
    "cadence_spm",
    "com_acceleration_mean",
    "com_acceleration_mean_log1p",
    "com_acceleration_std",
    "com_acceleration_std_log1p",
    "com_velocity_mean",
    "com_velocity_std",
    "dominant_frequency",
    "double_support_time_fraction",
    "elbow_angle_velocity_mean_left",
    "elbow_angle_velocity_mean_right",
    "elbow_angle_velocity_std_left",
    "elbow_angle_velocity_std_right",
    "foot_progression_angle_mean",
    "foot_progression_angle_std",
    "gait_speed_enhanced",
    "head_lateral_sway",
    "head_path_length",
    "head_rotation_speed_max",
    "head_rotation_speed_mean",
    "head_rotation_variability",
    "head_vertical_sway",
    "hip_angle_velocity_asymmetry_index",
    "hip_angle_velocity_mean_left",
    "hip_angle_velocity_mean_right",
    "hip_angle_velocity_std_left",
    "hip_angle_velocity_std_right",
    "jerk_magnitude",
    "knee_angle_velocity_asymmetry_index",
    "knee_angle_velocity_mean_left",
    "knee_angle_velocity_mean_right",
    "knee_angle_velocity_std_left",
    "knee_angle_velocity_std_right",
    "landmark_gap_filled_ratio",
    "landmark_large_gap_count",
    "landmark_mean_nan_ratio",
    "leg_step_length_asymmetry",
    "mean_gait_speed",
    "phase_locking_value",
    "power_3_7Hz",
    "shoulder_height_asymmetry",
    "stance_ratio_mean",
    "stance_time_mean",
    "step_events_left_count",
    "step_events_right_count",
    "step_length_asymmetry_index",
    "step_length_cv",
    "step_length_mean",
    "step_length_std",
    "step_time_asymmetry_index",
    "step_time_left_mean",
    "step_time_left_std",
    "step_time_mean",
    "step_time_right_mean",
    "step_time_right_std",
    "step_time_std",
    "step_time_variability",
    "stepping_symmetry_ratio",
    "stride_length_cv",
    "stride_length_mean",
    "stride_speed_mean",
    "stride_time_cv",
    "stride_time_mean",
    "stride_width_mean",
    "swing_time_mean",
    "tremor_cross_wrist_coherence",
    "tremor_intermittency",
    "tremor_regularity",
    "trunk_tilt_mean",
    "trunk_tilt_std",
    "wrist_tremor_amplitude",
]

# Backward-compatible alias (metadata must not appear here).
FEATURE_COLUMNS = list(MODEL_FEATURE_COLUMNS)
GAIT_NUMERIC_COLUMNS = list(MODEL_FEATURE_COLUMNS)

FeaturePoolMode = Literal["gait_numeric", "selected"]


def reject_leaky_columns(feature_cols: list[str], *, context: str = "") -> list[str]:
    """Drop or fail on metadata columns that must never enter training."""
    leaky_present = [c for c in feature_cols if c in LEAKY_METADATA_COLUMNS]
    if leaky_present:
        raise ValueError(
            f"Leaky metadata columns not allowed{': ' + context if context else ''}: "
            f"{leaky_present}"
        )
    return [c for c in feature_cols if c not in LEAKY_METADATA_COLUMNS]


def resolve_feature_columns(
    mode: FeaturePoolMode,
    *,
    selected: Optional[list[str]] = None,
) -> list[str]:
    if mode == "gait_numeric":
        return reject_leaky_columns(list(MODEL_FEATURE_COLUMNS), context="gait_numeric pool")
    if mode == "selected":
        if not selected:
            raise ValueError("feature pool mode 'selected' requires a feature list.")
        return reject_leaky_columns(list(selected), context="selected pool")
    raise ValueError(f"Unknown feature pool mode: {mode}")


def load_features_from_json(path: str | Path) -> list[str]:
    """Load feature names from JSON (list or {\"selected_features\": [...]})."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        names = [str(x) for x in payload]
    elif isinstance(payload, dict) and "selected_features" in payload:
        names = [str(x) for x in payload["selected_features"]]
    else:
        raise ValueError(
            f"{path}: expected a JSON list or object with 'selected_features' key."
        )
    return reject_leaky_columns(names, context=f"features json {path}")


def aggregate_top_features_by_frequency(
    fold_selected_features: dict[int, list[str]],
    top_k: int = 20,
) -> tuple[list[str], dict[str, int]]:
    """Pick top-k features by how often they appear across outer-fold RFECV selections."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    counts: Counter[str] = Counter()
    for features in fold_selected_features.values():
        counts.update(features)

    if not counts:
        raise ValueError("No features found in fold_selected_features.")

    ranked = counts.most_common()
    if len(ranked) < top_k:
        warnings.warn(
            f"Only {len(ranked)} unique features across folds; returning all of them."
        )
        top_k = len(ranked)

    top_items = ranked[:top_k]
    selected = [name for name, _ in top_items]
    frequencies = {name: int(freq) for name, freq in top_items}
    return selected, frequencies
