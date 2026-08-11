"""Feature groups for the ablation study (Project_discription.txt S4).

Groups are defined by **what the measurement means clinically**, not by name
prefix, so that removing a group answers a question a clinician would ask
("does the model still work without any variability measure?") rather than an
arbitrary string-matching question.

The groups must partition ``MODEL_FEATURE_COLUMNS`` exactly — no feature in two
groups, none left out — otherwise leave-one-group-out and group-only results
would not be comparable. ``validate_groups`` enforces this and is exercised by
``tests/test_feature_groups.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from qixing_fga.features.columns import MODEL_FEATURE_COLUMNS  # noqa: E402

FEATURE_GROUPS: dict[str, list[str]] = {
    # Pace / rhythm / distance: the classic spatiotemporal gait parameters.
    "pace_rhythm": [
        "cadence_hz",
        "cadence_spm",
        "gait_speed_enhanced",
        "mean_gait_speed",
        "stride_speed_mean",
        "step_length_mean",
        "stride_length_mean",
        "stride_width_mean",
        "step_time_mean",
        "step_time_left_mean",
        "step_time_right_mean",
        "stride_time_mean",
        "stance_time_mean",
        "stance_ratio_mean",
        "swing_time_mean",
        "double_support_time_fraction",
        "step_events_left_count",
        "step_events_right_count",
        "dominant_frequency",
        "foot_progression_angle_mean",
    ],
    # Stride-to-stride inconsistency (SD / CV style measures).
    "variability": [
        "step_length_cv",
        "step_length_std",
        "step_time_std",
        "step_time_left_std",
        "step_time_right_std",
        "step_time_variability",
        "stride_length_cv",
        "stride_time_cv",
        "foot_progression_angle_std",
        "com_acceleration_std",
        "com_acceleration_std_log1p",
        "com_velocity_std",
        "head_rotation_variability",
        "trunk_tilt_std",
    ],
    # Left/right imbalance and inter-limb coordination.
    "symmetry": [
        "arm_swing_amplitude_asymmetry",
        "arm_swing_peak_to_peak_asymmetry",
        "hip_angle_velocity_asymmetry_index",
        "knee_angle_velocity_asymmetry_index",
        "leg_step_length_asymmetry",
        "shoulder_height_asymmetry",
        "step_length_asymmetry_index",
        "step_time_asymmetry_index",
        "stepping_symmetry_ratio",
        "phase_locking_value",
    ],
    # Whole-body stability: trunk / head / centre-of-mass motion.
    "postural": [
        "com_acceleration_mean",
        "com_acceleration_mean_log1p",
        "com_velocity_mean",
        "head_lateral_sway",
        "head_path_length",
        "head_rotation_speed_max",
        "head_rotation_speed_mean",
        "head_vertical_sway",
        "trunk_tilt_mean",
        "jerk_magnitude",
        "power_3_7Hz",
    ],
    # Arm swing, elbow motion and wrist tremor.
    "upper_limb": [
        "arm_swing_peak_to_peak_left",
        "arm_swing_peak_to_peak_right",
        "elbow_angle_velocity_mean_left",
        "elbow_angle_velocity_mean_right",
        "elbow_angle_velocity_std_left",
        "elbow_angle_velocity_std_right",
        "tremor_cross_wrist_coherence",
        "tremor_intermittency",
        "tremor_regularity",
        "wrist_tremor_amplitude",
    ],
    # Lower-limb joint angular velocities.
    "joint_kinematics": [
        "hip_angle_velocity_mean_left",
        "hip_angle_velocity_mean_right",
        "hip_angle_velocity_std_left",
        "hip_angle_velocity_std_right",
        "knee_angle_velocity_mean_left",
        "knee_angle_velocity_mean_right",
        "knee_angle_velocity_std_left",
        "knee_angle_velocity_std_right",
    ],
    # Acquisition quality, not gait. Kept as its own group precisely so the
    # ablation can show whether the model leans on recording artefacts.
    # ``fps`` used to sit here; it is now rejected as leaky metadata (see
    # features/columns.py), which leaves this group entirely constant on the
    # current data — so its ablation rows are vacuous rather than evidence.
    "acquisition_quality": [
        "landmark_gap_filled_ratio",
        "landmark_large_gap_count",
        "landmark_mean_nan_ratio",
    ],
}

GROUP_LABELS: dict[str, str] = {
    "pace_rhythm": "Spatiotemporal (speed / rhythm / step length)",
    "variability": "Variability",
    "symmetry": "Symmetry and coordination",
    "postural": "Postural stability (trunk / head / COM)",
    "upper_limb": "Upper limb and tremor",
    "joint_kinematics": "Lower-limb joint angular velocity",
    "acquisition_quality": "Acquisition quality (not gait)",
}


def validate_groups(pool: list[str] | None = None) -> None:
    """Raise if the groups do not exactly partition the feature pool."""
    pool = list(MODEL_FEATURE_COLUMNS if pool is None else pool)
    assigned: list[str] = [f for feats in FEATURE_GROUPS.values() for f in feats]

    duplicates = {f for f in assigned if assigned.count(f) > 1}
    if duplicates:
        raise ValueError(f"Features assigned to more than one group: {sorted(duplicates)}")

    missing = sorted(set(pool) - set(assigned))
    extra = sorted(set(assigned) - set(pool))
    if missing or extra:
        raise ValueError(
            f"Feature groups do not partition the pool. Unassigned: {missing}. "
            f"Unknown to the pool: {extra}."
        )


def groups_present(available: list[str]) -> dict[str, list[str]]:
    """Groups restricted to columns actually present in the loaded table."""
    have = set(available)
    return {
        name: [f for f in feats if f in have]
        for name, feats in FEATURE_GROUPS.items()
        if any(f in have for f in feats)
    }


def summary() -> str:
    """Human-readable group sizes (used by the ablation report)."""
    lines = [f"{len(FEATURE_GROUPS)} groups over "
             f"{sum(len(v) for v in FEATURE_GROUPS.values())} features:"]
    for name, feats in FEATURE_GROUPS.items():
        lines.append(f"  {name:20s} {len(feats):3d}  {GROUP_LABELS[name]}")
    return "\n".join(lines)

