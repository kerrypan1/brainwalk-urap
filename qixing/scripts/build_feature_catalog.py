#!/usr/bin/env python3
"""Generate docs/FEATURE_CATALOG.md: every model feature, grouped, with
label-free screening information only (no MAE / importance — those would leak
into a manual selection)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from qixing_fga.protocol import load_protocol_data  # noqa: E402
from utils.feature_groups import FEATURE_GROUPS, GROUP_LABELS  # noqa: E402

# what the extractor actually computes (utils/feature_utils.py), caveat
D: dict[str, tuple[str, str]] = {
    # ---- pace_rhythm
    "cadence_hz": ("Steps per second, from the step-event count and the recording duration",
                   "Same quantity as `cadence_spm`, differing only by a factor of 60"),
    "cadence_spm": ("Steps per minute; same computation, per-minute scaling",
                    "Duplicate of `cadence_hz`"),
    "gait_speed_enhanced": ("Total 2D pelvis path length in the x-z plane divided by duration",
                            "Units are **landmark units per second, not m/s**; sensitive to camera distance"),
    "mean_gait_speed": ("Estimated from the pelvis x-coordinate series", "Also not a physical unit"),
    "stride_speed_mean": ("Mean of the per-stride speeds", ""),
    "step_length_mean": ("Mean ankle displacement between consecutive step events", "Not a physical unit"),
    "stride_length_mean": ("Displacement between consecutive ipsilateral foot contacts", "Not a physical unit"),
    "stride_width_mean": ("Mean lateral separation of the left and right ankle x-coordinates", "Not a physical unit"),
    "step_time_mean": ("Mean interval between consecutive step events", ""),
    "step_time_left_mean": ("Left-side step events only", ""),
    "step_time_right_mean": ("Right-side step events only", ""),
    "stride_time_mean": ("Interval between consecutive ipsilateral foot contacts", ""),
    "stance_time_mean": ("Mean stance duration per gait cycle", ""),
    "stance_ratio_mean": ("Stance duration divided by gait-cycle duration", ""),
    "swing_time_mean": ("Mean swing duration per gait cycle", ""),
    "double_support_time_fraction": ("Frames with both feet on the ground, divided by total frames",
                                     "Clinically a strong correlate of balance impairment"),
    "step_events_left_count": ("**Total** number of left-foot step events in the whole recording",
                               "**Scales with recording length** — a count, not a rate"),
    "step_events_right_count": ("Same, right foot", "Same caveat; highly correlated with the left count"),
    "dominant_frequency": ("Dominant tremor frequency of the wrist y-signals, averaged left/right",
                           "A tremor measure; its placement in the spatiotemporal group is questionable"),
    "foot_progression_angle_mean": ("Mean angle between foot direction and the walking axis", ""),
    # ---- variability
    "step_length_cv": ("SD of step length divided by its mean", ""),
    "step_length_std": ("SD of step length", "Same source as `step_length_cv`"),
    "step_time_std": ("SD of step time", ""),
    "step_time_left_std": ("Left side only", ""),
    "step_time_right_std": ("Right side only", ""),
    "step_time_variability": ("SD of step time divided by its mean", "Same source as `step_time_std`"),
    "stride_length_cv": ("SD of stride length divided by its mean",
                         "Its coefficient in the reported model **flips sign between folds** — unstable"),
    "stride_time_cv": ("SD of stride time divided by its mean", ""),
    "foot_progression_angle_std": ("SD of the foot progression angle", ""),
    "com_acceleration_std": ("SD of the centre-of-mass acceleration series",
                             "Heavily duplicated by `com_acceleration_std_log1p` and `com_velocity_std`"),
    "com_acceleration_std_log1p": ("log1p transform of the row above",
                                   "**Perfectly rank-identical to the untransformed column (r = 1.000)**"),
    "com_velocity_std": ("SD of the centre-of-mass velocity series", "r ~ 0.995 with COM acceleration SD"),
    "head_rotation_variability": ("SD of the rotation angular velocity divided by its mean",
                                  "**'Head' is inferred from shoulder width** — see the notes above"),
    "trunk_tilt_std": ("SD of the shoulder-to-hip line tilt angle", ""),
    # ---- symmetry
    "arm_swing_amplitude_asymmetry": ("Difference in wrist-relative-to-shoulder swing amplitude, left vs right", ""),
    "arm_swing_peak_to_peak_asymmetry": ("|left peak-to-peak - right| divided by their sum",
                                         "Computed directly from `arm_swing_peak_to_peak_left/right`"),
    "hip_angle_velocity_asymmetry_index": ("Asymmetry index of the mean left/right hip angular velocity",
                                           "Derived from two joint-kinematics columns"),
    "knee_angle_velocity_asymmetry_index": ("Asymmetry index of the mean left/right knee angular velocity",
                                            "Derived from two joint-kinematics columns"),
    "leg_step_length_asymmetry": ("Asymmetry of the left/right step-length series",
                                  "**r = 0.997** with `step_length_asymmetry_index`"),
    "shoulder_height_asymmetry": ("Difference between the left and right shoulder y-coordinates", ""),
    "step_length_asymmetry_index": ("Asymmetry index of the mean left/right step length",
                                    "Near-duplicate of `leg_step_length_asymmetry`"),
    "step_time_asymmetry_index": ("Asymmetry index of the mean left/right step time", ""),
    "stepping_symmetry_ratio": ("1 - |left count - right count| / (left + right)",
                                "**A deterministic function of the two step-event counts** — no independent information"),
    "phase_locking_value": ("Phase synchrony between the left and right wrist y-signals",
                            "Upper-limb coordination; same source signal as the tremor group"),
    # ---- postural
    "com_acceleration_mean": ("Mean centre-of-mass acceleration", "**r = 1.000** with its log1p version"),
    "com_acceleration_mean_log1p": ("log1p transform of the row above", "**Rank-identical to the untransformed column**"),
    "com_velocity_mean": ("Mean centre-of-mass velocity", "r ~ 0.989 with mean COM acceleration"),
    "head_lateral_sway": ("SD of the nose x-coordinate",
                          "Not a physical unit; **sensitive to camera distance and framing**"),
    "head_path_length": ("Sum of frame-to-frame nose displacement in the x-y plane",
                         "**Grows with recording length**; not normalised"),
    "head_rotation_speed_max": ("Maximum rotation angular velocity",
                                "**Inferred from shoulder width, not true head orientation**"),
    "head_rotation_speed_mean": ("Mean rotation angular velocity", "Same caveat"),
    "head_vertical_sway": ("SD of the nose y-coordinate", "Not a physical unit"),
    "trunk_tilt_mean": ("Mean sagittal-plane tilt of the shoulder-to-hip line", ""),
    "jerk_magnitude": ("Jerk of the wrist coordinates, averaged left/right",
                       "Really an **upper-limb** smoothness measure; its placement in the postural group is questionable"),
    "power_3_7Hz": ("(not produced by the current extractor)", "**Constant zero in this dataset — no information**"),
    # ---- upper_limb
    "arm_swing_peak_to_peak_left": ("Peak-to-peak swing of the left wrist relative to the left shoulder", ""),
    "arm_swing_peak_to_peak_right": ("Peak-to-peak swing of the right wrist relative to the right shoulder", ""),
    "elbow_angle_velocity_mean_left": ("Mean left elbow angular velocity", ""),
    "elbow_angle_velocity_mean_right": ("Mean right elbow angular velocity", ""),
    "elbow_angle_velocity_std_left": ("SD of the left elbow angular velocity", ""),
    "elbow_angle_velocity_std_right": ("SD of the right elbow angular velocity", ""),
    "tremor_cross_wrist_coherence": ("Spectral coherence between the left and right wrist y-signals", ""),
    "tremor_intermittency": ("Intermittency of wrist tremor, averaged left/right", ""),
    "tremor_regularity": ("Regularity of tremor in the 3-7 Hz band, averaged over both wrists", ""),
    "wrist_tremor_amplitude": ("Wrist tremor amplitude, averaged left/right", ""),
    # ---- joint_kinematics
    "hip_angle_velocity_mean_left": ("Mean left hip angular velocity", ""),
    "hip_angle_velocity_mean_right": ("Mean right hip angular velocity", ""),
    "hip_angle_velocity_std_left": ("SD of the left hip angular velocity", ""),
    "hip_angle_velocity_std_right": ("SD of the right hip angular velocity", ""),
    "knee_angle_velocity_mean_left": ("Mean left knee angular velocity", ""),
    "knee_angle_velocity_mean_right": ("Mean right knee angular velocity", ""),
    "knee_angle_velocity_std_left": ("SD of the left knee angular velocity", ""),
    "knee_angle_velocity_std_right": ("SD of the right knee angular velocity", ""),
    # ---- acquisition_quality
    "landmark_gap_filled_ratio": ("(not produced by the current extractor)", "**Constant zero in this dataset — no information**"),
    "landmark_large_gap_count": ("(not produced by the current extractor)", "**Constant zero in this dataset — no information**"),
    "landmark_mean_nan_ratio": ("(not produced by the current extractor)", "**Constant zero in this dataset — no information**"),
}

d = load_protocol_data(visit="all", root=ROOT)
X = d.X
const = {c for c in X.columns if X[c].nunique() <= 1}
Xv = X.drop(columns=list(const))
C = Xv.corr(method="spearman").abs().to_numpy().copy()
np.fill_diagonal(C, 0.0)
cols = list(Xv.columns)
partners: dict[str, list[str]] = {c: [] for c in cols}
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if C[i, j] >= 0.95:
            partners[cols[i]].append(f"`{cols[j]}` ({C[i, j]:.3f})")
            partners[cols[j]].append(f"`{cols[i]}` ({C[i, j]:.3f})")

# line numbers from the extractor, for traceability
src = (ROOT / "utils/feature_utils.py").read_text(encoding="utf-8", errors="replace").splitlines()
pat = re.compile(r"out\[['\"]([a-z0-9_]+)['\"]\]\s*=")
line_of: dict[str, int] = {}
for i, ln in enumerate(src):
    m = pat.search(ln)
    if m and m.group(1) not in line_of:
        line_of[m.group(1)] = i + 1

out: list[str] = []
A = out.append
A("# Feature catalogue (for manual screening)")
A("")
A(f"All **{sum(len(v) for v in FEATURE_GROUPS.values())}** gait/pose features that enter the "
  "models, in 7 groups.")
A("Generated by `scripts/build_feature_catalog.py`; descriptions follow the actual computation "
  "in `utils/feature_utils.py`.")
A("")
A("> ## Warning — the one rule for using this table")
A(">")
A("> **Selection criteria must come from outside this dataset** — literature, biomechanical "
  "reasoning,")
A("> measurement reliability, whether the unit is interpretable. **Do not** pick features by "
  "consulting")
A("> the ablation table, the nested-RFECV output, model coefficients or SHAP rankings: all of "
  "those are")
A("> computed from this dataset's **labels**, so choosing from them lets the test folds into the")
A("> selection and biases every MAE reported afterwards (measured here at about 0.045 MAE — enough")
A("> to turn the headline result from non-significant into significant).")
A(">")
A("> This table therefore carries **label-free information only**: group, actual computation, "
  "whether")
A("> the column is constant, and whether it duplicates another column. Everything "
  "performance-related")
A("> is deliberately excluded.")
A("")
A("## Read these whole-set issues first")
A("")
A("1. **Three exactly duplicated pairs** (Spearman = 1.000): `cadence_hz`/`cadence_spm` is the same")
A("   quantity in two units; `com_acceleration_mean`/`_log1p` and `com_acceleration_std`/`_log1p`")
A("   are a column and its own log transform. **Keep only one of each pair.**")
A("2. **Four features are constant zero in this dataset** (three landmark-quality measures plus")
A("   `power_3_7Hz`), and the current extractor does not produce them at all. They contribute")
A("   nothing to any model and can be dropped outright.")
A("3. **The three 'head rotation' columns are inferred from shoulder width**: `head_rotation_*` "
  "comes")
A("   from `arctan2(right_shoulder_x - left_shoulder_x, 1)`, i.e. the shoulder line used as an")
A("   orientation proxy, not true head orientation. The names are misleading.")
A("4. **Several features are unnormalised and depend on recording length or camera setup**:")
A("   `step_events_*_count` (a total count, so longer recordings give larger values),")
A("   `head_path_length` (a sum of frame-to-frame displacement), and every length/speed quantity")
A("   derived from landmark coordinates (units are normalised coordinates, not metres, and depend")
A("   on camera distance and framing). These are the main risk for multi-site generalisation.")
A("5. **`stepping_symmetry_ratio` is a deterministic function of `step_events_left/right_count`**")
A("   and carries no independent information.")
A("")
A("---")
A("")

for g, feats in FEATURE_GROUPS.items():
    A(f"## {g} — {GROUP_LABELS.get(g, g)} ({len(feats)} features)")
    A("")
    A("| Pick | Feature | What it computes | Notes / risks |")
    A("|:--:|---|---|---|")
    for f in sorted(feats):
        what, note = D.get(f, ("—", ""))
        flags = []
        if f in const:
            flags.append("**constant, no information**")
        if partners.get(f):
            flags.append("duplicates " + ", ".join(partners[f]))
        if note:
            flags.append(note)
        ln = f" <sub>L{line_of[f]}</sub>" if f in line_of else ""
        A(f"| [ ] | `{f}`{ln} | {what} | {'; '.join(flags) or '—'} |")
    A("")

A("---")
A("")
A("## Using a selection")
A("")
A("Write the chosen feature names to a JSON list, then:")
A("")
A("```bash")
A("python src/batch_training.py --config configs/fga_fw_2visit.yaml \\")
A("  --features-json <your-list>.json --output-dir results/manual_subset_n91")
A("```")
A("")
A("`reject_leaky_columns` validates the list before training — a misspelled name or a metadata")
A("column raises rather than passing silently.")
A("")
A("The resulting MAE **is reportable**, provided the selection genuinely did not use this dataset's")
A("labels. The methods section must state the reason for the selection (a citation or a")
A("biomechanical argument); reviewers will ask why these features.")
A("")
A("The <sub>L…</sub> line numbers point at the assignment for that feature in")
A("`utils/feature_utils.py`, so the computation can be checked directly.")

(ROOT / "docs" / "FEATURE_CATALOG.md").write_text("\n".join(out), encoding="utf-8")
print("written docs/FEATURE_CATALOG.md")
print("features documented:", sum(len(v) for v in FEATURE_GROUPS.values()),
      "| descriptions provided:", len(D))
missing = [f for v in FEATURE_GROUPS.values() for f in v if f not in D]
print("missing description:", missing)
