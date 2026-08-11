# Data contract (local only)

This directory is empty in the repository. Everything in it except this file is ignored by
Git, because it holds patient data. Place the files locally before running anything.

| File | Contents | Used by |
|---|---|---|
| `2026_05_17_FWOnly_2visits_features.csv` | MediaPipe gait/pose features, fast-walk trials, both visits. One row per recording, keyed by `sample_id`. | every FGA analysis |
| `2026_05_17_FWOnly_2visits_labels.csv` | `fga_estimate_score` (0–3) plus the 5 item sub-scores, one row per recording. | every FGA analysis |
| `bootstrap_indices.npz` | The shared per-fold participant indices used for the outer CV split. Stores **participant indices** in `sorted(unique(participant_id))` order, not row numbers. | all outer folds |
| `training_features_zeno_mentor_outcomes_wide.csv` | Zeno walkway measures for a separate, larger cohort (527 samples / 175 participants). | `scripts/run_multi_outcome.py`, `run_multi_outcome_fdr.py` |
| `BW_gait_videos_DPT_review.xlsx` | DPT review sheet; the source that confirms the label is a 0–3 estimate rather than the 30-point FGA total. | reference only |

## Conventions

- `sample_id` is `participant_id + "_" + video_index`, e.g. `<PATIENT>_1`, `<PATIENT>_2` for a patient's two visits.
- The study excludes `BW-0271_2` (its only available camera angle differs from every other
  recording). The exclusion is declared by `sample_id` in
  `qixing_fga.config.STUDY_EXCLUDED_SAMPLE_IDS` and applied before the visit filter, so the id
  is validated against the whole table. **Never edit the source CSV to drop a row.**
- After that exclusion the analysis sample is **91 rows from 46 participants**.

## Without these files

The code will not run, and four fold-alignment tests in `tests/test_bootstrap_alignment.py`
skip rather than fail. Everything already under `results/` was produced with these files.

Raw videos and MediaPipe landmark arrays are not part of this contract and are not needed to
reproduce any reported number — the feature table is the input.
