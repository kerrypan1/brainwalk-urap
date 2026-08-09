# Local data layout

This directory is intentionally excluded from Git. 

## Relevant folders

```text
data/
├── bath_fw/
│   └── PPPP_V.mp4
├── raw/
│   ├── bw_gait_videos/
│   │   └── <collection>/BW-PPPP/YYYY_MM_DD/gait_vertical_<protocol>_<trial>.mp4
│   └── zeno/
│       ├── BW_gait_videos_DPT_review.xlsx
│       └── 2025_12_03_BW_MS_ZenoData.xlsx
└── participant_stratified_groupkfold_split_seed42.csv
```

- `bath_fw/` contains the 91 curated fast-walk videos used by the final
  experiments. Filenames use a zero-padded participant number and visit:
  `PPPP_V.mp4`.
- `raw/bw_gait_videos/` contains the source video archive used by the data
  builders.
- `BW_gait_videos_DPT_review.xlsx` supplies the clinician-reviewed ordinal
  targets, including `FGA_estimate_score`.
- `2025_12_03_BW_MS_ZenoData.xlsx` supplies training-time auxiliary gait
  measurements for the Model 2 NTE adaptation.
- `participant_stratified_groupkfold_split_seed42.csv` is the locked
  patient-grouped five-fold split. Expected columns are `participant_id`,
  `split`, `label`, and `n_samples`.

## Evaluation cohort

- 91 fast-walk videos from 46 patients
- Four-class target distribution: `{0: 9, 1: 15, 2: 31, 3: 36}`
- Fixed seed-42 patient-grouped five-fold evaluation
