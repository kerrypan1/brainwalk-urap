# Local data layout

This directory is intentionally excluded from Git. It contains protected study
data, clinical labels, derived split files, and generated bootstrap indices.
Obtain the required files only through the approved UCSF/IRB data-sharing
process. Do not commit videos, spreadsheets, row-level labels, participant
identifiers, or derived patient-level predictions.

## Required inputs

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

An optional historical `bath_pws/` folder may contain preferred-walk videos
using the same filename convention. It is not required for the final
fast-walk results.

## Locked evaluation cohort

- 91 fast-walk videos from 46 patients
- Four-class target distribution: `{0: 9, 1: 15, 2: 31, 3: 36}`
- One labeled visit has no curated fast-walk video
- Fixed seed-42 patient-grouped five-fold evaluation

Identifiers appear in three forms:

| Level | Example |
|---|---|
| Participant | `BW-PPPP` |
| Label/sample ID | `P_V` |
| Curated video stem | `PPPP_V` |

## Generated local data

The following are also excluded from Git:

- `bootstrap_indices*.npz`
- Model 1 `gt.csv`, extracted clips, and per-clip VLM outputs
- Models 2–3 `artifacts/`, including indexed videos, labels, fold assignments,
  Zeno joins, and absolute local paths
- Feature/frame caches and all per-video or per-fold predictions

To rebuild the Models 2–3 data tables, run from
`brainwalk-clip-(model2-3)/data_build/`:

```powershell
python build_video_index.py
python build_fga_labels.py
python build_zeno_metrics.py
python join_video_zeno.py
python audit.py
```

Then build the labeled fast-walk table from
`brainwalk-clip-(model2-3)/src/`:

```powershell
python -m data.labeled_table
```

See the model-specific READMEs for feature extraction, inference, and training
commands.
