# FGA prediction from fast-walk video — gait-feature ML track

Predicting the clinician-rated **FGA estimate score (0–3 ordinal)** for BRAINWALK patients from
MediaPipe pose features extracted from their fast-walk (FW) recordings.

Results are reported on **91 fast-walk recordings from 46 patients** using a fixed seed-42,
patient-grouped five-fold split, with 76 gait/pose features and hyperparameters tuned in an
inner loop on the training folds only. The outer folds come from the shared
`bootstrap_indices.npz`, so this track and the others in this repository use the same patient
assignment.

**Full report — methods, results, limitations and interpretation:
[project document](https://docs.google.com/document/d/1DmRcMbAoZiMIggX1tVgz4MNGQprzl-WZK5LHD6tijtk/edit?usp=sharing).**
This repository is the code behind it.

---

## Headline result

| Model | MAE (5-fold mean ± SD) | ΔMAE vs baseline | 95% CI | one-sided P |
|---|---|---|---|---|
| **ordinal logistic (mord)** | **0.619 ± 0.202** | **−0.132** | [−0.307, +0.048] | 0.074 |
| SVR | 0.638 ± 0.161 | −0.113 | **[−0.228, −0.002]** | 0.022 |
| random forest | 0.650 ± 0.100 | −0.101 | [−0.223, +0.014] | 0.044 |
| ridge | 0.655 ± 0.121 | −0.096 | [−0.236, +0.056] | 0.099 |
| XGBoost | 0.660 ± 0.077 | −0.091 | [−0.228, +0.043] | 0.089 |
| median baseline | 0.751 ± 0.277 | — | — | — |

ΔMAE = MAE(model) − MAE(baseline); more negative is better. The ordinal model has the lowest
error and the largest effect, but **only SVR's confidence interval excludes zero**. The honest
summary is: lowest error, direction consistent across every model, but at n = 91 most models
are not separable from a median guess.

Nested RFECV — where feature selection happens inside each outer training fold, so the number
is properly validated — improves the ordinal model to **0.603**.

## Repository layout

```
src/qixing_fga/         importable package (data loading, CV, models, evaluation, reporting)
src/batch_training.py   CLI entry point
scripts/                one script per analysis
utils/                  shared helpers (feature groups, feature extraction, per-fold bootstrap, plotting)
configs/                YAML experiment configs
tests/                  leakage, split-alignment, bootstrap and artefact tests
data/                   empty; see data/README.md for the data contract
```

Run outputs are written to `results/`, which is not tracked: per-sample prediction files carry
participant identifiers, so nothing generated is committed.

## Process

### Entry point

- **`src/batch_training.py`** — Main driver. Runs nested-CV training and evaluation for a given
  YAML config and writes a full run directory (config, metrics, per-sample predictions,
  provenance, figures).
- **`configs/*.yaml`** — One config per experiment. `fga_fw_2visit.yaml` is the main
  experiment; `fga_nested_rfecv.yaml` adds fold-internal feature selection;
  `fga_fw_visit1.yaml` reproduces the single-visit limitation evidence.

### Dataset description

- **`scripts/describe_dataset.py`** — Cohort counts, label distribution, missingness and
  between-visit change. Applies the study exclusion automatically.

### Baselines

- **`scripts/run_baselines.py`** — The three naive baselines (training-fold median / mean /
  majority), the single best gait feature, and a 4-feature interpretable ridge.

### Model evaluation

- **`scripts/run_model_diagnostics.py`** — Learning curve and train-vs-validation gap.
- **`scripts/explain_delta_mae_ci.py`** — Regenerates the worked bootstrap example showing what
  the reported 95% CI column is, and what it is not.
- **`scripts/plot_model_comparison.py`** — The model-vs-baseline comparison figure.

### Feature analysis

- **`scripts/run_feature_ablation.py`** — Leave-one-group-out and group-only ablation over the
  7 feature groups.
- **`scripts/run_mord_coefficients.py`** — Standardised coefficients of the reported model, per
  outer fold. This is the interpretability artefact for the main table.
- **`scripts/run_shap_analysis.py`** — Random-forest SHAP, kept as a diagnostic only; the
  reported feature importances come from the coefficients above.
- **`scripts/build_feature_catalog.py`** — Generates a catalogue of all 76 features with what
  each one actually computes and its measurement caveats.

### Other outcomes

- **`scripts/run_multi_outcome.py`** — The same pipeline over the 5 FGA sub-scores and the 16
  Zeno walkway measures.
- **`scripts/run_multi_outcome_fdr.py`** — Correlation structure, PCA and Benjamini-Hochberg FDR
  correction across the multi-outcome test family.

## Running it

```bash
pip install -r requirements.txt
```

Place the data first — see [`data/README.md`](data/README.md). Then:

```bash
# Main experiment (n=91, ordinal). Run this first: run_model_diagnostics.py reads its predictions.
python src/batch_training.py --config configs/fga_fw_2visit.yaml --output-dir results/fga_fw_2visit_n91

# Nested RFECV (feature selection validated inside the outer folds)
python src/batch_training.py --config configs/fga_nested_rfecv.yaml --output-dir results/nested_rfecv_2visit_n91

# Dataset description
python scripts/describe_dataset.py

# Naive baselines
python scripts/run_baselines.py --visit all --output-dir results/baselines_2visit_n91

# Learning curve and generalisation gap
python scripts/run_model_diagnostics.py --visit all \
  --predictions results/fga_fw_2visit_n91/predictions.csv \
  --output-dir results/model_diagnostics_2visit_n91

# Feature-group ablation
python scripts/run_feature_ablation.py --visit all --output-dir results/ablation_2visit_n91

# Standardised coefficients of the reported model
python scripts/run_mord_coefficients.py

# Feature catalogue
python scripts/build_feature_catalog.py

# Multi-outcome extension with FDR control
python scripts/run_multi_outcome.py --output-root results/multi_outcome
python scripts/run_multi_outcome_fdr.py

# Tests
python -m pytest tests/ -q
```

Without the data, four fold-alignment tests skip and the rest pass.

**Do not pass `--task` to `batch_training.py`** unless you intend to override the config. It
defaults to `None` so that `task: regression` in the config takes effect.

`configs/fga_fw_visit2.yaml` currently fails: excluding one visit-2 recording leaves 45
participants in that view while the shared bootstrap split indexes 46, so the fold builder runs
off the end of the participant list. It is kept for reference and is not part of any reported
result.

## Method notes worth reading before the numbers

**One aggregation convention.** Every metric is the equal-weight mean over the five outer folds,
and the ± SD is the across-fold SD. ΔMAE and its confidence interval use the same convention,
via a fold-stratified participant bootstrap: participants are resampled within each fold, that
fold's mean difference is recomputed, and the five fold-level values are averaged with equal
weight. Do not replace that interval with a paired t-test over the five fold-level differences —
the outer training sets overlap by about 80%, so those five values are not independent and n = 5
has almost no power.

**Bootstrap resamples participants, not rows.** With two visits per person these are no longer
equivalent; the resampling unit has to be the independent unit.

**Frame rate was removed from the feature set.** The source video's frame rate is a property of
the recording session, not of the patient: 89 of the 91 samples share one value and two do not,
so after standardisation the column works as a near-identifier for two specific recordings.
Removing it cost real headline performance (MAE 0.597 → 0.619), which is exactly why it had to
go.

**Feature selection results are only reportable when the selection happens inside the fold.**
Nested RFECV (0.603) qualifies. Fixing a feature list chosen from cross-fold aggregates and then
retraining does not: measured on this data, that shortcut lowers the MAE by a further 0.045 and
flips the headline from non-significant to significant.

## Data

This track is **code and documentation only**. All contents of `data/`, except its README, are
ignored by Git. Place the data locally according to [`data/README.md`](data/README.md), which
lists the expected files and the `sample_id` convention. Contact the project lead for access.
