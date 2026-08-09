# BrainWalk Models 2–3

This package contains:

- **Model 2:** the constrained Vita-CLIP/KAPT/NTE adaptation; and
- **Model 3:** frozen OpenCLIP frame features with supervised classification
  and ordinal heads.

The final methods and results are reported in
[`../project_report.md`](../project_report.md). Generated data tables, caches,
weights, and outputs are local-only and ignored by Git.

## Environment

The final runs used Python 3.13, an NVIDIA RTX 4070, and CUDA 12.5.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Run data builders from `data_build/` and model modules from `src/`. Paths are
resolved relative to this repository, so the commands do not depend on the
current user's home directory.

## Build local data tables

After placing approved inputs according to
[`../data/README.md`](../data/README.md):

```powershell
cd data_build
python build_video_index.py
python build_fga_labels.py
python build_zeno_metrics.py
python join_video_zeno.py
python audit.py

cd ..\src
python -m data.labeled_table
```

These commands populate `artifacts/` with row-level local tables. Do not commit
that directory.

## Model 2: Vita-CLIP adaptation

Cache native-rate person crops:

```powershell
cd src
python -m features.cache_native
```

Run the fixed patient-grouped five-fold baseline:

```powershell
python -m train.train_vita_faithful `
  --frames_dir ../cache/frames_labeled_native_224 `
  --folds 5 --split fixed --epochs 8 --k_windows 8 --batch 4 `
  --run_name baseline_e8k8_n91_5fold_seed42
```

Add `--kapt`, `--nte`, or both flags for the three knowledge-augmentation
variants. Training is fold-resumable; local per-fold predictions and aggregate
metrics are written under `outputs/`.

## Model 3: frozen CLIP heads

Extract the final cropped T=32 features:

```powershell
cd src
python -m features.extract_frames --source labeled --num_frames 32
```

For the matched uncropped control:

```powershell
python -m features.extract_frames --source labeled --num_frames 32 --no_crop
```

Train and evaluate the supervised heads with the generated aggregate feature
file:

```powershell
python -m train.train_item_heads `
  --features ../cache/framefeat_labeled_crop_ViT-B-32-quickgelu_openai_T32.npz `
  --run_name n91_5fold_seed42_crop
```

The trainer evaluates class-balanced logistic regression, CORAL-style ordinal
thresholds, and Ridge regression on the fixed five folds. MAE uses each head's
raw continuous score; classification metrics use rounded-and-clipped classes.
Mean, median, and mode baselines are fitted on each outer training fold, and
headline values are the equal-weight mean and sample SD across held-out folds.

## Key code

```text
data_build/                         raw-input indexing and joins
src/data/                           labels, CV, person cropping, data tables
src/features/cache_native.py        native-rate crop cache for Model 2
src/features/extract_frames.py      frozen OpenCLIP features for Model 3
src/models/vita_clip_faithful.py    Vita-CLIP adaptation
src/train/train_vita_faithful.py    Model 2 fixed-fold training
src/train/train_item_heads.py       Model 3 fixed-fold heads
src/eval/metrics.py                 shared aggregate metrics
```
