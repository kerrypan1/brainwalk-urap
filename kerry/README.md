# BrainWalk video gait scoring

This repository contains three approaches for predicting the
clinician-assessed four-class `FGA_estimate_score` from fast-walk video:

1. zero-shot video-language models;
2. a constrained Vita-CLIP/KAPT/NTE adaptation inspired by Wang et al.; and
3. frozen OpenCLIP video features with lightweight supervised heads.

The canonical methods, results, limitations, and interpretation are in
[`project_report.md`](project_report.md). Results are reported on 91 videos
from 46 patients using a fixed seed-42, patient-grouped stratified five-fold
split for supervised models.

## Repository layout

```text
.
├── project_report.md              # final report and canonical results
├── data/
│   └── README.md                  # local-only data contract
├── brainwalk-vlm-(model1)/        # zero-shot VLM pipeline
└── brainwalk-clip-(model2-3)/     # Vita-CLIP adaptation and frozen-CLIP heads
```

Generated results, patient-level outputs, videos, clinical spreadsheets,
feature caches, model weights, and virtual environments are deliberately not
versioned.

## Data and privacy

The repository is code-and-documentation only. All contents of `data/`, except
its README, are ignored by Git. Obtain study data through an approved
UCSF/IRB channel and place it locally according to
[`data/README.md`](data/README.md).

Do not commit:

- raw or curated patient videos;
- clinician or Zeno spreadsheets;
- participant split files or derived labels;
- clips, frame/feature caches, or downloaded weights; or
- per-video and per-fold predictions.

## Model packages

### Model 1: zero-shot VLM

[`brainwalk-vlm-(model1)/README.md`](brainwalk-vlm-%28model1%29/README.md)
documents motion-based clip selection, InternVL/LLaVA inference, and the
visit-safe FGA evaluator.

Model 1 has its own dependency set:

```powershell
python -m venv "brainwalk-vlm-(model1)\.venv"
& "brainwalk-vlm-(model1)\.venv\Scripts\Activate.ps1"
pip install -r "brainwalk-vlm-(model1)\requirements.txt"
```

### Models 2–3: Vita-CLIP and frozen CLIP

[`brainwalk-clip-(model2-3)/README.md`](brainwalk-clip-%28model2-3%29/README.md)
documents data-table generation, feature extraction, the fixed-fold
Vita-CLIP variants, and the frozen-CLIP supervised heads.

Use a separate environment because its PyTorch/OpenCLIP requirements differ
from Model 1:

```powershell
python -m venv "brainwalk-clip-(model2-3)\.venv"
& "brainwalk-clip-(model2-3)\.venv\Scripts\Activate.ps1"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r "brainwalk-clip-(model2-3)\requirements.txt"
```

Python 3.13 and an NVIDIA CUDA GPU were used for the final local runs. Model 1
loads large Hugging Face checkpoints and may require a different CUDA/PyTorch
combination than Models 2–3.

## Reference

Wang D, Yuan K, Muller C, Blanc F, Padoy N, Seo H. *Enhancing Gait Video
Analysis in Neurodegenerative Diseases by Knowledge Augmentation in Vision
Language Model.* MICCAI 2024; arXiv:2403.13756.
