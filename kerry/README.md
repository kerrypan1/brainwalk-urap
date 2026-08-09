# BrainWalk video gait scoring

This repository contains three approaches for predicting the
clinician-assessed four-class `FGA_estimate_score` from fast-walk video:

1. zero-shot video-language models;
2. a constrained Vita-CLIP/KAPT/NTE adaptation inspired by Wang et al.; and
3. frozen OpenCLIP video features with lightweight supervised heads.

The methods, results, limitations, and interpretation are in
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

## Data

The repository is code-and-documentation only. All contents of `data/`, except
its README, are ignored by Git. Place data locally according to
[`data/README.md`](data/README.md).
