# BrainWalk Model 1: zero-shot VLM

This package evaluates off-the-shelf video-language models on the 91-video
fast-walk cohort. The canonical results and interpretation are in
[`../project_report.md`](../project_report.md).

Patient videos, generated clips, labels, model outputs, aggregate output files,
downloaded checkpoints, and virtual environments are local-only and ignored by
Git.

## Models

The wrappers under `models/` implement a shared `BaseVLM` interface:

| CLI key | Model |
|---|---|
| `intern_l` | InternVL2-2B |
| `intern_s` | InternVL2-1B |
| `llava_l` | LLaVA-NeXT-Video-7B |
| `llava_s` | LLaVA-OneVision-0.5B |

Prompt templates live under `prompts/` and are selected by model family,
in-context-learning mode, and prompt variant.

## Environment

Use an environment separate from Models 2–3 because the pinned PyTorch/CUDA
stack differs:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The large models require an NVIDIA GPU and download weights from Hugging Face
on first use.

## Pipeline

Run commands from `brainwalk-vlm-(model1)/`.

### 1. Build the local label table

```powershell
python scripts/xlsx_to_csv.py
```

This converts `../data/raw/zeno/BW_gait_videos_DPT_review.xlsx` to the local,
gitignored `gt.csv`.

### 2. Extract three high-motion clips per video

```powershell
python scripts/clip.py --fps 2 --clip_len 8
```

`clip.py` reads `../data/bath_fw/*.mp4`, finds high-motion windows, pads to
exactly three clips when needed, and writes:

```text
clips/clips_fps_<fps>_length_<seconds>/<video_stem>/clip_<1..3>.mp4
```

Use `--stems 0243_2 0270_1` for an incremental subset or `--overwrite` to
replace existing clips.

### 3. Run zero-shot inference

```powershell
python scripts/inference.py `
  --model llava_l --fps 2 --clip_len 8 `
  --icl n --dataset therapist
```

Inference is resumable and writes one response per clip under `vlm_output/`.
Other valid `--model` values are listed above. `--icl` accepts `n`, `y`, or
`generate`; `--dataset` selects the prompt suffix.

### 4. Evaluate the curated cohort

```powershell
python scripts/evaluate_fga.py `
  --pred_root "vlm_output/llava_l_noicl_therapist/clips_fps_2.0_length_8.0" `
  --expect_n 91
```

The evaluator preserves visit suffixes, intersects predictions with the
curated `bath_fw` cohort, averages up to three clip scores per video, and
reports raw clip-average MAE plus rounded-and-clipped accuracy. Mean, median,
and mode baselines are fitted independently on each outer training fold.
Headline values are the equal-weight mean and sample SD of the five held-out
fold metrics.

## Key code

```text
models/                     VLM wrappers
prompts/                    prompt templates and ablations
scripts/xlsx_to_csv.py      clinician workbook -> local gt.csv
scripts/clip.py             motion-based clip extraction
scripts/inference.py        resumable VLM inference
scripts/evaluate_fga.py     canonical n=91 FGA evaluation
```
