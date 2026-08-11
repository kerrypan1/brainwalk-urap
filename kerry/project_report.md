# BRAINWALK VLM methods and results

(also in https://docs.google.com/document/d/1Ni35AyJtCMB4sKvyZ4Su3uuhpwq4YgQQJb4ljaO6CZU/edit?usp=sharing)

## Dataset

- **91 fast-walk BRAINWALK videos from 46 patients**
- 92 clinician labels were available; one visit was excluded because no comparable fast-walk video from the standard camera angle was available
- Input: raw RGB video
- Prediction target: clinician-assessed `FGA_estimate_score`, an ordered
  four-class estimate from 0 (most impaired) to 3 (near normal)
- Intended inference input: video only

| `FGA_estimate_score` | Videos |
|---:|---:|
| 0 | 9 |
| 1 | 15 |
| 2 | 31 |
| 3 | 36 |

All supervised models use the same fixed, seed-42, **patient-grouped stratified
five-fold cross-validation** split. A patient never appears in both training
and test data. Model 1 is zero-shot and has no fold-specific training, but it is
evaluated on the same 91-video cohort.

For each outer fold, the mean, median, and mode baselines are fitted only on
that fold's training labels and applied to its held-out videos. Accuracy is
computed after rounding and clipping predictions to the nearest valid class.
MAE is computed on the raw continuous prediction before rounding. Headline
values are the equal-weight mean of the five held-out fold metrics; `±` is the
sample standard deviation across those five values. This differs from pooling
all 91 out-of-fold predictions into one metric.

The training-fold median is 2 in every fold. The mode is 3 in four folds and 2
in one fold; the training-fold mean ranges from 1.959 to 2.070.

## Model 1: zero-shot VLM

### Models tested

- InternVL2-2B
- InternVL2-1B
- LLaVA-NeXT-Video-7B
- LLaVA-OneVision-0.5B

### Method

1. Select three high-motion clips from each full video.
2. Provide each clip and a gait-scoring prompt independently to the VLM.
3. Average the three numeric outputs to obtain the video-level prediction.

Ablations included prompt wording, model family/size, and paired clip
fps-duration settings `(1,16)`, `(2,8)`, `(4,4)`, `(8,2)`, and `(16,1)`.

### Primary result

| Predictor | Accuracy | MAE |
|---|---:|---:|
| Fold-trained mean | 0.347 ± 0.228 | 0.768 ± 0.273 |
| Fold-trained median | 0.347 ± 0.228 | 0.751 ± 0.277 |
| Fold-trained mode | 0.305 ± 0.188 | 0.962 ± 0.206 |
| LLaVA, 2 fps × 8 s | 0.347 ± 0.228 | **0.748 ± 0.284** |
| LLaVA, 16 fps × 1 s | 0.347 ± 0.228 | 0.755 ± 0.271 |

Both evaluated LLaVA configurations rounded to class 2 for all 91 videos.
Therefore, the small raw-MAE difference from the median baseline is caused by
fractional clip averaging, not clinically useful discrimination.

## Model 2: knowledge-augmented Vita-CLIP

**Reference:** Wang D, Yuan K, Muller C, Blanc F, Padoy N, Seo H.
*Enhancing Gait Video Analysis in Neurodegenerative Diseases by Knowledge
Augmentation in Vision Language Model.* MICCAI 2024; arXiv:2403.13756.

### Goal and adaptations

Adapt the paper's Vita-CLIP and knowledge-augmentation ideas to predict the
four-class clinician score from fast-walk videos:

- **Vita-CLIP-style visual prompting:** frozen CLIP with learned
  summary/global/local prompts and cosine-to-text classification using focal
  loss.
- **KAPT-inspired descriptions:** detailed class descriptions with shared
  CoOp-style context. This is not full KAPT, KEPLER initialization and
  per-class KAPT MLPs were not implemented.
- **NTE-inspired numerical text:** Zeno gait metrics represented as numerical
  text and aligned to class text. This is an approximate adaptation using a
  different metric set.

All four variants were rerun on the same fixed five folds as Model 3.

| Variant                        |      Accuracy |               MAE |
| ------------------------------ | ------------: | ----------------: |
| Vita-CLIP baseline             | 0.341 ± 0.144 |     0.906 ± 0.194 |
| + descriptions (KAPT-inspired) | 0.267 ± 0.138 |     0.918 ± 0.211 |
| + NTE                          | 0.273 ± 0.110 |     0.930 ± 0.178 |
| + descriptions + NTE           | 0.272 ± 0.158 | **0.891 ± 0.179** |

The existing Model 2 caches retain only predicted classes, not averaged
softmax scores, so for Model 2, MAE is calculated from class-label rather than
continuous-score MAE. Therefore, Model 2 MAE is not directly identical to the continuous-score MAE reported for Models 1 and 3. The paper-inspired variants did not outperform the simple cropped-CLIP models.

Related experiments that also underperformed: zero-shot/CoOp prompting, an early VPT + temporal-transformer implementation, and sparse temporal-window models.

## Model 3: frozen CLIP encoder with supervised head

Instead of using a complete VLM, this approach uses a standalone frozen CLIP vision encoder to construct a video-level representation. A lightweight supervised classification head is then trained to predict the FGA score.

### Encoder and method

- OpenCLIP `ViT-B-32-quickgelu` with OpenAI weights
- Uniformly sample 32 frames per video
- Detect, track, and crop the person with YOLOv8n
- Encode every cropped frame with frozen CLIP
- Mean-pool frame embeddings into one video representation
- Train a lightweight head within each training fold

Heads tested: class-balanced logistic regression, CORAL-style ordinal
thresholds, and Ridge regression with rounding for classification metrics. For
logistic regression, the continuous score is the class-probability-weighted
expectation; for CORAL it is the sum of cumulative threshold probabilities.
Ridge uses its direct continuous output.

### Primary four-class endpoint

| Predictor               |          Accuracy |               MAE |
| ----------------------- | ----------------: | ----------------: |
| Fold-trained mean       |     0.347 ± 0.228 |     0.768 ± 0.273 |
| Fold-trained median     |     0.347 ± 0.228 |     0.751 ± 0.277 |
| Fold-trained mode       |     0.305 ± 0.188 |     0.962 ± 0.206 |
| Uncropped CLIP + logreg |     0.382 ± 0.089 |     0.757 ± 0.097 |
| Cropped CLIP + logreg   |     0.406 ± 0.144 |     0.700 ± 0.140 |
| Cropped CLIP + CORAL    | **0.427 ± 0.091** | **0.690 ± 0.095** |
| Cropped CLIP + Ridge    |     0.423 ± 0.113 |     0.760 ± 0.132 |

CORAL gives the strongest overall primary result by accuracy and raw MAE. Cropping improves every reported metric for the
matched logistic-regression control, although fold variability is substantial.

Confusion Matrix for best result (CLIP + CORAL):  
Rows = true FGA (0-3, top to bottom), columns = predicted (0-3, left to right)   
[[ 1,  5,  2,  1],  
 [ 1,  6,  7,  1],  
 [ 0,  5, 14, 12],  
 [ 1,  1, 16, 18]]  

### Exploratory three-class endpoint

Classes 0 and 1 are merged, producing ordered classes `{0–1, 2, 3}`, reindexed
to `{0,1,2}` for MAE. This is a changed and easier endpoint, not the primary
four-class result.

| Predictor             |          Accuracy |               MAE |
| --------------------- | ----------------: | ----------------: |
| Fold-trained mean     |     0.347 ± 0.228 |     0.687 ± 0.201 |
| Fold-trained median   |     0.347 ± 0.228 |     0.653 ± 0.228 |
| Fold-trained mode     |     0.305 ± 0.188 |     0.865 ± 0.187 |
| Cropped CLIP + logreg | **0.493 ± 0.129** | **0.573 ± 0.094** |
| Cropped CLIP + CORAL  |     0.492 ± 0.102 |     0.582 ± 0.066 |

## Overall interpretation

The main conclusions are:

1. Person cropping contributes more than added architectural complexity.
2. The zero-shot VLM effectively collapses to a constant prediction.
3. The constrained Vita-CLIP/KAPT/NTE adaptation does not transfer the paper's
   reported gain to this endpoint and cohort.
4. Cropped CLIP + CORAL gives the strongest primary four-class result, while cropped CLIP + logistic regression performs best on the exploratory three-class endpoint. The three-class result approaches 50% accuracy but represents a changed endpoint that requires clinical justification.
