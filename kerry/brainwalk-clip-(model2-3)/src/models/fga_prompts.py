"""FGA class descriptions, ordered by FGA_estimate_score (0=most impaired ... 3=near-normal).

Used as the CoOp class-token text and for the zero-shot reference. Kept concise
so `n_ctx` context tokens + description + EOT stay within CLIP's 77-token limit.
"""
from __future__ import annotations

# CoOp class-token strings (the learnable context supplies the leading prompt).
# Enriched with the clinical FGA rubric (KAPT-style knowledge prompt init): each
# class names its hallmark speed / step / balance / deviation / turning signs.
FGA_CLASSNAMES = {
    0: ("severe functional gait impairment, very slow unstable cautious walking, marked trunk sway, "
        "frequent balance corrections, short irregular steps, unable to keep a straight path or turn"),
    1: ("moderate functional gait impairment, slow walking with visible instability, reduced step length, "
        "increased step variability, lateral deviation, difficulty with head turns and pivot turns"),
    2: ("mild functional gait impairment, mostly independent and stable walking, mildly reduced speed, "
        "mild asymmetry, small balance corrections, slight trunk sway, reduced smoothness on turns"),
    3: ("normal dynamic gait, appropriate walking speed, symmetric steps, controlled trunk motion, "
        "balance maintained, confident turns, minimal visible instability or compensation"),
}

# Full sentences for zero-shot CLIP (no learnable context).
FGA_ZEROSHOT = {
    0: "a video of a person with severe functional gait impairment, walking very slowly and unstably with marked imbalance",
    1: "a video of a person with moderate functional gait impairment, walking slowly with visible instability and gait deviations",
    2: "a video of a person with mild functional gait impairment, walking mostly stably with subtle gait abnormalities",
    3: "a video of a person with normal gait, walking at an appropriate speed with symmetric steps and good balance",
}


# Short class labels for the faithful Baseline (CoOp learnable ctx + {label}, no KAPT).
FGA_BASELINE_LABELS = {
    0: "severe gait impairment",
    1: "moderate gait impairment",
    2: "mild gait impairment",
    3: "normal gait",
}


def baseline_labels_in_order() -> list[str]:
    return [FGA_BASELINE_LABELS[c] for c in sorted(FGA_BASELINE_LABELS)]


def classnames_in_order() -> list[str]:
    return [FGA_CLASSNAMES[c] for c in sorted(FGA_CLASSNAMES)]


def zeroshot_in_order() -> list[str]:
    return [FGA_ZEROSHOT[c] for c in sorted(FGA_ZEROSHOT)]
