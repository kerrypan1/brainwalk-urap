# run_vlm_on_clips.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# File Summary
# ------------
# Runs a selected VLM model over every generated clip under `clips/` and writes
# one text output per clip under `vlm_output/`
#
# Key responsibilities:
# - resolve model wrapper from --model,
# - resolve prompt file from model + ICL mode + dataset suffix,
# - run model safely on each clip,
# - skip already-generated outputs so runs can resume
#
# Put project root on sys.path before importing project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_NEW_DIR = Path(__file__).resolve().parents[1]
CLIPS_BASE = _NEW_DIR / "clips"
OUTPUTS_BASE = _NEW_DIR / "vlm_output"
PROMPTS_DIR = _NEW_DIR / "prompts"

MODEL_PROMPT_BASE = {
    "intern_l": "internvl",
    "intern_s": "internvl",
    "llava_l": "videollava",
    "llava_s": "videollava",
}

ICL_MODES = {
    "y":        "icl",
    "n":        "no-icl",
    "generate": "generate",
}


def ensure_dir(p: Path) -> None:
    """Create directory tree if it does not exist"""
    p.mkdir(parents=True, exist_ok=True)


def run_model_safe(model, video_path: Path) -> str:
    """Run one clip through the model and never raise to caller"""
    try:
        result = model.run(str(video_path))
        # Support wrappers that return plain string or (text, extra) tuple
        if isinstance(result, tuple) and len(result) >= 1:
            return result[0] or ""
        if isinstance(result, str):
            return result
        return ""
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


def load_prompt(model_key: str, icl_mode: str, dataset: str) -> str:
    """Load prompt text based on model family, ICL mode, and dataset tag"""
    base = MODEL_PROMPT_BASE[model_key]
    suffix = ICL_MODES[icl_mode]
    prompt_path = PROMPTS_DIR / f"{base}-prompt-{suffix}_{dataset}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def get_model(model_key: str):
    """Instantiate selected model wrapper with project defaults"""
    key = model_key.strip().lower()
    if key == "intern_l":
        from models.internvl import InternVL
        return "intern_l", InternVL(model_name="OpenGVLab/InternVL2-2B", num_frames=16)
    if key == "intern_s":
        from models.internvl_small import InternVLSmall
        return "intern_s", InternVLSmall(num_frames=16)
    if key == "llava_l":
        from models.videollava import VideoLLaVA
        return "llava_l", VideoLLaVA(model_id="llava-hf/LLaVA-NeXT-Video-7B-hf", num_frames=16)
    if key == "llava_s":
        from models.videollava_small import VideoLLaVASmall
        return "llava_s", VideoLLaVASmall(num_frames=16)
    raise ValueError('model must be one of: "intern_l", "intern_s", "llava_l", "llava_s"')


def main() -> None:
    ap = argparse.ArgumentParser()
    # Model and clip set selectors
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--clip_len", type=float, required=True)
    ap.add_argument("--icl", type=str, choices=["y", "n", "generate"], required=True,
                    help="y = ICL prompt, n = no-ICL prompt, generate = description-generation prompt")
    ap.add_argument("--dataset", type=str, required=True,
                    help="Prompt dataset/variant suffix (e.g., therapist, zeno, therapist1)")
    args = ap.parse_args()

    icl_mode = args.icl
    dataset = args.dataset
    model_name, model = get_model(args.model)

    # Inject prompt text directly into model wrapper for consistent use
    prompt_text = load_prompt(args.model.strip().lower(), icl_mode, dataset)
    model.prompt_text = prompt_text

    # Output namespace captures model family, ICL setting, and dataset variant
    icl_tag = {"y": "icl", "n": "noicl", "generate": "generate"}[icl_mode]
    output_name = f"{model_name}_{icl_tag}_{dataset}"

    clips_root = CLIPS_BASE / f"clips_fps_{args.fps}_length_{args.clip_len}"
    if not clips_root.exists():
        raise FileNotFoundError(f"Clips folder not found: {clips_root}")

    out_root = OUTPUTS_BASE / output_name / clips_root.name
    ensure_dir(out_root)

    clip_paths = sorted(clips_root.rglob("clip_*.mp4"))
    if not clip_paths:
        print(f"No clips found under {clips_root}")
        return

    # Process clip by clip while preserving folder hierarchy
    for clip_path in clip_paths:
        rel_parent = clip_path.parent.relative_to(clips_root)
        out_dir = out_root / rel_parent
        ensure_dir(out_dir)

        out_file = out_dir / clip_path.with_suffix(".txt").name
        if out_file.exists():
            # Skip existing outputs so interrupted jobs can resume
            continue

        text = run_model_safe(model, clip_path)
        out_file.write_text(text or "", encoding="utf-8")

        if text.startswith("[ERROR]"):
            print(f"Failed: {clip_path} -> {out_file}")
        else:
            print(f"Wrote: {out_file}")

    print(f"Done. Outputs in: {out_root}")


if __name__ == "__main__":
    main()
