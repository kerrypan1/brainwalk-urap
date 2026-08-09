import argparse
import cv2
import numpy as np
import subprocess
from pathlib import Path

# File Summary
# ------------
# Builds fixed-length clips from raw gait videos
# For each input video, it:
# 1) computes simple frame-difference motion scores,
# 2) picks 3 start times (prefer high-motion, non-overlapping windows; repeats if needed),
# 3) uses ffmpeg to cut clips and enforce output FPS
#
# Outputs are written under:
#   brainwalk-vlm-(model1)/clips/clips_fps_<fps>_length_<clip_len>/<video_id>/clip_i.mp4
#
# Paths are resolved from repo structure so cwd does not matter
SCRIPT_DIR = Path(__file__).resolve().parent
NEW_DIR = SCRIPT_DIR.parent
REPO_ROOT = NEW_DIR.parent
INPUT_DIR = REPO_ROOT / "data" / "bath_fw"  # folder with many .mp4
OUTPUT_DIR = NEW_DIR / "clips"               # where to create clips_fps_* folders

def motion_scores(video_path, sample_fps):
    """Return per-sampled-frame times and motion intensity scores for one video"""
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Sample near `sample_fps` by stepping every N source frames
    stride = max(1, int(src_fps // sample_fps))

    prev = None
    motions = []
    times = []
    frame_i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_i % stride != 0:
            frame_i += 1
            continue

        # Downsample spatially to make frame diff computation faster
        gray = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)

        if prev is None:
            motion = 0.0
        else:
            diff = cv2.absdiff(gray, prev)
            motion = float((diff > 15).mean())

        motions.append(motion)
        times.append(frame_i / src_fps)

        prev = gray
        frame_i += 1

    cap.release()

    # Treat extreme spikes as likely noise or transitions and zero them out
    max_m = max(motions) if motions else 0.0

    # It's worth noting that this is a hardcoded value that was determined by trial and error
    HIGH = 0.35 * max_m

    scores = []
    for m in motions:
        if m > HIGH:
            scores.append(0.0)
        else:
            scores.append(m)

    return np.array(times, dtype=np.float32), np.array(scores, dtype=np.float32)

NUM_CLIPS = 3


def ensure_three_starts(starts):
    """
    Always return exactly NUM_CLIPS start times for downstream ffmpeg cuts.
    Repeats the last chosen start when the video is too short for three windows.
    """
    if not starts:
        starts = [0.0]
    out = [float(s) for s in starts]
    pad_from = out[-1]
    while len(out) < NUM_CLIPS:
        out.append(pad_from)
    return out[:NUM_CLIPS]


def pick_top3_no_overlap_else_allow(times, scores, clip_len, skip_front_ratio=0.0):
    """
    Pick exactly 3 clip starts (padding with repeats when needed).
    Preference:
    - avoid first portion of video,
    - maximize summed motion within the clip window,
    - enforce non-overlap when possible, then allow overlap, then repeat starts
    """
    if len(times) == 0:
        return ensure_three_starts([])

    t_end = float(times[-1])
    if t_end <= clip_len:
        return ensure_three_starts([0.0])

    t_min = skip_front_ratio * t_end

    candidates = []
    for t0 in times:
        t0 = float(t0)
        if t0 < t_min or t0 + clip_len > t_end:
            continue
        mask = (times >= t0) & (times < (t0 + clip_len))
        candidates.append((float(scores[mask].sum()), t0))

    if not candidates:
        return ensure_three_starts([max(0.0, (t_end - clip_len) / 2.0)])

    candidates.sort(reverse=True)

    picked = []
    for score, t0 in candidates:
        # First pass enforces non-overlap
        if all(abs(t0 - s) >= clip_len for s in picked):
            picked.append(t0)
        if len(picked) == NUM_CLIPS:
            return ensure_three_starts(sorted(picked))

    # Second pass allows overlap if needed to reach 3 distinct candidates
    for score, t0 in candidates:
        if t0 not in picked:
            picked.append(t0)
        if len(picked) == NUM_CLIPS:
            break

    return ensure_three_starts(sorted(picked))

def cut_clip(video, t0, length, out_path, fps):
    """Use ffmpeg to cut one clip at start `t0` with fixed duration and output fps"""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{t0:.3f}",
        "-i", str(video),
        "-t", f"{length:.3f}",
        "-vf", f"fps={fps}",
        "-loglevel", "error",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fps", type=float, required=True)
    p.add_argument("--clip_len", type=float, required=True)
    p.add_argument(
        "--stems",
        nargs="+",
        default=None,
        help="optional video stems to process (for example: 0243_2 0270_1)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing non-empty clip files instead of skipping them",
    )
    args = p.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.clip_len <= 0:
        raise ValueError("--clip_len must be > 0")
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"INPUT_DIR does not exist: {INPUT_DIR}")

    out_root = OUTPUT_DIR / f"clips_fps_{args.fps}_length_{args.clip_len}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Process each source video independently
    videos = sorted(INPUT_DIR.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No .mp4 files found in INPUT_DIR: {INPUT_DIR}")
    if args.stems:
        requested = set(args.stems)
        videos = [video for video in videos if video.stem in requested]
        found = {video.stem for video in videos}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"Requested video stems not found in {INPUT_DIR}: {missing}")

    for idx, video in enumerate(videos, start=1):
        vid_dir = out_root / video.stem
        expected = [vid_dir / f"clip_{i}.mp4" for i in range(1, NUM_CLIPS + 1)]
        if not args.overwrite and all(p.exists() and p.stat().st_size > 0 for p in expected):
            print(f"[{idx}/{len(videos)}] cached, skip: {video.name}")
            continue

        times, scores = motion_scores(video, args.fps)
        starts = pick_top3_no_overlap_else_allow(times, scores, args.clip_len)

        vid_dir.mkdir(exist_ok=True)

        # Emit three clips per input video (starts may repeat on short videos)
        for i, t0 in enumerate(starts, start=1):
            out_clip = vid_dir / f"clip_{i}.mp4"
            if not args.overwrite and out_clip.exists() and out_clip.stat().st_size > 0:
                continue
            cut_clip(video, float(t0), args.clip_len, out_clip, args.fps)
            if not out_clip.exists() or out_clip.stat().st_size == 0:
                raise RuntimeError(f"ffmpeg produced empty clip: {out_clip}")

        print(
            f"[{idx}/{len(videos)}] processed:",
            video.name,
            "starts:",
            [round(float(x), 2) for x in starts],
        )

if __name__ == "__main__":
    main()
