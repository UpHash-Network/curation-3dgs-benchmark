#!/usr/bin/env python3
"""extract_frames.py — parallax-based keyframe extraction for 3DGS pipelines.

[video] -> extract_frames.py -> [images/ + colmap_run.sh] -> COLMAP -> 3DGS training

Three passes:
  Pass 1  video analysis: derive flow_thresh / sharp_floor / buffer_size / max_dim
          from ~300 sampled points (sharpness distribution + inter-frame flow stats).
  Pass 2  extraction: trigger a keyframe when the median feature-track flow from the
          previous keyframe exceeds flow_thresh; save the sharpest frame in the recent
          buffer. Frames below sharp_floor never enter the buffer.
  Pass 3  COLMAP config generation: QC the extracted frames and select the matcher.

stdout: the last line is the absolute path of images/ (for piping). All logs go to stderr.
Dependencies: OpenCV + NumPy only.
"""

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# constants (see README "How it works")
FLOW_THRESH_BASE = 0.03      # image-diagonal ratio
FLOW_THRESH_MIN = 0.012
FLOW_THRESH_MAX = 0.08
KEYFRAME_COUNT_MIN = 30
KEYFRAME_COUNT_MAX = 400
BUFFER_MIN = 4
BUFFER_MAX = 15
MAX_PIXELS = 2_000_000       # ~2 MP cap for saved frames
GAP_PARALLAX = 0.12          # inter-keyframe parallax above this = gap
ANALYSIS_SAMPLES = 300
EXHAUSTIVE_MAX_IMAGES = 120
LK_MIN_TRACKS = 12           # below this the track is considered lost


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def gray_small(frame, analysis_dim):
    h, w = frame.shape[:2]
    scale = analysis_dim / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def median_track_flow(prev_gray, cur_gray):
    """Median displacement of LK-tracked features, as a ratio of the image diagonal.

    Returns (flow_ratio, n_tracked). flow_ratio is None on tracking failure.
    """
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.01,
                                  minDistance=8, blockSize=7)
    if pts is None or len(pts) < LK_MIN_TRACKS:
        return None, 0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, cur_gray, pts, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if nxt is None:
        return None, 0
    good = st.reshape(-1) == 1
    if good.sum() < LK_MIN_TRACKS:
        return None, int(good.sum())
    d = np.linalg.norm((nxt - pts).reshape(-1, 2)[good], axis=1)
    diag = float(np.hypot(*prev_gray.shape))
    return float(np.median(d)) / diag, int(good.sum())


# ---------------------------------------------------------------------------
# Pass 1 — analysis

def analyze(video_path, analysis_dim):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log(f"error: cannot open video: {video_path}")
        sys.exit(1)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_samples = min(ANALYSIS_SAMPLES, max(2, n_frames - 1))
    idxs = np.linspace(0, max(0, n_frames - 2), n_samples).astype(int)

    sharps = []
    flows = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok1, f1 = cap.read()
        ok2, f2 = cap.read()
        if not ok1:
            continue
        g1 = gray_small(f1, analysis_dim)
        sharps.append(sharpness(g1))
        if ok2:
            g2 = gray_small(f2, analysis_dim)
            fl, _ = median_track_flow(g1, g2)
            if fl is not None:
                flows.append(fl)
    cap.release()

    if not sharps:
        log("error: could not read any frames for analysis")
        sys.exit(1)

    sharps = np.array(sharps)
    per_frame_flow = float(np.mean(flows)) if flows else 0.0
    total_motion = per_frame_flow * n_frames

    # flow_thresh: base 0.03; only adjusted (within 0.012-0.08) when the estimated
    # keyframe count would fall outside 30-400
    flow_thresh = FLOW_THRESH_BASE
    est = total_motion / flow_thresh if flow_thresh > 0 else 0
    if est and not (KEYFRAME_COUNT_MIN <= est <= KEYFRAME_COUNT_MAX):
        target = min(max(est, KEYFRAME_COUNT_MIN), KEYFRAME_COUNT_MAX)
        flow_thresh = min(max(total_motion / target, FLOW_THRESH_MIN), FLOW_THRESH_MAX)
        est = total_motion / flow_thresh

    # sharp_floor: the smaller of P30 sharpness and 0.6 x median
    sharp_floor = float(min(np.percentile(sharps, 30), 0.6 * np.median(sharps)))

    # buffer_size: half the estimated keyframe interval, clamped 4-15
    if est >= 1:
        interval = n_frames / est
        buffer_size = int(min(max(round(interval / 2), BUFFER_MIN), BUFFER_MAX))
    else:
        buffer_size = BUFFER_MIN

    # max_dim: cap saved resolution at ~2 MP
    long_edge = max(width, height)
    if width * height > MAX_PIXELS:
        max_dim = int(round(long_edge * (MAX_PIXELS / (width * height)) ** 0.5))
    else:
        max_dim = long_edge

    if total_motion < 0.5:
        log("warning: almost no camera motion detected; expect very few keyframes")

    return {
        "n_frames": n_frames, "fps": fps, "width": width, "height": height,
        "per_frame_flow": per_frame_flow, "total_motion": total_motion,
        "flow_thresh": flow_thresh, "sharp_floor": sharp_floor,
        "buffer_size": buffer_size, "max_dim": max_dim,
        "est_keyframes": int(round(est)),
        "sharp_median": float(np.median(sharps)),
        "sharp_p30": float(np.percentile(sharps, 30)),
    }


# ---------------------------------------------------------------------------
# Pass 2 — extraction

def extract(video_path, params, out_images, jpeg_quality, analysis_dim):
    cap = cv2.VideoCapture(str(video_path))
    flow_thresh = params["flow_thresh"]
    sharp_floor = params["sharp_floor"]
    buffer_size = params["buffer_size"]
    max_dim = params["max_dim"]

    out_images.mkdir(parents=True, exist_ok=True)

    buf = deque(maxlen=buffer_size)   # (frame_idx, sharpness, saved-size frame, analysis gray)
    key_gray = None                   # analysis-size gray of the reference keyframe
    saved = []                        # per-keyframe records
    n_saved = 0
    track_fail_events = []
    frame_idx = -1

    def save_frame(rec_idx, sharp, frame, trigger_flow, track_fail):
        nonlocal n_saved, key_gray
        n_saved += 1
        name = f"{n_saved:05d}.jpg"
        cv2.imwrite(str(out_images / name), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        saved.append({"image": name, "frame_idx": int(rec_idx),
                      "sharpness": round(float(sharp), 2),
                      "flow_from_prev": (round(float(trigger_flow), 4)
                                         if trigger_flow is not None else None),
                      "track_failure": bool(track_fail)})

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        g = gray_small(frame, analysis_dim)
        s = sharpness(g)

        # frames below sharp_floor never enter the buffer
        if s >= sharp_floor:
            h, w = frame.shape[:2]
            scale = max_dim / max(h, w)
            if scale < 1.0:
                small = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))),
                                   interpolation=cv2.INTER_AREA)
            else:
                small = frame
            buf.append((frame_idx, s, small, g))

        if key_gray is None:
            # first keyframe: first acceptable frame once the buffer has content
            if buf:
                bi, bs, bf, bg = max(buf, key=lambda t: t[1])
                save_frame(bi, bs, bf, None, False)
                key_gray = bg          # reference = the frame actually saved
                buf.clear()
            continue

        fl, _n = median_track_flow(key_gray, g)
        if fl is None:
            # tracking lost: force a keyframe from the buffer if possible
            track_fail_events.append(frame_idx)
            if buf:
                bi, bs, bf, bg = max(buf, key=lambda t: t[1])
                save_frame(bi, bs, bf, None, True)
                key_gray = bg          # reference = the frame actually saved
                buf.clear()
            else:
                key_gray = g           # nothing savable: restart tracking here
            continue

        if fl > flow_thresh and buf:
            bi, bs, bf, bg = max(buf, key=lambda t: t[1])
            save_frame(bi, bs, bf, fl, False)
            key_gray = bg              # reference = the frame actually saved
            buf.clear()

    cap.release()
    return saved, track_fail_events


# ---------------------------------------------------------------------------
# Pass 3 — COLMAP config generation

def make_colmap_config(saved, out_dir):
    n = len(saved)
    gap_positions = []
    for i, rec in enumerate(saved):
        f = rec["flow_from_prev"]
        if rec["track_failure"] or (f is not None and f > GAP_PARALLAX):
            gap_positions.append({"image": rec["image"], "index": i,
                                  "reason": ("track_failure" if rec["track_failure"]
                                             else f"parallax {f:.3f} > {GAP_PARALLAX}")})

    vocab_tree = os.environ.get("VOCAB_TREE_PATH", "")
    has_gaps = len(gap_positions) > 0

    if n <= EXHAUSTIVE_MAX_IMAGES:
        matcher = "exhaustive"
        reason = (f"{n} images <= {EXHAUSTIVE_MAX_IMAGES}: exhaustive matcher "
                  "(most robust; finishes in minutes at this scale)")
        matcher_cmd = 'colmap exhaustive_matcher --database_path "$DB"'
    elif not has_gaps:
        matcher = "sequential_o10"
        reason = (f"{n} images > {EXHAUSTIVE_MAX_IMAGES}, no parallax gaps: "
                  "sequential matcher, overlap=10 (fastest)")
        matcher_cmd = ('colmap sequential_matcher --database_path "$DB" '
                       '--SequentialMatching.overlap 10')
    else:
        matcher = "sequential_o20"
        reason = (f"{n} images > {EXHAUSTIVE_MAX_IMAGES}, {len(gap_positions)} "
                  "parallax gap(s): sequential matcher, overlap=20")
        matcher_cmd = ('colmap sequential_matcher --database_path "$DB" '
                       '--SequentialMatching.overlap 20')
        if vocab_tree:
            matcher += "_loop"
            reason += "; loop detection enabled (VOCAB_TREE_PATH set)"
            matcher_cmd += (' --SequentialMatching.loop_detection 1 '
                            f'--SequentialMatching.vocab_tree_path "{vocab_tree}"')

    script = f"""#!/usr/bin/env bash
# Auto-generated by extract_frames.py
# Matcher selection: {reason}
set -e
cd "$(dirname "$0")"
DB=colmap.db

colmap feature_extractor \\
    --database_path "$DB" \\
    --image_path images \\
    --ImageReader.single_camera 1 \\
    --ImageReader.camera_model SIMPLE_RADIAL

{matcher_cmd}

mkdir -p sparse
colmap mapper \\
    --database_path "$DB" \\
    --image_path images \\
    --output_path sparse
"""
    sh_path = out_dir / "colmap_run.sh"
    with open(sh_path, "w", newline="\n") as f:
        f.write(script)
    try:
        os.chmod(sh_path, 0o755)
    except OSError:
        pass
    return {"matcher": matcher, "reason": reason, "gap_positions": gap_positions}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Parallax-based keyframe extraction for 3DGS pipelines. "
                    "All parameters are auto-derived; the only required argument "
                    "is the video path.")
    ap.add_argument("video", help="input video file (mp4/mov/...)")
    ap.add_argument("-o", "--out", default=None,
                    help="output directory (default: <video_stem>_extracted next to video)")
    ap.add_argument("--flow-thresh", type=float, default=None,
                    help="keyframe parallax threshold (image-diagonal ratio)")
    ap.add_argument("--sharp-floor", type=float, default=None,
                    help="sharpness cutoff (absolute Laplacian variance)")
    ap.add_argument("--buffer-size", type=int, default=None,
                    help="best-of buffer length")
    ap.add_argument("--max-dim", type=int, default=None,
                    help="long edge of saved images")
    ap.add_argument("--analysis-dim", type=int, default=640,
                    help="long edge of analysis images (default 640)")
    ap.add_argument("--jpeg-quality", type=int, default=95,
                    help="saved jpg quality (default 95)")
    args = ap.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        log(f"error: video not found: {video_path}")
        sys.exit(1)
    out_dir = Path(args.out).resolve() if args.out else \
        video_path.parent / f"{video_path.stem}_extracted"
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"[pass 1] analyzing {video_path.name} ...")
    params = analyze(video_path, args.analysis_dim)
    for k in ("flow_thresh", "sharp_floor", "buffer_size", "max_dim"):
        ov = getattr(args, k.replace("-", "_"), None)
        if ov is not None:
            params[k] = ov
            log(f"  override: {k} = {ov}")
    log(f"  frames={params['n_frames']} fps={params['fps']:.2f} "
        f"{params['width']}x{params['height']}")
    log(f"  flow_thresh={params['flow_thresh']:.4f} "
        f"sharp_floor={params['sharp_floor']:.1f} "
        f"buffer_size={params['buffer_size']} max_dim={params['max_dim']} "
        f"(est ~{params['est_keyframes']} keyframes)")

    log("[pass 2] extracting keyframes ...")
    images_dir = out_dir / "images"
    saved, track_fails = extract(video_path, params, images_dir,
                                 args.jpeg_quality, args.analysis_dim)
    log(f"  saved {len(saved)} keyframes -> {images_dir}")
    if len(saved) < 10:
        log("warning: almost no camera motion — very few keyframes extracted")

    log("[pass 3] generating COLMAP config ...")
    qc = make_colmap_config(saved, out_dir)
    log(f"  matcher: {qc['matcher']} ({len(qc['gap_positions'])} gaps)")

    stats = {
        "video": str(video_path),
        "params": {k: params[k] for k in
                   ("flow_thresh", "sharp_floor", "buffer_size", "max_dim")},
        "analysis": {k: params[k] for k in
                     ("n_frames", "fps", "width", "height", "per_frame_flow",
                      "total_motion", "est_keyframes", "sharp_median", "sharp_p30")},
        "n_extracted": len(saved),
        "matcher": qc["matcher"],
        "matcher_reason": qc["reason"],
        "gap_positions": qc["gap_positions"],
        "track_failure_frames": track_fails,
        "keyframes": saved,
    }
    with open(out_dir / "extract_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # stdout contract: last line is the absolute images/ path
    print(str(images_dir))


if __name__ == "__main__":
    main()
