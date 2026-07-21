#!/usr/bin/env python3
"""prepare_scene.py — video -> held-out test views + training sets for both arms.

Protocol (fairness first):
  * Test views: one frame every --test-interval seconds (default 2 s, starting at
    1 s). If the nominal frame is not the sharpest within +/-5 frames, it is
    replaced by the sharpest one in that window (Laplacian variance).
  * Test frames are completely excluded from BOTH arms' selection pools:
    - Ours: extract_frames.py runs on the untouched video (its algorithm is not
      modified); afterwards any selected keyframe within +/-5 frames of a chosen
      test frame is removed. (Method noted in FINAL_REPORT.)
    - Uniform baseline: same FINAL count as Ours, evenly spaced over all frames
      excluding the same +/-5 test neighborhoods, same resolution cap, same JPEG
      quality, and NO blur handling (standard ffmpeg-style extraction).
  * All images (train of both arms + test) are saved at the same max_dim (the
    ~2 MP cap chosen by extract_frames' analysis pass) so COLMAP can register
    them as a single camera and metrics are computed at one resolution.

Outputs under data/scenes/<scene>/:
  extract/            raw extract_frames.py output (images/, extract_stats.json)
  work/ours/images/   train_*.jpg + test_*.jpg
  work/uniform/images/ train_*.jpg + test_*.jpg
  prepare_meta.json
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEST_NEIGHBORHOOD = 5  # +/- frames around a test frame excluded from selection


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def gray_small(frame, analysis_dim=640):
    h, w = frame.shape[:2]
    scale = analysis_dim / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def save_capped(frame, max_dim, path, quality=95):
    h, w = frame.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--test-interval", type=float, default=2.0)
    ap.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    ap.add_argument("--test-import", default=None, metavar="SCENE",
                    help="reuse the test views (images + frame indices) of another "
                         "prepared scene instead of selecting our own -- used by the "
                         "synthetic-blur experiment so every blur level is evaluated "
                         "against the SAME sharp ground-truth views")
    ap.add_argument("--frame-offset", type=int, default=0,
                    help="frame-index offset mapping the imported test indices onto "
                         "this scene's timeline (blur window (N-1)//2)")
    args = ap.parse_args()

    scene_dir = ROOT / "data" / "scenes" / args.scene
    video = scene_dir / "input.mp4"
    if not video.exists():
        log(f"error: {video} not found")
        sys.exit(1)
    meta_path = scene_dir / "prepare_meta.json"
    if meta_path.exists() and not args.force:
        log(f"{args.scene}: already prepared, skipping (use --force to rebuild)")
        return

    # --- Step 1: Ours extraction (untouched algorithm, full video) -----------
    extract_dir = scene_dir / "extract"
    stats_path = extract_dir / "extract_stats.json"
    if not stats_path.exists() or args.force:
        log(f"[{args.scene}] running extract_frames.py ...")
        r = subprocess.run(
            [sys.executable, str(ROOT / "extract_frames.py"), str(video),
             "-o", str(extract_dir)],
            stdout=subprocess.PIPE, stderr=sys.stderr)
        if r.returncode != 0:
            log("error: extract_frames.py failed")
            sys.exit(1)
    stats = json.loads(stats_path.read_text())
    max_dim = stats["params"]["max_dim"]
    fps = stats["analysis"]["fps"]
    n_frames = stats["analysis"]["n_frames"]

    work = scene_dir / "work"
    if work.exists() and args.force:
        shutil.rmtree(work)
    arms = {a: work / a / "images" for a in ("ours", "uniform")}
    for d in arms.values():
        d.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if args.test_import:
        # --- Step 2': import test views from another prepared scene ---------
        src_dir = ROOT / "data" / "scenes" / args.test_import
        meta0 = json.loads((src_dir / "prepare_meta.json").read_text())
        log(f"[{args.scene}] importing {meta0['n_test']} test views from "
            f"{args.test_import} (offset {args.frame_offset})")
        test_records = []
        for rec in meta0["test_frames"]:
            name = rec["image"]
            shutil.copy2(src_dir / "work" / "ours" / "images" / name,
                         arms["ours"] / name)
            shutil.copy2(src_dir / "work" / "ours" / "images" / name,
                         arms["uniform"] / name)
            # map the sharp-video index onto this (blurred) timeline
            idx = min(max(rec["frame_idx"] - args.frame_offset, 0), n_frames - 1)
            test_records.append({"image": name, "frame_idx": int(idx),
                                 "sharpness": rec["sharpness"],
                                 "imported_from": args.test_import})
    else:
        # --- Step 2: choose test frames -------------------------------------
        log(f"[{args.scene}] selecting test views (every {args.test_interval}s) ...")
        nominal = []
        t = 1.0
        while True:
            idx = int(round(t * fps))
            if idx >= n_frames - TEST_NEIGHBORHOOD:
                break
            nominal.append(idx)
            t += args.test_interval

        # sharpness for all frames in the candidate windows (sequential decode of
        # only needed frames via seek; windows are small)
        test_frames = []  # (chosen_idx, nominal_idx, sharpness)
        for nom in nominal:
            lo, hi = max(0, nom - TEST_NEIGHBORHOOD), min(n_frames - 1, nom + TEST_NEIGHBORHOOD)
            best = None
            cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
            for idx in range(lo, hi + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                s = sharpness(gray_small(frame))
                if best is None or s > best[1]:
                    best = (idx, s, frame.copy())
            if best is not None:
                test_frames.append(best)

        test_records = []
        for k, (idx, s, frame) in enumerate(test_frames, 1):
            name = f"test_{k:05d}.jpg"
            for d in arms.values():
                save_capped(frame, max_dim, d / name)
            test_records.append({"image": name, "frame_idx": int(idx),
                                 "sharpness": round(s, 2)})
    test_set = {r["frame_idx"] for r in test_records}
    excluded = set()
    for idx in test_set:
        excluded.update(range(idx - TEST_NEIGHBORHOOD, idx + TEST_NEIGHBORHOOD + 1))
    log(f"[{args.scene}] {len(test_records)} test views")

    # --- Step 3: Ours arm = extracted keyframes minus test neighborhoods ----
    keyframes = stats["keyframes"]
    ours_kept, ours_removed = [], []
    for kf in keyframes:
        (ours_removed if kf["frame_idx"] in excluded else ours_kept).append(kf)
    for i, kf in enumerate(ours_kept, 1):
        shutil.copy2(extract_dir / "images" / kf["image"],
                     arms["ours"] / f"train_{i:05d}.jpg")
    n_train = len(ours_kept)
    log(f"[{args.scene}] ours: {len(keyframes)} extracted, "
        f"{len(ours_removed)} removed near test views, {n_train} kept")

    # --- Step 4: Uniform baseline, same count, sequential decode ------------
    # container metadata may overcount decodable frames; only pick from frames the
    # extractor actually decoded (keyframe indices prove decodability up to there)
    last_decodable = max(kf["frame_idx"] for kf in keyframes) if keyframes else n_frames - 1
    eligible = [i for i in range(last_decodable + 1) if i not in excluded]
    pick_pos = np.round(np.linspace(0, len(eligible) - 1, n_train)).astype(int)
    uniform_idx = sorted({eligible[p] for p in pick_pos})
    # (dedup can only occur if n_train > len(eligible), which would be degenerate)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    want = set(uniform_idx)
    saved = 0
    frame_idx = -1
    order = {idx: i + 1 for i, idx in enumerate(uniform_idx)}
    while saved < len(uniform_idx):
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx in want:
            save_capped(frame, max_dim, arms["uniform"] / f"train_{order[frame_idx]:05d}.jpg")
            saved += 1
    cap.release()
    log(f"[{args.scene}] uniform: {saved} frames saved")

    meta = {
        "scene": args.scene,
        "video": str(video),
        "n_frames": n_frames, "fps": fps,
        "max_dim": max_dim,
        "test_interval_s": args.test_interval,
        "test_neighborhood": TEST_NEIGHBORHOOD,
        "test_import": args.test_import,
        "frame_offset": args.frame_offset,
        "test_selection": "sharpest Laplacian within +/-5 frames of nominal 2s grid (start 1s)",
        "ours_exclusion_method": "post-hoc removal of extracted keyframes within +/-5 frames of a test frame",
        "n_test": len(test_records),
        "n_extracted_raw": len(keyframes),
        "n_ours_removed": len(ours_removed),
        "n_train": n_train,
        "n_uniform": saved,
        "test_frames": test_records,
        "ours_frames": [kf["frame_idx"] for kf in ours_kept],
        "uniform_frames": uniform_idx,
        "extract_matcher": stats["matcher"],
        "extract_gaps": len(stats["gap_positions"]),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    log(f"[{args.scene}] prepared: {n_train} train x2 arms + {len(test_records)} test")


if __name__ == "__main__":
    main()
