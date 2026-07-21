#!/usr/bin/env python3
"""make_blur.py — synthesize exposure blur from 60 fps video (GoPro/REDS style).

Blurry frame = temporal average of N consecutive frames, computed in LINEAR
space (sRGB -> gamma-2.2 linear -> mean -> back), which approximates a longer
shutter time. All levels (including blur-0 / N=1) are re-encoded through the
identical decode -> encode path (libx264 CRF 18, veryfast, 60 fps) so codec
conditions are constant across levels; only the simulated shutter differs.

  python make_blur.py --scene dl3dv_032dee9f --levels 1,2,4,8

Outputs data/scenes/<scene>_b{N-1... naming: b0=N1, b2=N2? no:}
Naming: level N -> suffix _b{N} with _b1 meaning the re-encoded original.
For the paper we call them blur-0/2/4/8 where blur-0 == _b1.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GAMMA = 2.2

# 8-bit sRGB -> linear float32
LUT_TO_LIN = ((np.arange(256, dtype=np.float32) / 255.0) ** GAMMA)
# 16-bit linear -> 8-bit sRGB
LUT_TO_SRGB = np.clip(
    255.0 * ((np.arange(65536, dtype=np.float32) / 65535.0) ** (1.0 / GAMMA)),
    0, 255).astype(np.uint8)


def synth(src: Path, dst: Path, window: int):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dst.parent.mkdir(parents=True, exist_ok=True)

    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
         "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", str(dst)],
        stdin=subprocess.PIPE)

    ring = []           # linear float32 frames in the current window
    acc = None          # running sum of the window
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lin = LUT_TO_LIN[frame]
        if acc is None:
            acc = np.zeros_like(lin)
        ring.append(lin)
        acc += lin
        if len(ring) < window:
            continue
        avg = acc / window
        out = LUT_TO_SRGB[(avg * 65535.0).astype(np.uint16)]
        enc.stdin.write(out.tobytes())
        acc -= ring.pop(0)
        written += 1
        if written % 500 == 0:
            print(f"  {dst.name}: {written} frames", file=sys.stderr, flush=True)
    cap.release()
    enc.stdin.close()
    enc.wait()
    if enc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {dst}")
    print(f"{dst}: {written} frames (window={window}, src {n})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="source scene (sharp 60fps)")
    ap.add_argument("--levels", default="1,2,4,8")
    args = ap.parse_args()

    src = ROOT / "data" / "scenes" / args.scene / "input.mp4"
    if not src.exists():
        raise SystemExit(f"{src} not found")
    for n in [int(x) for x in args.levels.split(",")]:
        dst = ROOT / "data" / "scenes" / f"{args.scene}_b{n}" / "input.mp4"
        if dst.exists():
            print(f"skip {dst} (exists)", flush=True)
            continue
        synth(src, dst, n)


if __name__ == "__main__":
    main()
