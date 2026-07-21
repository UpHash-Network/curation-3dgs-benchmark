#!/usr/bin/env python3
"""extra_perceptual.py — DISTS and NVIDIA-FLIP over the eval canvases (WSL).

Robustness check that the headline comparison is not an artifact of the metric
choice. Reads val_step*_%04d.png canvases (GT|render), writes per-arm means to
benchmark/results/perceptual.json for the paper scenes.
"""

import glob
import json
import os
import sys

import numpy as np
import torch
import imageio.v2 as imageio
from DISTS_pytorch import DISTS
import flip_evaluator as flip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = "val_step14999_*.png"

scenes = [ln.strip() for ln in
          open(os.path.join(ROOT, "benchmark", "paper_scenes.txt"))
          if ln.strip() and not ln.startswith("#")]

device = "cuda" if torch.cuda.is_available() else "cpu"
dists = DISTS().to(device)

out = {}
for scene in scenes:
    row = {}
    for arm in ("ours", "uniform"):
        rd = os.path.join(ROOT, "data", "scenes", scene, "work", arm, "gs", "renders")
        files = sorted(glob.glob(os.path.join(rd, FINAL)))
        if not files:
            continue
        dv, fv = [], []
        for f in files:
            canvas = imageio.imread(f)[..., :3].astype(np.float32) / 255.0
            h, w, _ = canvas.shape
            gt, render = canvas[:, : w // 2], canvas[:, w // 2:]
            tg = torch.from_numpy(gt).permute(2, 0, 1)[None].to(device)
            tr = torch.from_numpy(render).permute(2, 0, 1)[None].to(device)
            with torch.no_grad():
                dv.append(float(dists(tr, tg)))
            # flip_evaluator: evaluate(ref, test, dynamic-range) -> (map, mean, params)
            _, fmean, _ = flip.evaluate(gt, render, "LDR")
            fv.append(float(fmean))
        row[arm] = {"dists": float(np.mean(dv)), "flip": float(np.mean(fv)),
                    "n": len(files)}
    if row:
        out[scene] = row
        print(scene, json.dumps(row), flush=True)

path = os.path.join(ROOT, "benchmark", "results", "perceptual.json")
json.dump(out, open(path, "w"), indent=1)
print("wrote", path)
