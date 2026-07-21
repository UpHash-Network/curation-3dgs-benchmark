#!/usr/bin/env python3
"""metrics.py — PSNR/SSIM/LPIPS over held-out test renders (runs in WSL, GPU).

Reads the side-by-side eval canvases written by simple_trainer's eval()
([GT | render] concatenated on width), splits them, and computes per-image and
mean PSNR / SSIM / LPIPS with the same torchmetrics configurations the trainer
uses (data_range=1, LPIPS alex normalize=True). This is the single metrics
source for the paper tables; the trainer's own val json is kept as cross-check.
"""

import argparse
import glob
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True, help="dir with val_step*_%04d.png canvases")
    ap.add_argument("--pattern", default="val_step14999_*.png")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    files = sorted(glob.glob(os.path.join(args.renders, args.pattern)))
    if not files:
        raise SystemExit(f"no renders match {args.pattern} in {args.renders}")

    per_image = []
    for f in files:
        canvas = imageio.imread(f)[..., :3].astype(np.float32) / 255.0
        h, w, _ = canvas.shape
        gt = torch.from_numpy(canvas[:, : w // 2]).permute(2, 0, 1)[None].to(device)
        render = torch.from_numpy(canvas[:, w // 2 :]).permute(2, 0, 1)[None].to(device)
        with torch.no_grad():
            per_image.append({
                "file": os.path.basename(f),
                "psnr": float(psnr_m(render, gt)),
                "ssim": float(ssim_m(render, gt)),
                "lpips": float(lpips_m(render, gt)),
            })

    out = {
        "n_images": len(per_image),
        "psnr": float(np.mean([m["psnr"] for m in per_image])),
        "ssim": float(np.mean([m["ssim"] for m in per_image])),
        "lpips": float(np.mean([m["lpips"] for m in per_image])),
        "per_image": per_image,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"PSNR {out['psnr']:.3f}  SSIM {out['ssim']:.4f}  LPIPS {out['lpips']:.3f} "
          f"({len(per_image)} views)")


if __name__ == "__main__":
    main()
