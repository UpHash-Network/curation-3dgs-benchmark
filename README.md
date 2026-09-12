# How Much Does Input Curation Alone Improve 3DGS from Casual Video?

Code release for the SIGGRAPH Asia 2026 poster
*"How Much Does Input Curation Alone Improve 3D Gaussian Splatting from
Casual Video?"* (Oshio et al., SA Posters '26, Kuala Lumpur).

This repository contains (1) the evaluated frame-extraction tool and (2) the
complete A/B benchmark protocol used in the paper, so that every number in the
abstract can be reproduced end to end.

```
[video] -> extract_frames.py -> [images/ + colmap_run.sh] -> COLMAP -> 3DGS
```

## Contents

| Path | Purpose |
|---|---|
| `extract_frames.py` | The evaluated selector: parallax-triggered, sharpness-filtered keyframe extraction with self-configured COLMAP scripts. Single file, OpenCV + NumPy only. See `README_tool.md`. |
| `benchmark/prepare_scene.py` | Video -> held-out test views (every 2 s, sharpest in ±5) + training sets for both arms (Ours / Uniform at the *same* frame budget, same resolution cap). |
| `benchmark/run_benchmark.py` | One-scene runner: COLMAP (identical settings for both arms, automatic fallback), gsplat `simple_trainer` (15 k iters, default hyperparameters, fixed seed), metrics. Resumable. |
| `benchmark/metrics.py` | PSNR / SSIM / LPIPS on the held-out views (same torchmetrics configs as the trainer). |
| `benchmark/make_tables.py` | Aggregation, common-test-view pairing, cluster-bootstrap 95% CIs, LaTeX table. |
| `benchmark/make_blur.py` | GoPro/REDS-style synthetic exposure blur (linear-space temporal averaging of 60 fps video; all levels re-encoded identically). |
| `benchmark/run_blur_batch.sh` | Dose-response experiment driver (blur-0/2/4/8, shared sharp test views). |
| `benchmark/extra_metrics.py` | Coverage-stratification analysis (per-view nearest-training-view distance), SfM statistics, Wilcoxon tests. |
| `benchmark/run_seed_check.py` | Seed-robustness spot check (seeds 42/43/44, both arms). |
| `benchmark/extra_perceptual.py` | DISTS / FLIP over the eval renders. |
| `benchmark/gsplat_examples/` | Vendored gsplat v1.5.3 examples with minimal patches (see `PATCHES.md`; identical for both arms). |
| `benchmark/results/` | Aggregated outputs backing the paper: `summary.csv`, `extra_metrics.json`, `seed_check.json`, `perceptual.json`. |

## Reproducing

1. **Environment.** Windows or Linux host with an NVIDIA GPU (paper: RTX 3080,
   10 GB). Install COLMAP (>= 3.12; note that vocabulary trees must be the
   faiss format, e.g. `vocab_tree_faiss_flickr100K_words256K.bin` from the
   COLMAP releases). Python 3.10+ with `benchmark/requirements.txt`; the
   training venv is created by `benchmark/setup_wsl_env.sh` (torch 2.4.1+cu124,
   gsplat 1.5.3 prebuilt wheel, torchmetrics).
2. **Data.** Place each scene as `data/scenes/<name>/input.mp4`. The paper uses
   the Tanks and Temples source videos (Barn, Horse, Ignatius), five
   DL3DV-Benchmark captures (hashes in `benchmark/paper_scenes.txt`; original
   videos from `DL3DV/DL3DV-ALL-video`), and two casual phone captures.
3. **Run.** `python benchmark/run_all.py` (scenes are processed independently
   and resumably), then `python benchmark/make_tables.py` and
   `python benchmark/extra_metrics.py`.
4. **Blur study.** `bash benchmark/run_blur_batch.sh cpu` then `... gpu`.

Fairness rules implemented by the protocol: identical frame budget per scene,
identical COLMAP configuration for both arms (chosen by the tool's own QC),
joint registration of held-out test frames, metrics computed only on test
views registered in *both* arms, fixed seed, and no blur handling in the
uniform arm (the standard `ffmpeg` recipe).

License: MIT (see LICENSE).

## Citation

```bibtex
@inproceedings{oshio2026curation,
  author    = {Oshio, Yuki and Fukiharu, Yuya and Shiki, Akane and
               Kiyoyama, Yukito and Kihara, Miki and Ito, Ai and Imai, Shota},
  title     = {How Much Does Input Curation Alone Improve 3D Gaussian
               Splatting from Casual Video?},
  booktitle = {SIGGRAPH Asia 2026 Posters (SA Posters '26)},
  year      = {2026},
  publisher = {ACM}
}
```
