# Vendored gsplat examples — patch log

`benchmark/gsplat_examples/` is a copy of `gsplat v1.5.3`'s `examples/`
(simple_trainer.py, utils.py, lib_bilagrid.py, datasets/). The following
minimal patches are applied. **Both benchmark arms (Ours / Uniform) are
trained with exactly the same code**, so none of these changes affect
fairness.

1. **simple_trainer.py**: viewer imports (viser / nerfview / gsplat_viewer)
   made optional via try/except; training runs headless with
   `--disable_viewer`. `from __future__ import annotations` added so the
   viewer type annotations are evaluated lazily.
2. **simple_trainer.py**: `fused_ssim` (a CUDA extension that could not be
   built in our environment due to a CUDA-toolkit / torch version mismatch)
   falls back to `utils.plain_ssim`, a mathematically identical plain-PyTorch
   SSIM (11x11 Gaussian, sigma = 1.5, valid padding), used in the training
   loss. Identical for both arms.
3. **utils.py**: `plain_ssim` added (implementation of patch 2).
4. **datasets/colmap.py**: the train/val split of `Dataset` uses the explicit
   `test_` filename prefix when such images exist (our held-out test views are
   jointly registered in the COLMAP model); otherwise the original every-Nth
   behavior is preserved.
5. **simple_trainer.py**: the hardcoded random seed 42 was promoted to
   `Config.seed` (default 42, settable via `--seed`). All main-paper runs use
   the default 42; seeds 43/44 are used only by the seed-robustness check.
   Both arms always share the same seed.
