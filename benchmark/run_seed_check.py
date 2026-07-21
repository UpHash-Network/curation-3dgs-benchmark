#!/usr/bin/env python3
"""run_seed_check.py — seed-robustness spot check.

Retrains both arms of two representative scenes (one curation-win, one
curation-loss) with seeds 43 and 44 (COLMAP models and images unchanged, so
the common test views are identical to the seed-42 run) and reports the
per-seed paired mean dPSNR. Outputs results/seed_check.json.
"""

import json
from pathlib import Path

from make_tables import common_test_paired
from run_benchmark import FINAL_STEP, TRAIN_STEPS, wsl_path, wsl_run, ROOT

BENCH = Path(__file__).resolve().parent
SCENES = ["chair", "dl3dv_2beaca31"]
SEEDS = [43, 44]


def train_and_measure(scene, arm, seed):
    arm_dir = ROOT / "data" / "scenes" / scene / "work" / arm
    gs_dir = arm_dir / f"gs_seed{seed}"
    stats_json = gs_dir / "stats" / f"val_step{FINAL_STEP:04d}.json"
    mj = arm_dir / f"metrics_seed{seed}.json"
    if not stats_json.exists():
        print(f"[train] {scene}/{arm} seed={seed}", flush=True)
        cmd = (
            "cd " + wsl_path(ROOT / "benchmark" / "gsplat_examples") + " && "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 python simple_trainer.py default "
            f"--data_dir {wsl_path(arm_dir)} --data_factor 1 "
            f"--result_dir {wsl_path(gs_dir)} --seed {seed} "
            f"--max_steps {TRAIN_STEPS} --eval_steps {TRAIN_STEPS} "
            f"--save_steps {TRAIN_STEPS} --disable_viewer --disable_video"
        )
        r = wsl_run(cmd, capture=True)
        if r.returncode != 0 or not stats_json.exists():
            print(f"  FAILED: {(r.stdout or b'')[-400:]}", flush=True)
            return False
    if not mj.exists():
        cmd = ("python " + wsl_path(ROOT / "benchmark" / "metrics.py") +
               f" --renders {wsl_path(gs_dir / 'renders')}"
               f" --pattern 'val_step{FINAL_STEP}_*.png'"
               f" --out {wsl_path(mj)}")
        r = wsl_run(cmd, capture=True)
        if r.returncode != 0 or not mj.exists():
            print(f"  metrics FAILED", flush=True)
            return False
    return True


def seed_delta(scene, seed):
    """Paired mean dPSNR over common views using metrics_seed<seed>.json."""
    import numpy as np
    from run_benchmark import read_registered_images
    per_arm = {}
    for arm in ("ours", "uniform"):
        arm_dir = ROOT / "data" / "scenes" / scene / "work" / arm
        names = sorted(n for n in read_registered_images(
            arm_dir / "sparse" / "0" / "images.bin") if n.startswith("test_"))
        src = (arm_dir / "metrics.json" if seed == 42
               else arm_dir / f"metrics_seed{seed}.json")
        per = json.loads(src.read_text())["per_image"]
        if len(names) != len(per):
            return None
        per_arm[arm] = dict(zip(names, per))
    common = sorted(set(per_arm["ours"]) & set(per_arm["uniform"]))
    d = [per_arm["ours"][n]["psnr"] - per_arm["uniform"][n]["psnr"] for n in common]
    return float(np.mean(d))


def main():
    out = {}
    for scene in SCENES:
        for seed in SEEDS:
            ok = all(train_and_measure(scene, arm, seed)
                     for arm in ("ours", "uniform"))
            if not ok:
                continue
        row = {}
        for seed in [42] + SEEDS:
            d = seed_delta(scene, seed)
            if d is not None:
                row[str(seed)] = round(d, 3)
        out[scene] = row
        print(f"{scene}: {row}", flush=True)
    (BENCH / "results" / "seed_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
