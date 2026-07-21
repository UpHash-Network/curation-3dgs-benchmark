#!/usr/bin/env python3
"""run_benchmark.py — one-scene A/B benchmark runner.

Per scene: prepare (if needed) -> per arm [ours, uniform]:
  COLMAP (identical settings for both arms, matcher chosen by extract_frames'
  own QC; automatic fallback sequential -> vocab_tree -> exhaustive when the
  registration rate is < 50%) -> gsplat simple_trainer in WSL (15k iters,
  default hyperparameters, seed 42) -> metrics.py (PSNR/SSIM/LPIPS on held-out
  test views) -> results/<scene>.json.

Re-runnable: completed stages are detected from their outputs and skipped.
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
COLMAP = os.environ.get("COLMAP_BIN", r"C:\Tools\colmap\COLMAP.bat")
# COLMAP >= 3.12 uses faiss-format vocab trees; the classic demuc.de .bin files
# are incompatible (matchers fail with 0 matches). Default to the faiss tree.
VOCAB_TREE = os.environ.get(
    "VOCAB_TREE_PATH",
    str(ROOT / "tools" / "vocab_tree_faiss_flickr100K_words256K.bin"))
WSL_VENV = "~/fx-venv"
TRAIN_STEPS = 15000
FINAL_STEP = TRAIN_STEPS - 1  # simple_trainer evaluates at step max_steps-1


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def sh(cmd, cwd=None, capture=False):
    log("  $ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], cwd=cwd,
                       stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.STDOUT if capture else None)
    return r


def wsl_path(p: Path) -> str:
    s = str(p).replace("\\", "/")
    return "/mnt/" + s[0].lower() + s[2:]


def wsl_run(command: str, capture=False):
    """Run a bash command inside WSL with the fx-venv activated."""
    full = f"source {WSL_VENV}/bin/activate && {command}"
    log(f"  $ wsl: {command[:160]}")
    r = subprocess.run(["wsl", "-d", "Ubuntu-22.04", "--", "bash", "-c", full],
                       stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.STDOUT if capture else None)
    return r


# --------------------------------------------------------------------------
# minimal COLMAP images.bin reader (registered image names only)

def read_registered_images(images_bin: Path):
    names = []
    with open(images_bin, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(4 + 32 + 24 + 4)  # image_id, qvec(4d), tvec(3d), camera_id
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00" or c == b"":
                    break
                name += c
            names.append(name.decode("utf-8"))
            (npts,) = struct.unpack("<Q", f.read(8))
            f.seek(npts * 24, 1)
    return names


# --------------------------------------------------------------------------
# COLMAP stage

def matcher_cmds(matcher: str, db: Path):
    """Return the matcher command for a given plan name."""
    if matcher == "exhaustive":
        return [COLMAP, "exhaustive_matcher", "--database_path", db]
    if matcher.startswith("sequential"):
        overlap = "20" if "o20" in matcher else "10"
        cmd = [COLMAP, "sequential_matcher", "--database_path", db,
               "--SequentialMatching.overlap", overlap]
        if matcher.endswith("_loop") and Path(VOCAB_TREE).exists():
            cmd += ["--SequentialMatching.loop_detection", "1",
                    "--SequentialMatching.vocab_tree_path", VOCAB_TREE]
        return cmd
    if matcher == "vocab_tree":
        return [COLMAP, "vocab_tree_matcher", "--database_path", db,
                "--VocabTreeMatching.vocab_tree_path", VOCAB_TREE]
    raise ValueError(matcher)


def run_colmap_arm(arm_dir: Path, initial_matcher: str, force=False):
    """COLMAP with fallback chain. Returns dict with status/registration info."""
    marker = arm_dir / "colmap_result.json"
    if marker.exists() and not force:
        return json.loads(marker.read_text())

    images_dir = arm_dir / "images"
    n_images = len(list(images_dir.glob("*.jpg")))

    # fallback order per the kickoff: initial -> sequential -> vocab_tree -> exhaustive
    chain = [initial_matcher]
    for fb in ["sequential_o20_loop", "vocab_tree", "exhaustive"]:
        if fb not in chain and fb.split("_")[0] != initial_matcher.split("_")[0]:
            chain.append(fb)
    if "exhaustive" not in chain:
        chain.append("exhaustive")

    attempts = []
    best = None  # (reg_rate, attempt_dir, model_dir, attempt_record)
    for matcher in chain:
        att_dir = arm_dir / f"colmap_{matcher}"
        att_dir.mkdir(parents=True, exist_ok=True)
        db = att_dir / "colmap.db"
        sparse = att_dir / "sparse"
        t0 = time.time()
        ok = True
        if not (sparse / "0" / "images.bin").exists():
            # remove the db AND any stale SQLite WAL/journal remnants (left by an
            # interrupted run; a fresh db next to an old -wal corrupts the session)
            for suffix in ("", "-wal", "-shm", "-journal"):
                p = Path(str(db) + suffix)
                if p.exists():
                    p.unlink()
            r = sh([COLMAP, "feature_extractor",
                    "--database_path", db, "--image_path", images_dir,
                    "--ImageReader.single_camera", "1",
                    "--ImageReader.camera_model", "SIMPLE_RADIAL"], capture=True)
            ok = r.returncode == 0
            if ok:
                r = sh(matcher_cmds(matcher, db), capture=True)
                ok = r.returncode == 0
            if ok:
                sparse.mkdir(exist_ok=True)
                r = sh([COLMAP, "mapper", "--database_path", db,
                        "--image_path", images_dir, "--output_path", sparse],
                       capture=True)
                ok = r.returncode == 0 and (sparse / "0" / "images.bin").exists()
        # pick largest model
        reg_names, model_dir = [], None
        if ok:
            for m in sorted(sparse.iterdir()):
                ib = m / "images.bin"
                if ib.exists():
                    names = read_registered_images(ib)
                    if len(names) > len(reg_names):
                        reg_names, model_dir = names, m
        n_test_total = len(list(images_dir.glob("test_*.jpg")))
        rec = {
            "matcher": matcher, "ok": bool(ok and model_dir),
            "n_images": n_images,
            "n_registered": len(reg_names),
            "reg_rate": len(reg_names) / n_images if n_images else 0.0,
            "n_test_total": n_test_total,
            "n_test_registered": sum(1 for n in reg_names if n.startswith("test_")),
            "time_s": round(time.time() - t0, 1),
            "model": str(model_dir) if model_dir else None,
        }
        attempts.append(rec)
        log(f"  [{matcher}] registered {rec['n_registered']}/{n_images} "
            f"({rec['reg_rate']:.0%}), test {rec['n_test_registered']}/{n_test_total}")
        if rec["ok"] and (best is None or rec["reg_rate"] > best[0]):
            best = (rec["reg_rate"], att_dir, model_dir, rec)
        if rec["ok"] and rec["reg_rate"] >= 0.5:
            break

    result = {"attempts": attempts}
    if best and best[0] >= 0.5:
        # stage the chosen model as arm_dir/sparse/0 for the gsplat parser
        dst = arm_dir / "sparse" / "0"
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(exist_ok=True)
        shutil.copytree(best[2], dst)
        result.update({"status": "ok", "chosen": best[3]})
    else:
        result.update({"status": "failed",
                       "chosen": best[3] if best else None,
                       "reason": "registration rate < 50% after all fallbacks"})
    marker.write_text(json.dumps(result, indent=2))
    return result


# --------------------------------------------------------------------------
# training + metrics (WSL)

def run_train_arm(arm_dir: Path, force=False):
    gs_dir = arm_dir / "gs"
    stats_json = gs_dir / "stats" / f"val_step{FINAL_STEP:04d}.json"
    train_marker = arm_dir / "train_result.json"
    if stats_json.exists() and train_marker.exists() and not force:
        return json.loads(train_marker.read_text())
    if gs_dir.exists() and force:
        shutil.rmtree(gs_dir)
    t0 = time.time()
    cmd = (
        "cd " + wsl_path(ROOT / "benchmark" / "gsplat_examples") + " && "
        "CUBLAS_WORKSPACE_CONFIG=:4096:8 python simple_trainer.py default "
        f"--data_dir {wsl_path(arm_dir)} --data_factor 1 "
        f"--result_dir {wsl_path(gs_dir)} "
        f"--max_steps {TRAIN_STEPS} --eval_steps {TRAIN_STEPS} "
        f"--save_steps {TRAIN_STEPS} --disable_viewer --disable_video"
    )
    r = wsl_run(cmd, capture=True)
    out = (r.stdout or b"").decode("utf-8", "replace")
    (arm_dir / "train_log.txt").write_text(out, encoding="utf-8")
    elapsed = round(time.time() - t0, 1)
    if r.returncode != 0 or not stats_json.exists():
        rec = {"status": "failed", "time_s": elapsed, "log_tail": out[-2000:]}
    else:
        val = json.loads(stats_json.read_text())
        rec = {"status": "ok", "time_s": elapsed, "trainer_val": val,
               "num_GS": val.get("num_GS")}
    train_marker.write_text(json.dumps(rec, indent=2))
    return rec


def run_metrics_arm(arm_dir: Path, force=False):
    out_json = arm_dir / "metrics.json"
    if out_json.exists() and not force:
        m = json.loads(out_json.read_text())
        m["status"] = "ok"
        return m
    cmd = (
        "python " + wsl_path(ROOT / "benchmark" / "metrics.py") +
        f" --renders {wsl_path(arm_dir / 'gs' / 'renders')}"
        f" --pattern 'val_step{FINAL_STEP}_*.png'"
        f" --out {wsl_path(out_json)}"
    )
    r = wsl_run(cmd, capture=True)
    if r.returncode != 0 or not out_json.exists():
        return {"status": "failed",
                "log": (r.stdout or b"").decode("utf-8", "replace")[-1000:]}
    m = json.loads(out_json.read_text())
    m["status"] = "ok"
    return m


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    scene_dir = ROOT / "data" / "scenes" / args.scene
    RESULTS.mkdir(exist_ok=True)
    result_path = RESULTS / f"{args.scene}.json"
    if result_path.exists() and not args.force:
        existing = json.loads(result_path.read_text())
        if existing.get("status") == "done":
            log(f"{args.scene}: already done, skipping")
            return

    t_start = time.time()
    # prepare
    r = subprocess.run([sys.executable, str(ROOT / "benchmark" / "prepare_scene.py"),
                        "--scene", args.scene], stderr=sys.stderr)
    if r.returncode != 0:
        result_path.write_text(json.dumps(
            {"scene": args.scene, "status": "failed", "stage": "prepare"}, indent=2))
        sys.exit(1)
    meta = json.loads((scene_dir / "prepare_meta.json").read_text())

    # COLMAP matcher plan from extract_frames' own QC (identical for both arms)
    stats = json.loads((scene_dir / "extract" / "extract_stats.json").read_text())
    initial_matcher = stats["matcher"]

    result = {"scene": args.scene, "status": "running", "prepare": meta,
              "initial_matcher": initial_matcher, "arms": {}}
    for arm in ("ours", "uniform"):
        arm_dir = scene_dir / "work" / arm
        log(f"[{args.scene}/{arm}] COLMAP ...")
        colmap_res = run_colmap_arm(arm_dir, initial_matcher, force=args.force)
        arm_rec = {"colmap": colmap_res}
        if colmap_res["status"] == "ok":
            log(f"[{args.scene}/{arm}] training (15k iters, seed 42) ...")
            train_res = run_train_arm(arm_dir, force=args.force)
            arm_rec["train"] = train_res
            if train_res["status"] == "ok":
                log(f"[{args.scene}/{arm}] metrics ...")
                arm_rec["metrics"] = run_metrics_arm(arm_dir, force=args.force)
        result["arms"][arm] = arm_rec
        result_path.write_text(json.dumps(result, indent=2))

    ok = all(result["arms"][a].get("metrics", {}).get("status") == "ok"
             for a in ("ours", "uniform"))
    result["status"] = "done" if ok else "partial"
    result["total_time_s"] = round(time.time() - t_start, 1)
    result_path.write_text(json.dumps(result, indent=2))
    log(f"[{args.scene}] {result['status']} in {result['total_time_s']/60:.1f} min")


if __name__ == "__main__":
    main()
