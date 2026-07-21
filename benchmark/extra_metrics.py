#!/usr/bin/env python3
"""extra_metrics.py — additional effect measures beyond PSNR/SSIM/LPIPS.

A. Coverage stratification: per held-out view, temporal distance to the nearest
   training frame in each arm; does curation lose exactly where its non-uniform
   spacing leaves local coverage sparser than uniform's?
B. SfM quality from the chosen COLMAP models: mean reprojection error, mean
   track length, #3D points, observations per registered image.
C. Paired Wilcoxon signed-rank tests on per-view PSNR deltas.
Plus worst-case (tail) statistics per arm.

Reads only existing artifacts; writes benchmark/results/extra_metrics.json.
"""

import json
import struct
from pathlib import Path

import numpy as np
from scipy import stats as sps

from make_tables import common_test_paired, paper_scene_list

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent


def read_points3d_stats(path: Path):
    """Mean reprojection error / track length / #points from points3D.bin."""
    errs, tracks = [], 0
    n_pts = 0
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(8 + 24 + 3)                      # id, xyz, rgb
            (err,) = struct.unpack("<d", f.read(8))
            (tl,) = struct.unpack("<Q", f.read(8))
            f.seek(tl * 8, 1)
            errs.append(err)
            tracks += tl
            n_pts += 1
    return {"n_points": n_pts,
            "mean_reproj_err": float(np.mean(errs)) if errs else None,
            "mean_track_len": tracks / n_pts if n_pts else None,
            "total_obs": tracks}


def scene_analysis(scene):
    sd = ROOT / "data" / "scenes" / scene
    meta = json.loads((sd / "prepare_meta.json").read_text())
    fps = meta["fps"]
    test_idx = {r["image"]: r["frame_idx"] for r in meta["test_frames"]}
    train_idx = {"ours": np.array(sorted(meta["ours_frames"])),
                 "uniform": np.array(sorted(meta["uniform_frames"]))}

    paired = common_test_paired(scene)
    if paired is None:
        return None

    views = []
    for name, o, u in paired["views"]:
        t = test_idx.get(name)
        if t is None:
            continue
        d = {arm: float(np.min(np.abs(train_idx[arm] - t)) / fps)
             for arm in ("ours", "uniform")}
        views.append({
            "name": name,
            "dpsnr": o["psnr"] - u["psnr"],
            "d_ours_s": d["ours"], "d_unif_s": d["uniform"],
            "psnr_ours": o["psnr"], "psnr_unif": u["psnr"],
        })
    if not views:
        return None

    dpsnr = np.array([v["dpsnr"] for v in views])
    ddist = np.array([v["d_ours_s"] - v["d_unif_s"] for v in views])
    sparser = ddist > 0        # ours' nearest train frame is farther
    rec = {"scene": scene, "n_views": len(views)}

    # A: coverage stratification
    rec["coverage"] = {
        "spearman_ddist_dpsnr": (float(sps.spearmanr(ddist, dpsnr).statistic)
                                 if len(views) > 5 else None),
        "n_ours_sparser": int(sparser.sum()),
        "mean_dpsnr_ours_sparser": float(dpsnr[sparser].mean()) if sparser.any() else None,
        "mean_dpsnr_ours_denser": float(dpsnr[~sparser].mean()) if (~sparser).any() else None,
        "mean_d_ours_s": float(np.mean([v["d_ours_s"] for v in views])),
        "mean_d_unif_s": float(np.mean([v["d_unif_s"] for v in views])),
    }

    # tails / worst case
    po = np.array([v["psnr_ours"] for v in views])
    pu = np.array([v["psnr_unif"] for v in views])
    rec["tail"] = {
        "p10_ours": float(np.percentile(po, 10)),
        "p10_unif": float(np.percentile(pu, 10)),
        "min_ours": float(po.min()), "min_unif": float(pu.min()),
        "collapsed_ours": int((po < 16).sum()), "collapsed_unif": int((pu < 16).sum()),
    }

    # C: Wilcoxon signed-rank on per-view deltas
    if len(views) > 10 and not np.allclose(dpsnr, 0):
        w = sps.wilcoxon(dpsnr)
        rec["wilcoxon_p"] = float(w.pvalue)
    else:
        rec["wilcoxon_p"] = None

    # B: SfM quality of the chosen models
    rec["sfm"] = {}
    for arm in ("ours", "uniform"):
        p3d = sd / "work" / arm / "sparse" / "0" / "points3D.bin"
        if p3d.exists():
            s = read_points3d_stats(p3d)
            res = json.loads((BENCH / "results" / f"{scene}.json").read_text())
            chosen = res["arms"][arm]["colmap"].get("chosen") or {}
            n_reg = chosen.get("n_registered") or 1
            s["obs_per_image"] = s["total_obs"] / n_reg
            rec["sfm"][arm] = s
    return rec


def main():
    scenes = paper_scene_list() or []
    out = {"scenes": {}}
    for s in scenes:
        try:
            r = scene_analysis(s)
        except Exception as e:
            print(f"{s}: skipped ({e})")
            continue
        if r:
            out["scenes"][s] = r
            c, t = r["coverage"], r["tail"]
            print(f"{s}: n={r['n_views']} spearman={c['spearman_ddist_dpsnr']} "
                  f"dPSNR[sparser]={c['mean_dpsnr_ours_sparser']} "
                  f"dPSNR[denser]={c['mean_dpsnr_ours_denser']} "
                  f"wilcoxon_p={r['wilcoxon_p']}")

    # pooled coverage summary
    allv = [(v, s) for s, r in out["scenes"].items() for v in [r]]
    sp_all, de_all, n_sp = [], [], 0
    for s, r in out["scenes"].items():
        c = r["coverage"]
        if c["mean_dpsnr_ours_sparser"] is not None:
            sp_all.append(c["mean_dpsnr_ours_sparser"])
        if c["mean_dpsnr_ours_denser"] is not None:
            de_all.append(c["mean_dpsnr_ours_denser"])
    out["pooled"] = {
        "mean_dpsnr_where_ours_sparser": float(np.mean(sp_all)) if sp_all else None,
        "mean_dpsnr_where_ours_denser": float(np.mean(de_all)) if de_all else None,
        "sfm": {},
    }
    for arm in ("ours", "uniform"):
        vals = {k: [] for k in ("mean_reproj_err", "mean_track_len", "n_points", "obs_per_image")}
        for r in out["scenes"].values():
            a = r["sfm"].get(arm)
            if a:
                for k in vals:
                    vals[k].append(a[k])
        out["pooled"]["sfm"][arm] = {k: float(np.mean(v)) for k, v in vals.items() if v}

    (BENCH / "results" / "extra_metrics.json").write_text(json.dumps(out, indent=1))
    print("\npooled:", json.dumps(out["pooled"], indent=1))


if __name__ == "__main__":
    main()
