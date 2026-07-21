#!/usr/bin/env python3
"""make_tables.py — aggregate results/*.json into LaTeX tables + CSV.

Outputs:
  benchmark/results/summary.csv
  paper/tables/main_results.tex   (booktabs; scene x {PSNR,SSIM,LPIPS,reg} x arms)
"""

import csv
import json
from pathlib import Path

from run_benchmark import read_registered_images

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
RESULTS = BENCH / "results"
TABLES = ROOT / "paper" / "tables"


def common_test_paired(scene):
    """Per-view paired metrics over the test views registered in BOTH arms.

    Returns {"views": [(name, ours_dict, unif_dict), ...]} or None.
    """
    per_arm = {}
    for arm in ("ours", "uniform"):
        arm_dir = ROOT / "data" / "scenes" / scene / "work" / arm
        ib = arm_dir / "sparse" / "0" / "images.bin"
        mj = arm_dir / "metrics.json"
        if not (ib.exists() and mj.exists()):
            return None
        names = sorted(n for n in read_registered_images(ib)
                       if n.startswith("test_"))
        per_image = json.loads(mj.read_text())["per_image"]
        if len(names) != len(per_image):
            return None
        per_arm[arm] = dict(zip(names, per_image))
    common = sorted(set(per_arm["ours"]) & set(per_arm["uniform"]))
    if not common:
        return None
    return {"views": [(n, per_arm["ours"][n], per_arm["uniform"][n])
                      for n in common]}


def bootstrap_delta_ci(scenes, metric, n_boot=10000, seed=42):
    """Cluster bootstrap (resample scenes, then views within each scene) of the
    mean-of-scene-means paired delta (ours - uniform) for one metric.

    Returns (mean_delta, lo95, hi95) or None.
    """
    import random
    per_scene = []
    for s in scenes:
        p = common_test_paired(s)
        if p:
            per_scene.append([o[metric] - u[metric] for _, o, u in p["views"]])
    if not per_scene:
        return None
    point = sum(sum(d) / len(d) for d in per_scene) / len(per_scene)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        tot = 0.0
        for _ in range(len(per_scene)):
            d = per_scene[rng.randrange(len(per_scene))]
            tot += sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d)
        stats.append(tot / len(per_scene))
    stats.sort()
    lo = stats[int(0.025 * n_boot)]
    hi = stats[int(0.975 * n_boot) - 1]
    return point, lo, hi


def common_test_metrics(scene):
    """Per-arm metric means over the test views registered in BOTH arms.

    The eval renders (val_step*_%04d.png) follow the sorted order of the
    registered test_* image names in each arm's chosen COLMAP model, which lets
    us map per-image metrics back to test-view names and intersect the sets.
    Returns {arm: {psnr, ssim, lpips}, "n_common": int} or None.
    """
    per_arm = {}
    for arm in ("ours", "uniform"):
        arm_dir = ROOT / "data" / "scenes" / scene / "work" / arm
        ib = arm_dir / "sparse" / "0" / "images.bin"
        mj = arm_dir / "metrics.json"
        if not (ib.exists() and mj.exists()):
            return None
        names = sorted(n for n in read_registered_images(ib)
                       if n.startswith("test_"))
        per_image = json.loads(mj.read_text())["per_image"]
        if len(names) != len(per_image):
            return None
        per_arm[arm] = dict(zip(names, per_image))
    common = sorted(set(per_arm["ours"]) & set(per_arm["uniform"]))
    if not common:
        return None
    out = {"n_common": len(common)}
    for arm in ("ours", "uniform"):
        for k in ("psnr", "ssim", "lpips"):
            out.setdefault(arm, {})[k] = sum(
                per_arm[arm][n][k] for n in common) / len(common)
    return out


def paper_scene_list():
    """Optional include-list (benchmark/paper_scenes.txt) for the paper table."""
    f = BENCH / "paper_scenes.txt"
    if not f.exists():
        return None
    return [ln.strip() for ln in f.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]


def load_results():
    include = paper_scene_list()
    rows = []
    ordered = sorted(RESULTS.glob("*.json"))
    if include:
        by_name = {p.stem: p for p in ordered}
        ordered = [by_name[s] for s in include if s in by_name]
    for p in ordered:
        r = json.loads(p.read_text())
        if r.get("status") not in ("done", "partial"):
            continue
        row = {"scene": r["scene"], "status": r["status"],
               "n_train": r.get("prepare", {}).get("n_train"),
               "n_test": r.get("prepare", {}).get("n_test")}
        # evaluate both arms on the SAME held-out views: intersection of the
        # test views registered in both arms (falls back to per-arm means)
        common = common_test_metrics(r["scene"])
        row["n_test_common"] = common["n_common"] if common else None
        for arm in ("ours", "uniform"):
            a = r.get("arms", {}).get(arm, {})
            m = (common[arm] if common else a.get("metrics", {}))
            c = a.get("colmap", {}).get("chosen") or {}
            t = a.get("train", {})
            row[f"{arm}_psnr"] = m.get("psnr")
            row[f"{arm}_ssim"] = m.get("ssim")
            row[f"{arm}_lpips"] = m.get("lpips")
            row[f"{arm}_reg"] = c.get("reg_rate")
            row[f"{arm}_test_reg"] = c.get("n_test_registered")
            row[f"{arm}_num_gs"] = t.get("num_GS")
            row[f"{arm}_train_time_s"] = t.get("time_s")
            row[f"{arm}_matcher"] = a.get("colmap", {}).get("chosen", {}).get("matcher") if a.get("colmap") else None
        rows.append(row)
    return rows


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(v, nd=2, pct=False):
    if v is None:
        return "--"
    if pct:
        return f"{100*v:.0f}\\%"
    return f"{v:.{nd}f}"


def make_latex(rows):
    complete = [r for r in rows if r["ours_psnr"] is not None
                and r["uniform_psnr"] is not None]
    lines = [
        "% Auto-generated by benchmark/make_tables.py -- do not edit by hand.",
        "\\begin{tabular}{l cc cc cc cc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{PSNR $\\uparrow$} & \\multicolumn{2}{c}{SSIM $\\uparrow$}"
        " & \\multicolumn{2}{c}{LPIPS $\\downarrow$} & \\multicolumn{2}{c}{Reg.\\ rate $\\uparrow$} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
        "Scene & Unif. & Ours & Unif. & Ours & Unif. & Ours & Unif. & Ours \\\\",
        "\\midrule",
    ]

    def cells(r):
        out = []
        for k, nd in (("psnr", 2), ("ssim", 3), ("lpips", 3)):
            u, o = r[f"uniform_{k}"], r[f"ours_{k}"]
            better_ours = (o is not None and u is not None and
                           ((o > u) if k != "lpips" else (o < u)))
            us, os_ = fmt(u, nd), fmt(o, nd)
            if better_ours:
                os_ = f"\\textbf{{{os_}}}"
            elif u is not None and o is not None:
                us = f"\\textbf{{{us}}}"
            out += [us, os_]
        u, o = r["uniform_reg"], r["ours_reg"]
        us, os_ = fmt(u, pct=True), fmt(o, pct=True)
        if u is not None and o is not None:
            if o > u:
                os_ = f"\\textbf{{{os_}}}"
            elif u > o:
                us = f"\\textbf{{{us}}}"
        out += [us, os_]
        return out

    for r in complete:
        name = r["scene"].replace("_", "\\_")
        lines.append(f"{name} & " + " & ".join(cells(r)) + " \\\\")

    if len(complete) > 1:
        lines.append("\\midrule")
        avg = {"scene": "Mean"}
        for arm in ("ours", "uniform"):
            for k in ("psnr", "ssim", "lpips", "reg"):
                avg[f"{arm}_{k}"] = mean([r[f"{arm}_{k}"] for r in complete])
        lines.append("Mean & " + " & ".join(cells(avg)) + " \\\\")
        # paired delta with cluster-bootstrap 95% CI (scenes, then views)
        ci_cells = []
        for k, nd in (("psnr", 2), ("ssim", 3), ("lpips", 3)):
            ci = bootstrap_delta_ci([r["scene"] for r in complete], k)
            ci_cells.append(
                f"\\multicolumn{{2}}{{c}}{{{ci[0]:+.{nd}f} "
                f"[{ci[1]:+.{nd}f}, {ci[2]:+.{nd}f}]}}" if ci else
                "\\multicolumn{2}{c}{--}")
        lines.append("$\\Delta$ (95\\% CI) & " + " & ".join(ci_cells) +
                     " & \\multicolumn{2}{c}{--} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    rows = load_results()
    if not rows:
        print("no results found")
        return
    RESULTS.mkdir(exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    fields = list(rows[0].keys())
    with open(RESULTS / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    (TABLES / "main_results.tex").write_text(make_latex(rows))

    # numeric macros for the paper (single source: results/*.json)
    complete = [r for r in rows if r["ours_psnr"] is not None
                and r["uniform_psnr"] is not None]
    if complete:
        m = {}
        for arm in ("ours", "uniform"):
            for k in ("psnr", "ssim", "lpips", "reg"):
                m[f"{arm}_{k}"] = mean([r[f"{arm}_{k}"] for r in complete])
        best = max(complete, key=lambda r: r["ours_psnr"] - r["uniform_psnr"])
        macros = [
            "% Auto-generated by benchmark/make_tables.py -- do not edit by hand.",
            f"\\newcommand{{\\NumScenes}}{{{len(complete)}}}",
            f"\\newcommand{{\\MeanPsnrOurs}}{{{m['ours_psnr']:.2f}}}",
            f"\\newcommand{{\\MeanPsnrUnif}}{{{m['uniform_psnr']:.2f}}}",
            f"\\newcommand{{\\MeanDeltaPsnr}}{{{m['ours_psnr']-m['uniform_psnr']:+.2f}}}",
            f"\\newcommand{{\\MeanSsimOurs}}{{{m['ours_ssim']:.3f}}}",
            f"\\newcommand{{\\MeanSsimUnif}}{{{m['uniform_ssim']:.3f}}}",
            f"\\newcommand{{\\MeanDeltaSsim}}{{{m['ours_ssim']-m['uniform_ssim']:+.3f}}}",
            f"\\newcommand{{\\MeanLpipsOurs}}{{{m['ours_lpips']:.3f}}}",
            f"\\newcommand{{\\MeanLpipsUnif}}{{{m['uniform_lpips']:.3f}}}",
            f"\\newcommand{{\\MeanDeltaLpips}}{{{m['ours_lpips']-m['uniform_lpips']:+.3f}}}",
            f"\\newcommand{{\\BestScene}}{{{best['scene'].replace('_', ' ')}}}",
            f"\\newcommand{{\\BestScenePsnrGain}}{{{best['ours_psnr']-best['uniform_psnr']:+.2f}}}",
        ]
        (TABLES / "macros.tex").write_text("\n".join(macros) + "\n")

    complete = [r for r in rows if r["ours_psnr"] is not None
                and r["uniform_psnr"] is not None]
    print(f"{len(rows)} scenes ({len(complete)} complete)")
    for k in ("psnr", "ssim", "lpips"):
        mo, mu = mean([r[f"ours_{k}"] for r in complete]), \
                 mean([r[f"uniform_{k}"] for r in complete])
        if mo is not None:
            print(f"  {k}: ours {mo:.3f} vs uniform {mu:.3f} (delta {mo-mu:+.3f})")


if __name__ == "__main__":
    main()
