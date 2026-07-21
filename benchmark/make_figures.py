#!/usr/bin/env python3
"""make_figures.py — teaser, qualitative comparison, selection-behavior plot.

  python make_figures.py teaser
  python make_figures.py qualitative [--scene <name>]   (best-gain scene by default)
  python make_figures.py selection --scene <name>

Outputs go to paper/figures/.
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
FIGS = ROOT / "paper" / "figures"
ACCENT = "#2547D0"
REJECT = "#B9BEC6"
INK = "#14161A"


def load_stats(scene):
    return json.loads(
        (ROOT / "data" / "scenes" / scene / "extract" / "extract_stats.json").read_text())


# ---------------------------------------------------------------------------

def teaser(args):
    """Pipeline overview: video strip -> analyze -> select -> COLMAP/3DGS."""
    # use real selection data when available for the timeline inset
    stats = None
    for p in sorted((ROOT / "data" / "scenes").glob("*/extract/extract_stats.json")):
        stats = json.loads(p.read_text())
        break

    fig = plt.figure(figsize=(13.2, 2.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 2.9)
    ax.axis("off")

    def box(x, w, title, tag):
        ax.add_patch(FancyBboxPatch((x, 0.42), w, 2.06,
                                    boxstyle="round,pad=0.06,rounding_size=0.10",
                                    fc="white", ec=INK, lw=1.1))
        ax.text(x + w / 2, 2.28, title, ha="center", va="center", fontsize=11.5,
                fontweight="bold", color=INK)
        ax.text(x + w / 2, 0.22, tag, ha="center", va="center", fontsize=8.2,
                color="#6A7076", family="monospace")

    def arrow(x0, x1):
        ax.add_patch(FancyArrowPatch((x0, 1.45), (x1, 1.45), arrowstyle="-|>",
                                     mutation_scale=16, color=INK, lw=1.4))

    # --- 1. input video strip
    box(0.25, 2.6, "Input video", "30 fps, handheld")
    rng = np.random.default_rng(7)
    for i in range(6):
        x = 0.55 + i * 0.34
        blur = i in (1, 4)
        ax.add_patch(plt.Rectangle((x, 0.85), 0.28, 1.05,
                                   fc=REJECT if blur else "#8E97A6",
                                   ec="white", lw=0.8, alpha=0.55 if blur else 0.95))
        if blur:
            ax.text(x + 0.14, 1.37, "~", ha="center", va="center", fontsize=13,
                    color="white", fontweight="bold")

    arrow(2.95, 3.35)

    # --- 2. analysis
    box(3.45, 2.9, "Pass 1 · Analyze", "thresholds derived from the video")
    xs = np.linspace(3.75, 5.95, 120)
    dist = np.exp(-0.5 * ((xs - 5.0) / 0.42) ** 2) + 0.45 * np.exp(
        -0.5 * ((xs - 4.15) / 0.28) ** 2)
    ax.plot(xs, 0.9 + dist * 0.85, color=INK, lw=1.3)
    thr_x = 4.42
    ax.plot([thr_x, thr_x], [0.85, 1.95], color=ACCENT, lw=1.4, ls="--")
    ax.text(thr_x - 0.06, 1.99, r"blur floor", fontsize=8, color=ACCENT, ha="center")
    ax.text(5.6, 1.05, "sharpness\ndistribution", fontsize=7.5, color="#6A7076",
            ha="center")

    arrow(6.35, 6.75)

    # --- 3. selection timeline (real keyframe positions if available)
    box(6.85, 3.3, "Pass 2 · Select", "parallax-triggered, sharpest-in-buffer")
    tl_x0, tl_x1, tl_y = 7.1, 9.9, 1.28
    ax.plot([tl_x0, tl_x1], [tl_y, tl_y], color=REJECT, lw=2.4)
    if stats:
        n = stats["analysis"]["n_frames"]
        pos = [kf["frame_idx"] / n for kf in stats["keyframes"]]
    else:
        pos = np.linspace(0.03, 0.97, 24)
    for p in pos:
        x = tl_x0 + p * (tl_x1 - tl_x0)
        ax.plot([x, x], [tl_y - 0.13, tl_y + 0.13], color=ACCENT, lw=1.0)
    ax.text((tl_x0 + tl_x1) / 2, 1.75, "keyframes fire where parallax accrues",
            fontsize=8, color="#6A7076", ha="center")
    if stats:
        ax.text((tl_x0 + tl_x1) / 2, 0.95,
                f"{stats['n_extracted']} of {stats['analysis']['n_frames']} frames kept",
                fontsize=8.2, color=INK, ha="center", family="monospace")

    arrow(10.35, 10.75)

    # --- 4. COLMAP + 3DGS
    box(10.85, 2.1, "Pass 3 · Configure", "matcher auto-selected + QC")
    ax.text(11.9, 1.62, "COLMAP script\n(single camera,\ngap-aware matcher)",
            fontsize=8.4, ha="center", color=INK)
    ax.text(11.9, 0.92, "→ 3DGS training", fontsize=9, ha="center",
            color=ACCENT, fontweight="bold")

    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "teaser.pdf")
    fig.savefig(FIGS / "teaser.png", dpi=200)
    print(f"wrote {FIGS / 'teaser.pdf'}")


# ---------------------------------------------------------------------------

def best_gain_scene():
    best, gain = None, -1e9
    for p in (BENCH / "results").glob("*.json"):
        r = json.loads(p.read_text())
        try:
            g = (r["arms"]["ours"]["metrics"]["psnr"] -
                 r["arms"]["uniform"]["metrics"]["psnr"])
        except (KeyError, TypeError):
            continue
        if g > gain:
            best, gain = r["scene"], g
    return best


def qualitative(args):
    scene = args.scene or best_gain_scene()
    if scene is None:
        raise SystemExit("no completed results yet")
    scene_dir = ROOT / "data" / "scenes" / scene
    # map each arm's render index to its registered test-view NAME (render order
    # follows the sorted registered test_* names), then compare per test view
    import sys
    sys.path.insert(0, str(BENCH))
    from run_benchmark import read_registered_images
    m, name2rec = {}, {}
    for arm in ("ours", "uniform"):
        arm_dir = scene_dir / "work" / arm
        m[arm] = json.loads((arm_dir / "metrics.json").read_text())
        names = sorted(n for n in read_registered_images(
            arm_dir / "sparse" / "0" / "images.bin") if n.startswith("test_"))
        assert len(names) == len(m[arm]["per_image"])
        name2rec[arm] = dict(zip(names, m[arm]["per_image"]))
    # largest per-view PSNR gap among views present in both arms, skipping
    # degenerate renders (mis-registered camera -> blank output) which would
    # read as a bug rather than a representative comparison
    gaps = []
    for n in set(name2rec["ours"]) & set(name2rec["uniform"]):
        o, u = name2rec["ours"][n], name2rec["uniform"][n]
        if o["psnr"] >= 16 and u["psnr"] >= 16:
            gaps.append((o["psnr"] - u["psnr"], o["file"], u["file"], n))
    gaps.sort(reverse=True)
    gain, f_ours, f_unif, view_name = gaps[0]

    def halves(arm, fname):
        canvas = cv2.imread(str(scene_dir / "work" / arm / "gs" / "renders" / fname))
        h, w, _ = canvas.shape
        return canvas[:, : w // 2], canvas[:, w // 2:]

    gt, ours = halves("ours", f_ours)
    _, unif = halves("uniform", f_unif)

    # crop: region with the largest local difference between unif and gt
    diff = cv2.absdiff(cv2.cvtColor(unif, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)).astype(np.float32)
    diff = cv2.GaussianBlur(diff, (0, 0), 15)
    h, w = diff.shape
    ch, cw = h // 3, w // 3
    cy, cx = np.unravel_index(np.argmax(diff[ch // 2: h - ch // 2,
                                             cw // 2: w - cw // 2]),
                              diff[ch // 2: h - ch // 2, cw // 2: w - cw // 2].shape)
    cy += ch // 2
    cx += cw // 2
    y0, x0 = max(0, cy - ch // 2), max(0, cx - cw // 2)
    crops = [im[y0:y0 + ch, x0:x0 + cw] for im in (unif, ours, gt)]

    h, w = unif.shape[:2]
    ch_, cw_ = crops[0].shape[:2]
    # figure height follows actual image aspect ratios -> no dead space
    fig_w = 9.0
    row1_h = fig_w / 3 * (h / w)
    row2_h = fig_w / 3 * (ch_ / cw_)
    fig, axes = plt.subplots(2, 3, figsize=(fig_w, row1_h + row2_h + 0.35),
                             gridspec_kw=dict(height_ratios=[row1_h, row2_h],
                                              wspace=0.02, hspace=0.03))
    titles = [f"Uniform ({name2rec['uniform'][view_name]['psnr']:.2f} dB)",
              f"Ours ({name2rec['ours'][view_name]['psnr']:.2f} dB)",
              "Ground truth"]
    for j, (im, crop, t) in enumerate(zip((unif, ours, gt), crops, titles)):
        for i, img in enumerate((im, crop)):
            axes[i][j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i][j].axis("off")
        axes[0][j].set_title(t, fontsize=10)
        rect = plt.Rectangle((x0, y0), cw, ch, fill=False, ec=ACCENT, lw=1.6)
        axes[0][j].add_patch(rect)
    fig.savefig(FIGS / "qualitative.png", dpi=200, bbox_inches="tight")
    print(f"wrote {FIGS / 'qualitative.png'} (scene={scene}, view gain {gain:+.2f} dB)")


# ---------------------------------------------------------------------------

def selection(args):
    """Timeline of selected frames + sharpness (optional figure)."""
    stats = load_stats(args.scene)
    n = stats["analysis"]["n_frames"]
    fps = stats["analysis"]["fps"]
    kf_t = [kf["frame_idx"] / fps for kf in stats["keyframes"]]
    kf_s = [kf["sharpness"] for kf in stats["keyframes"]]

    fig, ax = plt.subplots(figsize=(8.4, 2.2))
    ax.vlines(kf_t, 0, 1, color=ACCENT, lw=0.7, alpha=0.85,
              transform=ax.get_xaxis_transform(), label="selected keyframes")
    ax2 = ax.twinx()
    ax2.plot(kf_t, kf_s, ".", ms=3.5, color=INK, alpha=0.7, label="sharpness")
    ax.set_xlim(0, n / fps)
    ax.set_yticks([])
    ax.set_xlabel("time [s]")
    ax2.set_ylabel("Laplacian var.")
    ax.set_title(f"{args.scene}: {len(kf_t)} keyframes from {n} frames", fontsize=10)
    fig.savefig(FIGS / f"selection_{args.scene}.png", dpi=200, bbox_inches="tight")
    print(f"wrote {FIGS / f'selection_{args.scene}.png'}")


def blur(args):
    """Dose-response: delta(Ours-Uniform) vs synthetic blur level."""
    import sys
    sys.path.insert(0, str(BENCH))
    from make_tables import common_test_metrics

    scenes = {"dl3dv_032dee9f": "outdoor statue",
              "dl3dv_2beaca31": "toy store",
              "dl3dv_49381681": "greenhouse"}
    levels = [1, 2, 4, 8]
    xlab = [0, 2, 4, 8]  # paper naming: blur-0 == window 1

    data = {}
    for s in scenes:
        d = []
        for n in levels:
            c = common_test_metrics(f"{s}_b{n}")
            d.append(None if c is None else
                     (c["ours"]["psnr"] - c["uniform"]["psnr"],
                      c["ours"]["lpips"] - c["uniform"]["lpips"]))
        data[s] = d

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9))
    for ax, idx, ylabel, better in (
            (axes[0], 0, r"$\Delta$PSNR (Ours $-$ Uniform) [dB]", "up"),
            (axes[1], 1, r"$\Delta$LPIPS (Ours $-$ Uniform)", "down")):
        means = []
        for k, x in enumerate(xlab):
            vals = [data[s][k][idx] for s in scenes if data[s][k] is not None]
            means.append(sum(vals) / len(vals) if vals else None)
        for s, label in scenes.items():
            xs = [x for x, d in zip(xlab, data[s]) if d is not None]
            ys = [d[idx] for d in data[s] if d is not None]
            ax.plot(xs, ys, "o--", ms=4, lw=1.0, alpha=0.55, label=label)
        ax.plot(xlab, means, "s-", ms=5, lw=2.2, color=ACCENT, label="mean")
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_xlabel("synthetic blur level (averaged frames)")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(xlab)
        note = "curation better ↑" if better == "up" else "curation better ↓"
        ax.text(0.02, 0.96 if better == "up" else 0.06, note,
                transform=ax.transAxes, fontsize=7.5, color="#555",
                va="top" if better == "up" else "bottom")
    axes[0].legend(fontsize=7, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGS / "blur_doseresponse.pdf")
    fig.savefig(FIGS / "blur_doseresponse.png", dpi=200)
    print(f"wrote {FIGS / 'blur_doseresponse.pdf'}")


def coverage(args):
    """Mechanism figure: per-view dPSNR vs relative local coverage sparsity."""
    import sys
    sys.path.insert(0, str(BENCH))
    from make_tables import common_test_paired, paper_scene_list

    xs, ys = [], []
    for scene in paper_scene_list() or []:
        sd = ROOT / "data" / "scenes" / scene
        mp = sd / "prepare_meta.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text())
        fps = meta["fps"]
        tidx = {r["image"]: r["frame_idx"] for r in meta["test_frames"]}
        tr = {a: np.array(sorted(meta[f"{a}_frames" if a == "ours" else "uniform_frames"]))
              for a in ("ours", "uniform")}
        paired = common_test_paired(scene)
        if not paired:
            continue
        for name, o, u in paired["views"]:
            t = tidx.get(name)
            if t is None:
                continue
            d_o = np.min(np.abs(tr["ours"] - t)) / fps
            d_u = np.min(np.abs(tr["uniform"] - t)) / fps
            xs.append(d_o - d_u)
            ys.append(o["psnr"] - u["psnr"])
    xs, ys = np.array(xs), np.array(ys)

    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    ax.scatter(xs, ys, s=8, alpha=0.35, color="#5B739B", edgecolors="none")
    # binned means
    edges = np.quantile(xs, np.linspace(0, 1, 9))
    cx, cy = [], []
    for a, b in zip(edges, edges[1:]):
        m = (xs >= a) & (xs <= b)
        if m.sum() > 4:
            cx.append(xs[m].mean())
            cy.append(ys[m].mean())
    ax.plot(cx, cy, "s-", color=ACCENT, lw=2.0, ms=4.5, label="binned mean")
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(0, color="#999", lw=0.8)
    from scipy import stats as sps
    rho = sps.spearmanr(xs, ys).statistic
    ax.text(0.02, 0.04, f"Spearman $\\rho$ = {rho:.2f}  (n = {len(xs)} views)",
            transform=ax.transAxes, fontsize=8, color=INK)
    ax.set_xlabel("nearest-train-view distance: Ours $-$ Uniform [s]", fontsize=9)
    ax.set_ylabel(r"$\Delta$PSNR (Ours $-$ Uniform) [dB]", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "coverage.pdf")
    fig.savefig(FIGS / "coverage.png", dpi=200)
    print(f"wrote {FIGS / 'coverage.pdf'} (rho={rho:.3f}, n={len(xs)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["teaser", "qualitative", "selection", "blur",
                                     "coverage"])
    ap.add_argument("--scene", default=None)
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    {"teaser": teaser, "qualitative": qualitative, "selection": selection,
     "blur": blur, "coverage": coverage}[args.what](args)
