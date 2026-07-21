# extract_frames


Automatically extracts reconstruction-ready still frames from video for 3D Gaussian Splatting pipelines, and generates the COLMAP configuration to go with them.

```
[video] -> extract_frames.py -> [images/ + colmap_run.sh] -> COLMAP -> 3DGS training
```

Unlike naive fixed-interval frame extraction, this tool performs **parallax-based keyframe selection** and **blur rejection**, preventing the three main input-side causes of poor reconstructions: motion blur, redundant frames, and insufficient baseline. All parameters are derived automatically from an analysis pass over the video, so the only required argument is the video path.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. Install COLMAP separately (the generated script invokes it).

## Usage

```bash
# Basic (this is all you need)
python extract_frames.py input.mp4

# Specify output directory
python extract_frames.py input.mp4 -o ./dataset

# Pipeline integration (stdout is the images path only; logs go to stderr)
IMAGES_DIR=$(python extract_frames.py input.mp4)
bash "$(dirname "$IMAGES_DIR")/colmap_run.sh"
```

### Enabling loop detection (recommended)

Download a vocab tree (`vocab_tree_flickr100K_words32K.bin`) once from the [COLMAP release page](https://demuc.de/colmap/) and set:

```bash
export VOCAB_TREE_PATH=/path/to/vocab_tree_flickr100K_words32K.bin
```

When the extracted sequence contains parallax gaps (fragmentation risk from fast pans etc.), the generated matcher config will automatically enable loop detection.

## Outputs

```
<out_dir>/
├── images/               numbered jpgs (COLMAP input)
├── colmap_run.sh         auto-selected COLMAP script (with reasoning as comments)
└── extract_stats.json    extraction stats, gap positions, selection reasoning
```

The last line of stdout is the absolute path of `images/` (for piping into downstream systems).

## How it works

**Pass 1 — video analysis**: samples ~300 points and derives the following from the sharpness distribution and inter-frame flow statistics.

| Parameter | Derivation |
|---|---|
| `flow_thresh` | Base 0.03 (image-diagonal ratio). The estimated keyframe count is computed from cumulative motion; the threshold is only adjusted (within 0.012–0.08) when the count would fall outside 30–400 |
| `sharp_floor` | The smaller of the 30th-percentile sharpness and 0.6 × median. Aggressive filtering for blurry videos, conservative for stable ones |
| `buffer_size` | Half the estimated keyframe interval (4–15) |
| `max_dim` | Caps output resolution at ~2 MP (the sweet spot for feature matching) |

**Pass 2 — extraction**: a keyframe is triggered when the median feature-track flow from the previous keyframe exceeds `flow_thresh`; the sharpest frame in the recent buffer is saved. Frames below `sharp_floor` never enter the buffer.

**Pass 3 — COLMAP config generation**: QCs the extracted frames and selects the matcher.

| Condition | Selection |
|---|---|
| ≤ 120 images | exhaustive matcher (most robust; finishes in minutes at this scale) |
| > 120 images, no gaps | sequential matcher, overlap=10 (fastest) |
| > 120 images, gaps present | sequential, overlap=20; loop detection enabled if `VOCAB_TREE_PATH` is set |

A gap is an inter-keyframe parallax > 0.12 or a feature-tracking failure. Positions are recorded in `extract_stats.json`.

Generated COLMAP commands always use `--ImageReader.single_camera 1` (a video is a single camera) and the `SIMPLE_RADIAL` model.

## Options (manual overrides)

Normally unnecessary. For debugging cases where the auto-derived values are off.

| Option | Description |
|---|---|
| `--flow-thresh` | Keyframe parallax threshold (diagonal ratio). Raise if too many frames, lower if too few |
| `--sharp-floor` | Sharpness cutoff (absolute Laplacian variance) |
| `--buffer-size` | Best-of buffer length |
| `--max-dim` | Long edge of saved images |
| `--analysis-dim` | Long edge of analysis images (default 640; larger = more accurate but slower) |
| `--jpeg-quality` | Saved jpg quality (default 95) |

## Troubleshooting

- **Too few frames extracted / `almost no camera motion` warning**:
  the camera didn't move enough. When recapturing, walk while filming, avoid rotation-only sections, and take side steps to maintain baseline.
- **COLMAP mapper registers few images**:
  check `gap_positions` in `extract_stats.json`. If there are many gaps, set `VOCAB_TREE_PATH` and re-run, or swap the matcher in `colmap_run.sh` for `vocab_tree_matcher` / `exhaustive_matcher`.
- **Floaters / color flicker in the reconstruction**:
  usually caused by auto-exposure/white-balance drift at capture time, not extraction. Lock AE/AWB when filming.
- **Estimated count (`est ~N keyframes`) far from actual count**:
  can happen with highly non-uniform motion. Set `--flow-thresh` manually.

## License / dependencies

Dependencies are OpenCV and NumPy only. COLMAP is invoked as an external BSD-licensed tool.
