#!/usr/bin/env python3
"""run_all.py — run the A/B benchmark for every scene under data/scenes/.

Scenes with a results/<scene>.json marked "done" are skipped, so this is safe
to re-run whenever new scenes (e.g. late Tanks and Temples downloads) arrive.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"


def main():
    scenes = sorted(p.parent.name for p in (ROOT / "data" / "scenes").glob("*/input.mp4"))
    print(f"scenes: {scenes}", flush=True)
    for scene in scenes:
        rp = RESULTS / f"{scene}.json"
        if rp.exists() and json.loads(rp.read_text()).get("status") == "done":
            print(f"[{scene}] done, skipping", flush=True)
            continue
        print(f"[{scene}] running ...", flush=True)
        subprocess.run([sys.executable, str(ROOT / "benchmark" / "run_benchmark.py"),
                        "--scene", scene])
    subprocess.run([sys.executable, str(ROOT / "benchmark" / "make_tables.py")])


if __name__ == "__main__":
    main()
