#!/usr/bin/env bash
# Synthetic-blur dose-response experiment driver.
#   stage cpu: synthesize blur levels + prepare all scene-levels (no GPU)
#   stage gpu: run the A/B benchmark for every prepared blur scene-level
# Usage: bash run_blur_batch.sh cpu|gpu
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export VOCAB_TREE_PATH="$(pwd)/tools/vocab_tree_faiss_flickr100K_words256K.bin"

SCENES=(dl3dv_032dee9f dl3dv_2beaca31 dl3dv_49381681)
LEVELS=(1 2 4 8)          # window sizes; b1 == re-encoded sharp original
declare -A OFFSET=([1]=0 [2]=0 [4]=1 [8]=3)

stage="${1:-cpu}"

if [ "$stage" = "cpu" ]; then
    for s in "${SCENES[@]}"; do
        $PY benchmark/make_blur.py --scene "$s" --levels 1,2,4,8
        for n in "${LEVELS[@]}"; do
            sc="${s}_b${n}"
            [ -f "data/scenes/$sc/prepare_meta.json" ] && continue
            echo "[prepare] $sc"
            if [ "$n" = "1" ]; then
                $PY benchmark/prepare_scene.py --scene "$sc" 2>&1 | tail -1
            else
                $PY benchmark/prepare_scene.py --scene "$sc" \
                    --test-import "${s}_b1" --frame-offset "${OFFSET[$n]}" 2>&1 | tail -1
            fi
        done
    done
    echo "BLUR_CPU_DONE"
else
    for s in "${SCENES[@]}"; do
        for n in "${LEVELS[@]}"; do
            sc="${s}_b${n}"
            until $PY -c "import json;json.load(open('data/scenes/$sc/prepare_meta.json'))" 2>/dev/null; do
                sleep 60
            done
            echo "[bench] $sc"
            $PY benchmark/run_benchmark.py --scene "$sc"
        done
    done
    echo "BLUR_GPU_DONE"
fi
