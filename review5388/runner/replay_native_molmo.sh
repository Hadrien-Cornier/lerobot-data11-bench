#!/usr/bin/env bash
# Run after environment.sh on one Linux host. All inputs are public.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=${WORK:-/work/review5388}
PY="$WORK/venv/bin/python"
RUN_DIR=$(mktemp -d "$WORK/native-molmo.XXXXXX")
MANIFEST=aebacc429d54f284e103ff10613b925f5827c347e0678846b0b6f0015dded782

cd "$ROOT"
export HF_HUB_DISABLE_PROGRESS_BARS=1
for arm in grouped stock reuse; do
    revision=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["revision"])' \
        runner/native_molmo_pins.json "$arm")
    source="$RUN_DIR/$arm-source"
    cache="$RUN_DIR/$arm-cache"
    mkdir -p "$source" "$cache/tmp"
    curl -fsSL --retry 2 \
        "https://codeload.github.com/Hadrien-Cornier/lerobot/tar.gz/$revision" \
        -o "$RUN_DIR/$arm.tar.gz"
    tar -xzf "$RUN_DIR/$arm.tar.gz" -C "$source" --strip-components=1

    "$PY" catalog.py --dataset bucket --expect "$MANIFEST" \
        --out "$RUN_DIR/$arm-manifest-before.json"
    status=0
    HF_HOME="$cache/hub-home" HF_HUB_CACHE="$cache/hub-home/hub" HF_XET_CACHE="$cache/hub-home/xet" \
    HF_LEROBOT_HOME="$cache/lerobot" HF_DATASETS_CACHE="$cache/datasets" TMPDIR="$cache/tmp" \
    PYTHONPATH="$source/src:$ROOT" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ARROW_NUM_THREADS=1 \
        "$PY" -m runner.native_molmo_arm "$arm" "$RUN_DIR/$arm.json" || status=$?
    "$PY" catalog.py --dataset bucket --expect "$MANIFEST" \
        --out "$RUN_DIR/$arm-manifest-after.json"
    if (( status != 0 )); then
        exit "$status"
    fi
done
printf 'Result directory: %s\n' "$RUN_DIR"
