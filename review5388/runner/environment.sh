#!/usr/bin/env bash
# Build one environment for both readers. Select the source tree with PYTHONPATH.
set -Eeuo pipefail

WORK=${WORK:-/work/review5388}
mkdir -p "$WORK/src"
export DEBIAN_FRONTEND=noninteractive
export UV_LINK_MODE=copy UV_HTTP_TIMEOUT=120
BASE_SHA=fe1c54594f6ba275fccf210b9f7b11f0e2324b89
CANDIDATE_SHA=1a60cb4b800c2947834342752c32f064c8bf143c
DATASETS_SHA=464fe790b82a320350c9c686c1b7e409323a1cf3

apt-get -qq update
apt-get -qq install -y --no-install-recommends ca-certificates curl ffmpeg procps util-linux >/dev/null
python3 -m pip -q install 'uv==0.11.23' 'huggingface-hub==2.0.0'

source_archive() {
    local repo=$1 sha=$2 target=$3
    mkdir -p "$WORK/src/$target"
    curl -fsSL --retry 2 "https://codeload.github.com/huggingface/$repo/tar.gz/$sha" \
        -o "$WORK/src/$target.tar.gz"
    sha256sum "$WORK/src/$target.tar.gz"
    tar -xzf "$WORK/src/$target.tar.gz" -C "$WORK/src/$target" --strip-components=1
}
source_archive lerobot "$BASE_SHA" baseline
source_archive lerobot "$CANDIDATE_SHA" candidate
source_archive datasets "$DATASETS_SHA" datasets

uv venv --quiet --python "$(command -v python3)" "$WORK/venv"
PY="$WORK/venv/bin/python"
uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cpu \
    'torch==2.11.0' 'torchvision==0.26.0'
cat > "$WORK/constraints.txt" <<'CONSTRAINTS'
torch==2.11.0
torchvision==0.26.0
torchcodec==0.11.1
huggingface-hub==2.0.0
CONSTRAINTS
uv pip install --python "$PY" --prerelease allow --constraint "$WORK/constraints.txt" \
    "lerobot[dataset] @ file://$WORK/src/candidate" \
    "datasets @ file://$WORK/src/datasets" psutil pytest
uv pip freeze --python "$PY" > "$WORK/requirements-resolved.txt"
if ! uv pip install --python "$PY" --target "$WORK/datasets-release" --no-deps 'datasets==5.2.0'; then
    rm -rf "$WORK/datasets-release"
    echo "Released Datasets 5.2.0 control is unavailable"
fi

for reader in baseline candidate; do
    PYTHONPATH="$WORK/src/$reader/src" "$PY" -c '
import av, datasets, torch, torchcodec
import lerobot.datasets.streaming_dataset as streaming
print("imports", streaming.__file__, datasets.__version__, torch.__version__, torchcodec.__version__, av.__version__)
'
done
printf 'baseline=%s\ncandidate=%s\ndatasets=%s\n' "$BASE_SHA" "$CANDIDATE_SHA" "$DATASETS_SHA"
cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max
df -h "$WORK"
