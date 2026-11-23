set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive UV_LINK_MODE=copy UV_HTTP_TIMEOUT=90
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ARROW_NUM_THREADS=1 PYTHONUNBUFFERED=1
work=/work/episode-controls
mkdir -p "$work"
cd "$work"
printf '%s' "$1" | base64 -d | tar xzf -
apt-get -qq update
apt-get -qq install -y --no-install-recommends curl ffmpeg ca-certificates > /dev/null
python3 -m pip -q install 'uv==0.11.23'
fetch() {
    mkdir -p "$1"
    curl -fsSL --connect-timeout 10 --max-time 90 --retry 2         "https://codeload.github.com/$2/tar.gz/$3"         | tar xz --strip-components=1 -C "$1"
}
fetch grouped Hadrien-Cornier/lerobot adbdf5f0218898b187b4d20376a4bdee4a97d388
fetch controls Hadrien-Cornier/lerobot 8b6f1bdb33bdc556d51feb90beb2e9f563cf19c3
fetch hf-datasets huggingface/datasets 464fe790b82a320350c9c686c1b7e409323a1cf3
uv venv --quiet --python "$(command -v python3)" "$work/venv"
py="$work/venv/bin/python"
uv pip install --python "$py" --index-url https://download.pytorch.org/whl/cpu     'torch==2.11.0' 'torchvision==0.26.0'
cat > constraints.txt <<'CONSTRAINTS'
torch==2.11.0
torchvision==0.26.0
torchcodec==0.11.1
huggingface-hub==2.0.0
CONSTRAINTS
uv pip install --python "$py" --prerelease allow --constraint constraints.txt     "lerobot[dataset] @ file://$work/controls" "datasets @ file://$work/hf-datasets"
printf 'cpu quota: '
cat /sys/fs/cgroup/cpu.max
printf 'cpu model: '
awk -F: '/^model name/ {print $2; exit}' /proc/cpuinfo
ffmpeg -version | sed -n '1p'
manifest() {
    "$py" catalog.py --dataset bucket --out "$1"
}
arm() {
    local arm=$1 mode=$2 duration=$3
    local cache="$work/cache-$arm-$mode"
    mkdir -p "$cache/tmp"
    HF_HOME="$cache/hub" HF_HUB_CACHE="$cache/hub/hub" HF_XET_CACHE="$cache/hub/xet"         HF_LEROBOT_HOME="$cache/lerobot" HF_DATASETS_CACHE="$cache/datasets" TMPDIR="$cache/tmp"         PYTHONPATH="$work/$arm/src" timeout --kill-after=10s "$duration"         "$py" benchmark.py "$arm" "$mode"
}
manifest before.json
grouped_status=0
controls_status=0
parity_status=0
arm grouped measure 480s || grouped_status=$?
arm controls measure 480s || controls_status=$?
if [[ -f controls-probes.json ]]; then
    arm grouped parity 130s || parity_status=$?
else
    parity_status=99
fi
manifest after.json
"$py" - "$grouped_status" "$controls_status" "$parity_status" <<'PY'
import json
import sys

before = json.load(open("before.json"))["manifest_sha256"]
after = json.load(open("after.json"))["manifest_sha256"]
receipt = {"event":"summary", "before":before, "after":after, "grouped_status":int(sys.argv[1]),
           "controls_status":int(sys.argv[2]), "parity_status":int(sys.argv[3])}
print(json.dumps(receipt), flush=True)
raise SystemExit(before != after or any(receipt[key] for key in ("grouped_status", "controls_status", "parity_status")))
PY
