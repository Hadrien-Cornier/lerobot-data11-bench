#!/usr/bin/env bash
# DATA-11 round 4 job (HF Jobs, image python:3.12-slim-trixie, cpu-upgrade), one job per dataset (R4_DATASET).
# Same restart guard and cgroup memory logging as job/entrypoint_r3.sh. Bundle: bundle-r4/ of $RESULTS_REPO.
# Results: results/$RUN_NAME/ (RUN_NAME r4-<ds>-<UTC>[-smoke]).
#   slice    r4_slice.py    real file stats, slice of whole episodes in the first video file of every camera, downloads
#   encode   r4_encode.py   decode once, SEP + STACK with LeRobot's default encoder (PyAV-only process)
#   build    r4_build.py    -sep / -stack / -sep1 LeRobot v3 datasets (8 files per key), upload (private)
#   check    r4_check.py    PSNR, camera order, hf:// == local, read back through LeRobotDataset
#   stream   r4_stream.py   StreamingLeRobotDataset over hf:// (PR #4702 head), + DataLoader shard audit
#   map      r4_map.py      map-style LeRobotDataset on the local copies, lerobot-train DataLoader settings
#   summary  r4_summary.py
# Never enable `set -x` here: HF_TOKEN is in the environment.

set -Eeuo pipefail
umask 022
SMOKE="${SMOKE:-0}"
NO_UPLOAD="${NO_UPLOAD:-0}"
: "${R4_DATASET:?set R4_DATASET (droid | molmo | libero | hqf)}"
export SMOKE NO_UPLOAD R4_DATASET
TS="$(date -u +%Y%m%dT%H%M%SZ)"
if [ "$SMOKE" = 1 ]; then RUN_NAME="${RUN_NAME:-r4-$R4_DATASET-$TS-smoke}"; else RUN_NAME="${RUN_NAME:-r4-$R4_DATASET-$TS}"; fi
export RUN_NAME
RESULTS_ROOT="${RESULTS_ROOT:-/results}"
OUT="$RESULTS_ROOT/$RUN_NAME"
WORK="${WORK:-/work}"
mkdir -p "$OUT/logs" "$OUT/env" "$WORK"
exec > >(tee -a "$OUT/logs/entrypoint.log") 2>&1
echo "== DATA-11 r4 job $RUN_NAME start $(date -u +%FT%TZ) SMOKE=$SMOKE dataset=$R4_DATASET host=$(hostname)"

if [ "$NO_UPLOAD" != 1 ]; then
  : "${RESULTS_REPO:?set RESULTS_REPO (or NO_UPLOAD=1)}"
  if [ -z "${HF_TOKEN:-}" ]; then echo "HF_TOKEN is not set (pass --secrets HF_TOKEN)"; exit 2; fi
  echo "HF_TOKEN present (value not shown); results -> dataset $RESULTS_REPO:results/$RUN_NAME"
  python3 -m pip install -q --root-user-action=ignore "huggingface_hub>=1.0,<2"
  JOB_KEY="${JOB_ID:-$(hostname | sed -nE 's/^j-[^-]+-([0-9a-f]{24})-.*/\1/p')}"
  if [ -n "$JOB_KEY" ]; then
    if ! JOB_KEY="$JOB_KEY" RUN_NAME="$RUN_NAME" python3 - <<'PY'
import os, sys
from huggingface_hub import HfApi
api, repo, key = HfApi(), os.environ["RESULTS_REPO"], os.environ["JOB_KEY"]
path = f"attempts/{key}"
if api.file_exists(repo, path, repo_type="dataset"):
    print(f"RESTART DETECTED: {path} already exists in {repo}; refusing to run this job a second time")
    sys.exit(1)
api.upload_file(path_or_fileobj=os.environ["RUN_NAME"].encode(), path_in_repo=path, repo_id=repo,
                repo_type="dataset", commit_message=f"job attempt marker {key}")
print(f"attempt marker {path} recorded")
PY
    then
      exit 0
    fi
  else
    echo "WARNING: no job id found (JOB_ID unset, hostname $(hostname)); restart guard disabled"
  fi
fi

(
  printf 'utc\tcgroup_current_mb\tcgroup_peak_mb\tanon_mb\tfile_mb\ttop_rss_mb_cmd\n'
  while sleep 10; do
    cur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
    peak=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo 0)
    anon=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null || echo 0)
    file=$(awk '$1=="file"{print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null || echo 0)
    top=$(ps -eo rss=,args= --sort=-rss 2>/dev/null | head -3 | awk '{printf "%d:%s ", $1/1024, $3}')
    printf '%s\t%d\t%d\t%d\t%d\t%s\n' "$(date -u +%T)" $((cur / 1048576)) $((peak / 1048576)) $((${anon:-0} / 1048576)) $((${file:-0} / 1048576)) "$top"
  done
) > "$OUT/logs/mem.tsv" 2>/dev/null &

STATUS="$OUT/STATUS.tsv"
printf 'stage\tstatus\tseconds\tfinished_utc\tcgroup_peak_mb\n' > "$STATUS"
peak_mb() { echo $(( $(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo 0) / 1048576 )); }
record() { printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(date -u +%FT%TZ)" "$(peak_mb)" >> "$STATUS"; echo "== stage $1: $2 (${3}s, cgroup peak $(peak_mb) MB)"; }

HF="$(command -v hf || true)"
upload() {
  if [ "$NO_UPLOAD" = 1 ]; then return 0; fi
  if [ -z "$HF" ]; then echo "WARNING: no hf CLI yet, cannot upload ($1)"; return 0; fi
  if flock -w 900 "$WORK/.upload.lock" env HF_HUB_OFFLINE=0 "$HF" upload "$RESULTS_REPO" "$OUT" "results/$RUN_NAME" \
       --repo-type dataset --private --exclude "*.mp4" --exclude "*.tmp" --exclude "*.npz" --commit-message "DATA-11 r4 job $RUN_NAME: $1" \
       > "$OUT/logs/upload_last.log" 2>&1; then
    echo "uploaded results/$RUN_NAME ($1)"
  else
    echo "WARNING: upload failed ($1); see logs/upload_last.log"
  fi
}
on_exit() {
  local rc=$?
  echo "== exit code $rc at $(date -u +%FT%TZ)"
  printf 'exit\t%s\t-\t%s\t%s\n' "$rc" "$(date -u +%FT%TZ)" "$(peak_mb)" >> "$STATUS"
  cat /sys/fs/cgroup/memory.events > "$OUT/logs/memory_events_final.txt" 2>/dev/null || true
  upload "final (exit $rc)"
}
trap on_exit EXIT
trap 'exit 143' TERM
: "${PERIODIC_UPLOAD_S:=$([ "$SMOKE" = 1 ] && echo 180 || echo 600)}"
( while sleep "$PERIODIC_UPLOAD_S"; do upload "periodic" > /dev/null; done ) &

stage() {
  local name="$1"; shift
  local t0=$SECONDS
  echo "== stage $name start $(date -u +%FT%TZ)"
  if "$@" 2>&1 | tee "$OUT/logs/$name.log"; then record "$name" ok $((SECONDS - t0)); else record "$name" "FAILED rc=${PIPESTATUS[0]}" $((SECONDS - t0)); fi
  cat /sys/fs/cgroup/memory.events > "$OUT/logs/memory_events_after_$name.txt" 2>/dev/null || true
  upload "after $name"
}
# a stage whose output later stages need: stop the job if it failed
need() { [ -f "$1" ] || { echo "missing $1: stopping"; exit 7; }; }

# ------------------------------------------------------------------ system deps, bundle
t0=$SECONDS
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends ffmpeg procps util-linux ca-certificates coreutils curl >/dev/null
echo "system ffmpeg: $(ffmpeg -version 2>/dev/null | head -1)"
record system_deps ok $((SECONDS - t0))
t0=$SECONDS
if [ -n "${BUNDLE_DIR:-}" ]; then BUNDLE="$BUNDLE_DIR"; else
  command -v hf >/dev/null || python3 -m pip install -q --root-user-action=ignore "huggingface_hub>=1.0,<2"
  HF="$(command -v hf)"
  "$HF" download "$RESULTS_REPO" --repo-type dataset --include "bundle-r4/*" --local-dir "$WORK/dl_bundle" > "$OUT/logs/bundle_download.log" 2>&1
  BUNDLE="$WORK/dl_bundle/bundle-r4"
fi
BUNDLE="$(cd "$BUNDLE" && pwd)"
(cd "$BUNDLE" && sha256sum -c --quiet SHA256SUMS) || { echo "bundle checksum mismatch"; exit 4; }
echo "bundle checksums OK"
mkdir -p "$OUT/bundle" && cp -R "$BUNDLE"/. "$OUT/bundle/"
record bundle ok $((SECONDS - t0))

# ------------------------------------------------------------------ venv: lerobot at the PR #4702 head
t0=$SECONDS
: "${LEROBOT_PR:=4702}"
: "${LEROBOT_SHA_PIN:=32b671d437f8c50014f74a322ad3df09c6fab944}"  # refs/pull/4702/head on 2026-09-29
LEROBOT_SHA_NOW="$(curl -fsS "https://api.github.com/repos/huggingface/lerobot/pulls/$LEROBOT_PR" | python3 -c 'import json,sys; print(json.load(sys.stdin)["head"]["sha"])' 2>/dev/null || echo unknown)"
: "${LEROBOT_SHA:=$LEROBOT_SHA_NOW}"
[ "$LEROBOT_SHA" = unknown ] && LEROBOT_SHA="$LEROBOT_SHA_PIN"
export LEROBOT_SHA
echo "lerobot PR #$LEROBOT_PR head now $LEROBOT_SHA_NOW (pin $LEROBOT_SHA_PIN); installing $LEROBOT_SHA"
python3 -m pip install -q --root-user-action=ignore uv
export UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300
VENV="$WORK/venv"; PY="$VENV/bin/python"
uv venv --quiet --python "$(command -v python3)" "$VENV"
{
  printf 'torch==2.11.0\ntorchcodec==0.11.1\ntorchvision==0.26.0\nhuggingface_hub==1.30.0\n' > "$WORK/constraints.txt"
  uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0" "torchcodec==0.11.1" "torchvision==0.26.0" \
  && uv pip install --python "$PY" -c "$WORK/constraints.txt" psutil numpy pandas pyarrow "huggingface_hub==1.30.0" \
       "lerobot[dataset] @ https://github.com/huggingface/lerobot/archive/$LEROBOT_SHA.tar.gz"
} > "$OUT/logs/pip_install.log" 2>&1 || { tail -40 "$OUT/logs/pip_install.log"; exit 6; }
uv pip freeze --python "$PY" > "$OUT/env/pip_freeze.txt"
"$PY" -c "import lerobot, torch, torchcodec, datasets, huggingface_hub; from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset; print('lerobot', lerobot.__file__, 'torch', torch.__version__, 'torchcodec', torchcodec.__version__, 'datasets', datasets.__version__, 'hub', huggingface_hub.__version__)"
"$PY" -c "import av; print('av', av.__version__, {k: '.'.join(map(str, v)) for k, v in av.library_versions.items()}); av.codec.Codec('libsvtav1', 'w'); print('libsvtav1 encoder available in PyAV')"
HF="$VENV/bin/hf"
export PATH="$VENV/bin:$PATH"
record python_env ok $((SECONDS - t0))
{
  nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null; cat /sys/fs/cgroup/memory.max 2>/dev/null
  lscpu 2>/dev/null | head -20; free -m; df -h "$WORK"; echo "lerobot $LEROBOT_SHA (PR head now $LEROBOT_SHA_NOW, pin $LEROBOT_SHA_PIN)"
} > "$OUT/env/hardware.txt" 2>&1
env | grep -E '^(SMOKE|STAGES|R4_|RUN_NAME|LEROBOT_)' | sort > "$OUT/env/job_config.txt" || true
export CPU_QUOTA="$("$PY" -c 'import sys; sys.path.insert(0, "'"$BUNDLE"'"); import common; print(common.cpu_quota())')"
echo "cpu quota $CPU_QUOTA"
upload "setup done"

# ------------------------------------------------------------------ stages
: "${STAGES:=slice encode build check stream map summary}"
want() { [[ " $STAGES " == *" $1 "* ]]; }
cd "$BUNDLE"
export HF_LEROBOT_HOME="$WORK/lerobot_home" HF_HUB_DISABLE_PROGRESS_BARS=1
if want slice;  then stage slice  "$PY" r4_slice.py "$WORK" "$OUT"; need "$OUT/slice/slice.json"; fi
if want encode; then stage encode "$PY" r4_encode.py "$OUT/slice/slice.json" "$WORK" "$OUT"; need "$OUT/encode/encode.json"; fi
if want build;  then stage build  "$PY" r4_build.py "$OUT/slice/slice.json" "$OUT/encode/encode.json" "$WORK" "$OUT"; need "$OUT/build/build.json"; fi
want check   && stage check   "$PY" r4_check.py "$OUT/slice/slice.json" "$OUT/build/build.json" "$WORK" "$OUT"
want stream  && stage stream  "$PY" r4_stream.py "$OUT/slice/slice.json" "$OUT/build/build.json" "$OUT"
want map     && stage map     "$PY" r4_map.py "$OUT/slice/slice.json" "$OUT/build/build.json" "$OUT"
want summary && stage summary "$PY" r4_summary.py "$OUT"
du -sh "$WORK"/* > "$OUT/env/disk_usage.txt" 2>/dev/null || true
echo "== done $(date -u +%FT%TZ): $OUT"
