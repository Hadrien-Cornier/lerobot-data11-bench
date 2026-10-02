#!/usr/bin/env bash
# DATA-11 round 3 job (HF Jobs, image python:3.12-slim-trixie, cpu-upgrade). Same restart guard and cgroup memory
# logging as job/entrypoint.sh. Bundle: bundle-r3/ of $RESULTS_REPO. Results: results/$RUN_NAME/ (RUN_NAME r3-<UTC>).
#   prep     r3_prep.py      equal-size STACK / MULTI cuts (stream copy, verified), upload, #3917 datasets
#   size     r3_size.py      file size vs range-request latency inside HF (repeated at the end as size2)
#   access   r3_access.py    random access arms on equal-size files (+ control on round-2 files), 1 and N readers
#   pr3917   r3_pr3917.py    PR #3917 head on SEP vs STACK datasets, 1 and 8 processes, + multi-track check
#   summary  r3_summary.py
# Never enable `set -x` here: HF_TOKEN is in the environment.

set -Eeuo pipefail
umask 022
SMOKE="${SMOKE:-0}"
NO_UPLOAD="${NO_UPLOAD:-0}"
export SMOKE
TS="$(date -u +%Y%m%dT%H%M%SZ)"
if [ "$SMOKE" = 1 ]; then RUN_NAME="${RUN_NAME:-r3-$TS-smoke}"; else RUN_NAME="${RUN_NAME:-r3-$TS}"; fi
export RUN_NAME
RESULTS_ROOT="${RESULTS_ROOT:-/results}"
OUT="$RESULTS_ROOT/$RUN_NAME"
WORK="${WORK:-/work}"
mkdir -p "$OUT/logs" "$OUT/env" "$WORK"
exec > >(tee -a "$OUT/logs/entrypoint.log") 2>&1
echo "== DATA-11 r3 job $RUN_NAME start $(date -u +%FT%TZ) SMOKE=$SMOKE host=$(hostname)"

if [ "$NO_UPLOAD" != 1 ]; then
  : "${RESULTS_REPO:?set RESULTS_REPO (or NO_UPLOAD=1)}"
  if [ -z "${HF_TOKEN:-}" ]; then echo "HF_TOKEN is not set (pass --secrets HF_TOKEN)"; exit 2; fi
  echo "HF_TOKEN present (value not shown); results -> dataset $RESULTS_REPO:results/$RUN_NAME"
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
       --repo-type dataset --private --exclude "*.mp4" --exclude "*.tmp" --commit-message "DATA-11 r3 job $RUN_NAME: $1" \
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

# ------------------------------------------------------------------ system deps, bundle
t0=$SECONDS
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends ffmpeg procps util-linux ca-certificates coreutils >/dev/null
echo "system ffmpeg: $(ffmpeg -version 2>/dev/null | head -1)"
record system_deps ok $((SECONDS - t0))
t0=$SECONDS
if [ -n "${BUNDLE_DIR:-}" ]; then BUNDLE="$BUNDLE_DIR"; else
  command -v hf >/dev/null || python3 -m pip install -q --root-user-action=ignore "huggingface_hub>=1.0"
  HF="$(command -v hf)"
  "$HF" download "$RESULTS_REPO" --repo-type dataset --include "bundle-r3/*" --local-dir "$WORK/dl_bundle" > "$OUT/logs/bundle_download.log" 2>&1
  BUNDLE="$WORK/dl_bundle/bundle-r3"
fi
BUNDLE="$(cd "$BUNDLE" && pwd)"
(cd "$BUNDLE" && sha256sum -c --quiet SHA256SUMS) || { echo "bundle checksum mismatch"; exit 4; }
echo "bundle checksums OK"
mkdir -p "$OUT/bundle" && cp -R "$BUNDLE"/. "$OUT/bundle/"
record bundle ok $((SECONDS - t0))

# ------------------------------------------------------------------ venvs
t0=$SECONDS
python3 -m pip install -q --root-user-action=ignore uv
export UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300
VENV="$WORK/venv"; PY="$VENV/bin/python"
uv venv --quiet --python "$(command -v python3)" "$VENV"
{
  uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0" "torchcodec==0.11.1" \
  && uv pip install --python "$PY" "av>=15,<16" numpy pandas pyarrow psutil "datasets>=4.8,<5" "huggingface_hub==1.30.0"
} > "$OUT/logs/pip_install.log" 2>&1 || { tail -40 "$OUT/logs/pip_install.log"; exit 6; }
: "${PR3917_SHA:=9bea4a19181f3432caab55bc0b3d00442656ca68}"
export PR3917_SHA
VENV_PR3917="$WORK/venv_pr3917"
uv venv --quiet --python "$(command -v python3)" "$VENV_PR3917"
{
  printf 'torch==2.11.0\ntorchcodec==0.11.1\nhuggingface_hub==1.30.0\n' > "$WORK/constraints.txt"
  uv pip install --python "$VENV_PR3917/bin/python" --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0" "torchcodec==0.11.1" "torchvision==0.26.0" \
  && uv pip install --python "$VENV_PR3917/bin/python" -c "$WORK/constraints.txt" psutil \
       "lerobot[dataset] @ https://github.com/huggingface/lerobot/archive/$PR3917_SHA.tar.gz"
} > "$OUT/logs/pip_install_pr3917.log" 2>&1 || { tail -40 "$OUT/logs/pip_install_pr3917.log"; exit 6; }
uv pip freeze --python "$PY" > "$OUT/env/pip_freeze.txt"
uv pip freeze --python "$VENV_PR3917/bin/python" > "$OUT/env/pip_freeze_pr3917.txt"
"$VENV_PR3917/bin/python" -c "import lerobot, torch, torchcodec; from lerobot.streaming import episode_cache; print('pr3917 venv', lerobot.__file__, torch.__version__, torchcodec.__version__)"
"$PY" -c "import torch, torchcodec, av, datasets; print('main venv torch', torch.__version__, 'torchcodec', torchcodec.__version__, 'av', av.__version__)"
HF="$VENV/bin/hf"
export PATH="$VENV/bin:$PATH" VENV_PR3917
record python_env ok $((SECONDS - t0))
{
  nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null; cat /sys/fs/cgroup/memory.max 2>/dev/null
  lscpu 2>/dev/null | head -20; free -m; df -h "$WORK"; echo "pr3917 $PR3917_SHA"
} > "$OUT/env/hardware.txt" 2>&1
env | grep -E '^(SMOKE|STAGES|ACC_|SZ_|P_|R3_|RUN_NAME|PR3917_SHA)' | sort > "$OUT/env/job_config.txt" || true
export CPU_QUOTA="$("$PY" -c 'import sys; sys.path.insert(0, "'"$BUNDLE"'"); import common; print(common.cpu_quota())')"
echo "cpu quota $CPU_QUOTA"
upload "setup done"

# ------------------------------------------------------------------ stages
: "${STAGES:=prep size access pr3917 size2 summary}"
want() { [[ " $STAGES " == *" $1 "* ]]; }
cd "$BUNDLE"
export HF_LEROBOT_HOME="$WORK/lerobot_home"
want prep    && stage prep    "$PY" r3_prep.py selection.json "$WORK" "$OUT"
want size    && stage size    "$PY" r3_size.py "$OUT/prep/prep.json" "$OUT" size
want access  && stage access  "$PY" r3_access.py "$OUT/prep" "$OUT"
want pr3917  && stage pr3917  "$PY" r3_pr3917.py orchestrate "$OUT/prep/prep.json" "$OUT"
want size2   && stage size2   "$PY" r3_size.py "$OUT/prep/prep.json" "$OUT" size2
want summary && stage summary "$PY" r3_summary.py "$OUT"
du -sh "$WORK"/* > "$OUT/env/disk_usage.txt" 2>/dev/null || true
echo "== done $(date -u +%FT%TZ): $OUT"
