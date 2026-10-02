#!/usr/bin/env bash
# DATA-11 PR #3917 A/B round 3: sidecar header probe, BASE vs PATCH (probe-exact-reads.diff) vs option A (parallel tail read).
# Arms and method: ab3_3917.py docstring.
# Same restart guard and cgroup memory logging as job/entrypoint_ab.sh. Bundle: bundle-ab3/ of $RESULTS_REPO.
# Results: results/$RUN_NAME/ (ab3-<UTC>-j<AB3_JOB>). Fan out several jobs with AB3_JOB=0..n-1.
#   ab3      ab3_3917.py orchestrate   equality + interleaved speed runs on the 10 v3.0 datasets of ab3-files.json
# Never enable `set -x` here: HF_TOKEN is in the environment.

set -Eeuo pipefail
umask 022
SMOKE="${SMOKE:-0}"
NO_UPLOAD="${NO_UPLOAD:-0}"
export SMOKE
TS="$(date -u +%Y%m%dT%H%M%SZ)"
if [ "$SMOKE" = 1 ]; then RUN_NAME="${RUN_NAME:-ab3-$TS-j${AB3_JOB:-0}-smoke}"; else RUN_NAME="${RUN_NAME:-ab3-$TS-j${AB3_JOB:-0}}"; fi
export RUN_NAME
RESULTS_ROOT="${RESULTS_ROOT:-/results}"
OUT="$RESULTS_ROOT/$RUN_NAME"
WORK="${WORK:-/work}"
mkdir -p "$OUT/logs" "$OUT/env" "$WORK"
exec > >(tee -a "$OUT/logs/entrypoint.log") 2>&1
echo "== DATA-11 ab3 job $RUN_NAME start $(date -u +%FT%TZ) SMOKE=$SMOKE host=$(hostname)"

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
       --repo-type dataset --private --exclude "*.mp4" --exclude "*.tmp" --commit-message "DATA-11 ab3 job $RUN_NAME: $1" \
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
apt-get install -y -qq --no-install-recommends ffmpeg procps util-linux ca-certificates coreutils patch curl >/dev/null
echo "system ffmpeg: $(ffmpeg -version 2>/dev/null | head -1)"
record system_deps ok $((SECONDS - t0))
t0=$SECONDS
if [ -n "${BUNDLE_DIR:-}" ]; then BUNDLE="$BUNDLE_DIR"; else
  command -v hf >/dev/null || python3 -m pip install -q --root-user-action=ignore "huggingface_hub>=1.0"
  HF="$(command -v hf)"
  "$HF" download "$RESULTS_REPO" --repo-type dataset --include "bundle-ab3/*" --local-dir "$WORK/dl_bundle" > "$OUT/logs/bundle_download.log" 2>&1
  BUNDLE="$WORK/dl_bundle/bundle-ab3"
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
uv pip install --python "$PY" psutil "huggingface_hub==1.30.0" > "$OUT/logs/pip_install.log" 2>&1 || { tail -40 "$OUT/logs/pip_install.log"; exit 6; }
: "${PR3917_SHA:=6d94598523738d19700e345e9b42fb3d2db352f5}"
export PR3917_SHA
printf 'torch==2.11.0\ntorchcodec==0.11.1\nhuggingface_hub==1.30.0\n' > "$WORK/constraints.txt"
# BASE installs the PR head archive; PROBE and TFX install the same archive with one patch each.
curl -fsSL "https://github.com/huggingface/lerobot/archive/$PR3917_SHA.tar.gz" -o "$WORK/pr3917.tar.gz"
declare -A PATCH=([probe]=probe-exact-reads.diff)
for v in base probe; do
  mkdir -p "$WORK/src_$v"
  tar -xzf "$WORK/pr3917.tar.gz" -C "$WORK/src_$v" --strip-components=1
  if [ "$v" != base ]; then
    (cd "$WORK/src_$v" && patch -p1 --forward --batch < "$BUNDLE/patches-3917/${PATCH[$v]}") > "$OUT/logs/patch_$v.log" 2>&1 || { cat "$OUT/logs/patch_$v.log"; exit 7; }
    cat "$OUT/logs/patch_$v.log"
  fi
done
for v in base probe; do
  V="$WORK/venv_$v"
  uv venv --quiet --python "$(command -v python3)" "$V"
  {
    uv pip install --python "$V/bin/python" --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0" "torchcodec==0.11.1" "torchvision==0.26.0" \
    && uv pip install --python "$V/bin/python" -c "$WORK/constraints.txt" psutil py-spy "lerobot[dataset] @ file://$WORK/src_$v"
  } > "$OUT/logs/pip_install_$v.log" 2>&1 || { tail -40 "$OUT/logs/pip_install_$v.log"; exit 6; }
  uv pip freeze --python "$V/bin/python" > "$OUT/env/pip_freeze_$v.txt"
  "$V/bin/python" -c "import lerobot, inspect, torch, torchcodec; from lerobot.datasets import streaming_dataset as s; from lerobot.streaming import mp4; print('$v venv', lerobot.__file__, torch.__version__, torchcodec.__version__, 'probe_patch', hasattr(mp4, 'DEFAULT_HEADER_PROBE_BYTES'), 'tfx_patch', 'apply_image_transforms=False' not in inspect.getsource(s))"
done
VENV_BASE="$WORK/venv_base"; VENV_PROBE="$WORK/venv_probe"
VENV_PR3917="$VENV_BASE"
HF="$VENV/bin/hf"
export PATH="$VENV/bin:$PATH" VENV_PR3917 VENV_BASE VENV_PROBE
record python_env ok $((SECONDS - t0))
{
  nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null; cat /sys/fs/cgroup/memory.max 2>/dev/null
  lscpu 2>/dev/null | head -20; free -m; df -h "$WORK"; echo "pr3917 $PR3917_SHA"
  cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null; grep -i cap /proc/self/status
} > "$OUT/env/hardware.txt" 2>&1
env | grep -E '^(SMOKE|STAGES|PF_|AB3_|RUN_NAME|PR3917_SHA)' | sort > "$OUT/env/job_config.txt" || true
upload "setup done"

# ------------------------------------------------------------------ stages
: "${STAGES:=ab3}"
want() { [[ " $STAGES " == *" $1 "* ]]; }
cd "$BUNDLE"
export HF_LEROBOT_HOME="$WORK/lerobot_home"
want ab3 && stage ab3 "$PY" ab3_3917.py orchestrate "$OUT"
du -sh "$WORK"/* > "$OUT/env/disk_usage.txt" 2>/dev/null || true
echo "== done $(date -u +%FT%TZ): $OUT"
