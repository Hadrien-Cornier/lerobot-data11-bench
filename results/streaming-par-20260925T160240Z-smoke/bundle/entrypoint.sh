#!/usr/bin/env bash
# Streaming camera-parallelism benchmark job (HF Jobs, image python:3.12-slim-trixie, cpu-upgrade).
# Arms: base (main 1bc0bdfb + #4702), p1 (+ parallel cameras), p1p2 (+ cache lock), pr3917 (PR head).
# Results: results/$RUN_NAME/ of the private dataset $RESULTS_REPO (RUN_NAME=streaming-par-<UTC>[-smoke]).
# Bundle: bundle-streaming-par/ of $RESULTS_REPO (SHA256SUMS verified).
# Never enable `set -x` here: HF_TOKEN is in the environment.
set -Eeuo pipefail
umask 022

SMOKE="${SMOKE:-0}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
if [ "$SMOKE" = 1 ]; then RUN_NAME="${RUN_NAME:-streaming-par-$TS-smoke}"; else RUN_NAME="${RUN_NAME:-streaming-par-$TS}"; fi
OUT="/results/$RUN_NAME"
WORK=/work
mkdir -p "$OUT/logs" "$OUT/env" "$WORK"
exec > >(tee -a "$OUT/logs/entrypoint.log") 2>&1
echo "== streaming-par job $RUN_NAME start $(date -u +%FT%TZ) SMOKE=$SMOKE host=$(hostname)"
: "${RESULTS_REPO:?set RESULTS_REPO}"
if [ -z "${HF_TOKEN:-}" ]; then echo "HF_TOKEN is not set (pass --secrets HF_TOKEN)"; exit 2; fi
echo "HF_TOKEN present (value not shown)"

# Restart guard: one marker per job id; a restarted container (e.g. after an OOM kill) exits at once.
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
  echo "WARNING: no job id found; restart guard disabled"
fi

# Memory sampler (cgroup, includes page cache) every 10 s
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

HF="$(command -v hf || true)"
upload() {
  [ -n "$HF" ] || { echo "WARNING: no hf CLI, cannot upload ($1)"; return 0; }
  if flock -w 900 "$WORK/.upload.lock" "$HF" upload "$RESULTS_REPO" "$OUT" "results/$RUN_NAME" \
       --repo-type dataset --private --exclude "*.npz" --exclude "*.tmp" --commit-message "streaming-par $RUN_NAME: $1" \
       > "$OUT/logs/upload_last.log" 2>&1; then
    echo "uploaded results/$RUN_NAME ($1)"
  else
    echo "WARNING: upload failed ($1); see logs/upload_last.log"
  fi
}
on_exit() {
  local rc=$?
  echo "== exit code $rc at $(date -u +%FT%TZ)"
  cat /sys/fs/cgroup/memory.events > "$OUT/logs/memory_events_final.txt" 2>/dev/null || true
  upload "final (exit $rc)"
}
trap on_exit EXIT
trap 'exit 143' TERM
( while sleep "${PERIODIC_UPLOAD_S:-600}"; do upload "periodic" > /dev/null; done ) &

# ------------------------------------------------------------------ system deps
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends ffmpeg git procps util-linux ca-certificates >/dev/null
echo "system ffmpeg: $(ffmpeg -version 2>/dev/null | head -1)"

# ------------------------------------------------------------------ bundle
"$HF" download "$RESULTS_REPO" --repo-type dataset --include "bundle-streaming-par/*" --local-dir "$WORK/dl" > "$OUT/logs/bundle_download.log" 2>&1
B="$WORK/dl/bundle-streaming-par"
(cd "$B" && sha256sum -c --quiet SHA256SUMS) || { echo "bundle checksum mismatch"; exit 4; }
echo "bundle checksums OK"
mkdir -p "$OUT/bundle" && cp -R "$B"/. "$OUT/bundle/"

# ------------------------------------------------------------------ sources
BASE_SHA=1bc0bdfb20ad4f4f76dc8a68a0e0746d54f50de9
EXPECT_3917=9bea4a19
LR="$WORK/lerobot"
git clone -q --filter=blob:none https://github.com/huggingface/lerobot.git "$LR"
git -C "$LR" fetch -q origin "pull/3917/head:pr3917"
git -C "$LR" -c advice.detachedHead=false checkout -q "$BASE_SHA"
for arm in base p1 p1p2; do git -C "$LR" worktree add -q --detach "$WORK/src_$arm" "$BASE_SHA"; done
git -C "$LR" worktree add -q --detach "$WORK/src_pr3917" pr3917
git -C "$WORK/src_base" apply "$B/patches/pr4702.patch"
git -C "$WORK/src_p1" apply "$B/patches/pr4702.patch" "$B/patches/p1_parallel_cameras.patch"
git -C "$WORK/src_p1p2" apply "$B/patches/pr4702.patch" "$B/patches/p1_parallel_cameras.patch" "$B/patches/p2_cache_lock.patch"
{
  echo "base_sha $BASE_SHA"
  echo "pr3917_head $(git -C "$LR" rev-parse pr3917) (expected prefix $EXPECT_3917)"
  for arm in base p1 p1p2; do echo "src_$arm diff: $(git -C "$WORK/src_$arm" diff --stat | tail -1)"; done
} | tee "$OUT/env/revisions.txt"
case "$(git -C "$LR" rev-parse pr3917)" in $EXPECT_3917*) ;; *) echo "WARNING: PR #3917 head moved since the patches were prepared";; esac

# ------------------------------------------------------------------ venvs (CPU torch + torchcodec, then lerobot[dataset])
python3 -m pip install -q --root-user-action=ignore uv
export UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300
mkvenv() {  # venv source_tree
  uv venv --quiet --python "$(command -v python3)" "$1"
  uv pip install --python "$1/bin/python" --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0" "torchcodec==0.11.1"
  uv pip install --python "$1/bin/python" -e "$2[dataset]" "huggingface_hub==1.30.0" psutil
  uv pip freeze --python "$1/bin/python" > "$OUT/env/pip_freeze_$(basename "$1").txt"
}
mkvenv "$WORK/venv_main" "$WORK/src_base" > "$OUT/logs/pip_main.log" 2>&1 || { tail -30 "$OUT/logs/pip_main.log"; exit 6; }
mkvenv "$WORK/venv_pr3917" "$WORK/src_pr3917" > "$OUT/logs/pip_pr3917.log" 2>&1 || { tail -30 "$OUT/logs/pip_pr3917.log"; exit 6; }
"$WORK/venv_main/bin/python" -c "import torch, torchcodec, lerobot; print('main venv: torch', torch.__version__, 'torchcodec', torchcodec.__version__, lerobot.__file__)"
"$WORK/venv_pr3917/bin/python" -c "import torch, torchcodec, lerobot; print('pr3917 venv: torch', torch.__version__, 'torchcodec', torchcodec.__version__, lerobot.__file__)"
{
  nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null; cat /sys/fs/cgroup/memory.max 2>/dev/null
  grep -m1 "model name" /proc/cpuinfo; free -m; df -h "$WORK"; ffmpeg -version 2>/dev/null | head -1
} > "$OUT/env/hardware.txt" 2>&1
upload "setup done"

# ------------------------------------------------------------------ bench
export SRC_BASE="$WORK/src_base" SRC_P1="$WORK/src_p1" SRC_P1P2="$WORK/src_p1p2" SRC_PR3917="$WORK/src_pr3917"
export VENV_MAIN="$WORK/venv_main" VENV_PR3917="$WORK/venv_pr3917"
env | grep -E '^(SMOKE|ARMS|REPEATS|N_STEADY|CAP_S|BENCH_BUDGET_S|HASH_N|SAVE_N|REGIMES|PR3917_EPISODES|WARMUP_CAP_S)=' | sort > "$OUT/env/job_config.txt" || true
cd "$B"
python3 bench_stream.py orchestrate "$OUT" 2>&1 | tee "$OUT/logs/bench.log"
echo "== done $(date -u +%FT%TZ): $OUT"
