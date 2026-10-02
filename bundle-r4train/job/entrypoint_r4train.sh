#!/usr/bin/env bash
# DATA-11 round 4, train side (HF Jobs GPU: l4x1 / a100-large, image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime,
# Python 3.12). Same restart guard and cgroup memory logging as job/entrypoint_r3.sh.
# Bundle: bundle-r4train/ of $RESULTS_REPO. Results: results/$RUN_NAME/ (RUN_NAME r4train-<flavor>-<UTC>).
#   prefetch  hf download lerobot/smolvla_base (so the first SmolVLA run does not time the download)
#   bench     r4train_bench.py orchestrate: ACT + SmolVLA training step time, synthetic batches, 5 camera setups
# Never enable `set -x` here: HF_TOKEN is in the environment.

set -Eeuo pipefail
umask 022
SMOKE="${SMOKE:-0}"
NO_UPLOAD="${NO_UPLOAD:-0}"
FLAVOR_TAG="${FLAVOR_TAG:-gpu}"
export SMOKE
TS="$(date -u +%Y%m%dT%H%M%SZ)"
if [ "$SMOKE" = 1 ]; then RUN_NAME="${RUN_NAME:-r4train-$FLAVOR_TAG-$TS-smoke}"; else RUN_NAME="${RUN_NAME:-r4train-$FLAVOR_TAG-$TS}"; fi
export RUN_NAME
RESULTS_ROOT="${RESULTS_ROOT:-/results}"
OUT="$RESULTS_ROOT/$RUN_NAME"
WORK="${WORK:-/work}"
mkdir -p "$OUT/logs" "$OUT/env" "$WORK"
exec > >(tee -a "$OUT/logs/entrypoint.log") 2>&1
echo "== DATA-11 r4train job $RUN_NAME start $(date -u +%FT%TZ) SMOKE=$SMOKE host=$(hostname)"

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
  printf 'utc\tcgroup_current_mb\tcgroup_peak_mb\tanon_mb\tfile_mb\ttop_rss_mb_cmd\tgpu_mem_used_mb\tgpu_util_pct\n'
  while sleep 10; do
    cur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
    peak=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo 0)
    anon=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null || echo 0)
    file=$(awk '$1=="file"{print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null || echo 0)
    top=$(ps -eo rss=,args= --sort=-rss 2>/dev/null | head -3 | awk '{printf "%d:%s ", $1/1024, $3}')
    gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' | tr ',' '\t')
    printf '%s\t%d\t%d\t%d\t%d\t%s\t%s\n' "$(date -u +%T)" $((cur / 1048576)) $((peak / 1048576)) $((${anon:-0} / 1048576)) $((${file:-0} / 1048576)) "$top" "${gpu:--	-}"
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
       --repo-type dataset --private --exclude "*.mp4" --exclude "*.tmp" --commit-message "DATA-11 r4train job $RUN_NAME: $1" \
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
: "${PERIODIC_UPLOAD_S:=180}"
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
command -v ps >/dev/null && command -v flock >/dev/null || {
  apt-get update -qq && apt-get install -y -qq --no-install-recommends procps util-linux ca-certificates >/dev/null
}
record system_deps ok $((SECONDS - t0))
t0=$SECONDS
if [ -n "${BUNDLE_DIR:-}" ]; then BUNDLE="$BUNDLE_DIR"; else
  command -v hf >/dev/null || python3 -m pip install -q --root-user-action=ignore "huggingface_hub>=1.0,<2"
  HF="$(command -v hf)"
  "$HF" download "$RESULTS_REPO" --repo-type dataset --include "bundle-r4train/*" --local-dir "$WORK/dl_bundle" > "$OUT/logs/bundle_download.log" 2>&1
  BUNDLE="$WORK/dl_bundle/bundle-r4train"
fi
BUNDLE="$(cd "$BUNDLE" && pwd)"
(cd "$BUNDLE" && sha256sum -c --quiet SHA256SUMS) || { echo "bundle checksum mismatch"; exit 4; }
echo "bundle checksums OK"
mkdir -p "$OUT/bundle" && cp -R "$BUNDLE"/. "$OUT/bundle/"
record bundle ok $((SECONDS - t0))

# ------------------------------------------------------------------ python env (image torch kept as is)
t0=$SECONDS
: "${LEROBOT_SHA:=e0d50211ef236143ae867228662b7dfaba554f02}"
export LEROBOT_SHA PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
python3 -c "import torch; print('image torch', torch.__version__, 'cuda', torch.version.cuda)"
TORCH_V="$(python3 -c 'import torch; print(torch.__version__.split("+")[0])')"
TV_V="$(python3 -c 'import torchvision; print(torchvision.__version__.split("+")[0])' 2>/dev/null || true)"
CU_TAG="$(python3 -c 'import torch; print("cu" + torch.version.cuda.replace(".", ""))')"
{
  if [ -z "$TV_V" ]; then
    python3 -m pip install -q --index-url "https://download.pytorch.org/whl/$CU_TAG" "torch==$TORCH_V" torchvision
    TV_V="$(python3 -c 'import torchvision; print(torchvision.__version__.split("+")[0])')"
  fi
  printf 'torch==%s\ntorchvision==%s\nhuggingface_hub>=1.0,<2\n' "$TORCH_V" "$TV_V" > "$WORK/constraints.txt"
  cat "$WORK/constraints.txt"
  python3 -m pip install -q -c "$WORK/constraints.txt" \
    "lerobot[training,smolvla] @ https://github.com/huggingface/lerobot/archive/$LEROBOT_SHA.tar.gz" "huggingface_hub>=1.0,<2"
} > "$OUT/logs/pip_install.log" 2>&1 || { tail -40 "$OUT/logs/pip_install.log"; exit 6; }
python3 -m pip freeze > "$OUT/env/pip_freeze.txt"
python3 - <<'PY' | tee "$OUT/env/gpu.txt"
import torch, lerobot, transformers, accelerate
assert torch.cuda.is_available(), "torch does not see a GPU"
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("device", torch.cuda.get_device_name(), "capability", torch.cuda.get_device_capability(),
      "mem_gb", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
print("lerobot", lerobot.__file__, "transformers", transformers.__version__, "accelerate", accelerate.__version__)
PY
HF="$(command -v hf)"
record python_env ok $((SECONDS - t0))
{
  nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null; cat /sys/fs/cgroup/memory.max 2>/dev/null
  lscpu 2>/dev/null | head -20; free -m; df -h "$WORK"; nvidia-smi; echo "lerobot $LEROBOT_SHA"
} > "$OUT/env/hardware.txt" 2>&1
env | grep -E '^(SMOKE|STAGES|R4T_|RUN_NAME|LEROBOT_SHA|FLAVOR_TAG)' | sort > "$OUT/env/job_config.txt" || true
upload "setup done"

# ------------------------------------------------------------------ stages
export HF_HOME="$WORK/hf_home" HF_LEROBOT_HOME="$WORK/lerobot_home" TOKENIZERS_PARALLELISM=false
if [ "$SMOKE" = 1 ]; then
  : "${R4T_WARMUP:=2}" "${R4T_STEPS:=3}" "${R4T_SETUPS:=abc}" "${R4T_EXTRA:=}"
else
  : "${R4T_WARMUP:=10}" "${R4T_STEPS:=50}" "${R4T_SETUPS:=abc,droid,molmo,libero,hqf}" "${R4T_EXTRA:=smolvla:64}"
fi
: "${R4T_POLICIES:=act,smolvla}" "${R4T_BUDGET_S:=600}" "${R4T_RUN_TIMEOUT_S:=360}"
: "${STAGES:=prefetch bench}"
want() { [[ " $STAGES " == *" $1 "* ]]; }
cd "$BUNDLE"
want prefetch && stage prefetch "$HF" download lerobot/smolvla_base --exclude "*.gif" --exclude "*.ipynb"
want bench    && stage bench python3 r4train_bench.py orchestrate --out "$OUT/bench" \
  --policies "$R4T_POLICIES" --setups "$R4T_SETUPS" --extra-batch-sizes "$R4T_EXTRA" \
  --warmup "$R4T_WARMUP" --steps "$R4T_STEPS" --device cuda --smolvla-weights pretrained \
  --budget-s "$R4T_BUDGET_S" --run-timeout-s "$R4T_RUN_TIMEOUT_S"
du -sh "$WORK"/* > "$OUT/env/disk_usage.txt" 2>/dev/null || true
echo "== done $(date -u +%FT%TZ): $OUT"
