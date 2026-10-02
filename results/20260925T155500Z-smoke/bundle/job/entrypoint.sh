#!/usr/bin/env bash
# DATA-11 (camera stacking vs separate files) benchmark job for a fresh Linux container
# (HF Jobs, image python:3.12-slim-trixie: Debian FFmpeg 7.1, which torchcodec 0.11 supports).
#
# Stages (each stage's results are uploaded as soon as it finishes):
#   setup     restart guard, apt deps, bundle checksum, uv venv (torch CPU + torchcodec + PyAV), env info
#   fetch     s1_fetch.py     selected ABC v3 source MP4s at the pinned revision (sizes checked)
#   encode    s2_encode.py    decode sources once -> SEP (x3) + STACK per group, LeRobot default encoder
#   big       s3_big.py       BIG = stream-copy concat of 3 SEP groups; ffprobe packet checks
#   verify    s4_verify.py    timestamps exact, camera order (crop PSNR), BIG bit-identical to SEP
#   upload    s8_upload.py    push SEP / STACK / BIG of the ckpt_* families to the PRIVATE repo (variants/<run>/), read back at the commit
#   remote    s9_remote.py    PRIMARY: read the variants over hf:// (fsspec handle -> VideoDecoder), requests, bytes, rate-limit budget
#   opencost  s5_opencost.py  per-open cost vs frames/file, warm + cold, moov bytes, MB per decoder
#   loader    s6_loader.py    DataLoader arms (per-worker LRU, PR #4556 semantics), warm + cold
#   model     s7_model.py     fit, validate on the checkpoint family, EXTRAPOLATED predictions
#
# Results: $RESULTS_ROOT/<run name>/ (default /results/<UTC>[-smoke]/), uploaded to the dataset repo
# $RESULTS_REPO under results/<run name>/ (private). Encoded videos of the ckpt_* families are uploaded
# by stage upload to $UP_REPO (default $RESULTS_REPO, private) under variants/<run name>/ for stage remote.
# Scripts: bundle/ of $RESULTS_REPO (downloaded by the launch command); bundle/SHA256SUMS is verified.
#
# Env: RESULTS_REPO (required unless NO_UPLOAD=1), SMOKE=1, NO_UPLOAD=1, RESULTS_ROOT (/results),
#      WORK (/work), RUN_NAME, BUNDLE_DIR, SRC_ROOT (reuse downloaded sources), STAGES,
#      PERIODIC_UPLOAD_S, ENC_PROCS, plus every knob the stage scripts read (OC_*, DL_*, VERIFY_*, UP_*, RM_*).
#
# Never enable `set -x` here: HF_TOKEN is in the environment.

set -Eeuo pipefail
umask 022

SMOKE="${SMOKE:-0}"
NO_UPLOAD="${NO_UPLOAD:-0}"
export SMOKE
TS="$(date -u +%Y%m%dT%H%M%SZ)"
if [ "$SMOKE" = 1 ]; then RUN_NAME="${RUN_NAME:-$TS-smoke}"; else RUN_NAME="${RUN_NAME:-$TS}"; fi
export RUN_NAME  # stage upload names variants/<run name>/ after it
RESULTS_ROOT="${RESULTS_ROOT:-/results}"
OUT="$RESULTS_ROOT/$RUN_NAME"
WORK="${WORK:-/work}"
mkdir -p "$OUT/logs" "$OUT/env" "$WORK"
exec > >(tee -a "$OUT/logs/entrypoint.log") 2>&1
echo "== DATA-11 job $RUN_NAME start $(date -u +%FT%TZ) SMOKE=$SMOKE host=$(hostname)"

if [ "$NO_UPLOAD" != 1 ]; then
  : "${RESULTS_REPO:?set RESULTS_REPO (or NO_UPLOAD=1)}"
  if [ -z "${HF_TOKEN:-}" ]; then echo "HF_TOKEN is not set (pass --secrets HF_TOKEN)"; exit 2; fi
  echo "HF_TOKEN present (value not shown); results -> dataset $RESULTS_REPO:results/$RUN_NAME"
  # Restart guard: after an OOM kill HF Jobs restarted the container from scratch and --timeout did
  # not cap the total time. One marker per job id in the results repo; a second attempt exits.
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

# Memory sampler: cgroup usage (includes page cache) plus the largest processes, every 10 s.
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
upload() {  # commit message
  if [ "$NO_UPLOAD" = 1 ]; then echo "(NO_UPLOAD=1: not uploading: $1)"; return 0; fi
  if [ -z "$HF" ]; then echo "WARNING: no hf CLI yet, cannot upload ($1)"; return 0; fi
  if flock -w 900 "$WORK/.upload.lock" env HF_HUB_OFFLINE=0 "$HF" upload "$RESULTS_REPO" "$OUT" "results/$RUN_NAME" \
       --repo-type dataset --private --exclude "*.mp4" --commit-message "DATA-11 job $RUN_NAME: $1" \
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

stage() {  # non-fatal: stage NAME cmd...
  local name="$1"; shift
  local t0=$SECONDS
  echo "== stage $name start $(date -u +%FT%TZ)"
  if "$@" 2>&1 | tee "$OUT/logs/$name.log"; then record "$name" ok $((SECONDS - t0)); else record "$name" "FAILED rc=${PIPESTATUS[0]}" $((SECONDS - t0)); fi
  cat /sys/fs/cgroup/memory.events > "$OUT/logs/memory_events_after_$name.txt" 2>/dev/null || true
  upload "after $name"
}

# ------------------------------------------------------------------ 1. system deps
t0=$SECONDS
export DEBIAN_FRONTEND=noninteractive
if command -v apt-get >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends ffmpeg procps util-linux ca-certificates coreutils findutils >/dev/null
  apt-get install -y -qq --no-install-recommends util-linux-extra >/dev/null 2>&1 || true  # lscpu / fincore on newer Debian
fi
FFMPEG_MAJOR="$(ffmpeg -version 2>/dev/null | head -1 | sed -E 's/^ffmpeg version n?([0-9]+).*/\1/')"
echo "system ffmpeg: $(ffmpeg -version 2>/dev/null | head -1) (major $FFMPEG_MAJOR)"
case "$FFMPEG_MAJOR" in 4|5|6|7|8) ;; *) echo "ffmpeg major '$FFMPEG_MAJOR' is outside torchcodec 0.11's supported 4-8"; exit 3;; esac
record system_deps ok $((SECONDS - t0))

# ------------------------------------------------------------------ 2. bundle
t0=$SECONDS
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${BUNDLE_DIR:-}" ]; then
  BUNDLE="$BUNDLE_DIR"
elif [ -f "$HERE/../common.py" ] && [ -f "$HERE/../selection.json" ]; then
  BUNDLE="$(cd "$HERE/.." && pwd)"
else
  command -v hf >/dev/null || python3 -m pip install -q --root-user-action=ignore "huggingface_hub>=1.0"
  HF="$(command -v hf)"
  "$HF" download "$RESULTS_REPO" --repo-type dataset --include "bundle/*" --local-dir "$WORK/dl" > "$OUT/logs/bundle_download.log" 2>&1
  BUNDLE="$WORK/dl/bundle"
fi
BUNDLE="$(cd "$BUNDLE" && pwd)"
for f in common.py selection.json s1_fetch.py s2_encode.py s3_big.py s4_verify.py s5_opencost.py s6_loader.py s7_model.py s8_upload.py s9_remote.py \
         readers.py s10_prep.py s11_access.py s12_stream.py s13_summary.py; do
  [ -f "$BUNDLE/$f" ] || { echo "bundle is missing $f"; exit 4; }
done
if [ -f "$BUNDLE/SHA256SUMS" ]; then
  (cd "$BUNDLE" && sha256sum -c --quiet SHA256SUMS) || { echo "bundle checksum mismatch"; exit 4; }
  echo "bundle checksums OK"
else
  echo "WARNING: bundle has no SHA256SUMS"
fi
mkdir -p "$OUT/bundle"
cp -R "$BUNDLE"/. "$OUT/bundle/"
record bundle ok $((SECONDS - t0))

# ------------------------------------------------------------------ 3. python env
t0=$SECONDS
VENV="$WORK/venv"
python3 -m pip install -q --root-user-action=ignore uv
export UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300
[ -x "$VENV/bin/python" ] || uv venv --quiet --python "$(command -v python3)" "$VENV"
PY="$VENV/bin/python"
{
  # CPU builds of torch and torchcodec from the PyTorch CPU index (x86_64 only; the aarch64 PyPI
  # torchcodec wheel links CUDA libraries), everything else from PyPI
  uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0" "torchcodec==0.11.1" \
  && uv pip install --python "$PY" "av>=15,<16" numpy pandas pyarrow psutil "huggingface_hub==1.30.0"  # pinned: the hf:// spy patches its internals
} > "$OUT/logs/pip_install.log" 2>&1 || { cat "$OUT/logs/pip_install.log"; exit 6; }
# round 2: lerobot from the PR #4702 branch (streaming timestamps fix), only for the stages that import it.
# GitHub tarball of a pinned commit (no git needed); torch / torchcodec / huggingface_hub kept at the pins above.
: "${LEROBOT_REF:=a195a91ff815ad730373276ef8a4cc7c04c5c074}"  # Hadrien-Cornier/lerobot fix/streaming-video-timestamps-takeover
export LEROBOT_REF
if [[ " ${STAGES:-} " == *" stream "* || " ${STAGES:-} " == *" prep2 "* ]]; then
  {
    printf 'torch==2.11.0\ntorchcodec==0.11.1\nhuggingface_hub==1.30.0\n' > "$WORK/constraints.txt"
    uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cpu "torchvision==0.26.0" \
    && uv pip install --python "$PY" -c "$WORK/constraints.txt" \
         "lerobot[dataset] @ https://github.com/Hadrien-Cornier/lerobot/archive/$LEROBOT_REF.tar.gz"
  } > "$OUT/logs/pip_install_lerobot.log" 2>&1 || { tail -40 "$OUT/logs/pip_install_lerobot.log"; exit 6; }
  "$PY" -c "import lerobot, datasets; from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset; print('lerobot', lerobot.__file__, 'datasets', datasets.__version__)"
fi
uv pip freeze --python "$PY" > "$OUT/env/pip_freeze.txt"
HF="$VENV/bin/hf"
export PATH="$VENV/bin:$PATH"
# torchcodec and PyAV each bring FFmpeg libraries; they run in separate processes (as in the stages)
"$PY" -c "import torch, torchcodec; from torchcodec.decoders import VideoDecoder; print('torch', torch.__version__, '| torchcodec', torchcodec.__version__)"
"$PY" -c "import av; print('av', av.__version__, {k: '.'.join(map(str, v)) for k, v in av.library_versions.items()}); av.codec.Codec('libsvtav1', 'w'); print('libsvtav1 encoder available in PyAV')"
record python_env ok $((SECONDS - t0))

# ------------------------------------------------------------------ 4. hardware / env info
{
  echo "## date";        date -u +%FT%TZ
  echo "## uname";       uname -a
  echo "## os-release";  cat /etc/os-release 2>/dev/null || true
  echo "## nproc";       nproc; nproc --all
  echo "## cgroup cpu.max / memory.max"; cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a; cat /sys/fs/cgroup/memory.max 2>/dev/null || echo n/a
  echo "## lscpu";       lscpu 2>/dev/null || grep -m1 "model name" /proc/cpuinfo
  echo "## free -m";     free -m
  echo "## df -h";       df -h "$WORK" "$RESULTS_ROOT" /dev/shm 2>/dev/null || df -h
  echo "## mounts";      grep -E " (/|/work|/results|/dev/shm) " /proc/mounts 2>/dev/null || true
  echo "## block devices"; lsblk 2>/dev/null || cat /proc/partitions 2>/dev/null || true
  echo "## ffmpeg";      ffmpeg -version 2>/dev/null | head -3
  echo "## fincore";     command -v fincore || echo "fincore unavailable"
  echo "## libav sonames"; ls /usr/lib/*-linux-gnu/libav{util,codec,format}.so.* 2>/dev/null || true
  echo "## python";      "$PY" --version
} > "$OUT/env/hardware.txt" 2>&1
"$PY" - > "$OUT/env/python_versions.json" <<'EOF'
import json, os, sys
info = {"python": sys.version, "os_cpu_count": os.cpu_count(), "sched_affinity": len(os.sched_getaffinity(0))}
import torch, torchcodec
info.update(torch=torch.__version__, torch_num_threads=torch.get_num_threads(), torchcodec=torchcodec.__version__)
try:
    from torchcodec._core import get_ffmpeg_library_versions
    info["torchcodec_ffmpeg"] = get_ffmpeg_library_versions()
except Exception as exc:
    info["torchcodec_ffmpeg"] = f"unavailable: {exc}"
for m in ("numpy", "pandas", "pyarrow", "psutil", "huggingface_hub", "fsspec", "httpx"):
    info[m] = __import__(m).__version__
print(json.dumps(info, indent=1, default=str))
EOF
"$PY" -c "import av, json; print(json.dumps({'av': av.__version__, 'av_ffmpeg': {k: '.'.join(map(str, v)) for k, v in av.library_versions.items()}}))" > "$OUT/env/pyav_versions.json"
cat "$OUT/env/python_versions.json" "$OUT/env/pyav_versions.json"
env | grep -E '^(SMOKE|STAGES|ENC_PROCS|OC_|DL_|VERIFY_|UP_|RM_|RUN_NAME)' | sort > "$OUT/env/job_config.txt" || true
upload "setup done"

# ------------------------------------------------------------------ 5. stages
SEL="$BUNDLE/selection.json"
SRC="${SRC_ROOT:-$WORK/src}"
ENC="$WORK/enc"
: "${STAGES:=fetch encode big verify upload remote opencost loader model}"
want() { [[ " $STAGES " == *" $1 "* ]]; }
cd "$BUNDLE"
want fetch    && stage fetch    "$PY" s1_fetch.py "$SEL" "$SRC" "$OUT"
want encode   && stage encode   "$PY" s2_encode.py "$SEL" "$SRC" "$ENC" "$OUT"
want big      && stage big      "$PY" s3_big.py "$SEL" "$ENC" "$OUT"
want verify   && stage verify   "$PY" s4_verify.py "$SEL" "$ENC" "$OUT"
if [ "$NO_UPLOAD" = 1 ]; then
  want upload && record upload "skipped (NO_UPLOAD=1)" 0
else
  want upload && stage upload   "$PY" s8_upload.py "$SEL" "$ENC" "$OUT"
  want remote && stage remote   "$PY" s9_remote.py "$SEL" "$ENC" "$OUT"
fi
want opencost && stage opencost "$PY" s5_opencost.py "$SEL" "$ENC" "$OUT"
want loader   && stage loader   "$PY" s6_loader.py "$SEL" "$ENC" "$OUT"
want model    && stage model    "$PY" s7_model.py "$SEL" "$OUT"
# round 2 (reuse the round-1 variants; no fetch / encode)
export HF_LEROBOT_HOME="$WORK/lerobot_home"
want prep2    && stage prep2    "$PY" s10_prep.py "$SEL" "$WORK" "$OUT"
want stream   && stage stream   "$PY" s12_stream.py "$OUT/prep/prep.json" "$OUT"
want access   && stage access   "$PY" s11_access.py "$OUT/prep/prep.json" "$OUT"
want summary2 && stage summary2 "$PY" s13_summary.py "$OUT"
du -sh "$SRC" "$ENC" > "$OUT/env/disk_usage.txt" 2>/dev/null || true
df -h "$WORK" >> "$OUT/env/disk_usage.txt" 2>/dev/null || true

# ------------------------------------------------------------------ 6. index
{
  echo "# DATA-11 job $RUN_NAME"
  echo
  echo "SMOKE=$SMOKE. Source \`$(python3 -c "import json;s=json.load(open('$SEL'));print(s['repo_id']+'@'+s['revision'])")\`."
  echo
  echo '## Stages'; echo; echo '```'; cat "$STATUS"; echo '```'; echo
  [ -f "$OUT/summary.md" ] && { echo "---"; echo; cat "$OUT/summary.md"; }
  [ -f "$OUT/model/summary.md" ] && { echo "---"; echo; cat "$OUT/model/summary.md"; }
  [ ! -f "$OUT/model/summary.md" ] && [ -f "$OUT/remote/summary.md" ] && { echo "---"; echo; cat "$OUT/remote/summary.md"; }
} > "$OUT/README.md"
echo "== done $(date -u +%FT%TZ): $OUT"
