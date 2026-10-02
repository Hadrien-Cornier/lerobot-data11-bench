#!/usr/bin/env bash
# Run the #3917 streaming test suite on Linux for BASE (PR head) and BOTH (head + probe + transforms patches).
# Bundle: bundle-ab2/ of $RESULTS_REPO. Results: results/$RUN_NAME/. Never enable `set -x`: HF_TOKEN is in the environment.
set -Eeuo pipefail
RUN_NAME="${RUN_NAME:-ab2-tests-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="/results/$RUN_NAME"; WORK=/work; mkdir -p "$OUT" "$WORK"
exec > >(tee -a "$OUT/entrypoint.log") 2>&1
JOB_KEY="$(hostname | sed -nE 's/^j-[^-]+-([0-9a-f]{24})-.*/\1/p')"
python3 - <<'PY' || exit 0
import os, sys, socket, re
from huggingface_hub import HfApi
m = re.match(r"^j-[^-]+-([0-9a-f]{24})-", socket.gethostname()); key = m.group(1) if m else None
if key:
    api, repo, path = HfApi(), os.environ["RESULTS_REPO"], f"attempts/{key}"
    if api.file_exists(repo, path, repo_type="dataset"):
        print("RESTART DETECTED"); sys.exit(1)
    api.upload_file(path_or_fileobj=os.environ["RUN_NAME"].encode(), path_in_repo=path, repo_id=repo, repo_type="dataset")
PY
upload() { hf upload "$RESULTS_REPO" "$OUT" "results/$RUN_NAME" --repo-type dataset --private --commit-message "ab2 tests $RUN_NAME: $1" >/dev/null 2>&1 || echo "upload failed"; }
trap 'upload final' EXIT
apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg patch curl git >/dev/null
BUNDLE=/work/dl_bundle/bundle-ab2
SHA=6d94598523738d19700e345e9b42fb3d2db352f5
curl -fsSL "https://github.com/huggingface/lerobot/archive/$SHA.tar.gz" -o "$WORK/src.tar.gz"
python3 -m pip install -q --root-user-action=ignore uv
for v in base both; do
  mkdir -p "$WORK/src_$v" && tar -xzf "$WORK/src.tar.gz" -C "$WORK/src_$v" --strip-components=1
  if [ $v = both ]; then
    (cd "$WORK/src_$v" && patch -p1 --batch < "$BUNDLE/patches-3917/probe-exact-reads.diff" && patch -p1 --batch < "$BUNDLE/patches-3917/transforms-on-decode-threads.diff")
  fi
  V="$WORK/venv_$v"; uv venv -q --python "$(command -v python3)" "$V"
  uv pip install -q --python "$V/bin/python" --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0" "torchcodec==0.11.1" "torchvision==0.26.0"
  uv pip install -q --python "$V/bin/python" -c <(printf 'torch==2.11.0\ntorchcodec==0.11.1\n') "lerobot[dataset,test] @ file://$WORK/src_$v" pytest-timeout > "$OUT/pip_$v.log" 2>&1
done
for v in base both; do
  cd "$WORK/src_$v"
  "$WORK/venv_$v/bin/python" -m pytest -q -p no:cacheprovider --timeout=300 -rf \
    tests/datasets/test_episode_video_streaming.py tests/datasets/test_streaming*.py tests/test_streaming_sidecar.py tests/test_streaming_core_imports.py \
    > "$OUT/pytest_$v.log" 2>&1 || true
  echo "== $v: $(tail -1 "$OUT/pytest_$v.log")"
  grep -E "^FAILED" "$OUT/pytest_$v.log" | head -20 || true
  upload "$v done"
done
