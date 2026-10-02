# DATA-11 benchmark job (camera stacking vs separate files)

Scaled-down, extrapolation-oriented benchmark for LeRobot roadmap item DATA-11, run on Hugging Face Jobs. Source: `lerobot/abc_130k_v3_train@68651e49` (3 cameras, 224x224, 30 fps, AV1 g=2). Nothing here launches a job on its own.

## Layouts (same decoded pixels, same encoder: libsvtav1 crf 30 g 2 yuv420p, LeRobot default preset 12)

- SEP: one 224x224 file per camera per group.
- STACK: one 224x672 file per group, cameras top to bottom in order `top, left_wrist, right_wrist`.
- BIG: per camera, stream-copy concat of 3 consecutive SEP groups (3x frames per file, 3x fewer files).

Groups are whole episodes (`selection.json`, made locally by `select_slice.py` from the pinned meta). Families: `ckpt_1k`, `ckpt_5k`, `ckpt_full` (~24.6k frames, the real median frames/file of ABC v3), 3 groups each, for the open-cost curve; `thrash`, 48 single-episode groups of ~2k frames, for the cache-ratio sweep. Smoke uses `ckpt_1k`, `ckpt_5k` and 12 thrash groups (a subset of the same files).

## Stages (`job/entrypoint.sh`, order: fetch encode big verify upload remote opencost loader model)

| stage | script | what |
|---|---|---|
| fetch | `s1_fetch.py` | download only the selected source MP4s (4.34 GB full, 0.82 GB smoke), sizes checked |
| encode | `s2_encode.py` | PyAV process: decode sources once, write SEP x3 + STACK per group; records SVT's effective preset |
| big | `s3_big.py` | BIG via `ffmpeg -f concat -c copy`; ffprobe: packets == frames, pts increasing, pts == dts |
| verify | `s4_verify.py` | timestamps exact, camera order via crop PSNR matrix, BIG bit-identical to SEP |
| upload | `s8_upload.py` | push SEP / STACK / BIG of the `ckpt_*` families to the private repo under `variants/<run>/` (`HfApi.upload_folder`), record the commit, read back at that commit (size, LFS sha256, head/tail bytes through HfFileSystem) |
| remote | `s9_remote.py` | PRIMARY. Read the variants over `hf://datasets/<repo>@<rev>/...` like LeRobot: fsspec handle (5 MiB readahead) -> `CountingFile` -> `VideoDecoder`. Arms: seq / par / one camera x first touch / steady, plus N concurrent processes. HTTP requests (resolve / CDN / API), bytes, 429s, resolver budget. Checks: remote == local decode, SEP == BIG |
| opencost | `s5_opencost.py` | per-open cost (construct, first frame, close), steady decode, warm and cold, moov bytes, MB per decoder |
| loader | `s6_loader.py` | DataLoader arms, per-worker LRU (PR #4556 semantics), thrash ratios + checkpoint caches, warm and cold, CI stopping rule |
| model | `s7_model.py` | fits, checkpoint validation, EXTRAPOLATED predictions for ABC v3, DROID v3, a 2-file control; remote model (per-fetch, per-open, requests per open, resolver ceiling) for ABC train and DROID v30 |

Results go to `results/<run>/` of the private dataset repo `hadriencornier/lerobot-data11-bench` after every stage; `STATUS.tsv` lists stage status, seconds and cgroup peak memory. The entrypoint refuses to run twice for one job id (OOM restart guard) and samples cgroup memory every 10 s (`logs/mem.tsv`).

## Launch

```bash
cd /Users/HCornier/Documents/Personal/LeRobot/data11-bench
export HF_TOKEN="$(set -a; . ~/.env >/dev/null 2>&1; printf '%s' "$HUGGING_FACE_PERSONAL_KEY_USE_SPARINGLY")"
HF=/Users/HCornier/Documents/Personal/LeRobot/lerobot/.venv/bin/hf
bash job/make_bundle.sh
$HF upload hadriencornier/lerobot-data11-bench job/_bundle bundle --repo-type dataset --private --commit-message "DATA-11 job bundle"

# smoke
$HF jobs run --detach --flavor cpu-upgrade --timeout 45m \
  --secrets HF_TOKEN --env RESULTS_REPO=hadriencornier/lerobot-data11-bench --env SMOKE=1 \
  python:3.12-slim-trixie \
  bash -c 'set -e; pip install -q --root-user-action=ignore "huggingface_hub>=1.0"; hf download "$RESULTS_REPO" --repo-type dataset --include "bundle/*" --local-dir /work/dl >/dev/null; exec bash /work/dl/bundle/job/entrypoint.sh'
```

Main run (NOT launched; needs Hadrien's go). Smoke `6ab59b6d6b030d633f68f778` passed every stage in 10.8 min.

```bash
$HF jobs run --detach --name lerobot-data11-main --flavor cpu-upgrade --timeout 8h \
  --secrets HF_TOKEN --env RESULTS_REPO=hadriencornier/lerobot-data11-bench \
  --env DL_BUDGET_S=21600 --env DL_ARM_CAP_S=180 --env DL_REPEATS=2 \
  python:3.12-slim-trixie \
  bash -c 'set -e; pip install -q --root-user-action=ignore "huggingface_hub>=1.0"; hf download "$RESULTS_REPO" --repo-type dataset --include "bundle/*" --local-dir /work/dl >/dev/null; exec bash /work/dl/bundle/job/entrypoint.sh'
```

Monitor with `$HF jobs logs <id>`, `$HF jobs inspect <id>`, and `python job/watch_job.py <id> --max-running-min <n>` as a safety cancel.

## Local test

`docker run --rm -v "$PWD/job/_bundle:/bundle:ro" -e NO_UPLOAD=1 -e SMOKE=1 -e BUNDLE_DIR=/bundle python:3.12-slim-trixie bash /bundle/job/entrypoint.sh`

## Remote stage notes

- Every stage that decodes pins thread pools with `pin_threads()` (HF containers report 64 CPUs, the cgroup quota is 8).
- Resolver budget: the `ratelimit` header on resolve calls ("resolvers", 12000 per 300 s per token when checked on 2026-09-24). Each fsspec block fetch costs one resolve call plus one CDN call.
- `huggingface_hub` is pinned to 1.30.0 in the job venv because the HTTP spy patches its internals.
- Knobs: `UP_FAMILIES`, `UP_REPO`, `RM_FAMILY` (default ckpt_full, smoke ckpt_5k), `RM_N_FIRST`, `RM_N_STEADY`, `RM_REPEATS`, `RM_OPEN_N`, `RM_CONC` (default = CPU quota), `RM_CONC_S`, `RM_BUDGET_S`.
