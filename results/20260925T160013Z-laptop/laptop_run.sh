#!/bin/bash
# Item 4: laptop run over the internet, HARD CAP 500 MB of HTTP bytes in total (spy-counted), stages run in order,
# each one gets the budget left by the previous ones.
set -u
S=/private/tmp/claude-504/-Users-HCornier-Documents-Personal-LeRobot/37df4008-8773-4ef0-a5e8-e2bacc9534c8/scratchpad
B=/Users/HCornier/Documents/Personal/LeRobot/data11-bench
PY=/Users/HCornier/Documents/Personal/LeRobot/lerobot/.venv/bin/python
export HF_TOKEN="$(set -a; . ~/.env >/dev/null 2>&1; printf '%s' "$HUGGING_FACE_PERSONAL_KEY_USE_SPARINGLY")"
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@8/lib HF_LEROBOT_HOME=$S/lrhome_laptop HF_HUB_DISABLE_PROGRESS_BARS=1
PREP=$S/smoke/results/20260925T155500Z-smoke/prep/prep.json
OUT=$S/r2local/laptop
mkdir -p $OUT
CAP=500
used() { $PY - "$OUT" <<'P' 2>/dev/null
import json, sys, pathlib
t = 0
for f in ("stream/stream.json", "access_a/access/access.json", "access_b/access/access.json"):
    p = pathlib.Path(sys.argv[1]) / f
    if p.exists():
        t += json.loads(p.read_text()).get("remote_totals", {}).get("http_bytes", 0)
print(int(t / 1e6))
P
}
cd $B
echo "start $(date -u +%T)"
# 1. streaming order, S = 1 (one shard), warm, one reader: real StreamingLeRobotDataset (#4702 branch)
STREAM_ARMS="SEP_S1_c100_seq STACK_S1_c100" STREAM_REPEATS=1 STREAM_N=400 STREAM_WARM=50 STREAM_CONC=0 STREAM_CAP_S=120 STREAM_MB_CAP=80 \
  LEROBOT_REF=a195a91f PYTHONPATH=$S/lr4702/src $PY s12_stream.py $PREP $OUT 2>&1 | grep -E "^\[|MB cap|Error|rror" | grep -v objc
U=$(used); echo "used after stream: $U MB"
# 2. random access, warm, one reader: fs256K, rc256K (resolve once), exact (no precomputed index: 2 requests per open)
mkdir -p $OUT/access_a
ACC_ARMS="SEP|seq|fs256K|-|warm STACK|-|fs256K|-|warm SEP|seq|rc256K|-|warm STACK|-|rc256K|-|warm SEP|seq|exact|-|warm STACK|-|exact|-|warm" \
  ACC_N_WARM=15 ACC_REPEATS=2 ACC_CONC=0 ACC_CHECK_N=0 ACC_EXACT_HINT=0 ACC_MB_CAP=$(( (CAP - U) / 2 )) \
  $PY s11_access.py $PREP $OUT/access_a 2>&1 | grep -E "^\[|Error|rror" | grep -v objc | cut -c1-260
U=$(used); echo "used after access_a: $U MB"
# 3. fsspec default (5 MiB) with what is left
mkdir -p $OUT/access_b
ACC_ARMS="SEP|seq|fs5M|-|warm STACK|-|fs5M|-|warm" ACC_N_WARM=6 ACC_REPEATS=1 ACC_CONC=0 ACC_CHECK_N=0 ACC_EXACT_HINT=0 ACC_MB_CAP=$(( CAP - U - 20 )) \
  $PY s11_access.py $PREP $OUT/access_b 2>&1 | grep -E "^\[|Error|rror" | grep -v objc | cut -c1-260
U=$(used); echo "TOTAL used: $U MB (cap $CAP)"
echo "end $(date -u +%T)"
