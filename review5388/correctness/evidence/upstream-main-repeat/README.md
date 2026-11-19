# Repeating streams with unequal worker shards

LeRobot upstream `8ec1a474da5f3f90a670d9343386995ec9f81264` completed 23 CPU ACT updates with one rank, four workers, and four samples per batch. The 86-frame fixture assigned 31, 19, 18, and 18 frames to the four workers. In the first 86 training samples, 79 were unique, seven were repeat visits, and seven fixture frames had yet to appear. The first repeat was sample 72. The trainer logged `epch:1.02` after 88 samples, at which point 79 frames had appeared.

The trace in [batches.jsonl](batches.jsonl) records the anchor indices delivered to each completed training step. The harness calls the regular trainer entry point and leaves the data loader and optimizer calls in place.

## Reproduce the trace

Start in the directory that contains this file. Set `BENCH_CORRECTNESS` to the path of the fixture and trace scripts, then check out the upstream commit. Use a LeRobot environment with the dataset and training extras and an FFmpeg build with `libx264`. The recorded run used macOS arm64, Python 3.13.9, PyTorch 2.11.0, Datasets 4.8.5, Accelerate 1.15.0, and TorchCodec 0.11.1 with FFmpeg 8 libraries.

```bash
BENCH_CORRECTNESS="$(cd ../.. && pwd)"
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout --detach 8ec1a474da5f3f90a670d9343386995ec9f81264
uv sync --locked --extra dataset --extra training

run_dir="$(mktemp -d)"
uv run --no-sync python "$BENCH_CORRECTNESS/fixture.py" "$run_dir/fixture" --short
export STREAM5388_TRAIN_TRACE="$run_dir/batches.jsonl"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv run --no-sync python "$BENCH_CORRECTNESS/train_trace.py" \
    --dataset.repo_id=fixture/stream5388 --dataset.root="$run_dir/fixture" \
    --dataset.streaming=true --policy.type=act --policy.device=cpu \
    --policy.pretrained_backbone_weights=null --policy.push_to_hub=false \
    --policy.dim_model=32 --policy.n_heads=4 --policy.dim_feedforward=64 \
    --policy.n_encoder_layers=1 --policy.n_decoder_layers=1 --policy.n_vae_encoder_layers=1 \
    --policy.latent_dim=4 --policy.chunk_size=100 --policy.n_action_steps=100 \
    --batch_size=4 --prefetch_factor=3 --trackio.enable=false --steps=23 \
    --log_freq=1 --seed=1000 --env_eval_freq=0 --num_workers=4 \
    --save_checkpoint=false --output_dir="$run_dir/train"

uv run --no-sync python - "$run_dir/batches.jsonl" <<'PY'
from collections import Counter
import json
import sys

with open(sys.argv[1]) as trace:
    indices = [index for line in trace for index in json.loads(line)]
prefix = Counter(indices[:86])
print(f"Unique frames = {len(prefix)}")
print(f"Repeat visits = {86 - len(prefix)}")
print(f"Frames that have yet to appear = {sorted(set(range(86)) - set(prefix))}")
PY
```
