#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
bench="$(pwd)"
scratch="${1:-/tmp/lerobot5388-correct}"
bench_venv="${STREAM5388_VENV:-$scratch/venv}"
python="$bench_venv/bin/python"
export PATH="$bench_venv/bin:$PATH"
export HF_HOME="$scratch/hf-cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
if [[ -d /opt/homebrew/opt/ffmpeg@8/lib ]]; then
    export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@8/lib:${DYLD_LIBRARY_PATH:-}"
fi
root="$scratch/fixture"
if [[ ! -d "$root" ]]; then
    "$python" fixture.py "$root"
fi

source_env() {
    case "$1" in
        baseline) export PYTHONPATH="$scratch/base/src${STREAM5388_EXTRA_IMPORTS:+:$STREAM5388_EXTRA_IMPORTS}";;
        proposal) export PYTHONPATH="$scratch/proposal/src:$scratch/datasets/src${STREAM5388_EXTRA_IMPORTS:+:$STREAM5388_EXTRA_IMPORTS}";;
    esac
}

check() {
    local variant="$1"
    local name="$2"
    shift 2
    source_env "$variant"
    "$python" check_stream.py "$variant" "$root" "evidence/$name.json" "$@" >"$scratch/$name.log" 2>&1 || true
}

train_flags=(
    --dataset.repo_id=fixture/stream5388 "--dataset.root=$root" --dataset.streaming=true
    --policy.type=act --policy.device=cpu --policy.pretrained_backbone_weights=null
    --policy.push_to_hub=false --policy.dim_model=32 --policy.n_heads=4 --policy.dim_feedforward=64
    --policy.n_encoder_layers=1 --policy.n_decoder_layers=1 --policy.n_vae_encoder_layers=1
    --policy.latent_dim=4 --policy.chunk_size=100 --policy.n_action_steps=100
    --batch_size=4 --prefetch_factor=3 --trackio.enable=false
    --steps=8 --log_freq=1 --save_freq=3 --seed=1000 --env_eval_freq=0
)

train() {
    local variant="$1"
    local workers="$2"
    local out="$scratch/train-$variant-w$workers"
    source_env "$variant"
    if [[ "$variant" == "proposal" ]]; then
        # Run the untouched command. The error should precede any tracing hooks.
        lerobot-train "${train_flags[@]}" "--num_workers=$workers" "--output_dir=$out"             >"evidence/train-$variant-w$workers.log" 2>&1
        return
    fi
    STREAM5388_TRAIN_TRACE="$bench/evidence/train-$variant-w$workers-uninterrupted.jsonl"     "$python" train_trace.py "${train_flags[@]}" "--num_workers=$workers" "--output_dir=$out"         >"evidence/train-$variant-w$workers.log" 2>&1
    STREAM5388_TRAIN_TRACE="$bench/evidence/train-$variant-w$workers-resumed.jsonl"     "$python" train_trace.py --resume=true         "--config_path=$out/checkpoints/000003/pretrained_model/train_config.json" "--output_dir=$out-resumed"         >"evidence/train-$variant-w$workers-resumed.log" 2>&1
}

train_ranks() {
    local variant="$1"
    local workers="$2"
    local out="$scratch/train-$variant-w$workers-ranks"
    local stage="${3:-uninterrupted}"
    local port
    port="$("$python" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
    local launch=(torchrun --nnodes=1 --nproc-per-node=2 --rdzv-backend=static
        --rdzv-conf=timeout=20 --master-addr=127.0.0.1 "--master-port=$port")
    if [[ "$(uname -s)" == "Darwin" ]]; then
        export GLOO_SOCKET_IFNAME=lo0
    fi
    source_env "$variant"
    if [[ "$variant" == "proposal" ]]; then
        "${launch[@]}" --no-python "$(command -v lerobot-train)" \
            "${train_flags[@]}" "--num_workers=$workers" "--output_dir=$out" \
            >"evidence/train-$variant-w$workers-ranks.log" 2>&1
        return
    fi
    if [[ "$stage" == "resumed" ]]; then
        STREAM5388_TRAIN_TRACE="$bench/evidence/train-$variant-w$workers-ranks-resumed.jsonl" \
        "${launch[@]}" train_trace.py --resume=true \
            "--config_path=$out/checkpoints/000003/pretrained_model/train_config.json" "--output_dir=$out-resumed" \
            >"evidence/train-$variant-w$workers-ranks-resumed.log" 2>&1
        return
    fi
    STREAM5388_TRAIN_TRACE="$bench/evidence/train-$variant-w$workers-ranks-uninterrupted.jsonl" \
    "${launch[@]}" train_trace.py "${train_flags[@]}" "--num_workers=$workers" "--output_dir=$out" \
        >"evidence/train-$variant-w$workers-ranks.log" 2>&1
}

case "${2:-finite}" in
    finite)
        check baseline baseline-w1 --workers 1
        check baseline baseline-w4 --workers 4
        check baseline baseline-w4-selection --workers 4 --selection
        check proposal proposal-w1-normal-root --workers 1
        for mode in full numeric positive video anchors; do
            check proposal "proposal-w1-$mode" --local-id --workers 1 --mode "$mode"
        done
        check proposal proposal-video-probes --local-id --workers 0 --mode video --probe-video
        check proposal proposal-w4-full --local-id --workers 4
        check proposal proposal-w4-anchors --local-id --workers 4 --mode anchors
        check proposal proposal-w4-selection --local-id --workers 4 --mode anchors --selection
        ;;
    ranks)
        for workers in 1 4; do
            for rank in 0 1; do
                check baseline "baseline-w$workers-rank$rank" --workers "$workers" --rank "$rank" --world 2
                check proposal "proposal-w$workers-rank$rank" --local-id --mode anchors                     --workers "$workers" --rank "$rank" --world 2
            done
        done
        ;;
    restart)
        for workers in 0 1 4; do
            check baseline "baseline-w$workers-restart" --workers "$workers" --restart
            check proposal "proposal-w$workers-restart" --local-id --workers "$workers" --mode anchors --restart
        done
        ;;
    repeat)
        check baseline baseline-w1-repeat --workers 1 --mode anchors --repeat-coverage
        check baseline baseline-w4-repeat --workers 4 --mode anchors --repeat-coverage
        ;;
    shuffle)
        root="$scratch/fixture-short"
        if [[ ! -d "$root" ]]; then
            "$python" fixture.py "$root" --short
        fi
        check baseline baseline-shuffle --workers 0 --mode anchors --check-shuffle
        check proposal proposal-shuffle --workers 0 --mode anchors --local-id --check-shuffle
        ;;
    train-repeat)
        short_root="$scratch/fixture-short"
        if [[ ! -d "$short_root" ]]; then
            "$python" fixture.py "$short_root" --short
        fi
        source_env baseline
        STREAM5388_TRAIN_TRACE="$bench/evidence/train-baseline-w4-repeat.jsonl" \
        "$python" train_trace.py "${train_flags[@]}" "--dataset.root=$short_root" \
            --num_workers=4 --steps=23 --save_checkpoint=false "--output_dir=$scratch/train-baseline-w4-repeat" \
            >"evidence/train-baseline-w4-repeat.log" 2>&1
        ;;
    train-ranks-resume)
        train_ranks baseline "${3:?Specify the worker count}" resumed
        ;;
    train-ranks)
        for workers in 1 4; do
            train_ranks proposal "$workers" || true
            train_ranks baseline "$workers" || true
        done
        ;;
    train)
        for workers in 1 4; do
            train proposal "$workers" || true
            train baseline "$workers"
        done
        ;;
esac
