# Decoder creation and frame retrieval

On a fixed local request sequence, the draft creates a TorchCodec decoder for every camera request. The baseline reuses one per episode and camera. In the current-frame run, decoder creation accounts for 1.085 seconds of the 1.377-second difference in combined time inside the two TorchCodec calls. Keeping a decoder and its file handle per source MP4 cuts the draft's inclusive decode helper from 2.523 to 1.221 seconds in the same local test.

| Reader and request | `VideoDecoder` calls | `VideoDecoder` wall s | `get_frames_at` calls | `get_frames_at` wall s | Inclusive RGB decode wall s | Record |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| #3917, current frame | 48 | 0.038 | 2,358 | 0.791 | 0.861 | [JSON](measurements/decoder-stages-baseline.json) |
| #5388, current frame | 2,358 | 1.123 | 2,358 | 1.082 | 2.523 | [JSON](measurements/decoder-stages-candidate.json) |
| #5388, current frame with decoder reuse | 24 | 0.033 | 2,358 | 0.956 | 1.221 | [JSON](measurements/decoder-stages-candidate-reuse.json) |
| #3917, five camera frames | 48 | 0.023 | 2,358 | 2.617 | 2.690 | [JSON](measurements/decoder-stages-baseline-history.json) |
| #5388, five camera frames | 2,358 | 1.012 | 2,358 | 2.699 | 4.069 | [JSON](measurements/decoder-stages-candidate-history.json) |

Each arm sends the same shuffled 786 anchors to three cameras on a local H.264 fixture with 16 episodes and 24 source MP4s at 64 × 48 and 25 fps. All outputs match the FFmpeg fixture oracle pixel for pixel. The history run sends five timestamps per camera and clamps requests to the episode start in the harness. The harness calls the camera query methods directly, so its history result applies to frame retrieval. [Fixture generator](correctness/fixture.py) · [Request harness](decoder_stages.py) · [Reader correctness checks](correctness/README.md)

The experiment uses one serialized request loop on an Apple M1 Max with TorchCodec 0.17.0 and PyTorch 2.15.0. It reads the local source files before each arm and stages the baseline's synthesized episode payloads before the timed requests. The baseline opens decoders lazily as requests arrive. The draft opens local files through its usual `fsspec` path. The reuse control keeps that per-request file-open call and uses a persistent input handle for each retained decoder. The control has exclusive use of its decoders within the request loop. A concurrent cache would need to manage file-handle lifetime and serialize each decoder's use. [Baseline cache](https://github.com/huggingface/lerobot/blob/fe1c54594f6ba275fccf210b9f7b11f0e2324b89/src/lerobot/streaming/episode_cache.py#L205-L250) · [Draft helper](https://github.com/huggingface/lerobot/blob/1a60cb4b800c2947834342752c32f064c8bf143c/src/lerobot/datasets/video_utils.py#L259-L304)

The two inner timings include work done inside each API call. In the draft, the file-like object's reads can run inside both calls. The draft's inclusive helper starts after the outer `fsspec` file open. The baseline's inclusive request covers decoder lookup and potential wait for payload bytes or the decoder lock. Add only the two inner calls when comparing the combined TorchCodec API time. The raw results also include calling-thread CPU time and the full local request-loop time. Cloud prefetch and decode threads can overlap. Existing cloud results captured the inclusive decode timings and file-open counts, so the local split has its own evidence. [Instrumentation](telemetry.py) · [Cloud measurement method](method.md)

## Reproduce the fixture measurement

Set up `WORK` with `runner/environment.sh` as described in the [cloud measurement method](method.md). Run the following from `review5388/`. The recorded timings use the macOS environment in the JSON files. The same commands with the provided Linux setup reproduce the request sequence and counters.

```bash
fixture="$WORK/decoder-fixture"
"$WORK/venv/bin/python" correctness/fixture.py "$fixture"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for arm in baseline candidate; do
    for workload in current history; do
        flags=()
        if [[ "$workload" == history ]]; then flags+=(--history); fi
        PYTHONPATH="$WORK/src/$arm/src" \
            "$WORK/venv/bin/python" decoder_stages.py --reader "$arm" --fixture "$fixture" \
            --output "$WORK/decoder-$arm-$workload.json" "${flags[@]}"
    done
done
PYTHONPATH="$WORK/src/candidate/src" \
    "$WORK/venv/bin/python" decoder_stages.py --reader candidate --fixture "$fixture" \
    --candidate-reuse-control --output "$WORK/decoder-candidate-reuse.json"
```
