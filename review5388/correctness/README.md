# Streaming correctness and training checks for LeRobot #5388

The proposal fails in `lerobot-train` before it reads a sample. The baseline passes the finite sample oracle and the checkpoint resume checks. In repeating training, four baseline workers can repeat samples before the trainer consumes every sample once.

## Source revisions

- Baseline LeRobot `fe1c54594f6ba275fccf210b9f7b11f0e2324b89`.
- Proposal LeRobot `1a60cb4b800c2947834342752c32f064c8bf143c` from [LeRobot #5388](https://github.com/huggingface/lerobot/pull/5388).
- Proposal dependency `464fe790b82a320350c9c686c1b7e409323a1cf3` from [Datasets #8983](https://github.com/huggingface/datasets/pull/8983).

## Fixture and oracle

The full fixture has 786 frames in 16 episodes and eight Parquet files. Consecutive episodes share each file. Parquet row groups have at most 47 rows. The three H.264 cameras use distinct images and camera-specific video offsets.

The oracle computes state and action vectors from the sample ID. It checks 100 action steps, three state steps, three frames per camera and their padding masks. It compares the camera tensors with complete FFmpeg decodes of the fixture files. The RGB check allows at most four intensity levels per channel for decoder rounding. The baseline's maximum observed difference is zero.

The result files include the sample IDs in stream order, error counts and the first example of each error. The two-rank direct checks run each rank separately with `RANK` and `WORLD_SIZE`. The trainer checks launch actual processes with `torchrun` and use the trainer's normal data path.

## Finite streams

| Case | Baseline | Proposal |
| --- | --- | --- |
| Full fixture, 1 worker | 786 unique IDs. All fields match. | Temporal window assembly fails before the first sample. |
| Full fixture, 4 workers | 786 unique IDs. All fields match. | The temporal request fails at the same numeric window assembly. |
| Six selected episodes, 4 workers | 374 unique IDs. All fields match. | The anchors-only run returns all 786 IDs, including 412 outside the selection. |
| Two direct ranks, 1 worker per rank | 396 and 390 IDs. Union 786. Overlap zero. | Each reader returns all 786 IDs with temporal requests disabled. |
| Two direct ranks, 4 workers per rank | 396 and 390 IDs. Union 786. Overlap zero. | Each reader returns all 786 IDs with temporal requests disabled. |

With a normal `repo_id` and local `root`, the proposal attempts to load the Parquet data from the Hub. The direct proposal cases set `repo_id` to the absolute fixture path after this failure. This argument change points Datasets at the local files without source edits.

With that argument change, the proposal can decode the current camera frame. A full temporal request raises `TypeError: expected Tensor as element 0 in argument 0, but got dict`. A positive-only action request raises `KeyError: 'action_is_pad'`.

With camera deltas only, the proposal returns 112 samples before an out-of-bounds frame access. On those samples, each wrist camera reports 112 incorrect padding masks and each camera returns 12 incorrect windows. The frame query can use the exclusive episode stop time, and the comparison for padding uses timestamps with different origins. See [camera stream evidence](evidence/proposal-w1-video.json).

The direct-rank overlap measures the reader's native partition behavior. The proposal trainer raises an exception before it reaches Accelerate's DataLoader dispatch.

## ABC-style camera history

The `abc-history` fixture requests the current state, action steps 0 through 99, and five frames per camera at −0.8, −0.6, −0.4, −0.2 and 0 seconds. The camera offsets correspond to −20, −15, −10, −5 and 0 frames on the 25 fps fixture and −24, −18, −12, −6 and 0 frames on the 30 fps ABC source. The oracle checks the temporal values and padding masks at every anchor. It compares every camera frame with an independent full-file FFmpeg decode.

The baseline passed all 786 anchors with one and four workers. Each anchor appeared once, and the maximum RGB difference was zero. The proposal raised a `TypeError` in action-window assembly before the first anchor at both worker counts. These runs used Python 3.12.12, PyTorch 2.15.0 and TorchCodec 0.17.0. The baseline used Datasets 5.2.0 and the draft used the pinned Datasets pull request. [One-worker baseline](evidence/baseline-w1-abc-history.json) · [Four-worker baseline](evidence/baseline-w4-abc-history.json) · [One-worker draft](evidence/proposal-w1-abc-history.json) · [Four-worker draft](evidence/proposal-w4-abc-history.json)

## Shuffle controls

On the 86-frame fixture without delta timestamps, the baseline produces different first orders for seeds 42 and 43. With `shuffle=True`, iterating again on the same dataset changes the order. With `shuffle=False`, it repeats the first order. The proposal produces the same order for both seeds, both shuffle settings and both iterations. See the [baseline orders](evidence/baseline-shuffle.json) and [proposal orders](evidence/proposal-shuffle.json).

## Training and resume

The untouched proposal command fails with `TypeError: StreamingLeRobotDataset.__init__() got an unexpected keyword argument 'max_num_shards'`. This occurs while creating the dataset, before the trainer builds a DataLoader. See the [one-worker proposal log](evidence/train-proposal-w1.log) and [four-worker proposal log](evidence/train-proposal-w4.log).

The baseline trains a small CPU Action Chunking Transformer (ACT) for eight optimizer updates with one and four workers. The trace wraps the normal entry point. It records raw sample IDs only after the corresponding optimizer update succeeds. Both runs save a checkpoint after step three. Five resumed batches match the uninterrupted steps four through eight. They start at 12 consumed samples, even with three DataLoader batches of prefetch per worker.

The baseline also completes eight updates on each of two CPU ranks with one and four workers per rank. Each rank consumes 32 unique IDs, with no overlap between ranks. On every tested rank, the five resumed batches match the uninterrupted trace. The proposal fails with the same constructor error at both worker counts under `torchrun`.

The baseline direct state API reports offset 5 after exactly five yields with zero DataLoader workers. The test also compares a consumed offset of 20 samples against uninterrupted DataLoader order with one and four workers. Both pass. The proposal reader exposes neither `state_dict` nor `load_state_dict`.

## Repeating training

The normal trainer enables repetition. In a separate 86-frame fixture, the baseline completes 23 CPU ACT updates with a batch size of four and four workers. In the first 86 consumed samples, 79 IDs are unique. ID 5 repeats at position 72, in update 18. The other 7 expected IDs do not appear in that window. See the [training trace](evidence/train-baseline-w4-repeat.jsonl) and [training log](evidence/train-baseline-w4-repeat.log).

The full 786-frame fixture also shows 12 early repeated IDs with four workers. With one worker, its first 786 samples are unique. Finite iteration gives one visit to each ID with both worker counts. A repeating worker begins its next coverage epoch independently when its own shard ends.

## Reproduce

Create scratch worktrees named `base`, `proposal`, and `datasets` at the revisions above. Place them in the same scratch directory as the Python environment `venv`. Install the baseline `training` extra. Keep a `lerobot-train` console script in that environment. The runner sets the source import paths and generates the fixture with FFmpeg.

```bash
scratch=/tmp/lerobot5388-correct
uv venv --python 3.12 "$scratch/venv"
uv pip install --python "$scratch/venv/bin/python" -e "$scratch/base[training]" \
  'torch==2.15.0' 'torchvision==0.29.2' 'torchcodec==0.17.0' 'datasets==5.2.0'
for group in finite ranks restart repeat shuffle train train-repeat train-ranks; do
  ./run_cases.sh "$scratch" "$group"
done
./run_cases.sh "$scratch" abc-history
./run_cases.sh "$scratch" train-ranks-resume 1
./run_cases.sh "$scratch" train-ranks-resume 4
```

The finite checks and single-rank checkpoint traces used Apple silicon with Python 3.13.9, PyTorch 2.11.0, TorchCodec 0.11.1, Accelerate 1.15.0, Hub 1.33.0 and FFmpeg 8.1.2. The baseline in those runs used Datasets 4.8.5. The proposal used the Datasets pull request revision. The short repeat trainer and the distributed trainer used Python 3.12.12, PyTorch 2.15.0, TorchCodec 0.17.0, Accelerate 1.16.0, Hub 2.4.0 and Homebrew FFmpeg 9.0.2. The shuffle experiment also used this environment. The distributed and short-run baseline used Datasets 5.2.0. These checks measure data correctness and training integration for the local fixture.

## Packed video timestamp precision

The draft's no-delta Molmo run at native 0.1 ms tolerance yielded its first sample, then encountered a timestamp error before delivering a complete batch. Controlled real-decoder probes at 1,830.033333 and 4,556.133333 seconds show that rounding the absolute time to float32 exceeds the 0.1 ms tolerance. A 1 ms tolerance returned the intended frames, and both pinned readers passed the complete 786-frame current-image fixture at that setting. [Baseline probe](probe_tolerance.baseline.json) · [Draft probe](probe_tolerance.candidate.json)

The public map-style and streaming readers on main `8ec1a474` can read short episodes at both of these late offsets in a shared MP4 at 0.1 ms. The map-style path adds the local time to the file offset at higher precision. As a separate positive control, a single 30 fps episode whose *local* timestamp reached 2,048.033333 seconds failed at its native tolerance. An existing upstream change addresses that distinct case. [Packed-offset script and measurements](packed-offset) · [Existing PR #4882](https://github.com/huggingface/lerobot/pull/4882)
