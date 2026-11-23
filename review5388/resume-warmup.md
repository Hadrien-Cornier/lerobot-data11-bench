# Shuffle-buffer restore on the Datasets dependency of #5388

Quentin's [prototype](https://github.com/lhoestq/datasets/commit/cb0d9d14a4add5db04c10b83736cc9d08d55df51) applies cleanly to the source files of the [block-cache branch](https://github.com/huggingface/datasets/pull/8983). The branches have diverged. Neither pin alone supplies both implementations.

We generated 2,000 episodes of four unique frames each. We saved the Datasets iterator state after consuming either 1,001 or 4,096 IDs. We compared the uninterrupted suffix with a fresh iterator that loaded that state. The grouped case batches by episode, shuffles with a 32-episode buffer, then uses the reviewed draft’s frame-mixing rule inside 32-episode pools. These are in-process Datasets iterators; the test excludes PyTorch workers, prefetch queues, video and model training.

| Implementation | Consumed IDs | Missing IDs in complete pass | Duplicate deliveries | Matching positions in suffix |
|---|---:|---:|---:|---:|
| Block-cache branch | 1,001 | 93 | 33 | 15 of 6,999 |
| With warmup patch | 1,001 | 0 | 0 | 6,999 of 6,999 |
| Block-cache branch | 4,096 | 88 | 24 | 9 of 3,904 |
| With warmup patch | 4,096 | 44 | 52 | 9 of 3,904 |

The prototype reduced missing IDs in both examples. At the later checkpoint its delivered sequence had both omissions and repetitions. This does not measure the percentage of internal shuffle-buffer slots reconstructed. LeRobot's training entry point needs to save the state that corresponds to batches consumed on each rank for such a feature to work end to end.

## Reproduce

Use a Python environment with the [benchmark dependencies](runner/environment.sh). From the benchmark's `review5388` directory, set up two source trees and run the same script against each.

```bash
git clone https://github.com/huggingface/datasets.git /tmp/streaming-datasets
git -C /tmp/streaming-datasets fetch origin pull/8983/head
git -C /tmp/streaming-datasets checkout --detach 464fe790b82a320350c9c686c1b7e409323a1cf3
git -C /tmp/streaming-datasets worktree add --detach /tmp/streaming-warmup HEAD
curl -fsSL https://github.com/lhoestq/datasets/commit/cb0d9d14a4add5db04c10b83736cc9d08d55df51.patch -o /tmp/streaming-warmup.patch
git -C /tmp/streaming-warmup apply --include='src/*' /tmp/streaming-warmup.patch
PYTHONPATH=/tmp/streaming-datasets/src python correctness/compare_resume_warmup.py
PYTHONPATH=/tmp/streaming-warmup/src python correctness/compare_resume_warmup.py
```

[Control results](measurements/resume-warmup-control.json) · [Patched results](measurements/resume-warmup-prototype.json)
