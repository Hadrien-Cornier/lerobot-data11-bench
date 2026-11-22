# What does a larger video block buy?

On one camera MP4 from each dataset, switching from 1 MiB to 4 MiB reduced the number of remote range fetches and increased the bytes fetched. Both configurations returned the same RGB frames, in the same order. In both pairs per dataset the 4 MiB runs finished the 64 requested images sooner.

| Dataset and camera | Block | Range fetches | Data fetched | Time for 64 RGB images, two runs |
|---|---:|---:|---:|---:|
| MolmoAct2 right | 1 MiB | 5 | 5 MiB | 2.34–2.53 s |
| MolmoAct2 right | 4 MiB | 3 | 12 MiB | 2.27–2.32 s |
| ABC-130K top | 1 MiB | 5 | 6 MiB | 1.94–2.34 s |
| ABC-130K top | 4 MiB | 2 | 8 MiB | 1.70–1.93 s |

On ABC-130K, the first eight images took 0.75–1.13 s at 1 MiB and 1.16–1.28 s at 4 MiB. The larger blocks finished all 64 images sooner in each paired repeat.

The shared cache can combine consecutive missing blocks into one range fetch. Its range counter and byte counter measure the payload delivered to the cache. Redirects, retries, and lower-level transport requests can add wire traffic.

## Workload

The benchmark called the patched reader’s native grouped RGB method and decoder pool on one real video per dataset. It issued eight groups of eight timestamps in a fixed order. Some seeks were nearby and some returned to an earlier portion of the file. The first requested frame was 0.2 seconds into the file. The latest was about 48.3 seconds in. Groups were chosen for this test and do not represent draws from the episode sampler.

Each arm had a new Python process and empty application block cache. The Mac ran the experiment locally. Each dataset ran at 1 then 4 MiB, followed by a second pair at 4 then 1 MiB. Each arm checked the object hash before and after reading and recorded a digest over ordered frame IDs and RGB bytes. The timer covers the calls that open, fetch and decode video. It excludes startup of the Python process, dataset construction, hashing and teardown. Upstream caches can persist across arms. The result describes these two video traces and does not measure three-camera training throughput or establish the best default block size.

## Full-reader smoke test

A separate local smoke test tried eight batches of eight examples per dataset with three cameras, the current state and action, a four-episode pool, and zero DataLoader workers. Each default 1 MiB arm had an 85-second wall-clock limit including imports and dataset construction. Neither dataset completed the eight-batch test within that limit. The 4 MiB full-reader arms were skipped. There is no measured end-to-end block-size comparison here. The decoder test above completed all eight arms. No paid jobs were used.

## Reproduce

Use the [benchmark environment setup](runner/environment.sh) for the pinned Datasets commit `464fe790` and TorchCodec stack. Make the patched LeRobot source at `adbdf5f0218898b187b4d20376a4bdee4a97d388` available on `PYTHONPATH`. The benchmark verifies the file hashes of both implementations. From `review5388/`:

```bash
git clone --branch perf/5388-grouped-rgb https://github.com/Hadrien-Cornier/lerobot.git /tmp/block-size-lerobot
git -C /tmp/block-size-lerobot checkout --detach adbdf5f0218898b187b4d20376a4bdee4a97d388
PYTHONPATH="/tmp/block-size-lerobot/src" "$WORK/venv/bin/python" reader_block_size.py --out block-size-results.json
```

[Exact source objects, per-arm timings, cache counters, output hashes and software versions](measurements/reader-block-size.json) · [Benchmark script](reader_block_size.py)
