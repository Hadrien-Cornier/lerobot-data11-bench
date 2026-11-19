# MolmoAct2 current-frame crossover with a matching MP4 index

Quentin delivered the first full batch sooner in all three fresh-disk pairs. With the matching MP4 index, Pepijn took the lead between batches 374 and 940 and held it through each shared trace. One pair reached 4,000 batches on both readers. In two pairs, Quentin reached its time cap first.

## What the timer includes

One sample is an `(episode_index, frame_index)` anchor. It has three current camera frames, one state, and one action. Both readers used batch size 8 and a 1 ms timestamp tolerance. We requested seed 42 and set the episode pool to 32. A single CPU consumer took batches directly with zero DataLoader workers. The readers used their own sample order and decode concurrency. [Reader revisions and benchmark settings](method.md)

Each pair used the same 16-vCPU host. Each reader ran in a new process with a separate, empty application cache. The runner staged the matching 1.83 GB MP4 index for Pepijn before either reader started. Pepijn's measured time includes the receipt in every pair, including when Quentin ran first. It also includes construction, local memory map preparation, and delivery. The new repeats share one host. Upstream caches can persist between measurements.

The source inventory matched before index receipt and before and after each paired run. The bucket can change between inventory checks. The audit checks the transfer receipt against the pinned archive, the publisher audit, and the index in use. It verifies batch IDs against episode lengths and checks each stream for duplicates. The 1 ms current-frame fixture also checks all 786 expected anchors and sample values for both readers. [Fixture and sample checks](correctness/README.md) · [Benchmark commands](method.md#repeat-the-fresh-disk-comparison-with-a-matching-publisher-index)

We compare cumulative time at equal numbers of full batches. A lead change appears at the first observed batch where the other reader is faster. The audit retains the recorded deliveries on both sides of each change. A lead holds when it survives through the last batch both readers delivered. The same number of anchors can come from different episodes in each reader.

## Time to equal batch counts

The times in each cell are Pepijn / Quentin, in seconds. Pepijn's times include the receipt. The last shared batch is the latest observed point at which both readers delivered the same number of distinct anchors.

| Pair and reader order | Receipt | Batch 1 | Batch 1,000 | Last shared batch | Time at last shared batch | First lead change |
|---|---:|---:|---:|---:|---:|---|
| 1. Pepijn, then Quentin | 10.50 | 153.10 / 38.56 | 229.54 / 252.33 | 3,552 | 418.64 / 680.06 | Pepijn at 762 |
| 2. Quentin, then Pepijn | 10.42 | 118.14 / 37.01 | 181.70 / 244.02 | 3,978 | 366.11 / 680.08 | Pepijn at 374 |
| 3. Pepijn, then Quentin | 9.88 | 123.94 / 29.72 | 187.45 / 190.75 | 4,000 | 375.38 / 538.89 | Pepijn at 940 |

Quentin stopped at its 680-second consumer time cap in pairs 1 and 2. Pepijn reached 4,000 batches in each pair. Its first two runs took 451.73 and 367.46 seconds. The shared traces reached 28,416, 31,824, and 32,000 anchors per reader, or 0.0374%, 0.0418%, and 0.0421% of the source. Each reader delivered its first 8,000 anchors in the same order across all three pairs. [Pair summary CSV](measurements/molmo-prepared-repeats.pairs.csv) · [Delivery checkpoints as text](measurements/molmo-prepared-repeats.points.txt) · [Trace audit](measurements/molmo-prepared-repeats.json)

The earlier warm comparison reused both readers' local caches on its host and stopped at batch 1,000. Pepijn took the lead at batch 514 and reached batch 1,000 in 75.21 seconds, versus 91.14 seconds for Quentin. The original fresh-index pair also stopped at batch 1,000. Pepijn took the lead at batch 702 and reached batch 1,000 in 199.72 seconds, versus 224.34 seconds for Quentin. [Warm trace audit](measurements/warm-current-frame.json) · [Original fresh-index audit](measurements/prepared-current-frame.json)

As a check on episode mix, we group batches by the number of episodes, the number of samples from the largest episode, and the number of same-episode pairs within 0.2 seconds. We compare the mean time waiting for the next batch over the overlap. This wait excludes reader construction and index receipt. The delivery rates use the whole shared interval after batch 125.

| Pair | Batch interval | Episode mix overlap | Pepijn mean wait | Quentin mean wait | Anchors/s, Pepijn / Quentin |
|---|---:|---:|---:|---:|---:|
| 1 | 126–3,552 | 97.2% | 73.4 ms | 158.0 ms | 107.54 / 47.13 |
| 2 | 126–3,978 | 96.8% | 61.4 ms | 145.0 ms | 128.75 / 52.71 |
| 3 | 126–4,000 | 96.8% | 62.1 ms | 111.8 ms | 127.45 / 67.27 |

## What one full pass would take

The verified source contains **76,046,658 anchors in 32,246 episodes**. At batch size 8, one finite pass gives 9,505,832 full batches and a final batch of two samples if the loader retains that batch. A 1,000-batch trace covers 8,000 anchors or **0.0105%** of the pass. A 4,000-batch trace covers 32,000 anchors or **0.0421%**. The fractions apply to each reader's own distinct anchors.

The fresh-index and warm traces completed zero verified pool refills. The cold runs give longer measurements for the same current-frame workload. For the estimates, the rate window begins after batch 125 and includes two verified pool refills and at least 125 later batches. A refill counts when the reader emits all anchors from the prior cohort and at least one anchor from every span in the next cohort.

| Cold reader and run | Full batches delivered | Fraction of pass | Refill batches | Anchors timed after batch 125 | Time across window | Anchors/s | Estimated pass |
|---|---:|---:|---|---:|---:|---:|---:|
| Pepijn, first pair | 29,288 | 0.308% | 18,557 and 29,163 | 233,304 | 1,092.25 s | 213.60 | 4.13 days |
| Quentin, first pair | 13,300 | 0.140% | 6,878 and 13,175 | 105,400 | 959.16 s | 109.89 | 8.01 days |
| Pepijn, cold repeat | 25,000 | 0.263% | 18,557 | 199,000 | 1,131.12 s | 175.93 | One refill |
| Quentin, cold repeat, saved before deadline | 24,000 | 0.252% | 6,878 and 13,175 | 191,000 | 2,242.51 s | 85.17 | 10.33 days |

Each estimate charges the observed time to batch 125 and divides the remaining 76,045,658 anchors by the measured rate. It assumes the rate continues through the rest of the source. The estimate for Pepijn includes the local index build in the cold startup. These are CPU input times for one consumer. Using just the first 125 batches after the second refill gives 4.50 days for Pepijn and 16.29 days for Quentin in the first pair. It gives 17.25 days for Quentin in the deadline-limited repeat. Those short windows show the sensitivity to the part of the stream used to estimate the rate. [Cold repeat checkpoint and deadline evidence](measurements/molmo-tol02-checkpoint-crossover.json)

The finite pass describes a distinct visit to each source anchor in one single-consumer iteration. The normal baseline trainer repeats each worker's shard independently. In the small four-worker training fixture, the first 86 consumed samples contain 79 distinct anchors. An epoch defined by step count in that mode can repeat some anchors before it reaches others. [Finite fixture and repeating trainer trace](correctness/README.md)

[Time for every paired batch (compressed CSV)](measurements/molmo-prepared-repeats.curves.csv.gz) · [Lead changes and milestones (plain text)](measurements/molmo-prepared-repeats.points.txt)

## Reproduce the audit

Use the collected publisher and consumer job folders. Build `results/catalog-bucket.json` with `catalog.py`. Run `coverage_audit.py` with `--metadata-cache results/episode-lengths` to cache the episode lengths. The source manifest must match the timed job. The command below writes a JSON audit, one CSV row per pair, observed checkpoints, and a CSV with every paired delivery. [Run and collect the jobs](method.md#repeat-the-fresh-disk-comparison-with-a-matching-publisher-index) · [Crossover and full-pass audit](prepared_repeat_audit.py)

```bash
digest=$(python3 -c 'import json; print(json.load(open("results/catalog-bucket.json"))["manifest_sha256"])')
python3 prepared_repeat_audit.py --repeat-only --results results \
    --repeat-dir results/downloads/review5388/results/molmo-consumers \
    --episode-index "results/episode-lengths/bucket-$digest.json" \
    --output-prefix results/molmo-consumer-crossover
```
