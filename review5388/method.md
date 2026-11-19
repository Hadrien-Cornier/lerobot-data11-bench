# Comparing the LeRobot streaming readers

The unit of work is one training example at an identified control step. Three camera views and 100 target actions still count as one example. Its identity is the source snapshot, episode index, and frame index. Eight examples make one batch.

## Readers and input data

The baseline is the merge base of [LeRobot #5388](https://github.com/huggingface/lerobot/pull/5388), `fe1c54594f6ba275fccf210b9f7b11f0e2324b89`. It includes the merged seed, worker, resume, rank, and mapped-index changes after [#3917](https://github.com/huggingface/lerobot/pull/3917). The draft at `1a60cb4b800c2947834342752c32f064c8bf143c` requires [Datasets #8983](https://github.com/huggingface/datasets/pull/8983) at `464fe790b82a320350c9c686c1b7e409323a1cf3`. The paired cloud runs use that Datasets revision on both sides.

The draft groups Parquet rows by episode, shuffles a buffer of episodes, and maps video decoding over the sampled rows. Its block cache reuses remote byte ranges across threads in a process. The baseline builds or loads a per-MP4 seek index, schedules samples from an episode pool, and fetches video ranges using that index. The diff includes 5,414 deleted source lines and 5,893 deleted test lines. It removes explicit streaming resume and distributed sampling code from the trainer. [Proposed reader](https://github.com/huggingface/lerobot/blob/1a60cb4b800c2947834342752c32f064c8bf143c/src/lerobot/datasets/streaming_dataset.py#L227-L259) · [Full diff](https://github.com/huggingface/lerobot/pull/5388/files)

The inventory counts physical MP4s in the source tree. We also read the video references from every row of every episode metadata table. All 5,860 Molmo files and 38,164 ABC files are referenced. DROID has 813 physical MP4s and its 95,658 episodes reference 784. All references resolved. Video bytes below are compressed storage bytes across the physical files.

| Source | Episodes | Control steps | MP4s | Video bytes, decimal TB |
|---|---:|---:|---:|---:|
| [MolmoAct2 bucket](https://huggingface.co/buckets/pepijn223/MolmoAct2-BimanualYAM-Dataset-bucket) | 32,246 | 76,046,658 | 5,860 | 2.343 |
| [ABC-130K](https://huggingface.co/datasets/lerobot/abc_130k_v3_train/tree/68651e4929d9fb00f798937b2d62617cab5c771d) | 129,225 | 382,468,339 | 38,164 | 3.378 |
| [DROID](https://huggingface.co/datasets/lerobot/droid_1.0.1/tree/71da9dcba338f53889907ac5762a2ebc6c2e55ff) | 95,658 | 27,630,375 | 813 | 0.400 |

The bucket has mutable names. `catalog.py` sorts the relative path, byte size, and content identifier of each file and hashes that inventory. Check the digest before comparing readers. The benchmark routes the draft's unversioned repository video URLs to the matching immutable revision. The bucket stays at its original address.

The HF speed runs use one CPU XL host per pair, one reader process, no DataLoader subprocesses, a pool of 32 episodes, and one PyTorch compute thread. Each reader retains its native decoding concurrency. The baseline defaults to two decode threads and the draft uses 32. The finite tests separately exercise one and four DataLoader workers and two CPU ranks. In the earlier #3917 follow-up, four workers reduced within-batch diversity from about 6.6 to 4 episodes on another dataset at the default pool size. That measured trade-off motivates keeping reader speed and batch diversity together. [Prior multi-worker measurements](https://github.com/huggingface/lerobot/pull/3917#issuecomment-6053229372) The camera-history workload requests five frames per view at offsets of −0.8, −0.6, −0.4, −0.2 and 0 seconds.

The Molmo bucket lists three index objects under `meta/mp4-sidecars`. The two full names identify the same 1.83 GB object. The archives use schema 1 and lack the source fingerprint expected by the baseline. A file listed with the dataset therefore does not establish that a new consumer can reuse it. Use the loader's observed build, download, or local-cache path. Total logical file size can also exceed physical bytes transferred when the Hub deduplicates content.

## Interpreting the measurements

- Start the consumer clock before constructing the reader. Record delivery of batches 1, 125 and 1,000. For a simulated publisher download that happens before construction, include that interval in consumer wall time. Report raw deliveries and validated examples separately whenever they differ.
- Check exact coverage on a finite fixture. On a large source, count unique delivered identities and observed duplicates in the measured prefix. Overlapping action targets from different control steps are legal.
- Compare batch episode counts and nearby within-episode frame pairs. Interpret speed at the observed sampling diversity. In the original discussion, Pepijn measured about 5.2 distinct episodes per batch of eight from a pool of 11 versus 7.2 from a pool of 32. [Maintainer's review criterion](https://github.com/huggingface/lerobot/pull/3917#issuecomment-6035837027)
- For two complete prefixes, `matched_diversity.py` groups batches by the number of distinct episodes, the number of examples from the largest episode, and the number of same-episode pairs within 0.2 seconds. In each group use the smaller frequency across the two readers as its common weight. The sum of those frequencies is the distribution overlap, from zero to one. The weighted mean compares observed consumer waits within that overlap. Keep the measured wall time to 1,000 batches separate.
- Report the first 1,000 batches even if they fit inside the first episode pool. For full-pass projections, measure from batch 125 to the last batch of a run that spans two verified refills and includes at least 125 valid batches after the second refill. This charges the rate for stalls during both refills. Report the shorter post-refill rate as a sensitivity check. A CPU consumer's wait measures input delivery under the stated workload. Fetch and decode can overlap across threads.
- Estimate one full pass only with a qualified rate and compatible coverage evidence. At batch 125 the consumer has received 1,000 examples. With its measured wall time `t125`, `N` eligible anchors, and measured steady rate `r` after that batch, the estimate is `t125 + (N - 1000) / r`. This anchors the estimate to the same interval as the measured rate. Keep deployment, cache, and sampling conditions attached to estimates. An observed crossover compares cumulative delivery times at the same verified batch counts. Extrapolate an unobserved crossover only when both rates qualify and diversity is comparable.
- Record first-use construction, fresh receipt of a matching publisher index, and a new-process repeat with local caches separately. Include transfer and validation for a new consumer. The publisher's one-time build cost is shared across its consumers. Disk archive size and measured worker memory are separate quantities. [Baseline's sidecar resolution](https://github.com/huggingface/lerobot/blob/fe1c54594f6ba275fccf210b9f7b11f0e2324b89/src/lerobot/datasets/streaming_sidecar.py#L217-L234)

GR00T 2 also trained on ABC-130K and DROID. NVIDIA publishes eligible sample starts, chooses sampling order at training time, and stages video for decoding. Its three-camera test attributed about 90% of preparation time to decode. Its separate 1,024-GPU measurement reported p99 exposed input wait of 0.61 ms. These results motivate measuring both worker effort and consumer wait. [GR00T 2, section 4.7.2](https://research.nvidia.com/labs/gear/gr00t-2/assets/gr00t2_whitepaper.pdf#page=25)

## Reproduce a cloud reader measurement

In a Linux `python:3.12-slim-trixie` container, run the following from this directory. The installation script fetches the pinned source archives and installs the CPU video stack. This example creates a separate cache for each reader. Set Hugging Face authentication through your normal environment.

```bash
export WORK=/work/review5388
bash runner/environment.sh
for arm in candidate baseline; do
    HF_HOME="$WORK/replay/$arm/hf" \
    HF_LEROBOT_HOME="$WORK/replay/$arm/lerobot" \
    PYTHONPATH="$WORK/src/$arm/src" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$WORK/venv/bin/python" benchmark.py --reader "$arm" --dataset droid \
        --cache-state cold --workers 0 --output "$WORK/replay/$arm.json"
done
```

Use `--single-frame` for the current-frame diagnostic. Use `--dataset abc --history` for camera history. Run a second process with the same cache paths and `--cache-state warm` to measure a restart. The `--cache-state` option labels a run. The cache directories determine its actual state. The accompanying JSONL trace contains batch identities, consumer waits, milestones, and refill events. Run `summarize.py <downloaded-job-directory> --fixture-evidence correctness/evidence` to export the comparison as CSV.

`coverage_audit.py` verifies each traced `(episode, frame)` against the pinned episode-length tables. `resource_summary.py` exports measured I/O, timing and memory at each milestone. `matched_diversity.py` accepts one result file from each reader on the same host, source, workload, cache condition and tolerance. The three utilities read the raw job result folder. The [comparison](measurements/comparison.csv), [ID audit](measurements/coverage.csv) and [compressed resource table](measurements/resources.csv.gz) contain the resulting public aggregates.
