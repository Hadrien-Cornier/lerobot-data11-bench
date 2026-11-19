# Comparing the LeRobot streaming readers

The unit of work is one training example at an identified control step. Three camera views and 100 target actions still count as one example. Its identity is the source snapshot, episode index, and frame index. Eight examples make one batch.

## Readers and input data

The baseline is the merge base of [LeRobot #5388](https://github.com/huggingface/lerobot/pull/5388), `fe1c54594f6ba275fccf210b9f7b11f0e2324b89`. It includes the merged seed, worker, resume, rank, and mapped-index changes after [#3917](https://github.com/huggingface/lerobot/pull/3917). The draft at `1a60cb4b800c2947834342752c32f064c8bf143c` requires [Datasets #8983](https://github.com/huggingface/datasets/pull/8983) at `464fe790b82a320350c9c686c1b7e409323a1cf3`. The paired cloud runs use that Datasets revision on both sides.

The draft groups Parquet rows by episode, shuffles a buffer of episodes, and maps video decoding over the sampled rows. Its block cache reuses remote byte ranges across threads in a process. The baseline builds or loads a per-MP4 seek index, schedules samples from an episode pool, and fetches video ranges using that index. The diff includes 5,414 deleted source lines and 5,893 deleted test lines. It removes explicit streaming resume and distributed sampling code from the trainer. [Proposed reader](https://github.com/huggingface/lerobot/blob/1a60cb4b800c2947834342752c32f064c8bf143c/src/lerobot/datasets/streaming_dataset.py#L227-L259) · [Full diff](https://github.com/huggingface/lerobot/pull/5388/files)

The input inventory comes from each pinned Hub tree, the bucket object identifiers, and the corresponding `meta/info.json`. Video bytes are compressed storage bytes.

| Source | Episodes | Control steps | MP4s | Video bytes, decimal TB |
|---|---:|---:|---:|---:|
| [MolmoAct2 bucket](https://huggingface.co/buckets/pepijn223/MolmoAct2-BimanualYAM-Dataset-bucket) | 32,246 | 76,046,658 | 5,860 | 2.343 |
| [ABC-130K](https://huggingface.co/datasets/lerobot/abc_130k_v3_train/tree/68651e4929d9fb00f798937b2d62617cab5c771d) | 129,225 | 382,468,339 | 38,164 | 3.378 |
| [DROID](https://huggingface.co/datasets/lerobot/droid_1.0.1/tree/71da9dcba338f53889907ac5762a2ebc6c2e55ff) | 95,658 | 27,630,375 | 813 | 0.400 |

The bucket has mutable names. `catalog.py` sorts the relative path, byte size, and content identifier of each file and hashes that inventory. Check the digest before comparing readers. The benchmark routes the draft's unversioned repository video URLs to the matching immutable revision. The bucket stays at its original address.

## Interpreting the measurements

- Start the consumer clock before constructing the reader. Record delivery of batches 1, 125 and 1,000. For a simulated publisher download that happens before construction, include that interval in consumer wall time. Report reported raw deliveries and validated examples separately whenever they differ.
- Check exact coverage on a finite fixture. On a large source, count unique delivered identities and observed duplicates in the measured prefix. Overlapping action targets from different control steps are legal.
- Compare batch episode counts and nearby within-episode frame pairs. Interpret speed at the observed sampling diversity. In the original discussion, Pepijn measured about 5.2 distinct episodes per batch of eight from a pool of 11 versus 7.2 from a pool of 32. [Maintainer's review criterion](https://github.com/huggingface/lerobot/pull/3917#issuecomment-6035837027)
- Report the first 1,000 batches even if they fit inside the first episode pool. Qualify a sustained rate only after verifying two refills and observing subsequent valid batches. A CPU consumer's wait measures input delivery under the stated workload. Fetch and decode can overlap across threads.
- Estimate one full pass only with a qualified rate and compatible coverage evidence. With first-batch time `S`, eight valid examples delivered at that time, `N` eligible examples, and measured steady rate `r`, the estimate is `S + (N - 8) / r`. Keep deployment, cache, and sampling conditions attached to estimates. Solve the two time curves for a crossover only when both rates qualify.
- Record first-use construction, fresh receipt of a matching publisher index, and a new-process repeat with local caches separately. Include transfer and validation for a new consumer. The publisher's one-time build cost is shared across its consumers. Disk archive size and measured worker memory are separate quantities. [Baseline's sidecar resolution](https://github.com/huggingface/lerobot/blob/fe1c54594f6ba275fccf210b9f7b11f0e2324b89/src/lerobot/datasets/streaming_sidecar.py#L217-L234)

GR00T 2 also trained on ABC-130K and DROID. NVIDIA publishes eligible sample starts, chooses sampling order at training time, and stages video for decoding. Its three-camera test attributed about 90% of preparation time to decode. Its separate 1,024-GPU measurement reported p99 exposed input wait of 0.61 ms. These results motivate measuring both worker effort and consumer wait. [GR00T 2, section 4.7.2](https://research.nvidia.com/labs/gear/gr00t-2/assets/gr00t2_whitepaper.pdf#page=25)
