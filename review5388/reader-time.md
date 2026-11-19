# Where does the CPU reader spend its time?

On MolmoAct2, #3917 spent **494 seconds building the MP4 index** inside its 524-second constructor in the first cold run. #5388 skipped that build and opened in 7 seconds. After batch one, the pattern reversed. With a matching index available in repeat 3, the next 999 batches took **64 seconds with #3917** and **161 seconds with #5388**.

Here #3917 means the merged reader plus its fixes at `fe1c5459`. #5388 means the draft at `1a60cb4b`. The clock starts before dataset construction and includes any separate index receipt. Both readers delivered eight examples per batch, each with three current camera images, current state and current action. Both used a 1 ms timestamp tolerance. Each pair used fresh local caches and one 16-vCPU host. The no-index and matching-index pairs ran in different jobs. [Pins and workload](method.md) · [Measured batch times and source checks](measurements/molmo-prepared-repeats.json)

## What each reader does

```text
#3917  build or download frame-to-byte index
           -> fetch video ranges for an episode
           -> open a decoder for that episode and camera
           -> use the open decoder for later frames while it is cached

#5388  open dataset without building the index
           -> ask for a camera frame
           -> open a decoder and read compressed video blocks as needed
           -> keep the blocks for later requests
           -> next camera request opens another decoder
```

A decoder turns compressed video into pixels. Keeping the compressed bytes in memory saves repeat network reads. Keeping the decoder open also saves the work of opening the video again. In #3917 the decoder cache is keyed by episode and camera and has a limit. In #5388 the RGB helper constructs a decoder each time it is called. [#3917 decoder cache](https://github.com/huggingface/lerobot/blob/fe1c54594f6ba275fccf210b9f7b11f0e2324b89/src/lerobot/streaming/episode_cache.py#L193-L231) · [#5388 video read](https://github.com/huggingface/lerobot/blob/1a60cb4b800c2947834342752c32f064c8bf143c/src/lerobot/datasets/streaming_dataset.py#L416-L423) · [#5388 decoder construction](https://github.com/huggingface/lerobot/blob/1a60cb4b800c2947834342752c32f064c8bf143c/src/lerobot/datasets/video_utils.py#L282-L295)

## Where the consumer waits

The first four numeric columns are consecutive intervals. Adding them gives time to batch 1,000, within rounding. The constructor includes metadata work and, for #3917, opening or building its local index. The next interval includes all work needed to deliver the first eight examples.

| Fresh-disk setup | Receive and verify index | Constructor | Constructor end to batch 1 | Batches 2–1,000 | Total to batch 1,000 | Batches 1,001–4,000 |
|---|---:|---:|---:|---:|---:|---:|
| No matching index, #3917 | 0 s | 524.05 s | 47.57 s | 44.52 s | 616.15 s | 119.09 s |
| Same pair, #5388 | 0 s | 7.22 s | 9.12 s | 81.33 s | 97.67 s | 208.35 s |
| Matching index, repeat 3, #3917 | 9.88 s | 46.70 s | 67.36 s | 63.50 s | 187.45 s | 187.93 s |
| Same pair, #5388 | 0 s | 8.72 s | 21.00 s | 161.03 s | 190.75 s | 348.14 s |

In repeat 3, #5388 delivered batch one **94 seconds earlier**. By batch 1,000, #3917 had recovered that time and led by 3 seconds. At batch 4,000 the gap was 164 seconds. After grouping batches by episode count, largest episode share, and nearby same-episode frames, the distributions overlapped 96.8%. Within that overlap, mean consumer wait from batch 126 to 4,000 was **62 ms for #3917 and 112 ms for #5388**. [All three repeats](prepared-repeats.md)

## What the video counters show

These are counters sampled when the consumer received batch 1,000. The 8,000 examples contain 24,000 images. Prefetch can begin work on additional images before the consumer reaches that milestone. Video fetch counts exclude index-header requests.

| Counter through batch 1,000 | Cold #3917 | Cold #5388 | Matching index, #3917 | Same pair, #5388 |
|---|---:|---:|---:|---:|
| Index-header range fetches | 6,342 | 0 | 0 | 0 |
| Training-video range fetches | 120 | 1,533 | 120 | 1,530 |
| Training-video HTTP body bytes | 3.26 GB | 1.64 GB | 3.26 GB | 1.64 GB |
| Decoder opens on #3917 / calls that construct a decoder on #5388 | 97 | 24,086 | 97 | 24,065 |
| Video range fetch, summed elapsed seconds across calls | 440 | 559 | 619 | 384 |
| Decode call, summed elapsed seconds across calls, including waits | 133 | 2,016 | 185 | 3,014 |

#5388 moved fewer video bytes, issued more small range requests and opened many more decoders. In repeat 3 its summed fetch time was actually lower, while its consumer still waited longer. The decode timers have different boundaries. #5388 includes opening the decoder and waiting for reads inside its decode call. #3917 can fetch ahead and reuse a decoder. Concurrent calls overlap and fetch time can fall inside decode time. Adding these worker timers or dividing them by consumer wall time would give a misleading percentage. The cloud measurements identify repeated decoder construction as work worth removing. They do not isolate its contribution to the cloud gap. [Stage counters, compressed CSV](measurements/resources.csv.gz) · [Timed operations](telemetry.py)

## Local decoder experiment

We ran 2,358 RGB requests against 24 local MP4 files, in the same shuffled order, and checked every returned image against independent FFmpeg decodes. The files were already local. The test timed opening TorchCodec decoders separately from asking them for frames.

| Single-threaded request sequence | Decoder creations | Time creating decoders | Time retrieving frames inside TorchCodec |
|---|---:|---:|---:|
| #3917 as merged with fixes | 48 | 0.038 s | 0.791 s |
| #5388 as proposed | 2,358 | 1.123 s | 1.082 s |
| #5388 experiment, keep decoder and handle per MP4 | 24 | 0.033 s | 0.956 s |

Holding #5388's request order and pixels fixed, decoder reuse cut its RGB helper from **2.52 s to 1.22 s**. This isolates a cost on the small local fixture. It leaves the effect on networked multi-worker training open. [Reproduce the experiment](decoder-stages.md) · [#3917 trace](measurements/decoder-stages-baseline.json) · [#5388 trace](measurements/decoder-stages-candidate.json) · [#5388 reuse trace](measurements/decoder-stages-candidate-reuse.json)

## Can we publish the index once?

Yes. The publisher in the #3917 baseline can build a complete index for a pinned Hub dataset revision and pass `--push` to `lerobot-build-mp4-sidecar`. For a dataset repository it writes to the same repository's `lerobot-sidecars` branch, keeping the source commit fixed. For a bucket it writes under `meta/mp4-sidecars/`. The consumer checks its local index, tries the matching published index, and builds if it cannot reuse one. [Publisher command](https://github.com/huggingface/lerobot/blob/fe1c54594f6ba275fccf210b9f7b11f0e2324b89/src/lerobot/scripts/lerobot_build_mp4_sidecar.py#L37-L76) · [Consumer resolution](https://github.com/huggingface/lerobot/blob/fe1c54594f6ba275fccf210b9f7b11f0e2324b89/src/lerobot/datasets/streaming_sidecar.py#L181-L231)

The Molmo bucket's existing files used an older index format. The benchmark simulated compatible publication by staging the matching 1.83 GB archive privately. Each new consumer paid for transfer, verification and local setup. In the three prepared repeats, #3917 delivered batch one in **118–153 seconds** and #5388 in **30–39 seconds**. #3917 then passed #5388 at **batches 374–940**. Publication serves a team doing repeated runs well. Matching #5388's first-batch latency is a separate engineering target. [Reproduce the fresh-disk measurements](method.md#repeat-the-fresh-disk-comparison-with-a-matching-publisher-index)
