# Reusing decoders and grouping video frames on LeRobot #5388

On the same 1,000 MolmoAct2 batches, retaining decoders saved **37.94–48.68 seconds**. Grouping requests from the same MP4 within a batch saved **another 54.95–56.21 seconds**. Together they cut delivery time by **42–44%** versus stock #5388. At batch one, the total saving was **10.15–12.32 seconds**.

## Measured delivery time

Each cell gives elapsed seconds from before dataset construction. The first pass ran stock, reuse, then grouped. The second ran grouped, reuse, then stock. Each used a new process and fresh disk caches.

| Configuration | Batch 1, forward | Batch 1, reverse | Batch 1,000, forward | Batch 1,000, reverse |
|---|---:|---:|---:|---:|
| Stock #5388 | 36.17 | 34.54 | 234.44 | 222.84 |
| Decoder reuse | 35.40 | 29.88 | 185.76 | 184.90 |
| Reuse and same-MP4 grouping | 26.02 | 22.21 | 130.81 | 128.69 |

| Change, within each pass | Time saved at batch 1 | Time saved at batch 1,000 |
|---|---:|---:|
| Reuse versus stock | 0.78–4.65 s | 37.94–48.68 s (17–21%) |
| Grouping added to reuse | 7.67–9.37 s | 54.95–56.21 s (30%) |
| Both versus stock | 10.15–12.32 s | 94.15–103.63 s (42–44%) |

All six primary runs returned identical RGB, state, action and sample order. All fetched the same 1,642,613,961 bytes of MP4 blocks by batch 1,000. Stock opened about 24,100 decoders. Reuse reduced that to 4,377–4,407. Grouping reduced it to 100–102. With 32 samples decoding concurrently, reuse alone can open several decoders on one busy file and evict others. Grouping those requests reduced evictions from over 4,200 to 4–6. Counters include work already underway for later batches. [Milestone counters and output fingerprints](measurements/internvideo-molmo.json)

## How much came from a multi-frame decode call?

Two additional runs kept the grouped scheduler and borrowed the same decoder for each group, but retrieved one frame per call. Each used a fresh process and cache on the same host, after both primary comparisons.

| Frame retrieval with the grouped scheduler | Batch 1 | Batch 1,000 | Decoder opens by batch 1,000 | Frame-read calls by batch 1,000 |
|---|---:|---:|---:|---:|
| One frame per call | 23.64–23.95 s | 139.01–146.88 s | 103 | 24,107–24,120 |
| All requested indices for the MP4 in one call | 22.21–26.02 s | 128.69–130.81 s | 100–102 | 17,436–17,470 |

Most of the gain over per-sample reuse was already present with singleton reads. Grouping work by MP4 let each job keep its decoder through several requests. Sending the indices together then saved a further **8–18 seconds** by batch 1,000 in these runs. The multi-frame call had no consistent advantage at batch one. These controls ran later on the host, so the interval describes the observed timings across runs. It also includes the effect of sorting the grouped indices before calling TorchCodec.

All eight runs produced the same output fingerprint. By batch 1,000, the consumer had spent 10.0–13.2 seconds checking prior batches for stock or per-sample reuse, and 4.2–5.3 seconds for the grouped schedulers. Checking and background decode can overlap. The timings measure delivery to this CPU consumer on one host. [Per-arm results](measurements/internvideo-molmo.json) · [HF job](https://huggingface.co/jobs/hadriencornier/6aff5839efaf5e14f4b200b2)

## What each reader does

All configurations use the same episode sampler and return examples in the original order. Each example has three current camera images, the current state and the current action. The clock starts before the reader opens the MolmoAct2 bucket and stops when the CPU consumer receives the collated batch.

| Configuration | How it gets the 24 images in a batch of eight |
|---|---|
| Stock #5388 | Opens a decoder for each camera request. Each request retrieves one frame. |
| Decoder reuse | Keeps up to 96 open decoder and file-handle pairs. A sample can borrow an idle decoder for the same MP4. It still asks for each image separately. |
| Reuse and grouping | Plans eight examples together, groups their requests by MP4, then asks one decoder for all the required indices in each file. Puts the images back in sample order. |
| Grouped scheduler, singleton reads | Makes the same groups and borrows a decoder once per group. Calls the decoder once per image. Comparing this with the previous row tests the effect of sending several indices to TorchCodec in one call. |

The stock and reuse readers execute up to 32 sample tasks. The grouped readers admit up to four batches of eight at a time and use 32 executor threads for the file groups. The 96-decoder limit allows three views for each of the 32 episodes in the pool. Concurrent requests for the same MP4 can use different decoders.

The first batch drew 24 images from 16 MP4s. Across all 1,000 batches, 24,000 images came from 39 MP4s. Grouping inside each batch reduces calls from 24,000 to 17,365 (27.6%). Most groups still contain one image. Counters at delivery include prefetched work and can exceed these counts. [Grouping opportunity from the episode metadata](measurements/internvideo-opportunity.json)

## How to reproduce it

The reference contains the first 1,000 batches of the earlier third #5388 repeat. Every measured run checks all 8,000 ordered episode and frame IDs. It also hashes every RGB byte, state and action in order. The job checks the bucket manifest before and after each group of runs. Each arm starts a fresh Python process with empty application caches on the same HF CPU XL host. Upstream network and service caches can persist between arms.

On a separate finite fixture, all four paths returned the same 786 samples, including the last partial batch. Every output field matched. Stock RGB agreed pixel for pixel with an independent FFmpeg decode. [Fixture and oracle](correctness/check_internvideo.py) · [Fixture results](measurements/internvideo-fixture.json)

The test uses LeRobot at `1a60cb4b`, Datasets at `464fe790`, one PyTorch compute thread, zero DataLoader subprocesses, a 32-episode pool and 1 ms timestamp tolerance. The timing includes the consumer's checks on prior batches. The machine-readable results give the elapsed time spent on those checks separately. The 1,000 batches precede the first episode-pool refill.

Run from this directory after authenticating to Hugging Face. Supply a private dataset repository you control for the temporary bundle and results.

```bash
export REVIEW5388_RESULTS_REPO="YOUR_HF_ACCOUNT/YOUR_PRIVATE_DATASET"
uv run --no-project --python 3.12 --with huggingface-hub==2.0.0 \
    runner/launch.py molmo-internvideo01 --profile internvideo --minutes 44
# After the job finishes
uv run --no-project --python 3.12 --with huggingface-hub==2.0.0 \
    runner/collect.py molmo-internvideo01 --sync
uv run --no-project --python 3.12 internvideo_results.py results/downloads/review5388/results/molmo-internvideo01 \
    measurements/internvideo-molmo.json
```

[Benchmark entry point](internvideo_bench.py) · [Decoder variants](internvideo_variants.py) · [Ordered reference](internvideo_reference.json.gz) · [Output checks](internvideo_audit.py)
