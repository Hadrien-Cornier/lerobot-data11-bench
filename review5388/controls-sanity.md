# Check the first 1,000 MolmoAct2 batches with the new streaming controls

The integrated reader delivered batch 1 **3.54 seconds sooner** and batch 1,000 **4.65 seconds later** in one paired run. Both readers delivered 8,000 unique examples from 32 episodes.

| Reader | Constructor | Batch 1 | Batch 125 | Batch 1,000 | Mean distinct episodes per batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grouped RGB, `adbdf5f0` | 11.08 s | 29.60 s | 71.81 s | 154.53 s | 7.045 |
| Integrated controls, `8b6f1bdb` | 7.32 s | 26.06 s | 69.66 s | 159.17 s | 7.097 |

The runs used one 8-vCPU AMD EPYC 7R13 Hugging Face Jobs host, with the grouped reader first. Each run started in a new process with fresh disk caches. Both used three current RGB views, one current action, current state, a 32-episode pool, batch size 8, seed 42, zero DataLoader workers, and 1 ms timestamp tolerance. The integrated reader used eight episodes of Parquet read-ahead and a 2 GiB remote video cache. Timings start before dataset construction and include the consumer checks for earlier batches. With the measured consumer work removed, batch 1,000 took 150.09 s and 153.54 s. [Results](controls-sanity/result.json) · [Source revisions and reader hashes](controls-sanity/pins.json)

The bucket manifest matched before and after the pair. The grouped reader reproduced all 8,000 saved IDs in order. The integrated reader shuffled files and sampled from one episode pool, so its sample order changed. This single pair measures the first 1,000 batches under each reader's own sampling order. [Earlier grouped reader check](native-grouped-reader.md)

## Verify RGB frames with the same sample IDs

The integrated reader produced a pixel hash for one sample from batch 1 and from every 100th batch. The grouped reader decoded the same 11 sample IDs. All 33 camera frames matched byte for byte. On a separate 786-sample local fixture, every sample from both readers matched its Parquet row and each of the 2,358 camera frames matched an independent FFmpeg decode. [MolmoAct2 parity](controls-sanity/result.json) · [Grouped fixture](controls-sanity/local-grouped.json) · [Integrated fixture](controls-sanity/local-controls.json)

Hugging Face reports a total job duration of 527 seconds including scheduling. The conservative hardware cost was at most $0.005. The review's cumulative gross upper bound is $11.0222 against the $15 cap. [Cost receipt](controls-sanity/result.json)

## Reproduce

From `review5388/`, package the benchmark and run it on an 8-vCPU Hugging Face Jobs host. The script installs both pinned reader revisions and runs the two arms in order. The job prints timing, parity, and source manifest records to its log. The timeout bounds hardware cost to $0.013 at the recorded hourly rate. [Pricing and timeouts](https://huggingface.co/docs/hub/en/jobs-pricing)

```bash
payload=$(tar -czf - catalog.py -C controls-sanity benchmark.py pins.json | base64 | tr -d '\n')
hf jobs run --detach --flavor cpu-upgrade --timeout 25m \
  --secrets HF_TOKEN python:3.12-slim-trixie \
  bash -c "$(cat controls-sanity/boot.sh)" boot "$payload"
```
