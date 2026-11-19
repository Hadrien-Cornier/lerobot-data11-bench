# Run the streaming comparison on Hugging Face

The runner uses the active Hugging Face token from `hf auth`. It uploads the input bundle and results to the private DATA-11 artifact dataset. It records each bundle's commit and SHA-256 before the job starts.

Run the integration probe and inspect its evidence.

```bash
cd /Users/HCornier/Documents/Personal/LeRobot/data11-bench/review5388
export HF_HUB_DISABLE_PROGRESS_BARS=1
uv run --no-project --python 3.12 --with huggingface-hub==2.0.0 runner/launch.py probe-name --profile probe --minutes 15
uv run --no-project --python 3.12 --with huggingface-hub==2.0.0 runner/collect.py probe-name --logs --sync
```

Use `--profile matrix` to compare both pinned readers on one CPU XL host. The default dataset order is DROID, Molmo, and ABC. Set an explicit subset with `--datasets` if only some sources need another run. A job that changes the order of cold readers uses `--profile reverse`.

- The primary workload uses one current image per camera and 100 action steps. The runner attempts batch 1, 125, and 1,000 with a bounded consumer wait. Sustained throughput requires two measured pool refills and the subsequent measurement window.
- Cold runs use separate disk caches. Warm runs use new processes with the disk caches from the primary cold runs. Datasets #8983's in-process cache is reset when a process exits.
- The Molmo current-frame diagnostic gives the draft a fresh metadata cache. The baseline reuses its cold-run index for this diagnostic. The two diagnostics have different cache states.
- The controlled transfer copies the baseline's validated native index from the private artifact dataset into a fresh consumer cache. Add `consumer_prep_seconds` to the measured consumer time. That interval includes the transfer, link, and SHA-256 verification. Baseline construction includes native archive validation and memory-map preparation. The index build and staging upload are publisher costs.
- The native bucket reader syncs the entire `meta/` subtree. This includes the three legacy index objects. The exact baseline discards these files as sources for its automatic native index lookup. The audit records the expected native key and the actual object headers.
- The runner checks the exact bucket inventory before and after each comparison. If the source digest differs from the pinned digest, exclude that comparison.
- Each job has one hard timeout. The ledger reserves gross cost before launch. A completed job with a server duration uses its conservative minute-rounded cost. A job without a reported duration retains the reservation.

Use `runner/collect.py <run> --logs --sync` to save the status, the container logs, and its private result files at a recorded artifact revision.

## Compare current-frame delivery at 1 ms tolerance

The separate tolerance control uses the exact Molmo bucket. It passes 0.001 seconds to both readers, keeps batch size eight and the native pool size of 32, and starts with fresh process and disk caches. The native 0.1 ms runs remain separate.

```bash
uv run --no-project --python 3.12 --with huggingface-hub==2.0.0 runner/launch.py molmo-tol01 \
  --profile tolerance --minutes 92
uv run --no-project --python 3.12 --with huggingface-hub==2.0.0 runner/collect.py molmo-tol01 --logs --sync
```

To test prepared-index receipt, use `--prepared-from-run matrix03 --minutes 27` after that matrix saves its verified Molmo transfer. The control downloads and verifies that index before baseline construction. It attempts only 1,000 batches per reader. After it saves the timed pair, it reuses the prepared baseline disk cache for the Molmo worker memory probe.

Use `--profile tolerance-repeat --minutes 75` to reverse the cold reader order and attempt 25,000 batches per reader with the same 1 ms setting. This run keeps its first 125 batches after refill two and continues to the fixed target. It saves additional checkpoints at batches 15,000, 17,500, 20,000, and 25,000.

## Measure worker memory

After the main matrix verifies the prepared archives, use them for a separate baseline worker-memory job. The default source order is ABC then DROID. Each source transfers its index once and builds its local memory map before either reader starts. Both arms use fresh Python processes and the same disk cache. The runner starts one worker first and four workers second. The second arm can reuse metadata and pages cached by the operating system. Both arms attempt 125 batches. The runner records the available shared memory and uses the same feasible prefetch factor for both arms.

```bash
uv run --no-project --python 3.12 --with huggingface-hub==2.0.0 runner/launch.py worker-mem01 \
  --profile worker-memory --prepared-from-run matrix03 --minutes 39
uv run --no-project --python 3.12 --with huggingface-hub==2.0.0 runner/collect.py worker-mem01 --logs --sync
```
