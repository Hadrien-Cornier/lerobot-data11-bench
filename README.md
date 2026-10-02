# lerobot-data11-bench (GitHub copy)

This repo is a copy of the private Hugging Face dataset repo [`hadriencornier/lerobot-data11-bench`](https://huggingface.co/datasets/hadriencornier/lerobot-data11-bench), at commit `47a9b8e36b5140069efe565c16552e26381318a5` (2026-10-02).

It holds the benchmarks for DATA-11 ([huggingface/lerobot#3832](https://github.com/huggingface/lerobot/issues/3832)) and for PR [huggingface/lerobot#3917](https://github.com/huggingface/lerobot/pull/3917).

## Contents

| Folder | What it contains |
|---|---|
| `bundle*/` | The scripts and HF Jobs entrypoints of each benchmark round, with `SHA256SUMS`. |
| `results/` | The results of each HF job: JSON, Markdown summaries, logs, `STATUS.tsv` and memory logs. |
| `attempts/` | Restart-guard markers. Each HF job writes one marker, so that a restarted job does not run a second time. |

## What is not in this copy

The `variants/` folder (5.8 GB of generated test MP4 files) stays on Hugging Face only. GitHub does not accept files larger than 100 MB, and 10 of these files are larger. Download it with:

```bash
hf download hadriencornier/lerobot-data11-bench --repo-type dataset --include "variants/*" --local-dir .
```

## Main results

- Camera stacking (DATA-11): public report at https://github.com/Hadrien-Cornier/lerobot-camera-stacking-benchmark.
- PR #3917 per-sample fixes: `results/ab-20260928`. Comment: https://github.com/huggingface/lerobot/pull/3917#issuecomment-5894904188.
- PR #3917 header probe and transforms: `results/ab2-*`, `results/ab2tf-*`, `results/moov-survey-20261002`, `results/ab3-*`, `results/ab3-summary-20261002`. Comment: https://github.com/huggingface/lerobot/pull/3917#issuecomment-5960299347.
