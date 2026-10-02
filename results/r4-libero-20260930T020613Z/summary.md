# DATA-11 round 4: libero (lerobot/libero)

Source `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` (main now `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`), 1693 episodes, 273465 frames, fps 10. Cameras (STACK order, top to bottom): `image` 256x256, `image2` 256x256.
lerobot `32b671d437f8c50014f74a322ad3df09c6fab944` (PR #4702 head). SMOKE=0.

## Slice

- 36 whole consecutive episodes, 10109 frames per group (cap 60000, cap applied: False). Consecutive frames in the first file of every camera: 10109.
- Real median frames per video file: 6740 (slice / median = 1.50).
- Real video files per camera: image 37, image2 37; real parquet data files: 377 (episode table: 377); streaming load_dataset num_shards: 377.

## Encode

LeRobot defaults from the installed code: `{'g': '2', 'crf': '30', 'preset': '12', 'svtav1-params': 'fast-decode=0'}` (libsvtav1, yuv420p). SVT-AV1 reported: ['Preset M12 is mapped to M10.']; version ['SVT-AV1 Encoder Lib v3.0.0']; lp 8.
Stacked frame 512x256; pad pixels 0 per frame (0.0% of the stacked frame, 0 raw RGB bytes). Mux default (source faststart: False).

| file | shape | MB | encoder feed s | flush s |
|---|---|---|---|---|
| SEP/image | 256x256 | 36.10 | 5.3 | 0.1 |
| SEP/image2 | 256x256 | 24.70 | 5.4 | 0.0 |
| STACK | 512x256 | 60.15 | 11.1 | 0.0 |

SEP total 60.80 MB, STACK 60.15 MB (STACK / SEP = 0.989). Encode wall 33.9 s for 10109 frames (decode 11.4 s, process CPU 269.3 s); SEP and STACK share one loop, so their encode costs are not separated.

## Datasets

- sep: `hadriencornier/lerobot-data11-r4-libero-sep` commit `8329594de128496d19c23f8b5814aed22f3f9d22`, 8 files per key, 288 episodes, 80872 frames, keys ['observation.images.image', 'observation.images.image2'], private True.
- stack: `hadriencornier/lerobot-data11-r4-libero-stack` commit `0cee59f115657e830efaad4a0e052bbd64c9276a`, 8 files per key, 288 episodes, 80872 frames, keys ['observation.images.stacked'], private True.
- sep1: `hadriencornier/lerobot-data11-r4-libero-sep1` commit `ce0da929b59dfeb2cbe4b7930ce139fc922a67d0`, 8 files per key, 288 episodes, 80872 frames, keys ['observation.images.image'], private True.

## Checks

| camera | PSNR SEP vs source | STACK crop vs source | STACK crop vs SEP (min) |
|---|---|---|---|
| image | 41.32 | 41.24 | 44.69 (40.24) |
| image2 | 41.67 | 41.71 | 44.13 (37.69) |

- Camera order ok: True (2 same-shape pairs compared).
- Remote (hf://) == local, bit-exact: {'sep': {'frames_checked': 20, 'mismatch': 0, 'pass': True}, 'stack': {'frames_checked': 20, 'mismatch': 0, 'pass': True}, 'sep1': {'frames_checked': 20, 'mismatch': 0, 'pass': True}}.
- Through LeRobotDataset: SEP item == file frame 40/40; SEP1 == SEP first camera 20/20; STACK crop vs SEP item PSNR min {'image': 39.356068769045756, 'image2': 35.326706618495706}.
- Overall check ok: True (PSNR threshold 35.0 dB, STACK vs source min 38.40).

## StreamingLeRobotDataset over hf:// (one process = one DataLoader worker)

5 repeats, arms interleaved, 95% CI over repeats. Local copies instead of hf://: False.

| arm | frames/s | CPU ms/frame | mp4 fetch/frame | mp4 MB/frame | decoder hit | open ms | first yield s |
|---|---|---|---|---|---|---|---|
| SEP_S8_c100 | 367.5 +- 15.2 | 2.7 +- 0.1 | 0.0000 | 0.0000 | 1.000 | 0 | 10.3 |
| STACK_S8_c100 | 487.9 +- 21.8 | 2.1 +- 0.1 | 0.0000 | 0.0000 | 1.000 | 0 | 8.4 |
| SEP1_S8_c100 | 653.2 +- 28.0 | 1.5 +- 0.1 | 0.0000 | 0.0000 | 1.000 | 0 | 6.4 |
| SEP_S8_c12 | 5.5 +- 0.6 | 36.2 +- 3.1 | 1.5356 | 5.4662 | 0.744 | 351 | - |
| STACK_S8_c12 | 490.0 +- 19.3 | 2.0 +- 0.1 | 0.0000 | 0.0000 | 1.000 | 0 | 8.2 |

Audit, DataLoader num_workers = max_num_shards = 4 on the local SEP copy (8 parquet files): workers with frames 2; per worker w0: 200 frames, files [0, 2, 4, 6], max cache 8; w1: 200 frames, files [1, 3, 5, 7], max cache 8. Predicted active shards per worker [4, 4, 0, 0] (decoders [8, 8, 0, 0]).

## Map-style LeRobotDataset on the local copies (lerobot-train DataLoader settings)

cemu = max(1, round(100 x 8 / 37)) = 22. batch_size 8, EpisodeAwareSampler(shuffle=True), prefetch_factor 4, persistent_workers, spawn, return_uint8=True. 4 repeats, interleaved.

| arm | workers | cache | samples/s | CPU ms/sample | decoder hit | open ms |
|---|---|---|---|---|---|---|
| SEP_w0_c100 | 0 | 100 | 336.7 +- 7.5 | 4.2 +- 0.1 | 1.000 | 0.0 |
| SEP_w0_cemu | 0 | 22 | 339.8 +- 3.9 | 4.2 +- 0.0 | 1.000 | 0.0 |
| STACK_w0_c100 | 0 | 100 | 424.0 +- 11.0 | 2.4 +- 0.1 | 1.000 | 0.0 |
| STACK_w0_cemu | 0 | 22 | 419.2 +- 19.1 | 2.4 +- 0.1 | 1.000 | 0.0 |
| SEP1_w0_c100 | 0 | 100 | 620.0 +- 14.4 | 1.6 +- 0.0 | 1.000 | 0.0 |
| SEP1_w0_cemu | 0 | 22 | 626.5 +- 13.8 | 1.6 +- 0.0 | 1.000 | 0.0 |
| SEP_w4_c100 | 4 | 100 | 965.2 +- 44.6 | 5.8 +- 0.2 | 1.000 | 0.0 |
| SEP_w4_cemu | 4 | 22 | 966.9 +- 47.5 | 5.8 +- 0.2 | 1.000 | 0.0 |
| STACK_w4_c100 | 4 | 100 | 1138.8 +- 43.4 | 3.9 +- 0.2 | 1.000 | 0.0 |
| STACK_w4_cemu | 4 | 22 | 1128.9 +- 41.6 | 3.9 +- 0.2 | 1.000 | 0.0 |
| SEP1_w4_c100 | 4 | 100 | 1468.2 +- 80.0 | 2.8 +- 0.1 | 1.000 | 0.0 |
| SEP1_w4_cemu | 4 | 22 | 1459.1 +- 61.6 | 2.8 +- 0.1 | 1.000 | 0.0 |

## Extrapolation inputs

- Real video files per camera 37; real parquet data files 377; real median frames per file 6740 vs slice 10109.
- Map-style, full size: each worker's cache holds 100 decoders; random access over SEP 74 files vs STACK 37 files -> expected hit rate about 1.000 (SEP) vs 1.000 (STACK). The cemu arms measure this ratio on the slice.
- Streaming, lerobot-train (max_num_shards = num_workers, every worker iterates num_shards = min(P, W) shards; datasets then splits each shard's files across the W workers again, so worker w reads only the shards with more than w files). Decoders needed per worker = active shards x cameras (SEP) or active shards (STACK):

| num_workers W | workers that read | active shards: worker 0 / mean | SEP decoders (worker 0) | STACK decoders (worker 0) | SEP overflows 100 | STACK overflows 100 |
|---|---|---|---|---|---|---|
| 4 | 4 | 4 / 4.0 | 8 | 4 | False | False |
| 8 | 8 | 8 / 8.0 | 16 | 8 | False | False |
| 16 | 16 | 16 / 16.0 | 32 | 16 | False | False |
| 32 | 12 | 32 / 11.8 | 64 | 32 | False | False |
| 34 | 12 | 34 / 11.1 | 68 | 34 | False | False |
| 64 | 6 | 64 / 5.9 | 128 | 64 | True | False |
| 101 | 4 | 101 / 3.7 | 202 | 101 | True | True |
| 128 | 3 | 128 / 2.9 | 256 | 128 | True | True |

Worker 0 always reads min(P, W) shards, so the rule 'SEP overflows when min(P, W) x cameras > 100, STACK when min(P, W) > 100' holds for the busiest worker. Other workers read fewer shards, and all W workers carry the full load only when P >= W^2 (every shard has at least W files); with P < W some workers read nothing. The audit above tests this split on the SEP copy.

## Limits

- One group per dataset, copied 8 times: files are identical copies, so file-to-file variation (content, sizes) is not sampled.
- Streaming arms are one process (one DataLoader worker); the overflow case is emulated with cache 12 at S = 8, not run with 64 workers.
- Map-style arms read local disk (page cache warm after the first pass); they measure decode CPU, not network or cold-disk reads.
- STACK and SEP encodes share one loop, so encode time per layout is not measured separately.
- State / action values are random; only their shapes match the source.
- cpu-upgrade box (8 CPUs, 32 GB) shared with other stages' background uploads; compare layouts only inside this run.

