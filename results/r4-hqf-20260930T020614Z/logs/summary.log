# DATA-11 round 4: hqf (lerobot/high_quality_folding)

Source `lerobot/high_quality_folding@c9eb858d4b84e520edecbda84a3534c3c1e78436` (main now `c9eb858d4b84e520edecbda84a3534c3c1e78436`), 1200 episodes, 3254196 frames, fps 30. Cameras (STACK order, top to bottom): `left_wrist` 720x1280, `right_wrist` 720x1280, `base` 480x640.
lerobot `32b671d437f8c50014f74a322ad3df09c6fab944` (PR #4702 head). SMOKE=0.

## Slice

- 10 whole consecutive episodes, 18527 frames per group (cap 60000, cap applied: False). Consecutive frames in the first file of every camera: 18527.
- Real median frames per video file: 16956 (slice / median = 1.09).
- Real video files per camera: left_wrist 155, right_wrist 153, base 240; real parquet data files: 4 (episode table: 4); streaming load_dataset num_shards: 4.

## Encode

LeRobot defaults from the installed code: `{'g': '2', 'crf': '30', 'preset': '12', 'svtav1-params': 'fast-decode=0'}` (libsvtav1, yuv420p). SVT-AV1 reported: ['Preset M12 is mapped to M10.']; version ['SVT-AV1 Encoder Lib v3.0.0']; lp 8.
Stacked frame 1920x1280; pad pixels 307200 per frame (12.5% of the stacked frame, 921600 raw RGB bytes). Mux {'movflags': 'faststart'} (source faststart: True).

| file | shape | MB | encoder feed s | flush s |
|---|---|---|---|---|
| SEP/left_wrist | 720x1280 | 119.91 | 63.0 | 0.1 |
| SEP/right_wrist | 720x1280 | 118.37 | 54.8 | 0.1 |
| SEP/base | 480x640 | 188.07 | 21.9 | 0.1 |
| STACK | 1920x1280 | 436.93 | 262.0 | 0.2 |

SEP total 426.36 MB, STACK 436.93 MB (STACK / SEP = 1.025). Encode wall 480.9 s for 18527 frames (decode 42.0 s, process CPU 3759.9 s); SEP and STACK share one loop, so their encode costs are not separated.

## Datasets

- sep: `hadriencornier/lerobot-data11-r4-hqf-sep` commit `a285d620345057e86c620683e64407c6545afbc9`, 8 files per key, 80 episodes, 148216 frames, keys ['observation.images.left_wrist', 'observation.images.right_wrist', 'observation.images.base'], private True.
- stack: `hadriencornier/lerobot-data11-r4-hqf-stack` commit `ee2ae8822323caa094b1b07addbf5bd324161438`, 8 files per key, 80 episodes, 148216 frames, keys ['observation.images.stacked'], private True.
- sep1: `hadriencornier/lerobot-data11-r4-hqf-sep1` commit `6a787739c55234e0a1165f49fe1644c10ca78d27`, 8 files per key, 80 episodes, 148216 frames, keys ['observation.images.left_wrist'], private True.

## Checks

| camera | PSNR SEP vs source | STACK crop vs source | STACK crop vs SEP (min) |
|---|---|---|---|
| left_wrist | 44.08 | 43.97 | 47.49 (43.98) |
| right_wrist | 44.01 | 43.19 | 45.13 (40.64) |
| base | 42.57 | 42.08 | 43.02 (37.13) |

- Camera order ok: True (2 same-shape pairs compared).
- Remote (hf://) == local, bit-exact: {'sep': {'frames_checked': 18, 'mismatch': 0, 'pass': True}, 'stack': {'frames_checked': 20, 'mismatch': 0, 'pass': True}, 'sep1': {'frames_checked': 20, 'mismatch': 0, 'pass': True}}.
- Through LeRobotDataset: SEP item == file frame 60/60; SEP1 == SEP first camera 20/20; STACK crop vs SEP item PSNR min {'left_wrist': 43.78240520801975, 'right_wrist': 42.889490777722024, 'base': 38.10659623038615}.
- Overall check ok: True (PSNR threshold 35.0 dB, STACK vs source min 37.29).

## StreamingLeRobotDataset over hf:// (one process = one DataLoader worker)

5 repeats, arms interleaved, 95% CI over repeats. Local copies instead of hf://: False.

| arm | frames/s | CPU ms/frame | mp4 fetch/frame | mp4 MB/frame | decoder hit | open ms | first yield s |
|---|---|---|---|---|---|---|---|
| SEP_S8_c100 | 70.9 +- 4.3 | 14.1 +- 0.9 | 0.0000 | 0.0000 | 1.000 | 0 | 19.8 |
| STACK_S8_c100 | 71.0 +- 2.7 | 13.4 +- 0.6 | 0.0062 | 0.0327 | 1.000 | 0 | 16.0 |
| SEP1_S8_c100 | 190.1 +- 2.8 | 5.3 +- 0.1 | 0.0000 | 0.0000 | 1.000 | 0 | 8.0 |
| SEP_S8_c12 | 4.6 +- 0.3 | 62.9 +- 4.0 | 1.5157 | 8.0461 | 0.495 | 133 | - |
| STACK_S8_c12 | 75.4 +- 5.8 | 12.7 +- 1.0 | 0.0062 | 0.0327 | 1.000 | 0 | 15.1 |

Audit, DataLoader num_workers = max_num_shards = 4 on the local SEP copy (8 parquet files): workers with frames 2; per worker w0: 200 frames, files [0, 2, 4, 6], max cache 12; w1: 200 frames, files [1, 3, 5, 7], max cache 12. Predicted active shards per worker [4, 4, 0, 0] (decoders [12, 12, 0, 0]).

## Map-style LeRobotDataset on the local copies (lerobot-train DataLoader settings)

cemu = max(1, round(100 x 8 / 240)) = 3. batch_size 8, EpisodeAwareSampler(shuffle=True), prefetch_factor 4, persistent_workers, spawn, return_uint8=True. 4 repeats, interleaved.

| arm | workers | cache | samples/s | CPU ms/sample | decoder hit | open ms |
|---|---|---|---|---|---|---|
| SEP_w0_c100 | 0 | 100 | 98.7 +- 18.4 | 19.5 +- 2.4 | 1.000 | 0.0 |
| SEP_w0_cemu | 0 | 3 | 82.1 +- 4.7 | 22.3 +- 0.7 | 0.119 | 1.8 |
| STACK_w0_c100 | 0 | 100 | 59.7 +- 1.8 | 16.8 +- 0.5 | 1.000 | 0.0 |
| STACK_w0_cemu | 0 | 3 | 59.5 +- 1.4 | 16.8 +- 0.4 | 0.367 | 1.5 |
| SEP1_w0_c100 | 0 | 100 | 170.4 +- 3.7 | 5.9 +- 0.1 | 1.000 | 0.0 |
| SEP1_w0_cemu | 0 | 3 | 143.3 +- 4.3 | 7.0 +- 0.2 | 0.371 | 1.3 |
| SEP_w4_c100 | 4 | 100 | 247.7 +- 7.6 | 27.6 +- 0.7 | 1.000 | 1.1 |
| SEP_w4_cemu | 4 | 3 | 233.7 +- 2.4 | 29.5 +- 0.3 | 0.126 | 2.3 |
| STACK_w4_c100 | 4 | 100 | 168.9 +- 18.4 | 24.9 +- 1.3 | 1.000 | 1.0 |
| STACK_w4_cemu | 4 | 3 | 185.2 +- 2.8 | 23.5 +- 0.4 | 0.372 | 2.0 |
| SEP1_w4_c100 | 4 | 100 | 393.7 +- 62.4 | 10.8 +- 1.2 | 1.000 | 0.7 |
| SEP1_w4_cemu | 4 | 3 | 436.6 +- 26.0 | 10.3 +- 0.6 | 0.373 | 1.6 |

## Extrapolation inputs

- Real video files per camera 240; real parquet data files 4; real median frames per file 16956 vs slice 18527.
- Map-style, full size: each worker's cache holds 100 decoders; random access over SEP 720 files vs STACK 240 files -> expected hit rate about 0.139 (SEP) vs 0.417 (STACK). The cemu arms measure this ratio on the slice.
- Streaming, lerobot-train (max_num_shards = num_workers, every worker iterates num_shards = min(P, W) shards; datasets then splits each shard's files across the W workers again, so worker w reads only the shards with more than w files). Decoders needed per worker = active shards x cameras (SEP) or active shards (STACK):

| num_workers W | workers that read | active shards: worker 0 / mean | SEP decoders (worker 0) | STACK decoders (worker 0) | SEP overflows 100 | STACK overflows 100 |
|---|---|---|---|---|---|---|
| 4 | 1 | 4 / 1.0 | 12 | 4 | False | False |
| 8 | 1 | 4 / 0.5 | 12 | 4 | False | False |
| 16 | 1 | 4 / 0.2 | 12 | 4 | False | False |
| 32 | 1 | 4 / 0.1 | 12 | 4 | False | False |
| 34 | 1 | 4 / 0.1 | 12 | 4 | False | False |
| 64 | 1 | 4 / 0.1 | 12 | 4 | False | False |
| 101 | 1 | 4 / 0.0 | 12 | 4 | False | False |
| 128 | 1 | 4 / 0.0 | 12 | 4 | False | False |

Worker 0 always reads min(P, W) shards, so the rule 'SEP overflows when min(P, W) x cameras > 100, STACK when min(P, W) > 100' holds for the busiest worker. Other workers read fewer shards, and all W workers carry the full load only when P >= W^2 (every shard has at least W files); with P < W some workers read nothing. The audit above tests this split on the SEP copy.

## Limits

- One group per dataset, copied 8 times: files are identical copies, so file-to-file variation (content, sizes) is not sampled.
- Streaming arms are one process (one DataLoader worker); the overflow case is emulated with cache 12 at S = 8, not run with 64 workers.
- Map-style arms read local disk (page cache warm after the first pass); they measure decode CPU, not network or cold-disk reads.
- STACK and SEP encodes share one loop, so encode time per layout is not measured separately.
- State / action values are random; only their shapes match the source.
- cpu-upgrade box (8 CPUs, 32 GB) shared with other stages' background uploads; compare layouts only inside this run.

