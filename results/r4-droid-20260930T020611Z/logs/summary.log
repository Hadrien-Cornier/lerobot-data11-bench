# DATA-11 round 4: droid (lerobot/droid_1.0.1)

Source `lerobot/droid_1.0.1@0eabc778f959c54b8c5aa3626cc1128d2d2e54d4` (main now `0eabc778f959c54b8c5aa3626cc1128d2d2e54d4`), 95658 episodes, 27630375 frames, fps 15. Cameras (STACK order, top to bottom): `wrist_left` 180x320, `exterior_1_left` 180x320, `exterior_2_left` 180x320.
lerobot `32b671d437f8c50014f74a322ad3df09c6fab944` (PR #4702 head). SMOKE=0.

## Slice

- 197 whole consecutive episodes, 59826 frames per group (cap 60000, cap applied: True). Consecutive frames in the first file of every camera: 96183.
- Real median frames per video file: 94102 (slice / median = 0.64).
- Real video files per camera: wrist_left 183, exterior_1_left 302, exterior_2_left 299; real parquet data files: 156 (episode table: 86); streaming load_dataset num_shards: 156.

## Encode

LeRobot defaults from the installed code: `{'g': '2', 'crf': '30', 'preset': '12', 'svtav1-params': 'fast-decode=0'}` (libsvtav1, yuv420p). SVT-AV1 reported: ['Preset M12 is mapped to M10.']; version ['SVT-AV1 Encoder Lib v3.0.0']; lp 8.
Stacked frame 540x320; pad pixels 0 per frame (0.0% of the stacked frame, 0 raw RGB bytes). Mux {'movflags': 'faststart'} (source faststart: True).

| file | shape | MB | encoder feed s | flush s |
|---|---|---|---|---|
| SEP/wrist_left | 180x320 | 190.71 | 38.1 | 0.0 |
| SEP/exterior_1_left | 180x320 | 320.55 | 41.8 | 0.1 |
| SEP/exterior_2_left | 180x320 | 323.37 | 38.9 | 0.1 |
| STACK | 540x320 | 833.53 | 106.4 | 0.2 |

SEP total 834.63 MB, STACK 833.53 MB (STACK / SEP = 0.999). Encode wall 338.3 s for 59826 frames (decode 106.2 s, process CPU 2700.8 s); SEP and STACK share one loop, so their encode costs are not separated.

## Datasets

- sep: `hadriencornier/lerobot-data11-r4-droid-sep` commit `9be8467a34816e7faca89f273ac3155bc81f119c`, 8 files per key, 1576 episodes, 478608 frames, keys ['observation.images.wrist_left', 'observation.images.exterior_1_left', 'observation.images.exterior_2_left'], private True.
- stack: `hadriencornier/lerobot-data11-r4-droid-stack` commit `62b895ae483292c0389a361abfe3edba5094d90a`, 8 files per key, 1576 episodes, 478608 frames, keys ['observation.images.stacked'], private True.
- sep1: `hadriencornier/lerobot-data11-r4-droid-sep1` commit `0b2d75fc1582e498446fe45f6492613179f53e84`, 8 files per key, 1576 episodes, 478608 frames, keys ['observation.images.wrist_left'], private True.

## Checks

| camera | PSNR SEP vs source | STACK crop vs source | STACK crop vs SEP (min) |
|---|---|---|---|
| wrist_left | 39.74 | 39.43 | 41.51 (34.65) |
| exterior_1_left | 38.96 | 38.03 | 37.96 (34.33) |
| exterior_2_left | 38.98 | 38.28 | 38.45 (35.48) |

- Camera order ok: True (6 same-shape pairs compared).
- Remote (hf://) == local, bit-exact: {'sep': {'frames_checked': 18, 'mismatch': 0, 'pass': True}, 'stack': {'frames_checked': 20, 'mismatch': 0, 'pass': True}, 'sep1': {'frames_checked': 20, 'mismatch': 0, 'pass': True}}.
- Through LeRobotDataset: SEP item == file frame 60/60; SEP1 == SEP first camera 20/20; STACK crop vs SEP item PSNR min {'wrist_left': 36.27984529872011, 'exterior_1_left': 34.417391185945576, 'exterior_2_left': 36.33619170200489}.
- Overall check ok: False (PSNR threshold 35.0 dB, STACK vs source min 34.97).

## StreamingLeRobotDataset over hf:// (one process = one DataLoader worker)

5 repeats, arms interleaved, 95% CI over repeats. Local copies instead of hf://: False.

| arm | frames/s | CPU ms/frame | mp4 fetch/frame | mp4 MB/frame | decoder hit | open ms | first yield s |
|---|---|---|---|---|---|---|---|
| SEP_S8_c100 | 221.5 +- 6.8 | 4.5 +- 0.1 | 0.0000 | 0.0000 | 1.000 | 0 | 13.3 |
| STACK_S8_c100 | 328.4 +- 22.7 | 3.0 +- 0.2 | 0.0000 | 0.0000 | 1.000 | 0 | 9.2 |
| SEP1_S8_c100 | 653.8 +- 12.4 | 1.5 +- 0.0 | 0.0000 | 0.0000 | 1.000 | 0 | 7.4 |
| SEP_S8_c12 | 3.8 +- 0.7 | 53.8 +- 5.2 | 1.5254 | 8.0972 | 0.492 | 172 | - |
| STACK_S8_c12 | 333.9 +- 4.9 | 3.0 +- 0.0 | 0.0000 | 0.0000 | 1.000 | 0 | 8.8 |

Audit, DataLoader num_workers = max_num_shards = 4 on the local SEP copy (8 parquet files): workers with frames 2; per worker w0: 200 frames, files [0, 2, 4, 6], max cache 12; w1: 200 frames, files [1, 3, 5, 7], max cache 12. Predicted active shards per worker [4, 4, 0, 0] (decoders [12, 12, 0, 0]).

## Map-style LeRobotDataset on the local copies (lerobot-train DataLoader settings)

cemu = max(1, round(100 x 8 / 302)) = 3. batch_size 8, EpisodeAwareSampler(shuffle=True), prefetch_factor 4, persistent_workers, spawn, return_uint8=True. 4 repeats, interleaved.

| arm | workers | cache | samples/s | CPU ms/sample | decoder hit | open ms |
|---|---|---|---|---|---|---|
| SEP_w0_c100 | 0 | 100 | 237.8 +- 0.6 | 7.4 +- 0.0 | 1.000 | 0.0 |
| SEP_w0_cemu | 0 | 3 | 65.4 +- 4.6 | 19.0 +- 1.2 | 0.126 | 4.9 |
| STACK_w0_c100 | 0 | 100 | 254.5 +- 2.6 | 3.9 +- 0.0 | 1.000 | 0.0 |
| STACK_w0_cemu | 0 | 3 | 184.7 +- 3.2 | 5.4 +- 0.1 | 0.376 | 2.4 |
| SEP1_w0_c100 | 0 | 100 | 631.9 +- 6.6 | 1.6 +- 0.0 | 1.000 | 0.0 |
| SEP1_w0_cemu | 0 | 3 | 321.2 +- 6.9 | 3.1 +- 0.1 | 0.377 | 2.4 |
| SEP_w4_c100 | 4 | 100 | 764.2 +- 4.8 | 9.2 +- 0.0 | 1.000 | 0.0 |
| SEP_w4_cemu | 4 | 3 | 255.0 +- 7.0 | 20.3 +- 0.3 | 0.123 | 5.1 |
| STACK_w4_c100 | 4 | 100 | 860.0 +- 37.4 | 5.6 +- 0.1 | 1.000 | 0.0 |
| STACK_w4_cemu | 4 | 3 | 642.6 +- 6.4 | 7.2 +- 0.1 | 0.378 | 2.6 |
| SEP1_w4_c100 | 4 | 100 | 1498.2 +- 45.1 | 2.7 +- 0.0 | 1.000 | 0.0 |
| SEP1_w4_cemu | 4 | 3 | 1103.3 +- 18.9 | 4.4 +- 0.0 | 0.376 | 2.5 |

## Extrapolation inputs

- Real video files per camera 302; real parquet data files 156; real median frames per file 94102 vs slice 59826.
- Map-style, full size: each worker's cache holds 100 decoders; random access over SEP 906 files vs STACK 302 files -> expected hit rate about 0.110 (SEP) vs 0.331 (STACK). The cemu arms measure this ratio on the slice.
- Streaming, lerobot-train (max_num_shards = num_workers, every worker iterates num_shards = min(P, W) shards; datasets then splits each shard's files across the W workers again, so worker w reads only the shards with more than w files). Decoders needed per worker = active shards x cameras (SEP) or active shards (STACK):

| num_workers W | workers that read | active shards: worker 0 / mean | SEP decoders (worker 0) | STACK decoders (worker 0) | SEP overflows 100 | STACK overflows 100 |
|---|---|---|---|---|---|---|
| 4 | 4 | 4 / 4.0 | 12 | 4 | False | False |
| 8 | 8 | 8 / 8.0 | 24 | 8 | False | False |
| 16 | 10 | 16 / 9.8 | 48 | 16 | False | False |
| 32 | 5 | 32 / 4.9 | 96 | 32 | False | False |
| 34 | 5 | 34 / 4.6 | 102 | 34 | True | False |
| 64 | 3 | 64 / 2.4 | 192 | 64 | True | False |
| 101 | 2 | 101 / 1.5 | 303 | 101 | True | True |
| 128 | 2 | 128 / 1.2 | 384 | 128 | True | True |

Worker 0 always reads min(P, W) shards, so the rule 'SEP overflows when min(P, W) x cameras > 100, STACK when min(P, W) > 100' holds for the busiest worker. Other workers read fewer shards, and all W workers carry the full load only when P >= W^2 (every shard has at least W files); with P < W some workers read nothing. The audit above tests this split on the SEP copy.

## Limits

- One group per dataset, copied 8 times: files are identical copies, so file-to-file variation (content, sizes) is not sampled.
- Streaming arms are one process (one DataLoader worker); the overflow case is emulated with cache 12 at S = 8, not run with 64 workers.
- Map-style arms read local disk (page cache warm after the first pass); they measure decode CPU, not network or cold-disk reads.
- STACK and SEP encodes share one loop, so encode time per layout is not measured separately.
- State / action values are random; only their shapes match the source.
- cpu-upgrade box (8 CPUs, 32 GB) shared with other stages' background uploads; compare layouts only inside this run.

