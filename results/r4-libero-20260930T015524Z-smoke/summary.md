# DATA-11 round 4: libero (lerobot/libero)

Source `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` (main now `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`), 1693 episodes, 273465 frames, fps 10. Cameras (STACK order, top to bottom): `image` 256x256, `image2` 256x256.
lerobot `32b671d437f8c50014f74a322ad3df09c6fab944` (PR #4702 head). SMOKE=1.

## Slice

- 5 whole consecutive episodes, 1406 frames per group (cap 1500, cap applied: True). Consecutive frames in the first file of every camera: 10109.
- Real median frames per video file: 6740 (slice / median = 0.21).
- Real video files per camera: image 37, image2 37; real parquet data files: 377 (episode table: 377); streaming load_dataset num_shards: 377.

## Encode

LeRobot defaults from the installed code: `{'g': '2', 'crf': '30', 'preset': '12', 'svtav1-params': 'fast-decode=0'}` (libsvtav1, yuv420p). SVT-AV1 reported: ['Preset M12 is mapped to M10.']; version ['SVT-AV1 Encoder Lib v3.0.0']; lp 8.
Stacked frame 512x256; pad pixels 0 per frame (0.0% of the stacked frame, 0 raw RGB bytes). Mux default (source faststart: False).

| file | shape | MB | encoder feed s | flush s |
|---|---|---|---|---|
| SEP/image | 256x256 | 4.69 | 0.6 | 0.0 |
| SEP/image2 | 256x256 | 2.95 | 0.7 | 0.0 |
| STACK | 512x256 | 7.54 | 1.5 | 0.0 |

SEP total 7.64 MB, STACK 7.54 MB (STACK / SEP = 0.987). Encode wall 4.6 s for 1406 frames (decode 1.7 s, process CPU 35.3 s); SEP and STACK share one loop, so their encode costs are not separated.

## Datasets

- sep: `hadriencornier/lerobot-data11-r4-libero-sep` commit `3d54cfbd27d9a3689936ef503c6fac2d64aadfc2`, 8 files per key, 40 episodes, 11248 frames, keys ['observation.images.image', 'observation.images.image2'], private True.
- stack: `hadriencornier/lerobot-data11-r4-libero-stack` commit `203e6c92472e3f547d3ac2a83e603c9cf76a01f7`, 8 files per key, 40 episodes, 11248 frames, keys ['observation.images.stacked'], private True.
- sep1: `hadriencornier/lerobot-data11-r4-libero-sep1` commit `029505b4ce6dc4e525b552a0e392fbf1a2a97358`, 8 files per key, 40 episodes, 11248 frames, keys ['observation.images.image'], private True.

## Checks

| camera | PSNR SEP vs source | STACK crop vs source | STACK crop vs SEP (min) |
|---|---|---|---|
| image | 41.57 | 41.48 | 45.21 (40.52) |
| image2 | 41.70 | 41.67 | 44.26 (39.62) |

- Camera order ok: True (2 same-shape pairs compared).
- Remote (hf://) == local, bit-exact: {'sep': {'frames_checked': 20, 'mismatch': 0, 'pass': True}, 'stack': {'frames_checked': 20, 'mismatch': 0, 'pass': True}, 'sep1': {'frames_checked': 20, 'mismatch': 0, 'pass': True}}.
- Through LeRobotDataset: SEP item == file frame 40/40; SEP1 == SEP first camera 20/20; STACK crop vs SEP item PSNR min {'image': 39.83586170420037, 'image2': 39.88219067664305}.
- Overall check ok: True (PSNR threshold 35.0 dB, STACK vs source min 38.12).

## StreamingLeRobotDataset over hf:// (one process = one DataLoader worker)

2 repeats, arms interleaved, 95% CI over repeats. Local copies instead of hf://: False.

| arm | frames/s | CPU ms/frame | mp4 fetch/frame | mp4 MB/frame | decoder hit | open ms | first yield s |
|---|---|---|---|---|---|---|---|
| SEP_S8_c100 | 378.6 +- 254.7 | 2.6 +- 1.7 | 0.0000 | 0.0000 | 1.000 | 0 | 5.7 |
| STACK_S8_c100 | 512.0 +- 48.2 | 1.9 +- 0.1 | 0.0000 | 0.0000 | 1.000 | 0 | 5.6 |
| SEP1_S8_c100 | 641.6 +- 623.4 | 1.6 +- 1.5 | 0.0000 | 0.0000 | 1.000 | 0 | 4.1 |
| SEP_S8_c12 | 18.3 +- 0.6 | 13.2 +- 2.6 | 0.5007 | 1.9118 | 0.750 | 104 | - |
| STACK_S8_c12 | 513.9 +- 36.0 | 1.9 +- 0.1 | 0.0000 | 0.0000 | 1.000 | 0 | 5.5 |

Audit, DataLoader num_workers = max_num_shards = 4 on the local SEP copy (8 parquet files): workers with frames 2; per worker w0: 200 frames, files [0, 2, 4, 6], max cache 8; w1: 200 frames, files [1, 3, 5, 7], max cache 8. Predicted active shards per worker [4, 4, 0, 0] (decoders [8, 8, 0, 0]).

## Map-style LeRobotDataset on the local copies (lerobot-train DataLoader settings)

cemu = max(1, round(100 x 8 / 37)) = 22. batch_size 8, EpisodeAwareSampler(shuffle=True), prefetch_factor 4, persistent_workers, spawn, return_uint8=True. 2 repeats, interleaved.

| arm | workers | cache | samples/s | CPU ms/sample | decoder hit | open ms |
|---|---|---|---|---|---|---|
| SEP_w0_c100 | 0 | 100 | 330.9 +- 28.5 | 4.2 +- 0.2 | 1.000 | 0.0 |
| SEP_w0_cemu | 0 | 22 | 331.0 +- 17.5 | 4.2 +- 0.2 | 1.000 | 0.0 |
| STACK_w0_c100 | 0 | 100 | 441.9 +- 95.1 | 2.3 +- 0.5 | 1.000 | 0.0 |
| STACK_w0_cemu | 0 | 22 | 438.7 +- 48.5 | 2.3 +- 0.2 | 1.000 | 0.0 |
| SEP1_w0_c100 | 0 | 100 | 641.4 +- 43.1 | 1.6 +- 0.1 | 1.000 | 0.0 |
| SEP1_w0_cemu | 0 | 22 | 634.1 +- 38.6 | 1.6 +- 0.1 | 1.000 | 0.0 |
| SEP_w4_c100 | 4 | 100 | 982.0 +- 146.3 | 5.8 +- 0.6 | 1.000 | 0.0 |
| SEP_w4_cemu | 4 | 22 | 973.7 +- 82.2 | 5.8 +- 0.2 | 1.000 | 0.0 |
| STACK_w4_c100 | 4 | 100 | 1151.5 +- 249.4 | 3.8 +- 1.2 | 1.000 | 0.0 |
| STACK_w4_cemu | 4 | 22 | 1151.5 +- 82.8 | 3.8 +- 0.4 | 1.000 | 0.0 |
| SEP1_w4_c100 | 4 | 100 | 1477.5 +- 228.7 | 2.7 +- 0.1 | 1.000 | 0.0 |
| SEP1_w4_cemu | 4 | 22 | 1484.1 +- 8.2 | 2.7 +- 0.1 | 1.000 | 0.0 |

## Extrapolation inputs

- Real video files per camera 37; real parquet data files 377; real median frames per file 6740 vs slice 1406.
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

