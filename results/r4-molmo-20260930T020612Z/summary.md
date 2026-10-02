# DATA-11 round 4: molmo (allenai/MolmoAct2-BimanualYAM-Dataset)

Source `allenai/MolmoAct2-BimanualYAM-Dataset@e9f21ae15074330839f2ac25ed4b49d76dfa1f9c` (main now `e9f21ae15074330839f2ac25ed4b49d76dfa1f9c`), 32246 episodes, 76046658 frames, fps 30. Cameras (STACK order, top to bottom): `right` 360x640, `left` 360x640, `top` 360x640.
lerobot `32b671d437f8c50014f74a322ad3df09c6fab944` (PR #4702 head). SMOKE=0.

## Slice

- 16 whole consecutive episodes, 31115 frames per group (cap 60000, cap applied: False). Consecutive frames in the first file of every camera: 31115.
- Real median frames per video file: 38615 (slice / median = 0.81).
- Real video files per camera: right 1607, left 1765, top 2488; real parquet data files: 3576 (episode table: 3576); streaming load_dataset num_shards: 3576.

## Encode

LeRobot defaults from the installed code: `{'g': '2', 'crf': '30', 'preset': '12', 'svtav1-params': 'fast-decode=0'}` (libsvtav1, yuv420p). SVT-AV1 reported: ['Preset M12 is mapped to M10.']; version ['SVT-AV1 Encoder Lib v3.0.0']; lp 8.
Stacked frame 1080x640; pad pixels 0 per frame (0.0% of the stacked frame, 0 raw RGB bytes). Mux {'movflags': 'faststart'} (source faststart: True).

| file | shape | MB | encoder feed s | flush s |
|---|---|---|---|---|
| SEP/right | 360x640 | 195.87 | 52.2 | 0.1 |
| SEP/left | 360x640 | 215.19 | 47.2 | 0.1 |
| SEP/top | 360x640 | 301.52 | 46.8 | 0.1 |
| STACK | 1080x640 | 700.86 | 179.6 | 0.2 |

SEP total 712.57 MB, STACK 700.86 MB (STACK / SEP = 0.984). Encode wall 388.1 s for 31115 frames (decode 43.9 s, process CPU 3097.3 s); SEP and STACK share one loop, so their encode costs are not separated.

## Datasets

- sep: `hadriencornier/lerobot-data11-r4-molmo-sep` commit `8fc0fb143a7fd83802bf7f7658b6de1f7ffd3c28`, 8 files per key, 128 episodes, 248920 frames, keys ['observation.images.right', 'observation.images.left', 'observation.images.top'], private True.
- stack: `hadriencornier/lerobot-data11-r4-molmo-stack` commit `ef8b81635b08fab68a716096f0f50bdd5bffde14`, 8 files per key, 128 episodes, 248920 frames, keys ['observation.images.stacked'], private True.
- sep1: `hadriencornier/lerobot-data11-r4-molmo-sep1` commit `0b6e8789babd05fc3af4bdcc777a1efcad4798b4`, 8 files per key, 128 episodes, 248920 frames, keys ['observation.images.right'], private True.

## Checks

| camera | PSNR SEP vs source | STACK crop vs source | STACK crop vs SEP (min) |
|---|---|---|---|
| right | 40.79 | 40.53 | 42.77 (40.21) |
| left | 41.36 | 40.31 | 41.53 (37.77) |
| top | 39.88 | 39.27 | 40.46 (38.92) |

- Camera order ok: True (6 same-shape pairs compared).
- Remote (hf://) == local, bit-exact: {'sep': {'frames_checked': 18, 'mismatch': 0, 'pass': True}, 'stack': {'frames_checked': 20, 'mismatch': 0, 'pass': True}, 'sep1': {'frames_checked': 20, 'mismatch': 0, 'pass': True}}.
- Through LeRobotDataset: SEP item == file frame 60/60; SEP1 == SEP first camera 20/20; STACK crop vs SEP item PSNR min {'right': 38.33586588853815, 'left': 38.9168112615602, 'top': 38.194088135053725}.
- Overall check ok: True (PSNR threshold 35.0 dB, STACK vs source min 36.63).

## StreamingLeRobotDataset over hf:// (one process = one DataLoader worker)

5 repeats, arms interleaved, 95% CI over repeats. Local copies instead of hf://: False.

| arm | frames/s | CPU ms/frame | mp4 fetch/frame | mp4 MB/frame | decoder hit | open ms | first yield s |
|---|---|---|---|---|---|---|---|
| SEP_S8_c100 | 144.6 +- 10.1 | 6.9 +- 0.5 | 0.0000 | 0.0000 | 1.000 | 0 | 15.2 |
| STACK_S8_c100 | 179.4 +- 13.0 | 5.6 +- 0.4 | 0.0000 | 0.0000 | 1.000 | 0 | 9.6 |
| SEP1_S8_c100 | 438.1 +- 6.7 | 2.3 +- 0.0 | 0.0000 | 0.0000 | 1.000 | 0 | 6.0 |
| SEP_S8_c12 | 3.1 +- 0.7 | 55.2 +- 6.8 | 1.5206 | 8.0720 | 0.493 | 209 | - |
| STACK_S8_c12 | 183.7 +- 1.7 | 5.4 +- 0.1 | 0.0000 | 0.0000 | 1.000 | 0 | 9.7 |

Audit, DataLoader num_workers = max_num_shards = 4 on the local SEP copy (8 parquet files): workers with frames 2; per worker w0: 200 frames, files [0, 2, 4, 6], max cache 12; w1: 200 frames, files [1, 3, 5, 7], max cache 12. Predicted active shards per worker [4, 4, 0, 0] (decoders [12, 12, 0, 0]).

## Map-style LeRobotDataset on the local copies (lerobot-train DataLoader settings)

cemu = max(1, round(100 x 8 / 2488)) = 1. batch_size 8, EpisodeAwareSampler(shuffle=True), prefetch_factor 4, persistent_workers, spawn, return_uint8=True. 4 repeats, interleaved.

| arm | workers | cache | samples/s | CPU ms/sample | decoder hit | open ms |
|---|---|---|---|---|---|---|
| SEP_w0_c100 | 0 | 100 | 163.1 +- 4.9 | 11.7 +- 0.2 | 1.000 | 0.0 |
| STACK_w0_c100 | 0 | 100 | 123.6 +- 6.3 | 8.1 +- 0.4 | 1.000 | 0.0 |
| STACK_w0_cemu | 0 | 1 | 108.9 +- 0.7 | 9.2 +- 0.1 | 0.127 | 1.6 |
| SEP1_w0_c100 | 0 | 100 | 352.5 +- 2.8 | 2.8 +- 0.0 | 1.000 | 0.0 |
| SEP1_w0_cemu | 0 | 1 | 232.8 +- 3.6 | 4.3 +- 0.1 | 0.127 | 1.5 |
| SEP_w4_c100 | 4 | 100 | 510.5 +- 9.4 | 15.0 +- 0.2 | 1.000 | 2.9 |
| SEP_w4_cemu | 4 | 1 | 264.7 +- 3.8 | 22.3 +- 0.2 | 0.003 | 3.2 |
| STACK_w4_c100 | 4 | 100 | 381.7 +- 30.2 | 11.5 +- 0.4 | 1.000 | 1.4 |
| STACK_w4_cemu | 4 | 1 | 362.7 +- 2.4 | 12.2 +- 0.1 | 0.127 | 1.9 |
| SEP1_w4_c100 | 4 | 100 | 1013.4 +- 42.8 | 4.7 +- 0.1 | 1.000 | 1.6 |
| SEP1_w4_cemu | 4 | 1 | 777.5 +- 9.3 | 5.9 +- 0.1 | 0.127 | 1.7 |
| SEP_w0_cemu | 0 | 1 | 80.1 +- 15.2 | 18.2 +- 2.4 | 0.003 | 2.6 |

Errors: [('SEP_w0_cemu', ', in __call__\n    return self._op(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/work/venv/lib/python3.12/site-packages/fsspec/implementations/local.py", line 479, in seek\n    return self.f.seek(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nValueError: seek of closed file\n'), ('SEP_w0_cemu', ', in __call__\n    return self._op(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/work/venv/lib/python3.12/site-packages/fsspec/implementations/local.py", line 479, in seek\n    return self.f.seek(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nValueError: seek of closed file\n')]

## Extrapolation inputs

- Real video files per camera 2488; real parquet data files 3576; real median frames per file 38615 vs slice 31115.
- Map-style, full size: each worker's cache holds 100 decoders; random access over SEP 7464 files vs STACK 2488 files -> expected hit rate about 0.013 (SEP) vs 0.040 (STACK). The cemu arms measure this ratio on the slice.
- Streaming, lerobot-train (max_num_shards = num_workers, every worker iterates num_shards = min(P, W) shards; datasets then splits each shard's files across the W workers again, so worker w reads only the shards with more than w files). Decoders needed per worker = active shards x cameras (SEP) or active shards (STACK):

| num_workers W | workers that read | active shards: worker 0 / mean | SEP decoders (worker 0) | STACK decoders (worker 0) | SEP overflows 100 | STACK overflows 100 |
|---|---|---|---|---|---|---|
| 4 | 4 | 4 / 4.0 | 12 | 4 | False | False |
| 8 | 8 | 8 / 8.0 | 24 | 8 | False | False |
| 16 | 16 | 16 / 16.0 | 48 | 16 | False | False |
| 32 | 32 | 32 / 32.0 | 96 | 32 | False | False |
| 34 | 34 | 34 / 34.0 | 102 | 34 | True | False |
| 64 | 56 | 64 / 55.9 | 192 | 64 | True | False |
| 101 | 36 | 101 / 35.4 | 303 | 101 | True | True |
| 128 | 28 | 128 / 27.9 | 384 | 128 | True | True |

Worker 0 always reads min(P, W) shards, so the rule 'SEP overflows when min(P, W) x cameras > 100, STACK when min(P, W) > 100' holds for the busiest worker. Other workers read fewer shards, and all W workers carry the full load only when P >= W^2 (every shard has at least W files); with P < W some workers read nothing. The audit above tests this split on the SEP copy.

## Limits

- One group per dataset, copied 8 times: files are identical copies, so file-to-file variation (content, sizes) is not sampled.
- Streaming arms are one process (one DataLoader worker); the overflow case is emulated with cache 12 at S = 8, not run with 64 workers.
- Map-style arms read local disk (page cache warm after the first pass); they measure decode CPU, not network or cold-disk reads.
- STACK and SEP encodes share one loop, so encode time per layout is not measured separately.
- State / action values are random; only their shapes match the source.
- cpu-upgrade box (8 CPUs, 32 GB) shared with other stages' background uploads; compare layouts only inside this run.

