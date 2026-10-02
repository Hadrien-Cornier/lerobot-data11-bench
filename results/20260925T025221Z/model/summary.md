# DATA-11 benchmark model (EXTRAPOLATED from a scaled-down benchmark)

## Open-cost fits (ms per miss = construct + close + extra first-frame decode)

| layout | state | a ms | b us/frame | F range |
|---|---|---|---|---|
| SEP | warm | 1.021 | 0.0614 | [967.3333333333334, 24706] |
| STACK | warm | 1.389 | 0.0667 | [967.3333333333334, 24706] |
| BIG | warm | 1.095 | 0.0669 | [2902, 74118] |
| SEP+BIG | warm | 1.028 | 0.0674 | [967.3333333333334, 74118] |
| SEP | cold | 2.475 | 0.0978 | [967.3333333333334, 24706] |
| STACK | cold | 2.183 | 0.0986 | [967.3333333333334, 24706] |
| BIG | cold | 2.641 | 0.0779 | [2902, 74118] |
| SEP+BIG | cold | 2.611 | 0.0796 | [967.3333333333334, 74118] |

## Thrash sweep fits (worker-s per sample vs opens per sample)

| arm | pts | decode ms | loader open ms | isolated open ms | kappa |
|---|---|---|---|---|---|
| SEP|all|W8|warm | 3 | 8.720 | 1.082 | 1.164 | 0.929 |
| STACK|all|W8|warm | 3 | 7.468 | 0.521 | 1.525 | 0.342 |
| BIG|all|W8|warm | 3 | 8.224 | 1.807 | 1.438 | 1.256 |
| BIG|top|W8|warm | 3 | 3.479 | 2.328 | 1.438 | 1.619 |
| STACK|top|W8|warm | 3 | 6.297 | 1.500 | 1.525 | 0.984 |
| SEP|top|W8|warm | 3 | 3.391 | 2.949 | 1.164 | 2.533 |
| SEP|all|W8|cold | 3 | 14.413 | 3.513 | 2.772 | 1.267 |
| STACK|all|W8|cold | 3 | 10.749 | 3.306 | 2.383 | 1.388 |
| BIG|all|W8|cold | 3 | 13.895 | 4.153 | 3.096 | 1.342 |
| BIG|top|W8|cold | 3 | 5.526 | 3.863 | 3.096 | 1.248 |
| STACK|top|W8|cold | 3 | 8.835 | 3.675 | 2.383 | 1.542 |
| SEP|top|W8|cold | 3 | 5.593 | 3.275 | 2.772 | 1.181 |

Hit-rate model vs measured (thrash arms): max abs err 0.007, mean 0.001 over 36 arms.

## Validation on ckpt_full

| arm | measured sps | pred sps | err % | err % (measured opens) | h meas / model |
|---|---|---|---|---|---|
| ckpt_full|BIG|all|c1|8|warm | 260 | 259 | -0.3 | -0.3 | 0.000 / 0.000 |
| ckpt_full|STACK|all|c1|8|warm | 873 | 980 | +12.3 | +12.3 | 0.334 / 0.333 |
| ckpt_full|SEP|all|c1|8|warm | 406 | 493 | +21.5 | +21.5 | 0.000 / 0.000 |
| ckpt_full|SEP|all|call|8|warm | 1093 | 917 | -16.1 | -16.1 | 1.000 / 1.000 |
| ckpt_full|STACK|all|call|8|warm | 1127 | 1071 | -4.9 | -4.9 | 1.000 / 1.000 |
| ckpt_full|BIG|all|call|8|warm | 1095 | 973 | -11.1 | -11.1 | 1.000 / 1.000 |
| ckpt_full|SEP|top|c1|8|warm | 1235 | 1008 | -18.4 | -18.5 | 0.332 / 0.333 |
| ckpt_full|STACK|top|c1|8|warm | 705 | 965 | +36.8 | +37.2 | 0.340 / 0.333 |
| ckpt_full|BIG|top|c1|8|warm | 2067 | 2300 | +11.3 | +11.3 | 1.000 / 1.000 |
| ckpt_full|BIG|top|call|8|warm | 2120 | 2300 | +8.5 | +8.5 | 1.000 / 1.000 |
| ckpt_full|STACK|top|call|8|warm | 1336 | 1270 | -4.9 | -4.9 | 1.000 / 1.000 |
| ckpt_full|SEP|top|call|8|warm | 2275 | 2359 | +3.7 | +3.7 | 1.000 / 1.000 |
| ckpt_full|BIG|all|c1|8|cold | 191 | 166 | -13.0 | -13.0 | 0.000 / 0.000 |
| ckpt_full|STACK|all|c1|8|cold | 588 | 533 | -9.4 | -9.3 | 0.337 / 0.333 |
| ckpt_full|SEP|all|c1|8|cold | 287 | 251 | -12.5 | -12.5 | 0.000 / 0.000 |
| ckpt_full|SEP|all|call|8|cold | 601 | 555 | -7.6 | -7.6 | 1.000 / 1.000 |
| ckpt_full|STACK|all|call|8|cold | 763 | 744 | -2.5 | -2.5 | 1.000 / 1.000 |
| ckpt_full|BIG|all|call|8|cold | 600 | 576 | -4.1 | -4.1 | 1.000 / 1.000 |
| ckpt_full|SEP|top|c1|8|cold | 882 | 870 | -1.4 | -1.3 | 0.336 / 0.333 |
| ckpt_full|STACK|top|c1|8|cold | 616 | 589 | -4.3 | -4.4 | 0.333 / 0.333 |
| ckpt_full|BIG|top|c1|8|cold | 1391 | 1448 | +4.1 | +4.1 | 1.000 / 1.000 |
| ckpt_full|BIG|top|call|8|cold | 1460 | 1448 | -0.8 | -0.8 | 1.000 / 1.000 |
| ckpt_full|STACK|top|call|8|cold | 950 | 905 | -4.7 | -4.7 | 1.000 / 1.000 |
| ckpt_full|SEP|top|call|8|cold | 1421 | 1430 | +0.6 | +0.6 | 1.000 / 1.000 |

## Predictions (EXTRAPOLATED; cache 100 per worker)

| dataset | variant | pattern | W | state | h | opens/sample | pred sps | beyond F range |
|---|---|---|---|---|---|---|---|---|
| lerobot/abc_130k_v3_train | SEP (as stored) | all | 8 | warm | 0.003 | 2.99 | 494 |  |
| lerobot/abc_130k_v3_train | STACK, same frames/file (files / 3) | all | 8 | warm | 0.008 | 0.99 | 942 |  |
| lerobot/abc_130k_v3_train | STACK, same bytes/file (LeRobot size rollover) | all | 8 | warm | 0.003 | 1.00 | 984 |  |
| lerobot/abc_130k_v3_train | BIG (3x frames/file, files / 3) | all | 8 | warm | 0.008 | 2.98 | 261 |  |
| lerobot/abc_130k_v3_train | SEP (as stored) | top | 8 | warm | 0.008 | 0.99 | 788 |  |
| lerobot/abc_130k_v3_train | STACK, same frames/file (files / 3) | top | 8 | warm | 0.008 | 0.99 | 864 |  |
| lerobot/abc_130k_v3_train | STACK, same bytes/file (LeRobot size rollover) | top | 8 | warm | 0.003 | 1.00 | 974 |  |
| lerobot/abc_130k_v3_train | BIG (3x frames/file, files / 3) | top | 8 | warm | 0.024 | 0.98 | 616 |  |
| lerobot/abc_130k_v3_train | SEP (as stored) | all | 8 | cold | 0.003 | 2.99 | 252 |  |
| lerobot/abc_130k_v3_train | STACK, same frames/file (files / 3) | all | 8 | cold | 0.008 | 0.99 | 468 |  |
| lerobot/abc_130k_v3_train | STACK, same bytes/file (LeRobot size rollover) | all | 8 | cold | 0.003 | 1.00 | 536 |  |
| lerobot/abc_130k_v3_train | BIG (3x frames/file, files / 3) | all | 8 | cold | 0.008 | 2.98 | 167 |  |
| lerobot/abc_130k_v3_train | SEP (as stored) | top | 8 | cold | 0.008 | 0.99 | 730 |  |
| lerobot/abc_130k_v3_train | STACK, same frames/file (files / 3) | top | 8 | cold | 0.008 | 0.99 | 503 |  |
| lerobot/abc_130k_v3_train | STACK, same bytes/file (LeRobot size rollover) | top | 8 | cold | 0.003 | 1.00 | 594 |  |
| lerobot/abc_130k_v3_train | BIG (3x frames/file, files / 3) | top | 8 | cold | 0.024 | 0.98 | 504 |  |
| lerobot/droid_1.0.1 | SEP (as stored) | all | 8 | warm | 0.122 | 2.63 | 273 | yes |
| lerobot/droid_1.0.1 | STACK, same frames/file (files / 3) | all | 8 | warm | 0.369 | 0.63 | 774 | yes |
| lerobot/droid_1.0.1 | STACK, same bytes/file (LeRobot size rollover) | all | 8 | warm | 0.126 | 0.87 | 826 |  |
| lerobot/droid_1.0.1 | BIG (3x frames/file, files / 3) | all | 8 | warm | 0.365 | 1.90 | 131 | yes |
| lerobot/droid_1.0.1 | SEP (as stored) | top | 8 | warm | 0.369 | 0.63 | 484 | yes |
| lerobot/droid_1.0.1 | STACK, same frames/file (files / 3) | top | 8 | warm | 0.369 | 0.63 | 650 | yes |
| lerobot/droid_1.0.1 | STACK, same bytes/file (LeRobot size rollover) | top | 8 | warm | 0.126 | 0.87 | 768 |  |
| lerobot/droid_1.0.1 | BIG (3x frames/file, files / 3) | top | 8 | warm | 1.000 | 0.00 | 2003 | yes |
| lerobot/droid_1.0.1 | SEP (as stored) | all | 8 | cold | 0.122 | 2.63 | 153 | yes |
| lerobot/droid_1.0.1 | STACK, same frames/file (files / 3) | all | 8 | cold | 0.369 | 0.63 | 347 | yes |
| lerobot/droid_1.0.1 | STACK, same bytes/file (LeRobot size rollover) | all | 8 | cold | 0.126 | 0.87 | 418 |  |
| lerobot/droid_1.0.1 | BIG (3x frames/file, files / 3) | all | 8 | cold | 0.365 | 1.90 | 94 | yes |
| lerobot/droid_1.0.1 | SEP (as stored) | top | 8 | cold | 0.369 | 0.63 | 555 | yes |
| lerobot/droid_1.0.1 | STACK, same frames/file (files / 3) | top | 8 | cold | 0.369 | 0.63 | 363 | yes |
| lerobot/droid_1.0.1 | STACK, same bytes/file (LeRobot size rollover) | top | 8 | cold | 0.126 | 0.87 | 452 |  |
| lerobot/droid_1.0.1 | BIG (3x frames/file, files / 3) | top | 8 | cold | 1.000 | 0.00 | 1261 | yes |
| cadene/droid_1.0.1_v30 | SEP (as stored) | all | 8 | warm | 0.016 | 2.95 | 522 |  |
| cadene/droid_1.0.1_v30 | STACK, same frames/file (files / 3) | all | 8 | warm | 0.049 | 0.95 | 859 |  |
| cadene/droid_1.0.1_v30 | STACK, same bytes/file (LeRobot size rollover) | all | 8 | warm | 0.017 | 0.98 | 875 |  |
| cadene/droid_1.0.1_v30 | BIG (3x frames/file, files / 3) | all | 8 | warm | 0.048 | 2.85 | 349 |  |
| cadene/droid_1.0.1_v30 | SEP (as stored) | top | 8 | warm | 0.049 | 0.95 | 935 |  |
| cadene/droid_1.0.1_v30 | STACK, same frames/file (files / 3) | top | 8 | warm | 0.049 | 0.95 | 854 |  |
| cadene/droid_1.0.1_v30 | STACK, same bytes/file (LeRobot size rollover) | top | 8 | warm | 0.017 | 0.98 | 902 |  |
| cadene/droid_1.0.1_v30 | BIG (3x frames/file, files / 3) | top | 8 | warm | 0.146 | 0.85 | 872 |  |
| cadene/droid_1.0.1_v30 | SEP (as stored) | all | 8 | cold | 0.016 | 2.95 | 264 |  |
| cadene/droid_1.0.1_v30 | STACK, same frames/file (files / 3) | all | 8 | cold | 0.049 | 0.95 | 471 |  |
| cadene/droid_1.0.1_v30 | STACK, same bytes/file (LeRobot size rollover) | all | 8 | cold | 0.017 | 0.98 | 502 |  |
| cadene/droid_1.0.1_v30 | BIG (3x frames/file, files / 3) | all | 8 | cold | 0.048 | 2.85 | 209 |  |
| cadene/droid_1.0.1_v30 | SEP (as stored) | top | 8 | cold | 0.049 | 0.95 | 758 |  |
| cadene/droid_1.0.1_v30 | STACK, same frames/file (files / 3) | top | 8 | cold | 0.049 | 0.95 | 523 |  |
| cadene/droid_1.0.1_v30 | STACK, same bytes/file (LeRobot size rollover) | top | 8 | cold | 0.017 | 0.98 | 566 |  |
| cadene/droid_1.0.1_v30 | BIG (3x frames/file, files / 3) | top | 8 | cold | 0.146 | 0.85 | 637 |  |
| control: 2 files per camera | SEP (as stored) | all | 8 | warm | 1.000 | 0.00 | 917 |  |
| control: 2 files per camera | STACK, same frames/file (files / 3) | all | 8 | warm | 1.000 | 0.00 | 1071 |  |
| control: 2 files per camera | STACK, same bytes/file (LeRobot size rollover) | all | 8 | warm | 1.000 | 0.00 | 1071 |  |
| control: 2 files per camera | BIG (3x frames/file, files / 3) | all | 8 | warm | 1.000 | 0.00 | 973 |  |
| control: 2 files per camera | SEP (as stored) | top | 8 | warm | 1.000 | 0.00 | 2359 |  |
| control: 2 files per camera | STACK, same frames/file (files / 3) | top | 8 | warm | 1.000 | 0.00 | 1270 |  |
| control: 2 files per camera | STACK, same bytes/file (LeRobot size rollover) | top | 8 | warm | 1.000 | 0.00 | 1270 |  |
| control: 2 files per camera | BIG (3x frames/file, files / 3) | top | 8 | warm | 1.000 | 0.00 | 2300 |  |
| control: 2 files per camera | SEP (as stored) | all | 8 | cold | 1.000 | 0.00 | 555 |  |
| control: 2 files per camera | STACK, same frames/file (files / 3) | all | 8 | cold | 1.000 | 0.00 | 744 |  |
| control: 2 files per camera | STACK, same bytes/file (LeRobot size rollover) | all | 8 | cold | 1.000 | 0.00 | 744 |  |
| control: 2 files per camera | BIG (3x frames/file, files / 3) | all | 8 | cold | 1.000 | 0.00 | 576 |  |
| control: 2 files per camera | SEP (as stored) | top | 8 | cold | 1.000 | 0.00 | 1430 |  |
| control: 2 files per camera | STACK, same frames/file (files / 3) | top | 8 | cold | 1.000 | 0.00 | 905 |  |
| control: 2 files per camera | STACK, same bytes/file (LeRobot size rollover) | top | 8 | cold | 1.000 | 0.00 | 905 |  |
| control: 2 files per camera | BIG (3x frames/file, files / 3) | top | 8 | cold | 1.000 | 0.00 | 1448 |  |

## Assumptions

- LRU hit rate for independent uniform frame references: h = min(1, floor(C/k)/U); checked against the thrash sweep (hit_rate_check).
- Workers are CPU-bound and independent: samples/s = W / worker_s_per_sample; measured only for W in the loader config, on this flavor.
- open_cost(F) linear in frames per file, fitted on the isolated curve; scaled by kappa (in-loader / isolated open cost at the thrash length).
- decode_cost independent of frames per file (validated on the checkpoint family) and proportional to pixels when applied to 180x320 DROID.
- Open cost for DROID uses the 224x224 curve (index size depends on frames, not pixels); frames/file beyond the measured range is a linear extrapolation (flagged).
- STACK same-bytes variant: frames per file divided by (STACK bytes per stacked frame / SEP bytes per camera frame) = 2.940 (thrash family); file count multiplied by it.
- cold = posix_fadvise DONTNEED eviction on this container's local disk; real training storage (network FS, Hub streaming) is not modeled.
- Same encoder settings as the source (libsvtav1, crf 30, g 2); other codecs or GOPs are not covered.

# Remote (hf://) model (EXTRAPOLATED from a scaled-down benchmark)

Per-fetch fit on 12 single-reader arms: lat_ms = -4.8 + 158.0 x fetches/sample (R^2 0.9850616578458954). Requests per fetch 2.00, resolves per fetch 1.00, ms per request 79.0. Resolver limit 40.0/s per token (NOT reported to this job; using 40.0 (12000 per 300 s, read from a laptop probe on 2026-09-24)).

| open fit | a ms | b us/frame | F range | req/open | resolve/open | API/open | fetch/open |
|---|---|---|---|---|---|---|---|
| SEP+BIG | 156.7 | 0.843 | [967.3333333333334, 74118.0] | 2.1041666666666665 | 1.0 | 0.10416666666666667 | 1.0 |
| STACK | 138.8 | 1.882 | [967.3333333333334, 24706.0] | 2.125 | 1.0 | 0.125 | 1.0 |

| fetch-count check | file MB | measured fetches | model |
|---|---|---|---|
| open ckpt_1k|SEP | 2.1 | 1.00 | 1.00 |
| open ckpt_1k|STACK | 6.1 | 1.00 | 1.14 |
| open ckpt_1k|BIG | 6.2 | 1.12 | 1.16 |
| open ckpt_5k|SEP | 11.8 | 1.25 | 1.55 |
| open ckpt_5k|STACK | 34.4 | 2.00 | 1.85 |
| open ckpt_5k|BIG | 35.3 | 1.88 | 1.85 |
| open ckpt_full|SEP | 73.4 | 1.88 | 1.93 |
| open ckpt_full|STACK | 216.4 | 2.00 | 1.98 |
| open ckpt_full|BIG | 220.3 | 2.00 | 1.98 |
| steady steady|SEP|seq | 73.4 | 0.94 | 0.93 |
| steady steady|SEP|one | 73.4 | 0.95 | 0.93 |
| steady steady|STACK|seq | 216.4 | 0.97 | 0.98 |
| steady steady|STACK|one | 216.4 | 0.97 | 0.98 |
| steady steady|BIG|seq | 220.3 | 0.98 | 0.98 |
| steady steady|BIG|one | 220.3 | 0.99 | 0.98 |

## Remote predictions (EXTRAPOLATED unless marked ANALYTIC)

| dataset | variant | pattern | h | opens/smp | fetches/smp | resolves/smp | MB/smp | ms/smp 1 reader | sps 1 reader | sps cap (resolver limit) | beyond F range |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lerobot/abc_130k_v3_train | SEP (as stored) | all | 0.0026 | 2.99 | 5.78 | 5.78 | 30.3 | 908 | 1.10 | 6.9 |  |
| lerobot/abc_130k_v3_train | STACK, same frames/file (files / 3) | all | 0.0079 | 0.99 | 1.97 | 1.97 | 10.3 | 306 | 3.27 | 20.3 |  |
| lerobot/abc_130k_v3_train | BIG (3x frames/file, files / 3) | all | 0.0078 | 2.98 | 5.91 | 5.91 | 31.0 | 928 | 1.08 | 6.8 |  |
| lerobot/abc_130k_v3_train | SEP (as stored) | one | 0.0079 | 0.99 | 1.92 | 1.92 | 10.1 | 299 | 3.35 | 20.8 |  |
| lerobot/abc_130k_v3_train | STACK, same frames/file (files / 3) | one | 0.0079 | 0.99 | 1.97 | 1.97 | 10.3 | 306 | 3.27 | 20.3 |  |
| lerobot/abc_130k_v3_train | BIG (3x frames/file, files / 3) | one | 0.0236 | 0.98 | 1.95 | 1.95 | 10.2 | 304 | 3.29 | 20.5 |  |
| cadene/droid_1.0.1_v30 | SEP (as stored) | all | 0.0161 | 2.95 | 5.61 | 5.61 | 29.4 | 881 | 1.13 | 7.1 |  |
| cadene/droid_1.0.1_v30 | STACK, same frames/file (files / 3) | all | 0.0488 | 0.95 | 1.91 | 1.91 | 10.0 | 297 | 3.36 | 20.9 |  |
| cadene/droid_1.0.1_v30 | BIG (3x frames/file, files / 3) | all | 0.0483 | 2.85 | 5.74 | 5.74 | 30.1 | 902 | 1.11 | 7.0 |  |
| cadene/droid_1.0.1_v30 | SEP (as stored) | one | 0.0488 | 0.95 | 1.84 | 1.84 | 9.6 | 285 | 3.50 | 21.8 |  |
| cadene/droid_1.0.1_v30 | STACK, same frames/file (files / 3) | one | 0.0488 | 0.95 | 1.91 | 1.91 | 10.0 | 297 | 3.36 | 20.9 |  |
| cadene/droid_1.0.1_v30 | BIG (3x frames/file, files / 3) | one | 0.1465 | 0.85 | 1.82 | 1.82 | 9.5 | 282 | 3.55 | 22.0 |  |

| ANALYTIC sequential | variant | fetches/smp | sps cap (resolver limit) |
|---|---|---|---|
| lerobot/abc_130k_v3_train | SEP (as stored) | 0.00170 | 23514 |
| lerobot/abc_130k_v3_train | STACK | 0.00167 | 23942 |
| cadene/droid_1.0.1_v30 | SEP (as stored) | 0.00195 | 20483 |
| cadene/droid_1.0.1_v30 | STACK | 0.00192 | 20856 |

## Remote assumptions

- Access = uniform random frames (map-style shuffle) with a 100-decoder LRU per worker; LRU hit rate h = min(1, floor(C/k)/U).
- fsspec readahead cache holds one block of 5242880 bytes per open handle; a random frame needs a new block fetch with chance 1 - block/file bytes; an open costs one fetch (moov is at the start, faststart).
- Each block fetch = 1 resolve call (counts against the per-token 'resolvers' budget) + 1 CDN call; measured, see resolves_per_fetch.
- Latency per sample is linear in block fetches per sample (single reader, cameras one after another); fitted on this job's network path to the Hub.
- sps_cap_resolver_limit = resolver budget per second / resolves per sample: a per-token ceiling that no amount of hardware removes.
- DROID bytes per frame = ABC bytes per frame x pixel ratio (ASSUMED; not measured). Frames/file beyond the measured range are flagged.
- API calls per open (paths-info when the fsspec dircache misses) are an upper bound: a long-lived worker caches them.
- Sequential rows are analytic only (no measurement of in-order streaming reads here).

---

# DATA-11 remote (hf://) reads: family ckpt_full

Repo `hadriencornier/lerobot-data11-bench@93ad048ad8` (private). fsspec block 5242880 B, cache ReadAheadCache. Resolver limit used: 40.0/s per token (NOT reported to this job; using 40.0 (12000 per 300 s, read from a laptop probe on 2026-09-24)). Header budget at start: -1 of -1 per -1 s (-1 = header absent).

## Single stream (one reader)

| regime | layout | access | sps | +-CI | p50 ms | p95 ms | resolve/smp | CDN/smp | API/smp | MB/smp | dec MB/smp | open ms/smp | 429 | sps cap at resolver limit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| first | SEP | one | 3.88 | 0.18 | 256 | 336 | 1.95 | 1.95 | 0.00 | 10.09 | 0.28 | 128 | 0 | 20.5 |
| first | STACK | one | 3.06 | 0.15 | 320 | 478 | 1.98 | 1.98 | 0.00 | 10.37 | 0.28 | 134 | 0 | 20.2 |
| first | BIG | one | 3.79 | 0.42 | 250 | 371 | 1.98 | 1.98 | 0.00 | 10.33 | 0.58 | 125 | 0 | 20.2 |
| first | SEP | par | 2.41 | 0.18 | 397 | 612 | 5.81 | 5.81 | 0.00 | 29.82 | 0.84 | 571 | 0 | 6.9 |
| first | STACK | par | 3.12 | 0.23 | 310 | 453 | 1.98 | 1.98 | 0.00 | 10.37 | 0.28 | 129 | 0 | 20.2 |
| first | BIG | par | 2.41 | 0.43 | 401 | 550 | 5.94 | 5.94 | 0.00 | 30.86 | 1.74 | 550 | 0 | 6.7 |
| first | SEP | seq | 1.08 | 0.19 | 897 | 1332 | 5.81 | 5.81 | 0.00 | 29.82 | 0.84 | 466 | 0 | 6.9 |
| first | STACK | seq | 2.73 | 0.24 | 336 | 595 | 1.98 | 1.98 | 0.00 | 10.37 | 0.28 | 137 | 0 | 20.2 |
| first | BIG | seq | 1.08 | 0.33 | 859 | 1401 | 5.94 | 5.94 | 0.00 | 30.86 | 1.74 | 395 | 0 | 6.7 |
| steady | SEP | one | 8.01 | 0.54 | 121 | 185 | 0.95 | 0.95 | 0.00 | 4.97 | 0.07 | 0 | 0 | 42.1 |
| steady | STACK | one | 6.57 | 0.89 | 136 | 254 | 0.97 | 0.97 | 0.00 | 5.15 | 0.07 | 0 | 0 | 41.2 |
| steady | BIG | one | 7.98 | 0.59 | 119 | 169 | 0.99 | 0.99 | 0.00 | 5.19 | 0.07 | 0 | 0 | 40.5 |
| steady | SEP | par | 5.01 | 0.40 | 198 | 288 | 2.81 | 2.81 | 0.00 | 14.58 | 0.20 | 0 | 0 | 14.2 |
| steady | STACK | par | 6.13 | 1.24 | 147 | 272 | 0.97 | 0.97 | 0.00 | 5.15 | 0.07 | 0 | 0 | 41.2 |
| steady | BIG | par | 5.05 | 0.21 | 196 | 246 | 2.95 | 2.95 | 0.00 | 15.43 | 0.20 | 0 | 0 | 13.6 |
| steady | SEP | seq | 2.22 | 0.25 | 446 | 675 | 2.82 | 2.82 | 0.00 | 14.61 | 0.20 | 0 | 0 | 14.2 |
| steady | STACK | seq | 4.87 | 1.34 | 186 | 359 | 0.97 | 0.97 | 0.00 | 5.15 | 0.07 | 0 | 0 | 41.2 |
| steady | BIG | seq | 2.25 | 0.29 | 425 | 636 | 2.95 | 2.95 | 0.00 | 15.43 | 0.20 | 0 | 0 | 13.6 |

## Concurrent readers (processes, seq / all cameras)

| regime | layout | procs | sps | resolves/s | limit/s | MB/s | p50 ms | p95 ms | 429 | budget min seen |
|---|---|---|---|---|---|---|---|---|---|---|
| first | SEP | 8 | 9.0 | 51.4 | 40 | 265 | 904 | 1106 | 0 | None |
| first | STACK | 8 | 26.6 | 52.6 | 40 | 277 | 294 | 393 | 0 | None |
| first | BIG | 8 | 8.7 | 51.9 | 40 | 272 | 875 | 1220 | 0 | None |
| steady | SEP | 8 | 20.0 | 55.3 | 40 | 279 | 412 | 556 | 0 | None |
| steady | STACK | 8 | 61.2 | 59.6 | 40 | 312 | 128 | 173 | 0 | None |
| steady | BIG | 8 | 20.6 | 60.2 | 40 | 314 | 387 | 490 | 0 | None |

## Open cost over hf:// (fresh handle + decoder)

| family | layout | frames/file | MB/file | open ms | req/open | resolve/open | API/open | MB/open | first ms | fetch first | next ms | fetch next |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ckpt_1k | SEP | 967 | 2.1 | 104 | 2.1 | 1.0 | 0.1 | 2.10 | 1 | 0.0 | 1 | 0.0 |
| ckpt_1k | STACK | 967 | 6.1 | 125 | 2.1 | 1.0 | 0.1 | 5.12 | 3 | 0.0 | 3 | 0.1 |
| ckpt_1k | BIG | 2902 | 6.2 | 115 | 2.1 | 1.0 | 0.1 | 5.31 | 1 | 0.1 | 1 | 0.4 |
| ckpt_5k | SEP | 5156 | 11.8 | 198 | 2.1 | 1.0 | 0.1 | 5.31 | 1 | 0.2 | 153 | 0.9 |
| ckpt_5k | STACK | 5156 | 34.4 | 167 | 2.1 | 1.0 | 0.1 | 5.31 | 131 | 1.0 | 124 | 1.0 |
| ckpt_5k | BIG | 15469 | 35.3 | 127 | 2.1 | 1.0 | 0.1 | 5.31 | 112 | 0.9 | 128 | 1.0 |
| ckpt_full | SEP | 24706 | 73.4 | 319 | 2.0 | 1.0 | 0.0 | 5.31 | 243 | 0.9 | 111 | 0.8 |
| ckpt_full | STACK | 24706 | 216.4 | 182 | 2.1 | 1.0 | 0.1 | 5.31 | 185 | 1.0 | 278 | 1.0 |
| ckpt_full | BIG | 74118 | 220.3 | 181 | 2.1 | 1.0 | 0.1 | 5.31 | 229 | 1.0 | 252 | 1.0 |

## Checks

- remote decode == local decode (bit-exact): 36/36
- SEP == BIG remote samples: 12/12
- STACK crop vs SEP PSNR (separate lossy encodes): min 40.2 dB over 12 samples
- pts violations > 1e-4 s: 0 arms
- 429 responses in stage: 0; resolver budget start -1, end -1, min seen never (no ratelimit header seen)
- skipped arms: 0

