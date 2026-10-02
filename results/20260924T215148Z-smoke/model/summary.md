# DATA-11 benchmark model (EXTRAPOLATED from a scaled-down benchmark (SMOKE: numbers are not results))

## Open-cost fits (ms per miss = construct + close + extra first-frame decode)

| layout | state | a ms | b us/frame | F range |
|---|---|---|---|---|
| SEP | warm | 0.430 | 0.2608 | [967.3333333333334, 5156.333333333333] |
| STACK | warm | 0.540 | 0.2128 | [967.3333333333334, 5156.333333333333] |
| BIG | warm | 0.943 | 0.0359 | [2902, 15469] |
| SEP+BIG | warm | 0.943 | 0.0452 | [967.3333333333334, 15469] |
| SEP | cold | 2.386 | 0.1555 | [967.3333333333334, 5156.333333333333] |
| STACK | cold | 2.576 | 0.1020 | [967.3333333333334, 5156.333333333333] |
| BIG | cold | 2.862 | 0.0900 | [2902, 15469] |
| SEP+BIG | cold | 2.593 | 0.1120 | [967.3333333333334, 15469] |

## Thrash sweep fits (worker-s per sample vs opens per sample)

| arm | pts | decode ms | loader open ms | isolated open ms | kappa |
|---|---|---|---|---|---|
| SEP|all|W4|warm | 2 | 3.958 | 0.938 | 1.037 | 0.905 |
| STACK|all|W4|warm | 2 | 3.952 | 1.287 | 0.981 | 1.312 |
| BIG|all|W4|warm | 2 | 3.965 | 1.512 | 1.224 | 1.235 |
| SEP|top|W4|warm | 2 | 1.740 | 1.133 | 1.037 | 1.093 |
| STACK|top|W4|warm | 2 | 3.372 | 0.588 | 0.981 | 0.599 |
| BIG|top|W4|warm | 2 | 1.535 | 0.000 | 1.224 | 0.0 |
| SEP|all|W4|cold | 2 | 6.471 | 1.084 | 2.825 | 0.384 |
| STACK|all|W4|cold | 2 | 5.034 | 0.522 | 2.787 | 0.187 |
| BIG|all|W4|cold | 2 | 6.830 | 1.565 | 3.289 | 0.476 |
| SEP|top|W4|cold | 2 | 2.277 | 2.349 | 2.825 | 0.831 |
| STACK|top|W4|cold | 2 | 4.151 | 2.322 | 2.787 | 0.833 |
| BIG|top|W4|cold | 2 | 2.462 | 0.000 | 3.289 | 0.0 |

Hit-rate model vs measured (thrash arms): max abs err 0.027, mean 0.005 over 24 arms.

## Validation on ckpt_5k

| arm | measured sps | pred sps | err % | err % (measured opens) | h meas / model |
|---|---|---|---|---|---|
| ckpt_5k|SEP|all|c1|4|warm | 619 | 559 | -9.6 | -9.6 | 0.000 / 0.000 |
| ckpt_5k|STACK|all|c1|4|warm | 1017 | 743 | -27.0 | -26.7 | 0.342 / 0.333 |
| ckpt_5k|BIG|all|c1|4|warm | 459 | 398 | -13.3 | -13.3 | 0.000 / 0.000 |
| ckpt_5k|BIG|all|call|4|warm | 1182 | 1009 | -14.6 | -14.6 | 1.000 / 1.000 |
| ckpt_5k|STACK|all|call|4|warm | 1251 | 1012 | -19.1 | -19.1 | 1.000 / 1.000 |
| ckpt_5k|SEP|all|call|4|warm | 1175 | 1011 | -14.0 | -14.0 | 1.000 / 1.000 |
| ckpt_5k|SEP|all|c1|4|cold | 403 | 395 | -1.9 | -1.9 | 0.000 / 0.000 |
| ckpt_5k|STACK|all|c1|4|cold | 854 | 738 | -13.6 | -13.7 | 0.316 / 0.333 |
| ckpt_5k|BIG|all|c1|4|cold | 378 | 308 | -18.5 | -18.5 | 0.000 / 0.000 |
| ckpt_5k|BIG|all|call|4|cold | 718 | 586 | -18.4 | -18.4 | 1.000 / 1.000 |
| ckpt_5k|STACK|all|call|4|cold | 893 | 795 | -11.0 | -11.0 | 1.000 / 1.000 |
| ckpt_5k|SEP|all|call|4|cold | 641 | 618 | -3.5 | -3.5 | 1.000 / 1.000 |

## Predictions (EXTRAPOLATED; cache 100 per worker)

| dataset | variant | pattern | W | state | h | opens/sample | pred sps | beyond F range |
|---|---|---|---|---|---|---|---|---|
| lerobot/abc_130k_v3_train | SEP (as stored) | all | 4 | warm | 0.003 | 2.99 | 420 | yes |
| lerobot/abc_130k_v3_train | STACK, same frames/file (files / 3) | all | 4 | warm | 0.008 | 0.99 | 348 | yes |
| lerobot/abc_130k_v3_train | STACK, same bytes/file (LeRobot size rollover) | all | 4 | warm | 0.003 | 1.00 | 572 |  |
| lerobot/abc_130k_v3_train | BIG (3x frames/file, files / 3) | all | 4 | warm | 0.008 | 2.98 | 203 | yes |
| lerobot/abc_130k_v3_train | SEP (as stored) | top | 4 | warm | 0.008 | 0.99 | 1007 | yes |
| lerobot/abc_130k_v3_train | STACK, same frames/file (files / 3) | top | 4 | warm | 0.008 | 0.99 | 587 | yes |
| lerobot/abc_130k_v3_train | STACK, same bytes/file (LeRobot size rollover) | top | 4 | warm | 0.003 | 1.00 | 841 |  |
| lerobot/abc_130k_v3_train | BIG (3x frames/file, files / 3) | top | 4 | warm | 0.024 | 0.98 | 699 | yes |
| lerobot/abc_130k_v3_train | SEP (as stored) | all | 4 | cold | 0.003 | 2.99 | 317 | yes |
| lerobot/abc_130k_v3_train | STACK, same frames/file (files / 3) | all | 4 | cold | 0.008 | 0.99 | 669 | yes |
| lerobot/abc_130k_v3_train | STACK, same bytes/file (LeRobot size rollover) | all | 4 | cold | 0.003 | 1.00 | 705 |  |
| lerobot/abc_130k_v3_train | BIG (3x frames/file, files / 3) | all | 4 | cold | 0.008 | 2.98 | 180 | yes |
| lerobot/abc_130k_v3_train | SEP (as stored) | top | 4 | cold | 0.008 | 0.99 | 598 | yes |
| lerobot/abc_130k_v3_train | STACK, same frames/file (files / 3) | top | 4 | cold | 0.008 | 0.99 | 479 | yes |
| lerobot/abc_130k_v3_train | STACK, same bytes/file (LeRobot size rollover) | top | 4 | cold | 0.003 | 1.00 | 571 |  |
| lerobot/abc_130k_v3_train | BIG (3x frames/file, files / 3) | top | 4 | cold | 0.024 | 0.98 | 306 | yes |
| lerobot/droid_1.0.1 | SEP (as stored) | all | 4 | warm | 0.122 | 2.63 | 225 | yes |
| lerobot/droid_1.0.1 | STACK, same frames/file (files / 3) | all | 4 | warm | 0.369 | 0.63 | 174 | yes |
| lerobot/droid_1.0.1 | STACK, same bytes/file (LeRobot size rollover) | all | 4 | warm | 0.125 | 0.87 | 294 | yes |
| lerobot/droid_1.0.1 | BIG (3x frames/file, files / 3) | all | 4 | warm | 0.365 | 1.90 | 102 | yes |
| lerobot/droid_1.0.1 | SEP (as stored) | top | 4 | warm | 0.369 | 0.63 | 686 | yes |
| lerobot/droid_1.0.1 | STACK, same frames/file (files / 3) | top | 4 | warm | 0.369 | 0.63 | 326 | yes |
| lerobot/droid_1.0.1 | STACK, same bytes/file (LeRobot size rollover) | top | 4 | warm | 0.125 | 0.87 | 499 | yes |
| lerobot/droid_1.0.1 | BIG (3x frames/file, files / 3) | top | 4 | warm | 1.000 | 0.00 | 2270 | yes |
| lerobot/droid_1.0.1 | SEP (as stored) | all | 4 | cold | 0.122 | 2.63 | 185 | yes |
| lerobot/droid_1.0.1 | STACK, same frames/file (files / 3) | all | 4 | cold | 0.369 | 0.63 | 547 | yes |
| lerobot/droid_1.0.1 | STACK, same bytes/file (LeRobot size rollover) | all | 4 | cold | 0.125 | 0.87 | 590 | yes |
| lerobot/droid_1.0.1 | BIG (3x frames/file, files / 3) | all | 4 | cold | 0.365 | 1.90 | 97 | yes |
| lerobot/droid_1.0.1 | SEP (as stored) | top | 4 | cold | 0.369 | 0.63 | 402 | yes |
| lerobot/droid_1.0.1 | STACK, same frames/file (files / 3) | top | 4 | cold | 0.369 | 0.63 | 345 | yes |
| lerobot/droid_1.0.1 | STACK, same bytes/file (LeRobot size rollover) | top | 4 | cold | 0.125 | 0.87 | 434 | yes |
| lerobot/droid_1.0.1 | BIG (3x frames/file, files / 3) | top | 4 | cold | 1.000 | 0.00 | 1415 | yes |
| cadene/droid_1.0.1_v30 | SEP (as stored) | all | 4 | warm | 0.016 | 2.95 | 460 |  |
| cadene/droid_1.0.1_v30 | STACK, same frames/file (files / 3) | all | 4 | warm | 0.049 | 0.95 | 455 |  |
| cadene/droid_1.0.1_v30 | STACK, same bytes/file (LeRobot size rollover) | all | 4 | warm | 0.017 | 0.98 | 616 |  |
| cadene/droid_1.0.1_v30 | BIG (3x frames/file, files / 3) | all | 4 | warm | 0.048 | 2.85 | 279 | yes |
| cadene/droid_1.0.1_v30 | SEP (as stored) | top | 4 | warm | 0.049 | 0.95 | 1108 |  |
| cadene/droid_1.0.1_v30 | STACK, same frames/file (files / 3) | top | 4 | warm | 0.049 | 0.95 | 688 |  |
| cadene/droid_1.0.1_v30 | STACK, same bytes/file (LeRobot size rollover) | top | 4 | warm | 0.017 | 0.98 | 840 |  |
| cadene/droid_1.0.1_v30 | BIG (3x frames/file, files / 3) | top | 4 | warm | 0.146 | 0.85 | 969 | yes |
| cadene/droid_1.0.1_v30 | SEP (as stored) | all | 4 | cold | 0.016 | 2.95 | 331 |  |
| cadene/droid_1.0.1_v30 | STACK, same frames/file (files / 3) | all | 4 | cold | 0.049 | 0.95 | 617 |  |
| cadene/droid_1.0.1_v30 | STACK, same bytes/file (LeRobot size rollover) | all | 4 | cold | 0.017 | 0.98 | 631 |  |
| cadene/droid_1.0.1_v30 | BIG (3x frames/file, files / 3) | all | 4 | cold | 0.048 | 2.85 | 228 | yes |
| cadene/droid_1.0.1_v30 | SEP (as stored) | top | 4 | cold | 0.049 | 0.95 | 683 |  |
| cadene/droid_1.0.1_v30 | STACK, same frames/file (files / 3) | top | 4 | cold | 0.049 | 0.95 | 507 |  |
| cadene/droid_1.0.1_v30 | STACK, same bytes/file (LeRobot size rollover) | top | 4 | cold | 0.017 | 0.98 | 551 |  |
| cadene/droid_1.0.1_v30 | BIG (3x frames/file, files / 3) | top | 4 | cold | 0.146 | 0.85 | 449 | yes |
| control: 2 files per camera | SEP (as stored) | all | 4 | warm | 1.000 | 0.00 | 1011 | yes |
| control: 2 files per camera | STACK, same frames/file (files / 3) | all | 4 | warm | 1.000 | 0.00 | 1012 | yes |
| control: 2 files per camera | STACK, same bytes/file (LeRobot size rollover) | all | 4 | warm | 1.000 | 0.00 | 1012 |  |
| control: 2 files per camera | BIG (3x frames/file, files / 3) | all | 4 | warm | 1.000 | 0.00 | 1009 | yes |
| control: 2 files per camera | SEP (as stored) | top | 4 | warm | 1.000 | 0.00 | 2299 | yes |
| control: 2 files per camera | STACK, same frames/file (files / 3) | top | 4 | warm | 1.000 | 0.00 | 1186 | yes |
| control: 2 files per camera | STACK, same bytes/file (LeRobot size rollover) | top | 4 | warm | 1.000 | 0.00 | 1186 |  |
| control: 2 files per camera | BIG (3x frames/file, files / 3) | top | 4 | warm | 1.000 | 0.00 | 2605 | yes |
| control: 2 files per camera | SEP (as stored) | all | 4 | cold | 1.000 | 0.00 | 618 | yes |
| control: 2 files per camera | STACK, same frames/file (files / 3) | all | 4 | cold | 1.000 | 0.00 | 795 | yes |
| control: 2 files per camera | STACK, same bytes/file (LeRobot size rollover) | all | 4 | cold | 1.000 | 0.00 | 795 |  |
| control: 2 files per camera | BIG (3x frames/file, files / 3) | all | 4 | cold | 1.000 | 0.00 | 586 | yes |
| control: 2 files per camera | SEP (as stored) | top | 4 | cold | 1.000 | 0.00 | 1757 | yes |
| control: 2 files per camera | STACK, same frames/file (files / 3) | top | 4 | cold | 1.000 | 0.00 | 964 | yes |
| control: 2 files per camera | STACK, same bytes/file (LeRobot size rollover) | top | 4 | cold | 1.000 | 0.00 | 964 |  |
| control: 2 files per camera | BIG (3x frames/file, files / 3) | top | 4 | cold | 1.000 | 0.00 | 1625 | yes |

## Assumptions

- LRU hit rate for independent uniform frame references: h = min(1, floor(C/k)/U); checked against the thrash sweep (hit_rate_check).
- Workers are CPU-bound and independent: samples/s = W / worker_s_per_sample; measured only for W in the loader config, on this flavor.
- open_cost(F) linear in frames per file, fitted on the isolated curve; scaled by kappa (in-loader / isolated open cost at the thrash length).
- decode_cost independent of frames per file (validated on the checkpoint family) and proportional to pixels when applied to 180x320 DROID.
- Open cost for DROID uses the 224x224 curve (index size depends on frames, not pixels); frames/file beyond the measured range is a linear extrapolation (flagged).
- STACK same-bytes variant: frames per file divided by (STACK bytes per stacked frame / SEP bytes per camera frame) = 2.947 (thrash family); file count multiplied by it.
- cold = posix_fadvise DONTNEED eviction on this container's local disk; real training storage (network FS, Hub streaming) is not modeled.
- Same encoder settings as the source (libsvtav1, crf 30, g 2); other codecs or GOPs are not covered.
