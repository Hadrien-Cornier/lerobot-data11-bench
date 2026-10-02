# DATA-11 job 20260925T023326Z-smoke

SMOKE=1. Source `lerobot/abc_130k_v3_train@68651e4929d9fb00f798937b2d62617cab5c771d`.

## Stages

```
stage	status	seconds	finished_utc	cgroup_peak_mb
system_deps	ok	47	2026-09-25T02:34:14Z	835
bundle	ok	0	2026-09-25T02:34:14Z	835
python_env	ok	41	2026-09-25T02:34:55Z	2243
fetch	ok	12	2026-09-25T02:35:20Z	3075
encode	ok	228	2026-09-25T02:39:09Z	5229
big	ok	25	2026-09-25T02:39:35Z	5229
verify	ok	231	2026-09-25T02:43:27Z	5229
upload	ok	11	2026-09-25T02:43:39Z	5229
remote	ok	253	2026-09-25T02:47:53Z	6865
opencost	ok	47	2026-09-25T02:48:41Z	6865
loader	ok	78	2026-09-25T02:50:00Z	6865
model	ok	0	2026-09-25T02:50:01Z	6865
```

---

# DATA-11 benchmark model (EXTRAPOLATED from a scaled-down benchmark (SMOKE: numbers are not results))

## Open-cost fits (ms per miss = construct + close + extra first-frame decode)

| layout | state | a ms | b us/frame | F range |
|---|---|---|---|---|
| SEP | warm | 0.563 | 0.1864 | [967.3333333333334, 5156.333333333333] |
| STACK | warm | 1.009 | 0.0864 | [967.3333333333334, 5156.333333333333] |
| BIG | warm | 0.896 | 0.0238 | [2902, 15469] |
| SEP+BIG | warm | 0.927 | 0.0283 | [967.3333333333334, 15469] |
| SEP | cold | 1.782 | 0.0830 | [967.3333333333334, 5156.333333333333] |
| STACK | cold | 2.218 | 0.0068 | [967.3333333333334, 5156.333333333333] |
| BIG | cold | 1.403 | 0.1256 | [2902, 15469] |
| SEP+BIG | cold | 1.635 | 0.1072 | [967.3333333333334, 15469] |

## Thrash sweep fits (worker-s per sample vs opens per sample)

| arm | pts | decode ms | loader open ms | isolated open ms | kappa |
|---|---|---|---|---|---|
| SEP|all|W4|warm | 2 | 5.073 | 0.965 | 0.985 | 0.979 |
| STACK|all|W4|warm | 2 | 4.848 | 0.236 | 1.188 | 0.199 |
| BIG|all|W4|warm | 2 | 4.996 | 1.524 | 1.102 | 1.382 |
| SEP|top|W4|warm | 2 | 2.328 | 1.012 | 0.985 | 1.027 |
| STACK|top|W4|warm | 2 | 3.896 | 1.919 | 1.188 | 1.615 |
| BIG|top|W4|warm | 2 | 2.035 | 0.000 | 1.102 | 0.0 |
| SEP|all|W4|cold | 2 | 9.406 | 1.919 | 1.857 | 1.034 |
| STACK|all|W4|cold | 2 | 6.750 | 1.137 | 2.232 | 0.509 |
| BIG|all|W4|cold | 2 | 7.634 | 1.883 | 2.301 | 0.818 |
| SEP|top|W4|cold | 2 | 3.410 | 2.600 | 1.857 | 1.4 |
| STACK|top|W4|cold | 2 | 5.341 | 2.121 | 2.232 | 0.95 |
| BIG|top|W4|cold | 2 | 3.189 | 0.000 | 2.301 | 0.0 |

Hit-rate model vs measured (thrash arms): max abs err 0.027, mean 0.005 over 24 arms.

## Validation on ckpt_5k

| arm | measured sps | pred sps | err % | err % (measured opens) | h meas / model |
|---|---|---|---|---|---|
| ckpt_5k|SEP|all|c1|4|warm | 504 | 486 | -3.5 | -3.5 | 0.000 / 0.000 |
| ckpt_5k|STACK|all|c1|4|warm | 781 | 794 | +1.6 | +1.6 | 0.342 / 0.333 |
| ckpt_5k|BIG|all|c1|4|warm | 346 | 375 | +8.4 | +8.4 | 0.000 / 0.000 |
| ckpt_5k|BIG|all|call|4|warm | 882 | 801 | -9.3 | -9.3 | 1.000 / 1.000 |
| ckpt_5k|STACK|all|call|4|warm | 1001 | 825 | -17.6 | -17.6 | 1.000 / 1.000 |
| ckpt_5k|SEP|all|call|4|warm | 839 | 788 | -6.0 | -6.0 | 1.000 / 1.000 |
| ckpt_5k|SEP|all|c1|4|cold | 297 | 247 | -16.8 | -16.8 | 0.000 / 0.000 |
| ckpt_5k|STACK|all|c1|4|cold | 568 | 532 | -6.3 | -6.6 | 0.316 / 0.333 |
| ckpt_5k|BIG|all|c1|4|cold | 257 | 254 | -1.1 | -1.1 | 0.000 / 0.000 |
| ckpt_5k|BIG|all|call|4|cold | 535 | 524 | -2.0 | -2.0 | 1.000 / 1.000 |
| ckpt_5k|STACK|all|call|4|cold | 726 | 593 | -18.4 | -18.4 | 1.000 / 1.000 |
| ckpt_5k|SEP|all|call|4|cold | 517 | 425 | -17.8 | -17.8 | 1.000 / 1.000 |

## Predictions (EXTRAPOLATED; cache 100 per worker)

| dataset | variant | pattern | W | state | h | opens/sample | pred sps | beyond F range |
|---|---|---|---|---|---|---|---|---|
| lerobot/abc_130k_v3_train | SEP (as stored) | all | 4 | warm | 0.003 | 2.99 | 407 | yes |
| lerobot/abc_130k_v3_train | STACK, same frames/file (files / 3) | all | 4 | warm | 0.008 | 0.99 | 732 | yes |
| lerobot/abc_130k_v3_train | STACK, same bytes/file (LeRobot size rollover) | all | 4 | warm | 0.003 | 1.00 | 771 |  |
| lerobot/abc_130k_v3_train | BIG (3x frames/file, files / 3) | all | 4 | warm | 0.008 | 2.98 | 230 | yes |
| lerobot/abc_130k_v3_train | SEP (as stored) | top | 4 | warm | 0.008 | 0.99 | 1004 | yes |
| lerobot/abc_130k_v3_train | STACK, same frames/file (files / 3) | top | 4 | warm | 0.008 | 0.99 | 448 | yes |
| lerobot/abc_130k_v3_train | STACK, same bytes/file (LeRobot size rollover) | top | 4 | warm | 0.003 | 1.00 | 598 |  |
| lerobot/abc_130k_v3_train | BIG (3x frames/file, files / 3) | top | 4 | warm | 0.024 | 0.98 | 803 | yes |
| lerobot/abc_130k_v3_train | SEP (as stored) | all | 4 | cold | 0.003 | 2.99 | 177 | yes |
| lerobot/abc_130k_v3_train | STACK, same frames/file (files / 3) | all | 4 | cold | 0.008 | 0.99 | 503 | yes |
| lerobot/abc_130k_v3_train | STACK, same bytes/file (LeRobot size rollover) | all | 4 | cold | 0.003 | 1.00 | 506 |  |
| lerobot/abc_130k_v3_train | BIG (3x frames/file, files / 3) | all | 4 | cold | 0.008 | 2.98 | 129 | yes |
| lerobot/abc_130k_v3_train | SEP (as stored) | top | 4 | cold | 0.008 | 0.99 | 428 | yes |
| lerobot/abc_130k_v3_train | STACK, same frames/file (files / 3) | top | 4 | cold | 0.008 | 0.99 | 527 | yes |
| lerobot/abc_130k_v3_train | STACK, same bytes/file (LeRobot size rollover) | top | 4 | cold | 0.003 | 1.00 | 534 |  |
| lerobot/abc_130k_v3_train | BIG (3x frames/file, files / 3) | top | 4 | cold | 0.024 | 0.98 | 319 | yes |
| lerobot/droid_1.0.1 | SEP (as stored) | all | 4 | warm | 0.122 | 2.63 | 256 | yes |
| lerobot/droid_1.0.1 | STACK, same frames/file (files / 3) | all | 4 | warm | 0.369 | 0.63 | 588 | yes |
| lerobot/droid_1.0.1 | STACK, same bytes/file (LeRobot size rollover) | all | 4 | warm | 0.125 | 0.87 | 639 | yes |
| lerobot/droid_1.0.1 | BIG (3x frames/file, files / 3) | all | 4 | warm | 0.365 | 1.90 | 129 | yes |
| lerobot/droid_1.0.1 | SEP (as stored) | top | 4 | warm | 0.369 | 0.63 | 778 | yes |
| lerobot/droid_1.0.1 | STACK, same frames/file (files / 3) | top | 4 | warm | 0.369 | 0.63 | 276 | yes |
| lerobot/droid_1.0.1 | STACK, same bytes/file (LeRobot size rollover) | top | 4 | warm | 0.125 | 0.87 | 395 | yes |
| lerobot/droid_1.0.1 | BIG (3x frames/file, files / 3) | top | 4 | warm | 1.000 | 0.00 | 1713 | yes |
| lerobot/droid_1.0.1 | SEP (as stored) | all | 4 | cold | 0.122 | 2.63 | 89 | yes |
| lerobot/droid_1.0.1 | STACK, same frames/file (files / 3) | all | 4 | cold | 0.369 | 0.63 | 461 | yes |
| lerobot/droid_1.0.1 | STACK, same bytes/file (LeRobot size rollover) | all | 4 | cold | 0.125 | 0.87 | 452 | yes |
| lerobot/droid_1.0.1 | BIG (3x frames/file, files / 3) | all | 4 | cold | 0.365 | 1.90 | 64 | yes |
| lerobot/droid_1.0.1 | SEP (as stored) | top | 4 | cold | 0.369 | 0.63 | 266 | yes |
| lerobot/droid_1.0.1 | STACK, same frames/file (files / 3) | top | 4 | cold | 0.369 | 0.63 | 508 | yes |
| lerobot/droid_1.0.1 | STACK, same bytes/file (LeRobot size rollover) | top | 4 | cold | 0.125 | 0.87 | 490 | yes |
| lerobot/droid_1.0.1 | BIG (3x frames/file, files / 3) | top | 4 | cold | 1.000 | 0.00 | 1093 | yes |
| cadene/droid_1.0.1_v30 | SEP (as stored) | all | 4 | warm | 0.016 | 2.95 | 417 |  |
| cadene/droid_1.0.1_v30 | STACK, same frames/file (files / 3) | all | 4 | warm | 0.049 | 0.95 | 669 |  |
| cadene/droid_1.0.1_v30 | STACK, same bytes/file (LeRobot size rollover) | all | 4 | warm | 0.017 | 0.98 | 685 |  |
| cadene/droid_1.0.1_v30 | BIG (3x frames/file, files / 3) | all | 4 | warm | 0.048 | 2.85 | 288 | yes |
| cadene/droid_1.0.1_v30 | SEP (as stored) | top | 4 | warm | 0.049 | 0.95 | 1013 |  |
| cadene/droid_1.0.1_v30 | STACK, same frames/file (files / 3) | top | 4 | warm | 0.049 | 0.95 | 512 |  |
| cadene/droid_1.0.1_v30 | STACK, same bytes/file (LeRobot size rollover) | top | 4 | warm | 0.017 | 0.98 | 597 |  |
| cadene/droid_1.0.1_v30 | BIG (3x frames/file, files / 3) | top | 4 | warm | 0.146 | 0.85 | 975 | yes |
| cadene/droid_1.0.1_v30 | SEP (as stored) | all | 4 | cold | 0.016 | 2.95 | 198 |  |
| cadene/droid_1.0.1_v30 | STACK, same frames/file (files / 3) | all | 4 | cold | 0.049 | 0.95 | 451 |  |
| cadene/droid_1.0.1_v30 | STACK, same bytes/file (LeRobot size rollover) | all | 4 | cold | 0.017 | 0.98 | 451 |  |
| cadene/droid_1.0.1_v30 | BIG (3x frames/file, files / 3) | all | 4 | cold | 0.048 | 2.85 | 176 | yes |
| cadene/droid_1.0.1_v30 | SEP (as stored) | top | 4 | cold | 0.049 | 0.95 | 499 |  |
| cadene/droid_1.0.1_v30 | STACK, same frames/file (files / 3) | top | 4 | cold | 0.049 | 0.95 | 487 |  |
| cadene/droid_1.0.1_v30 | STACK, same bytes/file (LeRobot size rollover) | top | 4 | cold | 0.017 | 0.98 | 486 |  |
| cadene/droid_1.0.1_v30 | BIG (3x frames/file, files / 3) | top | 4 | cold | 0.146 | 0.85 | 457 | yes |
| control: 2 files per camera | SEP (as stored) | all | 4 | warm | 1.000 | 0.00 | 788 | yes |
| control: 2 files per camera | STACK, same frames/file (files / 3) | all | 4 | warm | 1.000 | 0.00 | 825 | yes |
| control: 2 files per camera | STACK, same bytes/file (LeRobot size rollover) | all | 4 | warm | 1.000 | 0.00 | 825 |  |
| control: 2 files per camera | BIG (3x frames/file, files / 3) | all | 4 | warm | 1.000 | 0.00 | 801 | yes |
| control: 2 files per camera | SEP (as stored) | top | 4 | warm | 1.000 | 0.00 | 1718 | yes |
| control: 2 files per camera | STACK, same frames/file (files / 3) | top | 4 | warm | 1.000 | 0.00 | 1027 | yes |
| control: 2 files per camera | STACK, same bytes/file (LeRobot size rollover) | top | 4 | warm | 1.000 | 0.00 | 1027 |  |
| control: 2 files per camera | BIG (3x frames/file, files / 3) | top | 4 | warm | 1.000 | 0.00 | 1966 | yes |
| control: 2 files per camera | SEP (as stored) | all | 4 | cold | 1.000 | 0.00 | 425 | yes |
| control: 2 files per camera | STACK, same frames/file (files / 3) | all | 4 | cold | 1.000 | 0.00 | 593 | yes |
| control: 2 files per camera | STACK, same bytes/file (LeRobot size rollover) | all | 4 | cold | 1.000 | 0.00 | 593 |  |
| control: 2 files per camera | BIG (3x frames/file, files / 3) | all | 4 | cold | 1.000 | 0.00 | 524 | yes |
| control: 2 files per camera | SEP (as stored) | top | 4 | cold | 1.000 | 0.00 | 1173 | yes |
| control: 2 files per camera | STACK, same frames/file (files / 3) | top | 4 | cold | 1.000 | 0.00 | 749 | yes |
| control: 2 files per camera | STACK, same bytes/file (LeRobot size rollover) | top | 4 | cold | 1.000 | 0.00 | 749 |  |
| control: 2 files per camera | BIG (3x frames/file, files / 3) | top | 4 | cold | 1.000 | 0.00 | 1254 | yes |

## Assumptions

- LRU hit rate for independent uniform frame references: h = min(1, floor(C/k)/U); checked against the thrash sweep (hit_rate_check).
- Workers are CPU-bound and independent: samples/s = W / worker_s_per_sample; measured only for W in the loader config, on this flavor.
- open_cost(F) linear in frames per file, fitted on the isolated curve; scaled by kappa (in-loader / isolated open cost at the thrash length).
- decode_cost independent of frames per file (validated on the checkpoint family) and proportional to pixels when applied to 180x320 DROID.
- Open cost for DROID uses the 224x224 curve (index size depends on frames, not pixels); frames/file beyond the measured range is a linear extrapolation (flagged).
- STACK same-bytes variant: frames per file divided by (STACK bytes per stacked frame / SEP bytes per camera frame) = 2.947 (thrash family); file count multiplied by it.
- cold = posix_fadvise DONTNEED eviction on this container's local disk; real training storage (network FS, Hub streaming) is not modeled.
- Same encoder settings as the source (libsvtav1, crf 30, g 2); other codecs or GOPs are not covered.

# Remote (hf://) model (EXTRAPOLATED from a scaled-down benchmark (SMOKE: numbers are not results))

Per-fetch fit on 12 single-reader arms: lat_ms = -0.1 + 131.5 x fetches/sample (R^2 0.9846369867337748). Requests per fetch 2.00, resolves per fetch 1.00, ms per request 65.7. Resolver limit nan/s per token.

| open fit | a ms | b us/frame | F range | req/open | resolve/open | API/open | fetch/open |
|---|---|---|---|---|---|---|---|
| SEP+BIG | 111.8 | 3.425 | [967.3333333333334, 15469.0] | 2.25 | 1.0 | 0.25 | 1.0 |
| STACK | 78.8 | 45.283 | [967.3333333333334, 5156.333333333333] | 2.3333333333333335 | 1.0 | 0.3333333333333333 | 1.0 |

| fetch-count check | file MB | measured fetches | model |
|---|---|---|---|
| open ckpt_1k|SEP | 2.1 | 1.00 | 1.00 |
| open ckpt_1k|STACK | 6.1 | 1.00 | 1.14 |
| open ckpt_1k|BIG | 6.2 | 1.33 | 1.16 |
| open ckpt_5k|SEP | 11.8 | 1.33 | 1.55 |
| open ckpt_5k|STACK | 34.4 | 1.67 | 1.85 |
| open ckpt_5k|BIG | 35.3 | 1.33 | 1.85 |
| steady steady|SEP|seq | 11.8 | 0.50 | 0.55 |
| steady steady|SEP|one | 11.8 | 0.50 | 0.55 |
| steady steady|STACK|seq | 34.4 | 0.75 | 0.85 |
| steady steady|STACK|one | 34.4 | 0.75 | 0.85 |
| steady steady|BIG|seq | 35.3 | 0.89 | 0.85 |
| steady steady|BIG|one | 35.3 | 1.00 | 0.85 |

## Remote predictions (EXTRAPOLATED unless marked ANALYTIC)

| dataset | variant | pattern | h | opens/smp | fetches/smp | resolves/smp | MB/smp | ms/smp 1 reader | sps 1 reader | sps cap (resolver limit) | beyond F range |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lerobot/abc_130k_v3_train | SEP (as stored) | all | 0.0026 | 2.99 | 5.71 | 5.71 | 30.0 | 751 | 1.33 | nan | yes |
| lerobot/abc_130k_v3_train | STACK, same frames/file (files / 3) | all | 0.0079 | 0.99 | 1.96 | 1.96 | 10.3 | 258 | 3.88 | nan | yes |
| lerobot/abc_130k_v3_train | BIG (3x frames/file, files / 3) | all | 0.0078 | 2.98 | 5.88 | 5.88 | 30.8 | 773 | 1.29 | nan | yes |
| lerobot/abc_130k_v3_train | SEP (as stored) | one | 0.0079 | 0.99 | 1.90 | 1.90 | 10.0 | 250 | 4.01 | nan | yes |
| lerobot/abc_130k_v3_train | STACK, same frames/file (files / 3) | one | 0.0079 | 0.99 | 1.96 | 1.96 | 10.3 | 258 | 3.88 | nan | yes |
| lerobot/abc_130k_v3_train | BIG (3x frames/file, files / 3) | one | 0.0236 | 0.98 | 1.95 | 1.95 | 10.2 | 256 | 3.91 | nan | yes |
| cadene/droid_1.0.1_v30 | SEP (as stored) | all | 0.0161 | 2.95 | 5.51 | 5.51 | 28.9 | 724 | 1.38 | nan |  |
| cadene/droid_1.0.1_v30 | STACK, same frames/file (files / 3) | all | 0.0488 | 0.95 | 1.90 | 1.90 | 10.0 | 250 | 4.00 | nan |  |
| cadene/droid_1.0.1_v30 | BIG (3x frames/file, files / 3) | all | 0.0483 | 2.85 | 5.71 | 5.71 | 29.9 | 750 | 1.33 | nan | yes |
| cadene/droid_1.0.1_v30 | SEP (as stored) | one | 0.0488 | 0.95 | 1.80 | 1.80 | 9.5 | 237 | 4.22 | nan |  |
| cadene/droid_1.0.1_v30 | STACK, same frames/file (files / 3) | one | 0.0488 | 0.95 | 1.90 | 1.90 | 10.0 | 250 | 4.00 | nan |  |
| cadene/droid_1.0.1_v30 | BIG (3x frames/file, files / 3) | one | 0.1465 | 0.85 | 1.80 | 1.80 | 9.5 | 237 | 4.22 | nan | yes |

| ANALYTIC sequential | variant | fetches/smp | sps cap (resolver limit) |
|---|---|---|---|
| lerobot/abc_130k_v3_train | SEP (as stored) | 0.00131 | nan |
| lerobot/abc_130k_v3_train | STACK | 0.00127 | nan |
| cadene/droid_1.0.1_v30 | SEP (as stored) | 0.00150 | nan |
| cadene/droid_1.0.1_v30 | STACK | 0.00146 | nan |

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

# DATA-11 remote (hf://) reads: family ckpt_5k

Repo `hadriencornier/lerobot-data11-bench@59b48590d1` (private). fsspec block 5242880 B, cache ReadAheadCache. Resolver limit -1 per -1 s = nan/s per token.

## Single stream (one reader)

| regime | layout | access | sps | +-CI | p50 ms | p95 ms | resolve/smp | CDN/smp | API/smp | MB/smp | dec MB/smp | open ms/smp | 429 | sps cap at resolver limit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| first | SEP | one | 7.47 | - | 117 | 209 | 1.17 | 1.17 | 0.00 | 6.19 | 0.13 | 114 | 0 | nan |
| first | STACK | one | 4.12 | - | 261 | 290 | 1.83 | 1.83 | 0.00 | 9.73 | 0.13 | 125 | 0 | nan |
| first | BIG | one | 4.57 | - | 235 | 259 | 1.83 | 1.83 | 0.00 | 9.73 | 0.20 | 117 | 0 | nan |
| first | SEP | par | 4.91 | - | 181 | 279 | 3.17 | 3.17 | 0.00 | 16.81 | 0.39 | 504 | 0 | nan |
| first | STACK | par | 4.70 | - | 228 | 256 | 1.83 | 1.83 | 0.00 | 9.73 | 0.13 | 111 | 0 | nan |
| first | BIG | par | 2.78 | - | 382 | 455 | 5.50 | 5.50 | 0.00 | 28.87 | 0.59 | 514 | 0 | nan |
| first | SEP | seq | 2.54 | - | 381 | 470 | 3.17 | 3.17 | 0.00 | 16.81 | 0.39 | 367 | 0 | nan |
| first | STACK | seq | 3.54 | - | 314 | 370 | 1.83 | 1.83 | 0.00 | 9.73 | 0.13 | 161 | 0 | nan |
| first | BIG | seq | 1.40 | - | 727 | 923 | 5.50 | 5.50 | 0.00 | 28.87 | 0.59 | 419 | 0 | nan |
| steady | SEP | one | 15.80 | - | 53 | 145 | 0.50 | 0.50 | 0.00 | 2.52 | 0.07 | 0 | 0 | nan |
| steady | STACK | one | 12.01 | - | 108 | 116 | 0.75 | 0.75 | 0.00 | 3.98 | 0.07 | 0 | 0 | nan |
| steady | BIG | one | 8.35 | - | 114 | 145 | 1.00 | 1.00 | 0.00 | 5.31 | 0.07 | 0 | 0 | nan |
| steady | SEP | par | 13.27 | - | 68 | 167 | 1.25 | 1.25 | 0.00 | 5.92 | 0.20 | 0 | 0 | nan |
| steady | STACK | par | 9.64 | - | 108 | 206 | 0.75 | 0.75 | 0.00 | 3.98 | 0.07 | 0 | 0 | nan |
| steady | BIG | par | 6.00 | - | 170 | 186 | 2.67 | 2.67 | 0.00 | 14.16 | 0.20 | 0 | 0 | nan |
| steady | SEP | seq | 5.33 | - | 200 | 407 | 1.50 | 1.50 | 0.00 | 7.30 | 0.20 | 0 | 0 | nan |
| steady | STACK | seq | 8.13 | - | 109 | 266 | 0.75 | 0.75 | 0.00 | 3.98 | 0.07 | 0 | 0 | nan |
| steady | BIG | seq | 2.57 | - | 368 | 618 | 2.67 | 2.67 | 0.00 | 14.16 | 0.20 | 0 | 0 | nan |

## Concurrent readers (processes, seq / all cameras)

| regime | layout | procs | sps | resolves/s | limit/s | MB/s | p50 ms | p95 ms | 429 | budget min seen |
|---|---|---|---|---|---|---|---|---|---|---|
| first | SEP | 8 | 13.8 | 65.1 | nan | 286 | 597 | 873 | 0 | None |
| first | STACK | 8 | 35.4 | 64.7 | nan | 325 | 230 | 297 | 0 | None |
| first | BIG | 8 | 11.0 | 63.6 | nan | 315 | 732 | 867 | 0 | None |
| steady | SEP | 8 | 39.0 | 72.8 | nan | 296 | 254 | 385 | 0 | None |
| steady | STACK | 8 | 74.8 | 65.4 | nan | 314 | 116 | 158 | 0 | None |
| steady | BIG | 8 | 24.1 | 60.5 | nan | 286 | 377 | 491 | 0 | None |

## Open cost over hf:// (fresh handle + decoder)

| family | layout | frames/file | MB/file | open ms | req/open | resolve/open | API/open | MB/open | first ms | fetch first | next ms | fetch next |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ckpt_1k | SEP | 967 | 2.1 | 86 | 2.3 | 1.0 | 0.3 | 1.66 | 1 | 0.0 | 1 | 0.0 |
| ckpt_1k | STACK | 967 | 6.1 | 123 | 2.3 | 1.0 | 0.3 | 5.15 | 2 | 0.0 | 2 | 0.0 |
| ckpt_1k | BIG | 2902 | 6.2 | 130 | 2.3 | 1.0 | 0.3 | 5.31 | 1 | 0.3 | 1 | 0.3 |
| ckpt_5k | SEP | 5156 | 11.8 | 160 | 2.0 | 1.0 | 0.0 | 5.31 | 1 | 0.3 | 102 | 1.0 |
| ckpt_5k | STACK | 5156 | 34.4 | 312 | 2.3 | 1.0 | 0.3 | 5.31 | 116 | 0.7 | 123 | 1.0 |
| ckpt_5k | BIG | 15469 | 35.3 | 155 | 2.3 | 1.0 | 0.3 | 5.31 | 1 | 0.3 | 127 | 1.0 |

## Checks

- remote decode == local decode (bit-exact): 36/36
- SEP == BIG remote samples: 12/12
- STACK crop vs SEP PSNR (separate lossy encodes): min 42.1 dB over 12 samples
- pts violations > 1e-4 s: 0 arms
- 429 responses in stage: 0; resolver budget start -1, end -1, min seen 1073741824
- skipped arms: 0

