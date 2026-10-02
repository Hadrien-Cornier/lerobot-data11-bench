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
