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
