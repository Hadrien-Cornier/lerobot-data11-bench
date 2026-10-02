# DATA-11 round 2: random access over hf:// (family ckpt_full, 3 cameras per sample)

Repo `hadriencornier/lerobot-data11-bench@b1a7e644fc`. sps = samples/s, one reader. fetch = one HTTP request that moves video bytes (CDN). fs* readers also pay 1 resolve per fetch; rc / exact resolve once per file (untimed, shown in `presigned`: 27 resolves for 27 files).

| regime | layout | cams | reader | mode | sps | +-CI | p50 ms | p95 ms | fetch/smp | resolve/smp | API/smp | MB/smp | dec MB/smp | open ms/smp | fallback/smp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold | MULTI | par | fs5M | shared | 1.63 | - | 611 | 780 | 1.75 | 1.75 | 0.00 | 10.487 | 1.130 | 334 | 0.00 |
| cold | MULTI | par | rc256K | shared | 2.92 | - | 315 | 423 | 2.75 | 0.00 | 0.00 | 0.786 | 1.130 | 188 | 0.00 |
| cold | MULTI | par | exact | shared | 5.58 | - | 180 | 204 | 2.00 | 0.00 | 0.00 | 0.619 | 1.130 | 92 | 0.00 |
| cold | SEP | par | fs5M | - | 1.61 | - | 583 | 843 | 6.00 | 6.00 | 0.00 | 31.855 | 0.840 | 669 | 0.00 |
| cold | SEP | par | rc256K | - | 2.71 | - | 303 | 619 | 6.00 | 0.00 | 0.00 | 1.966 | 0.840 | 349 | 0.00 |
| cold | SEP | par | exact | - | 4.24 | - | 235 | 295 | 6.00 | 0.00 | 0.00 | 1.450 | 0.840 | 277 | 0.00 |
| cold | SEP | seq | fs5M | - | 0.52 | - | 1923 | 2749 | 6.00 | 6.00 | 0.00 | 31.855 | 0.840 | 793 | 0.00 |
| cold | STACK | - | fs5M | - | 1.40 | - | 811 | 855 | 1.75 | 1.75 | 0.00 | 9.291 | 0.180 | 401 | 0.00 |
| cold | STACK | - | rc256K | - | 3.44 | - | 302 | 408 | 2.00 | 0.00 | 0.00 | 0.655 | 0.180 | 182 | 0.00 |
| cold | STACK | - | exact | - | 5.24 | - | 185 | 248 | 2.00 | 0.00 | 0.00 | 0.400 | 0.180 | 105 | 0.00 |
| warm | MULTI | par | fs5M | shared | 3.21 | - | 250 | 592 | 1.00 | 1.00 | 0.00 | 5.175 | 0.197 | 0 | 0.00 |
| warm | MULTI | par | rc256K | shared | 6.52 | - | 96 | 313 | 1.12 | 0.00 | 0.00 | 0.360 | 0.197 | 0 | 0.00 |
| warm | MULTI | par | exact | shared | 10.35 | - | 70 | 169 | 1.00 | 0.00 | 0.00 | 0.098 | 0.197 | 0 | 0.00 |
| warm | SEP | par | fs5M | - | 3.15 | - | 356 | 413 | 3.00 | 3.00 | 0.00 | 15.625 | 0.197 | 0 | 0.00 |
| warm | SEP | par | rc256K | - | 6.26 | - | 161 | 237 | 3.00 | 0.00 | 0.00 | 0.983 | 0.197 | 0 | 0.00 |
| warm | SEP | par | exact | - | 7.92 | - | 102 | 233 | 3.00 | 0.00 | 0.00 | 0.216 | 0.197 | 0 | 0.00 |
| warm | SEP | seq | fs5M | - | 1.29 | - | 870 | 981 | 3.00 | 3.00 | 0.00 | 15.625 | 0.197 | 0 | 0.00 |
| warm | STACK | - | fs5M | - | 3.94 | - | 220 | 386 | 1.00 | 1.00 | 0.00 | 4.807 | 0.066 | 0 | 0.00 |
| warm | STACK | - | rc256K | - | 6.94 | - | 115 | 279 | 1.00 | 0.00 | 0.00 | 0.328 | 0.066 | 0 | 0.00 |
| warm | STACK | - | exact | - | 10.29 | - | 76 | 180 | 1.00 | 0.00 | 0.00 | 0.085 | 0.066 | 0 | 0.00 |

## 8 concurrent reader processes

| config | sps total | fetch/s | resolves/s | MB/s | p50 ms | p95 ms | 429 | other err |
|---|---|---|---|---|---|---|---|---|
| MULTI|par|exact|shared|warm | 99.6 | 99.0 | 0.0 | 9.3 | 70 | 130 | 0 | 0 |
| MULTI|par|exact|shared|cold | 40.2 | 80.2 | 0.0 | 24.7 | 191 | 267 | 0 | 0 |
| SEP|par|exact|-|warm | 61.4 | 182.1 | 0.0 | 12.9 | 105 | 262 | 0 | 0 |
| SEP|par|exact|-|cold | 28.6 | 171.4 | 0.0 | 41.6 | 273 | 382 | 0 | 0 |
| STACK|-|exact|-|warm | 105.0 | 104.4 | 0.0 | 8.6 | 68 | 118 | 0 | 0 |
| STACK|-|exact|-|cold | 45.9 | 91.7 | 0.0 | 18.3 | 169 | 236 | 0 | 0 |

Checks: remote == local decode (bit-exact) 40/40; pts violations > 1e-4 s: 0; skipped arms 0; 429 total 0.
