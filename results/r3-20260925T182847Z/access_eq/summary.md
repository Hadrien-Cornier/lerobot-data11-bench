# DATA-11 round 2: random access over hf:// (family ckpt_full, 3 cameras per sample)

Repo `hadriencornier/lerobot-data11-bench@5efceaf32b`. sps = samples/s, one reader. fetch = one HTTP request that moves video bytes (CDN). fs* readers also pay 1 resolve per fetch; rc / exact resolve once per file (untimed, shown in `presigned`: 27 resolves for 27 files).

| regime | layout | cams | reader | mode | sps | +-CI | p50 ms | p95 ms | fetch/smp | resolve/smp | API/smp | MB/smp | dec MB/smp | open ms/smp | fallback/smp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold | MULTI | par | fs5M | shared | 2.13 | 0.73 | 470 | 759 | 1.92 | 1.92 | 0.00 | 10.062 | 1.143 | 208 | 0.00 |
| cold | MULTI | par | rc256K | shared | 3.05 | 1.05 | 323 | 582 | 2.75 | 0.00 | 0.00 | 0.764 | 1.143 | 150 | 0.00 |
| cold | MULTI | par | exact | shared | 3.57 | 0.91 | 245 | 426 | 2.00 | 0.00 | 0.00 | 0.613 | 1.143 | 123 | 0.00 |
| cold | SEP | par | fs5M | - | 2.15 | 0.77 | 421 | 763 | 5.60 | 5.60 | 0.00 | 29.466 | 0.846 | 622 | 0.00 |
| cold | SEP | par | rc256K | - | 4.11 | 1.29 | 235 | 450 | 6.00 | 0.00 | 0.00 | 1.966 | 0.846 | 270 | 0.00 |
| cold | SEP | par | exact | - | 3.77 | 1.39 | 251 | 469 | 6.00 | 0.00 | 0.00 | 1.455 | 0.846 | 318 | 0.00 |
| cold | SEP | seq | fs5M | - | 1.03 | 0.38 | 941 | 1725 | 5.60 | 5.60 | 0.00 | 29.466 | 0.846 | 470 | 0.00 |
| cold | STACK | - | fs5M | - | 2.12 | 0.74 | 448 | 989 | 1.92 | 1.92 | 0.00 | 9.893 | 0.177 | 214 | 0.00 |
| cold | STACK | - | rc256K | - | 3.97 | 1.09 | 228 | 442 | 2.00 | 0.00 | 0.00 | 0.654 | 0.177 | 109 | 0.00 |
| cold | STACK | - | exact | - | 3.60 | 2.66 | 229 | 680 | 2.00 | 0.00 | 0.00 | 0.398 | 0.177 | 130 | 0.00 |
| warm | MULTI | par | fs5M | shared | 4.02 | 2.10 | 268 | 542 | 0.81 | 0.81 | 0.00 | 4.286 | 0.196 | 0 | 0.00 |
| warm | MULTI | par | rc256K | shared | 6.20 | 2.93 | 163 | 331 | 0.99 | 0.00 | 0.00 | 0.321 | 0.196 | 0 | 0.00 |
| warm | MULTI | par | exact | shared | 7.28 | 0.57 | 141 | 280 | 1.00 | 0.00 | 0.00 | 0.097 | 0.196 | 0 | 0.00 |
| warm | SEP | par | fs5M | - | 4.58 | 0.70 | 219 | 373 | 2.72 | 2.72 | 0.00 | 13.296 | 0.197 | 0 | 0.00 |
| warm | SEP | par | rc256K | - | 7.73 | 1.16 | 115 | 267 | 3.00 | 0.00 | 0.00 | 0.974 | 0.197 | 0 | 0.00 |
| warm | SEP | par | exact | - | 7.68 | 5.05 | 104 | 295 | 3.00 | 0.00 | 0.00 | 0.215 | 0.197 | 0 | 0.00 |
| warm | SEP | seq | fs5M | - | 2.09 | 0.78 | 500 | 870 | 2.72 | 2.72 | 0.00 | 13.296 | 0.197 | 0 | 0.00 |
| warm | STACK | - | fs5M | - | 4.49 | 1.75 | 207 | 429 | 0.89 | 0.89 | 0.00 | 4.261 | 0.065 | 0 | 0.00 |
| warm | STACK | - | rc256K | - | 7.14 | 2.80 | 127 | 307 | 1.00 | 0.00 | 0.00 | 0.325 | 0.065 | 0 | 0.00 |
| warm | STACK | - | exact | - | 7.30 | 3.36 | 109 | 264 | 1.00 | 0.00 | 0.00 | 0.084 | 0.065 | 0 | 0.00 |

## 8 concurrent reader processes

| config | sps total | fetch/s | resolves/s | MB/s | p50 ms | p95 ms | 429 | other err |
|---|---|---|---|---|---|---|---|---|
| MULTI|par|exact|shared|warm | 111.2 | 109.9 | 0.0 | 10.2 | 66 | 109 | 0 | 0 |
| MULTI|par|exact|shared|cold | 45.8 | 91.4 | 0.0 | 28.1 | 168 | 228 | 0 | 0 |
| SEP|par|rc256K|-|warm | 73.4 | 219.3 | 0.0 | 71.7 | 100 | 179 | 0 | 0 |
| SEP|par|rc256K|-|cold | 37.0 | 221.5 | 0.0 | 72.5 | 199 | 346 | 0 | 0 |
| STACK|-|exact|-|warm | 108.4 | 107.1 | 0.0 | 8.7 | 65 | 113 | 0 | 0 |
| STACK|-|rc256K|-|cold | 49.8 | 99.4 | 0.0 | 32.6 | 154 | 220 | 0 | 0 |

Checks: remote == local decode (bit-exact) 40/40; pts violations > 1e-4 s: 0; skipped arms 0; 429 total 0.
