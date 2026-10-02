# DATA-11 round 2: random access over hf:// (family ckpt_full, 3 cameras per sample)

Repo `hadriencornier/lerobot-data11-bench@a3d31680db`. sps = samples/s, one reader. fetch = one HTTP request that moves video bytes (CDN). fs* readers also pay 1 resolve per fetch; rc / exact resolve once per file (untimed, shown in `presigned`: 15 resolves for 15 files).

| regime | layout | cams | reader | mode | sps | +-CI | p50 ms | p95 ms | fetch/smp | resolve/smp | API/smp | MB/smp | dec MB/smp | open ms/smp | fallback/smp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold | MULTI | par | exact | shared | 2.78 | - | 367 | 476 | 2.00 | 0.00 | 0.00 | 1.101 | 2.569 | 163 | 0.00 |
| cold | MULTI | seq | fs256K | naive | 0.76 | - | 1382 | 1460 | 12.00 | 12.00 | 0.00 | 3.757 | 2.569 | 858 | 0.00 |
| cold | SEP | par | rc256K | - | 3.21 | - | 336 | 367 | 6.00 | 0.00 | 0.00 | 1.966 | 0.843 | 313 | 0.00 |
| cold | SEP | seq | fs5M | - | 0.59 | - | 1646 | 2074 | 6.00 | 6.00 | 0.00 | 31.855 | 0.843 | 819 | 0.00 |
| cold | STACK | - | exact | - | 2.15 | - | 490 | 590 | 2.00 | 0.00 | 0.00 | 0.497 | 0.282 | 231 | 0.00 |
| warm | MULTI | par | exact | shared | 13.54 | - | 61 | 115 | 1.00 | 0.00 | 0.00 | 0.093 | 0.197 | 0 | 0.00 |
| warm | MULTI | seq | fs256K | naive | 2.79 | - | 323 | 525 | 3.00 | 3.00 | 0.00 | 0.985 | 0.197 | 0 | 0.00 |
| warm | SEP | par | rc256K | - | 7.28 | - | 123 | 227 | 3.00 | 0.00 | 0.00 | 0.983 | 0.197 | 0 | 0.00 |
| warm | SEP | seq | fs5M | - | 2.01 | - | 514 | 1047 | 2.00 | 2.00 | 0.00 | 10.618 | 0.197 | 0 | 0.00 |
| warm | STACK | - | exact | - | 5.42 | - | 118 | 384 | 1.00 | 0.00 | 0.00 | 0.081 | 0.066 | 0 | 0.00 |

## 8 concurrent reader processes

| config | sps total | fetch/s | resolves/s | MB/s | p50 ms | p95 ms | 429 | other err |
|---|---|---|---|---|---|---|---|---|
| STACK|-|exact|-|warm | 104.3 | 103.9 | 0.0 | 8.6 | 65 | 133 | 0 | 0 |
| MULTI|par|exact|shared|cold | 31.5 | 62.9 | 0.0 | 34.6 | 223 | 434 | 0 | 0 |

Checks: remote == local decode (bit-exact) 20/20; pts violations > 1e-4 s: 0; skipped arms 0; 429 total 0.
