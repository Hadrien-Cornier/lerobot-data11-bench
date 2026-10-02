# DATA-11 round 2: random access over hf:// (family ckpt_full, 3 cameras per sample)

Repo `hadriencornier/lerobot-data11-bench@5efceaf32b`. sps = samples/s, one reader. fetch = one HTTP request that moves video bytes (CDN). fs* readers also pay 1 resolve per fetch; rc / exact resolve once per file (untimed, shown in `presigned`: 6 resolves for 6 files).

| regime | layout | cams | reader | mode | sps | +-CI | p50 ms | p95 ms | fetch/smp | resolve/smp | API/smp | MB/smp | dec MB/smp | open ms/smp | fallback/smp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold | MULTI | par | exact | shared | 3.29 | 1.18 | 278 | 545 | 2.00 | 0.00 | 0.00 | 1.106 | 2.568 | 165 | 0.00 |
| cold | STACK | - | rc256K | - | 6.38 | 0.91 | 154 | 212 | 2.00 | 0.00 | 0.00 | 0.655 | 0.283 | 77 | 0.00 |
| cold | STACK | - | exact | - | 5.39 | 1.94 | 178 | 328 | 2.00 | 0.00 | 0.00 | 0.499 | 0.283 | 101 | 0.00 |
| warm | MULTI | par | exact | shared | 8.50 | 2.67 | 117 | 217 | 1.00 | 0.00 | 0.00 | 0.097 | 0.197 | 0 | 0.00 |
| warm | STACK | - | rc256K | - | 13.72 | 1.07 | 66 | 117 | 1.00 | 0.00 | 0.00 | 0.328 | 0.066 | 0 | 0.00 |
| warm | STACK | - | exact | - | 13.61 | 2.49 | 67 | 141 | 1.00 | 0.00 | 0.00 | 0.084 | 0.066 | 0 | 0.00 |

Checks: remote == local decode (bit-exact) 12/12; pts violations > 1e-4 s: 0; skipped arms 0; 429 total 0.
