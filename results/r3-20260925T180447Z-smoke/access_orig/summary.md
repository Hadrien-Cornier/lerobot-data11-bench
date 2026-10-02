# DATA-11 round 2: random access over hf:// (family ckpt_full, 3 cameras per sample)

Repo `hadriencornier/lerobot-data11-bench@b1a7e644fc`. sps = samples/s, one reader. fetch = one HTTP request that moves video bytes (CDN). fs* readers also pay 1 resolve per fetch; rc / exact resolve once per file (untimed, shown in `presigned`: 6 resolves for 6 files).

| regime | layout | cams | reader | mode | sps | +-CI | p50 ms | p95 ms | fetch/smp | resolve/smp | API/smp | MB/smp | dec MB/smp | open ms/smp | fallback/smp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold | MULTI | par | exact | shared | 2.10 | - | 484 | 542 | 2.00 | 0.00 | 0.00 | 1.099 | 2.524 | 188 | 0.00 |
| cold | STACK | - | rc256K | - | 6.47 | - | 149 | 177 | 2.00 | 0.00 | 0.00 | 0.655 | 0.281 | 73 | 0.00 |
| cold | STACK | - | exact | - | 3.90 | - | 266 | 301 | 2.00 | 0.00 | 0.00 | 0.498 | 0.281 | 145 | 0.00 |
| warm | MULTI | par | exact | shared | 5.79 | - | 147 | 291 | 1.00 | 0.00 | 0.00 | 0.098 | 0.197 | 0 | 0.00 |
| warm | STACK | - | rc256K | - | 13.67 | - | 72 | 88 | 1.00 | 0.00 | 0.00 | 0.328 | 0.066 | 0 | 0.00 |
| warm | STACK | - | exact | - | 10.41 | - | 92 | 129 | 1.00 | 0.00 | 0.00 | 0.085 | 0.066 | 0 | 0.00 |

Checks: remote == local decode (bit-exact) 12/12; pts violations > 1e-4 s: 0; skipped arms 0; 429 total 0.
