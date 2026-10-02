# DATA-11 round 2: random access over hf:// (family ckpt_full, 3 cameras per sample)

Repo `hadriencornier/lerobot-data11-bench@a3d31680db`. sps = samples/s, one reader. fetch = one HTTP request that moves video bytes (CDN). fs* readers also pay 1 resolve per fetch; rc / exact resolve once per file (untimed, shown in `presigned`: 12 resolves for 12 files).

| regime | layout | cams | reader | mode | sps | +-CI | p50 ms | p95 ms | fetch/smp | resolve/smp | API/smp | MB/smp | dec MB/smp | open ms/smp | fallback/smp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| warm | SEP | seq | fs256K | - | 1.11 | 0.74 | 921 | 1096 | 3.00 | 3.00 | 0.00 | 0.987 | 0.197 | 0 | 0.00 |
| warm | SEP | seq | rc256K | - | 1.42 | 0.78 | 699 | 887 | 3.00 | 0.00 | 0.00 | 0.983 | 0.197 | 0 | 0.00 |
| warm | SEP | seq | exact | - | 1.64 | 0.12 | 622 | 816 | 2.93 | 0.00 | 0.00 | 0.210 | 0.197 | 0 | 0.00 |
| warm | STACK | - | fs256K | - | 2.87 | 6.51 | 426 | 559 | 1.00 | 1.00 | 0.00 | 0.329 | 0.066 | 0 | 0.00 |
| warm | STACK | - | rc256K | - | 3.72 | 12.41 | 342 | 527 | 1.00 | 0.00 | 0.00 | 0.328 | 0.066 | 0 | 0.00 |
| warm | STACK | - | exact | - | 5.11 | 11.63 | 200 | 333 | 1.00 | 0.00 | 0.00 | 0.083 | 0.066 | 0 | 0.00 |

Checks: remote == local decode (bit-exact) 0/0; pts violations > 1e-4 s: 0; skipped arms 0; 429 total 0.
