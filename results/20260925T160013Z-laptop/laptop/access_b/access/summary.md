# DATA-11 round 2: random access over hf:// (family ckpt_full, 3 cameras per sample)

Repo `hadriencornier/lerobot-data11-bench@a3d31680db`. sps = samples/s, one reader. fetch = one HTTP request that moves video bytes (CDN). fs* readers also pay 1 resolve per fetch; rc / exact resolve once per file (untimed, shown in `presigned`: 12 resolves for 12 files).

| regime | layout | cams | reader | mode | sps | +-CI | p50 ms | p95 ms | fetch/smp | resolve/smp | API/smp | MB/smp | dec MB/smp | open ms/smp | fallback/smp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| warm | SEP | seq | fs5M | - | 0.56 | - | 1730 | 2165 | 3.00 | 3.00 | 0.00 | 15.929 | 0.197 | 0 | 0.00 |
| warm | STACK | - | fs5M | - | 1.32 | - | 631 | 1281 | 1.00 | 1.00 | 0.00 | 5.310 | 0.066 | 0 | 0.00 |

Checks: remote == local decode (bit-exact) 0/0; pts violations > 1e-4 s: 0; skipped arms 0; 429 total 0.
