# DATA-11 round 2: real StreamingLeRobotDataset over hf:// (PR #4702 branch)

lerobot a195a91ff815ad730373276ef8a4cc7c04c5c074; datasets SEP `hadriencornier/lerobot-data11-stream-sep`, STACK `hadriencornier/lerobot-data11-stream-stack` (8 video files per key, copies of the round-1 ckpt_full variants). Single process = one DataLoader worker. fps = frames made per second after warm-up (every step makes one frame).

| arm | S | cache | cams | fps | +-CI | mp4 fetch/frame | mp4 MB/frame | parquet fetch/frame | resolve/frame | decoder hit | opens/frame | open ms | CPU ms/frame | first yield s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SEP_S1_c100_seq | 1 | 100 | seq | 181.6 | 44.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 0.000 | 0 | 5.5 | 8, 6, 6 |
| SEP_S1_c100_par | 1 | 100 | par | 146.1 | 19.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 0.000 | 0 | 8.8 | 8, 8, 8 |
| SEP_S8_c100_seq | 8 | 100 | seq | 181.5 | 30.4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 0.000 | 0 | 5.5 | 13, 12, 11 |
| SEP_S8_c100_par | 8 | 100 | par | 136.3 | 13.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 0.000 | 0 | 9.3 | 16, 12, 14 |
| SEP_S8_c12_seq | 8 | 12 | seq | 4.6 | 1.5 | 1.4817 | 7.8656 | 0.0000 | 1.4817 | 0.506 | 1.482 | 143 | 72.5 | -, -, - |
| SEP_S8_c12_par | 8 | 12 | par | 4.9 | 1.1 | 1.4875 | 7.8962 | 0.0000 | 1.4875 | 0.504 | 1.487 | 160 | 67.9 | -, -, - |
| STACK_S1_c100 | 1 | 100 | seq | 250.9 | 66.2 | 0.0017 | 0.0088 | 0.0000 | 0.0017 | 1.000 | 0.000 | 0 | 3.7 | 4, 4, 5 |
| STACK_S8_c100 | 8 | 100 | seq | 342.2 | 17.3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 0.000 | 0 | 2.9 | 6, 6, 7 |
| STACK_S8_c12 | 8 | 12 | seq | 308.3 | 80.5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 0.000 | 0 | 3.3 | 6, 7, 6 |

## 8 concurrent processes (own dataset + seed each)

| arm | procs | fps total | resolves/s | CDN req/s | MB/s | decoder hit | 429 |
|---|---|---|---|---|---|---|---|
| SEP_S8_c100_seq | 8 | 1551.9 | 0.8 | 0.8 | 4.4 | 1.000 | 0 |
| SEP_S8_c100_par | 8 | 862.6 | 0.0 | 0.0 | 0.0 | 1.000 | 0 |
| STACK_S8_c100 | 8 | 1948.0 | 2.7 | 2.7 | 14.2 | 1.000 | 0 |
| SEP_S8_c12_seq | 8 | 36.4 | 53.5 | 53.5 | 284.2 | 0.510 | 0 |
| STACK_S8_c12 | 8 | 2409.6 | 3.6 | 3.6 | 19.2 | 1.000 | 0 |
