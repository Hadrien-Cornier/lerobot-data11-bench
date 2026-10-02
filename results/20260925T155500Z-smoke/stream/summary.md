# DATA-11 round 2: real StreamingLeRobotDataset over hf:// (PR #4702 branch)

lerobot a195a91ff815ad730373276ef8a4cc7c04c5c074; datasets SEP `hadriencornier/lerobot-data11-stream-sep`, STACK `hadriencornier/lerobot-data11-stream-stack` (8 video files per key, copies of the round-1 ckpt_full variants). Single process = one DataLoader worker. fps = frames made per second after warm-up (every step makes one frame).

| arm | S | cache | cams | fps | +-CI | mp4 fetch/frame | mp4 MB/frame | parquet fetch/frame | resolve/frame | decoder hit | opens/frame | open ms | CPU ms/frame | first yield s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SEP_S8_c100_seq | 8 | 100 | seq | 126.6 | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 0.000 | 0 | 7.8 | - |
| SEP_S8_c12_seq | 8 | 12 | seq | 4.6 | - | 1.6522 | 8.7704 | 0.0000 | 1.6522 | 0.449 | 1.652 | 128 | 74.8 | - |
| STACK_S8_c100 | 8 | 100 | seq | 256.4 | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 0.000 | 0 | 3.9 | - |
| SEP_S1_c100_par | 1 | 100 | par | 72.9 | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 0.000 | 0 | 10.8 | - |

## 8 concurrent processes (own dataset + seed each)

| arm | procs | fps total | resolves/s | CDN req/s | MB/s | decoder hit | 429 |
|---|---|---|---|---|---|---|---|
| SEP_S8_c100_seq | 8 | 1040.9 | 0.0 | 0.0 | 0.0 | 1.000 | 0 |
