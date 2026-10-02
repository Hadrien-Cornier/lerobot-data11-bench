# DATA-11 round 2: real StreamingLeRobotDataset over hf:// (PR #4702 branch)

lerobot a195a91f; datasets SEP `hadriencornier/lerobot-data11-stream-sep`, STACK `hadriencornier/lerobot-data11-stream-stack` (8 video files per key, copies of the round-1 ckpt_full variants). Single process = one DataLoader worker. fps = frames made per second after warm-up (every step makes one frame).

| arm | S | cache | cams | fps | +-CI | mp4 fetch/frame | mp4 MB/frame | parquet fetch/frame | resolve/frame | decoder hit | opens/frame | open ms | CPU ms/frame | first yield s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SEP_S1_c100_seq | 1 | 100 | seq | 501.6 | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 0.000 | 0 | 2.0 | - |
| STACK_S1_c100 | 1 | 100 | seq | 561.7 | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 0.000 | 0 | 1.8 | - |
