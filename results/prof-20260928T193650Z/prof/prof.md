# PR #3917 profile (5bf6c034ba293a075739dfb1ac6e1cbf79428f38)

Config: {"arms": ["D2", "D6", "D6T", "R6", "R6t1", "D6P8", "D6P8f"], "reps": 3, "cap_s": 150.0, "warm_s": 45.0, "batch": 8, "fetch_concurrency": 4, "repo": "hadriencornier/lerobot-data11-r3-sep", "rev": "31fd7b6605cd41c6c6c7c059ccd261924d74f033", "pr3917_sha": "5bf6c034ba293a075739dfb1ac6e1cbf79428f38"}

| arm | samples/s (mean +- sd, n) | CPU ms/sample | top thread groups (ms/sample) | first sample s | PSS GB |
|---|---|---|---|---|---|
| D2 | 268.8 +- 38.9 (n=3) | 6.2 | lerobot-decode 4.7, MainThread 1.0, Thread-1 (_serve) 0.4, QueueFeederThread 0.2, prof 0.1 | 18, 18, 16 | 2.3, 2.2, 2.0 |
| D6 | 207.1 +- 9.6 (n=3) | 8.4 | lerobot-decode 6.5, MainThread 1.0, Thread-1 (_serve) 0.7, QueueFeederThread 0.2, prof 0.1 | 18, 18, 16 | 2.3, 2.2, 2.0 |
| D6T | 59.2 +- 4.9 (n=3) | 19.8 | MainThread 14.2, lerobot-decode 5.1, Thread-1 (_serve) 0.4, prof 0.3, QueueFeederThread 0.2 | 19, 18, 18 | 2.2, 2.2, 2.0 |
| R6 | 326.2 +- 27.8 (n=3) | 6.5 | lerobot-decode 6.3, MainThread 0.3, prof 0.1, native:python 0.0, native:jemalloc_bg_thd 0.0 | 13, 7, 12 | 2.3, 2.2, 2.1 |
| R6t1 | 344.5 +- 33.1 (n=3) | 6.5 | lerobot-decode 6.2, MainThread 0.3, prof 0.1, native:python 0.0, native:jemalloc_bg_thd 0.0 | 14, 8, 11 | 2.4, 2.2, 2.0 |
| D6P8 | 252.9 +- 55.2 (n=2) | 8.1 | lerobot-decode 6.2, MainThread 0.9, Thread-1 (_serve) 0.6, QueueFeederThread 0.2, cache-fetch 0.1 | 9, 7 | 2.4, 2.9 |
| D6P8f | 255.4 +- 3.6 (n=2) | 8.5 | lerobot-decode 6.5, MainThread 0.9, Thread-1 (_serve) 0.7, QueueFeederThread 0.2, cache-fetch 0.1 | 9, 9 | 2.8, 2.9 |

## Timers (per sample, averaged over clean repeats)

- **D2**: planner_admit: 0.000 calls, 0.00 ms wall, 0.00 ms CPU; synthesize_mp4(clip): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; fetch_and_synthesize(clip): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; load_episode_parquet(episode): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; wait_clip_bytes(camera): 2.995 calls, 0.04 ms wall, 0.04 ms CPU; open_decoder(clip): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; get_frames(camera): 2.995 calls, 5.35 ms wall, 3.55 ms CPU; make_episode_item(sample): 0.998 calls, 7.49 ms wall, 4.63 ms CPU; apply_image_transforms(sample): 0.998 calls, 0.00 ms wall, 0.00 ms CPU
- **D6**: planner_admit: 0.000 calls, 0.00 ms wall, 0.00 ms CPU; load_episode_parquet(episode): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; synthesize_mp4(clip): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; fetch_and_synthesize(clip): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; wait_clip_bytes(camera): 2.998 calls, 0.05 ms wall, 0.04 ms CPU; open_decoder(clip): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; get_frames(camera): 2.998 calls, 12.86 ms wall, 4.43 ms CPU; make_episode_item(sample): 0.999 calls, 22.03 ms wall, 6.41 ms CPU; apply_image_transforms(sample): 0.999 calls, 0.00 ms wall, 0.00 ms CPU
- **D6T**: planner_admit: 0.000 calls, 0.00 ms wall, 0.00 ms CPU; load_episode_parquet(episode): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; synthesize_mp4(clip): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; fetch_and_synthesize(clip): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; wait_clip_bytes(camera): 3.006 calls, 0.05 ms wall, 0.04 ms CPU; open_decoder(clip): 0.001 calls, 0.00 ms wall, 0.00 ms CPU; get_frames(camera): 3.006 calls, 4.17 ms wall, 3.60 ms CPU; make_episode_item(sample): 1.002 calls, 6.23 ms wall, 4.93 ms CPU; apply_image_transforms(sample): 1.002 calls, 15.93 ms wall, 13.38 ms CPU
- **R6**: planner_admit: 0.000 calls, 0.00 ms wall, 0.00 ms CPU; load_episode_parquet(episode): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; synthesize_mp4(clip): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; fetch_and_synthesize(clip): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; wait_clip_bytes(camera): 3.002 calls, 0.04 ms wall, 0.04 ms CPU; open_decoder(clip): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; get_frames(camera): 3.002 calls, 10.31 ms wall, 4.35 ms CPU; make_episode_item(sample): 1.001 calls, 16.62 ms wall, 6.20 ms CPU; apply_image_transforms(sample): 1.001 calls, 0.00 ms wall, 0.00 ms CPU
- **R6t1**: planner_admit: 0.000 calls, 0.00 ms wall, 0.00 ms CPU; load_episode_parquet(episode): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; synthesize_mp4(clip): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; fetch_and_synthesize(clip): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; wait_clip_bytes(camera): 3.006 calls, 0.04 ms wall, 0.04 ms CPU; open_decoder(clip): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; get_frames(camera): 3.006 calls, 9.92 ms wall, 4.32 ms CPU; make_episode_item(sample): 1.002 calls, 15.94 ms wall, 6.15 ms CPU; apply_image_transforms(sample): 1.002 calls, 0.00 ms wall, 0.00 ms CPU
- **D6P8**: planner_admit: 0.000 calls, 0.00 ms wall, 0.00 ms CPU; synthesize_mp4(clip): 0.001 calls, 0.02 ms wall, 0.02 ms CPU; fetch_and_synthesize(clip): 0.001 calls, 0.93 ms wall, 0.13 ms CPU; load_episode_parquet(episode): 0.000 calls, 0.22 ms wall, 0.04 ms CPU; wait_clip_bytes(camera): 3.000 calls, 0.04 ms wall, 0.04 ms CPU; open_decoder(clip): 0.003 calls, 0.01 ms wall, 0.00 ms CPU; get_frames(camera): 3.000 calls, 13.38 ms wall, 4.24 ms CPU; make_episode_item(sample): 1.000 calls, 20.78 ms wall, 6.10 ms CPU; apply_image_transforms(sample): 1.000 calls, 0.00 ms wall, 0.00 ms CPU
- **D6P8f**: planner_admit: 0.000 calls, 0.00 ms wall, 0.00 ms CPU; synthesize_mp4(clip): 0.001 calls, 0.02 ms wall, 0.02 ms CPU; fetch_and_synthesize(clip): 0.001 calls, 0.91 ms wall, 0.14 ms CPU; load_episode_parquet(episode): 0.000 calls, 0.21 ms wall, 0.04 ms CPU; wait_clip_bytes(camera): 3.007 calls, 0.04 ms wall, 0.04 ms CPU; open_decoder(clip): 0.003 calls, 0.01 ms wall, 0.00 ms CPU; get_frames(camera): 3.007 calls, 13.15 ms wall, 4.42 ms CPU; make_episode_item(sample): 1.002 calls, 20.78 ms wall, 6.45 ms CPU; apply_image_transforms(sample): 1.002 calls, 0.00 ms wall, 0.00 ms CPU

## Stack samples (sampled runs, top inclusive per thread group)

### D2_r90_s
- lerobot-parquet (39552 samples): 
- lerobot-parquet (idle) (39552 samples): 
- cache-fetch (39552 samples): 
- cache-fetch (idle) (39552 samples): 
- lerobot-decode (19776 samples): run (threading.py) 19718; w (prof_3917.py) 19718; _worker (thread.py) 19718; decode_item (lerobot/datasets/streaming_dataset.py) 19718; _bootstrap (threading.py) 19718; run (thread.py) 19718; _bootstrap_inner (threading.py) 19718; _make_episode_item (lerobot/datasets/streaming_dataset.py) 19658; get_frames (lerobot/streaming/episode_cache.py) 14551; _get_frames (lerobot/streaming/episode_cache.py) 14400; get_frames_at (torchcodec/decoders/_video_decoder.py) 11158; get_frames_at_indices (torchcodec/_core/ops.py) 10978
- Thread-1 (_serve) (9888 samples): run (threading.py) 1059; _serve (resource_sharer.py) 1059; _bootstrap (threading.py) 1059; _bootstrap_inner (threading.py) 1059; accept (connection.py) 598; _send_bytes (connection.py) 571; _send (connection.py) 571; send_bytes (connection.py) 571; deliver_challenge (connection.py) 477; close (connection.py) 202; _close (connection.py) 202; __exit__ (connection.py) 202
- QueueFeederThread (9888 samples): run (threading.py) 595; _feed (queues.py) 595; _bootstrap_inner (threading.py) 595; _bootstrap (threading.py) 595; dumps (reduction.py) 352; reduce_storage (torch/multiprocessing/reductions.py) 352; fd_id (torch/multiprocessing/reductions.py) 324; _send_bytes (connection.py) 139; _send (connection.py) 139; send_bytes (connection.py) 139; DupFd (reduction.py) 27; __init__ (resource_sharer.py) 27
- MainThread (9888 samples): _worker_loop (torch/utils/data/_utils/worker.py) 3282; run (process.py) 3282; <module> (<string>) 3282; _bootstrap (process.py) 3282; _main (spawn.py) 3282; spawn_main (spawn.py) 3282; fetch (torch/utils/data/_utils/fetch.py) 2988; collate (torch/utils/data/_utils/collate.py) 2627; collate_tensor_fn (torch/utils/data/_utils/collate.py) 2627; default_collate (torch/utils/data/_utils/collate.py) 2627; _iter_once (lerobot/datasets/streaming_dataset.py) 361; _repeat_iterator (lerobot/datasets/streaming_dataset.py) 361
- QueueFeederThread (idle) (9293 samples): 
- Thread-1 (_serve) (idle) (8829 samples): 
- MainThread (idle) (6606 samples): 
- lerobot-decode (idle) (58 samples): 
### D6_r90_s
- lerobot-decode (58584 samples): run (thread.py) 50952; _bootstrap (threading.py) 50952; decode_item (lerobot/datasets/streaming_dataset.py) 50952; w (prof_3917.py) 50952; run (threading.py) 50952; _worker (thread.py) 50952; _bootstrap_inner (threading.py) 50952; _make_episode_item (lerobot/datasets/streaming_dataset.py) 50716; get_frames (lerobot/streaming/episode_cache.py) 29832; _get_frames (lerobot/streaming/episode_cache.py) 28184; get_frames_at (torchcodec/decoders/_video_decoder.py) 14181; get_frames_at_indices (torchcodec/_core/ops.py) 13718
- lerobot-parquet (39056 samples): 
- lerobot-parquet (idle) (39056 samples): 
- cache-fetch (39056 samples): 
- cache-fetch (idle) (39056 samples): 
- Thread-1 (_serve) (9764 samples): _bootstrap (threading.py) 2889; _serve (resource_sharer.py) 2889; run (threading.py) 2889; _bootstrap_inner (threading.py) 2889; accept (connection.py) 1406; _send (connection.py) 1258; _send_bytes (connection.py) 1258; send_bytes (connection.py) 1258; deliver_challenge (connection.py) 1101; send (resource_sharer.py) 623; send_handle (reduction.py) 623; close (connection.py) 597
- QueueFeederThread (9764 samples): _bootstrap (threading.py) 582; run (threading.py) 582; _feed (queues.py) 582; _bootstrap_inner (threading.py) 582; dumps (reduction.py) 345; reduce_storage (torch/multiprocessing/reductions.py) 345; fd_id (torch/multiprocessing/reductions.py) 323; _send (connection.py) 97; _send_bytes (connection.py) 97; send_bytes (connection.py) 97; __init__ (resource_sharer.py) 22; DupFd (reduction.py) 22
- MainThread (9764 samples): _main (spawn.py) 3509; _bootstrap (process.py) 3509; _worker_loop (torch/utils/data/_utils/worker.py) 3509; <module> (<string>) 3509; spawn_main (spawn.py) 3509; run (process.py) 3509; fetch (torch/utils/data/_utils/fetch.py) 2956; collate_tensor_fn (torch/utils/data/_utils/collate.py) 2548; collate (torch/utils/data/_utils/collate.py) 2548; default_collate (torch/utils/data/_utils/collate.py) 2548; _repeat_iterator (lerobot/datasets/streaming_dataset.py) 408; _iter_once (lerobot/datasets/streaming_dataset.py) 408
- QueueFeederThread (idle) (9182 samples): 
- lerobot-decode (idle) (7632 samples): 
- Thread-1 (_serve) (idle) (6875 samples): 
- MainThread (idle) (6255 samples): 
### D6T_r90_s
- lerobot-decode (60750 samples): _bootstrap (threading.py) 3511; w (prof_3917.py) 3511; _worker (thread.py) 3511; run (threading.py) 3511; _bootstrap_inner (threading.py) 3511; decode_item (lerobot/datasets/streaming_dataset.py) 3511; run (thread.py) 3511; _make_episode_item (lerobot/datasets/streaming_dataset.py) 3499; get_frames (lerobot/streaming/episode_cache.py) 2426; _get_frames (lerobot/streaming/episode_cache.py) 2412; get_frames_at (torchcodec/decoders/_video_decoder.py) 2092; get_frames_at_indices (torchcodec/_core/ops.py) 2084
- lerobot-decode (idle) (57239 samples): 
- lerobot-parquet (40500 samples): 
- lerobot-parquet (idle) (40500 samples): 
- cache-fetch (40500 samples): 
- cache-fetch (idle) (40500 samples): 
- Thread-1 (_serve) (10125 samples): _bootstrap (threading.py) 188; run (threading.py) 188; _bootstrap_inner (threading.py) 188; _serve (resource_sharer.py) 188; accept (connection.py) 110; _send (connection.py) 103; send_bytes (connection.py) 103; _send_bytes (connection.py) 103; deliver_challenge (connection.py) 85; send_handle (reduction.py) 40; send (resource_sharer.py) 40; __exit__ (connection.py) 33
- QueueFeederThread (10125 samples): _bootstrap (threading.py) 129; _feed (queues.py) 129; run (threading.py) 129; _bootstrap_inner (threading.py) 129; reduce_storage (torch/multiprocessing/reductions.py) 71; dumps (reduction.py) 71; fd_id (torch/multiprocessing/reductions.py) 67; _send (connection.py) 36; send_bytes (connection.py) 36; _send_bytes (connection.py) 36; DupFd (reduction.py) 4; __init__ (resource_sharer.py) 4
- MainThread (10125 samples): <module> (<string>) 10099; _worker_loop (torch/utils/data/_utils/worker.py) 10099; _bootstrap (process.py) 10099; spawn_main (spawn.py) 10099; _main (spawn.py) 10099; run (process.py) 10099; fetch (torch/utils/data/_utils/fetch.py) 10088; _repeat_iterator (lerobot/datasets/streaming_dataset.py) 9706; _iter_once (lerobot/datasets/streaming_dataset.py) 9706; _apply_image_transforms (lerobot/datasets/streaming_dataset.py) 9621; w (prof_3917.py) 9621; forward (lerobot/transforms/transforms.py) 9597
- QueueFeederThread (idle) (9996 samples): 
- Thread-1 (_serve) (idle) (9937 samples): 
- MainThread (idle) (26 samples): 
### R6_r90_s
- lerobot-decode (57144 samples): run (thread.py) 56036; _bootstrap_inner (threading.py) 56036; run (threading.py) 56036; _worker (thread.py) 56036; w (prof_3917.py) 56036; _bootstrap (threading.py) 56036; decode_item (lerobot/datasets/streaming_dataset.py) 56036; _make_episode_item (lerobot/datasets/streaming_dataset.py) 55791; get_frames (lerobot/streaming/episode_cache.py) 34787; _get_frames (lerobot/streaming/episode_cache.py) 32998; get_frames_at (torchcodec/decoders/_video_decoder.py) 17898; get_frames_at_indices (torchcodec/_core/ops.py) 17310
- lerobot-parquet (38096 samples): 
- lerobot-parquet (idle) (38096 samples): 
- cache-fetch (38096 samples): 
- cache-fetch (idle) (38096 samples): 
- tqdm_monitor (19048 samples): 
- tqdm_monitor (idle) (19048 samples): 
- MainThread (9524 samples): cmd_run (prof_3917.py) 3232; <module> (prof_3917.py) 3232; _iter_once (lerobot/datasets/streaming_dataset.py) 1; _repeat_iterator (lerobot/datasets/streaming_dataset.py) 1; w (prof_3917.py) 1
- MainThread (idle) (6292 samples): 
- lerobot-decode (idle) (1108 samples): 
### R6t1_r90_s
- lerobot-decode (56514 samples): _worker (thread.py) 53592; run (thread.py) 53592; run (threading.py) 53592; _bootstrap_inner (threading.py) 53592; _bootstrap (threading.py) 53592; decode_item (lerobot/datasets/streaming_dataset.py) 53589; w (prof_3917.py) 53589; _make_episode_item (lerobot/datasets/streaming_dataset.py) 53359; get_frames (lerobot/streaming/episode_cache.py) 33500; _get_frames (lerobot/streaming/episode_cache.py) 31917; get_frames_at (torchcodec/decoders/_video_decoder.py) 17861; get_frames_at_indices (torchcodec/_core/ops.py) 17327
- lerobot-parquet (37676 samples): 
- lerobot-parquet (idle) (37676 samples): 
- cache-fetch (37676 samples): 
- cache-fetch (idle) (37676 samples): 
- tqdm_monitor (18838 samples): 
- tqdm_monitor (idle) (18838 samples): 
- MainThread (9419 samples): <module> (prof_3917.py) 3045; cmd_run (prof_3917.py) 3045; _iter_once (lerobot/datasets/streaming_dataset.py) 3; _repeat_iterator (lerobot/datasets/streaming_dataset.py) 3; submit (thread.py) 2; __exit__ (threading.py) 1; acquire (threading.py) 1; _adjust_thread_count (thread.py) 1; __next__ (lerobot/streaming/episode_pool.py) 1
- MainThread (idle) (6374 samples): 
- lerobot-decode (idle) (2922 samples): 

py-spy self test: {'rc': 0, 'err': ''}
