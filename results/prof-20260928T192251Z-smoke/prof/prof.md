# PR #3917 profile (5bf6c034ba293a075739dfb1ac6e1cbf79428f38)

Config: {"arms": ["D2", "R6"], "reps": 1, "cap_s": 75.0, "warm_s": 20.0, "batch": 8, "fetch_concurrency": 4, "repo": "hadriencornier/lerobot-data11-r3-sep", "rev": "31fd7b6605cd41c6c6c7c059ccd261924d74f033", "pr3917_sha": "5bf6c034ba293a075739dfb1ac6e1cbf79428f38"}

| arm | samples/s (mean +- sd, n) | CPU ms/sample | top thread groups (ms/sample) | first sample s | PSS GB |
|---|---|---|---|---|---|
| D2 | 296.0 +- 0.0 (n=1) | 6.7 | lerobot-decode 5.0, MainThread 1.1, Thread-1 (_serve) 0.4, QueueFeederThread 0.2, prof 0.1 | 21 | 2.2 |
| R6 | 383.5 +- 0.0 (n=1) | 7.5 | lerobot-decode 7.2, MainThread 0.3, prof 0.1, native:python 0.0, native:jemalloc_bg_thd 0.0 | 14 | 2.3 |

## Timers (per sample, averaged over clean repeats)

- **D2**: synthesize_mp4(clip): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; fetch_and_synthesize(clip): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; load_episode_parquet(episode): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; open_decoder(clip): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; get_frames(camera): 3.023 calls, 4.87 ms wall, 3.80 ms CPU; make_episode_item(sample): 1.008 calls, 6.73 ms wall, 4.95 ms CPU; apply_image_transforms(sample): 1.008 calls, 0.00 ms wall, 0.00 ms CPU
- **R6**: synthesize_mp4(clip): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; fetch_and_synthesize(clip): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; load_episode_parquet(episode): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; open_decoder(clip): 0.000 calls, 0.00 ms wall, 0.00 ms CPU; get_frames(camera): 2.993 calls, 9.73 ms wall, 5.01 ms CPU; make_episode_item(sample): 0.998 calls, 15.51 ms wall, 7.13 ms CPU; apply_image_transforms(sample): 0.998 calls, 0.00 ms wall, 0.00 ms CPU

## Stack samples (sampled runs, top inclusive per thread group)

### D2_r90_s
- lerobot-parquet (20808 samples): run (threading.py) 20808; _bootstrap (threading.py) 20808; _bootstrap_inner (threading.py) 20808; _worker (thread.py) 20808
- cache-fetch (20808 samples): run (threading.py) 20808; _bootstrap (threading.py) 20808; _bootstrap_inner (threading.py) 20808; _worker (thread.py) 20808
- lerobot-decode (10404 samples): run (threading.py) 10404; _bootstrap_inner (threading.py) 10404; _worker (thread.py) 10404; run (thread.py) 10404; _bootstrap (threading.py) 10404; decode_item (lerobot/datasets/streaming_dataset.py) 10404; w (prof_3917.py) 10404; _make_episode_item (lerobot/datasets/streaming_dataset.py) 10375; get_frames (lerobot/streaming/episode_cache.py) 7641; _get_frames (lerobot/streaming/episode_cache.py) 7537; get_frames_at (torchcodec/decoders/_video_decoder.py) 5960; get_frames_at_indices (torchcodec/_core/ops.py) 5883
- Thread-1 (_serve) (5202 samples): run (threading.py) 5202; _bootstrap_inner (threading.py) 5202; _bootstrap (threading.py) 5202; _serve (resource_sharer.py) 5202; accept (connection.py) 4839; accept (socket.py) 3799; _recv (connection.py) 868; _recv_bytes (connection.py) 868; recv_bytes (connection.py) 734; answer_challenge (connection.py) 549; deliver_challenge (connection.py) 479; _send_bytes (connection.py) 281
- QueueFeederThread (5202 samples): _feed (queues.py) 5202; _bootstrap_inner (threading.py) 5202; _bootstrap (threading.py) 5202; run (threading.py) 5202; wait (threading.py) 4928; dumps (reduction.py) 158; reduce_storage (torch/multiprocessing/reductions.py) 158; fd_id (torch/multiprocessing/reductions.py) 148; _send_bytes (connection.py) 62; send_bytes (connection.py) 62; _send (connection.py) 62; __init__ (resource_sharer.py) 10
- MainThread (5202 samples): _main (spawn.py) 5202; _bootstrap (process.py) 5202; run (process.py) 5202; _worker_loop (torch/utils/data/_utils/worker.py) 5202; spawn_main (spawn.py) 5202; <module> (<string>) 5202; fetch (torch/utils/data/_utils/fetch.py) 5016; _iter_once (lerobot/datasets/streaming_dataset.py) 3334; _repeat_iterator (lerobot/datasets/streaming_dataset.py) 3334; wait (threading.py) 3126; result (_base.py) 3126; collate_tensor_fn (torch/utils/data/_utils/collate.py) 1682
### R6_r90_s
- lerobot-decode (32310 samples): run (threading.py) 32310; _bootstrap (threading.py) 32310; _worker (thread.py) 32310; _bootstrap_inner (threading.py) 32310; run (thread.py) 32017; decode_item (lerobot/datasets/streaming_dataset.py) 32017; w (prof_3917.py) 32017; _make_episode_item (lerobot/datasets/streaming_dataset.py) 31924; get_frames (lerobot/streaming/episode_cache.py) 20187; _get_frames (lerobot/streaming/episode_cache.py) 19198; get_frames_at (torchcodec/decoders/_video_decoder.py) 10789; get_frames_at_indices (torchcodec/_core/ops.py) 10523
- lerobot-parquet (21540 samples): run (threading.py) 21540; _bootstrap (threading.py) 21540; _worker (thread.py) 21540; _bootstrap_inner (threading.py) 21540
- cache-fetch (21540 samples): run (threading.py) 21540; _bootstrap (threading.py) 21540; _worker (thread.py) 21540; _bootstrap_inner (threading.py) 21540
- tqdm_monitor (10770 samples): wait (threading.py) 10770; _bootstrap (threading.py) 10770; run (tqdm/_monitor.py) 10770; _bootstrap_inner (threading.py) 10770
- MainThread (5385 samples): cmd_run (prof_3917.py) 5385; <module> (prof_3917.py) 5385; _repeat_iterator (lerobot/datasets/streaming_dataset.py) 3485; result (_base.py) 3485; wait (threading.py) 3485; _iter_once (lerobot/datasets/streaming_dataset.py) 3485

py-spy self test: {'rc': 0, 'err': ''}
