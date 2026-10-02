# Streaming camera-parallelism bench (streaming-par-20260925T160240Z-smoke)

`lerobot/abc_130k_v3_train@68651e49`, StreamingLeRobotDataset over hf://, batch 8, 3 cameras, no delta_timestamps.

| arm | regime | ok/runs | samples/s (mean +- 95% CI) | first batch s | build s | req/sample (resolve, cdn) | MB/sample | 429s | peak PSS GB | peak cgroup GB |
|---|---|---|---|---|---|---|---|---|---|---|
| base | w0_cachedef | 0/1 | - | - | - | - (-, -) | - | 0 | 0.31 | 3.16 |
| base | w8_cache3 | 0/1 | - | - | - | - (-, -) | - | 0 | 0.22 | 3.11 |
| p1p2 | w0_cachedef | 0/1 | - | - | - | - (-, -) | - | 0 | 0.22 | 3.11 |
| p1p2 | w8_cache3 | 0/1 | - | - | - | - (-, -) | - | 0 | 0.31 | 3.16 |
| pr3917 | w0_dt2 | 0/1 | - | - | - | - (-, -) | - | 0 | 0.22 | 3.11 |
| pr3917 | w1_dt8 | 0/1 | - | - | - | - (-, -) | - | 0 | 0.22 | 3.11 |

## Correctness

```
{
 "n_hashed": {
  "base": 0,
  "p1p2": 0,
  "pr3917": 0
 }
}
```

## Prep / hash runs (num_workers=0, discarded for timing)

```
{
 "base": {
  "status": "error",
  "build_s": null,
  "first_batch_s": null,
  "peak_pss_gb": 0.32752037048339844,
  "error": "RuntimeError: operator torchvision::nms does not exist",
  "status_429_total": null,
  "build_http": null
 },
 "p1p2": {
  "status": "error",
  "build_s": null,
  "first_batch_s": null,
  "peak_pss_gb": 0.22011280059814453,
  "error": "RuntimeError: operator torchvision::nms does not exist",
  "status_429_total": null,
  "build_http": null
 },
 "pr3917": {
  "status": "error",
  "build_s": null,
  "first_batch_s": null,
  "peak_pss_gb": 0.31017494201660156,
  "error": "RuntimeError: operator torchvision::nms does not exist",
  "status_429_total": null,
  "build_http": null
 }
}
```
