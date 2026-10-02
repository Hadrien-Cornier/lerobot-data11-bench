# Streaming camera-parallelism bench (streaming-par-20260925T163334Z-smoke)

`lerobot/abc_130k_v3_train@68651e49`, StreamingLeRobotDataset over hf://, batch 8, 3 cameras, no delta_timestamps.

| arm | regime | ok/runs | samples/s (mean +- 95% CI) | first batch s | build s | req/sample (resolve, cdn) | MB/sample | 429s | peak PSS GB | peak cgroup GB |
|---|---|---|---|---|---|---|---|---|---|---|
| base | w0_cachedef | 1/1 | 4.46 | 39.9 | 4.3 | 0.518 (0.250, 0.250) | 2.345 | 0 | 10.41 | 16.82 |
| base | w8_cache3 | 0/1 | - | - | - | - (-, -) | - | 0 | 0.00 | 0.00 |
| p1p2 | w0_cachedef | 1/1 | 5.44 | 27.2 | 4.3 | 0.607 (0.250, 0.250) | 2.375 | 0 | 10.44 | 12.67 |
| p1p2 | w8_cache3 | 0/1 | - | - | - | - (-, -) | - | 0 | 0.00 | 0.00 |
| pr3917 | w0_dt2 | 1/1 | 9.84 | 49.1 | 6.0 | 1.339 (0.625, 0.696) | 5.181 | 0 | 4.37 | 6.09 |
| pr3917 | w1_dt8 | 1/1 | 8.25 | 43.8 | 3.8 | 1.357 (0.643, 0.696) | 5.181 | 0 | 4.56 | 6.28 |

## Correctness

```
{
 "n_hashed": {
  "base": 32,
  "p1p2": 32,
  "pr3917": 32
 },
 "p1p2_vs_base_identical": true,
 "p1p2_vs_base_first_mismatch": null,
 "mapcheck_pr3917": {
  "n_samples": 16,
  "n_camera_frames": 48,
  "n_equal": 48,
  "max_abs_diff": 0
 },
 "mapcheck_base": {
  "n_samples": 16,
  "n_camera_frames": 48,
  "n_equal": 48,
  "max_abs_diff": 0
 }
}
```

## Prep / hash runs (num_workers=0, discarded for timing)

```
{
 "base": {
  "status": "ok",
  "build_s": 16.781810245942324,
  "first_batch_s": 42.598760323016904,
  "peak_pss_gb": 10.394042015075684,
  "error": null,
  "status_429_total": 0,
  "build_http": {
   "requests": 60,
   "resolve_requests": 8,
   "cdn_requests": 2,
   "api_requests": 50,
   "http_bytes": 16673337,
   "http_s": 5.141382005880587,
   "status_429": 0,
   "status_other_err": 4,
   "cdn_hit": 0,
   "cdn_miss": 0,
   "fetch_calls": 2,
   "fetch_bytes": 279506,
   "fetch_s": 0.38398678600788116,
   "resolver_min_seen": 1073741824,
   "pid": 3276
  }
 },
 "p1p2": {
  "status": "ok",
  "build_s": 4.690831515006721,
  "first_batch_s": 30.119870071997866,
  "peak_pss_gb": 10.460278511047363,
  "error": null,
  "status_429_total": 0,
  "build_http": {
   "requests": 11,
   "resolve_requests": 5,
   "cdn_requests": 2,
   "api_requests": 4,
   "http_bytes": 3373761,
   "http_s": 0.6887028267374262,
   "status_429": 0,
   "status_other_err": 4,
   "cdn_hit": 0,
   "cdn_miss": 0,
   "fetch_calls": 2,
   "fetch_bytes": 279506,
   "fetch_s": 0.26503475510980934,
   "resolver_min_seen": 1073741824,
   "pid": 3364
  }
 },
 "pr3917": {
  "status": "ok",
  "build_s": 224.06114488106687,
  "first_batch_s": 45.36036950489506,
  "peak_pss_gb": 5.343954086303711,
  "error": null,
  "status_429_total": 0,
  "build_http": {
   "requests": 3625,
   "resolve_requests": 1170,
   "cdn_requests": 2340,
   "api_requests": 115,
   "http_bytes": 131084936184,
   "http_s": 385.8892785127973,
   "status_429": 0,
   "status_other_err": 1,
   "cdn_hit": 0,
   "cdn_miss": 0,
   "fetch_calls": 0,
   "fetch_bytes": 0,
   "fetch_s": 0.0,
   "resolver_min_seen": 1073741824,
   "pid": 4744
  }
 }
}
```
