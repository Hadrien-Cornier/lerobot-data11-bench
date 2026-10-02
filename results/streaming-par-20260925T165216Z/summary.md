# Streaming camera-parallelism bench (streaming-par-20260925T165216Z)

`lerobot/abc_130k_v3_train@68651e49`, StreamingLeRobotDataset over hf://, batch 8, 3 cameras, no delta_timestamps.

| arm | regime | ok/runs | samples/s (mean +- 95% CI) | first batch s | build s | req/sample (resolve, cdn) | MB/sample | 429s | peak PSS GB | peak cgroup GB |
|---|---|---|---|---|---|---|---|---|---|---|
| base | w0_cache3 | 3/3 | 1.09 +- 0.21 | 42.4 | 3.3 | 5.776 (2.886, 2.886) | 15.583 | 0 | 10.16 | 12.74 |
| base | w0_cachedef | 3/3 | 40.25 +- 20.63 | 34.4 | 3.3 | 0.095 (0.046, 0.046) | 0.432 | 0 | 10.41 | 16.70 |
| base | w2_cache3 | 3/3 | 2.05 +- 0.42 | 43.3 | 3.5 | 6.566 (3.230, 3.230) | 19.478 | 0 | 21.14 | 24.15 |
| base | w2_cachedef | 3/3 | 31.52 +- 26.04 | 35.0 | 3.3 | 0.961 (0.428, 0.428) | 4.599 | 0 | 21.53 | 28.25 |
| base | w8_cache3 | 0/3 | - | - | - | - (-, -) | - | 0 | 0.00 | 0.00 |
| base | w8_cachedef | 0/3 | - | - | - | - (-, -) | - | 0 | 0.00 | 0.00 |
| p1 | w0_cache3 | 3/3 | 0.85 +- 0.10 | 46.2 | 3.3 | 8.279 (2.885, 2.885) | 16.515 | 0 | 11.02 | 13.72 |
| p1 | w0_cachedef | 3/3 | 28.28 +- 5.40 | 35.4 | 3.3 | 0.112 (0.046, 0.046) | 0.438 | 0 | 10.57 | 13.28 |
| p1 | w2_cache3 | 3/3 | 1.94 +- 0.20 | 45.7 | 3.6 | 7.100 (3.230, 3.230) | 19.660 | 0 | 21.48 | 24.53 |
| p1 | w2_cachedef | 3/3 | 34.86 +- 11.65 | 35.5 | 3.4 | 1.005 (0.428, 0.428) | 4.614 | 0 | 21.65 | 24.84 |
| p1 | w8_cache3 | 0/3 | - | - | - | - (-, -) | - | 0 | 0.00 | 0.00 |
| p1 | w8_cachedef | 0/3 | - | - | - | - (-, -) | - | 0 | 0.00 | 0.00 |
| p1p2 | w0_cache3 | 3/3 | 2.14 +- 0.07 | 28.2 | 3.5 | 8.379 (2.868, 2.868) | 16.316 | 0 | 11.78 | 14.31 |
| p1p2 | w0_cachedef | 3/3 | 38.45 +- 18.52 | 23.8 | 3.4 | 0.112 (0.046, 0.046) | 0.438 | 0 | 10.56 | 13.15 |
| p1p2 | w2_cache3 | 3/3 | 5.35 +- 0.39 | 26.8 | 3.4 | 7.061 (3.230, 3.230) | 19.647 | 0 | 21.52 | 24.54 |
| p1p2 | w2_cachedef | 3/3 | 52.27 +- 10.33 | 23.0 | 3.4 | 1.000 (0.428, 0.428) | 4.612 | 0 | 21.62 | 24.59 |
| p1p2 | w8_cache3 | 0/3 | - | - | - | - (-, -) | - | 0 | 0.00 | 0.00 |
| p1p2 | w8_cachedef | 0/3 | - | - | - | - (-, -) | - | 0 | 0.00 | 0.00 |
| pr3917 | w0_dt2 | 3/3 | 47.05 +- 20.82 | 37.7 | 3.7 | 0.243 (0.113, 0.127) | 0.947 | 0 | 4.62 | 5.96 |
| pr3917 | w0_dt8 | 3/3 | 22.95 +- 7.30 | 34.9 | 3.0 | 0.243 (0.115, 0.125) | 0.937 | 0 | 4.56 | 5.93 |
| pr3917 | w1_dt2 | 3/3 | 54.42 +- 28.24 | 34.5 | 2.9 | 0.243 (0.115, 0.125) | 0.937 | 0 | 4.71 | 6.10 |
| pr3917 | w1_dt8 | 3/3 | 51.29 +- 3.02 | 36.5 | 3.0 | 0.243 (0.115, 0.125) | 0.937 | 0 | 4.70 | 6.03 |

## Correctness

```
{
 "n_hashed": {
  "base": 32,
  "p1": 32,
  "p1p2": 32,
  "pr3917": 32
 },
 "p1_vs_base_identical": true,
 "p1_vs_base_first_mismatch": null,
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
  "build_s": 15.90906325681135,
  "first_batch_s": 35.92140450421721,
  "peak_pss_gb": 10.425856590270996,
  "error": null,
  "status_429_total": 0,
  "build_http": {
   "requests": 60,
   "resolve_requests": 8,
   "cdn_requests": 2,
   "api_requests": 50,
   "http_bytes": 16673337,
   "http_s": 5.053688401822001,
   "status_429": 0,
   "status_other_err": 4,
   "cdn_hit": 0,
   "cdn_miss": 0,
   "fetch_calls": 2,
   "fetch_bytes": 279506,
   "fetch_s": 0.40049449913203716,
   "resolver_min_seen": 1073741824,
   "pid": 3404
  }
 },
 "p1": {
  "status": "ok",
  "build_s": 3.1284902030602098,
  "first_batch_s": 33.68289295816794,
  "peak_pss_gb": 10.43012523651123,
  "error": null,
  "status_429_total": 0,
  "build_http": {
   "requests": 11,
   "resolve_requests": 5,
   "cdn_requests": 2,
   "api_requests": 4,
   "http_bytes": 3373761,
   "http_s": 0.3413266632705927,
   "status_429": 0,
   "status_other_err": 4,
   "cdn_hit": 0,
   "cdn_miss": 0,
   "fetch_calls": 2,
   "fetch_bytes": 279506,
   "fetch_s": 0.24767269426956773,
   "resolver_min_seen": 1073741824,
   "pid": 3579
  }
 },
 "p1p2": {
  "status": "ok",
  "build_s": 3.361741990316659,
  "first_batch_s": 22.411071362905204,
  "peak_pss_gb": 10.17856502532959,
  "error": null,
  "status_429_total": 0,
  "build_http": {
   "requests": 11,
   "resolve_requests": 5,
   "cdn_requests": 2,
   "api_requests": 4,
   "http_bytes": 3373761,
   "http_s": 0.3838298413902521,
   "status_429": 0,
   "status_other_err": 4,
   "cdn_hit": 0,
   "cdn_miss": 0,
   "fetch_calls": 2,
   "fetch_bytes": 279506,
   "fetch_s": 0.3063228181563318,
   "resolver_min_seen": 1073741824,
   "pid": 5031
  }
 },
 "pr3917": {
  "status": "ok",
  "build_s": 167.7505399179645,
  "first_batch_s": 32.07235125172883,
  "peak_pss_gb": 5.284336090087891,
  "error": null,
  "status_429_total": 0,
  "build_http": {
   "requests": 3613,
   "resolve_requests": 1170,
   "cdn_requests": 2340,
   "api_requests": 103,
   "http_bytes": 131080689956,
   "http_s": 318.67496893554926,
   "status_429": 0,
   "status_other_err": 1,
   "cdn_hit": 0,
   "cdn_miss": 0,
   "fetch_calls": 0,
   "fetch_bytes": 0,
   "fetch_s": 0.0,
   "resolver_min_seen": 1073741824,
   "pid": 6462
  }
 }
}
```
