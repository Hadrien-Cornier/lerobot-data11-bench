# DATA-11 round 3 (r3-20260925T182847Z): equal-size files, and PR #3917 on layouts

Job `6ab6bd546b030d633f691efe` (cpu-upgrade, 37 min, about $0.02). It ran 18:28-19:05 UTC. `lerobot-streaming-par-main` had finished about 18:27, and no `lerobot-streaming-par-*` job overlapped, so no arm is affected. 95% CIs use a t-interval over 3 repeats, or a bootstrap of the median for the size check. The 8-reader numbers are one 45 s run each, so they have no CI.

## Analysis (session report)

### 1. File size vs request latency inside HF
Signed link resolved once per file, then 60 random 256 KiB reads per file, interleaved. Median time to first byte (TTFB) in ms, [95% CI]. The diff is the median paired difference against SEP in the same round.

| file | MB | start pass | diff vs SEP | end pass | diff vs SEP |
|---|---|---|---|---|---|
| SEP | 80 | 66 [62, 82] | - | 62 [60, 68] | - |
| BIG | 253 | 94 [79, 112] | +16 [+2, +43] | 69 [62, 85] | +5 [-2, +18] |
| STACK cut | 87 | 81 [70, 88] | +10 [-7, +27] | 66 [61, 71] | +3 [-2, +7] |
| STACK round 2 | 261 | 101 [87, 118] | +27 [+13, +39] | 72 [63, 81] | 0 [-6, +16] |
| MULTI cut | 88 | 107 [69, 136] | +18 [+3, +45] | 80 [72, 102] | +13 [+5, +27] |

- Inside HF every read is 2 to 4 times faster than on the laptop (62 to 107 ms, against 182 to 332 ms).
- There is a small size effect only in the start pass: the 3x bigger files were 16 to 27 ms slower. It is gone in the end pass. The 88 MB MULTI cut was slower than SEP in both passes, so file size alone does not explain the extra latency.
- Response headers arrive after about 31 ms for every file. The rest of the wait is before the first body byte. Host: `cas-bridge-direct.xethub.hf.co`, and x-cache is empty.
- Verdict: inside HF, the file-size effect is small (0 to 27 ms) and unstable. It is far from the laptop's +90 to +150 ms.

### 2. Random access on equal-size files (samples/s, 1 reader, mean +- 95% CI over 3 repeats)

| regime | reader | SEP (parallel cams) | STACK | MULTI (shared cache) |
|---|---|---|---|---|
| warm | fs5M (today) | 4.6 +- 0.7 | 4.5 +- 1.8 | 4.0 +- 2.1 |
| warm | rc256K | 7.7 +- 1.2 | 7.1 +- 2.8 | 6.2 +- 2.9 |
| warm | exact | 7.7 +- 5.1 | 7.3 +- 3.4 | 7.3 +- 0.6 |
| cold | fs5M | 2.2 +- 0.8 | 2.1 +- 0.7 | 2.1 +- 0.7 |
| cold | rc256K | 4.1 +- 1.3 | 4.0 +- 1.1 | 3.1 +- 1.1 |
| cold | exact | 3.8 +- 1.4 | 3.6 +- 2.7 | 3.6 +- 0.9 |

For reference, SEP with sequential cameras and fs5M (LeRobot today): warm 2.1, cold 1.0.

- With 1 reader and files of the same size, the three layouts are equal: every CI overlaps. Fixing the reader matters much more than the layout (about 1.7x warm, about 1.8x cold).
- The best config with 8 readers (samples/s, fetches/s, MB/s):
  - warm: STACK exact 108 (107, 8.7); MULTI exact 111 (110, 10.2); SEP rc256K 73 (219, 71.7).
  - cold: STACK rc256K 50 (99, 32.6); MULTI exact 46 (91, 28.1); SEP rc256K 37 (221, 72.5).
  - Stacking gives about 1.5x warm and about 1.3x cold. SEP makes 3 requests per sample, so at the same request rate it gets fewer samples. There were no 429s.
- Round 2 (3x bigger files), warm, 1 reader: SEP rc256K 8.7, STACK exact 6.8, MULTI exact 12.9 +- 3.3. With 8 readers: STACK 117, MULTI 123, SEP exact 92.
  - Round 3 on equal-size files: STACK 7.3 (unchanged), MULTI 7.3 (its round-2 lead is gone), SEP 7.7.
- The in-job control (round-2 files, run right after): STACK exact warm 13.6 +- 2.5, rc256K 13.7 +- 1.1, and MULTI exact 8.5 +- 2.7. Round 2 measured STACK exact at 6.8 on the same files, so identical files differ by about 2x between runs. The control also ran at a different time (18:49) than the equal-size arms (18:34-18:41), and the round-2 files had been read many times before, while the cuts were new uploads.
- Verdict: round 2's STACK/MULTI edge with 1 reader does not come from file size. Bigger files were not slower here. Most of it looks like run-to-run noise. The edge that holds up is under concurrency: at 8 readers, stacking needs 3x fewer requests.

### 3. PR #3917 on SEP vs STACK (mean +- 95% CI, 3 repeats)
Our datasets are small. With 1 rank the pool (32 episodes plus 8 prefetched = 40) already held every episode the run touched, and with 8 ranks each rank held all 29 of its episodes. So the "second half" rate is the rate of decoding from RAM after the pool fill. It does not cover eviction and refill.

| layout | ranks | rate, 2nd half | first sample s | range GETs/ep | MB/ep | HTTP req/ep | mean GET s | peak PSS GB | decode CPU ms/smp | process CPU ms/smp |
|---|---|---|---|---|---|---|---|---|---|---|
| SEP | 1 | 138 +- 80 | 24 +- 15 | 3 | 25.0 | 9.1 | 3.1 | 2.8 | 3.9 +- 0.8 | 76 +- 77 |
| STACK | 1 | 212 +- 38 | 18 +- 2 | 1 | 24.2 | 6.5 | 7.6 | 2.8 | 3.1 +- 0.3 | 38 +- 6 |
| SEP | 8 | 1320 +- 240 | 16 +- 5 | 3 | 24.7 | 8.4 | 4.5 | 16.3 total | 4.0 +- 0.4 | 13 +- 6 |
| STACK | 8 | 1900 +- 521 | 19 +- 1 | 1 | 23.8 | 6.4 | 12.1 | 17.0 total | 3.2 +- 0.1 | 10 +- 4 |

- STACK is about 1.4 to 1.5x faster once the pool is full. That speed comes from CPU: fewer decode calls per sample and less process CPU. The network adds nothing here.
- Both layouts fetch the same bytes per episode, and STACK makes 3x fewer range GETs. The time to the first sample is about equal, because the byte count is the same.
- Peak memory is the same for both layouts: about 2.8 GB per process, and 16 to 17 GB for 8 ranks. Close to 24 GB of the 32 GB box was in use, counting page cache.
- The decode CPU I measure is thread CPU inside `_get_frames` only. FFmpeg worker threads may be missed.

### 4. Correctness, and what is not verified
- Checks that passed:
  - All 18 cuts: packets, pts, keyframes and packet bytes identical, plus sampled frames bit-exact against the originals and against SEP.
  - Remote decode equal to local decode: 40 of 40 (equal-size arms) and 12 of 12 (control).
  - No pts violations and no 429s.
  - #3917 on SEP and STACK: 40 of 40 samples bit-exact.
- #3917 on the multi-track file does not fail, it returns wrong data. Every key returns track 0 (40 of 40 samples of `left_wrist` and `right_wrist` equal `top`).
- Not verified:
  - Why identical files run about 2x apart between runs (time of day, how new the file is, Xet caching).
  - Whether new uploads are slower on their first reads.
  - Any 8-reader CI (one run each).
  - #3917 with pool turnover, and on larger datasets.
  - Decode CPU in FFmpeg threads.
  - "8 workers" means 8 ranks, not DataLoader workers.
  - The MULTI test with #3917 was local only.
  - One rank of SEP p8 repeat 2 failed with an httpx ReadTimeout after #3917's retries. That repeat counts 7 of 8 ranks.

---


## Equal-size variants

Cuts uploaded to `variants/r3-20260925T182847Z` at `5efceaf32b5cdcc90e949f391a2ffc3b6d2d714a`. Stream copy (PyAV packet remux), each cut starts at a keyframe, pts rebased to 0.

| layout | group | part | frames [a, b) | frames | MB | check |
|---|---|---|---|---|---|---|
| STACK | 0 | 0 | [0, 8266) | 8266 | 70.5 | pass (32 frames, packets True, pts True, bytes True) |
| MULTI | 0 | 0 | [0, 8266) | 8266 | 71.9 | pass (192 frames, packets True, pts True, bytes True) |
| STACK | 0 | 1 | [8266, 14066) | 5800 | 70.5 | pass (32 frames, packets True, pts True, bytes True) |
| MULTI | 0 | 1 | [8266, 14066) | 5800 | 71.6 | pass (192 frames, packets True, pts True, bytes True) |
| STACK | 0 | 2 | [14066, 23380) | 9314 | 70.5 | pass (32 frames, packets True, pts True, bytes True) |
| MULTI | 0 | 2 | [14066, 23380) | 9314 | 71.9 | pass (192 frames, packets True, pts True, bytes True) |
| STACK | 1 | 0 | [0, 6626) | 6626 | 86.8 | pass (32 frames, packets True, pts True, bytes True) |
| MULTI | 1 | 0 | [0, 6626) | 6626 | 88.1 | pass (192 frames, packets True, pts True, bytes True) |
| STACK | 1 | 1 | [6626, 17092) | 10466 | 86.9 | pass (32 frames, packets True, pts True, bytes True) |
| MULTI | 1 | 1 | [6626, 17092) | 10466 | 88.6 | pass (192 frames, packets True, pts True, bytes True) |
| STACK | 1 | 2 | [17092, 25759) | 8667 | 86.8 | pass (32 frames, packets True, pts True, bytes True) |
| MULTI | 1 | 2 | [17092, 25759) | 8667 | 88.5 | pass (192 frames, packets True, pts True, bytes True) |
| STACK | 2 | 0 | [0, 7336) | 7336 | 59.1 | pass (32 frames, packets True, pts True, bytes True) |
| MULTI | 2 | 0 | [0, 7336) | 7336 | 60.3 | pass (192 frames, packets True, pts True, bytes True) |
| STACK | 2 | 1 | [7336, 14600) | 7264 | 59.1 | pass (32 frames, packets True, pts True, bytes True) |
| MULTI | 2 | 1 | [7336, 14600) | 7264 | 60.3 | pass (192 frames, packets True, pts True, bytes True) |
| STACK | 2 | 2 | [14600, 24979) | 10379 | 59.1 | pass (32 frames, packets True, pts True, bytes True) |
| MULTI | 2 | 2 | [14600, 24979) | 10379 | 60.8 | pass (192 frames, packets True, pts True, bytes True) |

SEP files (unchanged): 80, 69, 66, 105, 82, 78, 68, 57, 57 MB.
- dataset SEP: hadriencornier/lerobot-data11-r3-sep @ 31fd7b6605, 232 episodes, 592944 frames, 24 video files per key
- dataset STACK: hadriencornier/lerobot-data11-r3-stack @ aa84869aa0, 232 episodes, 592944 frames, 72 video files per key
- dataset MULTI: local @ None, 10 episodes, 23380 frames, 3 video files per key

## File size vs range-request latency inside HF (size)

Signed CDN link resolved once per file; 60 random 256 KiB range GETs per file, interleaved, one at a time. TTFB = time from sending the request to the first body byte. ms.

| file | MB | n | TTFB p50 | IQR (p25-p75) | headers p50 | total p50 | total p90 | x-cache |
|---|---|---|---|---|---|---|---|---|
| SEP (`ckpt_full_g000__top.mp4`) | 80 | 60 | 66 | 59-115 | 30 | 72 | 139 | {'': 60} |
| BIG (`ckpt_full_b000__top.mp4`) | 253 | 60 | 94 | 62-146 | 32 | 100 | 203 | {'': 60} |
| STACK_eq (`ckpt_full_g001_p0.mp4`) | 87 | 60 | 81 | 61-106 | 32 | 89 | 131 | {'': 60} |
| STACK_r2 (`ckpt_full_g001.mp4`) | 261 | 60 | 101 | 69-160 | 31 | 110 | 203 | {'': 60} |
| MULTI_eq (`ckpt_full_g001_p0.mp4`) | 88 | 60 | 107 | 59-154 | 31 | 117 | 194 | {'': 60} |

Hosts: ['cas-bridge-direct.xethub.hf.co']. Wall 35 s, start 2026-09-25T18:33:07Z.


## File size vs range-request latency inside HF (size2)

Signed CDN link resolved once per file; 60 random 256 KiB range GETs per file, interleaved, one at a time. TTFB = time from sending the request to the first body byte. ms.

| file | MB | n | TTFB p50 | IQR (p25-p75) | headers p50 | total p50 | total p90 | x-cache |
|---|---|---|---|---|---|---|---|---|
| SEP (`ckpt_full_g000__top.mp4`) | 80 | 60 | 62 | 57-92 | 32 | 68 | 118 | {'': 60} |
| BIG (`ckpt_full_b000__top.mp4`) | 253 | 60 | 69 | 57-104 | 32 | 77 | 175 | {'': 60} |
| STACK_eq (`ckpt_full_g001_p0.mp4`) | 87 | 60 | 66 | 59-80 | 32 | 73 | 105 | {'': 60} |
| STACK_r2 (`ckpt_full_g001.mp4`) | 261 | 60 | 72 | 60-91 | 32 | 79 | 137 | {'': 60} |
| MULTI_eq (`ckpt_full_g001_p0.mp4`) | 88 | 60 | 80 | 64-130 | 31 | 83 | 163 | {'': 60} |

Hosts: ['cas-bridge-direct.xethub.hf.co']. Wall 29 s, start 2026-09-25T19:05:09Z.


## Random access, equal-size files

Repo `hadriencornier/lerobot-data11-bench@5efceaf32b`. sps = samples/s, one reader. fetch = one HTTP request that moves video bytes (CDN). fs* readers also pay 1 resolve per fetch; rc / exact resolve once per file (untimed, shown in `presigned`: 27 resolves for 27 files).

| regime | layout | cams | reader | mode | sps | +-CI | p50 ms | p95 ms | fetch/smp | resolve/smp | API/smp | MB/smp | dec MB/smp | open ms/smp | fallback/smp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold | MULTI | par | fs5M | shared | 2.13 | 0.73 | 470 | 759 | 1.92 | 1.92 | 0.00 | 10.062 | 1.143 | 208 | 0.00 |
| cold | MULTI | par | rc256K | shared | 3.05 | 1.05 | 323 | 582 | 2.75 | 0.00 | 0.00 | 0.764 | 1.143 | 150 | 0.00 |
| cold | MULTI | par | exact | shared | 3.57 | 0.91 | 245 | 426 | 2.00 | 0.00 | 0.00 | 0.613 | 1.143 | 123 | 0.00 |
| cold | SEP | par | fs5M | - | 2.15 | 0.77 | 421 | 763 | 5.60 | 5.60 | 0.00 | 29.466 | 0.846 | 622 | 0.00 |
| cold | SEP | par | rc256K | - | 4.11 | 1.29 | 235 | 450 | 6.00 | 0.00 | 0.00 | 1.966 | 0.846 | 270 | 0.00 |
| cold | SEP | par | exact | - | 3.77 | 1.39 | 251 | 469 | 6.00 | 0.00 | 0.00 | 1.455 | 0.846 | 318 | 0.00 |
| cold | SEP | seq | fs5M | - | 1.03 | 0.38 | 941 | 1725 | 5.60 | 5.60 | 0.00 | 29.466 | 0.846 | 470 | 0.00 |
| cold | STACK | - | fs5M | - | 2.12 | 0.74 | 448 | 989 | 1.92 | 1.92 | 0.00 | 9.893 | 0.177 | 214 | 0.00 |
| cold | STACK | - | rc256K | - | 3.97 | 1.09 | 228 | 442 | 2.00 | 0.00 | 0.00 | 0.654 | 0.177 | 109 | 0.00 |
| cold | STACK | - | exact | - | 3.60 | 2.66 | 229 | 680 | 2.00 | 0.00 | 0.00 | 0.398 | 0.177 | 130 | 0.00 |
| warm | MULTI | par | fs5M | shared | 4.02 | 2.10 | 268 | 542 | 0.81 | 0.81 | 0.00 | 4.286 | 0.196 | 0 | 0.00 |
| warm | MULTI | par | rc256K | shared | 6.20 | 2.93 | 163 | 331 | 0.99 | 0.00 | 0.00 | 0.321 | 0.196 | 0 | 0.00 |
| warm | MULTI | par | exact | shared | 7.28 | 0.57 | 141 | 280 | 1.00 | 0.00 | 0.00 | 0.097 | 0.196 | 0 | 0.00 |
| warm | SEP | par | fs5M | - | 4.58 | 0.70 | 219 | 373 | 2.72 | 2.72 | 0.00 | 13.296 | 0.197 | 0 | 0.00 |
| warm | SEP | par | rc256K | - | 7.73 | 1.16 | 115 | 267 | 3.00 | 0.00 | 0.00 | 0.974 | 0.197 | 0 | 0.00 |
| warm | SEP | par | exact | - | 7.68 | 5.05 | 104 | 295 | 3.00 | 0.00 | 0.00 | 0.215 | 0.197 | 0 | 0.00 |
| warm | SEP | seq | fs5M | - | 2.09 | 0.78 | 500 | 870 | 2.72 | 2.72 | 0.00 | 13.296 | 0.197 | 0 | 0.00 |
| warm | STACK | - | fs5M | - | 4.49 | 1.75 | 207 | 429 | 0.89 | 0.89 | 0.00 | 4.261 | 0.065 | 0 | 0.00 |
| warm | STACK | - | rc256K | - | 7.14 | 2.80 | 127 | 307 | 1.00 | 0.00 | 0.00 | 0.325 | 0.065 | 0 | 0.00 |
| warm | STACK | - | exact | - | 7.30 | 3.36 | 109 | 264 | 1.00 | 0.00 | 0.00 | 0.084 | 0.065 | 0 | 0.00 |

### 8 concurrent reader processes

| config | sps total | fetch/s | resolves/s | MB/s | p50 ms | p95 ms | 429 | other err |
|---|---|---|---|---|---|---|---|---|
| MULTI|par|exact|shared|warm | 111.2 | 109.9 | 0.0 | 10.2 | 66 | 109 | 0 | 0 |
| MULTI|par|exact|shared|cold | 45.8 | 91.4 | 0.0 | 28.1 | 168 | 228 | 0 | 0 |
| SEP|par|rc256K|-|warm | 73.4 | 219.3 | 0.0 | 71.7 | 100 | 179 | 0 | 0 |
| SEP|par|rc256K|-|cold | 37.0 | 221.5 | 0.0 | 72.5 | 199 | 346 | 0 | 0 |
| STACK|-|exact|-|warm | 108.4 | 107.1 | 0.0 | 8.7 | 65 | 113 | 0 | 0 |
| STACK|-|rc256K|-|cold | 49.8 | 99.4 | 0.0 | 32.6 | 154 | 220 | 0 | 0 |

Checks: remote == local decode (bit-exact) 40/40; pts violations > 1e-4 s: 0; skipped arms 0; 429 total 0.


## Random access, control: round-2 files (3x bigger)

Repo `hadriencornier/lerobot-data11-bench@5efceaf32b`. sps = samples/s, one reader. fetch = one HTTP request that moves video bytes (CDN). fs* readers also pay 1 resolve per fetch; rc / exact resolve once per file (untimed, shown in `presigned`: 6 resolves for 6 files).

| regime | layout | cams | reader | mode | sps | +-CI | p50 ms | p95 ms | fetch/smp | resolve/smp | API/smp | MB/smp | dec MB/smp | open ms/smp | fallback/smp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold | MULTI | par | exact | shared | 3.29 | 1.18 | 278 | 545 | 2.00 | 0.00 | 0.00 | 1.106 | 2.568 | 165 | 0.00 |
| cold | STACK | - | rc256K | - | 6.38 | 0.91 | 154 | 212 | 2.00 | 0.00 | 0.00 | 0.655 | 0.283 | 77 | 0.00 |
| cold | STACK | - | exact | - | 5.39 | 1.94 | 178 | 328 | 2.00 | 0.00 | 0.00 | 0.499 | 0.283 | 101 | 0.00 |
| warm | MULTI | par | exact | shared | 8.50 | 2.67 | 117 | 217 | 1.00 | 0.00 | 0.00 | 0.097 | 0.197 | 0 | 0.00 |
| warm | STACK | - | rc256K | - | 13.72 | 1.07 | 66 | 117 | 1.00 | 0.00 | 0.00 | 0.328 | 0.066 | 0 | 0.00 |
| warm | STACK | - | exact | - | 13.61 | 2.49 | 67 | 141 | 1.00 | 0.00 | 0.00 | 0.084 | 0.066 | 0 | 0.00 |

Checks: remote == local decode (bit-exact) 12/12; pts violations > 1e-4 s: 0; skipped arms 0; 429 total 0.


## DATA-11 round 3: PR #3917 on SEP vs STACK (equal-size files)

p1 = 1 process; p8 = 8 processes as ranks 0..7 of WORLD_SIZE 8 (#3917 allows at most 1 DataLoader worker per rank). sps = samples/s after the first sample, summed over processes; sps 2nd half = rate over the second half of each run's window (after the pool fill) (1 sample = 1 time step, 3 camera images). mean +- sd over repeats. dec CPU = thread CPU inside #3917's decode calls, per sample; proc CPU = process CPU per sample.

| layout | mode | reps | sps | sps 2nd half | first sample s | range GETs/episode | MB/episode | HTTP req/episode | mean GET s | dec CPU ms/smp | proc CPU ms/smp | peak PSS GB (sum) | cgroup peak MB | anon peak MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SEP | p1 | 3 | 117.2 +- 48.0 | 137.5 +- 43.6 | 24.29 +- 8.02 | 3.00 +- 0.00 | 25.0 +- 3.0 | 9.1 +- 0.1 | 3.12 +- 0.77 | 3.93 +- 0.44 | 76.23 +- 41.90 | 2.84 +- 0.22 | 9247 +- 234 | 2666 +- 226 |
| SEP | p8 | 3 | 650.1 +- 179.1 | 1319.7 +- 130.6 | 15.96 +- 2.64 | 3.00 +- 0.00 | 24.7 +- 0.0 | 8.4 +- 0.1 | 4.52 +- 0.55 | 3.95 +- 0.22 | 13.25 +- 3.44 | 16.25 +- 1.71 | 23310 +- 1135 | 16494 +- 1180 |
| STACK | p1 | 3 | 198.8 +- 27.4 | 212.4 +- 20.6 | 18.26 +- 1.14 | 1.00 +- 0.00 | 24.2 +- 2.8 | 6.5 +- 0.1 | 7.55 +- 0.85 | 3.10 +- 0.14 | 38.15 +- 3.19 | 2.83 +- 0.19 | 9287 +- 199 | 2710 +- 198 |
| STACK | p8 | 3 | 875.3 +- 161.4 | 1899.6 +- 283.5 | 18.60 +- 0.54 | 1.00 +- 0.00 | 23.8 +- 0.0 | 6.4 +- 0.1 | 12.13 +- 1.14 | 3.23 +- 0.05 | 10.05 +- 2.03 | 17.02 +- 0.22 | 23584 +- 268 | 16776 +- 258 |

### Correctness checks (return_uint8=True, decoded frame == local TorchCodec decode of the source frame)

- SEP: status ok, build 22.9 s, pass True, shape [3, 224, 224], per key {"observation.images.left_wrist": {"equal_own": 40, "equal_cam": [0, 40, 0], "n": 40}, "observation.images.right_wrist": {"equal_own": 40, "equal_cam": [0, 0, 40], "n": 40}, "observation.images.top": {"equal_own": 40, "equal_cam": [40, 0, 0], "n": 40}} 
- STACK: status ok, build 11.5 s, pass True, shape [3, 672, 224], per key {"observation.images.stack": {"equal_own": 40, "equal_cam": [0, 0, 0], "n": 40}} 
- MULTI: status ok, build 4.3 s, pass False, shape [3, 224, 224], per key {"observation.images.left_wrist": {"equal_own": 0, "equal_cam": [40, 0, 0], "n": 40}, "observation.images.right_wrist": {"equal_own": 0, "equal_cam": [40, 0, 0], "n": 40}, "observation.images.top": {"equal_own": 40, "equal_cam": [40, 0, 0], "n": 40}} 

MULTI = local dataset (not on the Hub) whose 3 video keys point at the multi-track cuts. equal_cam[i] counts samples equal to SEP camera i; a correct reader gives equal_own == n for every key.


## Stages

```
stage	status	seconds	finished_utc	cgroup_peak_mb
system_deps	ok	48	2026-09-25T18:29:36Z	835
bundle	ok	1	2026-09-25T18:29:37Z	835
python_env	ok	84	2026-09-25T18:31:01Z	3216
prep	ok	123	2026-09-25T18:33:06Z	7281
size	ok	35	2026-09-25T18:33:42Z	7281
access	ok	968	2026-09-25T18:49:51Z	9478
pr3917	ok	914	2026-09-25T19:05:07Z	23991
size2	ok	29	2026-09-25T19:05:38Z	23991
```
