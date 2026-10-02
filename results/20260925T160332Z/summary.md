# DATA-11 round 2 (20260925T160332Z)

Question: does STACK (3 cameras in one video) beat SEP (one MP4 per camera) over hf://, and do cheaper reader fixes close the gap? Option C = one MP4 with 3 video tracks (stream copy of SEP). Raw numbers: prep/prep.json, stream/stream.json, access/access.json.

## Option C build (multi-track MP4, stream copy)

| group | MB | SEP sum MB | same-timestamp span p50 B | p95 B | gap p50 B | p95 B | packets per run | identity (bit-exact) |
|---|---|---|---|---|---|---|---|---|
| 0 | 215.3 | 215.0 | 6414 | 25158 | 4504 | 18285 | 1.00 | PASS 120/120 |
| 1 | 265.2 | 264.9 | 7890 | 28628 | 5726 | 19607 | 1.00 | PASS 120/120 |
| 2 | 181.4 | 181.1 | 6924 | 19189 | 5507 | 12642 | 1.00 | PASS 120/120 |
- muxer `-max_interleave_delta 0 -movflags +faststart`: span p50 6414 B, p95 25158 B, packets per run 1.00, decode True
- muxer `-movflags +frag_keyframe+empty_moov+default_base_moof -frag_duration 66667`: span p50 14148 B, p95 25404 B, packets per run 2.00, decode True

MULTI uploaded at `a3d31680dba3ed626ee7c5afca85753b23941b52` (read-back ok: True).

---

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


---

# DATA-11 round 2: random access over hf:// (family ckpt_full, 3 cameras per sample)

Repo `hadriencornier/lerobot-data11-bench@a3d31680db`. sps = samples/s, one reader. fetch = one HTTP request that moves video bytes (CDN). fs* readers also pay 1 resolve per fetch; rc / exact resolve once per file (untimed, shown in `presigned`: 15 resolves for 15 files).

| regime | layout | cams | reader | mode | sps | +-CI | p50 ms | p95 ms | fetch/smp | resolve/smp | API/smp | MB/smp | dec MB/smp | open ms/smp | fallback/smp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold | MULTI | par | fs256K | naive | 1.10 | 0.24 | 858 | 1222 | 12.00 | 12.00 | 0.00 | 3.726 | 2.538 | 747 | 0.00 |
| cold | MULTI | par | fs256K | shared | 2.60 | 0.53 | 380 | 512 | 4.62 | 4.62 | 0.00 | 1.276 | 2.538 | 280 | 0.00 |
| cold | MULTI | par | fs5M | naive | 1.84 | 0.33 | 513 | 724 | 5.81 | 5.81 | 0.00 | 30.589 | 2.538 | 337 | 0.00 |
| cold | MULTI | par | fs5M | shared | 3.67 | 1.71 | 228 | 557 | 1.94 | 1.94 | 0.00 | 10.225 | 2.538 | 121 | 0.00 |
| cold | MULTI | par | exact | shared | 5.37 | 1.06 | 162 | 275 | 2.00 | 0.00 | 0.00 | 1.096 | 2.538 | 101 | 0.00 |
| cold | MULTI | seq | fs256K | naive | 0.89 | 0.02 | 1085 | 1481 | 12.00 | 12.00 | 0.00 | 3.726 | 2.538 | 763 | 0.00 |
| cold | MULTI | seq | fs256K | shared | 2.52 | 0.58 | 384 | 642 | 4.67 | 4.67 | 0.00 | 1.276 | 2.538 | 278 | 0.00 |
| cold | MULTI | seq | fs5M | naive | 1.33 | 0.48 | 744 | 1221 | 5.81 | 5.81 | 0.00 | 30.589 | 2.538 | 349 | 0.00 |
| cold | MULTI | seq | fs5M | shared | 3.62 | 1.32 | 234 | 475 | 1.94 | 1.94 | 0.00 | 10.225 | 2.538 | 125 | 0.00 |
| cold | MULTI | seq | exact | shared | 4.75 | 1.21 | 174 | 469 | 2.00 | 0.00 | 0.00 | 1.096 | 2.538 | 111 | 0.00 |
| cold | SEP | par | fs64K | - | 2.97 | 0.88 | 315 | 550 | 9.00 | 9.00 | 0.00 | 1.043 | 0.840 | 525 | 0.00 |
| cold | SEP | par | fs256K | - | 3.74 | 2.07 | 292 | 433 | 6.00 | 6.00 | 0.00 | 1.966 | 0.840 | 298 | 0.00 |
| cold | SEP | par | fs1M | - | 3.45 | 2.17 | 261 | 548 | 5.85 | 5.85 | 0.00 | 6.477 | 0.840 | 338 | 0.00 |
| cold | SEP | par | fs5M | - | 2.63 | 0.53 | 388 | 523 | 5.77 | 5.77 | 0.00 | 29.906 | 0.840 | 487 | 0.00 |
| cold | SEP | par | rc256K | - | 4.52 | 0.90 | 197 | 310 | 6.00 | 0.00 | 0.00 | 1.964 | 0.840 | 246 | 0.00 |
| cold | SEP | par | exact | - | 4.10 | 1.46 | 244 | 413 | 5.94 | 0.00 | 0.00 | 1.444 | 0.840 | 280 | 0.00 |
| cold | SEP | seq | fs64K | - | 1.14 | 0.39 | 828 | 1534 | 9.00 | 9.00 | 0.00 | 1.043 | 0.840 | 516 | 0.00 |
| cold | SEP | seq | fs256K | - | 1.72 | 0.36 | 534 | 850 | 6.00 | 6.00 | 0.00 | 1.966 | 0.840 | 266 | 0.00 |
| cold | SEP | seq | fs1M | - | 1.63 | 0.41 | 568 | 900 | 5.85 | 5.85 | 0.00 | 6.477 | 0.840 | 300 | 0.00 |
| cold | SEP | seq | fs5M | - | 1.19 | 0.48 | 782 | 1771 | 5.77 | 5.77 | 0.00 | 29.906 | 0.840 | 393 | 0.00 |
| cold | SEP | seq | rc256K | - | 1.95 | 0.87 | 438 | 872 | 6.00 | 0.00 | 0.00 | 1.964 | 0.840 | 235 | 0.00 |
| cold | SEP | seq | exact | - | 1.90 | 0.40 | 508 | 725 | 5.94 | 0.00 | 0.00 | 1.444 | 0.840 | 269 | 0.00 |
| cold | STACK | - | fs64K | - | 2.68 | 0.69 | 354 | 648 | 3.00 | 3.00 | 0.00 | 0.349 | 0.281 | 185 | 0.00 |
| cold | STACK | - | fs256K | - | 3.40 | 0.73 | 258 | 579 | 2.00 | 2.00 | 0.00 | 0.657 | 0.281 | 108 | 0.00 |
| cold | STACK | - | fs1M | - | 3.32 | 0.62 | 278 | 621 | 1.98 | 1.98 | 0.00 | 2.203 | 0.281 | 111 | 0.00 |
| cold | STACK | - | fs5M | - | 2.27 | 0.96 | 409 | 883 | 1.94 | 1.94 | 0.00 | 10.196 | 0.281 | 165 | 0.00 |
| cold | STACK | - | rc256K | - | 3.95 | 0.69 | 225 | 573 | 2.00 | 0.00 | 0.00 | 0.655 | 0.281 | 90 | 0.00 |
| cold | STACK | - | exact | - | 3.86 | 0.57 | 212 | 551 | 2.00 | 0.00 | 0.00 | 0.496 | 0.281 | 101 | 0.00 |
| warm | MULTI | par | fs256K | naive | 8.28 | 1.04 | 95 | 247 | 3.00 | 3.00 | 0.00 | 0.985 | 0.197 | 0 | 0.00 |
| warm | MULTI | par | fs256K | shared | 10.47 | 2.24 | 76 | 237 | 1.02 | 1.02 | 0.00 | 0.324 | 0.197 | 0 | 0.00 |
| warm | MULTI | par | fs5M | naive | 5.43 | 1.85 | 169 | 361 | 2.97 | 2.97 | 0.00 | 15.679 | 0.197 | 0 | 0.00 |
| warm | MULTI | par | fs5M | shared | 8.28 | 2.68 | 110 | 338 | 0.83 | 0.83 | 0.00 | 4.486 | 0.197 | 0 | 0.00 |
| warm | MULTI | par | exact | shared | 12.95 | 3.29 | 64 | 154 | 1.00 | 0.00 | 0.00 | 0.094 | 0.197 | 0 | 0.00 |
| warm | MULTI | seq | fs256K | naive | 3.45 | 0.95 | 260 | 563 | 3.00 | 3.00 | 0.00 | 0.985 | 0.197 | 0 | 0.00 |
| warm | MULTI | seq | fs256K | shared | 10.52 | 1.15 | 84 | 173 | 1.02 | 1.02 | 0.00 | 0.324 | 0.197 | 0 | 0.00 |
| warm | MULTI | seq | fs5M | naive | 2.58 | 0.28 | 351 | 610 | 2.97 | 2.97 | 0.00 | 15.679 | 0.197 | 0 | 0.00 |
| warm | MULTI | seq | fs5M | shared | 7.93 | 5.58 | 110 | 354 | 0.83 | 0.83 | 0.00 | 4.486 | 0.197 | 0 | 0.00 |
| warm | MULTI | seq | exact | shared | 12.84 | 3.32 | 66 | 206 | 1.00 | 0.00 | 0.00 | 0.094 | 0.197 | 0 | 0.00 |
| warm | SEP | par | fs64K | - | 7.64 | 1.57 | 110 | 235 | 3.00 | 3.00 | 0.00 | 0.395 | 0.197 | 0 | 0.00 |
| warm | SEP | par | fs256K | - | 7.76 | 2.34 | 110 | 241 | 3.00 | 3.00 | 0.00 | 0.985 | 0.197 | 0 | 0.00 |
| warm | SEP | par | fs1M | - | 7.39 | 2.37 | 112 | 240 | 2.97 | 2.97 | 0.00 | 3.300 | 0.197 | 0 | 0.00 |
| warm | SEP | par | fs5M | - | 4.25 | 2.60 | 183 | 594 | 2.80 | 2.80 | 0.00 | 14.629 | 0.197 | 0 | 0.00 |
| warm | SEP | par | rc256K | - | 8.66 | 3.83 | 103 | 210 | 3.00 | 0.00 | 0.00 | 0.983 | 0.197 | 0 | 0.00 |
| warm | SEP | par | exact | - | 8.41 | 6.38 | 91 | 325 | 2.94 | 0.00 | 0.00 | 0.210 | 0.197 | 0 | 0.00 |
| warm | SEP | seq | fs64K | - | 3.38 | 1.86 | 246 | 627 | 3.00 | 3.00 | 0.00 | 0.395 | 0.197 | 0 | 0.00 |
| warm | SEP | seq | fs256K | - | 3.47 | 1.51 | 249 | 527 | 3.00 | 3.00 | 0.00 | 0.985 | 0.197 | 0 | 0.00 |
| warm | SEP | seq | fs1M | - | 3.43 | 1.28 | 254 | 512 | 2.97 | 2.97 | 0.00 | 3.300 | 0.197 | 0 | 0.00 |
| warm | SEP | seq | fs5M | - | 2.55 | 1.58 | 351 | 972 | 2.80 | 2.80 | 0.00 | 14.629 | 0.197 | 0 | 0.00 |
| warm | SEP | seq | rc256K | - | 4.08 | 1.91 | 209 | 463 | 3.00 | 0.00 | 0.00 | 0.983 | 0.197 | 0 | 0.00 |
| warm | SEP | seq | exact | - | 4.37 | 0.76 | 222 | 356 | 2.94 | 0.00 | 0.00 | 0.210 | 0.197 | 0 | 0.00 |
| warm | STACK | - | fs64K | - | 6.28 | 1.45 | 168 | 343 | 1.00 | 1.00 | 0.00 | 0.132 | 0.066 | 0 | 0.00 |
| warm | STACK | - | fs256K | - | 6.45 | 1.33 | 151 | 326 | 1.00 | 1.00 | 0.00 | 0.328 | 0.066 | 0 | 0.00 |
| warm | STACK | - | fs1M | - | 5.91 | 1.96 | 153 | 436 | 1.00 | 1.00 | 0.00 | 1.115 | 0.066 | 0 | 0.00 |
| warm | STACK | - | fs5M | - | 4.37 | 2.16 | 194 | 625 | 0.99 | 0.99 | 0.00 | 5.226 | 0.066 | 0 | 0.00 |
| warm | STACK | - | rc256K | - | 7.16 | 3.04 | 131 | 320 | 1.00 | 0.00 | 0.00 | 0.328 | 0.066 | 0 | 0.00 |
| warm | STACK | - | exact | - | 6.77 | 2.97 | 124 | 343 | 1.00 | 0.00 | 0.00 | 0.082 | 0.066 | 0 | 0.00 |

## 8 concurrent reader processes

| config | sps total | fetch/s | resolves/s | MB/s | p50 ms | p95 ms | 429 | other err |
|---|---|---|---|---|---|---|---|---|
| SEP|seq|fs5M|-|warm | 25.0 | 68.9 | 68.9 | 347.4 | 329 | 432 | 0 | 0 |
| STACK|-|fs5M|-|warm | 68.7 | 66.7 | 66.7 | 348.8 | 108 | 163 | 0 | 0 |
| SEP|par|rc256K|-|warm | 89.8 | 267.8 | 0.0 | 87.5 | 77 | 119 | 0 | 0 |
| SEP|par|exact|-|warm | 92.1 | 272.4 | 0.0 | 18.8 | 76 | 128 | 0 | 0 |
| STACK|-|exact|-|warm | 117.3 | 116.1 | 0.0 | 9.5 | 61 | 107 | 0 | 0 |
| MULTI|par|exact|shared|warm | 123.3 | 121.9 | 0.0 | 11.4 | 60 | 102 | 0 | 0 |
| MULTI|par|fs5M|shared|warm | 104.5 | 65.7 | 65.7 | 344.3 | 103 | 157 | 0 | 0 |
| SEP|par|exact|-|cold | 36.8 | 220.5 | 0.0 | 53.4 | 199 | 325 | 0 | 0 |
| STACK|-|exact|-|cold | 50.6 | 101.1 | 0.0 | 25.2 | 141 | 232 | 0 | 0 |
| MULTI|par|exact|shared|cold | 42.7 | 85.2 | 0.0 | 47.0 | 180 | 235 | 0 | 0 |

Checks: remote == local decode (bit-exact) 112/112; pts violations > 1e-4 s: 0; skipped arms 0; 429 total 0.



---

# Round 2 final analysis (written after the job, from stream.json and access.json)

Job `6ab69b4b6b030d633f691998`, 3663 s on cpu-upgrade (about $0.03; round 2 total about $0.035). 3 repeats per arm; +-CI = 95% t-interval over the 3 repeat means (t = 4.3, so intervals are wide).

## 1. Real streaming (StreamingLeRobotDataset, #4702 branch, one process)
Each pair below has the same shard count (S), so the comparisons are clean.
- Cache fits, S = 8: SEP 182 +-30 frames/s, STACK 342 +-17 (1.9x). S = 1: SEP 182 +-44, STACK 251 +-66. Network is about 0 for both (one 5 MiB block holds about 1,700 SEP or 600 STACK frames; measured 0.0017 fetch/frame for STACK S = 1).
- Cache overflows (cache 12 with S = 8, same hit rate as S = 64 with cache 100): SEP 4.6 +-1.5 frames/s, decoder hit 0.51, 1.48 opens and 7.9 MB per frame. STACK still fits (8 <= 12): 308 +-81, hit 1.00. This is a 67x gap.
- Parallel cameras (thread pool per frame, like dataset_reader.py:419) are slower when the cache fits: S = 8, 136 +-13 vs 182 +-30 sequential; S = 1, 146 +-19 vs 182 +-44. CPU per frame goes from 5.5 to 9.3 ms. With 8 processes: 863 vs 1552 frames/s. When the cache overflows, parallel does not help (4.9 +-1.1 vs 4.6 +-1.5), because VideoDecoderCache holds its lock while it opens a file.
- 8 processes: STACK 1948, SEP 1552 frames/s (cache fits). With overflow, SEP falls to 36 frames/s at 54 resolves/s and 284 MB/s. STACK with cache 12: 2410.

## 2. Random access, one reader: see the access table above
- The default reader (fs5M) is the worst warm reader for every layout. SEP seq warm: 2.6 +-1.6 samples/s.
- Block size sweep (warm): 64 KiB, 256 KiB and 1 MiB are about equal (SEP par 7.4 to 7.8, STACK 5.9 to 6.5); 5 MiB is worse (4.3 to 4.4). Cold: 64 KiB needs 9 fetches per sample (vs 6) and is slower, so 256 KiB is the best cold block size.
- Reusing the signed link (rc256K vs fs256K): +12 to 17% warm, +16 to 30% cold. CIs overlap.
- Exact-range: the fewest bytes (SEP 0.21 MB, STACK 0.08, MULTI 0.09 per warm sample). Its speed is about the same as rc256K.
- Parallel cameras help SEP in random access: about 2x warm, 2x cold.
- Best configs, warm: MULTI exact shared 12.9 +-3.3 (par) and 12.8 +-3.3 (seq), SEP par rc256K 8.7 +-3.8, SEP par exact 8.4 +-6.4, STACK rc256K 7.2 +-3.0, STACK exact 6.8 +-3.0. Cold: MULTI exact par 5.4 +-1.1, SEP par rc256K 4.5 +-0.9, SEP par exact 4.1 +-1.5, STACK rc256K 4.0 +-0.7.
- C (MULTI) with 3 naive handles reads the same bytes 3 times (fs5M warm: 15.7 MB per sample). A shared cache fixes this (4.5 MB fs5M, 0.32 MB fs256K, 0.09 MB exact).

## 3. 8 reader processes
| config | sps | fetch/s | resolves/s | MB/s | what binds |
|---|---|---|---|---|---|
| SEP seq fs5M warm | 25.0 | 69 | 69 | 347 | bandwidth, about 345 MB/s (same for all fs5M rows) |
| STACK fs5M warm | 68.7 | 67 | 67 | 349 | bandwidth |
| MULTI par fs5M shared warm | 104.5 | 66 | 66 | 344 | bandwidth |
| SEP par rc256K warm | 89.8 | 268 | 0 | 88 | neither wall; round trips per reader |
| SEP par exact warm | 92.1 | 272 | 0 | 19 | neither wall; round trips per reader |
| STACK exact warm | 117.3 | 116 | 0 | 9.5 | neither wall; 1 request in flight per reader |
| MULTI par exact shared warm | 123.3 | 122 | 0 | 11.4 | neither wall |
| SEP par exact cold | 36.8 | 221 | 0 | 53 | opens (2 fetches per file) |
| STACK exact cold | 50.6 | 101 | 0 | 25 | opens |
| MULTI par exact shared cold | 42.7 | 85 | 0 | 47 | opens (bigger moov) |
With 5 MiB fetches every layout stops at about 67 fetches/s and 345 MB/s, so bandwidth binds. With small fetches, 272 fetches/s were reached without any 429, so about 67 fetches/s is not a request ceiling. Small-fetch configs reached neither wall: each reader waits for its own round trips. No 429 in the whole job; the rate-limit header was still absent.

## 4. Time per fetch vs size (fs readers, warm, one reader, seq; derived from sample time / fetches, so decode is inside the fixed part)
- SEP, repeats 1-2: fixed 83 ms + size / 138 MB/s, R^2 0.99 (all 3 repeats: 93 ms + size / 114 MB/s, R^2 0.47).
- STACK, repeats 1-2: fixed 147 ms + size / 93 MB/s, R^2 0.91 (all: 154 ms, 68 MB/s, R^2 0.65).
- The resolve step costs about 14 ms per fetch (SEP fs256K 86 ms vs rc256K 72 ms).
- Repeat 0 was about 13% slower than repeats 1-2 (median over arms). It ran mostly before the other session's job, so a cold CDN is the likely cause (not verified).
- STACK fetches are about 60 ms slower than SEP fetches of the same size. Local random-access decode is only about 1.5 ms per sample for all three layouts (laptop, ckpt_1k), so decode does not explain it. Unexplained; maybe per-file CDN or xet latency.

## 5. Smoke vs main
The conclusions held. Streaming STACK/SEP: smoke 2.0x, main 1.9x. Parallel cameras were slower in streaming, now on equal shard counts. MULTI exact warm: smoke 13.5, main 12.9. STACK exact 8 readers: smoke 104, main 117 samples/s. What changed: the smoke's parallel-vs-sequential streaming comparison was not clean (S 1 vs 8); main fixes that. MULTI fs5M shared, broken in the smoke, now works: 8.3 warm, 3.7 cold.

## 6. Checks, overlap, not verified
- Checks: 112/112 remote samples bit-exact vs local decode; 0 pts violations; 0 errors, 0 skipped arms, 0 429.
- Overlap with the other session (same token): `lerobot-streaming-par-smoke2` (16:10 to 16:19 UTC) overlapped 19 of 32 streaming runs. `-smoke3` (16:33 to 16:48) overlapped 120 of 168 single-stream access runs. `-main` (from 16:52) overlapped all 10 of our 8-reader access runs. Repeat 0 (mostly outside the overlap) was the slowest repeat, so no slowdown from overlap is visible. But the 8-reader numbers shared bandwidth with -main.
- Not verified: the 64-shard case is reproduced with cache 12, not measured. Copied video files share the CDN cache. The exact reader uses a moov index known in advance (like Lance). The shared cache keeps 16 blocks (80 MB at 5 MiB), more memory than one handle has. The STACK per-fetch gap is unexplained. The fetch fit uses sample time, not a dedicated fetch benchmark.
