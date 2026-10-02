# DATA-11 round 2 (20260925T155500Z-smoke)

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
| SEP_S8_c100_seq | 8 | 100 | seq | 126.6 | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 0.000 | 0 | 7.8 | - |
| SEP_S8_c12_seq | 8 | 12 | seq | 4.6 | - | 1.6522 | 8.7704 | 0.0000 | 1.6522 | 0.449 | 1.652 | 128 | 74.8 | - |
| STACK_S8_c100 | 8 | 100 | seq | 256.4 | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 0.000 | 0 | 3.9 | - |
| SEP_S1_c100_par | 1 | 100 | par | 72.9 | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 0.000 | 0 | 10.8 | - |

## 8 concurrent processes (own dataset + seed each)

| arm | procs | fps total | resolves/s | CDN req/s | MB/s | decoder hit | 429 |
|---|---|---|---|---|---|---|---|
| SEP_S8_c100_seq | 8 | 1040.9 | 0.0 | 0.0 | 0.0 | 1.000 | 0 |


---

# DATA-11 round 2: random access over hf:// (family ckpt_full, 3 cameras per sample)

Repo `hadriencornier/lerobot-data11-bench@a3d31680db`. sps = samples/s, one reader. fetch = one HTTP request that moves video bytes (CDN). fs* readers also pay 1 resolve per fetch; rc / exact resolve once per file (untimed, shown in `presigned`: 15 resolves for 15 files).

| regime | layout | cams | reader | mode | sps | +-CI | p50 ms | p95 ms | fetch/smp | resolve/smp | API/smp | MB/smp | dec MB/smp | open ms/smp | fallback/smp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold | MULTI | par | exact | shared | 2.78 | - | 367 | 476 | 2.00 | 0.00 | 0.00 | 1.101 | 2.569 | 163 | 0.00 |
| cold | MULTI | seq | fs256K | naive | 0.76 | - | 1382 | 1460 | 12.00 | 12.00 | 0.00 | 3.757 | 2.569 | 858 | 0.00 |
| cold | SEP | par | rc256K | - | 3.21 | - | 336 | 367 | 6.00 | 0.00 | 0.00 | 1.966 | 0.843 | 313 | 0.00 |
| cold | SEP | seq | fs5M | - | 0.59 | - | 1646 | 2074 | 6.00 | 6.00 | 0.00 | 31.855 | 0.843 | 819 | 0.00 |
| cold | STACK | - | exact | - | 2.15 | - | 490 | 590 | 2.00 | 0.00 | 0.00 | 0.497 | 0.282 | 231 | 0.00 |
| warm | MULTI | par | exact | shared | 13.54 | - | 61 | 115 | 1.00 | 0.00 | 0.00 | 0.093 | 0.197 | 0 | 0.00 |
| warm | MULTI | seq | fs256K | naive | 2.79 | - | 323 | 525 | 3.00 | 3.00 | 0.00 | 0.985 | 0.197 | 0 | 0.00 |
| warm | SEP | par | rc256K | - | 7.28 | - | 123 | 227 | 3.00 | 0.00 | 0.00 | 0.983 | 0.197 | 0 | 0.00 |
| warm | SEP | seq | fs5M | - | 2.01 | - | 514 | 1047 | 2.00 | 2.00 | 0.00 | 10.618 | 0.197 | 0 | 0.00 |
| warm | STACK | - | exact | - | 5.42 | - | 118 | 384 | 1.00 | 0.00 | 0.00 | 0.081 | 0.066 | 0 | 0.00 |

## 8 concurrent reader processes

| config | sps total | fetch/s | resolves/s | MB/s | p50 ms | p95 ms | 429 | other err |
|---|---|---|---|---|---|---|---|---|
| STACK|-|exact|-|warm | 104.3 | 103.9 | 0.0 | 8.6 | 65 | 133 | 0 | 0 |
| MULTI|par|exact|shared|cold | 31.5 | 62.9 | 0.0 | 34.6 | 223 | 434 | 0 | 0 |

Checks: remote == local decode (bit-exact) 20/20; pts violations > 1e-4 s: 0; skipped arms 0; 429 total 0.


