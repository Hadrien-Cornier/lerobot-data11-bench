# DATA-11 round 2: laptop run (outside HF) + Option E check

Run 2026-09-25 16:00-16:03 UTC from Hadrien's Mac (home internet, laptop possibly busy). One reader, warm regime, family ckpt_full at `hadriencornier/lerobot-data11-bench@a3d31680db`. HTTP bytes counted by the huggingface_hub/httpx spy: **369 MB total, under the 500 MB hard cap**. Setup and test traffic before this run (not in the 369 MB): about 0.5 GB (tiny test files and a tiny streaming test).

## RTT from the laptop

- ICMP ping huggingface.co: avg 49 ms (min 45); cas-bridge.xethub.hf.co: avg 47 ms
- HTTP: resolve request (huggingface.co, 1 byte, no redirect) median 74 ms; 1-byte Range GET on the signed CDN URL (us.aws.cdn.hf.co, warm connection) median 169 ms, min 147 ms
- 5 MiB range GET: 0.46-0.66 s (8-11 MB/s)

## Random access, warm, one reader (sps = samples/s, 3 cameras per sample, cameras sequential)

| layout | reader | sps per repeat | fetch/smp | resolve/smp | MB/smp |
|---|---|---|---|---|---|
| SEP | fs256K | 1.17, 1.06 | 3.00 | 3.00 | 0.987 |
| STACK | fs256K | 2.45, 3.47 | 1.00 | 1.00 | 0.329 |
| SEP | rc256K | 1.49, 1.36 | 3.00 | 0.00 | 0.983 |
| STACK | rc256K | 2.98, 4.93 | 1.00 | 0.00 | 0.328 |
| SEP | exact | 1.63, 1.65 | 2.93 | 0.00 | 0.210 |
| STACK | exact | 4.35, 6.19 | 1.00 | 0.00 | 0.083 |
| SEP | fs5M | 0.56 | 3.00 | 3.00 | 15.929 |
| STACK | fs5M | 1.32 | 1.00 | 1.00 | 5.310 |

exact on the laptop runs without the precomputed moov index (ACC_EXACT_HINT=0), so its open costs 2 requests; opens are untimed in the warm regime. fs5M: 6 samples, 1 repeat (budget). Local bit-exact checks were off here (no local copies of the 1.3 GB files); the HF job checks them.

## Streaming order, real StreamingLeRobotDataset (#4702 branch), S = 1 shard, warm

- SEP_S1_c100_seq: 502 frames/s over 400 frames after 50 warm-up frames, mp4 fetches in the window 0, CPU 2.0 ms/frame
- STACK_S1_c100: 562 frames/s over 400 frames after 50 warm-up frames, mp4 fetches in the window 0, CPU 1.8 ms/frame

With one shard every camera file is read forward in order. One 5 MiB block holds about 1,700 SEP frames or 600 STACK frames, so 400 frames needed no fetch. The network cost is the file opens (5.3 MB each).

## Option E: can TorchCodec 0.11 decode a frame without reading the moov?

Local test on the ckpt_1k SEP `top` file (moov 6.2 KB at offset 32), torchcodec 0.11.1, FFmpeg 8. Script and raw output in optionE/.

| test | result |
|---|---|
| default open | {"moov_bytes_read": 6201, "moov_size": 6201, "reads": 2, "bit_exact": true} |
| custom_frame_mappings (ffprobe JSON) | {"moov_bytes_read": 6201, "bit_exact": true} |
| moov bytes zeroed | {"decoded": false, "error": "ValueError: The best video stream is unknown and there is no specified stream. \nThis should never happen. Please report an issue following the steps in\nhttps://github.co |
| moov zeroed + custom_frame_mappings | {"decoded": false, "error": "ValueError: The best video stream is unknown and there is no specified stream. \nThis should never happen. Please report an issue following the steps in\nhttps://github.co |
| ftyp + mdat of the needed packets, no moov | {"decoded": false, "error": "RuntimeError: SingleStreamDecoder, /Users/runner/work/torchcodec/torchcodec/meta-pytorch/torchcodec/src/torchcodec/_core/SingleStreamDecoder.cpp:82, Failed to open input b |
| synthetic mini-MP4: packets cut with an external index + 17-byte av1C, re-muxed in memory | {"packet_bytes": 4664, "extradata_bytes": 17, "mini_file_bytes": 5485, "bit_exact": true, "frames_in_mini": 2} |
| raw AV1 OBU stream, no container | {"decoded": false, "error": "RuntimeError: SingleStreamDecoder, /Users/runner/work/torchcodec/torchcodec/meta-pytorch/torchcodec/src/torchcodec/_core/SingleStreamDecoder.cpp:82, Failed to open input b |

Answer: **No** for TorchCodec on its own. It reads the whole moov at open, also with custom_frame_mappings, and it fails when the moov is missing or zeroed. **Yes with a workaround**: keep the sample index (offset, size, keyframe flag, about 8 bytes per frame) and the codec config outside the MP4, cut the packet bytes, wrap them in a tiny in-memory MP4 (5.5 KB here); TorchCodec decodes it bit-exact. The re-mux cost per sample was not timed.
