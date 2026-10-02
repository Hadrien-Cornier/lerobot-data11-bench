# File size vs range-request latency inside HF (size2)

Signed CDN link resolved once per file; 10 random 256 KiB range GETs per file, interleaved, one at a time. TTFB = time from sending the request to the first body byte. ms.

| file | MB | n | TTFB p50 | IQR (p25-p75) | headers p50 | total p50 | total p90 | x-cache |
|---|---|---|---|---|---|---|---|---|
| SEP (`ckpt_full_g000__top.mp4`) | 80 | 10 | 58 | 53-110 | 30 | 62 | 140 | {'': 10} |
| BIG (`ckpt_full_b000__top.mp4`) | 253 | 10 | 103 | 64-147 | 37 | 105 | 182 | {'': 10} |
| STACK_eq (`ckpt_full_g001_p0.mp4`) | 87 | 10 | 68 | 66-71 | 32 | 72 | 85 | {'': 10} |
| STACK_r2 (`ckpt_full_g001.mp4`) | 261 | 10 | 56 | 56-77 | 31 | 63 | 88 | {'': 10} |
| MULTI_eq (`ckpt_full_g001_p0.mp4`) | 88 | 10 | 83 | 59-133 | 32 | 93 | 352 | {'': 10} |

Hosts: ['cas-bridge-direct.xethub.hf.co']. Wall 6 s, start 2026-09-25T18:20:04Z.
