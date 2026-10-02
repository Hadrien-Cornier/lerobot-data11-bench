# File size vs range-request latency inside HF (size2)

Signed CDN link resolved once per file; 60 random 256 KiB range GETs per file, interleaved, one at a time. TTFB = time from sending the request to the first body byte. ms.

| file | MB | n | TTFB p50 | IQR (p25-p75) | headers p50 | total p50 | total p90 | x-cache |
|---|---|---|---|---|---|---|---|---|
| SEP (`ckpt_full_g000__top.mp4`) | 80 | 60 | 62 | 57-92 | 32 | 68 | 118 | {'': 60} |
| BIG (`ckpt_full_b000__top.mp4`) | 253 | 60 | 69 | 57-104 | 32 | 77 | 175 | {'': 60} |
| STACK_eq (`ckpt_full_g001_p0.mp4`) | 87 | 60 | 66 | 59-80 | 32 | 73 | 105 | {'': 60} |
| STACK_r2 (`ckpt_full_g001.mp4`) | 261 | 60 | 72 | 60-91 | 32 | 79 | 137 | {'': 60} |
| MULTI_eq (`ckpt_full_g001_p0.mp4`) | 88 | 60 | 80 | 64-130 | 31 | 83 | 163 | {'': 60} |

Hosts: ['cas-bridge-direct.xethub.hf.co']. Wall 29 s, start 2026-09-25T19:05:09Z.
