# File size vs range-request latency inside HF (size)

Signed CDN link resolved once per file; 60 random 256 KiB range GETs per file, interleaved, one at a time. TTFB = time from sending the request to the first body byte. ms.

| file | MB | n | TTFB p50 | IQR (p25-p75) | headers p50 | total p50 | total p90 | x-cache |
|---|---|---|---|---|---|---|---|---|
| SEP (`ckpt_full_g000__top.mp4`) | 80 | 60 | 66 | 59-115 | 30 | 72 | 139 | {'': 60} |
| BIG (`ckpt_full_b000__top.mp4`) | 253 | 60 | 94 | 62-146 | 32 | 100 | 203 | {'': 60} |
| STACK_eq (`ckpt_full_g001_p0.mp4`) | 87 | 60 | 81 | 61-106 | 32 | 89 | 131 | {'': 60} |
| STACK_r2 (`ckpt_full_g001.mp4`) | 261 | 60 | 101 | 69-160 | 31 | 110 | 203 | {'': 60} |
| MULTI_eq (`ckpt_full_g001_p0.mp4`) | 88 | 60 | 107 | 59-154 | 31 | 117 | 194 | {'': 60} |

Hosts: ['cas-bridge-direct.xethub.hf.co']. Wall 35 s, start 2026-09-25T18:33:07Z.
