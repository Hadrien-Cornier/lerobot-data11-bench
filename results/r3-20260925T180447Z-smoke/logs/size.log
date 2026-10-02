# File size vs range-request latency inside HF (size)

Signed CDN link resolved once per file; 10 random 256 KiB range GETs per file, interleaved, one at a time. TTFB = time from sending the request to the first body byte. ms.

| file | MB | n | TTFB p50 | IQR (p25-p75) | headers p50 | total p50 | total p90 | x-cache |
|---|---|---|---|---|---|---|---|---|
| SEP (`ckpt_full_g000__top.mp4`) | 80 | 10 | 73 | 55-85 | 29 | 76 | 147 | {'': 10} |
| BIG (`ckpt_full_b000__top.mp4`) | 253 | 10 | 188 | 142-229 | 34 | 191 | 281 | {'': 10} |
| STACK_eq (`ckpt_full_g001_p0.mp4`) | 87 | 10 | 72 | 61-130 | 37 | 90 | 234 | {'': 10} |
| STACK_r2 (`ckpt_full_g001.mp4`) | 261 | 10 | 96 | 59-179 | 33 | 101 | 212 | {'': 10} |
| MULTI_eq (`ckpt_full_g001_p0.mp4`) | 88 | 10 | 133 | 95-209 | 33 | 136 | 363 | {'': 10} |

Hosts: ['cas-bridge-direct.xethub.hf.co']. Wall 9 s, start 2026-09-25T18:08:24Z.
