# DATA-11 round 3: PR #3917 on SEP vs STACK (equal-size files)

p1 = 1 process; p8 = 8 processes as ranks 0..7 of WORLD_SIZE 8 (#3917 allows at most 1 DataLoader worker per rank). sps = samples/s after the first sample, summed over processes; sps 2nd half = rate over the second half of each run's window (after the pool fill) (1 sample = 1 time step, 3 camera images). mean +- sd over repeats. dec CPU = thread CPU inside #3917's decode calls, per sample; proc CPU = process CPU per sample.

| layout | mode | reps | sps | sps 2nd half | first sample s | range GETs/episode | MB/episode | HTTP req/episode | mean GET s | dec CPU ms/smp | proc CPU ms/smp | peak PSS GB (sum) | cgroup peak MB | anon peak MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SEP | p1 | 3 | 117.2 +- 48.0 | 137.5 +- 43.6 | 24.29 +- 8.02 | 3.00 +- 0.00 | 25.0 +- 3.0 | 9.1 +- 0.1 | 3.12 +- 0.77 | 3.93 +- 0.44 | 76.23 +- 41.90 | 2.84 +- 0.22 | 9247 +- 234 | 2666 +- 226 |
| SEP | p8 | 3 | 650.1 +- 179.1 | 1319.7 +- 130.6 | 15.96 +- 2.64 | 3.00 +- 0.00 | 24.7 +- 0.0 | 8.4 +- 0.1 | 4.52 +- 0.55 | 3.95 +- 0.22 | 13.25 +- 3.44 | 16.25 +- 1.71 | 23310 +- 1135 | 16494 +- 1180 |
| STACK | p1 | 3 | 198.8 +- 27.4 | 212.4 +- 20.6 | 18.26 +- 1.14 | 1.00 +- 0.00 | 24.2 +- 2.8 | 6.5 +- 0.1 | 7.55 +- 0.85 | 3.10 +- 0.14 | 38.15 +- 3.19 | 2.83 +- 0.19 | 9287 +- 199 | 2710 +- 198 |
| STACK | p8 | 3 | 875.3 +- 161.4 | 1899.6 +- 283.5 | 18.60 +- 0.54 | 1.00 +- 0.00 | 23.8 +- 0.0 | 6.4 +- 0.1 | 12.13 +- 1.14 | 3.23 +- 0.05 | 10.05 +- 2.03 | 17.02 +- 0.22 | 23584 +- 268 | 16776 +- 258 |

## Correctness checks (return_uint8=True, decoded frame == local TorchCodec decode of the source frame)

- SEP: status ok, build 22.9 s, pass True, shape [3, 224, 224], per key {"observation.images.left_wrist": {"equal_own": 40, "equal_cam": [0, 40, 0], "n": 40}, "observation.images.right_wrist": {"equal_own": 40, "equal_cam": [0, 0, 40], "n": 40}, "observation.images.top": {"equal_own": 40, "equal_cam": [40, 0, 0], "n": 40}} 
- STACK: status ok, build 11.5 s, pass True, shape [3, 672, 224], per key {"observation.images.stack": {"equal_own": 40, "equal_cam": [0, 0, 0], "n": 40}} 
- MULTI: status ok, build 4.3 s, pass False, shape [3, 224, 224], per key {"observation.images.left_wrist": {"equal_own": 0, "equal_cam": [40, 0, 0], "n": 40}, "observation.images.right_wrist": {"equal_own": 0, "equal_cam": [40, 0, 0], "n": 40}, "observation.images.top": {"equal_own": 40, "equal_cam": [40, 0, 0], "n": 40}} 

MULTI = local dataset (not on the Hub) whose 3 video keys point at the multi-track cuts. equal_cam[i] counts samples equal to SEP camera i; a correct reader gives equal_own == n for every key.
