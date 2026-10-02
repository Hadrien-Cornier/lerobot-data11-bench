# DATA-11 round 3: PR #3917 on SEP vs STACK (equal-size files)

p1 = 1 process; p8 = 8 processes as ranks 0..7 of WORLD_SIZE 8 (#3917 allows at most 1 DataLoader worker per rank). sps = samples/s after the first sample, summed over processes (1 sample = 1 time step, 3 camera images). mean +- sd over repeats. dec CPU = thread CPU inside #3917's decode calls, per sample; proc CPU = process CPU per sample.

| layout | mode | reps | sps | first sample s | range GETs/episode | MB/episode | HTTP req/episode | mean GET s | dec CPU ms/smp | proc CPU ms/smp | peak PSS GB (sum) | cgroup peak MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SEP | p1 | 1 | 23.1 | 41.66 | 3.00 | 27.7 | 9.1 | 5.25 | 6.48 | 252.25 | 2.89 | 9304 |
| SEP | p8 | 1 | 178.8 | 20.28 | 3.00 | 24.7 | 8.4 | 4.89 | 4.98 | 41.79 | 17.26 | 22718 |
| STACK | p1 | 1 | 60.8 | 32.15 | 1.00 | 26.7 | 6.4 | 13.38 | 3.81 | 98.32 | 2.94 | 9395 |
| STACK | p8 | 1 | 227.5 | 19.12 | 1.00 | 23.8 | 6.2 | 12.22 | 3.74 | 35.52 | 17.43 | 23918 |

## Correctness checks (return_uint8=True, decoded frame == local TorchCodec decode of the source frame)

- SEP: status ok, build 20.7 s, pass True, shape [3, 224, 224], per key {"observation.images.left_wrist": {"equal_own": 20, "equal_cam": [0, 20, 0], "n": 20}, "observation.images.right_wrist": {"equal_own": 20, "equal_cam": [0, 0, 20], "n": 20}, "observation.images.top": {"equal_own": 20, "equal_cam": [20, 0, 0], "n": 20}} 
- STACK: status ok, build 12.7 s, pass True, shape [3, 672, 224], per key {"observation.images.stack": {"equal_own": 20, "equal_cam": [0, 0, 0], "n": 20}} 
- MULTI: status ok, build 4.7 s, pass False, shape [3, 224, 224], per key {"observation.images.left_wrist": {"equal_own": 0, "equal_cam": [20, 0, 0], "n": 20}, "observation.images.right_wrist": {"equal_own": 0, "equal_cam": [20, 0, 0], "n": 20}, "observation.images.top": {"equal_own": 20, "equal_cam": [20, 0, 0], "n": 20}} 

MULTI = local dataset (not on the Hub) whose 3 video keys point at the multi-track cuts. equal_cam[i] counts samples equal to SEP camera i; a correct reader gives equal_own == n for every key.
