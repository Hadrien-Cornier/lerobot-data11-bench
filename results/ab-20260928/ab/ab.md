# #3917 per-sample fixes A/B (5bf6c034ba293a075739dfb1ac6e1cbf79428f38)

Digest: `{"n_base": 9096, "n_fix": 9096, "dup_base": 0, "dup_fix": 0, "common": 9096, "only_base": 0, "only_fix": 0, "differing": 0, "differing_fields": [], "differing_examples": [], "identical": true}`

| arm | BASE sps | FIX sps | paired FIX/BASE | BASE cpu ms | FIX cpu ms |
|---|---|---|---|---|---|
| D2 | 305 ± 25 | 396 ± 29 | 1.299 ± 0.054 | 5.95 | 5.01 |
| D6 | 258 ± 18 | 349 ± 35 | 1.352 ± 0.125 | 8.57 | 6.76 |
| R6 | 375 ± 38 | 558 ± 47 | 1.494 ± 0.115 | 6.54 | 4.99 |
