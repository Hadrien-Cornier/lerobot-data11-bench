# #3917 ab2 job 0 (6d94598523738d19700e345e9b42fb3d2db352f5)

## Probe

| arm | files/s | MB per file | reads per file | file s p50 | runs | non-faststart |
|---|---|---|---|---|---|---|
| BASE@default | 9.8 | 4.194 | 1.00 | 0.909 | 1 | 0 |
| PROBE@64K | 11.9 | 0.203 | 2.00 | 0.597 | 1 | 0 |
| PROBE@256K | 12.1 | 0.293 | 1.23 | 0.530 | 1 | 0 |
| PROBE@default | 12.0 | 0.524 | 1.00 | 0.579 | 1 | 0 |

Equality: `{"BASE@default": {"status": "ok", "files": 20, "same_digest_as_base": 20, "errors": 0, "non_faststart": 0}, "PROBE@64K": {"status": "ok", "files": 20, "same_digest_as_base": 20, "errors": 0, "non_faststart": 0}, "PROBE@256K": {"status": "ok", "files": 20, "same_digest_as_base": 20, "errors": 0, "non_faststart": 0}, "PROBE@default": {"status": "ok", "files": 20, "same_digest_as_base": 20, "errors": 0, "non_faststart": 0}}`

## Transforms

Digests: `{"fixed_BASE_vs_TFX": {"a": 4733, "b": 4733, "common": 4733, "identical_samples": 4733, "all_identical": true}, "random_BASE_twice": {"a": 4733, "b": 4733, "common": 4733, "identical_samples": 4733, "all_identical": true}, "random_TFX_twice": {"a": 4733, "b": 4733, "common": 4733, "identical_samples": 0, "all_identical": false}}`

| arm | BASE sps | TFX sps | paired TFX/BASE |
|---|---|---|---|
| D6T | [83] | [113] | [1.36] |
