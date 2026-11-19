# How many Hub datasets publish a reader-ready MP4 index?

A published index is a file that maps a video frame to its location in an MP4. Here “reader-ready” means the merged loader from #3917 can find it on the Hub when a user's local cache is empty. The count covers the snapshot of publicly discoverable repositories listed below. [Reader's lookup code](https://github.com/huggingface/lerobot/blob/fe1c54594f6ba275fccf210b9f7b11f0e2324b89/src/lerobot/datasets/streaming_sidecar.py#L179-L214)

**November 19, 2026: 0 of the 7,147 public repositories returned by the Hub's LeRobot tag filter had the publication branch.** We also checked every repository under the official organization and the owner of the MolmoAct2 example bucket, including their untagged repositories. All 7,576 distinct repositories returned a missing-branch result. No branch checks were inconclusive. The lookups ran from 17:45 to 17:54 UTC.

| Public Hub scope | Dataset repositories | With #3917 publication branch | With reader-ready index |
|---|---:|---:|---:|
| Repositories returned by the LeRobot tag filter | 7,147 | 0 | 0 |
| Official `lerobot` organization | 195 | 0 | 0 |
| Dataset repositories owned by the MolmoAct2 bucket owner | 264 | 0 | 0 |
| **Unique repositories across all three groups** | **7,576** | **0** | **0** |

The scopes overlap. Nine official repositories and 21 repositories owned by the example bucket's owner had the LeRobot tag. A Hub tag is optional. An untagged dataset outside the other two groups, a private dataset, or a saved index on a user's machine falls outside this count. Hub full-text search indexes dataset cards and cannot supply a global count of files on side branches. [Catalog filter](https://huggingface.co/datasets?other=lerobot) · [What Hub search indexes](https://huggingface.co/docs/hub/en/search)

## What the audit looked for

For a dataset repository the reader resolves a source commit and consults a dedicated branch named `lerobot-sidecars`. The file lives under `meta/mp4-sidecars/` and its name includes a key for the source. Publishing on that branch preserves the commit whose videos were indexed. For all 195 official repos and the 100 most-liked tagged repos, the audit queried the directory on both the side branch and the listed source commit. None of the 292 distinct repos had it in either place.

The extended audit checked the publication branch for every additional tagged or bucket-owner repo. An anonymous resolver request can explicitly report a missing revision. For a gated response, a missing file, or any ambiguous status, the script asks the repository tree API for the exact directory. An access error is recorded separately from absence. The catalog and branch lookups span a short interval; repository owners can add an index after the observation. [Two-location audit](measurements/hub-index-audit.json) · [Extended branch audit](measurements/hub-index-all-tagged.json)

## A bucket is a separate case

The public MolmoAct2 bucket has three archive filenames under `meta/mp4-sidecars/`, pointing to two distinct objects. Their manifests use schema 1 and lack source-file fingerprints. The pinned #3917 reader expects a matching schema-3 index and rebuilt it in the as-published benchmark. The staged schema-3 archive used in the prepared benchmark was private. None of those files adds a public reader-ready dataset repository to the table. The audit checked this one bucket. [Public bucket directory](https://huggingface.co/buckets/pepijn223/MolmoAct2-BimanualYAM-Dataset-bucket/tree/meta/mp4-sidecars) · [Object hashes and archive headers](measurements/hub-index-bucket.json)

## Reproduce the repository scan

Use Python and `requests` from the `review5388` directory. These calls read public Hub metadata anonymously and honor the server's rate limit. On a new date, the catalog and results can change.

```bash
python hub_index_audit.py official-and-top.json
python hub_index_audit.py all-tagged.json --scoped official-and-top.json
```
