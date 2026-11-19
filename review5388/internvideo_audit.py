"""Export saved sample order and verify the MolmoAct2 loader outputs."""

import argparse
import gzip
import json
from collections import Counter
from hashlib import file_digest, sha256
from pathlib import Path
from tempfile import TemporaryDirectory

BATCH_SIZE = 8
REFERENCE_BATCHES = 1000
SCHEMA_VERSION = 1
MOLMO_MANIFEST = "aebacc429d54f284e103ff10613b925f5827c347e0678846b0b6f0015dded782"
EPISODE_PARQUET = "meta/episodes/chunk-000/file-000.parquet"


def canonical_json(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("ascii")


def anchor_digest(batches):
    digest = sha256()
    for batch in batches:
        digest.update(canonical_json(batch) + b"\n")
    return digest.hexdigest()


def validate_anchors(batches):
    if not isinstance(batches, list) or len(batches) != REFERENCE_BATCHES:
        raise ValueError(f"Expected {REFERENCE_BATCHES} reference batches")
    seen = set()
    for number, ids in enumerate(batches, 1):
        if len(ids) != BATCH_SIZE:
            raise ValueError(
                f"Reference batch {number} does not contain {BATCH_SIZE} samples"
            )
        for pair in ids:
            if len(pair) != 2 or any(
                type(index) is not int or index < 0 for index in pair
            ):
                raise ValueError(
                    f"Invalid sample ID at reference batch {number}: {pair!r}"
                )
            key = tuple(pair)
            if key in seen:
                raise ValueError(
                    f"Repeated sample ID at reference batch {number}: {pair!r}"
                )
            seen.add(key)


def export_reference(trace_path, output_path):
    trace_path, output_path = Path(trace_path), Path(output_path)
    opener = gzip.open if trace_path.suffix == ".gz" else open
    batches = []
    with opener(trace_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["event"] != "batch":
                continue
            if (
                record["batch"] != len(batches) + 1
                or record["contract_errors"]
                or (record["unique"], record["usable_unique"])
                != (BATCH_SIZE, BATCH_SIZE)
            ):
                raise ValueError(f"Invalid delivery event at batch {record['batch']}")
            batches.append(record["ids"])
            if len(batches) == REFERENCE_BATCHES:
                break
    validate_anchors(batches)
    with trace_path.open("rb") as stream:
        trace_sha256 = file_digest(stream, "sha256").hexdigest()
    reference = {
        "schema_version": SCHEMA_VERSION,
        "trace_file": trace_path.name,
        "trace_sha256": trace_sha256,
        "ordered_anchor_sha256": anchor_digest(batches),
        "batches": batches,
    }
    output_path.write_bytes(gzip.compress(canonical_json(reference) + b"\n", mtime=0))
    return reference


def load_reference(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        reference = json.load(stream)
    if reference["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unknown reference schema in {path}")
    batches = reference["batches"]
    validate_anchors(batches)
    if anchor_digest(batches) != reference["ordered_anchor_sha256"]:
        raise ValueError(f"Reference sample digest mismatch in {path}")
    return reference


def packed_video_opportunity(reference):
    """Count same-path reads inside measured batches from the bucket episode metadata."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    from catalog import snapshot

    before = snapshot("bucket")
    if before["manifest_sha256"] != MOLMO_MANIFEST:
        raise ValueError("MolmoAct2 bucket inventory differs from the saved trace")
    cameras = sorted(before["cameras"])
    columns = ["episode_index", "length"] + [
        f"videos/{camera}/{part}_index" for camera in cameras for part in ("chunk", "file")
    ]
    with TemporaryDirectory(prefix="internvideo-molmo-meta-") as temp:
        path = Path(temp) / "episodes.parquet"
        HfApi().download_bucket_files(before["source"], [(EPISODE_PARQUET, path)], raise_on_missing_files=True)
        metadata_bytes = path.stat().st_size
        with path.open("rb") as stream:
            metadata_sha256 = file_digest(stream, "sha256").hexdigest()
        rows = pq.read_table(path, columns=columns).to_pylist()
    if snapshot("bucket")["manifest_sha256"] != MOLMO_MANIFEST:
        raise ValueError("MolmoAct2 bucket inventory changed during the metadata check")
    episodes = {row["episode_index"]: row for row in rows}
    if len(episodes) != len(rows) or len(episodes) != before["total_episodes"]:
        raise ValueError("MolmoAct2 episode metadata count differs from the bucket inventory")

    groups_by_size, groups_per_batch = Counter(), Counter()
    distinct_paths = set()
    first_batch = None
    for number, ids in enumerate(reference["batches"], 1):
        paths = Counter()
        for episode, frame in ids:
            row = episodes[episode]
            if not 0 <= frame < row["length"]:
                raise ValueError(f"Out-of-range frame in batch {number}, episode {episode}: {frame}")
            for camera in cameras:
                path_key = camera, row[f"videos/{camera}/chunk_index"], row[f"videos/{camera}/file_index"]
                paths[path_key] += 1
        groups_by_size.update(paths.values())
        groups_per_batch[len(paths)] += 1
        distinct_paths.update(paths)
        if number == 1:
            first_batch = {
                "frame_requests": sum(paths.values()),
                "distinct_mp4_paths": len(paths),
                "path_groups_by_request_count": dict(sorted(Counter(paths.values()).items())),
            }
    requests = sum(size * count for size, count in groups_by_size.items())
    groups = sum(groups_by_size.values())
    return {
        "source": before["source"],
        "manifest_sha256": MOLMO_MANIFEST,
        "ordered_anchor_sha256": reference["ordered_anchor_sha256"],
        "episode_metadata": {"path": EPISODE_PARQUET, "bytes": metadata_bytes, "sha256": metadata_sha256},
        "batch_1": first_batch,
        "batches_checked": len(reference["batches"]),
        "frame_requests": requests,
        "path_groups_summed_over_batches": groups,
        "potential_backend_calls_saved": requests - groups,
        "distinct_mp4_paths_across_batches": len(distinct_paths),
        "path_groups_by_request_count": dict(sorted(groups_by_size.items())),
        "batches_by_distinct_mp4_count": dict(sorted(groups_per_batch.items())),
    }


class BatchAudit:
    """Check sample identities and hash each field over all delivered batches."""

    def __init__(self, reference, cameras):
        from xxhash import xxh3_128

        self.reference = reference
        self.checked = 0
        self.anchor_hash = sha256()
        self.fields = {
            **{
                camera: ("|u1", (BATCH_SIZE, 3, 360, 640)) for camera in sorted(cameras)
            },
            "action": ("<f4", (BATCH_SIZE, 14)),
            "observation.state": ("<f4", (BATCH_SIZE, 14)),
        }
        self.payload_hashes = {
            field: xxh3_128(canonical_json([field, dtype, shape]) + b"\n")
            for field, (dtype, shape) in self.fields.items()
        }

    def check(self, batch_number, batch):
        if batch_number != self.checked + 1 or batch_number > len(
            self.reference["batches"]
        ):
            raise ValueError(f"Unexpected delivered batch number: {batch_number}")
        ids = [
            list(pair)
            for pair in zip(
                batch["episode_index"].tolist(),
                batch["frame_index"].tolist(),
                strict=True,
            )
        ]
        expected = self.reference["batches"][batch_number - 1]
        if ids != expected:
            raise ValueError(
                f"Batch {batch_number} sample order differs: expected {expected!r}, got {ids!r}"
            )

        arrays = {}
        for field, (dtype, shape) in self.fields.items():
            array = batch[field].detach().numpy()
            if array.shape != shape or array.dtype.str[1:] != dtype[1:]:
                raise ValueError(
                    f"Batch {batch_number} {field} has {array.dtype} {array.shape}, expected {dtype} {shape}"
                )
            if array.dtype.str != dtype:
                array = array.astype(dtype, copy=False)
            if not array.flags.c_contiguous:
                array = array.copy(order="C")
            arrays[field] = array

        self.anchor_hash.update(canonical_json(ids) + b"\n")
        for field, array in arrays.items():
            self.payload_hashes[field].update(memoryview(array).cast("B"))
        self.checked = batch_number

    def summary(self):
        fields = {
            field: digest.hexdigest() for field, digest in self.payload_hashes.items()
        }
        anchors = self.anchor_hash.hexdigest()
        return {
            "batches_checked": self.checked,
            "reference_batches": len(self.reference["batches"]),
            "reference_trace_sha256": self.reference["trace_sha256"],
            "reference_ordered_anchor_sha256": self.reference["ordered_anchor_sha256"],
            "ordered_anchor_sha256": anchors,
            "payload_xxh3_128_by_field": fields,
            "ordered_output_sha256": sha256(
                canonical_json([self.checked, anchors, fields])
            ).hexdigest(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    export = actions.add_parser(
        "export", help="Save the first 1,000 valid batch identities from a trace"
    )
    export.add_argument("trace", type=Path)
    export.add_argument("output", type=Path)
    opportunity = actions.add_parser(
        "opportunity", help="Count same-file requests using the public episode metadata"
    )
    opportunity.add_argument("reference", type=Path)
    opportunity.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "export":
        reference = export_reference(args.trace, args.output)
        print(reference["trace_file"], reference["ordered_anchor_sha256"], len(reference["batches"]))
    else:
        report = packed_video_opportunity(load_reference(args.reference))
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
