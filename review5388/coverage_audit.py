"""Check cloud trace anchor IDs against the episode tables of each source snapshot."""

import argparse
import csv
import gzip
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from catalog import SOURCES, snapshot

MILESTONES = (1, 125, 1000)
TRACE_SUFFIX = ".events.jsonl"
ROOT = Path(__file__).resolve().parent
CATALOG_FIELDS = ("source", "revision", "manifest_sha256", "total_episodes", "total_frames")
EPISODE_COLUMNS = ("episode_index", "length", "dataset_from_index", "dataset_to_index")
RECEIPT_FIELDS = ("repo", "revision", "path", "relative_cache_path", "sha256", "size_bytes")


def read_json(path):
    return json.loads(path.read_text())


def catalog_key(record):
    return tuple(record[key] for key in CATALOG_FIELDS)


def episode_lengths(dataset, catalog, cache):
    """Cache a compact episode index for one pinned source or verified bucket inventory."""
    repo, revision = SOURCES[dataset]
    if (repo, revision) != (catalog["source"], catalog["revision"]):
        raise ValueError(f"{dataset}: catalog source does not match the configured snapshot")

    root = f"datasets/{repo}@{revision}" if revision else f"buckets/{repo}"
    expected_episodes, expected_frames = catalog["total_episodes"], catalog["total_frames"]
    cache_file = cache / f"{dataset}-{catalog['manifest_sha256']}.json"
    if cache_file.is_file():
        index = read_json(cache_file)
        if tuple(index["catalog_key"]) != catalog_key(catalog):
            raise ValueError(f"{dataset}: cached metadata has a different catalog")
        lengths = {int(episode): length for episode, length in index["lengths"].items()}
    else:
        if revision is None and catalog_key(snapshot(dataset)) != catalog_key(catalog):
            raise ValueError(f"{dataset}: bucket inventory changed before metadata read")
        fs = HfFileSystem()
        lengths = {}
        paths = sorted(fs.glob(f"{root}/meta/episodes/*/*.parquet"))
        if not paths:
            raise ValueError(f"{dataset}: no episode metadata files")
        for path in paths:
            with fs.open(path, "rb", block_size=131072, cache_type="readahead") as stream:
                table = pq.ParquetFile(stream).read(columns=EPISODE_COLUMNS)
            episode_ids, sizes, start, stop = [column.to_pylist() for column in table.columns]
            for episode, size, first, last in zip(episode_ids, sizes, start, stop, strict=True):
                if episode in lengths or size != last - first:
                    raise ValueError(f"{dataset}: inconsistent episode metadata for episode {episode}")
                lengths[episode] = size
        if revision is None and catalog_key(snapshot(dataset)) != catalog_key(catalog):
            raise ValueError(f"{dataset}: bucket inventory changed during metadata read")

    if (
        len(lengths) != expected_episodes
        or sum(lengths.values()) != expected_frames
        or any(key < 0 or value <= 0 for key, value in lengths.items())
    ):
        raise ValueError(f"{dataset}: summed episode metadata does not match the catalog")

    if not cache_file.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.with_suffix(".tmp")
        temporary.write_text(json.dumps({"catalog_key": catalog_key(catalog), "lengths": lengths}))
        temporary.replace(cache_file)
    return lengths


def worker_receipt_check(summary_path, record):
    sidecar = record.get("sidecar", {})
    receipt = sidecar.get("private_transfer")
    path = summary_path.parent / f"{record['dataset']}-private-transfer.json"
    if not receipt or not path.is_file():
        return "worker-receipt-missing"
    transferred = read_json(path)
    expected = transferred.get("private_transfer", {})
    if (
        transferred.get("dataset") != record["dataset"]
        or any(not expected.get(key) or expected[key] != receipt.get(key) for key in RECEIPT_FIELDS)
        or not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"])
        or not re.fullmatch(r"[0-9a-f]{40}", expected["revision"])
        or sidecar.get("sha256") != expected["sha256"]
        or sidecar.get("archive_bytes") != expected["size_bytes"]
        or sidecar.get("archive_name") != Path(expected["relative_cache_path"]).name
        or sidecar.get("origin") != "verified_prepared_archive"
    ):
        return "worker-receipt-mismatch"
    return "verified"


def source_check(summary_path, record, catalog, allow_partial, *, worker_memory=False):
    source = record.get("source", {})
    if source.get("repo") != catalog["source"] or source.get("revision") != catalog["revision"]:
        return "source-mismatch" if source else "source-not-recorded"
    observed = source if worker_memory and "frames" in source else source.get("observed")
    if observed and (observed.get("frames"), observed.get("episodes")) != (
        catalog["total_frames"],
        catalog["total_episodes"],
    ):
        return "catalog-count-mismatch"
    if worker_memory and not observed:
        return "snapshot-not-opened"
    if catalog["revision"]:
        if not observed:
            return "snapshot-not-opened"
        if worker_memory:
            return worker_receipt_check(summary_path, record)
        return (
            "verified"
            if observed["metadata_snapshot"] == catalog["revision"]
            and observed["reader_revision"] == catalog["revision"]
            else "revision-mismatch"
        )

    name = summary_path.stem
    reader = record["reader"]
    log = summary_path.parent / "run.jsonl"
    events = read_trace(log, allow_partial) if log.is_file() else ()
    for event in events:
        if worker_memory and event["event"] == "worker-memory-pair-done" and event["dataset"] == "bucket":
            label = "worker-memory"
            break
        if event["event"] == "comparison-source" and event["label"] in (name, name.removeprefix("bucket-")):
            label = name.removeprefix("bucket-")
            break
        if (
            event["event"] == "pair-done"
            and event["dataset"] == "bucket"
            and name == f"bucket-{event['pair']}-{reader}"
        ):
            label = event["pair"]
            break
    else:
        return "bucket-check-pending"

    if event.get("source_valid") is not True:
        return "bucket-check-failed"
    paths = [summary_path.parent / f"bucket-manifest-{label}-{phase}.json" for phase in ("before", "after")]
    if not all(path.is_file() for path in paths):
        return "bucket-manifest-missing"
    if not all(catalog_key(read_json(path)) == catalog_key(catalog) for path in paths):
        return "bucket-manifest-mismatch"
    return worker_receipt_check(summary_path, record) if worker_memory else "verified"


def trace_files(directory):
    """Prefer the completed compressed trace when a prior sync also left a plain copy."""
    paths = {}
    for path in sorted(directory.rglob(f"*{TRACE_SUFFIX}*")):
        name, suffix, compression = path.name.partition(TRACE_SUFFIX)
        if suffix and compression in ("", ".gz"):
            key = path.with_name(f"{name}.json")
            if key not in paths or compression:
                paths[key] = path
    return sorted(paths.items())


def read_trace(path, allow_partial):
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    if allow_partial and not line.endswith("\n"):
                        return
                    raise
    except EOFError:
        if not allow_partial:
            raise


def audit(trace, lengths, total_frames, allow_partial, *, worker_memory=False):
    seen, admitted, delivered = set(), set(), set()
    raw = unique_valid = out_of_range = batches = 0
    # The worker-memory harness logs only measured batches and omits the phase and admission events.
    measured = worker_memory
    terminal = None
    rows = []

    def row(scope):
        return {
            "prefix": scope,
            "batches": batches,
            "raw_anchors": raw,
            "unique_anchors": len(seen),
            "unique_in_bounds": unique_valid,
            "duplicates": raw - len(seen),
            "out_of_range": out_of_range,
            "admitted_episodes": None if worker_memory else len(admitted),
            "delivered_episodes": len(delivered),
            "source_frames": total_frames,
            "source_episodes": len(lengths),
            "source_coverage_fraction": f"{unique_valid / total_frames:.9f}",
        }

    for event in read_trace(trace, allow_partial):
        kind = event["event"]
        if kind == "phase":
            measured = event.get("phase") == "measurement"
        elif kind == "terminal":
            terminal = event
        elif measured and kind == "pool_admission":
            admitted.update(span[0] for span in event["spans"])
        elif measured and kind == "batch":
            batches += 1
            if event["batch"] != batches:
                raise ValueError(f"{trace}: missing or out-of-order batch {batches}")
            for episode, frame in event["ids"]:
                if type(episode) is not int or type(frame) is not int:
                    raise ValueError(f"{trace}: non-integer anchor in batch {batches}")
                anchor = (episode, frame)
                raw += 1
                if not 0 <= frame < lengths.get(episode, 0):
                    out_of_range += 1
                elif anchor not in seen:
                    unique_valid += 1
                    delivered.add(episode)
                seen.add(anchor)
            if batches in MILESTONES:
                rows.append(row("batch_" + str(batches)))

    if terminal is None and not allow_partial:
        raise ValueError(f"{trace}: no terminal event (use --allow-partial for in-progress traces)")
    if terminal is not None:
        counts = (batches, raw, len(seen), raw - len(seen))
        keys = ("batches", "raw_samples", "unique_samples", "duplicates")
        if counts != tuple(terminal[key] for key in keys):
            raise ValueError(f"{trace}: independently counted IDs do not match the terminal event")
    rows.append(row("end"))
    return rows, terminal


def trace_format(record):
    """Distinguish native benchmark and worker-memory summaries from unrelated traces."""
    required = {"dataset", "reader", "workload", "config", "progress"}
    if not isinstance(record, dict) or not required <= record.keys():
        return None
    if {"timings", "contract", "turnover"} <= record.keys():
        return "native"
    config = record["config"]
    if (
        isinstance(config, dict)
        and "milestones" in record
        and {"workers", "sample_interval_s", "maps_interval_s"} <= config.keys()
    ):
        return "worker_memory"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Job folder, or parent with one or more job folders")
    catalogs_dir = ROOT / "catalog" if (ROOT / "catalog").is_dir() else ROOT / "results"
    parser.add_argument("--catalog-dir", type=Path, default=catalogs_dir)
    parser.add_argument("--metadata-cache", type=Path, default=Path("/tmp/lerobot5388-coverage-metadata"))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    catalogs = {}
    lengths = {}
    results = []
    for summary_path, trace in trace_files(args.directory):
        if not summary_path.is_file():
            print(f"Skipping {trace.name}: no matching summary", file=sys.stderr)
            continue
        record = read_json(summary_path)
        format_name = trace_format(record)
        if format_name is None or record["dataset"] not in SOURCES:
            continue
        worker_memory = format_name == "worker_memory"
        dataset = record["dataset"]
        if dataset not in catalogs:
            catalog = read_json(args.catalog_dir / f"catalog-{dataset}.json")
            lengths[dataset] = episode_lengths(dataset, catalog, args.metadata_cache)
            catalogs[dataset] = catalog
        catalog = catalogs[dataset]
        checked = source_check(summary_path, record, catalog, args.allow_partial, worker_memory=worker_memory)
        rows, terminal = audit(
            trace, lengths[dataset], catalog["total_frames"], args.allow_partial, worker_memory=worker_memory
        )
        context = {
            "run": trace.parent.name,
            "cell": summary_path.stem,
            "dataset": dataset,
            "reader": record["reader"],
            "workload": record["workload"],
            "tolerance_ms": 1000 * record["config"].get("tolerance_s", 0.0001),
            "source_check": checked,
            "catalog_manifest_sha256": catalog["manifest_sha256"],
            "trace_status": terminal["status"] if terminal else "partial",
        }
        results.extend(context | row for row in rows)

    if not results:
        parser.error("No auditable benchmark traces found")
    stream = args.out.open("w", newline="") if args.out else sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    finally:
        if args.out:
            stream.close()


if __name__ == "__main__":
    main()
