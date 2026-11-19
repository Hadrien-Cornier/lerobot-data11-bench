"""Export measured resource metrics from downloaded DATA-11 cells as a long-form CSV."""

import argparse
from bisect import bisect_right
import csv
import gzip
import json
from pathlib import Path
import sys
from typing import NamedTuple, Optional

from summarize import prepared_transfer_path

MILESTONES = (1, 125, 1000)
TERMINAL = {"ok", "invalid_output", "empty", "interrupted", "error"}
PURPOSES = ("metadata", "parquet", "sidecar_download", "sidecar_probe", "video", "unclassified_http")
STAGES = {
    "baseline": {
        "baseline_parquet_read_and_assemble": "parquet",
        "baseline_video_fetch_and_synthesize": "video_fetch",
        "video_native_http_range_fetch": "video_fetch",
        "baseline_rgb_decoder_open": "decode",
        "baseline_rgb_decode_inclusive_of_cache_wait": "decode",
        "sidecar_probe_native_http_range_fetch": "index_build",
    },
    "candidate": {
        "candidate_pool_assembly": "parquet",
        "video_fsspec_range_fetch": "video_fetch",
        "candidate_rgb_decode_inclusive_of_io": "decode",
    },
}
COMMON_STAGES = {
    "metadata_fsspec_range_fetch": "metadata",
    "parquet_fsspec_range_fetch": "parquet",
    "sidecar_download_fsspec_range_fetch": "index_transfer",
}
HTTP_NOTE = (
    "Observed raw HTTP response bodies. This count can include redirects and failed reads. "
    "Native Xet traffic and network overhead can be absent."
)
FILES_NOTE = (
    "Observed distinct named source files for this purpose. Opens or HTTP responses can count. "
    "The same MP4 can occur in both sidecar_probe and video."
)
STAGE_NOTE = (
    "Per-call sums can overlap and include prefetch that supplies later batches. "
    "CPU covers the calling thread. Decode can include I/O or cache wait."
)
MEMORY_NOTE = "Sampled process-tree RSS sums can count shared pages for multiple processes."


class BatchEvent(NamedTuple):
    process_elapsed_s: float
    measurement_elapsed_s: float
    wait_s: Optional[float]
    raw_samples: int
    unique_samples: Optional[int]
    usable_unique_samples: Optional[int]


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    open_file = gzip.open if path.suffix == ".gz" else open
    if not path.is_file():
        return
    with open_file(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if line.endswith("\n"):
                    raise
                # A killed process can leave one final partial line.


def trace_data(path, start_s):
    trace = path.with_suffix(".events.jsonl")
    if not trace.exists() and trace.with_suffix(".jsonl.gz").exists():
        trace = trace.with_suffix(".jsonl.gz")
    resources, batches = [], {}
    first_sample = None
    sidecar_start = {}
    seen_elapsed_s = None
    for event in jsonl(trace):
        elapsed_s = event["elapsed_s"]
        seen_elapsed_s = elapsed_s if seen_elapsed_s is None else max(elapsed_s, seen_elapsed_s)
        if start_s is None or elapsed_s < start_s:
            continue
        kind = event["event"]
        if kind == "resources":
            resources.append(event)
        elif kind == "batch":
            batches[event["batch"]] = BatchEvent(
                elapsed_s, event.get("measurement_elapsed_s", elapsed_s - start_s), event.get("wait_s"),
                len(event.get("ids", [])), event.get("unique"), event.get("usable_unique"),
            )
        elif kind == "first_sample":
            first_sample = event.get("measurement_elapsed_s")
        elif kind == "sidecar_start":
            sidecar_start[event["phase"]] = elapsed_s
        elif kind == "sidecar_end":
            sidecar_start.pop(event["phase"], None)
    resources.sort(key=lambda event: event["elapsed_s"])
    return trace.name, resources, batches, first_sample, sidecar_start, seen_elapsed_s


def source_metadata(path, record, run_events):
    source = record.get("source", {})
    repo, revision = source.get("repo"), source.get("revision")
    if not repo:
        return "", "not_recorded", None, None
    if record["dataset"] != "bucket":
        observed = source.get("observed")
        if observed is None:
            verified = "revision_requested"
        elif observed.get("reader_revision") == observed.get("metadata_snapshot") == revision:
            verified = "revision_verified"
        else:
            verified = "revision_mismatch"
        return f"hf://datasets/{repo}@{revision}", verified, None, None

    cell = path.stem
    pair = cell.removeprefix("bucket-").removesuffix(f"-{record['reader']}")
    standalone = cell.removeprefix("bucket-")
    manifest_key = standalone if (path.parent / f"bucket-manifest-{standalone}-before.json").exists() else pair
    digests = []
    for bracket in ("before", "after"):
        manifest = path.parent / f"bucket-manifest-{manifest_key}-{bracket}.json"
        digests.append(read(manifest).get("manifest_sha256") if manifest.is_file() else None)
    before, after = digests
    verified = "bucket_manifest_unverified"
    if before and after:
        verified = "bucket_manifest_match" if before == after else "bucket_manifest_mismatch"
    for event in run_events:
        relevant = (event.get("event") == "comparison-source" and event.get("label") == cell) or (
            event.get("event") == "pair-done" and event.get("dataset") == "bucket" and event.get("pair") == pair
        )
        if relevant and event.get("source_valid") is False:
            verified = "bucket_manifest_mismatch"
    return f"hf://buckets/{repo}", verified, before, after


def quantiles(waits):
    values = sorted(waits)
    result = {}
    for percentile in (50, 95, 99):
        point = (len(values) - 1) * percentile / 100
        lower = int(point)
        upper = min(lower + 1, len(values) - 1)
        result[f"p{percentile}"] = values[lower] + (point - lower) * (values[upper] - values[lower])
    return result


def rows(path, record, run_events):
    timing = record["timings"]
    start = timing.get("measurement_started_elapsed_process_s")
    trace, samples, batches, traced_first, open_sidecars, last_event_s = trace_data(path, start)
    sample_times = [event["elapsed_s"] for event in samples]
    final_sample = samples[-1] if samples else {}
    measured_last = max(batches, default=0)
    reader, dataset = record["reader"], record["dataset"]
    root, verification, before, after = source_metadata(path, record, run_events)
    stop = next((event for event in reversed(run_events)
                 if event.get("event") == "stop" and event.get("label") == path.stem), {})
    progress = record.get("progress", {})
    io_final = record.get("io") or {}
    resources = record.get("resources") or {}
    status = record.get("status")
    final_scope = "terminal" if status in TERMINAL else "last_checkpoint"
    if final_scope != "terminal" and final_sample.get("elapsed_s", -1) > record.get("elapsed_process_s", 0):
        io_final, resources = final_sample.get("io", {}), final_sample
        final_scope = "last_resource_sample"
    final_artifact = trace if final_scope == "last_resource_sample" else path.name
    context = {
        "run": path.parent.name,
        "cell": path.stem,
        "reader": reader,
        "dataset": dataset,
        "workload": record.get("workload"),
        "cache_state": record.get("cache"),
        "tolerance_s": record.get("config", {}).get("tolerance_s", 0.0001),
        "cell_status": status,
        "stop_reason": record.get("stop_reason"),
        "failure": (record.get("error") or "").split("\n", 1)[0],
        "supervisor_exit_code": stop.get("code"),
        "supervisor_timed_out": stop.get("timed_out"),
        "source_root": root,
        "source_revision": record.get("source", {}).get("revision"),
        "source_verification": verification,
        "bucket_manifest_before": before,
        "bucket_manifest_after": after,
        "last_recorded_batch": progress.get("batches"),
        "last_traced_batch": measured_last if (path.parent / trace).exists() else None,
    }

    def metric(scope, target_batch, checkpoint, category, name, value, unit, artifact, note="", measurement=None):
        if measurement is None:
            measurement = "observed" if value is not None else "unavailable"
        return {
            **context, "scope": scope, "target_batch": target_batch, "checkpoint": checkpoint,
            "category": category, "metric": name, "value": value, "unit": unit,
            "measurement": measurement, "artifact": artifact, "note": note,
        }

    def total(category, name, value, unit, note="", measurement=None, artifact=None):
        return metric(
            "cell", None, final_scope, category, name, value, unit,
            artifact or final_artifact, note=note, measurement=measurement,
        )

    def snapshot(scope, target_batch, checkpoint, io, memory, io_artifact, memory_artifact, io_sampled=False):
        provenance = "sampled_before_delivery" if io_sampled else "observed"

        def io_status(value):
            return provenance if value is not None else "unavailable"

        for purpose in PURPOSES:
            counters = io.get("by_purpose", {}).get(purpose, {})
            file_metric = "distinct_mp4s_touched" if purpose in ("sidecar_probe", "video") else "distinct_named_files_touched"
            yield metric(scope, target_batch, checkpoint, purpose, file_metric,
                         counters.get("files_touched"), "files", io_artifact, FILES_NOTE,
                         measurement=io_status(counters.get("files_touched")))
            yield metric(scope, target_batch, checkpoint, purpose, "http_body_bytes",
                         counters.get("http_body_bytes"), "bytes", io_artifact, HTTP_NOTE,
                         measurement=io_status(counters.get("http_body_bytes")))
        for stage, category in (COMMON_STAGES | STAGES.get(reader, {})).items():
            timers = io.get("stage_durations", {}).get(stage, {})
            for key, unit in (("wall_s_sum", "s"), ("calling_thread_cpu_s_sum", "s"),
                              ("calls", "calls"), ("errors", "calls")):
                yield metric(scope, target_batch, checkpoint, category, f"{stage}:{key}",
                             timers.get(key), unit, io_artifact, STAGE_NOTE,
                             measurement=io_status(timers.get(key)))
        for key, note in (
            ("rss_sampled_peak_bytes", MEMORY_NOTE),
            ("cgroup_sampled_peak_bytes", "Sampled peak for the job cgroup, including filesystem cache."),
            ("net_namespace_rx_bytes", "Namespace total can include non-reader traffic and protocol overhead."),
        ):
            yield metric(scope, target_batch, checkpoint, "resources", key,
                         memory.get(key), "bytes", memory_artifact, note)

    yield total("timing", "constructor_s", timing.get("constructor_s"), "s", artifact=path.name)
    first = timing.get("first_sample_s")
    yield total("timing", "first_sample_s", traced_first if first is None else first, "s",
                artifact=trace if first is None else path.name)
    yield total("progress", "raw_samples", progress.get("raw_samples"), "samples", artifact=path.name)
    yield total("progress", "usable_unique_samples", progress.get("usable_unique_samples"), "samples", artifact=path.name)

    sidecar = io_final.get("sidecar", {})
    observed = record.get("source", {}).get("observed", {})
    archive_size = observed.get("baseline_sidecar_bytes")
    archive_state = "not_applicable" if reader == "candidate" else None
    yield total("index", "installed_archive_bytes", archive_size, "bytes",
                "Compressed index archive on disk.", measurement=archive_state, artifact=path.name)
    for phase, name in (("sidecar_probe", "native_build_s"), ("sidecar_download", "native_download_attempt_s")):
        result = sidecar.get(phase, {})
        seconds = result.get("s")
        state = result.get("status")
        artifact = final_artifact
        if seconds is None and phase in open_sidecars and last_event_s is not None:
            seconds = last_event_s - open_sidecars[phase]
            state = "observed_lower_bound"
            artifact = trace
        if state is None and archive_size is not None and "sidecar" in io_final:
            state = "not_invoked"
        yield total("index", name, seconds, "s", measurement=state or archive_state, artifact=artifact)

    prepared = reader == "baseline" and "-prepared-" in path.stem
    transfer_path = prepared_transfer_path(path, dataset, run_events)
    transfer = read(transfer_path).get("private_transfer", {}) if prepared and transfer_path.is_file() else {}
    origin = None
    if archive_size is not None:
        if sidecar.get("sidecar_probe", {}).get("status") == "ok":
            origin = "native_build"
        elif sidecar.get("sidecar_download", {}).get("status") == "ok":
            origin = "native_publisher_download"
        elif transfer.get("size_bytes") == archive_size:
            origin = "private_prepared_archive"
        elif not prepared and "sidecar" in io_final:
            origin = "local_archive"
    yield total("index", "installed_archive_origin", origin, "label", measurement=archive_state)
    for key, unit in (("consumer_prep_seconds", "s"), ("network_transfer_and_link_seconds", "s"), ("size_bytes", "bytes")):
        yield total("index_transfer", key, transfer.get(key), unit,
                    "Controlled private archive transfer recorded separately from the consumer.",
                    measurement=None if prepared else "not_applicable", artifact=transfer_path.name if prepared else path.name)

    yield from snapshot("cell", None, final_scope, io_final, resources, final_artifact, final_artifact)

    windows = record.get("windows", {})
    prefix = windows.get("to_1000", {})
    for target in MILESTONES:
        milestone = timing.get("milestones", {}).get(str(target))
        traced = batches.get(target)
        checkpoint = "reached" if milestone or traced else "not_reached"
        yield metric("prefix", target, checkpoint, "timing", "delivery_elapsed_s",
                     milestone.get("elapsed_s") if milestone else (traced.measurement_elapsed_s if traced else None),
                     "s", path.name if milestone else trace)
        anchor = (
            start + milestone["elapsed_s"] if milestone and start is not None
            else traced.process_elapsed_s if traced else None
        )
        sample_index = bisect_right(sample_times, anchor) - 1 if anchor is not None else -1
        sampled = samples[sample_index] if sample_index >= 0 else {}
        prefix_io = milestone.get("io", {}) if milestone else (sampled.get("io", {}) if traced else {})
        io_artifact = path.name if milestone else trace
        yield metric("prefix", target, checkpoint, "resources", "resource_sample_before_delivery_s",
                     anchor - sampled["elapsed_s"] if sampled else None, "s", trace)
        yield from snapshot("prefix", target, checkpoint, prefix_io, sampled, io_artifact, trace,
                            io_sampled=not milestone and bool(traced))
        batch_prefix = [batches.get(number) for number in range(1, target + 1)]
        trace_has_prefix = all(batch is not None for batch in batch_prefix)
        for key in ("raw_samples", "unique_samples", "usable_unique_samples"):
            count = milestone.get(key) if milestone else None
            if not milestone and trace_has_prefix:
                counts = [getattr(batch, key) for batch in batch_prefix]
                if all(count is not None for count in counts):
                    count = sum(counts)
            yield metric("prefix", target, checkpoint, "progress", key, count, "samples", path.name if milestone else trace)
        if trace_has_prefix and all(batch.wait_s is not None for batch in batch_prefix):
            waits, artifact = quantiles([batch.wait_s for batch in batch_prefix]), trace
        elif milestone and prefix.get("batches") == target:
            waits, artifact = prefix.get("batch_wait_s", {}), path.name
        else:
            waits, artifact = {}, trace
        for percentile in ("p50", "p95", "p99"):
            yield metric("prefix", target, checkpoint, "wait", f"batch_wait_{percentile}_s",
                         waits.get(percentile), "s", artifact, "Prefix percentiles include the first batch wait.")

    for key, window in (("available_prefix", prefix), ("after_batch_125", windows.get("after_batch_125", {})),
                        ("after_refill_2", windows.get("after_refill_2", {}))):
        count = window.get("batches")
        waits = window.get("batch_wait_s", {})
        artifact = path.name
        reached = min(measured_last, 1000)
        if key == "available_prefix" and reached > (count or 0):
            trace_waits = [batches[number].wait_s for number in range(1, reached + 1) if number in batches]
            if len(trace_waits) == reached and all(wait is not None for wait in trace_waits):
                count, waits, artifact = reached, quantiles(trace_waits), trace
        checkpoint = (
            "unavailable" if not count else "partial" if key == "available_prefix" and count < 1000 else "observed"
        )
        yield metric(key, None, checkpoint, "progress", "batches", count, "batches", artifact)
        for percentile in ("p50", "p95", "p99"):
            yield metric(key, None, checkpoint, "wait", f"batch_wait_{percentile}_s",
                         waits.get(percentile), "s", artifact)



def export(directory, stream):
    worker_fields = {"reader", "dataset", "timings", "progress"}
    run_events = {}
    writer = None
    for path in sorted(directory.rglob("*.json")):
        if ".cache" in path.relative_to(directory).parts:
            continue
        record = read(path)
        if not isinstance(record, dict) or not worker_fields <= record.keys():
            continue
        if path.parent not in run_events:
            run_events[path.parent] = list(jsonl(path.parent / "run.jsonl"))
        for row in rows(path, record, run_events[path.parent]):
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
    return writer is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Root directory of downloaded native worker results.")
    parser.add_argument("--output", type=Path, help="CSV destination. The default is standard output.")
    args = parser.parse_args()
    if args.output:
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            found = export(args.directory, stream)
    else:
        found = export(args.directory, sys.stdout)
    if not found:
        parser.exit(1, f"No worker result JSON files found under {args.directory}\n")


if __name__ == "__main__":
    main()
