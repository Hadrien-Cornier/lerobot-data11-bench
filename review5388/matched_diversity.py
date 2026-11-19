"""Compare observed DataLoader waits at the same episode mix.

Reader construction and simulated-publication receipt are outside the wait clock.
"""

import argparse
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from resource_summary import source_metadata
from summarize import read, run_events, source_verified

TRANSFER_FIELDS = ("path", "revision", "sha256", "relative_cache_path")
WAIT_CLOCK = "DataLoader next() wait. Reader construction and prior index receipt are outside this clock."


def one_event(events, kind, label):
    found = [event for event in events if event.get("event") == kind and event.get("label") == label]
    if len(found) != 1:
        raise ValueError(f"Expected one {kind} event for {label}.")
    return found[0]


def argument_values(arguments, option):
    return arguments[arguments.index(option) + 1:arguments.index(option) + 2] if arguments.count(option) == 1 else []


def verify_receipt_arguments(start, transfer, dataset, receipt_name):
    arguments = start.get("argv", [])
    required = {
        "--dataset": dataset,
        "--private-repo": transfer["repo"],
        "--private-path": transfer["path"],
        "--private-revision": transfer["revision"],
        "--private-destination": transfer["relative_cache_path"],
        "--private-sha256": transfer["sha256"],
    }
    if not any(Path(argument).as_posix().endswith("runner/publisher.py") for argument in arguments):
        raise ValueError("No native publisher command was recorded for the receipt.")
    for option, value in required.items():
        if argument_values(arguments, option) != [value]:
            raise ValueError(f"Private receipt disagrees with the publisher argument {option}.")
    output = argument_values(arguments, "--out")
    if not output or Path(output[0]).name != receipt_name:
        raise ValueError("The recorded publisher command wrote a different receipt.")


def verify_reader_command(start, path, row):
    args = start.get("argv", [])
    output = argument_values(args, "--output")
    if (
        not args or Path(args[0]).name != "benchmark.py"
        or argument_values(args, "--reader") != [row["reader"]]
        or argument_values(args, "--dataset") != [row["dataset"]]
        or argument_values(args, "--cache-state") != [row["cache"]]
        or "--warm-batches" in args
        or not output or Path(output[0]).name != path.name
    ):
        raise ValueError("The logged reader command does not match the fresh stock benchmark cell.")


def prepared_receipt(paths, rows, events):
    """Validate the isolated prepared baseline and fresh stock candidate."""
    by_reader = {row["reader"]: (path, row) for path, row in zip(paths, rows)}
    base_path, base = by_reader["baseline"]
    candidate_path, candidate = by_reader["candidate"]
    directory = base_path.parent
    dataset = base["dataset"]
    if directory.resolve() != candidate_path.parent.resolve():
        raise ValueError("Use both prepared pair cells from the same run directory.")
    pair = base_path.stem.removeprefix(f"{dataset}-").removesuffix("-baseline")
    repeat = re.fullmatch(r"current-frame-tolerance-1ms-prepared-(r[0-9]{2})", pair)
    states = {
        "prepared": ("unclassified", "unclassified"),
        "current-frame-tolerance-1ms-prepared": ("unclassified", "cold"),
    }
    expected_states = states["current-frame-tolerance-1ms-prepared"] if repeat else states.get(pair)
    if (
        base_path.stem != f"{dataset}-{pair}-baseline"
        or candidate_path.stem != f"{dataset}-{pair}-candidate"
        or (base["cache"], candidate["cache"]) != expected_states
        or any(row["config"].get("warm_batches", 0) for row in rows)
    ):
        raise ValueError("Use the logged prepared baseline and fresh stock candidate pair.")
    completed = [
        event for event in events if event.get("event") == "pair-done"
        and event.get("dataset") == dataset and event.get("pair") == pair
    ]
    if len(completed) != 1 or completed[0].get("source_valid") is not True:
        raise ValueError("The prepared pair source validation is missing or failed.")
    expected_workload = ("single_frame" if completed[0].get("single_frame") else
                         "action100_history_0p8s" if completed[0].get("history") else "action100")
    if base["workload"] != expected_workload or base["config"].get("tolerance_s", 0.0001) != completed[0].get(
        "tolerance_s", 0.0001
    ):
        raise ValueError("The prepared pair ledger disagrees with the worker workload.")

    sources = [source_metadata(path, row, events) for path, row in zip(paths, rows)]
    verification = "bucket_manifest_match" if dataset == "bucket" else "revision_verified"
    if any(source[1] != verification for source in sources) or sources[0] != sources[1]:
        raise ValueError("The prepared pair did not verify one shared source revision or bucket inventory.")

    suffix = f"-{repeat[1]}" if repeat else ""
    label = f"{dataset}-private-transfer{suffix}"
    receipt_path = directory / f"{label}.json"
    receipt = read(receipt_path)
    transfer = receipt["private_transfer"]
    prep = transfer["consumer_prep_seconds"]
    transfer_and_link = transfer["network_transfer_and_link_seconds"]
    archive_path = Path(transfer["path"])
    relative = Path(transfer["relative_cache_path"])
    if (
        receipt["dataset"] != dataset
        or not re.fullmatch(r"[0-9a-f]{64}", transfer["sha256"])
        or not re.fullmatch(r"[0-9a-f]{40}", transfer["revision"])
        or not (0 < transfer_and_link <= prep and math.isfinite(prep))
        or archive_path.name != relative.name
        or relative.is_absolute()
        or relative.parts[:1] != ("streaming-sidecars",)
        or ".." in relative.parts
        or transfer["size_bytes"] <= 0
    ):
        raise ValueError("The private transfer receipt is incomplete or invalid.")

    start = one_event(events, "start", label)
    stop = one_event(events, "stop", label)
    base_start = one_event(events, "start", base_path.stem)
    candidate_start = one_event(events, "start", candidate_path.stem)
    completed_rows = [one_event(events, "stop", path.stem) for path in (base_path, candidate_path)]
    if stop.get("code") != 0 or stop.get("timed_out") is not False or stop["seconds"] < prep:
        raise ValueError("The checksum-verifying transfer did not complete in the recorded interval.")
    if not all(
        events.index(start) < events.index(stop) < events.index(measurement_start)
        < events.index(measurement_stop) < events.index(completed[0])
        for measurement_start, measurement_stop in zip((base_start, candidate_start), completed_rows)
    ):
        raise ValueError("Run order does not place the transfer before both measured readers.")
    baseline_cache, candidate_cache = base_start.get("cache"), candidate_start.get("cache")
    if (
        not baseline_cache or not candidate_cache
        or baseline_cache == candidate_cache
        or start.get("cache") != baseline_cache
    ):
        raise ValueError("Transfer and baseline must share one cache, separate from the fresh candidate.")
    earlier_starts = [
        event for event in events[:events.index(base_start)]
        if event.get("event") == "start" and event.get("cache") == baseline_cache
    ]
    if earlier_starts != [start] or any(
        event.get("event") == "start" and event.get("cache") == candidate_cache
        for event in events[:events.index(candidate_start)]
    ):
        raise ValueError("A prior stage already used a reader cache.")
    verify_receipt_arguments(start, transfer, dataset, receipt_path.name)
    verify_reader_command(base_start, base_path, base)
    verify_reader_command(candidate_start, candidate_path, candidate)

    audit_directory = directory
    if pair == "prepared":
        staged = [
            event for event in events[:events.index(start)] if event.get("event") == "private-transfer-staged"
            and event.get("dataset") == dataset
        ]
        if len(staged) != 1 or any(staged[0].get(key) != transfer[key] for key in TRANSFER_FIELDS[:-1]) or (
            staged[0].get("bytes") != transfer["size_bytes"]
        ):
            raise ValueError("The receipt does not match the native index staged by this job.")
    else:
        setups = [
            event for event in events[:events.index(start)]
            if event.get("event") == "tolerance-control-setup" and event.get("pair") in (None, pair)
        ]
        if len(setups) != 1:
            raise ValueError("The tolerance prepared-control provenance is absent.")
        setup = setups[0]
        reference = setup.get("prepared_index") or {}
        source_run = reference.get("source_run", "")
        if (
            dataset != "bucket" or base["workload"] != "single_frame" or setup.get("tolerance_s") != 0.001
            or setup.get("baseline_start_state") != "fresh_metadata_and_verified_private_transfer"
            or setup.get("candidate_start_state") != "fresh"
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,39}", source_run)
            or not re.fullmatch(r"[0-9a-f]{40}", reference.get("source_proof_revision", ""))
            or any(reference.get(key) != transfer[key] for key in TRANSFER_FIELDS)
        ):
            raise ValueError("The fresh transfer does not match the pinned publisher control.")
        pre_receipt_label = f"bucket-manifest-tolerance-1ms-before-transfer{suffix}"
        if read(directory / f"{pre_receipt_label}.json")["manifest_sha256"] != sources[0][2]:
            raise ValueError("The bucket inventory before receipt differs from the paired inventory.")
        if repeat:
            order = sorted((base_start, candidate_start), key=events.index)
            stops_by_label = {event["label"]: event for event in completed_rows}
            if (
                setup.get("pair") != pair
                or setup.get("repeat") != int(repeat[1][1:])
                or setup.get("reader_order") != [event["reader"] for event in order]
                or events.index(stops_by_label[order[0]["label"]]) >= events.index(order[1])
                or tuple(event.get("reader") for event in (base_start, candidate_start)) != ("baseline", "candidate")
            ):
                raise ValueError("The fresh reader stages overlap or differ from the declared order.")
            fixed_batches = completed[0].get("fixed_target")
            if (
                not fixed_batches or setup.get("max_batches") != fixed_batches
                or setup.get("transfer_label") != label
                or setup.get("before_transfer_label") != pre_receipt_label.removeprefix("bucket-manifest-")
                or any(row["config"].get("max_batches") != fixed_batches
                       or row["config"].get("continue_to_max_batches") is not True
                       or row["config"].get("seed_requested") != setup.get("seed_requested")
                       or row["config"].get("pool_size") != setup.get("pool_size") for row in rows)
            ):
                raise ValueError("The fresh repeat differs from the declared benchmark settings.")
            before = one_event(events, "stop", f"bucket-manifest-{pair}-before")
            after = one_event(events, "stop", f"bucket-manifest-{pair}-after")
            pre_receipt = one_event(events, "stop", pre_receipt_label)
            if (
                any(event.get("code") != 0 or event.get("timed_out") is not False
                    for event in (before, after, pre_receipt))
                or not events.index(pre_receipt) < events.index(start) < events.index(stop)
                < events.index(before) < events.index(order[0])
                or not max(map(events.index, completed_rows)) < events.index(after) < events.index(completed[0])
            ):
                raise ValueError("The three successful bucket inventories do not bracket receipt and delivery.")
        audit_directory = directory.parent / source_run
        original = read(audit_directory / f"{dataset}-private-transfer.json")["private_transfer"]
        if any(original.get(key) != transfer[key] for key in (*TRANSFER_FIELDS, "repo", "size_bytes")):
            raise ValueError("The original publisher receipt identifies a different private archive.")
        for phase in ("before", "after"):
            if read(audit_directory / f"bucket-manifest-prepared-{phase}.json")["manifest_sha256"] != sources[0][2]:
                raise ValueError("The index audit came from a different bucket inventory.")

    audit_path = audit_directory / f"{dataset}-publisher-audit.json"
    audit = read(audit_path)
    audited = [entry for entry in audit.get("publisher_files", []) if entry.get("url") == audit.get("expected_url")]
    if (
        audit.get("dataset") != dataset or audit.get("data_root") != sources[0][0]
        or audit.get("relative_cache_path") != transfer["relative_cache_path"]
        or len(audited) != 1 or audited[0].get("status") not in {"absent", "matching_header", "incompatible"}
        or base["source"].get("observed", {}).get("baseline_sidecar_bytes") != transfer["size_bytes"]
        or base.get("io", {}).get("sidecar") != {}
    ):
        raise ValueError("The publisher audit or native baseline archive does not match the verified receipt.")
    return {
        "mode": "simulated_publication",
        "metadata_start_state": "fresh isolated reader caches; the baseline cache initially contains the transferred index",
        "public_archive_status_at_audit": audited[0]["status"],
        "baseline_consumer_prep_seconds": prep,
        "baseline_transfer_and_link_seconds": transfer_and_link,
        "baseline_archive_bytes": transfer["size_bytes"],
        "baseline_archive_sha256": transfer["sha256"],
        "private_archive_revision": transfer["revision"],
        "audit": str(audit_path),
        "receipt": str(receipt_path),
    }


def read_prefix(path, limit, batch_size, after_batch):
    trace = path.with_suffix(".events.jsonl.gz")
    if not trace.is_file():
        trace = path.with_suffix(".events.jsonl")
    open_trace = gzip.open if trace.suffix == ".gz" else open
    strata = defaultdict(list)
    seen = set()
    issues = []
    observed = 0
    with open_trace(trace, "rt", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] != "batch":
                continue
            batch = event["batch"]
            if batch > limit:
                break
            observed += 1
            ids = event["ids"]
            malformed = any(len(anchor) != 2 or any(type(value) is not int or value < 0 for value in anchor)
                            for anchor in ids)
            if malformed:
                issues.append(batch)
                continue
            anchors = set(map(tuple, ids))
            valid = (
                type(batch) is int and batch == observed and len(ids) == batch_size
                and len(anchors - seen) == batch_size
                and event["unique"] == batch_size and event["usable_unique"] == batch_size
                and not event["contract_errors"] and math.isfinite(event["wait_s"]) and event["wait_s"] >= 0
            )
            seen.update(anchors)
            if not valid:
                issues.append(batch)
                continue
            if batch > after_batch:
                # Integer counts avoid grouping on rounded proportions.
                key = (event["distinct_episodes"], round(event["largest_episode_share"] * batch_size),
                       event["near_0p2s_pairs"])
                strata[key].append(event["wait_s"])
    return strata, issues, observed


def compare(left, right, limit, *, prepared=False, after_batch=0):
    if not 0 <= after_batch < limit:
        raise ValueError("The interval must satisfy 0 <= after_batch < limit.")
    paths = [left, right]
    rows = [read(p) for p in paths]
    if rows[0]['dataset'] != rows[1]['dataset'] or rows[0]['workload'] != rows[1]['workload']:
        raise ValueError('The two measurements must use the same dataset and workload.')
    if {r['reader'] for r in rows} != {'baseline', 'candidate'}:
        raise ValueError('Specify one result from each reader.')
    for key, default in [('tolerance_s', 0.0001), ('batch_size', None), ('workers', None), ('pool_size', None)]:
        if rows[0]['config'].get(key, default) != rows[1]['config'].get(key, default):
            raise ValueError(f'Reader settings differ for {key}.')
    if (rows[0]['source']['revision'] != rows[1]['source']['revision'] or
            any(not source_verified(p, r, run_events(p)) for p, r in zip(paths, rows))):
        raise ValueError('Verify both source revisions or both bucket manifests before comparison.')
    hosts = [read(path.parent / "host.json") for path in paths]
    if hosts[0]["job_id"] != hosts[1]["job_id"] or (
        not prepared and rows[0]["cache"] != rows[1]["cache"]
    ):
        raise ValueError("Use runs from the same job with the same initial cache state.")
    if (prepared or after_batch) and (
        not hosts[0].get("job_id")
        or not hosts[0].get("hostname")
        or hosts[0]["hostname"] != hosts[1].get("hostname")
    ):
        raise ValueError("Use results from the same recorded host.")
    if after_batch and not prepared:
        sources = [source_metadata(path, row, run_events(path)) for path, row in zip(paths, rows)]
        required_verification = "bucket_manifest_match" if rows[0]["dataset"] == "bucket" else "revision_verified"
        if sources[0] != sources[1] or sources[0][1] != required_verification:
            raise ValueError("Use the same verified source revision or bucket inventory.")
    publication = prepared_receipt(paths, rows, run_events(left)) if prepared else None
    batch_size = rows[0]["config"]["batch_size"]
    prefixes = [read_prefix(path, limit, batch_size, after_batch) for path in paths]
    a, b = [prefix[0] for prefix in prefixes]
    included_batches = limit - after_batch
    report = {
        "cells": [path.stem for path in paths], "cutoff_batches": limit, "after_batch": after_batch,
        "included_batches_per_reader": included_batches,
        "prefix_batches_seen": [prefix[2] for prefix in prefixes],
        "valid_unique_batches": [sum(map(len, strata.values())) for strata in (a, b)],
        "invalid_or_repeated_batches": [len(prefix[1]) for prefix in prefixes],
        "first_invalid_or_repeated_batch": [prefix[1][0] if prefix[1] else None for prefix in prefixes],
        "wait_clock": WAIT_CLOCK,
    }
    if publication:
        report["publication"] = publication
    if (
        report["prefix_batches_seen"] != [limit, limit]
        or report["valid_unique_batches"] != [included_batches, included_batches]
        or any(report["invalid_or_repeated_batches"])
    ):
        return report | {"status": "incomplete or invalid prefix"}
    strata = []
    weights = []
    for key in sorted(a.keys() & b.keys()):
        waits = [a[key], b[key]]
        counts = [len(s) for s in waits]
        weight = min(counts) / included_batches
        weights.append(weight)
        strata.append({'episodes': key[0], 'largest_episode_examples': key[1],
                       'same_episode_pairs_within_0p2s': key[2], 'batches': counts,
                       'mean_wait_s': [mean(s) for s in waits], 'median_wait_s': [median(s) for s in waits]})
    overlap = sum(weights)
    standardized = [sum(w * s['mean_wait_s'][i] for w, s in zip(weights, strata)) / overlap
                    for i in range(2)] if overlap else None
    return {
        **report, "status": "complete", "distribution_overlap": overlap,
        "overlap_weighted_mean_consumer_wait_s": standardized,
        "unconditioned_mean_consumer_wait_s": [
            mean(wait for waits in distribution.values() for wait in waits) for distribution in (a, b)
        ],
        "unconditioned_mean_distinct_episodes_per_batch": [
            sum(key[0] * len(waits) for key, waits in distribution.items()) / included_batches
            for distribution in (a, b)
        ],
        "strata": strata,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument("--limit", type=int, default=1000, help="Last included batch, inclusive.")
    parser.add_argument("--after-batch", type=int, default=0, help="Exclude deliveries through this batch from estimates.")
    parser.add_argument("--prepared", action="store_true", help="Verify the simulated-publication transfer and audit.")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error('The prefix must have a positive batch count.')
    try:
        result = compare(args.baseline, args.candidate, args.limit, prepared=args.prepared, after_batch=args.after_batch)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
