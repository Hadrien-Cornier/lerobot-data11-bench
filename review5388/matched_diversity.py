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


def verify_receipt_arguments(start, transfer, dataset):
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
    if not output or Path(output[0]).name != f"{dataset}-private-transfer.json":
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
    states = {
        "prepared": ("unclassified", "unclassified"),
        "current-frame-tolerance-1ms-prepared": ("unclassified", "cold"),
    }
    if (
        pair not in states
        or base_path.stem != f"{dataset}-{pair}-baseline"
        or candidate_path.stem != f"{dataset}-{pair}-candidate"
        or (base["cache"], candidate["cache"]) != states[pair]
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

    label = f"{dataset}-private-transfer"
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
    verify_receipt_arguments(start, transfer, dataset)
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
        setups = [event for event in events[:events.index(start)] if event.get("event") == "tolerance-control-setup"]
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
        if read(directory / "bucket-manifest-tolerance-1ms-before-transfer.json")["manifest_sha256"] != sources[0][2]:
            raise ValueError("The bucket inventory before receipt differs from the paired inventory.")
        audit_directory = directory.parent / source_run
        original = read(audit_directory / receipt_path.name)["private_transfer"]
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


def read_prefix(path, limit, batch_size):
    trace = path.with_suffix('.events.jsonl.gz')
    if not trace.is_file():
        trace = path.with_suffix('.events.jsonl')
    open_trace = gzip.open if trace.suffix == '.gz' else open
    strata = defaultdict(list)
    issues = []
    with open_trace(trace, 'rt') as stream:
        for line in stream:
            event = json.loads(line)
            if event['event'] != 'batch':
                continue
            if event['batch'] > limit:
                break
            if event['usable_unique'] != batch_size or event['contract_errors']:
                issues.append(event['batch'])
                continue
            # Integer counts avoid grouping on rounded proportions.
            key = (event['distinct_episodes'], round(event['largest_episode_share'] * batch_size),
                   event['near_0p2s_pairs'])
            strata[key].append(event['wait_s'])
    return strata, issues


def compare(left, right, limit, *, prepared=False):
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
    publication = None
    if prepared:
        if not hosts[0].get("job_id") or not hosts[0].get("hostname") or hosts[0]["hostname"] != hosts[1].get("hostname"):
            raise ValueError("The prepared pair must run on the same recorded host.")
        publication = prepared_receipt(paths, rows, run_events(left))
    batch_size = rows[0]['config']['batch_size']
    a, b = [read_prefix(p, limit, batch_size) for p in paths]
    report = {'cells': [p.stem for p in paths], 'cutoff_batches': limit,
              'valid_unique_batches': [sum(map(len, v[0].values())) for v in (a, b)],
              'invalid_or_repeated_batches': [len(a[1]), len(b[1])], 'wait_clock': WAIT_CLOCK}
    if publication:
        report["publication"] = publication
    if report['valid_unique_batches'] != [limit, limit] or a[1] or b[1]:
        return {**report, 'status': 'incomplete or invalid prefix'}
    strata = []
    weights = []
    for key in sorted(a[0].keys() & b[0].keys()):
        waits = [a[0][key], b[0][key]]
        counts = [len(s) for s in waits]
        weight = min(counts) / limit
        weights.append(weight)
        strata.append({'episodes': key[0], 'largest_episode_examples': key[1],
                       'same_episode_pairs_within_0p2s': key[2], 'batches': counts,
                       'mean_wait_s': [mean(s) for s in waits], 'median_wait_s': [median(s) for s in waits]})
    overlap = sum(weights)
    standardized = [sum(w * s['mean_wait_s'][i] for w, s in zip(weights, strata)) / overlap
                    for i in range(2)] if overlap else None
    return {**report, 'status': 'complete', 'distribution_overlap': overlap,
            'overlap_weighted_mean_consumer_wait_s': standardized, 'strata': strata}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument("--prepared", action="store_true", help="Verify the simulated-publication transfer and audit.")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error('The prefix must have a positive batch count.')
    try:
        result = compare(args.baseline, args.candidate, args.limit, prepared=args.prepared)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
