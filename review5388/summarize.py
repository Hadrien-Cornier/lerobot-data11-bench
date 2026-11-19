"""Summarize measured HF cells, including unfinished runs, from downloaded job folders."""

import argparse
import csv
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def fixture_passes(directory, reader, workload):
    if directory is None:
        return False
    # The baseline's full temporal fixture includes all the checks for a current frame.
    # The draft's temporal and current-frame cases must qualify independently.
    candidates = ["baseline-w1.json"] if reader == "baseline" else [
        "proposal-w1-anchors.json" if workload == "single_frame" else "proposal-w1-full.json"
    ]
    for name in candidates:
        path = directory / name
        if not path.is_file():
            continue
        record = read(path)
        count = record.get("expected_full")
        ids = record.get("ids", [])
        if (count and len(ids) == len(set(ids)) == count and not record.get("error")
                and not record.get("unexpected") and not record.get("checks_failed")):
            return True
    return False


def source_verified(path, record):
    if record["dataset"] != "bucket":
        return bool(record.get("source", {}).get("revision"))
    events = path.parent / "run.jsonl"
    if not events.is_file():
        return None
    for line in events.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "comparison-source" and event.get("label") == path.stem:
            return event.get("source_valid")
        if event.get("event") == "pair-done" and event.get("dataset") == "bucket":
            prefix = f"bucket-{event['pair']}-"
            if path.stem == prefix + record["reader"]:
                return event.get("source_valid")
    return None


def row(path, record, fixture):
    name = path.stem
    reader, dataset, workload = (record[k] for k in ("reader", "dataset", "workload"))
    prep = 0
    if "prepared" in name and reader == "baseline":
        transfer = path.parent / f"{dataset}-private-transfer.json"
        prep = read(transfer)["private_transfer"]["consumer_prep_seconds"] if transfer.exists() else None

    timing = record["timings"]
    milestones = timing["milestones"]
    def elapsed(n, require_valid=False):
        event = milestones.get(str(n))
        if prep is None or event is None:
            return None
        if require_valid and event.get("usable_unique_samples") != n * record["config"]["batch_size"]:
            return None
        return prep + event["elapsed_s"]

    windows = record["windows"]
    initial = windows.get("to_1000", {})
    diversity = initial.get("episode_diversity", {})
    tail = windows.get("after_refill_2", {})
    refills = record["turnover"].get("refills_verified", 0)
    tail_valid = (refills >= 2 and tail.get("batches", 0) >= record["config"]["min_steady_batches"]
                  and tail.get("raw_samples") == tail.get("usable_unique_samples")
                  and not record["contract"].get("invalid_batches"))
    after_start = windows.get("after_batch_125", {})
    refill_indices = record["turnover"].get("refill_batches", [])
    across_refills = (
        tail_valid and sum(index > 125 for index in refill_indices) >= 2
        and after_start.get("raw_samples") == after_start.get("usable_unique_samples")
    )
    rate = after_start.get("usable_unique_samples_per_s") if across_refills else None
    counts = record["progress"]
    milestone = milestones.get("1000", {}).get("io", {})
    io = milestone.get("by_purpose", {})
    size = record.get("source", {}).get("observed", {}).get("frames")
    actual = record.get("source", {}).get("observed", {})
    index_build = record.get("io", {}).get("sidecar", {}).get("sidecar_probe", {})
    verified = source_verified(path, record)
    near_pairs = diversity.get("near_0p2s_pairs")
    pairs = diversity.get("unordered_pairs")
    first = elapsed(1, True)
    projection = None
    if (verified and rate and size and first is not None and counts.get("duplicates") == 0
            and fixture_passes(fixture, reader, workload)):
        projection = (first + (size - record["config"]["batch_size"]) / rate) / 86400
    return {
        "run": path.parent.name, "cell": name, "dataset": dataset, "reader": reader,
        "workload": workload, "tolerance_ms": 1000 * record["config"].get("tolerance_s", 0.0001),
        "cache": record["cache"], "status": record["status"],
        "source_verified": verified,
        "raw_examples": counts["raw_samples"], "usable_unique": counts["usable_unique_samples"],
        "duplicates": counts["duplicates"], "added_consumer_prep_s": prep,
        "constructor_s": timing.get("constructor_s"),
        "first_sample_s": (prep + timing["first_sample_s"]
                           if prep is not None and "first_sample_s" in timing else None),
        "batch_1_raw_s": elapsed(1), "batch_1_valid_s": first, "batch_125_valid_s": elapsed(125, True),
        "batch_1000_raw_s": elapsed(1000), "batch_1000_valid_s": elapsed(1000, True),
        "episodes_per_batch": diversity.get("distinct_episodes_per_batch_mean"),
        "largest_episode_share": diversity.get("largest_episode_share_mean"),
        "near_sample_pair_share_0p2s": near_pairs / pairs if pairs and near_pairs is not None else None,
        "verified_refills": refills,
        "post_refill_batches": tail.get("batches"), "steady_windows_per_s": rate,
        "post_refill_tail_windows_per_s": tail.get("usable_unique_samples_per_s") if tail_valid else None,
        "prefix_median_wait_s": initial.get("batch_wait_s", {}).get("p50"),
        "prefix_p95_wait_s": initial.get("batch_wait_s", {}).get("p95"),
        "prefix_p99_wait_s": initial.get("batch_wait_s", {}).get("p99"),
        "video_files_by_batch_1000": io.get("video", {}).get("files_touched"),
        "http_video_bytes_by_batch_1000": io.get("video", {}).get("http_body_bytes"),
        "index_archive_bytes": actual.get("baseline_sidecar_bytes"),
        "native_index_build_s": index_build.get("s") if index_build.get("status") == "ok" else None,
        "rss_peak_bytes": record["resources"].get("rss_sampled_peak_bytes"),
        "full_pass_days_estimate": projection,
        "stop": record.get("stop_reason") or record.get("error", "").partition("\n")[0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--fixture-evidence", type=Path)
    args = parser.parse_args()
    writer = None
    for path in sorted(args.directory.rglob("*.json")):
        record = read(path)
        if not isinstance(record, dict) or not {"timings", "reader", "turnover"} <= set(record):
            continue
        result = row(path, record, args.fixture_evidence)
        if writer is None:
            writer = csv.DictWriter(sys.stdout, fieldnames=list(result))
            writer.writeheader()
        writer.writerow(result)


if __name__ == "__main__":
    main()
