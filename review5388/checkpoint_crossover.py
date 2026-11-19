"""Audit the interrupted Molmo repeat without changing its native result or ledger."""

import argparse
import csv
import gzip
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from statistics import mean

from benchmark import DATASETS_PIN, INPUTS, PINS, diversity
from matched_diversity import (
    WAIT_CLOCK,
    argument_values,
    one_event,
    verify_reader_command,
)
from observed_crossover import (
    CONTRACT_SETTINGS,
    PROGRESS_FIELDS,
    load_lengths,
    trace_path,
)

PAIR = "current-frame-tolerance-1ms-repeat"
REPO = "pepijn223/MolmoAct2-BimanualYAM-Dataset-bucket"
CATALOG_FIELDS = (
    "source", "revision", "manifest_sha256", "files", "video_files", "video_bytes",
    "data_parquets", "data_parquet_bytes", "sidecars", "total_frames", "total_episodes", "fps", "cameras",
)


def read(path):
    return json.loads(path.read_text())


def utc(value):
    time = datetime.fromisoformat(value)
    if time.utcoffset() is None:
        raise ValueError("Evidence has an unqualified timestamp")
    return time


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def identity(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def supervisor_proof(events, job, host, pre, post, rows, paths, pre_path):
    """Establish comparable starts, deadline termination, and the post-run source audit."""
    require(job["job_id"] == host["job_id"] and job["status"] == "COMPLETED"
            and bool(host["hostname"]) and bool(re.fullmatch(r"[0-9a-f]{40}", job["artifact_revision"])),
            "Completed job and synced artifact revision could not be established")
    before_label = pre_path.stem
    before_start = one_event(events, "start", before_label)
    before_stop = one_event(events, "stop", before_label)
    starts = [one_event(events, "start", path.stem) for path in paths]
    stops = [one_event(events, "stop", path.stem) for path in paths]
    missing_after = one_event(events, "skipped", f"bucket-manifest-{PAIR}-after")
    pair, = [event for event in events if event.get("event") == "pair-done" and event.get("pair") == PAIR]
    run_end, = [event for event in events if event.get("event") == "finish"]
    require(
        unavailable_after(events, missing_after) and pair.get("source_valid") is False
        and pair["dataset"] == "bucket" and pair["single_frame"] is True and pair["history"] is False
        and pair["tolerance_s"] == 0.001,
        "Native pair does not establish that its after-manifest was skipped"
    )
    require(
        all(pre[key] == post[key] for key in CATALOG_FIELDS)
        and pre["source"] == REPO and pre["revision"] is None
        and host["source_bucket_sha256"] == pre["manifest_sha256"]
        and before_start["argv"][0] == "catalog.py"
        and argument_values(before_start["argv"], "--dataset") == ["bucket"]
        and argument_values(before_start["argv"], "--expect") == [pre["manifest_sha256"]]
        and Path(argument_values(before_start["argv"], "--out")[0]).name == pre_path.name
        and before_stop["code"] == 0 and before_stop["timed_out"] is False,
        "Pre-run and independent post-run stock bucket manifests do not match"
    )
    bracket = [
        utc(before_start["utc"]), utc(pre["recorded_utc"]), utc(before_stop["utc"]),
        utc(starts[0]["utc"]), utc(stops[0]["utc"]), utc(starts[1]["utc"]), utc(stops[1]["utc"]),
        utc(missing_after["utc"]), utc(pair["utc"]), utc(run_end["utc"]), utc(job["finish"]),
        utc(post["recorded_utc"]),
    ]
    require(all(left < right for left, right in zip(bracket, bracket[1:])),  # noqa: RUF007 - local Python 3.9 support
            "Evidence does not place the independent source check after the actual stop and job finish")

    base, draft = rows
    for row, path, start in zip(rows, paths, starts):
        verify_reader_command(start, path, row)
        config = row["config"]
        observed = row["source"]["observed"]
        environment = row["environment"]
        pinned_revision, pinned_digest = PINS[row["reader"]]
        require(
            environment["lerobot_revision"] == pinned_revision
            and environment["lerobot_streaming_module_sha256"] == pinned_digest
            and environment["datasets_source"]["revision"] == DATASETS_PIN,
            f"Native reader or Datasets pin differs for {row['reader']}"
        )
        require(
            row["dataset"] == "bucket" and row["workload"] == "single_frame" and row["cache"] == "cold"
            and row["source"]["repo"] == REPO and row["source"]["revision"] is None
            and config["max_batches"] == pair["fixed_target"] and config["continue_to_max_batches"] is True
            and config["tolerance_s"] == 0.001 and config["workers"] == 0 and config["batch_size"] == 8
            and config.get("warm_batches", 0) == 0
            and "--continue-to-max-batches" in start["argv"] and "--single-frame" in start["argv"]
            and argument_values(start["argv"], "--max-batches") == [str(pair["fixed_target"])]
            and argument_values(start["argv"], "--tolerance-s") == ["0.001"]
            and observed["frames"] == pre["total_frames"] and observed["episodes"] == pre["total_episodes"]
            and observed["fps"] == pre["fps"] and sorted(observed["video_keys"]) == sorted(pre["cameras"]),
            f"Source or fixed current-frame workload differs for {row['reader']}"
        )
    require(
        (base["reader"], draft["reader"]) == ("baseline", "candidate")
        and all(base["config"].get(key, default) == draft["config"].get(key, default)
                for key, default in CONTRACT_SETTINGS.items()),
        "Reader settings differ"
    )
    for start in starts:
        require(bool(start.get("cache")) and not any(
            event.get("event") == "start" and event.get("cache") == start["cache"]
            for event in events[:events.index(start)]
        ), "The ledger shows a reused reader cache")
    require(starts[0]["cache"] != starts[1]["cache"], "Reader caches were not isolated")

    base_stop, draft_stop = stops
    require(
        base["status"] == "ok" and base_stop["code"] == 0 and base_stop["timed_out"] is False
        and base["progress"]["batches"] == pair["fixed_target"],
        "Baseline did not finish the fixed target"
    )
    memory_events = dict(line.split() for line in draft_stop["cgroup_memory_events"].splitlines())
    require(
        draft_stop["code"] == -9 and draft_stop["timed_out"] is True and draft["status"] == "running"
        and not draft.get("error") and 0 < draft["progress"]["batches"] <= 24000
        and draft["progress"]["batches"] < pair["fixed_target"],
        "Candidate is not the deadline-interrupted native checkpoint"
    )
    require(
        starts[1]["timeout_seconds"] <= draft_stop["seconds"] <= starts[1]["timeout_seconds"] + 20
        and starts[1]["timeout_seconds"] < draft["config"]["max_seconds"]
        and all(memory_events.get(key) == "0" for key in ("oom", "oom_kill", "oom_group_kill")),
        "SIGKILL does not match the recorded supervisory deadline or an OOM cannot be excluded"
    )
    return {"baseline": base_stop, "candidate": draft_stop, "native_pair": pair, "native_after_manifest": missing_after}


def unavailable_after(events, after):
    return after.get("reason") == "job time exhausted" and not any(
        event.get("label") == after["label"] and event.get("event") == "start" for event in events
    )


def verify_prefix(path, row, lengths, stop, *, interrupted):
    """Recount all identities and metrics through an actual atomically saved worker snapshot."""
    batch_size = row["config"]["batch_size"]
    progress = row["progress"]
    limit = progress["batches"]
    held_count = limit * batch_size
    require(
        progress == dict(zip(PROGRESS_FIELDS, (limit, held_count, held_count, held_count, 0)))
        and row["contract"]["invalid_batches"] == 0,
        f"{row['reader']} snapshot does not describe a full valid prefix"
    )
    _fps, vector_size, image_shape, _suffixes = INPUTS["bucket"]
    schema = row["contract"]["first_schema"]
    expected = {
        "observation.state": {"shape": [batch_size, vector_size], "dtype": "torch.float32"},
        "action": {"shape": [batch_size, vector_size], "dtype": "torch.float32"},
    }
    expected.update({
        camera: {"shape": [batch_size, *image_shape], "dtype": "torch.uint8"}
        for camera in row["source"]["observed"]["video_keys"]
    })
    require(all(schema.get(key) == value for key, value in expected.items()),
            f"{row['reader']} first checked schema disagrees with current-frame workload")
    trace = trace_path(path)
    opener = gzip.open if trace.suffix == ".gz" else open
    counts = set()
    timeline = []
    first_sample = None
    terminal = None
    last_trace_elapsed = None
    start_measurement = row["timings"]["measurement_started_elapsed_process_s"]
    started_wall = utc(row["started_utc"])
    require(
        row["elapsed_process_s"] <= (utc(stop["utc"]) - started_wall).total_seconds() + 1,
        f"{row['reader']} snapshot was time-stamped after its stop"
    )
    with opener(trace, "rt", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            kind = event["event"]
            if kind == "first_sample":
                require(first_sample is None, "Repeated first-sample trace event")
                first_sample = event["measurement_elapsed_s"]
            elif kind == "terminal":
                terminal = event
            if kind != "batch":
                continue

            index = len(timeline) + 1
            elapsed, wait = event["measurement_elapsed_s"], event["wait_s"]
            require(
                index <= limit and terminal is None and type(event["batch"]) is int and event["batch"] == index
                and math.isfinite(elapsed) and elapsed >= 0 and math.isfinite(wait) and 0 <= wait <= elapsed
                and (not timeline or elapsed >= timeline[-1][0])
                and event["elapsed_s"] <= row["elapsed_process_s"] + 1e-5
                and event["elapsed_s"] + 1e-5 >= start_measurement + elapsed,
                f"{row['reader']} batch {index} is missing, non-monotonic, or outside the saved checkpoint"
            )
            ids = event["ids"]
            require(
                len(ids) == batch_size and all(
                    type(episode) is int and type(frame) is int and 0 <= frame < lengths.get(episode, 0)
                    for episode, frame in ids
                ),
                f"{row['reader']} batch {index} does not contain a full in-bounds batch"
            )
            anchors = {tuple(anchor) for anchor in ids}
            require(len(anchors - counts) == batch_size and event["unique"] == event["usable_unique"] == batch_size
                    and event["contract_errors"] == [], f"{row['reader']} batch {index} is invalid or repeated")
            counts.update(anchors)

            episode_count, _, near, _, largest_share = diversity(ids, row["source"]["observed"]["fps"])
            require((episode_count, near, largest_share) == (
                event["distinct_episodes"], event["near_0p2s_pairs"], event["largest_episode_share"]
            ), f"{row['reader']} batch {index} episode-mix counters disagree with emitted identities")
            timeline.append((elapsed, wait, (episode_count, round(largest_share * batch_size), near)))
            last_trace_elapsed = event["elapsed_s"]
            if interrupted and index == limit:
                break

    require(len(timeline) == limit and len(counts) == held_count,
            f"{row['reader']} trace does not reach the saved checkpoint")
    if not interrupted:
        require(terminal is not None and terminal["status"] == row["status"]
                and all(terminal[key] == progress[key] for key in PROGRESS_FIELDS),
                "Baseline terminal event does not agree with its completed summary")
    require(first_sample == row["timings"]["first_sample_s"] and 0 <= first_sample <= timeline[0][0],
            f"{row['reader']} first-sample marker disagrees with its native summary")
    for batch, milestone in row["timings"]["milestones"].items():
        require(math.isclose(timeline[int(batch) - 1][0], milestone["elapsed_s"], rel_tol=1e-9),
                f"{row['reader']} milestone {batch} disagrees with the trace")
    return timeline, {
        "native_status": row["status"], "first_sample_s": first_sample,
        "lerobot_revision": row["environment"]["lerobot_revision"],
        "datasets_revision": row["environment"]["datasets_source"]["revision"],
        "saved_batches": limit, "in_bounds_unique_samples": held_count, "duplicates": 0, "invalid_batches": 0,
        "saved_elapsed_process_s": row["elapsed_process_s"], "last_valid_delivery_elapsed_process_s": last_trace_elapsed,
        "native_refill_batches": row["turnover"]["refill_batches"],
        "cursor": "last atomically saved worker checkpoint" if interrupted else "verified terminal event",
        "tail_not_examined_beyond_checkpoint": interrupted,
    }


def lead_changes(series, minimum):
    runs = []
    previous = None
    for entry in series:
        if entry["lead"] == "tie":
            continue
        if not runs or runs[-1]["reader"] != entry["lead"]:
            runs.append({
                "reader": entry["lead"], "first_batch": entry["batch"], "strict_lead_batches": 0,
                "previous_leader_last_delivery": previous, "leader_first_delivery": entry,
            })
        runs[-1]["strict_lead_batches"] += 1
        runs[-1]["last_strict_lead_batch"] = entry["batch"]
        previous = entry
    changes = runs[1:]
    final_lead = runs[-1] if runs else None
    return {
        "lead_change_count": len(changes), "lead_changes": changes, "initial_lead": runs[0] if runs else None,
        "first_durable_lead_change": next(
            (run for run in changes if run["strict_lead_batches"] >= minimum), None
        ),
        "final_lead_through_checkpoint": final_lead,
        "durable_lead_threshold_batches": minimum,
        "ties": sum(entry["lead"] == "tie" for entry in series),
    }


def matched_window(timelines, after, stop):
    included = stop - after
    distributions = [defaultdict(list), defaultdict(list)]
    segments = [values[after:stop] for values in timelines]
    for strata, segment in zip(distributions, segments):
        for _elapsed, wait, key in segment:
            strata[key].append(wait)
    baseline, candidate = distributions
    overlap_counts = {key: min(len(baseline[key]), len(candidate[key])) for key in baseline.keys() & candidate.keys()}
    matches = sum(overlap_counts.values())
    return {
        "after_batch": after, "through_batch": stop, "batches_per_reader": included,
        "distribution_overlap": matches / included,
        "unconditioned_mean_consumer_wait_s": [mean(item[1] for item in segment) for segment in segments],
        "unconditioned_mean_distinct_episodes_per_batch": [mean(item[2][0] for item in segment) for segment in segments],
        "overlap_weighted_mean_consumer_wait_s": [
            sum(count * mean(strata[key]) for key, count in overlap_counts.items()) / matches
            for strata in distributions
        ] if matches else None,
        "wait_clock": WAIT_CLOCK,
    }


def audit(directory, job_path, post_path, episode_index, minimum):
    paths = [directory / f"bucket-{PAIR}-{reader}.json" for reader in ("baseline", "candidate")]
    ledger = directory / "run.jsonl"
    host_path = directory / "host.json"
    pre_path = directory / f"bucket-manifest-{PAIR}-before.json"
    pre, post, host, job = (read(path) for path in (pre_path, post_path, host_path, job_path))
    rows = [read(path) for path in paths]
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    proof = supervisor_proof(events, job, host, pre, post, rows, paths, pre_path)
    lengths = load_lengths(episode_index, rows[0]["source"], pre["manifest_sha256"], rows[0]["source"]["observed"])
    traces = [
        verify_prefix(path, row, lengths, proof[row["reader"]], interrupted=row["reader"] == "candidate")
        for path, row in zip(paths, rows)
    ]
    timelines = [item[0] for item in traces]
    horizon = min(map(len, timelines))
    series = []
    for index, (base, draft) in enumerate(zip(*timelines), 1):
        delta = base[0] - draft[0]
        series.append({
            "batch": index, "baseline_s": base[0], "candidate_s": draft[0],
            "baseline_minus_candidate_s": delta,
            "lead": "candidate" if delta > 0 else "baseline" if delta < 0 else "tie",
        })
    batch_size = rows[0]["config"]["batch_size"]
    return {
        "status": "externally_audited_saved_checkpoint",
        "scope": "Recorded shared deliveries through the candidate's last atomically saved checkpoint",
        "native_pair_complete": False, "native_source_valid": proof["native_pair"]["source_valid"],
        "source_verification": "Matching pre-run and separately captured post-run inventories. The bucket remained mutable between checks.",
        "source_root": f"hf://buckets/{pre['source']}",
        "manifest_sha256": pre["manifest_sha256"],
        "manifest_before_utc": pre["recorded_utc"], "external_manifest_after_utc": post["recorded_utc"],
        "job_id": host["job_id"], "job_finished_utc": job["finish"], "artifact_revision": job["artifact_revision"],
        "supervisor": proof,
        "workload": rows[0]["workload"], "cache": rows[0]["cache"], "tolerance_s": rows[0]["config"]["tolerance_s"],
        "reader_audits": {row["reader"]: trace[1] for row, trace in zip(rows, traces)},
        "last_paired_batch": horizon, "paired_unique_samples_per_reader": horizon * batch_size,
        "native_target_batches": proof["native_pair"]["fixed_target"],
        "observed_milestones": {str(i): series[i - 1] for i in sorted({1, 125, 1000, 13300, horizon}) if i <= horizon},
        "delivery_leads": lead_changes(series, minimum),
        "diversity_prefix_1000": matched_window(timelines, 0, 1000) if horizon >= 1000 else None,
        "diversity_after_125": matched_window(timelines, 125, horizon) if horizon > 125 else None,
        "after_125_samples_per_s": [
            (horizon - 125) * batch_size / (timeline[horizon - 1][0] - timeline[124][0])
            for timeline in timelines
        ] if horizon > 125 else None,
        "raw_evidence": [identity(path) for path in (
            ledger, host_path, job_path, pre_path, post_path, episode_index, *paths,
            *(trace_path(path) for path in paths),
        )],
    }, series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--job", type=Path, required=True, help="Synced terminal Hub job metadata.")
    parser.add_argument("--external-post", type=Path, required=True, help="Externally captured post-run inventory.")
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--min-durable-batches", type=int, default=125)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--series", type=Path)
    args = parser.parse_args()
    if args.min_durable_batches < 1:
        parser.error("Minimum durable batch count must be positive.")
    protected = {path.resolve() for path in (args.job, args.external_post, args.episode_index)}
    for output in (args.out, args.series):
        if output and (output.resolve() in protected or output.resolve().is_relative_to(args.directory.resolve())):
            parser.error("Write derived output outside the raw evidence paths.")
    if args.out and args.series and args.out.resolve() == args.series.resolve():
        parser.error("Use separate destinations for the report and series.")
    try:
        result, series = audit(args.directory, args.job, args.external_post, args.episode_index, args.min_durable_batches)
    except (OSError, EOFError, ValueError, KeyError, TypeError, IndexError) as exc:
        result, series = {"status": "unresolved", "reason": str(exc)}, []
    result_json = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(result_json)
    else:
        sys.stdout.write(result_json)
    if args.series and series:
        with args.series.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(series[0]))
            writer.writeheader()
            writer.writerows(series)
    return int(result["status"] == "unresolved")


if __name__ == "__main__":
    sys.exit(main())
