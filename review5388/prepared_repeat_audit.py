"""Audit Molmo current-frame crossover repeats and finite-pass rate projections."""

import argparse
import csv
import gzip
import json
import math
import sys
from hashlib import sha256
from pathlib import Path
from statistics import median

from benchmark import DATASETS_PIN, PINS
from checkpoint_crossover import CATALOG_FIELDS, identity, require
from checkpoint_crossover import audit as audit_checkpoint
from matched_diversity import compare as compare_diversity
from matched_diversity import one_event
from observed_crossover import compare, load_lengths, trace_path
from summarize import fixture_passes, read, run_events
from turnover import PoolTurnover

ROOT = Path(__file__).resolve().parent
READERS = ("baseline", "candidate")
PAIR = "current-frame-tolerance-1ms"
CHECKPOINTS = (1, 125, 500, 1000, 2000, 3000, 4000)
START_BATCH, TAIL_BATCHES = 125, 125


def cell_paths(directory, pair):
    return [directory / f"bucket-{pair}-{reader}.json" for reader in READERS]


def window(times, start, stop, batch_size):
    if stop <= start:
        return None
    anchors = (stop - start) * batch_size
    seconds = times[stop - 1] - (times[start - 1] if start else 0)
    require(seconds > 0 and math.isfinite(seconds), "Window duration must be positive and finite")
    return {"after_batch": start, "through_batch": stop, "anchors": anchors,
            "elapsed_s": seconds, "anchors_per_s": anchors / seconds}


def finite_days(rate, elapsed, delivered, source_total):
    return (elapsed + (source_total - delivered) / rate) / 86400


class RefillEvents:
    def __init__(self):
        self.refills = []

    def write(self, kind, **fields):
        if kind == "pool_refill_verified":
            self.refills.append(fields)


def reader_metrics(path, row, lengths, *, offset, paired_horizon, fixture, stop, all_refills=False):
    """Take counts from audited delivery events and replay native pool admissions."""
    batch_size = row["config"]["batch_size"]
    delivered_batches = row["progress"]["batches"]
    source_total = sum(lengths.values())
    trace = trace_path(path)
    opener = gzip.open if trace.suffix == ".gz" else open
    replay_events = RefillEvents()
    pool = PoolTurnover(row["reader"], replay_events, refill_limit=None if all_refills else 2)
    digest = sha256()
    times, reported_refills, signatures = [], [], {}
    first_sample = None
    with opener(trace, "rt", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            kind = event["event"]
            if kind == "batch":
                batch = event["batch"]
                if batch > delivered_batches:
                    break
                times.append(event["measurement_elapsed_s"])
                digest.update((json.dumps(event["ids"], separators=(",", ":")) + "\n").encode())
                if batch in CHECKPOINTS or batch == paired_horizon:
                    signatures[str(batch)] = digest.hexdigest()
                pool.batch(batch, set(map(tuple, event["ids"])))
            elif kind == "pool_admission":
                spans = event["spans"]
                require(event["offset"] == pool.admission_count and all(
                    0 <= start <= end < lengths.get(episode, 0) and end - start + 1 == count
                    for episode, start, end, count in spans
                ), f"Invalid native pool admission in {path.name}")
                pool.add(spans)
            elif kind == "pool_refill_verified":
                reported_refills.append({key: value for key, value in event.items() if key not in ("event", "elapsed_s")})
            elif kind == "first_sample":
                require(first_sample is None, f"Repeated first-sample event in {path.name}")
                first_sample = event["measurement_elapsed_s"]

    refills = pool.refill_batches
    # The original native benchmark stops writing verified-refill events after two. Replay
    # all recorded admissions and deliveries offline, and check every native marker.
    checked = slice(2 if all_refills and len(reported_refills) <= 2 else None)
    require(len(times) == delivered_batches and replay_events.refills[checked] == reported_refills
            and refills[checked] == row["turnover"]["refill_batches"]
            and len(refills[checked]) == row["turnover"]["refills_verified"],
            f"Replayed coverage or refills disagree for {path.name}")
    require(first_sample == row["timings"]["first_sample_s"] and 0 <= first_sample <= times[0],
            f"First-sample trace and worker summary disagree for {path.name}")
    require(fixture_passes(fixture, row["reader"], row["workload"], row["config"]["tolerance_s"]),
            f"Current-frame finite fixture check failed for {path.name}")

    across = window(times, START_BATCH, delivered_batches, batch_size)
    tail = None
    projection = None
    qualifies = (row["status"] == "ok" or stop["timed_out"]) and len(refills) >= 2
    qualifies = qualifies and refills[0] > START_BATCH and delivered_batches >= refills[1] + TAIL_BATCHES
    if qualifies:
        tail = window(times, refills[1], refills[1] + TAIL_BATCHES, batch_size)
        batch_125_consumer_s = offset + times[START_BATCH - 1]
        projection = {
            "days_at_across_refills_rate": finite_days(
                across["anchors_per_s"], batch_125_consumer_s, START_BATCH * batch_size, source_total),
            "days_at_short_post_refill_rate": finite_days(
                tail["anchors_per_s"], batch_125_consumer_s, START_BATCH * batch_size, source_total),
            "startup_through_batch_125_s": batch_125_consumer_s,
        }

    paired = window(times, START_BATCH, paired_horizon, batch_size)
    return {
        "status": row["status"], "stop_reason": row.get("stop_reason"), "supervisor_code": stop["code"],
        "supervisor_timed_out": stop["timed_out"], "delivered_batches": delivered_batches,
        "delivered_anchors": delivered_batches * batch_size, "coverage_percent": delivered_batches * batch_size / source_total * 100,
        "native_refills_replayed": refills, "paired_after_125": paired, "own_after_125": across,
        "post_refill_2_first_125": tail, "finite_pass_projection": projection,
        "ordered_anchor_sha256": signatures, "reader_first_sample_s": first_sample,
        "reader_constructor_s": row["timings"]["constructor_s"], "receipt_s": offset,
        "consumer_milestones_s": {
            str(batch): offset + times[batch - 1]
            for batch in sorted({*CHECKPOINTS, delivered_batches, paired_horizon}) if batch <= delivered_batches
        },
    }


def compare_native(directory, pair, catalog, episode_index, fixture, *, prepared=False):
    paths = cell_paths(directory, pair)
    result, series = compare(paths, TAIL_BATCHES, CHECKPOINTS, episode_index, prepared=prepared)
    if not series:
        return result, []
    manifests = [directory / f"bucket-manifest-{pair}-{suffix}.json" for suffix in ("before", "after")]
    if prepared and pair != f"{PAIR}-prepared":
        manifests.append(directory / f"bucket-manifest-tolerance-1ms-before-transfer-{pair.split('-')[-1]}.json")
    for manifest in manifests:
        observed = read(manifest)
        require(all(observed[key] == catalog[key] for key in CATALOG_FIELDS),
                f"Inventory differs from the finite-pass catalog in {manifest.name}")
    rows = [read(path) for path in paths]
    events = run_events(paths[0])
    starts = [one_event(events, "start", path.stem) for path in paths]
    stops = [one_event(events, "stop", path.stem) for path in paths]
    intended = next(event for event in events if event.get("event") == "pair-done" and event.get("pair") == pair)
    lengths = load_lengths(episode_index, rows[0]["source"], catalog["manifest_sha256"], rows[0]["source"]["observed"])
    horizon = result["horizon"]["last_shared_batch"]
    prep = result.get("publication", {}).get("baseline_consumer_prep_seconds", 0)
    metrics = {}
    for path, row, stop, offset in zip(paths, rows, stops, (prep, 0)):
        audit = result["trace_checks"][row["reader"]]
        environment = row["environment"]
        config = row["config"]
        require(
            (environment["lerobot_revision"], environment["lerobot_streaming_module_sha256"]) == PINS[row["reader"]]
            and environment["datasets_source"]["revision"] == DATASETS_PIN
            and row["workload"] == "single_frame" and config["tolerance_s"] == 0.001
            and (config["batch_size"], config["workers"], config["pool_size"], config["seed_effective"]) == (8, 0, 32, 42)
            and config["camera_offsets_s"] == [0.0] and config["action_steps"] == 1 and config["torch_num_threads"] == 1,
            f"Reader code or current-frame settings differ in {path.name}",
        )
        require(audit["full_valid_unique_prefix_batches"] == audit["observed_batches"],
                f"An excluded batch occurs in {path.name}")
        metrics[row["reader"]] = reader_metrics(
            path, row, lengths, offset=offset, paired_horizon=horizon, fixture=fixture, stop=stop)
    result.update(
        reader_metrics=metrics, reader_order=[event["reader"] for event in sorted(starts, key=events.index)],
        target_batches=intended.get("fixed_target"),
        censored_before_target=bool(intended.get("fixed_target") and horizon < intended["fixed_target"]),
    )
    if prepared and horizon > START_BATCH:
        matched = compare_diversity(*paths, horizon, prepared=True, after_batch=START_BATCH)
        result["matched_diversity_after_125"] = {key: value for key, value in matched.items()
                                                  if key not in ("strata", "publication")}
    if prepared:
        result["raw_evidence"] = [identity(path) for path in (
            directory / "host.json", directory / "run.jsonl", *manifests, *paths,
            *(trace_path(path) for path in paths), Path(result["publication"]["receipt"]),
            Path(result["publication"]["audit"]),
        )]
    return result, series


def cold_checkpoint(directory, inputs, catalog, episode_index, fixture):
    result, _ = audit_checkpoint(directory, inputs / "molmo-tol02.job.json",
                                 inputs / "molmo-tol02-external-post-manifest.json", episode_index, TAIL_BATCHES)
    require(result["manifest_sha256"] == catalog["manifest_sha256"], "Cold repeat catalog differs")
    paths = cell_paths(directory, f"{PAIR}-repeat")
    rows = [read(path) for path in paths]
    lengths = load_lengths(episode_index, rows[0]["source"], catalog["manifest_sha256"], rows[0]["source"]["observed"])
    result["reader_metrics"] = {
        row["reader"]: reader_metrics(
            path, row, lengths, offset=0, paired_horizon=result["last_paired_batch"],
            fixture=fixture, stop=result["supervisor"][row["reader"]],
        ) for path, row in zip(paths, rows)
    }
    result["censored_before_target"] = result["last_paired_batch"] < result["native_target_batches"]
    return result


def repeats_summary(repeats, curves):
    verified = [result for result in repeats.values() if result.get("reader_metrics")]
    result = {"attempts": len(repeats), "audited_pairs": len(verified)}
    if not verified:
        return result
    result["audited_hosts"] = len({item["host_name"] for item in verified})
    result["censored_pairs"] = sum(item["censored_before_target"] for item in verified)
    result["receipt_seconds"] = [item["publication"]["baseline_consumer_prep_seconds"] for item in verified]
    shared_batch = min(item["horizon"]["last_shared_batch"] for item in verified)
    result["last_batch_shared_across_audited_pairs"] = {
        "batch": shared_batch,
        "points": {label: curves[label][shared_batch - 1] for label in repeats if label in curves},
    }
    result["prepared_timing_s"] = {}
    result["same_ordered_ids_between_pairs_at_batch"] = {}
    for batch in (1000, 4000):
        available = [item for item in verified if item["horizon"]["last_shared_batch"] >= batch]
        if not available:
            continue
        result["prepared_timing_s"][str(batch)] = {"pairs": len(available)}
        for reader in READERS:
            values = [item["observed_batches"][str(batch)][f"{reader}_s"] for item in available]
            result["prepared_timing_s"][str(batch)][reader] = {
                "min": min(values), "median": median(values), "max": max(values),
            }
        if len(available) > 1:
            result["same_ordered_ids_between_pairs_at_batch"][str(batch)] = {
                reader: len({item["reader_metrics"][reader]["ordered_anchor_sha256"][str(batch)]
                             for item in available}) == 1 for reader in READERS
            }
    return result


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def write_curves(prefix, curves, trials):
    series, compact = [], []
    attempts = []
    shared_repeat_batch = min((len(rows) for pair, rows in curves.items() if pair.startswith("r")), default=0)
    for pair, result in trials.items():
        first_change = result.get("first_lead_change") or {}
        final_lead = result.get("final_sustained_lead_within_horizon") or {}
        record = {
            "pair": pair, "audit_status": result["status"], "unavailable_reason": result.get("reason"),
            "same_host_job_id": result.get("host_job_id"), "reader_order": "/".join(result.get("reader_order", [])),
            "target_batches": result.get("target_batches"), "last_shared_batch": result.get("horizon", {}).get("last_shared_batch"),
            "source_coverage_pct_per_reader": (
                100 * result["horizon"]["source_coverage_fraction_per_reader"] if pair in curves else None),
            "censored_before_target": result.get("censored_before_target"),
            "baseline_receipt_s": result.get("publication", {}).get("baseline_consumer_prep_seconds", 0) if pair in curves else None,
            "first_lead_change_batch": first_change.get("first_batch_with_new_lead", {}).get("batch"),
            "final_leader": final_lead.get("reader"), "final_lead_start_batch": final_lead.get("first_batch"),
        }
        for batch in (1, 1000, 4000):
            sample = result.get("observed_batches", {}).get(str(batch)) or {}
            for reader in READERS:
                record[f"{reader}_batch_{batch}_s"] = sample.get(f"{reader}_s")
        for reader in READERS:
            metrics = result.get("reader_metrics", {}).get(reader, {})
            final_batch = metrics.get("delivered_batches")
            record[f"{reader}_final_batch"] = final_batch
            record[f"{reader}_final_consumer_s"] = metrics.get("consumer_milestones_s", {}).get(str(final_batch))
            record[f"{reader}_stop_reason"] = metrics.get("stop_reason")
        attempts.append(record)
    write_csv(prefix.with_suffix(".pairs.csv"), attempts)
    for pair, rows in curves.items():
        chosen = {1, len(rows), *CHECKPOINTS}
        if pair.startswith("r"):
            chosen.add(shared_repeat_batch)
        for before, after in zip(rows, rows[1:]):  # noqa: RUF007 - supports Python 3.9
            if before["lead"] != after["lead"]:
                chosen.update((before["batch"], after["batch"]))
        for row in rows:
            point = {"pair": pair, "batch": row["batch"], "unique_anchors": row["unique_anchors"],
                     "baseline_s": row["baseline_s"], "candidate_s": row["candidate_s"],
                     "baseline_minus_candidate_s": row["baseline_minus_candidate_s"], "lead": row["lead"]}
            series.append(point)
            if row["batch"] in chosen:
                compact.append(point)
    write_csv(prefix.with_suffix(".curves.csv"), series)
    write_csv(prefix.with_suffix(".points.csv"), compact)
    lines = [f" {'Pair':15s} {'Batch':>7s} {'Anchors':>8s} {'Pepijn s':>10s} {'Quentin s':>10s} {'Pepijn - Quentin s':>18s}"]
    lines.extend(
        f" {row['pair']:15s} {row['batch']:7,d} {row['unique_anchors']:8,d} {row['baseline_s']:10.2f}"
        f" {row['candidate_s']:10.2f} {row['baseline_minus_candidate_s']:18.2f}"
        for row in compact
    )
    prefix.with_suffix(".points.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", required=True, type=Path, help="Verified cache from coverage_audit.py")
    parser.add_argument("--results", default=ROOT / "results", type=Path)
    parser.add_argument("--repeat-dir", type=Path)
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument("--repeat-only", action="store_true", help="Skip the older reference run directories.")
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    if args.repeat_only and not args.repeat_dir:
        parser.error("--repeat-only requires --repeat-dir")
    if not 1 <= args.repeat_count <= 99:
        parser.error("--repeat-count must be between 1 and 99")
    downloads = args.results / "downloads/review5388/results"
    fixture = ROOT / "correctness/evidence"
    catalog = read(args.results / "catalog-bucket.json")
    full_batches, remainder = divmod(catalog["total_frames"], 8)
    report = {
        "full_pass": {
            "manifest_sha256": catalog["manifest_sha256"], "anchors": catalog["total_frames"],
            "episodes": catalog["total_episodes"], "batch_size": 8, "full_batches": full_batches,
            "final_partial_batch_anchors": remainder, "delivered_batches_including_tail": full_batches + bool(remainder),
            "scope": "One finite current-frame iteration, one process, no DataLoader workers, incomplete final batch retained.",
            "projection_formula": "(consumer time to batch 125 + (source anchors - 1,000) / observed anchors per second) / 86,400",
            "projection_assumption": "The measured CPU consumer delivery rate holds through this snapshot. No model steps are timed.",
            "minimum_projection_window": "After batch 125 through two audited pool refills and at least 125 further full batches.",
            "coverage_percent_per_reader": {str(batch): 100 * (8 * batch / catalog["total_frames"]) for batch in CHECKPOINTS},
            "repeating_trainer_fixture": "Four workers delivered 79 unique anchors in the first 86 consumed samples. See correctness/README.md.",
        }
    }
    references = [
        ("cold", downloads / "molmo-tol01", PAIR, False),
        ("warm", downloads / "molmo-tol01", f"{PAIR}-warm", False),
        ("prepared 1k", downloads / "molmo-prepared01", f"{PAIR}-prepared", True),
    ]
    profiles = [] if args.repeat_only else references
    if args.repeat_dir:
        profiles.extend((f"r{i:02}", args.repeat_dir, f"{PAIR}-prepared-r{i:02}", True)
                        for i in range(1, args.repeat_count + 1))
    trials, curves = {}, {}
    for label, directory, pair, prepared in profiles:
        try:
            result, curve = compare_native(directory, pair, catalog, args.episode_index, fixture, prepared=prepared)
        except (OSError, EOFError, ValueError, KeyError, TypeError) as exc:
            result, curve = {"status": "ineligible_pair", "reason": str(exc)}, []
        trials[label] = result
        if curve:
            curves[label] = curve
    report["trials"] = trials
    report["prepared_repeats"] = repeats_summary(
        {label: result for label, result in trials.items() if label.startswith("r")}, curves)
    if not args.repeat_only:
        report["cold_checkpoint_repeat"] = cold_checkpoint(
            downloads / "molmo-tol02", args.results, catalog, args.episode_index, fixture)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    write_curves(args.output_prefix, curves, trials)
    return int(len(curves) != len(profiles))


if __name__ == "__main__":
    sys.exit(main())
