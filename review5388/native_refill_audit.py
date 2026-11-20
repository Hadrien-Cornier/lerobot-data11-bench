"""Audit the two-refill MolmoAct2 experiment on the grouped #5388 revision."""

import argparse
import csv
import gzip
import json
import math
from collections import Counter
from hashlib import sha256
from pathlib import Path

from benchmark import DATASETS_PIN, PINS, diversity
from checkpoint_crossover import CATALOG_FIELDS, lead_changes, require
from matched_diversity import one_event, verify_receipt_arguments
from observed_crossover import load_lengths, read_cell, trace_path
from prepared_repeat_audit import (
    START_BATCH,
    TAIL_BATCHES,
    finite_days,
    reader_metrics,
    window,
)
from summarize import read, run_events

ROOT = Path(__file__).resolve().parent
READERS = ("baseline", "candidate")
SETTINGS = {
    "batch_size": 8,
    "workers": 0,
    "pool_size": 32,
    "seed_effective": 42,
    "torch_num_threads": 1,
    "tolerance_s": 0.001,
    "action_steps": 1,
    "camera_offsets_s": [0.0],
    "warm_batches": 0,
    "collate": "default_collate",
}


def source_and_receipt(directory, historical, catalog):
    manifests = sorted(directory.glob("bucket-manifest-*.json"))
    expected = {"bucket-manifest-before-transfer"} | {
        f"bucket-manifest-{phase}-{reader}"
        for reader in READERS
        for phase in ("before", "after")
    }
    require(expected <= {p.stem for p in manifests}, "A bucket inventory is missing.")
    for path in manifests:
        current = read(path)
        require(
            all(current[key] == catalog[key] for key in CATALOG_FIELDS),
            f"Different inventory in {path.name}",
        )
    receipt = read(directory / "bucket-private-transfer.json")["private_transfer"]
    published = historical["publication"]
    require(
        receipt["sha256"] == published["baseline_archive_sha256"]
        and receipt["size_bytes"] == published["baseline_archive_bytes"]
        and receipt["revision"] == published["private_archive_revision"]
        and 0
        < receipt["network_transfer_and_link_seconds"]
        <= receipt["consumer_prep_seconds"]
        and math.isfinite(receipt["consumer_prep_seconds"]),
        "The transfer did not identify the matching publisher archive and its acquisition time.",
    )
    return manifests, receipt


def supervisor(directory, paths, catalog, grouped, receipt):
    host = read(directory / "host.json")
    require(
        host["profile"] == "native-refills"
        and host["hostname"]
        and host["source_bucket_sha256"] == catalog["manifest_sha256"],
        "Host or source metadata does not identify the paired run.",
    )
    events = run_events(paths["baseline"])
    (setup,) = (e for e in events if e["event"] == "refills_setup")
    order = setup["order"]
    require(
        setup["patched"] == grouped
        and sorted(order) == sorted(READERS)
        and setup["prepared"]["sha256"] == receipt["sha256"]
        and host["run"] == directory.name
        and host["job_id"],
        "Setup does not identify the expected patch, run, two readers and matching index.",
    )
    receipt_label = "bucket-private-transfer"
    labels = ["bucket-manifest-before-transfer"]
    for reader in order:
        labels.append(f"bucket-manifest-before-{reader}")
        if reader == "baseline":
            labels.append(receipt_label)
        labels.extend([paths[reader].stem, f"bucket-manifest-after-{reader}"])
    cache_reads = {}
    previous = events.index(setup)
    for label in labels:
        start, stop = (one_event(events, phase, label) for phase in ("start", "stop"))
        require(
            previous < events.index(start) < events.index(stop)
            and stop["code"] == 0
            and not stop["timed_out"],
            f"Failed or out-of-sequence stage {label}.",
        )
        previous = events.index(stop)
        cache_reads[label] = start["cache"]
        if label == receipt_label:
            verify_receipt_arguments(start, receipt, "bucket", f"{receipt_label}.json")
        if label in (p.stem for p in paths.values()):
            require(
                start["argv"][:2] == ["-m", "runner.refill_arm"]
                and start["datasets_source"] == "pull-8983",
                f"Unexpected reader entry point in {label}.",
            )
    for reader, expected_labels in (
        ("baseline", [receipt_label, paths["baseline"].stem]),
        ("candidate", [paths["candidate"].stem]),
    ):
        used_cache = cache_reads[paths[reader].stem]
        cache_starts = [
            e["label"]
            for e in events
            if e.get("event") == "start" and e.get("cache") == used_cache
        ]
        require(
            cache_starts == expected_labels,
            f"Additional uses of the {reader} cache: {cache_starts}",
        )
    metadata = {
        key: host[key]
        for key in (
            "cpu_model",
            "cpu_quota",
            "memory_limit_bytes",
            "job_id",
            "bundle_revision",
        )
    }
    quota, period = (int(value) for value in host["cpu_quota"].split())
    return {**metadata, "vcpus": quota / period, "reader_order": order}


def read_arm(path, row, reader, catalog, episode_index, grouped):
    env, config, source = row["environment"], row["config"], row["source"]
    pin = (
        (grouped["revision"], grouped["module_sha256"]["streaming_dataset.py"])
        if reader == "candidate"
        else PINS["baseline"]
    )
    require(
        (env["lerobot_revision"], env["lerobot_streaming_module_sha256"]) == pin
        and env["datasets_source"]["revision"] == DATASETS_PIN
        and (row["reader"], row["dataset"], row["workload"])
        == (reader, "bucket", "single_frame")
        and all(config[k] == v for k, v in SETTINGS.items())
        and source["repo"] == catalog["source"]
        and source["revision"] == catalog["revision"],
        f"Source, reader code or workload settings differ in {path.name}",
    )
    observed = source["observed"]
    require(
        observed["frames"] == catalog["total_frames"]
        and observed["episodes"] == catalog["total_episodes"]
        and observed["fps"] == catalog["fps"]
        and len(observed["video_keys"]) == 3,
        "Observed source does not match the pinned three-camera catalog.",
    )
    lengths = load_lengths(episode_index, source, catalog["manifest_sha256"], observed)
    elapsed, audit = read_cell(path, row, lengths)
    require(not audit["snapshot_errors"], f"{path.name}: {audit['snapshot_errors']}")
    require(
        not audit["first_excluded_batch"]
        and len(elapsed) == audit["observed_batches"]
        and elapsed,
        f"An invalid, missing, empty or duplicate batch occurred in {path.name}.",
    )
    return elapsed, lengths


def episode_mix(path, horizon, fps):
    """Measure the episode mix from the emitted IDs for the same number of batches per reader."""
    trace = trace_path(path)
    opener = gzip.open if trace.suffix == ".gz" else open
    unique_episodes = set()
    distinct_counts = Counter()
    digest = sha256()
    sum_largest_share = 0.0
    batches_with_near_pair = 0
    with opener(trace, "rt") as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] != "batch":
                continue
            if event["batch"] > horizon:
                break
            digest.update(
                (json.dumps(event["ids"], separators=(",", ":")) + "\n").encode()
            )
            counts = diversity(event["ids"], fps)
            distinct, near_pairs, largest = counts[0], counts[2], counts[4]
            require(
                (distinct, near_pairs, largest)
                == (
                    event["distinct_episodes"],
                    event["near_0p2s_pairs"],
                    event["largest_episode_share"],
                ),
                "Episode mix disagrees with trace.",
            )
            unique_episodes.update(ep for ep, _frame in event["ids"])
            distinct_counts[distinct] += 1
            sum_largest_share += largest
            batches_with_near_pair += near_pairs > 0
    require(sum(distinct_counts.values()) == horizon, "Incomplete episode mix counts.")
    return {
        "batches": horizon,
        "unique_episodes": len(unique_episodes),
        "ordered_anchor_sha256": digest.hexdigest(),
        "mean_distinct_episodes_per_batch": sum(
            n * count for n, count in distinct_counts.items()
        )
        / horizon,
        "mean_largest_episode_share": sum_largest_share / horizon,
        "batches_by_distinct_episode_count": dict(sorted(distinct_counts.items())),
        "batches_with_a_same_episode_pair_within_0p2s": batches_with_near_pair,
    }


def delivery_curve(baseline, candidate, receipt_seconds):
    series = []
    for batch, (base_s, candidate_s) in enumerate(zip(baseline, candidate), 1):
        baseline_s = base_s + receipt_seconds
        delta = baseline_s - candidate_s
        series.append(
            {
                "batch": batch,
                "baseline_s": baseline_s,
                "candidate_s": candidate_s,
                "baseline_minus_candidate_s": delta,
                "lead": "baseline"
                if delta < 0
                else "candidate"
                if delta > 0
                else "tie",
            }
        )
    return series


def crossover_sensitivity(series, stop):
    """Extrapolate from the measured gap at this batch with several trailing slopes."""
    point = series[stop - 1]
    gap = point["baseline_minus_candidate_s"]
    starting_leader = point["lead"]
    reverse = next(
        (
            later["batch"]
            for later in series[stop:]
            if later["lead"] not in (starting_leader, "tie")
        ),
        None,
    )
    estimates = []
    for start in sorted(
        {START_BATCH, stop - 10000, stop - 5000, stop - 1000, stop - 125}
    ):
        if not 0 < start < stop:
            continue
        start_gap = series[start - 1]["baseline_minus_candidate_s"]
        slope = (gap - start_gap) / (stop - start)
        crossing = None
        if gap * slope < 0:
            crossing = stop + math.floor(abs(gap / slope)) + 1
        estimates.append(
            {
                "from_batch": start,
                "through_batch": stop,
                "baseline_minus_candidate_seconds_per_batch": slope,
                "extrapolated_first_strict_lead_reversal_batch": crossing,
            }
        )
    return {
        "observed_leader": starting_leader,
        "baseline_minus_candidate_s": gap,
        "observed_first_opposite_leader_after_checkpoint": reverse,
        "observed_through_batch": len(series),
        "rate_windows": estimates,
    }


def epoch_windows(times, refills, source_total, offset):
    # Hold observed delivery through the last batch fixed in every scenario.
    # Vary only the rate for the remaining examples.
    last = len(times)
    if (
        len(refills) < 2
        or refills[0] <= START_BATCH
        or last < refills[1] + TAIL_BATCHES
    ):
        return {}
    first, second = refills[:2]
    intervals = {
        "through_both_refills": (START_BATCH, last),
        "before_first_refill": (START_BATCH, first),
        "between_refills": (first, second),
        "all_after_second_refill": (second, last),
    }
    for i in range(2, len(refills)):
        if refills[i] > refills[i - 1]:
            intervals[f"refill_{i}_to_{i + 1}"] = (refills[i - 1], refills[i])
    projections = {}
    for name, (start, stop) in intervals.items():
        rate = window(times, start, stop, SETTINGS["batch_size"])
        projections[name] = rate | {
            "cpu_epoch_days": finite_days(
                rate["anchors_per_s"],
                times[-1] + offset,
                last * SETTINGS["batch_size"],
                source_total,
            )
        }
    return projections


def audit_pair(directory, episode_index):
    prior = read(ROOT / "measurements/molmo-prepared-repeats.json")
    indexed = prior["trials"]["r03"]
    stock_reference = prior["cold_checkpoint_repeat"]
    catalog_path = ROOT / "catalog/catalog-bucket.json"
    catalog = read(
        catalog_path if catalog_path.exists() else ROOT / "results/catalog-bucket.json"
    )
    manifests, receipt = source_and_receipt(directory, indexed, catalog)
    paths = {r: directory / f"bucket-native-refills-{r}.json" for r in READERS}
    rows = {r: read(p) for r, p in paths.items()}
    grouped = read(ROOT / "runner/native_molmo_pins.json")["grouped"]
    host = supervisor(directory, paths, catalog, grouped, receipt)
    expected = read(ROOT / "measurements/native-grouping-fixture.json")
    require(
        expected["revisions"]["grouped"]["commit"] == grouped["revision"]
        and "grouped" in expected["revisions_match_stock_in_all_cases"],
        "Missing grouped-reader parity evidence.",
    )
    require(
        rows["baseline"]["source"]["observed"]["baseline_sidecar_bytes"]
        == receipt["size_bytes"]
        and rows["baseline"]["io"]["sidecar"] == {},
        "Baseline did not reuse the matching transferred index.",
    )
    receipts = {"baseline": receipt["consumer_prep_seconds"], "candidate": 0.0}
    traces = {
        r: read_arm(paths[r], rows[r], r, catalog, episode_index, grouped)
        for r in READERS
    }
    series = delivery_curve(
        traces["baseline"][0], traces["candidate"][0], receipts["baseline"]
    )
    end = len(series)
    result = {}
    for reader in READERS:
        times, lengths = traces[reader]
        row = rows[reader]
        metrics = reader_metrics(
            paths[reader],
            row,
            lengths,
            offset=receipts[reader],
            # Compare anchor order over the longest shared prefix of the old and new traces.
            paired_horizon=min(end, stock_reference["last_paired_batch"]),
            fixture=ROOT / "correctness/evidence",
            stop={"code": None, "timed_out": False},
            all_refills=True,
        )
        refills = metrics["native_refills_replayed"]
        expected_hashes = stock_reference["reader_metrics"][reader][
            "ordered_anchor_sha256"
        ]
        result[reader] = {
            "lerobot_revision": row["environment"]["lerobot_revision"],
            "batches": len(times),
            "first_batch_s": times[0] + receipts[reader],
            "last_batch_s": times[-1] + receipts[reader],
            "status": row["status"],
            "stop_reason": row.get("stop_reason"),
            "refill_batches": refills,
            "native_refill_events": row["turnover"]["refill_batches"],
            "refill_elapsed_s": [times[n - 1] + receipts[reader] for n in refills],
            "distinct_usable_examples": row["progress"]["usable_unique_samples"],
            "shared_prefix_episode_diversity": episode_mix(
                paths[reader], end, catalog["fps"]
            ),
            "invalid_batches": row["contract"]["invalid_batches"],
            "duplicates": row["progress"]["duplicates"],
            "after_batch_125": {
                key: row["windows"]["after_batch_125"][key]
                for key in ("batches", "batch_wait_s", "episode_diversity")
            }
            if "after_batch_125" in row["windows"]
            else None,
            "first_1000_match_saved_id_order": (
                metrics["ordered_anchor_sha256"].get("1000")
                == indexed["reader_metrics"][reader]["ordered_anchor_sha256"]["1000"]
                if len(times) >= 1000
                else None
            ),
            "order_check_against_long_stock_run": {
                batch: {
                    "sha256": digest,
                    "matches_stock": expected_hashes[batch] == digest,
                }
                for batch, digest in metrics["ordered_anchor_sha256"].items()
                if batch in expected_hashes
            },
            "window_sensitivity": epoch_windows(
                times, refills, catalog["total_frames"], receipts[reader]
            )
            if row["status"] == "ok"
            else {},
        }
    transitions = lead_changes(series, TAIL_BATCHES)
    milestones = {1, 125, 1000, 4000, 10000, 20000, 30000, 40000, 50000, 60000, end}
    milestones.update(n for arm in result.values() for n in arm["refill_batches"])
    return {
        "source": catalog["source"],
        "source_manifest_sha256": catalog["manifest_sha256"],
        "matching_inventories": [p.name for p in manifests],
        "run": directory.name,
        "host": host,
        "settings": SETTINGS,
        "matching_index": {
            "mode": "simulated_publication",
            "bytes": receipt["size_bytes"],
            "sha256": receipt["sha256"],
            "consumer_receipt_s": receipts["baseline"],
        },
        "source_examples": catalog["total_frames"],
        "batches_per_epoch": math.ceil(catalog["total_frames"] / 8),
        "final_batch_size": catalog["total_frames"] % 8 or 8,
        "last_shared_batch": end,
        "shared_fraction_of_epoch": end * 8 / catalog["total_frames"],
        "readers": result,
        "delivery_leads": transitions,
        "crossover_rate_window_sensitivity": {
            str(stop): crossover_sensitivity(series, stop)
            for stop in sorted({min(30000, end), end})
        },
        "milestones": [series[n - 1] for n in sorted(milestones) if n <= end],
        "epoch_method": "Observed time to the final delivery plus remaining examples / measured examples per second. "
        "Primary rate spans batches after 125 through the final delivery. Other windows test sensitivity. "
        "Two verified cohort refills and 125 further batches are required.",
    }, series


def plot(series, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    fig, ax = plt.subplots(figsize=(8.4, 4.4), layout="constrained")
    x = [point["batch"] for point in series]
    y = [point["baseline_minus_candidate_s"] for point in series]
    ax.axhline(0, color="#333333", lw=1)
    ax.plot(x, y, color="#31659a", lw=1.6)
    ax.set(
        xlim=(1, max(2, x[-1])),
        xlabel="Full batches of eight delivered",
        ylabel="Time for #3917 minus time for patched #5388 (s)",
    )
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", color="#dedede", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Patched #5388 leads above zero. #3917 leads below zero.", fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--curve",
        type=Path,
        help="CSV or gzip CSV with every shared batch and both clocks.",
    )
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    result, series = audit_pair(args.directory, args.episode_index)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if args.curve:
        opener = gzip.open if args.curve.suffix == ".gz" else open
        with opener(args.curve, "wt", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(series[0]))
            writer.writeheader()
            writer.writerows(series)
    if args.plot:
        plot(series, args.plot)


if __name__ == "__main__":
    main()
