"""Compare elapsed time to equal batches in complete, verified reader traces."""

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

from matched_diversity import compare as compare_diversity
from matched_diversity import prepared_receipt
from resource_summary import source_metadata
from summarize import read, run_events, source_verified

CONTRACT_SETTINGS = {
    "tolerance_s": 0.0001,
    "batch_size": None,
    "workers": None,
    "pool_size": None,
    "torch_num_threads": None,
    "seed_effective": None,
    "warm_batches": 0,
    "camera_offsets_s": None,
    "action_steps": None,
    "collate": None,
}
PROGRESS_FIELDS = ("batches", "raw_samples", "unique_samples", "usable_unique_samples", "duplicates")
TERMINAL = {"ok", "error", "empty", "invalid_output", "interrupted"}
MILESTONES = (1, 125, 1000)


def trace_path(path):
    compressed = path.with_suffix(".events.jsonl.gz")
    return compressed if compressed.is_file() else path.with_suffix(".events.jsonl")


def read_cell(path, record, lengths):
    """Audit the complete snapshot. Stop the comparison prefix at its first invalid batch."""
    trace = trace_path(path)
    opener = gzip.open if trace.suffix == ".gz" else open
    times, errors = [], []
    seen, usable = set(), set()
    count = raw = 0
    last_s = 0.0
    first_invalid = None
    first_sample = record.get("timings", {}).get("first_sample_s")
    terminal = None
    batch_size = record["config"]["batch_size"]
    episode_count = record.get("source", {}).get("observed", {}).get("episodes")
    try:
        with opener(trace, "rt", encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                kind = event["event"]
                if kind == "terminal":
                    if terminal is not None:
                        raise ValueError("More than one terminal event")
                    terminal = event
                    continue
                if kind == "first_sample" and first_sample is None:
                    first_sample = event["measurement_elapsed_s"]
                if kind != "batch":
                    continue
                if terminal is not None:
                    raise ValueError("Delivery recorded after terminal event")
                count += 1
                elapsed = event["measurement_elapsed_s"]
                if event["batch"] != count or not math.isfinite(elapsed) or elapsed < last_s:
                    raise ValueError(f"Missing, out-of-order, or non-monotonic batch {count}")
                last_s = elapsed
                ids = event["ids"]
                raw += len(ids)
                if any(
                    len(anchor) != 2 or any(type(value) is not int for value in anchor)
                    for anchor in ids
                ):
                    raise ValueError(f"Invalid anchor encoding at batch {count}")
                anchors = set(map(tuple, ids))
                new = anchors - seen
                contract_errors = event["contract_errors"]
                new_usable = anchors - usable if not contract_errors else set()
                if (event["unique"], event["usable_unique"]) != (len(new), len(new_usable)):
                    raise ValueError(f"Recorded coverage disagrees with IDs at batch {count}")
                seen.update(new)
                usable.update(new_usable)
                if first_invalid is None:
                    invalid_ids = any(
                        ep < 0 or frame < 0
                        or (episode_count is not None and ep >= episode_count)
                        or (lengths is not None and frame >= lengths.get(ep, 0))
                        for ep, frame in anchors
                    )
                    failures = []
                    if len(ids) != batch_size:
                        failures.append("partial_batch")
                    if len(new) != len(ids):
                        failures.append("duplicate_anchor")
                    if contract_errors:
                        failures.append("contract_violation")
                    if invalid_ids:
                        failures.append("anchor_out_of_bounds")
                    if failures:
                        first_invalid = {"batch": count, "reasons": failures, "contract_errors": contract_errors}
                    else:
                        times.append(elapsed)
    except (OSError, EOFError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"{trace.name}: {exc}")

    expected = dict(zip(PROGRESS_FIELDS, (count, raw, len(seen), len(usable), raw - len(seen))))
    if terminal is None:
        errors.append("Terminal trace event is missing")
    else:
        if any(terminal.get(key) != value for key, value in expected.items()):
            errors.append("Trace counts disagree with its terminal event")
        if terminal.get("status") != record.get("status") or record.get("status") not in TERMINAL:
            errors.append("Terminal trace status disagrees with the worker summary")
    if any(record.get("progress", {}).get(key) != value for key, value in expected.items()):
        errors.append("Trace counts disagree with the worker summary")
    for batch, milestone in record.get("timings", {}).get("milestones", {}).items():
        index = int(batch) - 1
        if 0 <= index < len(times) and not math.isclose(times[index], milestone["elapsed_s"], rel_tol=1e-9):
            errors.append(f"Elapsed time for batch {batch} disagrees with the worker summary")
    return times, {
        "trace": trace.name,
        "snapshot_errors": errors,
        "status": record.get("status"),
        "error": (record.get("error") or "").split("\n", 1)[0] or None,
        "first_sample_s": first_sample,
        "observed_batches": count,
        "full_valid_unique_prefix_batches": len(times),
        "first_excluded_batch": first_invalid,
        "duplicate_anchors_in_trace": raw - len(seen),
    }


def same_pair(paths, rows, events, hosts, *, prepared=False):
    """Check source identity and matched comparison settings before emitting a paired curve."""
    a, b = rows
    if (a["reader"], b["reader"]) != ("baseline", "candidate"):
        return "Pass the baseline result first and the candidate result second."
    for key in ("dataset", "workload") if prepared else ("dataset", "workload", "cache"):
        if a[key] != b[key]:
            return f"Reader results differ in {key}."
    if not prepared and a["cache"] not in ("cold", "warm"):
        return "Both cache states must be classified and match."
    for key, default in CONTRACT_SETTINGS.items():
        if a["config"].get(key, default) != b["config"].get(key, default):
            return f"Reader settings differ for {key}."
    for key in ("job_id", "hostname"):
        if not hosts[0].get(key) or hosts[0].get(key) != hosts[1].get(key):
            return f"Same host not verified ({key})."
    if a["source"].get("repo") != b["source"].get("repo"):
        return "Source IDs differ."
    observed = [row["source"].get("observed") for row in rows]
    if all(observed) and any(observed[0].get(key) != observed[1].get(key) for key in ("fps", "frames", "episodes")):
        return "Readers report different source frame or episode counts."
    sources = [source_metadata(path, row, log) for path, row, log in zip(paths, rows, events)]
    if sources[0][0] != sources[1][0]:
        return "Source roots differ."
    if a["dataset"] == "bucket":
        fingerprints = {digest for source in sources for digest in source[2:]}
        if len(fingerprints) != 1 or None in fingerprints:
            return "Source bucket pre/post manifests are missing or disagree."
        if any(source[1] != "bucket_manifest_match" for source in sources):
            return "Source bucket changed or the check did not pass."
        if any(host.get("source_bucket_sha256") not in (None, sources[0][2]) for host in hosts):
            return "Source bucket manifest differs from the job's expected manifest."
        if not all(source_verified(path, row, log) is True for path, row, log in zip(paths, rows, events)):
            return "Source bucket checks are not finalized in the run ledger."
    elif any(source[1] != "revision_verified" for source in sources):
        return "Both dataset revisions must resolve to the same verified commit."
    return None


def load_lengths(path, source, manifest, observed):
    if path is None:
        return None
    index = read(path)
    expected = [source["repo"], source["revision"], manifest, observed["episodes"], observed["frames"]]
    catalog = index["catalog_key"]
    if manifest is None:
        expected[2] = catalog[2]  # repository revision is immutable
    if catalog != expected:
        raise ValueError("The episode index does not match the verified source.")
    lengths = {int(episode): count for episode, count in index["lengths"].items()}
    if len(lengths) != observed["episodes"] or sum(lengths.values()) != observed["frames"]:
        raise ValueError("Episode counts in the supplied index are inconsistent.")
    return lengths


def compare(paths, minimum, query_batches, episode_index=None, diversity_prefix=1000, *, prepared=False):
    result = {"cells": [path.stem for path in paths]}
    try:
        rows = [read(path) for path in paths]
        events = [run_events(path) for path in paths]
        hosts = [read(path.parent / "host.json") for path in paths]
    except (OSError, json.JSONDecodeError) as exc:
        return result | {"status": "incomplete_snapshot", "reason": str(exc)}, []
    required_fields = {"reader", "dataset", "workload", "cache", "source", "config", "progress", "status"}
    if any(not isinstance(row, dict) or not required_fields <= row.keys() for row in rows):
        return result | {"status": "incomplete_snapshot", "reason": "Required worker metadata is missing."}, []
    result.update(
        dataset=rows[0].get("dataset"), workload=rows[0].get("workload"),
        cache=rows[0].get("cache"), host_job_id=hosts[0].get("job_id"), host_name=hosts[0].get("hostname"),
        tolerance_s=rows[0]["config"].get("tolerance_s", 0.0001),
        batch_size=rows[0]["config"].get("batch_size"),
    )
    reason = same_pair(paths, rows, events, hosts, prepared=prepared)
    if reason:
        return result | {"status": "ineligible_pair", "reason": reason}, []
    prep = 0.0
    if prepared:
        try:
            publication = prepared_receipt(paths, rows, events[0])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            return result | {"status": "ineligible_pair", "reason": str(exc)}, []
        prep = publication["baseline_consumer_prep_seconds"]
        result.pop("cache")
        result.update(publication=publication, cache_states={row["reader"]: row["cache"] for row in rows})
    source, _, manifest, _ = source_metadata(paths[0], rows[0], events[0])
    observed = rows[0]["source"].get("observed")
    if episode_index and not observed:
        return result | {"status": "invalid_episode_index", "reason": "Source episode counts are missing."}, []
    try:
        lengths = load_lengths(episode_index, rows[0]["source"], manifest, observed)
    except (OSError, ValueError, KeyError) as exc:
        return result | {"status": "invalid_episode_index", "reason": str(exc)}, []
    audits = [read_cell(path, row, lengths) for path, row in zip(paths, rows)]
    times = [audit[0] for audit in audits]
    result.update(
        source_root=source, source_bucket_manifest_sha256=manifest,
        trace_checks={row["reader"]: audit[1] for row, audit in zip(rows, audits)},
        anchor_validation="source_episode_index" if lengths is not None else "integer_ids_and_episode_count",
        same_sample_ids_required_across_readers=False,
    )
    if any(audit[1]["snapshot_errors"] for audit in audits):
        return result | {"status": "incomplete_snapshot"}, []
    if prepared:
        breakdown = {}
        for row, (reader_times, audit), offset in zip(rows, audits, (prep, 0.0)):
            first_sample = audit["first_sample_s"]
            audit["reader_first_sample_s"] = first_sample
            audit["first_sample_s"] = first_sample + offset if first_sample is not None else None
            breakdown[row["reader"]] = {
                "consumer_prep_seconds": offset,
                "reader_constructor_s": row.get("timings", {}).get("constructor_s"),
                "reader_first_sample_s": first_sample,
                "consumer_first_sample_s": audit["first_sample_s"],
                "milestones": {
                    str(batch): {
                        "reader_elapsed_s": reader_times[batch - 1],
                        "consumer_elapsed_s": reader_times[batch - 1] + offset,
                    }
                    for batch in sorted(set(MILESTONES) | set(query_batches))
                    if batch <= len(reader_times)
                },
            }
        result["timing_breakdown"] = breakdown
    horizon = min(map(len, times))
    result["horizon"] = {
        "last_shared_batch": horizon,
        "unique_anchors_per_reader": horizon * result["batch_size"],
        "source_coverage_fraction_per_reader": horizon * result["batch_size"] / observed["frames"]
        if observed and observed.get("frames") else None,
    }
    if not horizon:
        return result | {"status": "no_shared_valid_prefix"}, []

    series = []
    changes = []
    previous = None
    initial = None
    strict_run_start = None
    for batch, (baseline_reader, candidate) in enumerate(zip(*times), start=1):
        baseline = baseline_reader + prep
        delta = baseline - candidate
        leader = "baseline" if delta < 0 else "candidate" if delta > 0 else "tie"
        item = {"batch": batch, "unique_anchors": batch * result["batch_size"],
                "baseline_s": baseline, "candidate_s": candidate, "baseline_minus_candidate_s": delta, "lead": leader}
        if prepared:
            item["baseline_reader_s"] = baseline_reader
            item["baseline_consumer_prep_s"] = prep
        series.append(item)
        if leader == "tie":
            continue
        if previous is None:
            initial = item
            strict_run_start = item
        elif leader != previous["lead"]:
            changes.append({"last_batch_with_old_lead": previous, "first_batch_with_new_lead": item})
            strict_run_start = item
        previous = item

    final = None
    if previous is not None:
        start = strict_run_start["batch"]
        strictly_ahead = sum(item["lead"] == previous["lead"] for item in series[start - 1:])
        final = {
            "reader": previous["lead"], "first_batch": start, "through_batch": horizon,
            "strictly_ahead_batches": strictly_ahead, "minimum_requested_batches": minimum,
            "meets_minimum": strictly_ahead >= minimum,
        }

    diversity = {"status": "insufficient_common_valid_batches", "requested_prefix_batches": diversity_prefix}
    if horizon >= diversity_prefix:
        try:
            comparison = compare_diversity(*paths, diversity_prefix, prepared=prepared)
            diversity = {key: value for key, value in comparison.items() if key != "strata"}
        except (OSError, ValueError, KeyError) as exc:
            diversity = {"status": "unavailable", "reason": str(exc)}
    queries = sorted(set(MILESTONES) | set(query_batches) | {horizon})
    return result | {
        "status": "observed_crossover" if changes else "no_observed_crossover",
        "observed_elapsed_ranges_s": {
            "baseline": [series[0]["baseline_s"], series[-1]["baseline_s"]],
            "candidate": [series[0]["candidate_s"], series[-1]["candidate_s"]],
        },
        "first_strict_lead": initial,
        "first_lead_change": changes[0] if changes else None,
        "lead_change_count": len(changes),
        "lead_changes": changes,
        "final_sustained_lead_within_horizon": final,
        "observed_batches": {str(batch): series[batch - 1] if batch <= horizon else None for batch in queries},
        "matched_diversity_consumer_wait": diversity,
        "crossover_conditions": "Observed cumulative delivery time without conditioning on sampling diversity.",
        "tie_handling": "An exact tie cannot establish or reverse a lead. It does not count as strictly ahead.",
    }, series


def plot(path, result, series):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    fig, ax = plt.subplots(figsize=(8.6, 4.8), layout="constrained")
    fig.patch.set_facecolor("white")
    batch = [row["batch"] for row in series]
    baseline_label = "Current reader"
    if "publication" in result:
        baseline_label += f" (+{result['publication']['baseline_consumer_prep_seconds']:.2f} s receipt)"
    ax.plot(batch, [row["baseline_s"] for row in series], label=baseline_label, color="#245786", lw=2.2)
    ax.plot(batch, [row["candidate_s"] for row in series], label="Draft reader", color="#b7501a", lw=2.2)
    for milestone in MILESTONES:
        if milestone <= batch[-1]:
            ax.axvline(milestone, ls=":", lw=1, color="#9ca3af", zorder=0)
            for column, color in (("baseline_s", "#245786"), ("candidate_s", "#b7501a")):
                ax.scatter(milestone, series[milestone - 1][column], color=color, s=17, zorder=4)
    switch = result["first_lead_change"]
    if switch:
        previous = switch["last_batch_with_old_lead"]["batch"]
        change = switch["first_batch_with_new_lead"]["batch"]
        ax.axvspan(previous, change, color="#8594a5", alpha=0.22, label=f"First lead change ({previous:,}–{change:,})")
    ax.set_xscale("log" if batch[-1] > 1000 else "linear")
    if batch[-1] > 1000:
        ax.set_xticks(sorted(set(MILESTONES) | {batch[-1]}))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("Full valid batches delivered")
    ax.set_ylabel("Elapsed time (s)")
    ax.grid(axis="y", color="#e0e4ec", lw=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", frameon=False)
    title = {"single_frame": "Current-frame streaming", "action100": "100-step action streaming"}.get(
        result["workload"], result["workload"])
    dataset = "MolmoAct2 Bimanual YAM" if result["dataset"] == "bucket" else result["dataset"].upper()
    fig.suptitle(f"{dataset} · {title}", fontsize=15, fontweight="semibold")
    cache_label = "simulated index publication" if result.get("publication") else f"{result['cache']} cache"
    ax.set_title(
        f"{result['tolerance_s']*1000:g} ms tolerance · {cache_label} · "
        f"{result['batch_size']} examples per batch · {batch[-1]:,} paired batches",
        fontsize=10,
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)


def positive_int(text):
    number = int(text)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--episode-index", type=Path, help="Verified cache JSON produced by coverage_audit.py.")
    parser.add_argument("--batch", type=positive_int, action="append", default=[], help="Report this batch in JSON.")
    parser.add_argument("--min-sustained-batches", type=positive_int, default=125)
    parser.add_argument("--diversity-prefix", type=positive_int, default=1000)
    parser.add_argument("--prepared", action="store_true", help="Verify and charge the simulated-publication receipt.")
    parser.add_argument("--out", type=Path, help="Write JSON to this file instead of standard output.")
    parser.add_argument("--series", type=Path, help="Write all paired batch measurements as CSV.")
    parser.add_argument("--plot", type=Path, help="Write an SVG or PNG curve if a paired prefix exists.")
    args = parser.parse_args()
    result, series = compare(
        [args.baseline, args.candidate], args.min_sustained_batches, args.batch,
        args.episode_index, args.diversity_prefix, prepared=args.prepared,
    )
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(output)
    else:
        print(output, end="")
    if args.series and series:
        with args.series.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(series[0]))
            writer.writeheader()
            writer.writerows(series)
    if args.plot and series:
        plot(args.plot, result, series)


if __name__ == "__main__":
    main()
