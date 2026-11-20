"""Compare exposed DataLoader waits at overlapping episode mixes in validated native refill runs."""
import argparse
import json
from pathlib import Path
from statistics import fmean

from matched_diversity import WAIT_CLOCK, read_prefix
from native_refill_audit import READERS, audit_pair


def interval(directory, after, end, size, deliveries):
    count = end - after
    distributions = {}
    for reader in READERS:
        path = directory / f"bucket-native-refills-{reader}.json"
        groups, issues, observed = read_prefix(path, end, size, after)
        if issues or observed != end or sum(map(len, groups.values())) != count:
            raise ValueError(f"Incomplete or invalid batches for {reader} through {end}.")
        distributions[reader] = groups
    common = distributions["baseline"].keys() & distributions["candidate"].keys()
    weights = {key: min(len(groups[key]) for groups in distributions.values()) for key in common}
    overlap = sum(weights.values())
    if not overlap:
        raise ValueError("The two readers have no overlapping episode mix.")
    results = {}
    for reader, groups in distributions.items():
        # Each shared class receives exactly the same weight for both readers.
        weighted_mean = sum(weights[key] * fmean(groups[key]) for key in sorted(common)) / overlap
        wait_ms = 1000 * fmean(wait for waits in groups.values() for wait in waits)
        interval_s = deliveries[end - 1][reader + "_s"] - deliveries[after - 1][reader + "_s"]
        cycle_ms = 1000 * interval_s / count
        results[reader] = {
            "mean_distinct_episodes": sum(key[0] * len(waits) for key, waits in groups.items()) / count,
            "mean_wait_ms": wait_ms,
            "mean_wall_ms_per_delivery": cycle_ms,
            "mean_ms_between_next_calls": cycle_ms - wait_ms,
            "episode_mix_matched_mean_wait_ms": 1000 * weighted_mean,
        }
    return {
        "from_batch": after + 1, "through_batch": end,
        "overlap_batches_per_reader": overlap, "batches_per_reader": count,
        "overlap_fraction": overlap / count,
        "matched_mean_distinct_episodes": sum(key[0] * w for key, w in weights.items()) / overlap,
        "reader_results": results,
    }


def compare(directory, index):
    audit, deliveries = audit_pair(directory, index)
    horizon = audit["last_shared_batch"]
    periods = [(125, horizon)]
    if horizon > 30000:
        periods.extend([(125, 30000), (30000, horizon)])
    return {
        "run": directory.name,
        "vcpus": audit["host"]["vcpus"],
        "valid_shared_batches": horizon,
        "full_prefix_mean_distinct_episodes": {
            reader: audit["readers"][reader]["shared_prefix_episode_diversity"]["mean_distinct_episodes_per_batch"]
            for reader in READERS
        },
        "intervals": [interval(directory, start, end, audit["settings"]["batch_size"], deliveries) for start, end in periods],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="+")
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = {
        "clock": WAIT_CLOCK,
        "method": "Match on distinct episodes, largest same-episode sample count, and number of same-episode pairs within 0.2 s. "
                  "Weight each class by the smaller observed count across the two readers. Overlap is the sum of those counts divided by the number of batches per reader.",
        "pairs": [compare(directory, args.episode_index) for directory in args.directory],
    }
    args.out.write_text(json.dumps(output, indent=2) + "\n")
