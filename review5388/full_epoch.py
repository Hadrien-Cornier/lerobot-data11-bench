"""Project one finite, single-consumer CPU epoch from public benchmark results."""

import argparse
import csv
import json
from pathlib import Path

DATASETS = {"bucket": "MolmoAct2", "abc": "ABC-130K", "droid": "DROID"}
PROFILES = ("cold", "prepared", "warm", "warm-reversed")
BATCH_SIZE = 8
START_BATCH = 125


def epoch_days(anchors, elapsed_to_125, rate):
    return (float(elapsed_to_125) + (anchors - START_BATCH * BATCH_SIZE) / float(rate)) / 86400


def table(headers, rows):
    for row in (headers, ["---"] * len(headers), *rows):
        print("| " + " | ".join(map(str, row)) + " |")
    print()


def primary_epochs(cells, sizes):
    counts, projections = [], []
    totals = {profile: [] for profile in PROFILES}
    total_batches = 0
    for dataset, name in DATASETS.items():
        anchors = sizes[dataset]
        full, tail = divmod(anchors, BATCH_SIZE)
        batches = full + bool(tail)
        counts.append([name, f"{anchors:,}", f"{full:,}", tail, f"{batches:,}"])
        total_batches += batches
        row = [name]
        for profile in PROFILES:
            cell = cells[f"{dataset}-{profile}-baseline"]
            if cell["source_verified"] != "True" or cell["workload"] != "action100":
                raise ValueError(f"Invalid workload or source in {cell['cell']}")
            if int(cell["verified_refills"]) < 2 or int(cell["post_refill_batches"] or 0) < 125:
                row.append("Insufficient refills")
                continue
            days = epoch_days(anchors, cell["batch_125_valid_s"], cell["steady_windows_per_s"])
            totals[profile].append(days)
            row.append(f"{days:.2f}")
        for profile in PROFILES:
            candidate = cells[f"{dataset}-{profile}-candidate"]
            if candidate["status"] != "error" or int(candidate["usable_unique"]) != 0:
                raise ValueError(f"Recheck #5388's 100-action results for {candidate['cell']}")
        projections.append([*row, "0 usable batches"])
    counts.append(["All three, iterated separately", f"{sum(sizes.values()):,}", "", "", f"{total_batches:,}"])
    print("## One epoch at batch size eight, final partial batch kept\n")
    table(["Dataset", "Examples", "Full batches", "Final batch size", "Batches per epoch"], counts)
    for_total = [f"{sum(vals):.2f}" if len(vals) == len(DATASETS) else "Incomplete" for vals in totals.values()]
    projections.append(["All three, in sequence", *for_total, "0 usable batches"])
    print("## Three cameras, current state and 100 target actions, projected CPU input days\n")
    table(["Dataset", "#3917 fresh build", "#3917 fresh index", "#3917 warm 1", "#3917 warm 2",
           "#5388, all four setups"], projections)


def current_frame_epochs(audit, anchors):
    if audit["full_pass"]["anchors"] != anchors or audit["full_pass"]["batch_size"] != BATCH_SIZE:
        raise ValueError("The Molmo audit and catalog disagree")
    trials = audit["trials"]
    cold = []
    for name, trial in (("Cold pair 1", trials["cold"]), ("Cold pair 2", audit["cold_checkpoint_repeat"])):
        row = [name]
        for reader in ("baseline", "candidate"):
            m = trial["reader_metrics"][reader]
            if m["finite_pass_projection"] is None:
                row.append("Insufficient refills")
                continue
            rate = m["own_after_125"]["anchors_per_s"]
            days = epoch_days(anchors, m["consumer_milestones_s"]["125"], rate)
            row.append(f"{days:.2f} ({rate:.2f}/s)")
        cold.append(row)
    print("## MolmoAct2 with current images and current action, 1 ms, two-refill projections\n")
    table(["Run", "#3917 epoch days (examples/s)", "#5388 epoch days (examples/s)"], cold)
    repeats = []
    for name in ("r01", "r02", "r03"):
        trial = trials[name]
        horizon = trial["horizon"]["last_shared_batch"]
        row = [name, f"{horizon:,}"]
        for reader in ("baseline", "candidate"):
            m = trial["reader_metrics"][reader]
            window = m["paired_after_125"]
            if (window["after_batch"], window["through_batch"]) != (START_BATCH, horizon):
                raise ValueError(f"Unmatched batch interval for {name}, {reader}")
            rate = window["anchors_per_s"]
            days = epoch_days(anchors, m["consumer_milestones_s"]["125"], rate)
            row.append(f"{days:.2f} ({rate:.2f}/s)")
        repeats.append(row)
    print("## What if each fresh-index pair's observed prefix rate held for an entire Molmo epoch?\n")
    print("These prefixes end before a verified pool refill. #3917's elapsed time includes index receipt.\n")
    table(["Pair", "Last shared batch", "#3917 epoch days (examples/s)", "#5388 epoch days (examples/s)"],
          repeats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Benchmark directory containing measurements/ and catalog/.")
    parser.add_argument("--catalog-dir", type=Path, help="Override the catalog directory.")
    args = parser.parse_args()
    catalog = args.catalog_dir or args.root / "catalog"
    if args.catalog_dir is None and not catalog.exists():
        catalog = args.root / "results"
    measurements = args.root / "measurements"
    sizes = {name: json.loads((catalog / f"catalog-{name}.json").read_text())["total_frames"]
             for name in DATASETS}
    with (measurements / "comparison.csv").open(newline="") as stream:
        cells = {row["cell"]: row for row in csv.DictReader(stream) if row["run"] == "matrix03"}
    primary_epochs(cells, sizes)
    current_frame_epochs(json.loads((measurements / "molmo-prepared-repeats.json").read_text()), sizes["bucket"])


if __name__ == "__main__":
    main()
