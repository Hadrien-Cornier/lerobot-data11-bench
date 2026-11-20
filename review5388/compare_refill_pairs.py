"""Compare independently audited MolmoAct2 pairs on different CPU hosts."""

import argparse
import json
from pathlib import Path

from checkpoint_crossover import require
from native_refill_audit import READERS, SETTINGS
from summarize import read


def compare_pairs(paths):
    runs = [read(path) for path in paths]
    first, second = runs
    for key in (
        "source_manifest_sha256",
        "source_examples",
        "batches_per_epoch",
        "settings",
    ):
        require(first[key] == second[key], f"Runs differ in {key}.")
    require(
        len({r["host"]["job_id"] for r in runs}) == 2
        and len({r["host"]["vcpus"] for r in runs}) == 2
        and first["host"]["reader_order"]
        == list(reversed(second["host"]["reader_order"])),
        "Different CPU quotas, jobs, and opposite reader orders are required for this comparison.",
    )
    require(
        first["settings"] == SETTINGS
        and first["matching_index"]["sha256"] == second["matching_index"]["sha256"],
        "The workload or publisher-index identity differs.",
    )
    summary = []
    for run in runs:
        leads = run["delivery_leads"]
        phases = [leads["initial_lead"], *leads["lead_changes"]]
        first_index_lead = next(
            (
                phase["first_batch"]
                for phase in phases
                if phase and phase["reader"] == "baseline"
            ),
            None,
        )
        final = leads["final_lead_through_checkpoint"]
        days = {
            r: run["readers"][r]["window_sensitivity"]
            .get("through_both_refills", {})
            .get("cpu_epoch_days")
            for r in READERS
        }
        checkpoints = {
            point["batch"]: [point["baseline_s"], point["candidate_s"]]
            for point in run["milestones"]
        }
        summary.append(
            {
                "run": run["run"],
                "vcpus": run["host"]["vcpus"],
                "order": run["host"]["reader_order"],
                "receipt_s": run["matching_index"]["consumer_receipt_s"],
                "horizon": run["last_shared_batch"],
                "baseline_ahead_from_first_batch": first_index_lead == 1,
                "first_baseline_ahead_batch": first_index_lead,
                "lead_changes": leads["lead_change_count"],
                "final_leader": final["reader"] if final else None,
                "final_lead_from_batch": final["first_batch"] if final else None,
                "final_gap_baseline_minus_candidate_s": run["milestones"][-1][
                    "baseline_minus_candidate_s"
                ],
                "milestone_seconds_baseline_then_candidate": checkpoints,
                "epoch_days_by_reader": days,
                "epoch_days_saved_by_baseline": days["candidate"] - days["baseline"]
                if all(days.values())
                else None,
                "refill_batches_by_reader": {
                    r: run["readers"][r]["refill_batches"] for r in READERS
                },
                "mean_distinct_episodes_per_batch": {
                    r: run["readers"][r]["shared_prefix_episode_diversity"][
                        "mean_distinct_episodes_per_batch"
                    ]
                    for r in READERS
                },
            }
        )
    replicated_ids = {}
    for reader in READERS:
        require(
            first["readers"][reader]["lerobot_revision"]
            == second["readers"][reader]["lerobot_revision"],
            f"{reader} revisions differ.",
        )
        prefix_hashes = []
        for run in runs:
            sample = run["readers"][reader]
            checkpoints = {
                int(n): row["sha256"]
                for n, row in sample["order_check_against_long_stock_run"].items()
            }
            checkpoints[run["last_shared_batch"]] = sample[
                "shared_prefix_episode_diversity"
            ]["ordered_anchor_sha256"]
            prefix_hashes.append(checkpoints)
        shared = max(prefix_hashes[0].keys() & prefix_hashes[1].keys())
        replicated_ids[reader] = {
            "through_batch": shared,
            "matches": prefix_hashes[0][shared] == prefix_hashes[1][shared],
        }
    return {
        "common_batches_per_reader": min(run["last_shared_batch"] for run in runs),
        "common_source_examples": first["source_examples"],
        "same_ordered_ids_for_each_reader_across_hosts": replicated_ids,
        "pairs": summary,
        "limits": "CPU host size and reader order both changed between runs. Each host had one pair. "
        "Epoch times project measured CPU batch delivery; no model steps were timed.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pairs", nargs=2, type=Path, help="Two summaries from native_refill_audit.py."
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(compare_pairs(args.pairs), indent=2) + "\n")


if __name__ == "__main__":
    main()
