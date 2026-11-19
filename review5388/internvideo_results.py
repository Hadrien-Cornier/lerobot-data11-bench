"""Build a public summary from verified, same-host MolmoAct2 job artifacts."""

import argparse
import json
from pathlib import Path

from internvideo_audit import load_reference


def read(path):
    return json.loads(path.read_text())


def summarize(folder, reference_path):
    reference = load_reference(reference_path)
    expected_order = reference["ordered_anchor_sha256"]
    events = [json.loads(line) for line in (folder / "run.jsonl").read_text().splitlines()]
    rows = []
    payload_hashes = set()
    hosts = set()
    first_env = None
    for done in events:
        if done["event"] not in {"internvideo-triplet-done", "internvideo-control-done"}:
            continue
        if not done["source_valid"] or done.get("failed_arms") or done.get("complete") is False:
            continue
        label = done["label"]
        for path in sorted(folder.glob(f"{label}-*.json")):
            r = read(path)
            if "mode" not in r or r["status"] != "complete":
                continue
            a = r["audit"]
            if a["batches_checked"] != 1000 or a["ordered_anchor_sha256"] != expected_order:
                raise ValueError(f"Wrong sample order or length: {path}")
            if first_env is None:
                first_env = r["environment"]
            elif first_env != r["environment"]:
                raise ValueError("Environment differed across arms")
            hosts.add(r["host"])
            payload_hashes.add(a["ordered_output_sha256"])
            row = {"run": path.stem, "mode": r["mode"], "constructor_s": r["constructor_s"],
                   "milestones": {}, "audit": a, "settings": r["settings"]}
            for step, m in r["milestones"].items():
                io = m["io"]
                stages = io["stage_durations"]
                row["milestones"][step] = dict(
                    delivered_s=m["delivered_s"], prior_consumer_audit_s=m["prior_consumer_audit_s"],
                    process_peak_rss_bytes=m["process_peak_rss_bytes"],
                    decoder_opens=stages["candidate_rgb_torchcodec_create"]["calls"],
                    frame_read_calls=stages["candidate_rgb_torchcodec_get_frames_at"]["calls"],
                    decoder_create=stages["candidate_rgb_torchcodec_create"],
                    frame_read=stages["candidate_rgb_torchcodec_get_frames_at"],
                    video=io["by_purpose"].get("video", {}),
                    cache=m.get("decoder_cache"),
                )
            rows.append(row)
    if len(hosts) != 1 or len(payload_hashes) != 1:
        raise ValueError("Expected one host and identical RGB, vectors and order across arms")
    for row in rows:
        if set(row["milestones"]) != {"1", "125", "1000"}:
            raise ValueError(f"Missing required delivery in {row['run']}")
    triplets = {}
    for event in events:
        if event["event"] == "internvideo-triplet-done" and event["source_valid"] and not event["failed_arms"]:
            label = event["label"]
            modes = {r["mode"]: r for r in rows if r["run"].startswith(label + "-")}
            if set(modes) != {"stock", "reuse", "grouped"}:
                raise ValueError(f"Incomplete triplet {label}")
            gains = {}
            for base, variant in (("stock", "reuse"), ("reuse", "grouped"), ("stock", "grouped")):
                deltas = {}
                for step in ("1", "1000"):
                    b = modes[base]["milestones"][step]["delivered_s"]
                    v = modes[variant]["milestones"][step]["delivered_s"]
                    deltas[step] = {"seconds_saved": b-v, "percent_saved": 100*(b-v)/b}
                gains[f"{variant}_vs_{base}"] = deltas
            triplets[label] = gains
    if not triplets:
        raise ValueError("No complete verified primary comparison")
    return {"environment": first_env, "reference": reference["trace_file"], "ordered_anchor_sha256": expected_order,
            "measured_examples_per_arm": 8000, "paired_savings": triplets, "arms": rows}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    summary = summarize(args.folder, Path(__file__).with_name("internvideo_reference.json.gz"))
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
