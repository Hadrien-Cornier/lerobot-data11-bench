import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Check and summarize the paired MolmoAct2 run.")
parser.add_argument("log", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
log = args.log
events = []
for line in log.read_text().splitlines():
    if not line.startswith("{"):
        continue
    row = json.loads(line)
    if "event" in row:
        events.append(row)
result = {}
for arm in ("grouped", "controls"):
    start = next(row for row in events if row.get("arm") == arm and row["event"] == "start")
    milestones = {
        str(row["number"]): row for row in events if row.get("arm") == arm and row["event"] == "batch"
    }
    finish = next(row for row in events if row.get("arm") == arm and row["event"] == "finish")
    assert set(milestones) == {"1", "125", "1000"}
    assert finish["count"] == milestones["1000"]["unique_anchors"] == 8000
    result[arm] = {"start":start, "milestones":milestones, "finish":finish}
assert result["grouped"]["start"]["host"] == result["controls"]["start"]["host"]
assert result["grouped"]["start"]["software"] == result["controls"]["start"]["software"]
manifest = next(row for row in events if row["event"] == "summary")
parity = next(row for row in events if row["event"] == "parity")
assert manifest["before"] == manifest["after"]
assert manifest["grouped_status"] == manifest["controls_status"] == 0
for details in result.values():
    details["start"].pop("host")
    for milestone in details["milestones"].values():
        milestone.pop("first_anchors")
result.update(
    parity=parity, manifest=manifest,
    old_order_matches_saved_trace=result["grouped"]["finish"]["ordered_anchor_sha256"] ==
       "fc96b52afef08706deee06effe9bea3d0ca9d54d672be4f60c27f2ea6a786ea4",
    changed_order=result["grouped"]["finish"]["ordered_anchor_sha256"] !=
       result["controls"]["finish"]["ordered_anchor_sha256"],
)
args.output.write_text(json.dumps(result, indent=2)+"\n")
for arm in ("grouped", "controls"):
    arm_result=result[arm]
    print(arm,
          *(f"{float(arm_result['milestones'][milestone]['wall_s']):.3f}s" for milestone in ("1", "125", "1000")),
          "episodes/batch", arm_result["finish"]["mean_distinct_episodes"],
          "unique episodes", arm_result["finish"]["episodes"])
print("parity", parity["camera_frames"], len(parity["mismatches"]), "mismatched anchors")
print("matching old order", result["old_order_matches_saved_trace"], "changed candidate order", result["changed_order"])
