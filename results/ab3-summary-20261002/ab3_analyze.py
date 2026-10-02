"""Summarize ab3 jobs: files/s per arm, paired ratios, bytes and round trips, and the full sidecar build time.

Extrapolation: (video files in the dataset) / (median files/s of the arm), at 16 workers on one HF Jobs
cpu-upgrade machine inside the HF network. A dataset with one video file uses the median time for that file.

Usage: python ab3_analyze.py ab3-files.json RESULTS_DIR [RESULTS_DIR ...]  (each holds ab3/ab3.json)
"""

import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ARMS = ["BASE", "PATCH", "A1M", "A3M", "A1M-c32"]


def ci(vals, n=4000, seed=0):
    rng = random.Random(seed)
    meds = sorted(st.median(rng.choices(vals, k=len(vals))) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n)]


def fmt_time(s):
    return f"{s:.1f} s" if s < 90 else f"{s / 60:.1f} min" if s < 5400 else f"{s / 3600:.1f} h"


def main():
    catalog = json.loads(Path(sys.argv[1]).read_text())
    runs, eq = [], defaultdict(lambda: [0, 0])
    for d in sys.argv[2:]:
        r = json.loads((Path(d) / "ab3" / "ab3.json").read_text())
        for row in r["runs"]:
            runs.append({**row, "job": r["job"]})
        for repo, e in r["equality"].items():
            eq[repo][0] += e["same"]
            eq[repo][1] += e["files"]
    bad = [x for x in runs if x.get("status") != "ok"]
    out = {"errors": bad, "datasets": {}}
    lines = ["| Dataset | Files | Arm | Files/s | vs BASE (95% CI) | vs PATCH (95% CI) | KiB/file | Round trips/file | Full build |",
             "|---|---|---|---|---|---|---|---|---|"]
    for repo in catalog:
        rows = [x for x in runs if x["repo"] == repo and x.get("status") == "ok"]
        if not rows:
            continue
        n_files = len(catalog[repo]["files"])
        by = {(x["job"], x["rep"], x["arm"]): x for x in rows}
        ds = {"files": n_files, "equality": eq[repo], "arms": {}}
        for arm in ARMS:
            xs = [x for x in rows if x["arm"] == arm]
            if not xs:
                continue
            fps = st.median(x["files_per_s"] for x in xs)
            a = {"runs": len(xs), "files_per_s": fps, "kib_per_file": st.mean(x["kib_per_file"] for x in xs),
                 "round_trips": st.mean(x["round_trips_per_file"] for x in xs),
                 "tail_hit": sum(x["tail_hit"] for x in xs) / max(1, sum(x["files"] for x in xs))}
            for ref in ("BASE", "PATCH"):
                pairs = [by[(j, rp, arm)]["files_per_s"] / by[(j, rp, ref)]["files_per_s"]
                         for (j, rp, ar) in by if ar == arm and (j, rp, ref) in by]
                if pairs and arm != ref:
                    a[f"vs_{ref}"] = (st.median(pairs), *ci(pairs), sum(p > 1 for p in pairs), len(pairs))
            if n_files == 1:
                a["build_s"] = st.median(x["file_s_median"] for x in xs)
            else:
                a["build_s"] = n_files / fps
            ds["arms"][arm] = a

            def r(key):
                v = a.get(key)
                return f"{v[0]:.2f} ({v[1]:.2f} to {v[2]:.2f})" if v else "1"
            lines.append(f"| {repo} | {n_files:,} | {arm} | {fps:.1f} | {r('vs_BASE') if arm != 'BASE' else '1'} | "
                         f"{r('vs_PATCH') if arm != 'PATCH' else '1'} | {a['kib_per_file']:,.0f} | {a['round_trips']:.2f} | "
                         f"{fmt_time(a['build_s'])} |")
        out["datasets"][repo] = ds
    tot = {arm: sum(d["arms"][arm]["build_s"] for d in out["datasets"].values() if arm in d["arms"]) for arm in ARMS}
    out["total_build_s"] = tot
    lines.append("")
    lines.append("Total full build, all datasets: " + ", ".join(f"{a} {fmt_time(s)}" for a, s in tot.items()))
    lines.append("Equality (same index in BASE, PATCH, A1M): " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in eq.items()))
    lines.append(f"Failed runs: {len(bad)}")
    print("\n".join(lines))
    Path(sys.argv[2]).parent.joinpath("ab3-summary.json").write_text(json.dumps(out, indent=1))
    Path(sys.argv[2]).parent.joinpath("ab3-summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
