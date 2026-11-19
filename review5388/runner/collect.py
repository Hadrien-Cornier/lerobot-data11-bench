"""Save a job's safe status fields, logs and private result snapshot."""

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import re

from huggingface_hub import HfApi

from launch import LEDGER, REPO, ROOT, TERMINAL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--logs", action="store_true")
    parser.add_argument("--sync", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,39}", args.run):
        parser.error("Invalid run name.")
    rows = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    reservation = next(row for row in rows if row.get("run") == args.run and
                       row.get("event") == "reservation")
    job_id = next(row["job_id"] for row in rows if row.get("run") == args.run and
                  row.get("event") == "submitted")
    result_dir = ROOT / "results"
    api = HfApi()
    job = api.inspect_job(job_id=job_id)
    observed = dt.datetime.now(dt.timezone.utc)
    seconds = job.durations.total_secs if job.durations else None
    duration_source = "hf_server"
    if seconds is None and job.status.stage in TERMINAL and job.created_at:
        seconds = (observed - job.created_at).total_seconds()
        duration_source = "observed_terminal_time_upper_bound"
    cost_so_far_upper_bound = None
    if seconds is not None:
        cost_so_far_upper_bound = reservation["usd_per_hour"] * (math.ceil(seconds / 60) + 1) / 60
        cost_so_far_upper_bound = min(cost_so_far_upper_bound, reservation["usd_reserved"])
    record = {
        "job_id": job.id, "name": (job.labels or {}).get("name"), "flavor": job.flavor,
        "status": job.status.stage, "url": job.url,
        "start": job.started_at, "creation": job.created_at, "finish": job.finished_at,
        "duration": vars(job.durations) if job.durations else None,
        "gross_running_time_upper_bound_usd": cost_so_far_upper_bound,
        "remaining_reservation_usd": reservation["usd_reserved"] if job.status.stage not in TERMINAL or
        cost_so_far_upper_bound is None else cost_so_far_upper_bound,
        "queried_utc": observed.isoformat(),
        "duration_source": duration_source,
        "artifact_repo": REPO,
    }
    if args.logs:
        with (result_dir / f"{args.run}.job.log").open("w") as stream:
            for line in api.fetch_job_logs(job_id=job_id):
                stream.write(line.rstrip("\n") + "\n")

    if args.sync:
        info = api.dataset_info(REPO)
        if info.private is not True:
            raise RuntimeError("Evidence source must be private.")
        prefix = f"review5388/results/{args.run}"
        path = result_dir / "downloads"
        api.snapshot_download(REPO, repo_type="dataset", revision=info.sha, local_dir=path,
                              allow_patterns=[f"{prefix}/*"])
        record["artifact_revision"] = info.sha
    payload = json.dumps(record, indent=2, default=str) + "\n"
    (result_dir / f"{args.run}.job.json").write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
