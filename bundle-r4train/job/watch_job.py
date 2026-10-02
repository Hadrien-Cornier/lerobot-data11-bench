#!/usr/bin/env python
"""Wait for an HF job to finish; cancel it if it has been RUNNING longer than --max-running-min.

The running clock starts when the job first reports RUNNING (queue time is not counted).
If the status cannot be read 5 times in a row, exit with an error instead of guessing.
Token: HF_TOKEN from the environment (never printed).

    python watch_job.py <job_id> --max-running-min 40
"""

import argparse
import sys
import time

from huggingface_hub import HfApi

TERMINAL = {"COMPLETED", "ERROR", "CANCELED", "DELETED"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("--max-running-min", type=float, required=True)
    ap.add_argument("--poll-s", type=float, default=60)
    args = ap.parse_args()

    api = HfApi()
    running_since = None
    failures = 0
    while True:
        try:
            job = api.inspect_job(job_id=args.job_id)
            failures = 0
        except Exception as exc:
            failures += 1
            print(f"status read failed ({failures}/5): {type(exc).__name__}", flush=True)
            if failures >= 5:
                sys.exit("could not read job status 5 times in a row; not cancelling blindly")
            time.sleep(args.poll_s)
            continue
        stage = str(job.status.stage).split(".")[-1]
        if stage in TERMINAL:
            print(f"final stage: {stage} message: {job.status.message}", flush=True)
            return
        if stage == "RUNNING" and running_since is None:
            running_since = time.time()
            print(f"running since {time.strftime('%H:%M:%S')}", flush=True)
        if running_since is not None and time.time() - running_since > args.max_running_min * 60:
            print(f"SAFETY CANCEL: running more than {args.max_running_min} min", flush=True)
            api.cancel_job(job_id=args.job_id)
            return
        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
