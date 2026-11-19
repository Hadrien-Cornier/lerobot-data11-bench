"""Upload a private, immutable input bundle and reserve each job under the gross cap."""

import argparse
import base64
import datetime as dt
import fcntl
import hashlib
import io
import json
import math
import os
import re
import sys
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "jobs.jsonl"
REPO = os.environ.get("REVIEW5388_RESULTS_REPO", "hadriencornier/lerobot-data11-bench")
RATES = {"cpu-xl": 1.0, "cpu-upgrade": 0.03}
GROSS_RESERVATION_LIMIT = 9.75
TERMINAL = {"COMPLETED", "ERROR", "CANCELED", "FAILED", "TIMEOUT"}
IMAGE = "python:3.12-slim-trixie"
BOOT = r"""
set -Eeuo pipefail
umask 077
export WORK=/work/review5388
export JOB_BOOT_STARTED_AT="$(date +%s)"
mkdir -p "$WORK/input"
cd "$WORK/input"
python3 -m pip -q install --root-user-action=ignore 'huggingface-hub==2.0.0'
hf download "$RESULTS_REPO" "$BUNDLE_PATH" --repo-type dataset --revision "$BUNDLE_REVISION" --local-dir "$WORK/input"
base64 -d "$BUNDLE_PATH" > "$WORK/input-bundle.tar.gz"
printf '%s  %s\n' "$BUNDLE_SHA256" "$WORK/input-bundle.tar.gz" | sha256sum -c
tar -xzf "$WORK/input-bundle.tar.gz"
bash runner/environment.sh
exec "$WORK/venv/bin/python" -m runner.remote
"""


def bundle(profile):
    paths = [
        ROOT / "benchmark.py",
        ROOT / "telemetry.py",
        ROOT / "turnover.py",
        ROOT / "catalog.py",
        ROOT / "runner" / "environment.sh",
        ROOT / "runner" / "publisher.py",
        ROOT / "runner" / "remote.py",
    ]
    paths.extend(sorted((ROOT / "correctness").glob("*.py")))
    if profile in ("worker-memory", "tolerance"):
        paths.append(ROOT / "worker_memory.py")
    if profile == "internvideo":
        paths.extend(ROOT / name for name in (
            "internvideo_bench.py",
            "internvideo_variants.py",
            "internvideo_audit.py",
            "internvideo_reference.json.gz",
        ))
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=path.relative_to(ROOT), recursive=False)
    content = stream.getvalue()
    return content, hashlib.sha256(content).hexdigest()


def append(ledger, row):
    ledger.seek(0, io.SEEK_END)
    ledger.write(json.dumps(row, sort_keys=True) + "\n")
    ledger.flush()


def prior_reservation(api, rows):
    jobs = {row.get("run"): row["job_id"] for row in rows if row.get("event") == "submitted"}
    observations = {row["job_id"]: row for row in rows if row.get("event") == "terminal_observation"}
    total = 0.0
    for row in rows:
        if "usd_reserved" not in row:
            continue
        reservation = row["usd_reserved"]
        job_id = row.get("job_id") or jobs.get(row["run"])
        if job_id:
            try:
                job = api.inspect_job(job_id=job_id)
                seconds = job.durations.total_secs if job.durations else None
                if job.status.stage in TERMINAL:
                    evidence = observations.get(job_id)
                    if seconds is None and job.created_at and evidence is not None:
                        original = (LEDGER.parent / evidence["source"]).read_bytes()
                        if hashlib.sha256(original).hexdigest() == evidence["response_sha256"]:
                            ended = dt.datetime.fromisoformat(evidence["conservative_observed_utc"])
                            seconds = (ended - job.created_at).total_seconds()
                    observed = LEDGER.parent / f"{row.get('run')}.job.json"
                    if seconds is None and job.created_at and observed.is_file():
                        recorded = json.loads(observed.read_text())
                        if recorded.get("job_id") == job_id and recorded.get("status") in TERMINAL:
                            ended = dt.datetime.fromisoformat(recorded["queried_utc"])
                            seconds = (ended - job.created_at).total_seconds()
                    if seconds is not None and seconds >= 0:
                        billable_minutes_upper_bound = math.ceil(seconds / 60) + 1
                        reservation = min(reservation, row["usd_per_hour"] * billable_minutes_upper_bound / 60)
            except Exception as exc:  # noqa: BLE001
                print(f"Kept full reservation for {job_id} ({type(exc).__name__}).", file=sys.stderr)
        total += reservation
    return total


def prepared_index(api, source_run, dataset="bucket"):
    repo_revision = api.dataset_info(REPO).sha
    proof_path = f"review5388/results/{source_run}/{dataset}-private-transfer.json"
    proof = api.hf_hub_download(REPO, proof_path, repo_type="dataset", revision=repo_revision)
    transfer = json.loads(Path(proof).read_text())["private_transfer"]
    expected_prefix = f"review5388/prepared/{source_run}/{dataset}/"
    path = transfer["path"]
    relative = Path(transfer["relative_cache_path"])
    if transfer["repo"] != REPO or not path.startswith(expected_prefix) or relative.name != Path(path).name:
        raise ValueError("The private transfer proof points to a different index.")
    if not re.fullmatch(r"[0-9a-f]{40}", transfer["revision"]) or not re.fullmatch(
        r"[0-9a-f]{64}", transfer["sha256"]
    ):
        raise ValueError("The archive revision and checksum must be exact.")
    return {
        "path": path,
        "revision": transfer["revision"],
        "sha256": transfer["sha256"],
        "relative_cache_path": str(relative),
        "source_run": source_run,
        "source_proof_revision": repo_revision,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="Unique private result folder and job label suffix.")
    parser.add_argument("--profile", choices=("probe", "matrix", "reverse", "correctness", "tolerance", "tolerance-repeat",
                                 "tolerance-prepared-repeat", "worker-memory", "abc-history", "internvideo"),
                        required=True)
    parser.add_argument("--minutes", type=int, required=True)
    parser.add_argument("--wait-for-job", help="Delay workload until this earlier job stops.")
    parser.add_argument("--prepared-from-run", help="Use verified indexes from this matrix run.")

    parser.add_argument("--datasets", nargs="+", choices=("droid", "bucket", "abc"))
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,39}", args.run):
        parser.error("Run name must use 3 to 40 lower-case letters, numbers, or hyphens.")
    prepared_profiles = ("tolerance", "tolerance-prepared-repeat", "worker-memory", "abc-history")
    if args.prepared_from_run and (args.profile not in prepared_profiles or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{2,39}", args.prepared_from_run
    )):
        parser.error("Invalid prepared-index run name or unsupported profile.")
    if args.profile in prepared_profiles[1:] and not args.prepared_from_run:
        parser.error("The selected profile requires --prepared-from-run.")
    if args.profile in ("tolerance", "tolerance-repeat", "tolerance-prepared-repeat", "internvideo"):
        args.datasets = ["bucket"]
    elif args.profile in ("worker-memory", "abc-history"):
        args.datasets = ["abc"] if args.profile == "abc-history" else args.datasets or ["abc", "droid"]
    else:
        args.datasets = args.datasets or ["droid", "bucket", "abc"]
    if args.wait_for_job and not re.fullmatch(r"[0-9a-f]{24}", args.wait_for_job):
        parser.error("Expected a Hugging Face Job ID.")
    if args.minutes < 5:
        parser.error("A job must have at least 5 minutes for environment setup.")
    flavor = "cpu-upgrade" if args.profile == "correctness" else "cpu-xl"
    reservation = RATES[flavor] * (args.minutes + 1) / 60
    api = HfApi()
    token = get_token()
    if not token:
        raise SystemExit("Hugging Face credentials are unavailable.")
    if api.dataset_info(REPO).private is not True:
        raise SystemExit("The configured artifact dataset must be private.")
    prepared = None
    if args.profile == "worker-memory":
        prepared = {dataset: prepared_index(api, args.prepared_from_run, dataset)
                    for dataset in dict.fromkeys(args.datasets)}
    elif args.prepared_from_run:
        prepared = prepared_index(api, args.prepared_from_run, args.datasets[0])
    content, digest = bundle(args.profile)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a+", encoding="utf-8") as ledger:
        fcntl.flock(ledger, fcntl.LOCK_EX)
        ledger.seek(0)
        rows = [json.loads(line) for line in ledger if line.strip()]
        if any(row.get("run") == args.run for row in rows):
            raise SystemExit("The run name has a reservation already. Use a new name for each launch.")
        total = prior_reservation(api, rows) + reservation
        if total > GROSS_RESERVATION_LIMIT:
            raise SystemExit(f"Job reservation would raise the gross cap to ${total:.3f}.")

        path = f"review5388/bundles/{args.run}-{digest[:12]}.txt"
        commit = api.upload_file(path_or_fileobj=base64.encodebytes(content), path_in_repo=path,
                                 repo_id=REPO, repo_type="dataset")
        observed = api.hf_hub_download(REPO, path, repo_type="dataset", revision=commit.oid,
                                       force_download=True, etag_timeout=12)
        if hashlib.sha256(base64.b64decode(Path(observed).read_bytes())).hexdigest() != digest:
            raise RuntimeError("The stored bundle failed its SHA-256 check before submission.")
        concurrent = [job.id for job in api.list_jobs(status="RUNNING")]
        append(ledger, {
            "event": "reservation",
            "run": args.run,
            "profile": args.profile,
            "flavor": flavor,
            "minutes_reserved": args.minutes,
            "datasets": list(args.datasets),
            "gross_upper_bound_including_reservation": total,
            "utc": dt.datetime.now(dt.UTC).isoformat(),
            "usd_per_hour": RATES[flavor],
            "usd_reserved": reservation,
            "bundle_revision": commit.oid,
            "bundle_sha256": digest,
            "bundle_path": path,
            "wait_for_job": args.wait_for_job,
            "prepared_index": prepared,
            "concurrent_jobs_at_launch": concurrent,
        })
        try:
            job = api.run_job(
                image=IMAGE,
                command=["bash", "-c", BOOT],
                flavor=flavor,
                name=f"lerobot-review5388-{args.run}",
                timeout=f"{args.minutes}m",
                secrets={"HF_TOKEN": token},
                env={
                    "RESULTS_REPO": REPO,
                    "BUNDLE_PATH": path,
                    "BUNDLE_REVISION": commit.oid,
                    "BUNDLE_SHA256": digest,
                    "RUN_NAME": args.run,
                    "PROFILE": args.profile,
                    "RUN_DATASETS": ",".join(dict.fromkeys(args.datasets)),
                    "JOB_HARD_CAP_SECONDS": str(args.minutes * 60),
                    "WAIT_FOR_JOB": args.wait_for_job or "",
                    "PREPARED_INDEX": json.dumps(prepared) if prepared else "",
                    "CONCURRENT_JOBS_AT_LAUNCH": json.dumps(concurrent),
                },
            )
        # The server can accept a request before a network error reaches this client.
        except Exception as exc:  # noqa: BLE001
            append(ledger, {"event": "submission_error", "run": args.run, "error_type": type(exc).__name__})
            raise SystemExit(f"Job submission failed ({type(exc).__name__}). Reservation remains counted.") from None
        append(ledger, {"event": "submitted", "run": args.run, "job_id": job.id, "url": job.url})
        print(job.id)


if __name__ == "__main__":
    main()
