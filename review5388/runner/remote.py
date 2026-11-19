"""Run paired cells sequentially on one CPU host and save private evidence after each cell."""

import datetime as dt
import gzip
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("WORK", "/work/review5388"))
PY = WORK / "venv/bin/python"
RESULTS = WORK / "results"
CACHE = WORK / "caches"
REPO = os.environ["RESULTS_REPO"]
RUN = os.environ["RUN_NAME"]
PROFILE = os.environ["PROFILE"]
JOB_SECONDS = int(os.environ["JOB_HARD_CAP_SECONDS"])
STARTED = time.monotonic()
DEADLINE = STARTED + JOB_SECONDS - 240
if PROFILE == "internvideo":
    DEADLINE -= max(0, time.time() - int(os.environ["JOB_BOOT_STARTED_AT"]))
API = HfApi()
DEST = f"review5388/results/{RUN}"
BUCKET_MANIFEST = "aebacc429d54f284e103ff10613b925f5827c347e0678846b0b6f0015dded782"
DATASETS = tuple(os.environ.get("RUN_DATASETS", "droid,bucket,abc").split(","))
COLD_CAP = {"bucket": 2100, "droid": 1000, "abc": 5600}


def now():
    return dt.datetime.now(dt.UTC).isoformat()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def append(data):
    with (RESULTS / "run.jsonl").open("a") as stream:
        stream.write(json.dumps({"utc": now(), **data}, sort_keys=True) + "\n")
    print(json.dumps(data, sort_keys=True), flush=True)


def upload(label):
    try:
        commit = API.upload_folder(
            folder_path=RESULTS,
            path_in_repo=DEST,
            repo_id=REPO,
            repo_type="dataset",
            commit_message=f"LeRobot #5388 {RUN} {label}",
            ignore_patterns=["*.tmp"],
        )
        print(f"Saved {label} to {DEST} at {commit.oid}", flush=True)
    except Exception as exc:  # noqa: BLE001
        append({"event": "upload-failed", "label": label, "error_type": type(exc).__name__})


def read_file(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def env_for(reader, cache, *, release=False):
    source = WORK / "src" / reader / "src"
    python_path = f"{source}:{WORK / 'datasets-release'}" if release else str(source)
    return {
        **os.environ,
        "HF_HOME": str(cache / "hub-home"),
        "HF_HUB_CACHE": str(cache / "hub-home/hub"),
        "HF_XET_CACHE": str(cache / "hub-home/xet"),
        "HF_LEROBOT_HOME": str(cache / "lerobot"),
        "HF_DATASETS_CACHE": str(cache / "datasets"),
        "TMPDIR": str(cache / "tmp"),
        "PYTHONPATH": python_path,
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "ARROW_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    }


def run(label, argv, timeout, reader="baseline", cache=None, *, release=False):
    time_left = int(DEADLINE - time.monotonic())
    timeout = min(timeout, time_left)
    if timeout < 10:
        append({"event": "skipped", "label": label, "reason": "job time exhausted"})
        return None
    cache = cache if cache is not None else CACHE / label
    (cache / "tmp").mkdir(parents=True, exist_ok=True)
    logfile = RESULTS / f"{label}.log"
    started = time.monotonic()
    append({
        "event": "start", "label": label, "reader": reader, "argv": [str(x) for x in argv],
        "cache": str(cache), "timeout_seconds": timeout,
        "datasets_source": "pypi-5.2.0" if release else "pull-8983",
    })
    timed_out = False
    with logfile.open("w") as stream:
        child = subprocess.Popen(
            [str(PY), *map(str, argv)],
            cwd=ROOT, env=env_for(reader, cache, release=release), stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while child.poll() is None:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                break
            try:
                child.wait(timeout=min(remaining, 55))
            except subprocess.TimeoutExpired:
                report = RESULTS / f"{label}.json"
                details = {}
                if report.is_file():
                    try:
                        row = json.loads(report.read_text())
                        details = {"status": row.get("status"), "milestones": list(
                            row.get("milestones", row.get("timings", {}).get("milestones", {}))
                        ), "progress": row.get("checked_batches", row.get("progress"))}
                    except (FileNotFoundError, json.JSONDecodeError):
                        pass
                print(json.dumps({"event": "heartbeat", "label": label,
                                  "seconds": round(time.monotonic() - started, 1), **details}), flush=True)
        if child.poll() is None:
            timed_out = True
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            except ProcessLookupError:
                child.wait()
        return_code = child.returncode
    append({
        "event": "stop", "label": label, "reader": reader, "code": return_code,
        "timed_out": timed_out, "seconds": time.monotonic() - started,
        "cgroup_memory_events": read_file("/sys/fs/cgroup/memory.events"),
    })
    for path in RESULTS.glob(f"{label}.*"):
        if path.suffix == ".jsonl" and path.stat().st_size >= 2 * 1024**2:
            with path.open("rb") as source, gzip.open(f"{path}.gz", "wb", compresslevel=1) as target:
                shutil.copyfileobj(source, target)
            path.unlink()
    upload(label)
    return return_code


def verify_bucket(label):
    out = RESULTS / f"bucket-manifest-{label}.json"
    code = run(f"bucket-manifest-{label}", ["catalog.py", "--dataset", "bucket", "--out", out,
                                          "--expect", BUCKET_MANIFEST], 180)
    return code == 0


def benchmark(dataset, reader, label, cache, timeout, *, single_frame=False,
              batches=125000, warm_batches=0, cache_state="cold", history=False, release=False,
              tolerance_s=None, fixed_target=False, seed=None):
    out = RESULTS / f"{label}.json"
    args = [
        "benchmark.py", "--dataset", dataset, "--reader", reader, "--output", out,
        "--max-batches", str(batches), "--workers", "0", "--cache-state", cache_state,
        "--max-seconds", str(max(10, timeout - 20)),
    ]
    if single_frame:
        args.append("--single-frame")
    if history:
        args.append("--history")
    if tolerance_s is not None:
        args += ["--tolerance-s", str(tolerance_s)]
    if warm_batches:
        args += ["--warm-batches", str(warm_batches)]
    if fixed_target:
        args.append("--continue-to-max-batches")
    if seed is not None:
        args += ["--seed", str(seed)]
    return run(label, args, timeout, reader, cache, release=release)


def publisher_audit(dataset):
    label = f"{dataset}-publisher-audit"
    out = RESULTS / f"{label}.json"
    code = run(label, ["runner/publisher.py", "--dataset", dataset, "--audit", "--out", out], 220)
    if code == 0 and out.exists():
        return json.loads(out.read_text())
    return {}


def native_archive(cache):
    archives = list((cache / "lerobot" / "streaming-sidecars").rglob("mp4-v*.npz"))
    if len(archives) == 1:
        return archives[0]
    return None


def download_prepared(dataset, cache, reference, timeout=900, *, measure_wire=False, label=None):
    label = label if label is not None else f"{dataset}-private-transfer"
    out = RESULTS / f"{label}.json"
    command = [
        "runner/publisher.py", "--dataset", dataset, "--out", out, "--private-repo", REPO,
        "--private-path", reference["path"], "--private-revision", reference["revision"],
        "--private-destination", reference["relative_cache_path"], "--private-sha256", reference["sha256"],
    ]
    if measure_wire:
        command = [
            "worker_memory.py", "receipt", "--output", RESULTS / f"{label}-wire.json",
            "--receipt-manifest", out, "--", PY, *command,
        ]
    code = run(label, command, timeout, cache=cache)
    if code != 0 or not out.exists():
        return False
    observed = json.loads(out.read_text())["private_transfer"]["sha256"]
    if observed != reference["sha256"]:
        raise RuntimeError(f"Prepared-index checksum changed for {dataset}")
    return True


def capture_transfer(dataset, archive, source_cache, cache):
    if time.monotonic() + 180 > DEADLINE:
        return False
    path = f"review5388/prepared/{RUN}/{dataset}/{archive.name}"
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    relative = archive.relative_to(source_cache / "lerobot")
    start = time.monotonic()
    try:
        commit = API.upload_file(
            path_or_fileobj=archive, path_in_repo=path, repo_id=REPO, repo_type="dataset",
            commit_message=f"Native LeRobot #5388 prepared index for {dataset}",
        )
    except Exception as exc:  # noqa: BLE001
        append({"event": "transfer-stage-failed", "dataset": dataset, "error_type": type(exc).__name__})
        return False
    append({
        "event": "private-transfer-staged", "dataset": dataset, "revision": commit.oid,
        "path": path, "sha256": checksum, "bytes": archive.stat().st_size,
        "seconds": time.monotonic() - start,
    })
    reference = {
        "path": path, "revision": commit.oid,
        "relative_cache_path": str(relative), "sha256": checksum,
    }
    return download_prepared(dataset, cache, reference)


def pair(dataset, state, readers, caps, caches, *, cache_state="cold", max_batches=125000,
         single_frame=False, history=False, warm_batches=0, tolerance_s=None, fixed_target=False, seed=None):
    checks = {"dataset": dataset, "pair": state, "single_frame": single_frame, "history": history,
              "tolerance_s": tolerance_s or 1e-4, "fixed_target": max_batches if fixed_target else None}
    if dataset == "bucket" and not verify_bucket(f"{state}-before"):
        append({"event": "pair-skipped", "reason": "source manifest", **checks})
        return False
    for reader in readers:
        reader_state = cache_state[reader] if isinstance(cache_state, dict) else cache_state
        # Leave time to save a capped cell and recheck the mutable source after both readers.
        remaining = int(DEADLINE - time.monotonic() - (120 if dataset == "bucket" else 0))
        benchmark(
            dataset, reader, f"{dataset}-{state}-{reader}", caches[reader], min(caps[reader], remaining),
            single_frame=single_frame, history=history, batches=max_batches,
            warm_batches=warm_batches, cache_state=reader_state, tolerance_s=tolerance_s, fixed_target=fixed_target,
            seed=seed,
        )
    source_valid = dataset != "bucket" or verify_bucket(f"{state}-after")
    append({"event": "pair-done", "source_valid": source_valid, **checks})
    upload(f"{dataset}-{state}-pair")
    return source_valid


def correctness():
    fixture = WORK / "synthetic-v3"
    code = run("fixture-build", ["correctness/fixture.py", fixture], 180)
    if code != 0:
        return
    for reader in ("baseline", "candidate"):
        check_variant = "baseline" if reader == "baseline" else "proposal"
        for mode in ("anchors", "full", "numeric", "positive", "video"):
            label = f"fixture-{reader}-{mode}"
            command = [
                "correctness/check_stream.py", check_variant, fixture, RESULTS / f"{label}.json",
                "--mode", mode, "--workers", "0",
            ]
            if reader == "candidate":
                command.append("--local-id")
            run(label, command, 150, reader)


def cold_cache(dataset):
    return {reader: CACHE / f"{dataset}-cold-{reader}" for reader in ("baseline", "candidate")}


def matrix():
    reverse = PROFILE == "reverse"
    first = ("baseline", "candidate") if reverse else ("candidate", "baseline")
    second = tuple(reversed(first))
    archive = {}
    measurement_caps = {"baseline": 1200, "candidate": 1200}

    # Establish the action-100 comparison on all sources before diagnostics.
    for dataset in DATASETS:
        caps = {"baseline": COLD_CAP[dataset], "candidate": 900}
        pair(dataset, "cold", first, caps, cold_cache(dataset))
        index = native_archive(cold_cache(dataset)["baseline"])
        if index is not None:
            archive[dataset] = index

        if dataset == "bucket":
            # The native 10-second startup claim uses the current-frame workload.
            # The candidate gets a fresh metadata cache. Reuse the baseline index
            # to avoid measuring the same cold index build twice.
            label = "cold-single-frame-candidate"
            if verify_bucket(f"{label}-before"):
                benchmark(dataset, "candidate", f"bucket-{label}", CACHE / f"bucket-{label}",
                          1200, single_frame=True)
                valid = verify_bucket(f"{label}-after")
                append({"event": "comparison-source", "dataset": "bucket", "label": label, "source_valid": valid})
            if dataset in archive:
                label = "warm-single-frame-baseline"
                if verify_bucket(f"{label}-before"):
                    benchmark(dataset, "baseline", f"bucket-{label}", cold_cache(dataset)["baseline"],
                              1200, single_frame=True, cache_state="warm")
                    valid = verify_bucket(f"{label}-after")
                    append({"event": "comparison-source", "dataset": "bucket", "label": label, "source_valid": valid})

    # Restart in new processes with the same local disk state. No iterator is prewarmed.
    for dataset in DATASETS:
        if dataset in archive:
            pair(dataset, "warm", second, measurement_caps, cold_cache(dataset), cache_state="warm")
        else:
            append({"event": "warm-blocked", "dataset": dataset, "reason": "no validated index"})

    if "abc" in DATASETS and "abc" in archive:
        pair("abc", "warm-history", first, measurement_caps, cold_cache("abc"),
             cache_state="warm", history=True)

    # Time each transfer separately, then open the consumer on a clean metadata cache.
    for dataset in DATASETS:
        audit = publisher_audit(dataset)
        if dataset not in archive:
            append({"event": "prepared-blocked", "dataset": dataset, "reason": "no validated index"})
            continue
        if any(x["status"] == "matching_header" and x["url"] == audit["expected_url"]
               for x in audit.get("publisher_files", [])):
            append({"event": "native-publication-available", "dataset": dataset})
        prepared = {reader: CACHE / f"{dataset}-prepared-{reader}" for reader in first}
        if capture_transfer(dataset, archive[dataset], cold_cache(dataset)["baseline"], prepared["baseline"]):
            pair(dataset, "prepared", first, measurement_caps, prepared, cache_state="unclassified")

    if "droid" in DATASETS and "droid" in archive:
        label = "droid-warm-release-control"
        if (WORK / "datasets-release").is_dir():
            benchmark("droid", "baseline", label, cold_cache("droid")["baseline"], 210, batches=2,
                      cache_state="warm", release=True)
        else:
            append({"event": "release-control-skipped", "reason": "release wheel unavailable"})

    # Reverse the primary warm order on this host, with no repeat of cold indexing.
    for dataset in DATASETS:
        if dataset in archive:
            pair(dataset, "warm-reversed", first, measurement_caps, cold_cache(dataset), cache_state="warm")


def probe():
    dataset = "droid"
    caps = {"baseline": 600, "candidate": 240}
    pair(dataset, "probe", ("candidate", "baseline"), caps, cold_cache(dataset),
         single_frame=True, max_batches=2)


def tolerance():
    raw_reference = os.environ.get("PREPARED_INDEX")
    reference = json.loads(raw_reference) if raw_reference else None
    order = ("baseline", "candidate") if reference else ("candidate", "baseline")
    caches = {reader: CACHE / f"bucket-tolerance-{reader}" for reader in order}
    append({
        "event": "tolerance-control-setup", "tolerance_s": 0.001,
        "baseline_start_state": "fresh_metadata_and_verified_private_transfer" if reference else "fresh",
        "candidate_start_state": "fresh", "prepared_index": reference,
        "concurrent_jobs_at_launch": json.loads(os.environ.get("CONCURRENT_JOBS_AT_LAUNCH", "[]")),
    })
    if reference:
        if not verify_bucket("tolerance-1ms-before-transfer"):
            raise RuntimeError("Molmo inventory has changed.")
        if not download_prepared("bucket", caches["baseline"], reference, timeout=180):
            raise RuntimeError("Prepared Molmo index could not be transferred.")
        valid_source = pair(
            "bucket", "current-frame-tolerance-1ms-prepared", order, {"baseline": 330, "candidate": 330}, caches,
            cache_state={"baseline": "unclassified", "candidate": "cold"}, single_frame=True,
            tolerance_s=0.001, max_batches=1000,
        )
        if valid_source and verify_bucket("worker-memory-before"):
            prefetch = worker_prefetch("bucket")
            if prefetch is not None:
                append({"event": "worker-memory-cache-reuse", "dataset": "bucket",
                        "source_cell": "bucket-current-frame-tolerance-1ms-prepared-baseline",
                        "cache_state": "same prepared archive and mmap, warm metadata and OS pages"})
                measure_workers("bucket", caches["baseline"], prefetch)
        return

    state = "current-frame-tolerance-1ms"
    valid_source = pair("bucket", state, order, {"baseline": 2200, "candidate": 1800}, caches,
                        single_frame=True, tolerance_s=0.001)
    if not valid_source:
        return

    means = []
    for reader in order:
        result_path = RESULTS / f"bucket-{state}-{reader}.json"
        if not result_path.exists():
            return
        summary = json.loads(result_path.read_text())
        window = summary["windows"].get("to_1000", {})
        if summary.get("status") != "ok" or window.get("usable_unique_samples", 0) < 8000:
            return
        means.append(window["usable_unique_samples_per_s"])
    seconds = int((DEADLINE - time.monotonic() - 120) / 2)
    if seconds < 300:
        return
    append({"event": "tolerance-warm-repeat", "cold_prefix_valid_samples_per_s": means})
    pair("bucket", f"{state}-warm", tuple(reversed(order)), {reader: seconds for reader in order},
         caches, cache_state="warm", single_frame=True, tolerance_s=0.001, max_batches=1000)


def tolerance_prepared_repeat():
    reference = json.loads(os.environ["PREPARED_INDEX"])
    dataset = "bucket"
    batches = 4000
    seed = 42
    forward = ("baseline", "candidate")
    orders = (forward, tuple(reversed(forward)), forward)
    for repeat, order in enumerate(orders, 1):
        suffix = f"r{repeat:02}"
        state = f"current-frame-tolerance-1ms-prepared-{suffix}"
        caches = {reader: CACHE / f"{dataset}-{state}-{reader}" for reader in order}
        if any(cache.exists() for cache in caches.values()):
            raise FileExistsError(f"A prepared repetition cache already exists for {state}.")
        transfer_label = f"{dataset}-private-transfer-{suffix}"
        before_transfer = f"tolerance-1ms-before-transfer-{suffix}"
        append({
            "event": "tolerance-control-setup", "pair": state, "repeat": repeat, "reader_order": order,
            "tolerance_s": 0.001, "seed_requested": seed, "max_batches": batches, "pool_size": 32,
            "baseline_start_state": "fresh_metadata_and_verified_private_transfer",
            "candidate_start_state": "fresh", "prepared_index": reference,
            "transfer_label": transfer_label, "before_transfer_label": before_transfer,
            "concurrent_jobs_at_launch": json.loads(os.environ.get("CONCURRENT_JOBS_AT_LAUNCH", "[]")),
        })
        if not verify_bucket(before_transfer):
            raise RuntimeError(f"Molmo inventory check before transfer failed for {state}.")
        if not download_prepared(dataset, caches["baseline"], reference, timeout=180, label=transfer_label):
            raise RuntimeError(f"Prepared Molmo index transfer failed for {state}.")
        if not pair(
            dataset, state, order, {"baseline": 700, "candidate": 700}, caches,
            cache_state={"baseline": "unclassified", "candidate": "cold"}, single_frame=True,
            tolerance_s=0.001, max_batches=batches, fixed_target=True, seed=seed,
        ):
            raise RuntimeError(f"Molmo paired inventory check failed for {state}.")


def tolerance_repeat():
    order = ("baseline", "candidate")
    caches = {reader: CACHE / f"bucket-tolerance-repeat-{reader}" for reader in order}
    pair("bucket", "current-frame-tolerance-1ms-repeat", order, {"baseline": 2000, "candidate": 2550}, caches,
         single_frame=True, tolerance_s=0.001, max_batches=25000, fixed_target=True)


def internvideo_arm(label, mode, reference):
    arm = f"{label}-{mode}"
    cache = CACHE / arm
    if cache.exists():
        raise FileExistsError(f"Cache already exists for {arm}.")
    out = RESULTS / f"{arm}.json"
    before = time.monotonic()
    timeout = min(450, int(DEADLINE - before - 120))
    code = run(arm, ["internvideo_bench.py", "--mode", mode, "--reference", reference,
                     "--output", out], timeout, reader="candidate", cache=cache)
    return code == 0 and out.is_file(), time.monotonic() - before


def internvideo():
    reference = ROOT / "internvideo_reference.json.gz"
    shutil.copyfile(reference, RESULTS / reference.name)
    with reference.open("rb") as stream:
        reference_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    forward = ("stock", "reuse", "grouped")
    timings = {}
    for repeat, order in enumerate((forward, tuple(reversed(forward))), 1):
        label = f"internvideo-r{repeat:02}"
        if repeat == 2:
            remaining = DEADLINE - time.monotonic()
            planned_seconds = 1.25 * sum(timings.values()) + 180
            if remaining < planned_seconds:
                append({"event": "reverse-skipped", "label": label, "seconds_remaining": remaining,
                        "seconds_needed": planned_seconds})
                return
        append({"event": "internvideo-setup", "label": label, "arm_order": order,
                "reference_sha256": reference_sha256, "cache_state": "fresh_per_arm"})
        if not verify_bucket(f"{label}-before"):
            raise RuntimeError(f"Source check failed before {label}.")
        failures = []
        for mode in order:
            complete, timings[mode] = internvideo_arm(label, mode, reference)
            if not complete:
                failures.append(f"{label}-{mode}")
        valid_source = verify_bucket(f"{label}-after")
        append({"event": "internvideo-triplet-done", "label": label, "source_valid": valid_source,
                "failed_arms": failures, "arm_elapsed_seconds_including_upload": timings})
        upload(label)
        if failures or not valid_source:
            raise RuntimeError(f"Incomplete comparison for {label}. See the private artifacts.")

    control_seconds = max(timings.values())
    for repeat in (1, 2):
        label = f"internvideo-control-r{repeat:02}"
        remaining = DEADLINE - time.monotonic()
        planned_seconds = 1.25 * control_seconds + 180
        if remaining < planned_seconds:
            append({"event": "control-skipped", "label": label, "seconds_remaining": remaining,
                    "seconds_needed": planned_seconds})
            break
        if not verify_bucket(f"{label}-before"):
            append({"event": "control-skipped", "label": label, "reason": "source manifest"})
            break
        complete, control_seconds = internvideo_arm(label, "batch_singletons", reference)
        valid_source = verify_bucket(f"{label}-after")
        append({"event": "internvideo-control-done", "label": label, "source_valid": valid_source,
                "complete": complete, "arm_elapsed_seconds_including_upload": control_seconds})
        upload(label)
        if not complete or not valid_source:
            break


def worker_prefetch(dataset):
    from benchmark import BATCH_SIZE, INPUTS

    shm = shutil.disk_usage("/dev/shm") if Path("/dev/shm").is_dir() else None
    _, _, (channels, height, width), cameras = INPUTS[dataset]
    image_bytes_per_batch = BATCH_SIZE * len(cameras) * channels * height * width
    # Allow 128 KiB per batch for actions, state and other tensors. Reserve
    # two additional batches and 20% of available shared memory.
    batch_bytes = image_bytes_per_batch + 128 * 1024
    prefetch = next(
        (factor for factor in (4, 2, 1) if shm and (4 * factor + 2) * batch_bytes <= 0.8 * shm.free),
        None,
    )
    append({"event": "worker-memory-config", "dataset": dataset, "worker_counts": [1, 4],
            "prefetch_factor": prefetch, "dev_shm_total_bytes": shm.total if shm else None,
            "dev_shm_free_bytes": shm.free if shm else None, "image_bytes_per_batch": image_bytes_per_batch})
    if prefetch is None:
        append({"event": "worker-memory-skipped", "dataset": dataset,
                "reason": "shared memory is too small or unavailable for four workers at prefetch factor 1"})
    return prefetch


def measure_workers(dataset, cache, prefetch):
    for workers in (1, 4):
        label = f"{dataset}-worker-memory-w{workers}"
        run(label, ["worker_memory.py", "measure", "--dataset", dataset, "--workers", str(workers),
                    "--output", RESULTS / f"{label}.json", "--max-batches", "125",
                    "--prefetch-factor", str(prefetch), "--require-mapped-index",
                    "--sidecar-manifest", RESULTS / f"{dataset}-private-transfer.json"], 360, cache=cache)
    valid_source = dataset != "bucket" or verify_bucket("worker-memory-after")
    append({"event": "worker-memory-pair-done", "dataset": dataset, "source_valid": valid_source,
            "worker_counts": [1, 4]})
    upload(f"{dataset}-worker-memory")


def worker_memory():
    references = json.loads(os.environ["PREPARED_INDEX"])
    map_index = (
        "from pathlib import Path; import sys; "
        "from lerobot.streaming.sidecar_utils import mapped_sidecar; "
        "print(mapped_sidecar(Path(sys.argv[1]))[0])"
    )
    for dataset in DATASETS:
        if dataset == "bucket" and not verify_bucket("worker-memory-before"):
            continue
        prefetch = worker_prefetch(dataset)
        if prefetch is None:
            continue
        cache = CACHE / f"{dataset}-worker-memory"
        reference = references[dataset]
        transfer_cap = 540 if dataset == "abc" else 300
        if not download_prepared(dataset, cache, reference, timeout=transfer_cap, measure_wire=True):
            append({"event": "worker-memory-skipped", "dataset": dataset, "reason": "transfer failed"})
            continue
        archive = cache / "lerobot" / reference["relative_cache_path"]
        if run(f"{dataset}-worker-memory-map-index", ["-c", map_index, archive], 300, cache=cache) != 0:
            append({"event": "worker-memory-skipped", "dataset": dataset, "reason": "mapping failed"})
            continue
        measure_workers(dataset, cache, prefetch)


def abc_history():
    reference = json.loads(os.environ["PREPARED_INDEX"])
    cache = CACHE / "abc-history-prepared"
    if not download_prepared("abc", cache, reference, timeout=540):
        raise RuntimeError("Prepared ABC index could not be transferred.")
    # The earlier candidate failed before any batch for this exact history workload.
    benchmark("abc", "baseline", "abc-prepared-history-baseline", cache, 2600,
              history=True, cache_state="unclassified")


def wait_for_previous():
    job_id = os.environ.get("WAIT_FOR_JOB")
    if not job_id:
        return
    terminal = {"COMPLETED", "ERROR", "CANCELED", "FAILED", "TIMEOUT"}
    append({"event": "waiting-for-job", "job_id": job_id})
    while time.monotonic() < DEADLINE:
        stage = API.inspect_job(job_id=job_id).status.stage
        if stage in terminal:
            append({"event": "previous-job-done", "job_id": job_id, "stage": stage})
            return
        time.sleep(15)
    raise TimeoutError("The earlier workload is still active at the comparison deadline.")


def main():
    if API.dataset_info(REPO).private is not True:
        raise RuntimeError("Evidence destination must be private.")
    RESULTS.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(WORK)
    shm = shutil.disk_usage("/dev/shm") if Path("/dev/shm").is_dir() else None
    metadata = {
        "profile": PROFILE, "run": RUN, "job_id": os.environ.get("JOB_ID"),
        "bundle_revision": os.environ["BUNDLE_REVISION"], "bundle_sha256": os.environ["BUNDLE_SHA256"],
        "hostname": os.uname().nodename, "kernel": os.uname().release,
        "cpu_model": next((line.partition(":")[2].strip()
                           for line in (read_file("/proc/cpuinfo") or "").splitlines()
                           if line.startswith("model name")), None),
        "cpu_quota": read_file("/sys/fs/cgroup/cpu.max"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "memory_limit_bytes": read_file("/sys/fs/cgroup/memory.max"),
        "disk_total_bytes": usage.total, "disk_free_bytes": usage.free, "started_utc": now(),
        "dev_shm_total_bytes": shm.total if shm else None,
        "dev_shm_free_bytes": shm.free if shm else None,
        "job_cap_seconds": JOB_SECONDS, "source_bucket_sha256": BUCKET_MANIFEST,
        "concurrent_jobs_at_launch": json.loads(os.environ.get("CONCURRENT_JOBS_AT_LAUNCH", "[]")),
    }
    write_json(RESULTS / "host.json", metadata)
    shutil.copyfile(WORK / "requirements-resolved.txt", RESULTS / "requirements-resolved.txt")
    upload("environment")
    try:
        wait_for_previous()
        if PROFILE == "correctness":
            correctness()
        elif PROFILE == "probe":
            probe()
        elif PROFILE == "tolerance":
            tolerance()
        elif PROFILE == "tolerance-repeat":
            tolerance_repeat()
        elif PROFILE == "tolerance-prepared-repeat":
            tolerance_prepared_repeat()
        elif PROFILE == "internvideo":
            internvideo()
        elif PROFILE == "worker-memory":
            worker_memory()
        elif PROFILE == "abc-history":
            abc_history()
        else:
            matrix()
    finally:
        append({"event": "finish", "on_host_seconds": time.monotonic() - STARTED})
        upload("final")


if __name__ == "__main__":
    main()
