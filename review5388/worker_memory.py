"""Measure native baseline DataLoader process-tree memory or sidecar receipt traffic."""

import argparse
import json
import math
import signal
import subprocess
import sys
import traceback
from contextlib import ExitStack
from hashlib import file_digest
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter
from unittest.mock import patch

from benchmark import (
    BATCH_SIZE,
    INPUTS,
    NATIVE_TOLERANCE_S,
    atomic_json,
    provenance,
    redact,
    shape,
)
from telemetry import Trace, cgroup_value

MILESTONES = (1, 125)
PROC = Path("/proc")
KIB = 1024


def network_bytes(path=PROC / "net/dev"):
    """Read non-loopback namespace counters. They include traffic from other processes."""
    try:
        rx = tx = 0
        for line in path.read_text().splitlines()[2:]:
            interface, stats = line.split(":", 1)
            if interface.strip() != "lo":
                fields = stats.split()
                rx += int(fields[0])
                tx += int(fields[8])
        return {"rx": rx, "tx": tx}
    except (OSError, ValueError, IndexError):
        return None


def network_delta(before, after):
    if (
        before is None
        or after is None
        or any(after[key] < before[key] for key in before)
    ):
        return None
    return {key: after[key] - before[key] for key in before}


def proc_memory(path, mapped_path=None):
    """Read total RSS/PSS from smaps_rollup or selected mappings from smaps."""
    output = {"Rss": 0, "Pss": 0}
    found = set()
    matched = mapped_path is None
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if mapped_path is not None and "-" in line.partition(" ")[0]:
                    fields = line.split(maxsplit=5)
                    name = (
                        fields[5].strip().removesuffix(" (deleted)")
                        if len(fields) == 6
                        else ""
                    )
                    matched = name == mapped_path
                if not matched:
                    continue
                key, _, value = line.partition(":")
                if key in output:
                    output[key] += int(value.split()[0]) * KIB
                    found.add(key)
    except (OSError, ValueError, IndexError, UnicodeError):
        return None
    # An available smaps file can correctly report no sidecar mappings yet.
    if mapped_path is None and found != output.keys():
        return None
    return {"rss_bytes": output["Rss"], "pss_bytes": output["Pss"]}


class Monitor:
    def __init__(self, trace, interval, maps_interval, workers):
        self.trace = trace
        self.interval = interval
        self.maps_interval = maps_interval
        self.last_mapping = -math.inf
        self.expected_workers = workers
        self.done = Event()
        self.lock = Lock()
        self.thread = Thread(target=self.run, name="worker-memory-monitor", daemon=True)
        self.worker_pids = set()
        self.observed_pids = set()
        self.mapped_path = None
        self.network_start = None
        self.latest = {}
        self.peaks = {
            "rss_sampled_peak_bytes": None,
            "pss_sampled_peak_bytes": None,
            "mapped_sidecar_rss_sampled_peak_bytes": None,
            "mapped_sidecar_pss_sampled_peak_bytes": None,
            "cgroup_sampled_peak_bytes": None,
            "active_workers_sampled_peak": 0,
            "samples": 0,
            "complete_pss_samples": 0,
            "complete_mapped_samples": 0,
            "rss_all_workers_sampled_peak_bytes": None,
            "pss_all_workers_sampled_peak_bytes": None,
            "pss_all_workers_samples": 0,
        }

    def register(self, workers):
        with self.lock:
            self.worker_pids = {
                worker.pid for worker in workers if worker.pid is not None
            }

    def sample(self):
        import psutil

        now = perf_counter()
        with self.lock:
            worker_pids = self.worker_pids.copy()
            mapped_path = self.mapped_path
        scan_mappings = (
            mapped_path is not None
            and self.maps_interval > 0
            and now - self.last_mapping >= self.maps_interval
        )
        if scan_mappings:
            self.last_mapping = now
        root = psutil.Process()
        processes = []
        for process in (root, *root.children(recursive=True)):
            pid = process.pid
            role = (
                "parent"
                if pid == root.pid
                else "worker"
                if pid in worker_pids
                else "descendant"
            )
            try:
                if process.status() == psutil.STATUS_ZOMBIE:
                    continue
                parsed = proc_memory(PROC / str(pid) / "smaps_rollup")
                if parsed is None:
                    parsed = {"rss_bytes": process.memory_info().rss, "pss_bytes": None}
                item = {"pid": pid, "role": role, **parsed}
                if scan_mappings:
                    mapping = proc_memory(PROC / str(pid) / "smaps", mapped_path)
                    item["mapped_sidecar"] = mapping
                processes.append(item)
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                processes.append(
                    {"pid": pid, "role": role, "rss_bytes": None, "pss_bytes": None}
                )
        active_workers = sum(item["role"] == "worker" for item in processes)
        complete_rss = bool(processes) and all(
            item["rss_bytes"] is not None for item in processes
        )
        complete_pss = bool(processes) and all(
            item["pss_bytes"] is not None for item in processes
        )
        complete_maps = (
            bool(processes)
            and scan_mappings
            and all(item.get("mapped_sidecar") is not None for item in processes)
        )
        memory = {
            "rss_sum_bytes": sum(item["rss_bytes"] for item in processes)
            if complete_rss
            else None,
            "pss_sum_bytes": sum(item["pss_bytes"] for item in processes)
            if complete_pss
            else None,
            "mapped_sidecar_rss_sum_bytes": sum(
                item["mapped_sidecar"]["rss_bytes"] for item in processes
            )
            if complete_maps
            else None,
            "mapped_sidecar_pss_sum_bytes": sum(
                item["mapped_sidecar"]["pss_bytes"] for item in processes
            )
            if complete_maps
            else None,
            "cgroup_current_bytes": cgroup_value("memory.current"),
            "cgroup_lifetime_peak_bytes": cgroup_value("memory.peak"),
            "active_workers": active_workers,
            "processes": processes,
            "namespace_wire_bytes": network_delta(self.network_start, network_bytes()),
        }
        fields = {
            "rss_sum_bytes": "rss_sampled_peak_bytes",
            "pss_sum_bytes": "pss_sampled_peak_bytes",
            "mapped_sidecar_rss_sum_bytes": "mapped_sidecar_rss_sampled_peak_bytes",
            "mapped_sidecar_pss_sum_bytes": "mapped_sidecar_pss_sampled_peak_bytes",
            "cgroup_current_bytes": "cgroup_sampled_peak_bytes",
            "active_workers": "active_workers_sampled_peak",
        }
        with self.lock:
            self.latest = memory
            self.observed_pids.update(
                item["pid"] for item in processes if item["role"] == "worker"
            )
            self.peaks["samples"] += 1
            self.peaks["complete_pss_samples"] += complete_pss
            self.peaks["complete_mapped_samples"] += complete_maps
            for key, peak_key in fields.items():
                if memory[key] is not None:
                    self.peaks[peak_key] = max(self.peaks[peak_key] or 0, memory[key])
            if active_workers == self.expected_workers:
                self.peaks["pss_all_workers_samples"] += complete_pss
                for metric in ("rss", "pss"):
                    value = memory[f"{metric}_sum_bytes"]
                    key = f"{metric}_all_workers_sampled_peak_bytes"
                    if value is not None:
                        self.peaks[key] = max(self.peaks[key] or 0, value)
        self.trace.write("memory", **memory)

    def run(self):
        while not self.done.wait(self.interval):
            try:
                self.sample()
            except (OSError, RuntimeError) as exc:
                self.trace.write("memory_sample_error", error=redact(str(exc)))

    def snapshot(self):
        with self.lock:
            return {
                **self.peaks,
                "configured_worker_pids": sorted(self.worker_pids),
                "sampled_worker_pids": sorted(self.observed_pids),
                "last_sample": self.latest,
                "rss_basis": "process_tree_sum; repeated shared pages can contribute once per process",
                "pss_basis": "sum of Linux smaps_rollup Pss for a complete sampled process tree",
                "mapped_basis": "sparse smaps sums for the native derived sidecar path only",
            }

    def start(self):
        self.network_start = network_bytes()
        self.sample()
        self.thread.start()

    def stop(self):
        self.done.set()
        self.thread.join()


def sidecar_input(args):
    if args.fixture_root:
        return None, {}
    from lerobot.utils.constants import HF_LEROBOT_HOME

    if args.sidecar_manifest:
        receipt = json.loads(args.sidecar_manifest.read_text())
        if receipt["dataset"] != args.dataset:
            raise ValueError(
                "Sidecar receipt dataset does not match the benchmark dataset"
            )
        transfer = receipt["private_transfer"]
        relative = Path(transfer["relative_cache_path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[:1] != ("streaming-sidecars",)
        ):
            raise ValueError("Invalid relative sidecar path in receipt")
        path = (HF_LEROBOT_HOME / relative).resolve()
        expected_sha = transfer["sha256"]
    else:
        path, expected_sha = args.sidecar.resolve(), args.sidecar_sha256
        transfer = {}
    begin_verify = perf_counter()
    with path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    verify_seconds = perf_counter() - begin_verify
    if digest != expected_sha.lower():
        raise ValueError("Prepared sidecar SHA-256 does not match")
    size = path.stat().st_size
    if transfer and transfer["size_bytes"] != size:
        raise ValueError("Prepared sidecar size does not match receipt")
    return path, {
        "sha256": digest,
        "archive_bytes": size,
        "private_transfer": transfer or None,
        "archive_recheck_seconds_before_measurement": verify_seconds,
    }


def verify_batch(batch, specs, lengths):
    import torch

    schema = {key: shape(batch.get(key)) for key in specs}
    errors = [key for key, expected in specs.items() if schema[key] != expected]
    ids = list(
        zip(batch["episode_index"].tolist(), batch["frame_index"].tolist(), strict=True)
    )
    valid_ids = all(
        0 <= episode < len(lengths) and 0 <= frame < lengths[episode]
        for episode, frame in ids
    )
    if not valid_ids:
        errors.append("anchor_out_of_bounds")
    if valid_ids and "action_is_pad" not in errors:
        episode_lengths = batch["frame_index"].new_tensor(
            [int(lengths[episode]) for episode, frame in ids]
        )
        required = (
            batch["frame_index"][:, None] + torch.arange(100)
            >= episode_lengths[:, None]
        )
        if not batch["action_is_pad"].equal(required):
            errors.append("action_is_pad:values")
    for key in specs:
        if (
            key.endswith("_is_pad")
            and key != "action_is_pad"
            and key not in errors
            and batch[key].any()
        ):
            errors.append(f"{key}:values")
    return schema, ids, errors


def expected_schema(meta):
    def spec(dimensions, dtype):
        return {"shape": [BATCH_SIZE, *dimensions], "dtype": dtype}

    expected = {
        "observation.state": spec(
            meta.features["observation.state"]["shape"], "torch.float32"
        ),
        "action": spec([100, *meta.features["action"]["shape"]], "torch.float32"),
        "action_is_pad": spec([100], "torch.bool"),
    }
    for camera in meta.video_keys:
        height, width, channels = meta.features[camera]["shape"]
        expected[camera] = spec([channels, height, width], "torch.uint8")
        expected[f"{camera}_is_pad"] = spec([1], "torch.bool")
    return expected


def measure(args):
    t0 = perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace = Trace(args.output.with_suffix(".events.jsonl"), t0)
    monitor = Monitor(trace, args.sample_interval_s, args.maps_interval_s, args.workers)
    progress = {
        "batches": 0,
        "raw_samples": 0,
        "unique_samples": 0,
        "duplicates": 0,
        "invalid_batches": 0,
    }
    result = {
        "dataset": args.dataset if not args.fixture_root else "fixture",
        "reader": "baseline",
        "workload": "action100",
        "status": "started",
        "config": {
            "workers": args.workers,
            "max_batches": args.max_batches,
            "batch_size": BATCH_SIZE,
            "prefetch_factor": args.prefetch_factor,
            "persistent_workers": True,
            "context": "spawn",
            "tolerance_s": NATIVE_TOLERANCE_S,
            "seed": 42,
            "return_uint8": True,
            "action_steps": 100,
            "sample_interval_s": args.sample_interval_s,
            "maps_interval_s": args.maps_interval_s,
        },
        "progress": progress,
        "milestones": {},
    }
    started = False
    it = None
    signal.signal(signal.SIGTERM, signal_interrupt)
    atomic_json(args.output, result)
    try:
        import torch
        from catalog import SOURCES
        from lerobot.datasets import streaming_dataset, streaming_sidecar
        from lerobot.streaming.sidecar_utils import _cache_path
        from lerobot.utils.collate import lerobot_collate_fn

        result["environment"] = provenance("baseline", streaming_dataset)
        torch.set_num_threads(1)
        expected_path, sidecar = sidecar_input(args)
        result["sidecar"] = sidecar
        if expected_path is not None:
            mapped_path = _cache_path(expected_path)
            result["sidecar"]["mapped_index_existed_before_construction"] = (
                mapped_path.is_file()
            )
            monitor.mapped_path = str(mapped_path.resolve())
        if args.fixture_root is not None:
            repo, revision, root, repo_type = (
                "fixture/stream5388",
                None,
                args.fixture_root,
                "dataset",
            )
            info = json.loads((root / "meta/info.json").read_text())
            fps = info["fps"]
            cameras = [
                key
                for key, spec in info["features"].items()
                if spec["dtype"] == "video"
            ]
        else:
            repo, revision = SOURCES[args.dataset]
            root, repo_type = None, "dataset" if revision else "bucket"
            fps, _, _, suffixes = INPUTS[args.dataset]
            cameras = [f"observation.images.{suffix}" for suffix in suffixes]
        deltas = {
            "action": [step / fps for step in range(100)],
            **{camera: [0.0] for camera in cameras},
        }
        result["source"] = {
            "repo": repo,
            "revision": revision,
            "repo_type": repo_type,
            "cameras": cameras,
            "fps": fps,
        }

        def prepared_only(*unused_args, **unused_kwargs):
            raise RuntimeError(
                "The verified local sidecar did not match the reader's native source specification"
            )

        monitor.start()
        started = True
        measurement_start = perf_counter()
        result["status"] = "constructing"
        atomic_json(args.output, result)
        with ExitStack() as guards:
            if expected_path is not None:
                for name in ("build_mp4_sidecar", "download_published_sidecar"):
                    guards.enter_context(
                        patch.object(streaming_sidecar, name, prepared_only)
                    )
            dataset = streaming_dataset.StreamingLeRobotDataset(
                repo,
                root=root,
                revision=revision,
                repo_type=repo_type,
                video_backend="torchcodec",
                return_uint8=True,
                seed=42,
                delta_timestamps=deltas,
                tolerance_s=NATIVE_TOLERANCE_S,
            )
        result["constructor_s"] = perf_counter() - measurement_start
        actual_path = Path(dataset._sidecar_path).resolve()
        if expected_path is not None and actual_path != expected_path:
            raise ValueError("The native reader selected a different sidecar")
        if revision and (
            dataset.revision != revision or dataset.meta.root.name != revision
        ):
            raise ValueError("The native reader opened a different metadata revision")
        if (
            dataset.fps != fps
            or set(dataset.meta.video_keys) != set(cameras)
            or len(cameras) != 3
        ):
            raise ValueError("Source does not match the expected three-camera workload")
        actual_map = _cache_path(actual_path)
        with monitor.lock:
            monitor.mapped_path = str(actual_map.resolve())
        result["sidecar"].update(
            archive_bytes=actual_path.stat().st_size,
            derived_index_bytes=actual_map.stat().st_size,
            archive_name=actual_path.name,
            derived_index_name=actual_map.name,
            origin="verified_prepared_archive"
            if expected_path is not None
            else "fixture",
        )
        result["source"].update(
            episodes=dataset.num_episodes, frames=dataset.num_frames
        )
        lengths = dataset._episode_frame_counts
        specs = expected_schema(dataset.meta)
        collate = lerobot_collate_fn if dataset.meta.has_language_columns else None
        result["config"].update(
            rank_episode_pool=dataset.episode_pool_size,
            per_worker_pool=max(1, round(dataset.episode_pool_size / args.workers)),
            rank_byte_budget=dataset.byte_budget,
            per_worker_byte_budget=max(1, dataset.byte_budget // args.workers),
            collate="lerobot_collate_fn" if collate else "default_collate",
            max_fetch_threads_per_worker=dataset.max_num_shards,
            decode_threads_per_worker=dataset.decode_threads,
            decoded_queue_per_worker=dataset.decoded_queue_size,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=args.workers,
            pin_memory=False,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
            multiprocessing_context="spawn",
            generator=torch.Generator().manual_seed(42),
            collate_fn=collate,
        )
        it = iter(loader)
        monitor.register(it._workers)
        result["status"] = "iterating"
        atomic_json(args.output, result)
        seen = set()
        for index in range(1, args.max_batches + 1):
            before = perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                break
            delivered = perf_counter()
            schema, ids, errors = verify_batch(batch, specs, lengths)
            new = set(ids) - seen
            seen.update(new)
            progress.update(
                batches=index,
                raw_samples=progress["raw_samples"] + len(ids),
                unique_samples=len(seen),
                duplicates=progress["duplicates"] + len(ids) - len(new),
                invalid_batches=progress["invalid_batches"] + bool(errors),
            )
            event = {
                "batch": index,
                "measurement_elapsed_s": delivered - measurement_start,
                "wait_s": delivered - before,
                "ids": ids,
                "new_anchors": len(new),
                "errors": errors,
            }
            trace.write("batch", **event)
            if index == 1:
                result["first_batch_schema"] = schema
            if index in MILESTONES:
                result["milestones"][str(index)] = {
                    "elapsed_s": event["measurement_elapsed_s"],
                    **progress,
                }
                result["status"] = "running"
                result["memory"] = monitor.snapshot()
                atomic_json(args.output, result)
            del batch
        result["status"] = (
            "ok"
            if (
                progress["batches"] == args.max_batches
                and not progress["duplicates"]
                and not progress["invalid_batches"]
            )
            else "incomplete_or_invalid"
        )
    except BaseException as exc:  # noqa: BLE001 - keep a terminal record for worker failures and interrupts
        result["status"] = (
            "interrupted"
            if isinstance(exc, (InterruptedError, KeyboardInterrupt))
            else "error"
        )
        result["error"] = redact(f"{type(exc).__name__}: {exc}")
        print(redact(traceback.format_exc()), file=sys.stderr, flush=True)
    finally:
        if started:
            monitor.stop()
            monitor.sample()
        result["memory"] = monitor.snapshot()
        trace.write("terminal", status=result["status"], **progress)
        atomic_json(args.output, result)
        if it is not None:
            it._shutdown_workers()
        trace.close()
    return (
        0
        if result["status"] == "ok"
        else 130
        if result["status"] == "interrupted"
        else 1
    )


def signal_interrupt(_signum, _frame):
    raise InterruptedError("Received SIGTERM")


def receipt(args):
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("Supply the publisher command after --")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    before, t0 = network_bytes(), perf_counter()
    process = subprocess.run(command, check=False)
    elapsed, after = perf_counter() - t0, network_bytes()
    result = {
        "status": "ok" if process.returncode == 0 else "error",
        "return_code": process.returncode,
        "command_elapsed_s": elapsed,
        "namespace_wire_bytes": network_delta(before, after),
        "wire_scope": "non-loopback namespace counters around the publisher command; includes protocol and other traffic",
    }
    return_code = process.returncode
    if return_code == 0 and args.receipt_manifest is not None:
        try:
            transfer = json.loads(args.receipt_manifest.read_text())["private_transfer"]
            result["logical_archive_bytes"] = transfer["size_bytes"]
            result["verified_archive_sha256"] = transfer["sha256"]
            result["consumer_prep_seconds"] = transfer["consumer_prep_seconds"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            result["status"] = "receipt_manifest_error"
            result["error"] = redact(str(exc))
            return_code = 1
    atomic_json(args.output, result)
    return return_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command_name", required=True)
    memory = commands.add_parser("measure")
    inputs = memory.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--dataset", choices=INPUTS)
    inputs.add_argument(
        "--fixture-root",
        type=Path,
        help="Run the same loader on a local three-camera fixture.",
    )
    memory.add_argument("--workers", type=int, choices=(1, 4), required=True)
    memory.add_argument("--output", type=Path, required=True)
    memory.add_argument("--sidecar-manifest", type=Path)
    memory.add_argument("--sidecar", type=Path)
    memory.add_argument("--sidecar-sha256")
    memory.add_argument("--max-batches", type=int, default=125)
    memory.add_argument("--prefetch-factor", type=int, default=4)
    memory.add_argument("--sample-interval-s", type=float, default=1.0)
    memory.add_argument(
        "--maps-interval-s",
        type=float,
        default=5.0,
        help="Set 0 to skip individual mapping scans.",
    )
    wire = commands.add_parser("receipt")
    wire.add_argument("--output", type=Path, required=True)
    wire.add_argument("--receipt-manifest", type=Path)
    wire.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command_name == "receipt":
        return receipt(args)
    if args.dataset and not (
        bool(args.sidecar_manifest) ^ bool(args.sidecar and args.sidecar_sha256)
    ):
        parser.error(
            "Supply either --sidecar-manifest or both --sidecar and --sidecar-sha256."
        )
    if args.max_batches < 1 or args.prefetch_factor < 1:
        parser.error("Batch count and prefetch factor must be positive.")
    if not math.isfinite(args.sample_interval_s) or args.sample_interval_s <= 0:
        parser.error("Sample interval must be finite and positive.")
    if not math.isfinite(args.maps_interval_s) or args.maps_interval_s < 0:
        parser.error("Mapping sample interval must be finite and nonnegative.")
    return measure(args)


if __name__ == "__main__":
    sys.exit(main())
