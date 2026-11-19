"""Run one pinned LeRobot reader against one large remote source."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import wraps
from hashlib import sha256
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import signal
import statistics
import sys
from time import perf_counter
import traceback

from telemetry import IORecorder, ResourceSampler, Trace
from turnover import PoolTurnover

BATCH_SIZE = 8
NATIVE_TOLERANCE_S = 1e-4
MILESTONES = (1, 125, 1000)
HISTORY_OFFSETS_S = [-0.8, -0.6, -0.4, -0.2, 0.0]
PINS = {
    "baseline": ("fe1c54594f6ba275fccf210b9f7b11f0e2324b89",
                 "05cb8a76041a35681bf04cd2fc740d28c2ddfe55bcc626cdb618f7b38cb0b615"),
    "candidate": ("1a60cb4b800c2947834342752c32f064c8bf143c",
                  "17ec1e996af3e83de1e64b08b4550ff4c25914f41abf06329abbeb5c0f50a656"),
}
DATASETS_PIN = "464fe790b82a320350c9c686c1b7e409323a1cf3"
DATASETS_CACHE_SHA = "16ab95f92d6d147e0ab15de55d0a4afab2553331fa520df0e6e8ae4b58ab2447"
INPUTS = {
    "bucket": (30, 14, (3, 360, 640), ("right", "left", "top")),
    "abc": (30, 14, (3, 224, 224), ("top", "left_wrist", "right_wrist")),
    "droid": (15, 8, (3, 180, 320), ("wrist_left", "exterior_1_left", "exterior_2_left")),
}


def source_digest(module):
    return sha256(Path(module.__file__).read_bytes()).hexdigest()


def provenance(reader, streaming_module):
    dependencies = {}
    for dependency in ("datasets", "huggingface_hub", "fsspec", "torch", "torchvision", "torchcodec",
                       "pyarrow", "numpy", "httpx", "httpx2", "hf_xet", "psutil", "lerobot"):
        try:
            version = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            version = None
        dependencies[dependency] = version
    revision, expected_hash = PINS[reader]
    actual_hash = source_digest(streaming_module)
    if actual_hash != expected_hash:
        raise ValueError(f"Unexpected LeRobot streaming source hash {actual_hash}; expected {expected_hash}")
    data_source = {"revision": None, "cache_module_sha256": None}
    try:
        from datasets.filesystems import caching
    except ImportError:
        if reader == "candidate":
            raise
    else:
        actual_hash = source_digest(caching)
        if actual_hash != DATASETS_CACHE_SHA:
            raise ValueError(f"Unexpected Datasets cache source hash {actual_hash}; expected {DATASETS_CACHE_SHA}")
        data_source = {"revision": DATASETS_PIN, "cache_module_sha256": actual_hash}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": dependencies,
        "lerobot_revision": revision,
        "lerobot_streaming_module_sha256": expected_hash,
        "datasets_source": data_source,
        "hf_hub_disable_xet": os.environ.get("HF_HUB_DISABLE_XET", "unset"),
    }


def build_reader(args, dataset_class, repo_id, revision, fps, cameras):
    deltas = None if args.single_frame else {
        "action": [step / fps for step in range(100)],
        **{camera: HISTORY_OFFSETS_S if args.history else [0.0] for camera in cameras},
    }
    kwargs = {"video_backend": "torchcodec", "episode_pool_size": args.pool_size}
    if args.reader == "candidate":
        kwargs = {"buffer_size": args.pool_size}
    return dataset_class(
        repo_id,
        revision=revision,
        repo_type="bucket" if revision is None else "dataset",
        seed=args.seed,
        return_uint8=True,
        delta_timestamps=deltas,
        tolerance_s=args.tolerance_s,
        **kwargs,
    )


def check_identity(dataset, revision, fps, cameras):
    if dataset.fps != fps or set(dataset.meta.video_keys) != set(cameras):
        raise ValueError(f"Unexpected source metadata: fps={dataset.fps}, cameras={dataset.meta.video_keys!r}")
    if revision is not None and (dataset.revision != revision or dataset.meta.root.name != revision):
        raise ValueError(f"Unexpected revision or metadata snapshot: {dataset.revision!r}, {dataset.meta.root!s}")
    return {
        "video_keys": list(dataset.meta.video_keys),
        "fps": dataset.fps,
        "frames": dataset.num_frames,
        "episodes": dataset.num_episodes,
        "reader_revision": dataset.revision,
        "metadata_snapshot": dataset.meta.root.name,
        "baseline_sidecar_bytes": (
            Path(dataset._sidecar_path).stat().st_size
            if getattr(dataset, "_sidecar_path", None) is not None else None
        ),
        "reader_shards": dataset.num_shards,
        "metadata_files_on_disk": sum(1 for path in (dataset.meta.root / "meta").rglob("*") if path.is_file()),
        "metadata_logical_bytes_on_disk": sum(
            path.stat().st_size for path in (dataset.meta.root / "meta").rglob("*") if path.is_file()
        ),
    }


def shape(value):
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "length": len(value), "item": shape(value[0]) if value else None}
    return {"type": type(value).__name__}


def check_batch(batch, *, cameras, image_shape, vector_size, fps, history, single_frame):
    schema = {key: shape(value) for key, value in batch.items()}
    expected = {
        "observation.state": ([BATCH_SIZE, vector_size], "torch.float32"),
        "action": ([BATCH_SIZE, vector_size] if single_frame else [BATCH_SIZE, 100, vector_size], "torch.float32"),
    }
    if not single_frame:
        expected["action_is_pad"] = ([BATCH_SIZE, 100], "torch.bool")
    for camera in cameras:
        expected[camera] = (
            [BATCH_SIZE, len(HISTORY_OFFSETS_S), *image_shape] if history else [BATCH_SIZE, *image_shape],
            "torch.uint8",
        )
        if not single_frame:
            expected[f"{camera}_is_pad"] = ([BATCH_SIZE, len(HISTORY_OFFSETS_S) if history else 1], "torch.bool")
    errors = [
        key for key, (required_shape, required_dtype) in expected.items()
        if schema.get(key) != {"shape": required_shape, "dtype": required_dtype}
    ]
    if not single_frame:
        frame_ids = batch["frame_index"]
        offsets = HISTORY_OFFSETS_S if history else [0.0]
        required_pad = frame_ids[:, None] + frame_ids.new_tensor([round(offset * fps) for offset in offsets]) < 0
        for camera in cameras:
            mask = f"{camera}_is_pad"
            if mask not in errors and not batch[mask].equal(required_pad):
                errors.append(f"{mask}:values")
    return schema, errors


def diversity(ids, fps):
    same_episode = near_02 = near_10 = 0
    for (episode, frame), (other_episode, other_frame) in itertools.combinations(ids, 2):
        if episode != other_episode:
            continue
        same_episode += 1
        distance = abs(frame - other_frame)
        near_02 += distance <= fps * 0.2
        near_10 += distance <= fps
    counts = Counter(episode for episode, _ in ids)
    largest_share = max(counts.values()) / len(ids) if ids else None
    return (len(counts), same_episode, near_02, near_10, largest_share)


def quantiles(values):
    if not values:
        return {}
    sorted_values = sorted(values)
    result = {}
    for percentile in (50, 95, 99):
        point = (len(sorted_values) - 1) * percentile / 100
        lo = int(point)
        hi = min(lo + 1, len(sorted_values) - 1)
        result[f"p{percentile}"] = sorted_values[lo] + (point - lo) * (sorted_values[hi] - sorted_values[lo])
    return result


class Window:
    def __init__(self, start_s, start_batch):
        self.start_s = self.end_s = start_s
        self.start_batch = start_batch
        self.waits = []
        self.unique = self.usable = self.raw = 0
        self.episode_counts = []
        self.largest_shares = []
        self.pairs = self.same = self.near_02 = self.near_10 = 0

    def add(self, elapsed, wait, raw, unique, usable, stats):
        self.end_s = elapsed
        self.waits.append(wait)
        self.unique += unique
        self.usable += usable
        self.raw += raw
        self.episode_counts.append(stats[0])
        self.pairs += raw * (raw - 1) // 2
        self.same += stats[1]
        self.near_02 += stats[2]
        self.near_10 += stats[3]
        self.largest_shares.append(stats[4])

    def snapshot(self):
        seconds = self.end_s - self.start_s
        return {
            "start_after_batch": self.start_batch,
            "batches": len(self.waits),
            "elapsed_s": seconds,
            "raw_samples": self.raw,
            "unique_samples": self.unique,
            "usable_unique_samples": self.usable,
            "raw_samples_per_s": self.raw / seconds if seconds > 0 else None,
            "unique_samples_per_s": self.unique / seconds if seconds > 0 else None,
            "usable_unique_samples_per_s": self.usable / seconds if seconds > 0 else None,
            "batch_wait_s": quantiles(self.waits),
            "episode_diversity": {
                "distinct_episodes_per_batch_mean": statistics.fmean(self.episode_counts) if self.episode_counts else None,
                "distinct_episodes_per_batch": quantiles(self.episode_counts),
                "largest_episode_share_mean": statistics.fmean(self.largest_shares) if self.largest_shares else None,
                "largest_episode_share": quantiles(self.largest_shares),
                "unordered_pairs": self.pairs,
                "same_episode_pairs": self.same,
                "near_0p2s_pairs": self.near_02,
                "near_1p0s_pairs": self.near_10,
            },
        }


def atomic_json(path, value):
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(value, indent=2) + "\n")
    tmp_path.replace(path)


def redact(text):
    text = re.sub(r"https?://[^\s\"'<>]+", lambda m: m[0].split("?", 1)[0], text)
    return re.sub(r"\bhf_[a-zA-Z0-9]{10,}\b", "[redacted]", text)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader", choices=PINS, required=True)
    parser.add_argument("--dataset", choices=INPUTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    workload = parser.add_mutually_exclusive_group()
    workload.add_argument("--history", action="store_true", help="Add ABC camera frames from -0.8 s through 0 s in 0.2 s steps.")
    workload.add_argument("--single-frame", action="store_true", help="Request no deltas for a diagnostic run.")
    parser.add_argument("--cache-state", choices=("cold", "warm", "unclassified"), default="unclassified")
    parser.add_argument("--warm-batches", type=positive_int, default=0)
    parser.add_argument("--max-batches", type=positive_int, default=125000)
    parser.add_argument("--max-seconds", type=positive_int)
    parser.add_argument("--min-steady-batches", type=positive_int, default=125)
    parser.add_argument("--pool-size", type=positive_int, default=32)
    parser.add_argument("--workers", type=int, choices=(0,), default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tolerance-s", type=float, default=NATIVE_TOLERANCE_S,
        help="Maximum timestamp difference in seconds for decoded frames (default: 0.0001).",
    )
    args = parser.parse_args()
    if not math.isfinite(args.tolerance_s) or args.tolerance_s < 0:
        parser.error("--tolerance-s requires a finite nonnegative value")
    if args.history and args.dataset != "abc":
        parser.error("--history requires --dataset abc")
    if args.warm_batches and args.cache_state != "warm":
        parser.error("--warm-batches requires --cache-state warm")
    return args


def main():
    args = parse_args()
    t0 = perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace = Trace(args.output.with_suffix(".events.jsonl"), t0)
    io = IORecorder(trace)
    resources = ResourceSampler(trace, io)
    pool = PoolTurnover(args.reader, trace)
    workload = "single_frame" if args.single_frame else "action100_history_0p8s" if args.history else "action100"
    fps, vector_size, image_shape, suffixes = INPUTS[args.dataset]
    cameras = [f"observation.images.{suffix}" for suffix in suffixes]
    progress = {"batches": 0, "raw_samples": 0, "unique_samples": 0, "usable_unique_samples": 0, "duplicates": 0}
    contract = {"invalid_batches": 0, "errors_by_field": {}, "first_schema": None}
    timings = {"milestones": {}}
    result = {
        "schema_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "reader": args.reader,
        "dataset": args.dataset,
        "workload": workload,
        "cache": args.cache_state,
        "config": {
            "max_batches": args.max_batches,
            "max_seconds": args.max_seconds,
            "min_steady_batches": args.min_steady_batches,
            "warm_batches": args.warm_batches,
            "batch_size": BATCH_SIZE,
            "workers": 0,
            "torch_num_threads": 1,
            "pool_size": args.pool_size,
            "seed_requested": args.seed,
            "seed_effective": 42 if args.reader == "candidate" else args.seed,
            "camera_offsets_s": HISTORY_OFFSETS_S if args.history else [0.0],
            "action_steps": 1 if args.single_frame else 100,
            "tolerance_s": args.tolerance_s,
        },
        "progress": progress,
        "contract": contract,
        "timings": timings,
    }
    windows = {}
    failures = Counter()
    seen = set()
    usable_seen = set()
    resource_started = False
    iterator = None

    def dump(status):
        result["status"] = status
        result["elapsed_process_s"] = perf_counter() - t0
        result["turnover"] = pool.snapshot()
        result["io"] = io.snapshot()
        result["resources"] = resources.latest
        result["windows"] = {name: window.snapshot() for name, window in windows.items()}
        contract["errors_by_field"] = dict(failures)
        atomic_json(args.output, result)

    def finish_signal(signum, _frame):
        raise InterruptedError(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, finish_signal)
    dump("started")

    try:
        # Imports and source verification precede the timed reader construction.
        import numpy as np
        import torch
        from torch.utils.data import DataLoader

        from catalog import SOURCES
        from lerobot.datasets import streaming_dataset

        repo_id, revision = SOURCES[args.dataset]
        result["source"] = {
            "repo": repo_id,
            "revision": revision,
            "candidate_uses_video_url_revision_adapter": args.reader == "candidate" and revision is not None,
        }
        result["environment"] = provenance(args.reader, streaming_dataset)
        torch.set_num_threads(1)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        io.install_http()
        io.install_workload(args.reader, repo_id, revision)
        dataset_class = streaming_dataset.StreamingLeRobotDataset
        pool.install(dataset_class)
        native_iter = dataset_class.__iter__
        measured_start = None

        @wraps(native_iter)
        def first_sample(dataset):
            source_iterator = native_iter(dataset)
            try:
                first_item = next(source_iterator)
            except StopIteration:
                return
            try:
                if measured_start is not None and "first_sample_s" not in timings:
                    timings["first_sample_s"] = perf_counter() - measured_start
                    trace.write("first_sample", measurement_elapsed_s=timings["first_sample_s"])
                yield first_item
                del first_item
                yield from source_iterator
            finally:
                source_iterator.close()

        dataset_class.__iter__ = first_sample
        resources.start()
        resource_started = True

        def make_dataset():
            return build_reader(args, dataset_class, repo_id, revision, fps, cameras)

        def make_iterator(dataset):
            from lerobot.utils.collate import lerobot_collate_fn

            # Match the training entry point. ABC stores variable-length language events.
            collate = lerobot_collate_fn if dataset.meta.has_language_columns else None
            result["config"]["collate"] = "lerobot_collate_fn" if collate else "default_collate"
            return iter(DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=collate))

        if args.warm_batches:
            trace.write("phase", phase="prewarm")
            dump("prewarming")
            begin_warmup = perf_counter()
            prewarmed_dataset = make_dataset()
            check_identity(prewarmed_dataset, revision, fps, cameras)
            iterator = make_iterator(prewarmed_dataset)
            for batch_number in range(1, args.warm_batches + 1):
                batch = next(iterator)
                trace.write("prewarm_batch", batch=batch_number, episodes=batch["episode_index"].tolist())
                del batch
            del iterator, prewarmed_dataset
            iterator = None
            result["prewarm"] = {"elapsed_s": perf_counter() - begin_warmup, "io": io.snapshot()}
            io.reset()
            resources.reset()
            pool.reset()

        trace.write("phase", phase="measurement")
        measured_start = perf_counter()
        timings["measurement_started_elapsed_process_s"] = measured_start - t0
        dump("constructing")
        dataset = make_dataset()
        timings["constructor_s"] = perf_counter() - measured_start
        result["source"]["observed"] = check_identity(dataset, revision, fps, cameras)
        iterator_start = perf_counter()
        iterator = make_iterator(dataset)
        timings["iterator_setup_s"] = perf_counter() - iterator_start
        dump("iterating")
        windows["to_1000"] = Window(0.0, 0)
        status, stop_reason = "ok", "max_batches"
        for batch_number in range(1, args.max_batches + 1):
            wait_start = perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                stop_reason = "source_exhausted"
                break
            now = perf_counter()
            elapsed = now - measured_start
            wait_s = now - wait_start
            schema, errors = check_batch(
                batch, cameras=cameras, image_shape=image_shape, vector_size=vector_size, fps=fps,
                history=args.history, single_frame=args.single_frame,
            )
            if contract["first_schema"] is None:
                contract["first_schema"] = schema
            contract["invalid_batches"] += bool(errors)
            failures.update(errors)
            ids = list(zip(batch["episode_index"].tolist(), batch["frame_index"].tolist(), strict=True))
            raw = len(ids)
            new = set(ids) - seen
            usable_new = set(ids) - usable_seen if not errors else set()
            seen.update(new)
            usable_seen.update(usable_new)
            progress.update(
                batches=batch_number, raw_samples=progress["raw_samples"] + raw, unique_samples=len(seen),
                usable_unique_samples=len(usable_seen), duplicates=progress["duplicates"] + raw - len(new),
            )
            stats = diversity(ids, fps)
            trace.write(
                "batch", batch=batch_number, measurement_elapsed_s=elapsed, wait_s=wait_s,
                ids=ids, unique=len(new), usable_unique=len(usable_new),
                distinct_episodes=stats[0], same_episode_pairs=stats[1], near_0p2s_pairs=stats[2],
                near_1p0s_pairs=stats[3], largest_episode_share=stats[4], contract_errors=errors,
            )
            for name, window in windows.items():
                if name != "to_1000" or batch_number <= 1000:
                    window.add(elapsed, wait_s, raw, len(new), len(usable_new), stats)
            pool.batch(batch_number, new)
            if batch_number in MILESTONES:
                timings["milestones"][str(batch_number)] = {"elapsed_s": elapsed, **progress, "io": io.snapshot()}
                dump("running")
                print(
                    f"{args.reader} {args.dataset} {workload} batch {batch_number}: "
                    f"{elapsed:.3f} s, {len(usable_seen)} usable unique samples",
                    flush=True,
                )
            if batch_number == 125:
                windows["after_batch_125"] = Window(elapsed, batch_number)
            if len(pool.refill_batches) == 2 and "after_refill_2" not in windows:
                windows["after_refill_2"] = Window(elapsed, batch_number)
                dump("running")
            if (batch_number % 100 == 0 and batch_number < 1000) or (batch_number % 1000 == 0 and batch_number != 1000):
                dump("running")
            del batch
            if (
                batch_number >= 1000
                and "after_refill_2" in windows
                and len(windows["after_refill_2"].waits) >= args.min_steady_batches
            ):
                stop_reason = "milestones_and_steady_window"
                break
            if args.max_seconds is not None and elapsed >= args.max_seconds:
                stop_reason = "max_seconds"
                break

        if contract["invalid_batches"]:
            status = "invalid_output"
        elif not progress["batches"]:
            status = "empty"
        result["stop_reason"] = stop_reason
        trace.write("terminal", status=status, stop_reason=stop_reason, **progress)
        dump(status)
        return 0
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "error"
        result["error"] = redact(f"{type(exc).__name__}: {exc}")
        result["traceback"] = redact(traceback.format_exc())
        trace.write("terminal", status=status, error=result["error"], **progress)
        dump(status)
        print(result["traceback"], file=sys.stderr, flush=True)
        return 130 if status == "interrupted" else 1
    finally:
        if resource_started:
            resources.stop()
        dump(result["status"])
        io.write_files(args.output.with_suffix(".files.jsonl.gz"))
        # Close the iterator before the trace because shutdown can finish in-flight reads.
        del iterator
        trace.close()


if __name__ == "__main__":
    sys.exit(main())
