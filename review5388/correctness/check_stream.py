"""Check finite stream contents and consumed-position restart on the fixture."""

import argparse
import json
import os
import time
import traceback
from collections import Counter
from contextlib import closing
from itertools import islice
from pathlib import Path

import datasets
import numpy as np
import torch

from fixture import ACTION, CAMERAS, FPS, STATE, TASK, vectors
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

DELTAS = {ACTION: list(range(100)), STATE: [-2, 0, 2], **{cam: [-2, 0, 3] for cam in CAMERAS}}
BATCH_SIZE = 4
SELECTION = [1, 4, 7, 9, 12, 15]


def mode_deltas(mode):
    if mode == "anchors":
        return {}
    if mode == "numeric":
        return {key: DELTAS[key] for key in (ACTION, STATE)}
    if mode == "positive":
        return {ACTION: [1, 2]}
    if mode == "video":
        return {key: DELTAS[key] for key in CAMERAS}
    return DELTAS


class Oracle:
    def __init__(self, root, mode, selection):
        episodes = json.loads((root / "oracle/episodes.json").read_text())
        self.rows = {row["episode_index"]: row for row in episodes}
        self.ids = sorted(
            index for row in episodes if selection is None or row["episode_index"] in selection
            for index in range(row["dataset_from_index"], row["dataset_to_index"])
        )
        self.root = root
        self.deltas = mode_deltas(mode)
        self.errors = Counter()
        self.examples = {}
        self.max_pixel_error = 0
        self.videos = {}

    def check(self, key, passed, index, failure):
        if not passed:
            self.errors[key] += 1
            self.examples.setdefault(key, {"index": index, "detail": failure})

    def sample(self, sample):
        index, episode, frame = [int(sample[key]) for key in ("index", "episode_index", "frame_index")]
        row = self.rows[episode]
        length, start = row["length"], row["dataset_from_index"]
        self.check("identity", index == start + frame and 0 <= frame < length, index, [episode, frame])
        self.check("task", sample["task"] == TASK, index, str(sample["task"]))
        self.check("timestamp", abs(float(sample["timestamp"]) - frame / FPS) < 1e-6, index, str(sample["timestamp"]))
        for key in (ACTION, STATE, *CAMERAS):
            offsets = np.array(self.deltas.get(key, [0]))
            requested = frame + offsets
            clipped = requested.clip(0, length - 1)
            if key in CAMERAS:
                camera = CAMERAS.index(key)
                file_index = row[f"videos/{key}/file_index"]
                video_offset = round(row[f"videos/{key}/from_timestamp"] * FPS)
                video = self.videos.get((file_index, camera))
                if video is None:
                    video = np.load(self.root / f"oracle/{file_index:03}-{camera}.npy", mmap_mode="r")
                    self.videos[(file_index, camera)] = video
                expected = np.moveaxis(video[video_offset + clipped], -1, 1)
            else:
                expected = vectors(start + clipped, key)
            if key not in self.deltas:
                expected = expected[0]
            value = sample.get(key)
            if value is None:
                self.check(key, False, index, "field missing")
                continue
            actual = np.asarray(value)
            detail = [list(actual.shape), list(expected.shape), str(actual.dtype), str(expected.dtype)]
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                self.check(key, False, index, detail)
            elif key in CAMERAS:
                max_err = int(np.abs(actual.astype(np.int16) - expected).max())
                self.max_pixel_error = max(self.max_pixel_error, max_err)
                self.check(key, max_err <= 4, index, {"max_absolute_rgb8_difference": max_err})
            else:
                self.check(key, np.array_equal(actual, expected), index, "numeric values differ")

            if key in self.deltas:
                actual_mask = sample.get(f"{key}_is_pad")
                expected_mask = requested != clipped
                self.check(
                    f"{key}_is_pad", actual_mask is not None and np.array_equal(actual_mask, expected_mask),
                    index, "field missing" if actual_mask is None else {
                        "actual": np.asarray(actual_mask).tolist(), "expected": expected_mask.tolist()
                    },
                )
        return index


def dataset(args, repeat=False, *, seed=1000, shuffle=True):
    is_baseline = args.variant == "baseline"
    kwargs = {"seed": seed, "shuffle": shuffle, "return_uint8": True, "buffer_size": 4}
    if is_baseline:
        kwargs.update(repeat=repeat)
    selection = SELECTION if args.selection else None
    offsets = {key: [offset / FPS for offset in offsets] for key, offsets in mode_deltas(args.mode).items()}
    return StreamingLeRobotDataset(
        str(args.root) if args.local_id else "fixture/stream5388",
        root=args.root, episodes=selection, delta_timestamps=offsets or None, **kwargs,
    )


def loader(dataset, workers, batch_size):
    kwargs = {"multiprocessing_context": "spawn", "prefetch_factor": 3} if workers else {}
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers, **kwargs)


def finite(args, result):
    oracle = Oracle(args.root, args.mode, SELECTION if args.selection else None)
    result["ids"] = []
    data = dataset(args)
    result["dataset_num_shards"] = getattr(data, "num_shards", None)
    try:
        for item in loader(data, args.workers, None):
            result["ids"].append(oracle.sample(item))
    finally:
        ids = result["ids"]
        result["count"] = len(ids)
        result["duplicates"] = sum(value - 1 for value in Counter(ids).values())
        result["expected_full"] = len(oracle.ids)
        result["missing_from_full"] = sorted(set(oracle.ids) - set(ids))
        result["unexpected"] = sorted(set(ids) - set(oracle.ids))
        result["checks_failed"] = dict(oracle.errors)
        result["first_failure"] = oracle.examples
        result["max_pixel_error"] = oracle.max_pixel_error


def batches(dataloader, count, pause_after_first=False):
    iterator = iter(dataloader)
    try:
        result = []
        for batch in islice(iterator, count):
            result.append([int(index) for index in batch["index"]])
            if pause_after_first and len(result) == 1:
                time.sleep(0.6)
        return result
    finally:
        if hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()


def restart(args, result):
    original = dataset(args, repeat=True)
    result["has_state_dict"] = hasattr(original, "state_dict")
    result["has_load_state_dict"] = hasattr(original, "load_state_dict")
    result["consumed_before_restart"] = 5
    result["batch_size"] = BATCH_SIZE
    result["worker_prefetch_factor"] = 3 if args.workers else None
    total = 23
    if not result["has_load_state_dict"]:
        result["status"] = "no supported loader state API"
        result["fresh_restart"] = batches(loader(dataset(args), args.workers, BATCH_SIZE), total)
        return
    full = batches(loader(original, args.workers, BATCH_SIZE), total)
    result["uninterrupted"] = full
    consumed = result["consumed_before_restart"]
    stopped = dataset(args, repeat=True)
    prefix = batches(loader(stopped, args.workers, BATCH_SIZE), consumed, pause_after_first=True)
    restarted = dataset(args, repeat=True)
    state = {"epoch": 0, "offset": consumed * BATCH_SIZE, "batch_size": BATCH_SIZE}
    restarted.load_state_dict(state)
    suffix = batches(loader(restarted, args.workers, BATCH_SIZE), total - consumed)
    result.update(consumed=prefix, resumed=suffix, checkpoint_state=state, matches=prefix + suffix == full)
    if args.workers == 0:
        direct = dataset(args)
        with closing(iter(direct)) as iterator:
            ids = [int(item["index"]) for item in islice(iterator, consumed)]
            direct_state = direct.state_dict()
            follow = [int(item["index"]) for item in islice(iterator, 12)]
        resumed = dataset(args)
        resumed.load_state_dict(direct_state)
        with closing(iter(resumed)) as iterator:
            resumed_ids = [int(item["index"]) for item in islice(iterator, 12)]
        result.update(direct_ids=ids, direct_state=direct_state, direct_state_matches=follow == resumed_ids)


def probe_video(args, result):
    data = dataset(args)
    oracle = Oracle(args.root, args.mode, None)
    for episode, frame in [(0, 4), (0, 106), (1, 10), (15, 112)]:
        row = oracle.rows[episode]
        index = row["dataset_from_index"] + frame
        anchor = {
            "index": index, "episode_index": episode, "frame_index": frame, "task_index": 0,
            "timestamp": frame / FPS, ACTION: vectors(index, ACTION), STATE: vectors(index, STATE),
        }
        oracle.errors.clear()
        oracle.examples.clear()
        try:
            oracle.sample(data.query_frames(anchor))
            result.setdefault("probes", []).append({
                "episode": episode, "frame": frame, "checks_failed": dict(oracle.errors),
                "details": dict(oracle.examples),
            })
        except Exception as exc:
            result.setdefault("probes", []).append({
                "episode": episode, "frame": frame, "error": type(exc).__name__, "detail": str(exc),
            })


def repeat_coverage(args, result):
    data = dataset(args, repeat=True)
    expected = data.num_frames_for_rank(args.rank, args.world, args.workers)
    sequence = batches(loader(data, args.workers, BATCH_SIZE), (expected + BATCH_SIZE - 1) // BATCH_SIZE)
    ids = [index for batch in sequence for index in batch][:expected]
    counts = Counter(ids)
    result.update(expected=expected, ids=ids, returned=len(ids), unique=len(counts),
                  repeats_before_epoch_end=len(ids) - len(counts))
    seen = set()
    for position, index in enumerate(ids, start=1):
        if index in seen:
            result.update(first_duplicate={"position": position, "sample_id": index})
            break
        seen.add(index)


def check_shuffle(args, result):
    orders = {}
    for shuffle in (True, False):
        for seed in (42, 43):
            data = dataset(args, seed=seed, shuffle=shuffle)
            key = f"shuffle_{str(shuffle).lower()}_seed_{seed}"
            orders[key] = [[int(sample["index"]) for sample in data] for _ in range(2)]
    result["orders"] = orders
    result["true_seed_changes_order"] = orders["shuffle_true_seed_42"][0] != orders["shuffle_true_seed_43"][0]
    result["false_seed_changes_order"] = orders["shuffle_false_seed_42"][0] != orders["shuffle_false_seed_43"][0]
    result["shuffle_flag_changes_order"] = orders["shuffle_true_seed_42"][0] != orders["shuffle_false_seed_42"][0]
    result["repeat_changes_order"] = {key: first != second for key, (first, second) in orders.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("baseline", "proposal"))
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--mode", choices=("full", "anchors", "numeric", "positive", "video"), default="full")
    parser.add_argument("--local-id", action="store_true",
                        help="Pass the fixture directory as repo_id to bypass the proposal's local parquet lookup.")
    parser.add_argument("--selection", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--probe-video", action="store_true")
    parser.add_argument("--repeat-coverage", action="store_true")
    parser.add_argument("--check-shuffle", action="store_true")
    args = parser.parse_args()
    os.environ.update(RANK=str(args.rank), WORLD_SIZE=str(args.world))
    torch.set_num_threads(1)
    result = vars(args) | {"datasets": datasets.__version__, "torch": torch.__version__}
    started = time.monotonic()
    try:
        if args.check_shuffle:
            check_shuffle(args, result)
        elif args.probe_video:
            probe_video(args, result)
        elif args.repeat_coverage:
            repeat_coverage(args, result)
        elif args.restart:
            restart(args, result)
        else:
            finite(args, result)
    except Exception as exc:
        result.update(error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
    result["seconds"] = time.monotonic() - started
    args.out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    if "error" in result:
        print(result["error"]["traceback"], end="")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
