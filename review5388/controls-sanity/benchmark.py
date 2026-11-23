import json
import os
import platform
import sys
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from lerobot.datasets import streaming_dataset, video_utils
from lerobot.utils.collate import lerobot_collate_fn

ARM, MODE = sys.argv[1:3]
HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "pins.json").read_text())
REPO = "pepijn223/MolmoAct2-BimanualYAM-Dataset-bucket"
CAMERAS = [f"observation.images.{name}" for name in ("right", "left", "top")]


def event(name, **fields):
    print(json.dumps({"event": name, "arm": ARM, **fields}, sort_keys=True), flush=True)


def camera_hashes(sample, row=None):
    return {
        key: sha256(memoryview((sample[key] if row is None else sample[key][row]).contiguous().numpy()).cast("B")).hexdigest()
        for key in CAMERAS
    }


def main():
    torch.set_num_threads(1)
    for module in (streaming_dataset, video_utils):
        path = Path(module.__file__)
        actual = sha256(path.read_bytes()).hexdigest()
        assert actual == PINS[ARM][path.name], f"Wrong source for {ARM}: {path}"
    kwargs = {
        "repo_type": "bucket",
        "tolerance_s": 0.001,
        "seed": 42,
        "return_uint8": True,
    }
    if ARM == "grouped":
        kwargs["buffer_size"] = 32
    else:
        kwargs.update(episode_pool_size=32, prefetch_episodes=8, video_cache_bytes=2 << 30)
    event(
        "start", revision=PINS[ARM]["revision"], mode=MODE, host=platform.node(),
        software={key: version(key) for key in ("torch", "torchcodec", "av", "datasets", "huggingface_hub", "fsspec", "pyarrow")},
        settings={**kwargs, "batch_size": 8, "workers": 0},
    )
    start = perf_counter()
    dataset = streaming_dataset.StreamingLeRobotDataset(REPO, **kwargs)
    constructor_s = perf_counter() - start
    assert set(dataset.meta.video_keys) == set(CAMERAS)
    if MODE == "parity":
        probes = json.loads((HERE / "controls-probes.json").read_text())
        mismatches = []
        for probe in probes:
            item = dataset.query_frames(probe["anchor"])
            actual = camera_hashes(item)
            if actual != probe["sha256"]:
                mismatches.append({"anchor": probe["anchor"], "cameras": [key for key in CAMERAS if actual[key] != probe["sha256"][key]]})
        event("parity", anchors=len(probes), camera_frames=len(probes) * len(CAMERAS), mismatches=mismatches)
        return bool(mismatches)

    collate = lerobot_collate_fn if dataset.meta.has_language_columns else None
    loader = iter(DataLoader(dataset, batch_size=8, num_workers=0, collate_fn=collate))
    seen = set()
    mix = Counter()
    anchor_hash = sha256()
    consumer_s = 0.0
    probes = []
    for number in range(1, 1001):
        batch = next(loader)
        delivered = perf_counter() - start
        identifiers = list(zip(batch["episode_index"].tolist(), batch["frame_index"].tolist(), strict=True))
        assert len(identifiers) == 8
        assert len(set(identifiers)) == 8 and seen.isdisjoint(identifiers), identifiers
        seen.update(identifiers)
        mix[len({episode for episode, _ in identifiers})] += 1
        anchor_hash.update(json.dumps(identifiers, separators=(",", ":")).encode() + b"\n")
        for key in CAMERAS:
            assert batch[key].dtype == torch.uint8 and tuple(batch[key].shape) == (8, 3, 360, 640)
        for key in ("action", "observation.state"):
            assert batch[key].dtype == torch.float32 and tuple(batch[key].shape) == (8, 14)
        if ARM == "controls" and (number == 1 or number % 100 == 0):
            task = int(batch["task_index"][0])
            episode, frame = identifiers[0]
            probes.append({
                "anchor": {"episode_index": episode, "frame_index": frame, "task_index": task},
                "sha256": camera_hashes(batch, 0),
            })
        if number in (1, 125, 1000):
            event(
                "batch", number=number, wall_s=delivered, constructor_s=constructor_s,
                delivery_less_consumer_s=delivered-consumer_s,
                distinct_episodes_in_batch=len({episode for episode, _ in identifiers}),
                episode_mix=dict(mix), first_anchors=identifiers if number == 1 else None,
                unique_anchors=len(seen),
            )
        del batch
        consumer_s += perf_counter() - start - delivered
    if probes:
        (HERE / "controls-probes.json").write_text(json.dumps(probes))
    event(
        "finish", count=len(seen), episodes=len({episode for episode, _ in seen}),
        mean_distinct_episodes=sum(k * v for k, v in mix.items()) / 1000,
        episode_mix=dict(mix), ordered_anchor_sha256=anchor_hash.hexdigest(),
    )
    del loader
    return False


if __name__ == "__main__":
    raise SystemExit(main())
