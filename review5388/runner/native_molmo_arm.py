"""Measure exact consumer delivery and verify the delivered tensors for one revision."""
import json
import platform
import random
import sys
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from datasets.filesystems import caching
from torch.utils.data import DataLoader

from catalog import SOURCES
from internvideo_audit import BATCH_SIZE, REFERENCE_BATCHES, BatchAudit, load_reference
from lerobot.datasets import streaming_dataset, video_utils
from lerobot.utils.collate import lerobot_collate_fn


ROOT = Path(__file__).resolve().parents[1]
PINS = json.loads(Path(__file__).with_name("native_molmo_pins.json").read_text())
EXPECTED = "76313c64bf9870e8a60f52d47b8faec13752adffd3af6236900ad5c059bf7c66"
DATASETS_CACHE_SHA = "16ab95f92d6d147e0ab15de55d0a4afab2553331fa520df0e6e8ae4b58ab2447"
CAMERAS = [f"observation.images.{name}" for name in ("right", "left", "top")]


def save(path, result):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(path)


def main(label, output):
    pin = PINS[label]
    for module in (streaming_dataset, video_utils):
        path = Path(module.__file__)
        if sha256(path.read_bytes()).hexdigest() != pin["module_sha256"][path.name]:
            raise ValueError(f"Wrong source for {label}: {path}")
    if sha256(Path(caching.__file__).read_bytes()).hexdigest() != DATASETS_CACHE_SHA:
        raise ValueError("Wrong Datasets cache source.")
    reference = ROOT / "internvideo_reference.json.gz"
    audit = BatchAudit(load_reference(reference), CAMERAS)
    torch.set_num_threads(1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    result = {
        "arm": label,
        "status": "started",
        "source": SOURCES["bucket"][0],
        "host": platform.node(),
        "source_pin": pin,
        "reference_sha256": sha256(reference.read_bytes()).hexdigest(),
        "settings": {
            "seed": 42, "buffer_size": 32, "batch_size": BATCH_SIZE, "workers": 0,
            "tolerance_s": 0.001, "return_uint8": True, "delta_timestamps": None,
            "streaming_module": streaming_dataset.__file__,
        },
        "software": {key: version(key) for key in (
            "torch", "torchcodec", "av", "datasets", "huggingface_hub", "fsspec",
            "hf_xet", "numpy", "pyarrow",
        )},
        "milestones": {},
    }
    save(output, result)
    start = perf_counter()
    dataset = streaming_dataset.StreamingLeRobotDataset(
        SOURCES["bucket"][0], repo_type="bucket", return_uint8=True,
        buffer_size=32, seed=42, tolerance_s=0.001,
    )
    result["constructor_s"] = perf_counter() - start
    if dataset.fps != 30 or set(dataset.meta.video_keys) != set(CAMERAS):
        raise ValueError("Unexpected source metadata.")
    collate = lerobot_collate_fn if dataset.meta.has_language_columns else None
    loader = iter(DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=collate))
    consumer_s = 0.0
    for number in range(1, REFERENCE_BATCHES + 1):
        batch = next(loader)
        delivered = perf_counter()
        checkpoint = number in (1, 125, REFERENCE_BATCHES)
        if checkpoint:
            wall_s = delivered - start
            result["milestones"][str(number)] = {
                "wall_delivery_s": wall_s,
                "prior_consumer_work_s": consumer_s,
                "delivery_less_prior_consumer_work_s": wall_s - consumer_s,
            }
        audit.check(number, batch)
        del batch
        if checkpoint:
            result["checked_batches"] = number
            save(output, result)
            print(f"{label}: verified batch {number} at {wall_s:.3f} s", flush=True)
        consumer_s += perf_counter() - delivered
    result["audit"] = audit.summary()
    result["expected_ordered_output_sha256"] = EXPECTED
    result["matched_stock_output"] = result["audit"]["ordered_output_sha256"] == EXPECTED
    result["status"] = "complete" if result["matched_stock_output"] else "digest-mismatch"
    save(output, result)
    if not result["matched_stock_output"]:
        raise ValueError("The native Linux field digest differs from the saved stock digest.")
    del loader


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
