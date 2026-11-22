"""Replay the grouped RGB decode path against two real MP4s with fresh shared block caches."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter


# Each group is one native multi-frame call, with positions expressed in frames at 30 Hz.
# Alternate near and far requests, and revisit an earlier region late in the trace.
STARTS = (0, 240, 480, 60, 960, 245, 1440, 10)
OFFSETS = (6, 0, 8, 2, 1, 4, 3, 7)
FILES = {
    "MolmoAct2": (
        "buckets/pepijn223/MolmoAct2-BimanualYAM-Dataset-bucket/"
        "videos/observation.images.right/chunk-000/file-000.mp4",
        "9b85f78ee61d2a6a6cb1e8009b670ac519a21483af268dd567f28f1bea2d4f85",
    ),
    "ABC-130K": (
        "datasets/lerobot/abc_130k_v3_train@68651e4929d9fb00f798937b2d62617cab5c771d/"
        "videos/observation.images.top/chunk-000/file-000.mp4",
        "b774e019e32f38a84c193ab262e96cba05ed6b5842a0b6c114762b3ea0aaf4b7",
    ),
}
ROOT = Path(__file__).resolve().parent


def one(dataset, mib):
    import fsspec
    import torch
    from datasets.filesystems import caching
    from huggingface_hub import HfFileSystem
    from lerobot.datasets import streaming_dataset, video_utils

    pins = json.loads((ROOT / "runner/native_molmo_pins.json").read_text())["grouped"]
    for module in (streaming_dataset, video_utils):
        path = Path(module.__file__)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pins["module_sha256"][path.name]
    assert hashlib.sha256(Path(caching.__file__).read_bytes()).hexdigest() == (
        "16ab95f92d6d147e0ab15de55d0a4afab2553331fa520df0e6e8ae4b58ab2447"
    )
    torch.set_num_threads(1)
    path, xet_hash = FILES[dataset]
    hub = HfFileSystem()
    assert hub.info(path)["xet_hash"] == xet_hash
    store = caching.SharedBlockCacheFileSystem.default_store(block_size=mib * 2**20)
    fsspec.config.conf["sharedblockcache"] = {"block_size": mib * 2**20}
    assert store.stats()["cached_bytes"] == 0
    reader = object.__new__(streaming_dataset.StreamingLeRobotDataset)
    reader.tolerance_s = 0.001
    reader._return_uint8 = True
    pool = video_utils._StreamingVideoDecoderPool(96)
    digest = hashlib.sha256()
    elapsed = 0.0
    stages = []
    try:
        for first in STARTS:
            frame_ids = [first + off for off in OFFSETS]
            start = perf_counter()
            frames = reader._read_rgb_group(pool, "sharedblockcache::hf://" + path, [idx / 30 for idx in frame_ids])
            elapsed += perf_counter() - start
            for idx, frame in zip(frame_ids, frames, strict=True):
                digest.update(idx.to_bytes(8, "little"))
                digest.update(frame.contiguous().numpy().tobytes())
            stages.append(elapsed)
    finally:
        pool.close()
    assert HfFileSystem(skip_instance_cache=True).info(path)["xet_hash"] == xet_hash
    stats = store.stats()
    assert stats["requests"] > 0 and stats["block_size"] == mib * 2**20
    return {
        "dataset": dataset, "block_MiB": mib, "path": path, "object_hash": xet_hash,
        "first_group_seconds": stages[0], "cumulative_decode_seconds": elapsed,
        "rgb_and_frame_id_sha256": digest.hexdigest(), "cache": stats,
        "software": {name: version(name) for name in ("datasets", "fsspec", "huggingface_hub", "torch", "torchcodec")},
        "hardware": platform.machine(), "platform": platform.system(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=FILES)
    parser.add_argument("--mib", type=int, choices=(1, 4))
    args = parser.parse_args()
    if args.arm:
        args.out.write_text(json.dumps(one(args.arm, args.mib), indent=2) + "\n")
        return
    rows = []
    for order in ((1, 4), (4, 1)):
        for dataset in FILES:
            for mib in order:
                with TemporaryDirectory(prefix="lerobot-block-probe-") as tmp:
                    out = Path(tmp) / "result.json"
                    env = os.environ | {"HF_HOME": tmp, "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
                    subprocess.run([sys.executable, __file__, "--arm", dataset, "--mib", str(mib), "--out", str(out)], env=env, check=True)
                    row = json.loads(out.read_text())
                    assert all(old["rgb_and_frame_id_sha256"] == row["rgb_and_frame_id_sha256"]
                               for old in rows if old["dataset"] == dataset)
                    rows.append(row)
                    args.out.write_text(json.dumps({"starts": STARTS, "offsets": OFFSETS, "rows": rows}, indent=2) + "\n")
                    print(dataset, mib, round(row["cumulative_decode_seconds"], 3), row["cache"]["requests"], row["cache"]["downloaded_bytes"], flush=True)


if __name__ == "__main__":
    main()
