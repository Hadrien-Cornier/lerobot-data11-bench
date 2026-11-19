"""Compare the installed reader's first 1,000 MolmoAct2 batches to the saved stock result."""

import argparse
import json
import platform
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import torch
from catalog import SOURCES, snapshot
from internvideo_audit import MOLMO_MANIFEST, BatchAudit, load_reference
from lerobot.datasets import streaming_dataset
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write-stock",
        type=Path,
        help="Capture the unmodified PR on this decoder stack",
    )
    mode.add_argument(
        "--against", type=Path, help="Check all outputs against a local stock capture"
    )
    args = parser.parse_args()
    software = {key: version(key) for key in ("torch", "torchcodec", "av", "datasets")}
    software["platform"] = platform.platform()
    if args.write_stock:
        code = Path(streaming_dataset.__file__).read_bytes()
        if (
            sha256(code).hexdigest()
            != "17ec1e996af3e83de1e64b08b4550ff4c25914f41abf06329abbeb5c0f50a656"
        ):
            raise ValueError(
                "--write-stock requires LeRobot at the original #5388 commit"
            )
    else:
        stock = json.loads(args.against.read_text())
        if stock["software"] != software:
            raise ValueError(
                "Run stock and the proposed reader on the same decoder stack"
            )
    here = Path(__file__).resolve().parent
    ref = load_reference(here / "internvideo_reference.json.gz")
    if snapshot("bucket")["manifest_sha256"] != MOLMO_MANIFEST:
        raise ValueError("The bucket has changed since the reference run")

    torch.set_num_threads(1)
    start = perf_counter()
    dataset = streaming_dataset.StreamingLeRobotDataset(
        SOURCES["bucket"][0],
        repo_type="bucket",
        return_uint8=True,
        buffer_size=32,
        seed=42,
        tolerance_s=0.001,
    )
    audit = BatchAudit(ref, dataset.meta.video_keys)
    loader = iter(DataLoader(dataset, batch_size=8, num_workers=0))
    for number in range(1, 1001):
        batch = next(loader)
        audit.check(number, batch)
        if number in (1, 125, 1000):
            print(
                f"Checked batch {number} after {perf_counter() - start:.2f} seconds",
                flush=True,
            )
    del loader
    result = audit.summary()
    if snapshot("bucket")["manifest_sha256"] != MOLMO_MANIFEST:
        raise ValueError("The bucket changed during the run")
    capture = {"software": software, "result": result}
    if args.write_stock:
        args.write_stock.write_text(json.dumps(capture, indent=2) + "\n")
    elif result != stock["result"]:
        raise ValueError(
            f"Output differs from stock on the same decoder stack: {capture}"
        )
    print(json.dumps(capture, indent=2))


if __name__ == "__main__":
    main()
