"""Measure the two pinned readers with the common streaming benchmark."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import benchmark

if __name__ == "__main__":
    label = sys.argv[sys.argv.index("--reader") + 1]
    if label == "candidate":
        from lerobot.datasets import video_utils

        pins = json.loads(Path(__file__).with_name("native_molmo_pins.json").read_text())["grouped"]
        if sha256(Path(video_utils.__file__).read_bytes()).hexdigest() != pins["module_sha256"]["video_utils.py"]:
            raise ValueError("Unexpected video decoder module")
        benchmark.PINS["candidate"] = (pins["revision"], pins["module_sha256"]["streaming_dataset.py"])
    raise SystemExit(benchmark.main())
