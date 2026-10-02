"""Check the read model of moov_survey.py against the real #3917 `fetch_mp4_index` code.

Runs the patched probe on sampled files, and the base (6d945985) probe where the model predicts
16 MiB or less. Compares reads and bytes with the model, and the index digests where both run.

Usage: python moov_survey_verify.py BASE_MP4_PY PATCHED_MP4_PY SURVEY_DIR [SURVEY_DIR ...]
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np

from moov_survey import MIB, SESSION, read_range
from huggingface_hub import hf_hub_url

PER_DATASET = 3
BASE_BYTES_LIMIT = 16 * MIB


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def digest(index) -> str:
    h = hashlib.sha1()
    for f in dataclasses.fields(index):
        v = getattr(index, f.name)
        if f.name == "file_path":
            continue
        h.update(np.ascontiguousarray(v).tobytes() if isinstance(v, np.ndarray) else repr(v).encode())
    return h.hexdigest()


def run(mod, url: str, size: int):
    calls = []

    def rr(path, start, length):
        calls.append(length)
        return read_range(path, start, length)[0]

    index = mod.fetch_mp4_index(url, rr, file_size=size)
    return index, len(calls), sum(calls)


def main() -> None:
    base, patched = load(sys.argv[1], "mp4_base"), load(sys.argv[2], "mp4_patched")
    rows = []
    for d in sys.argv[3:]:
        for p in sorted(Path(d).glob("*.json")):
            if p.name == "summary.json":
                continue
            r = json.loads(p.read_text())
            files = [f for f in r["files"] if "error" not in f]
            rng = random.Random(r["repo"])
            # Prefer the interesting layouts: a trailing moov, or a moov over the first read.
            hard = [f for f in files if f["reads_patched"] > 1]
            pick = rng.sample(hard, min(PER_DATASET, len(hard))) or rng.sample(files, min(1, len(files)))
            for f in pick:
                url = hf_hub_url(r["repo"], f["path"], repo_type="dataset", revision=r["revision"])
                row = {"repo": r["repo"], "path": f["path"], "model_patched": [f["reads_patched"], f["bytes_patched"]]}
                ip, n, b = run(patched, url, f["file_size"])
                row["real_patched"] = [n, b]
                if f["bytes_now"] <= BASE_BYTES_LIMIT:
                    ib, n, b = run(base, url, f["file_size"])
                    row["model_now"], row["real_now"] = [f["reads_now"], f["bytes_now"]], [n, b]
                    row["same_index"] = digest(ib) == digest(ip)
                rows.append(row)
                print(json.dumps(row), flush=True)
    ok = all(r["model_patched"] == r["real_patched"] and r.get("model_now") == r.get("real_now") for r in rows)
    print(f"model matches real code on {len(rows)} files: {ok}; same index where both ran: "
          f"{all(r.get('same_index', True) for r in rows)}")
    Path(sys.argv[3], "verify.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    _ = SESSION
    main()
