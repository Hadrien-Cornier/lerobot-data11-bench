"""Survey MP4 `moov` box sizes in popular LeRobot datasets with small HTTP range reads.

For each sampled video file, walk the top-level MP4 boxes by reading only box headers
(16 bytes per box), and record the `moov` size and whether it is before or after `mdat`.
From the layout, derive the reads that the #3917 header probe makes now (4 MiB first read,
doubling from byte 0) and with the proposed change (512 KiB first read, exact follow-up reads).

Usage: uv run --no-project --with huggingface_hub --with requests python moov_survey.py OUT_DIR
"""

from __future__ import annotations

import json
import os
import random
import statistics
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

# Top 10 LeRobot-tagged datasets by Hub downloads (last 30 days), listed on 2026-10-02.
DATASETS = [
    "cadene/droid_1.0.1",
    "IPEC-COMMUNITY/droid_lerobot",
    "IPEC-COMMUNITY/bridge_orig_lerobot",
    "IPEC-COMMUNITY/language_table_lerobot",
    "IPEC-COMMUNITY/kuka_lerobot",
    "IPEC-COMMUNITY/fractal20220817_data_lerobot",
    "cadene/agibot_alpha_v30",
    "yaak-ai/L2D",
    "cadene/droid",
    "physical-intelligence/libero",
]
# Top 10 v3.0 datasets with video by the same ranking (the format that #3917 reads).
DATASETS_V3 = [
    "cadene/agibot_alpha_v30",
    "yaak-ai/L2D",
    "zekaiwang/trex_dataset",
    "allenai/MolmoAct2-BimanualYAM-Dataset",
    "lerobot/aloha_sim_transfer_cube_human",
    "lerobot/aloha_sim_insertion_human",
    "lerobot/aloha_sim_insertion_scripted",
    "lerobot/pusht",
    "BitRobot/HIW-500-LeRobot",
    "lerobot/droid_1.0.1",
]
FILES_PER_DATASET = 40
SEED = 0
KIB = 1024
MIB = 1024 * KIB

TOKEN = os.environ.get("HF_TOKEN")
SESSION = requests.Session()
if TOKEN:
    SESSION.headers["Authorization"] = f"Bearer {TOKEN}"


def read_range(url: str, start: int, length: int) -> tuple[bytes, int]:
    r = SESSION.get(url, headers={"Range": f"bytes={start}-{start + length - 1}"}, timeout=60)
    r.raise_for_status()
    if r.status_code != 206:
        raise RuntimeError(f"range not honored for {url}")
    total = int(r.headers["Content-Range"].rsplit("/", 1)[1])
    return r.content, total


def walk_boxes(url: str) -> dict:
    boxes = []
    offset, file_size = 0, None
    while file_size is None or offset < file_size:
        head, file_size = read_range(url, offset, 16)
        size, kind = struct.unpack(">I4s", head[:8])
        if size == 1:
            size = struct.unpack(">Q", head[8:16])[0]
        elif size == 0:
            size = file_size - offset
        boxes.append((kind.decode("latin-1"), offset, size))
        if kind == b"moov":
            break
        offset += size
    types = [b[0] for b in boxes]
    moov = next(b for b in boxes if b[0] == "moov")
    return {
        "file_size": file_size,
        "boxes": boxes,
        "moov_size": moov[2],
        "moov_end": moov[1] + moov[2],
        "moov_first": "mdat" not in types[: types.index("moov")],
    }


def tail_bytes(layout: dict, cap: int = 64 * MIB) -> int:
    """`_fetch_tail_moov_index` reads from the end of `mdat` to the end of the file (capped)."""
    mdat = next(b for b in layout["boxes"] if b[0] == "mdat")
    return min(cap, layout["file_size"] - (mdat[1] + mdat[2]))


def reads_now(layout: dict, first: int = 4 * MIB, cap: int = 64 * MIB) -> tuple[int, int]:
    """Reads and bytes of the 6d945985 probe: the prefix doubles and is re-read from byte 0."""
    limit = min(cap, layout["file_size"])
    # A moov-first file is complete when the prefix holds moov and the next (mdat) box header.
    need = layout["moov_end"] + 8 if layout["moov_first"] else float("inf")
    n = min(first, limit)
    reads, total = 1, n
    while n < need and n < limit:
        n = min(2 * n, limit)
        reads, total = reads + 1, total + n
    if layout["moov_first"] or n >= layout["file_size"]:
        # A prefix that holds the whole file also holds a trailing moov: no tail read.
        return reads, total
    return reads + 1, total + tail_bytes(layout, cap)


def reads_patched(layout: dict, first: int = 512 * KIB) -> tuple[int, int]:
    """Reads and bytes of the proposed probe (exact follow-up read, tail read for a trailing moov)."""
    n = min(first, layout["file_size"])
    if n >= layout["file_size"]:
        return 1, n
    if layout["moov_first"]:
        if n >= layout["moov_end"] + 8:
            return 1, n
        return 2, min(layout["moov_end"] + 16, layout["file_size"])
    return 2, n + tail_bytes(layout)


def sample_paths(api: HfApi, repo: str, sha: str, info: dict, rng: random.Random) -> list[str]:
    keys = [k for k, f in info["features"].items() if f.get("dtype") == "video"]
    if not keys:
        return []
    if info["codebase_version"].startswith("v3"):
        files = [
            f.path
            for f in api.list_repo_tree(repo, path_in_repo="videos", recursive=True, repo_type="dataset", revision=sha)
            if f.path.endswith(".mp4")
        ]
        return rng.sample(files, min(FILES_PER_DATASET, len(files)))
    chunks_size = info.get("chunks_size", 1000)
    paths = set()
    while len(paths) < min(FILES_PER_DATASET, info["total_episodes"] * len(keys)):
        ep = rng.randrange(info["total_episodes"])
        key = rng.choice(keys)
        paths.add(info["video_path"].format(episode_chunk=ep // chunks_size, video_key=key, episode_index=ep))
    return sorted(paths)


def survey(repo: str, out: Path) -> dict:
    api = HfApi(token=TOKEN)
    sha = api.dataset_info(repo).sha
    info = json.loads(Path(hf_hub_download(repo, "meta/info.json", repo_type="dataset", revision=sha)).read_text())
    rng = random.Random(f"{SEED}:{repo}")
    paths = sample_paths(api, repo, sha, info, rng)
    result = {"repo": repo, "revision": sha, "codebase_version": info["codebase_version"], "fps": info.get("fps"),
              "total_videos": info.get("total_videos"), "video_keys": [k for k, f in info["features"].items() if f.get("dtype") == "video"],
              "files": []}
    def one(path):
        try:
            layout = walk_boxes(hf_hub_url(repo, path, repo_type="dataset", revision=sha))
        except Exception as exc:  # noqa: BLE001 - record and continue
            return {"path": path, "error": repr(exc)}
        r0, b0 = reads_now(layout)
        r1, b1 = reads_patched(layout)
        return {"path": path, **layout, "reads_now": r0, "bytes_now": b0, "reads_patched": r1, "bytes_patched": b1}
    with ThreadPoolExecutor(16) as pool:
        result["files"] = list(pool.map(one, paths))
    (out / f"{repo.replace('/', '__')}.json").write_text(json.dumps(result, indent=1))
    return result


def summarize(r: dict) -> dict:
    ok = [f for f in r["files"] if "error" not in f]
    if not ok:
        return {"repo": r["repo"], "version": r["codebase_version"], "n": 0, "errors": len(r["files"])}
    moov = [f["moov_size"] / KIB for f in ok]
    return {
        "repo": r["repo"], "version": r["codebase_version"], "n": len(ok), "errors": len(r["files"]) - len(ok),
        "file_mib_median": statistics.median(f["file_size"] / MIB for f in ok),
        "moov_kib_mean": statistics.mean(moov), "moov_kib_median": statistics.median(moov), "moov_kib_max": max(moov),
        "needs_2nd_read": sum(f["reads_patched"] > 1 for f in ok),
        "moov_after_mdat": sum(not f["moov_first"] for f in ok),
        "kib_now_mean": statistics.mean(f["bytes_now"] / KIB for f in ok),
        "kib_patched_mean": statistics.mean(f["bytes_patched"] / KIB for f in ok),
        "reads_now_mean": statistics.mean(f["reads_now"] for f in ok),
        "reads_patched_mean": statistics.mean(f["reads_patched"] for f in ok),
    }


def resummarize(out: Path) -> None:
    """Recompute the read model and the summary from saved layouts, without network reads."""
    rows = []
    for p in sorted(out.glob("*.json")):
        if p.name in ("summary.json", "verify.json"):
            continue
        r = json.loads(p.read_text())
        for f in r["files"]:
            if "error" not in f:
                f["reads_now"], f["bytes_now"] = reads_now(f)
                f["reads_patched"], f["bytes_patched"] = reads_patched(f)
        p.write_text(json.dumps(r, indent=1))
        rows.append(summarize(r))
        print(json.dumps(rows[-1]))
    (out / "summary.json").write_text(json.dumps(rows, indent=1))


def main() -> None:
    if sys.argv[1] == "--resummarize":
        for d in sys.argv[2:]:
            resummarize(Path(d))
        return
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    repos = DATASETS_V3 if os.environ.get("SURVEY_SET") == "v3" else DATASETS
    for repo in repos:
        try:
            rows.append(summarize(survey(repo, out)))
        except Exception as exc:  # noqa: BLE001
            rows.append({"repo": repo, "error": repr(exc)})
        print(json.dumps(rows[-1]), flush=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
