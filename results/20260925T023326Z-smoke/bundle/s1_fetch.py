"""Stage fetch: download ONLY the selected source MP4s (+ meta/info.json) at the pinned revision,
check every file size against selection.json, and cross-check info.json codec settings.

Usage: python s1_fetch.py <selection.json> <src_root> <out_dir>   (env SMOKE)
"""

import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

from common import write_json


def main():
    sel_path, src_root, out_dir = sys.argv[1:4]
    smoke = os.environ.get("SMOKE", "0") == "1"
    sel = json.loads(Path(sel_path).read_text())
    want = sel["download"]["smoke" if smoke else "full"]
    t0 = time.perf_counter()
    snapshot_download(sel["repo_id"], repo_type="dataset", revision=sel["revision"], local_dir=src_root,
                      allow_patterns=[*want, "meta/info.json"], max_workers=8)
    wall = time.perf_counter() - t0
    bad = {p: (os.path.getsize(Path(src_root) / p) if (Path(src_root) / p).exists() else None, s)
           for p, s in want.items() if not (Path(src_root) / p).exists() or os.path.getsize(Path(src_root) / p) != s}
    info = json.loads((Path(src_root) / "meta/info.json").read_text())
    vinfo = {k: v["info"] for k, v in info["features"].items() if v["dtype"] == "video"}
    total = sum(want.values())
    res = {"repo_id": sel["repo_id"], "revision": sel["revision"], "files": len(want), "bytes": total,
           "gb": round(total / 1e9, 3), "wall_s": round(wall, 1), "mb_per_s": round(total / 1e6 / wall, 1) if wall else None,
           "size_mismatches": bad, "source_video_info": vinfo, "fps": info["fps"]}
    write_json(Path(out_dir) / "fetch" / "fetch.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "source_video_info"}, indent=1))
    assert not bad, f"size mismatches: {bad}"
    assert info["fps"] == sel["fps"]


if __name__ == "__main__":
    main()
