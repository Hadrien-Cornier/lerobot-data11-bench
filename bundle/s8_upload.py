"""Stage upload: push the encoded SEP / STACK / BIG files of selected families to a PRIVATE HF
dataset repo, so stage remote can read them over hf:// exactly like a streamed LeRobot dataset.

  repo     $UP_REPO (default $RESULTS_REPO), must already exist and be private (checked, never created public)
  prefix   variants/<run name>/<family>/<layout>/<file>.mp4 (same relative layout as $WORK/enc)
  commit   the commit oid of the upload is recorded; the read-back check uses THAT revision:
           every file's size and LFS sha256 from get_paths_info(revision) vs the local file, plus
           the first and last 64 KiB read through HfFileSystem at hf://datasets/<repo>@<rev>/...

Usage: python s8_upload.py <selection.json> <enc_root> <out_dir>
       env: SMOKE, RUN_NAME, UP_REPO / RESULTS_REPO, UP_FAMILIES (default: every active ckpt_* family)
"""

import hashlib
import os
import sys
import time
from pathlib import Path

from common import LAYOUTS, Family, load_selection, write_json


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def main():
    sel_path, enc_root, out_dir = sys.argv[1:4]
    smoke = os.environ.get("SMOKE", "0") == "1"
    from huggingface_hub import HfApi, HfFileSystem

    repo = os.environ.get("UP_REPO") or os.environ["RESULTS_REPO"]
    run = os.environ.get("RUN_NAME") or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    sel = load_selection(sel_path, smoke)
    fams = os.environ.get("UP_FAMILIES", "").split() or [f for f in sel["active"] if f.startswith("ckpt")]
    api = HfApi()
    info = api.repo_info(repo, repo_type="dataset")
    assert info.private, f"{repo} is not private; refusing to upload"
    prefix = f"variants/{run}"
    files = []
    for fam in fams:
        F = Family(enc_root, fam, sel["active"][fam], sel["cams"], sel["fps"])
        for lay in LAYOUTS:
            nfs = F.frames_per_file(lay)
            per_file = [n for n in nfs for _ in (F.cams if lay != "STACK" else [0])]
            for p, n in zip(F.files(lay), per_file, strict=True):
                rel = os.path.relpath(p, enc_root)
                files.append({"family": fam, "layout": lay, "rel": rel, "path_in_repo": f"{prefix}/{rel}", "bytes": os.path.getsize(p),
                              "frames": n, "sha256": sha256(p)})
    total = sum(f["bytes"] for f in files)
    print(f"upload: {len(files)} files, {total / 1e6:.1f} MB, families {fams} -> {repo}:{prefix}", flush=True)
    t0 = time.perf_counter()
    ci = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=enc_root, path_in_repo=prefix,
                           allow_patterns=[f["rel"] for f in files], commit_message=f"DATA-11 variants {run}: {' '.join(fams)}")
    up_s = time.perf_counter() - t0
    rev = ci.oid
    print(f"committed {rev} in {up_s:.1f}s ({total / 1e6 / up_s:.1f} MB/s)", flush=True)

    # ---- read back via the recorded revision
    problems = []
    remote = {}
    paths = [f["path_in_repo"] for f in files]
    for i in range(0, len(paths), 50):
        for pi in api.get_paths_info(repo, paths[i : i + 50], repo_type="dataset", revision=rev):
            remote[pi.path] = pi
    fs = HfFileSystem()
    for f in files:
        pi = remote.get(f["path_in_repo"])
        if pi is None:
            problems.append(f"missing at {rev}: {f['path_in_repo']}")
            continue
        lfs_sha = getattr(pi.lfs, "sha256", None) if pi.lfs else None
        f["remote_bytes"], f["remote_lfs_sha256"], f["xet_hash"] = pi.size, lfs_sha, getattr(pi, "xet_hash", None)
        if pi.size != f["bytes"]:
            problems.append(f"size {f['path_in_repo']}: remote {pi.size} local {f['bytes']}")
        if lfs_sha is not None and lfs_sha != f["sha256"]:
            problems.append(f"sha256 {f['path_in_repo']}: remote {lfs_sha} local {f['sha256']}")
        if lfs_sha is None:
            problems.append(f"no LFS/xet pointer for {f['path_in_repo']} (stored as a regular git blob?)")
        loc = Path(enc_root) / f["rel"]
        with fs.open(f"datasets/{repo}@{rev}/{f['path_in_repo']}", "rb", block_size=0) as fh, open(loc, "rb") as lf:
            head = fh.read(65536)
            ok_head = head == lf.read(65536)
        with fs.open(f"datasets/{repo}@{rev}/{f['path_in_repo']}", "rb") as fh, open(loc, "rb") as lf:
            fh.seek(max(0, f["bytes"] - 65536))
            lf.seek(max(0, f["bytes"] - 65536))
            ok_tail = fh.read(65536) == lf.read()
        if not (ok_head and ok_tail):
            problems.append(f"bytes differ through HfFileSystem: {f['path_in_repo']} head {ok_head} tail {ok_tail}")
    res = {"repo": repo, "private": info.private, "revision": rev, "commit_url": ci.commit_url, "prefix": prefix, "families": fams,
           "files": files, "bytes": total, "upload_s": round(up_s, 1), "readback_problems": problems,
           "hf_url_base": f"hf://datasets/{repo}@{rev}/{prefix}"}
    write_json(Path(out_dir) / "upload" / "upload.json", res)
    print(f"read back {len(files)} files at {rev}: {len(problems)} problems", flush=True)
    for p in problems[:20]:
        print("  PROBLEM", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
