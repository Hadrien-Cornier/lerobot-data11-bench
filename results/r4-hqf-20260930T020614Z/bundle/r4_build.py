"""Round 4 stage build: 3 real LeRobot v3 datasets from the encoded group, and their upload (private repos).

  -sep    all camera keys, one MP4 per camera per video file
  -stack  one key (observation.images.stacked); per-camera crop rows in info.json ("r4_stack_layout") and in the
          sidecar meta/r4_stack_layout.json
  -sep1   only the first camera key, the SAME files as SEP for that camera (hard links; the Hub dedups the bytes)
R4_NCOPY video files per key (default 8), each a copy of the one encoded group at a distinct path, one parquet data
file per video file (file i <-> data file i), episode lengths of the slice, observation.state / action float32 with
the source dims (random values), timestamps episode-relative, videos/*/from_timestamp file-relative (the s10_prep
approach). Upload with upload_folder (delete_patterns keep the repo equal to the local folder), commit recorded,
sizes read back.

Usage: python r4_build.py <slice.json> <encode.json> <work_dir> <out_dir>
Env: R4_NCOPY (8), R4_REPO_PREFIX (hadriencornier/lerobot-data11-r4), NO_UPLOAD
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from common import pin_threads
from r4_common import STACK_KEY, env, env_int, key, log, read_json, write_json

pin_threads(2)


def build_dataset(kind, sl, enc, ncopy, root):
    import datasets
    import numpy as np
    import pandas as pd

    root = Path(root)
    if root.exists():
        subprocess.run(["rm", "-rf", str(root)], check=True)
    cams, fps = sl["cams"], sl["fps"]
    lay = sl["stack_layout"]
    if kind == "sep":
        keys = {key(c): enc["files"][f"SEP/{c}"]["path"] for c in cams}
        shapes = {key(c): sl["shapes"][c] for c in cams}
    elif kind == "sep1":
        keys = {key(cams[0]): enc["files"][f"SEP/{cams[0]}"]["path"]}
        shapes = {key(cams[0]): sl["shapes"][cams[0]]}
    else:
        keys = {STACK_KEY: enc["files"]["STACK"]["path"]}
        shapes = {STACK_KEY: [lay["height"], lay["width"]]}
    lens = [e["length"] for e in sl["slice"]["episodes"]]
    n = sum(lens)
    sd, ad = sl["state_dim"], sl["action_dim"]
    rng = np.random.default_rng(0)
    feats = datasets.Features({"observation.state": datasets.Sequence(datasets.Value("float32"), length=sd),
                               "action": datasets.Sequence(datasets.Value("float32"), length=ad),
                               "timestamp": datasets.Value("float32"), "frame_index": datasets.Value("int64"),
                               "episode_index": datasets.Value("int64"), "index": datasets.Value("int64"),
                               "task_index": datasets.Value("int64")})
    ep_rows, gidx, ep_idx = [], 0, 0
    for fi in range(ncopy):
        for k, src in keys.items():
            dst = root / "videos" / k / "chunk-000" / f"file-{fi:03d}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(src, dst)
        cols = {"observation.state": rng.standard_normal((n, sd), dtype=np.float32).tolist(),
                "action": rng.standard_normal((n, ad), dtype=np.float32).tolist(),
                "timestamp": [], "frame_index": [], "episode_index": [], "index": list(range(gidx, gidx + n)), "task_index": [0] * n}
        vfrom = 0
        for L in lens:
            cols["timestamp"] += [j / fps for j in range(L)]
            cols["frame_index"] += list(range(L))
            cols["episode_index"] += [ep_idx] * L
            row = {"episode_index": ep_idx, "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
                   "data/chunk_index": 0, "data/file_index": fi, "dataset_from_index": gidx, "dataset_to_index": gidx + L,
                   "tasks": ["task"], "length": L}
            for k in keys:
                row[f"videos/{k}/chunk_index"] = 0
                row[f"videos/{k}/file_index"] = fi
                row[f"videos/{k}/from_timestamp"] = vfrom / fps
                row[f"videos/{k}/to_timestamp"] = (vfrom + L) / fps
            ep_rows.append(row)
            vfrom += L
            gidx += L
            ep_idx += 1
        dp = root / "data" / "chunk-000" / f"file-{fi:03d}.parquet"
        dp.parent.mkdir(parents=True, exist_ok=True)
        datasets.Dataset.from_dict(cols, features=feats).to_parquet(str(dp))
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ep_rows).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["task"], name="task")).to_parquet(root / "meta" / "tasks.parquet")
    ed = sl["encoder_defaults"]
    features = {"observation.state": {"dtype": "float32", "shape": [sd], "names": None},
                "action": {"dtype": "float32", "shape": [ad], "names": None}}
    for k, (h, w) in shapes.items():
        features[k] = {"dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channels"],
                       "info": {"video.height": h, "video.width": w, "video.codec": "av1", "video.pix_fmt": ed["pix_fmt"],
                                "video.is_depth_map": False, "video.fps": fps, "video.channels": 3, "has_audio": False,
                                "video.g": ed["g"], "video.crf": ed["crf"], "video.preset": ed["preset"], "video.fast_decode": ed["fast_decode"]}}
    for k, dt in (("timestamp", "float32"), ("frame_index", "int64"), ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64")):
        features[k] = {"dtype": dt, "shape": [1], "names": None}
    info = {"codebase_version": "v3.0", "robot_type": sl.get("robot_type") or "unknown", "total_episodes": ep_idx, "total_frames": gidx,
            "total_tasks": 1, "chunks_size": 1000, "data_files_size_in_mb": 100, "video_files_size_in_mb": 500, "fps": fps,
            "splits": {"train": f"0:{ep_idx}"}, "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4", "features": features}
    side = None
    if kind == "stack":
        side = {"key": STACK_KEY, "order_top_to_bottom": lay["order"], "crops_y0_h_w": lay["crops"], "height": lay["height"],
                "width": lay["width"], "pad": "right-pad narrower cameras with black; odd heights padded by one black row",
                "source": f"{sl['repo']}@{sl['revision']}"}
        info["r4_stack_layout"] = side
        (root / "meta" / "r4_stack_layout.json").write_text(json.dumps(side, indent=2))
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    return {"kind": kind, "root": str(root), "video_keys": list(keys), "files_per_key": ncopy, "episodes": ep_idx, "frames": gidx,
            "frames_per_file": n, "episodes_per_file": len(lens), "stack_layout": side,
            "video_bytes_per_file": {k: os.path.getsize(p) for k, p in keys.items()}}


def upload_dataset(repo, root, kind, run):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.dataset_info(repo).private:
        api.update_repo_settings(repo, private=True, repo_type="dataset")
    t0 = time.perf_counter()
    ci = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(root), delete_patterns=["data/**", "videos/**", "meta/**"],
                           commit_message=f"DATA-11 round 4 test dataset ({kind}), job {run}")
    info = api.dataset_info(repo, revision=ci.oid, files_metadata=True)
    remote = {s.rfilename: s.size for s in info.siblings}
    local = {str(p.relative_to(root)): p.stat().st_size for p in Path(root).rglob("*") if p.is_file()}
    bad = [k for k, v in local.items() if remote.get(k) != v]
    return {"repo": repo, "commit": ci.oid, "upload_s": round(time.perf_counter() - t0, 1), "files": len(local), "readback_mismatch": bad,
            "private": info.private}


def main():
    sl, enc = read_json(sys.argv[1]), read_json(sys.argv[2])
    work, out = Path(sys.argv[3]), Path(sys.argv[4])
    D = out / "build"
    D.mkdir(parents=True, exist_ok=True)
    ncopy = env_int("R4_NCOPY", "8")
    prefix = env("R4_REPO_PREFIX", "hadriencornier/lerobot-data11-r4")
    no_up = env("NO_UPLOAD", "0") == "1"
    res = {"ncopy": ncopy, "no_upload": no_up, "datasets": {}}
    for kind in ("sep", "stack", "sep1"):
        st = build_dataset(kind, sl, enc, ncopy, work / "ds" / kind)
        st["repo"] = f"{prefix}-{sl['dataset']}-{kind}"
        if not no_up:
            st.update(upload_dataset(st["repo"], st["root"], kind, os.environ.get("RUN_NAME", "?")))
        res["datasets"][kind] = st
        log(f"dataset {kind}: {json.dumps(st)[:500]}")
        write_json(D / "build.json", res)
    res["ok"] = all(not d.get("readback_mismatch") for d in res["datasets"].values())
    write_json(D / "build.json", res)
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
