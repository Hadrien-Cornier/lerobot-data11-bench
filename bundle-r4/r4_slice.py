"""Round 4 stage slice: real file statistics of the source dataset, the slice, and the source downloads.

1. Pin: dataset_info at the pinned full sha (and the current main sha, recorded).
2. meta/info.json: video keys in info.json order = camera order (STACK top to bottom), shapes, fps, state/action dims.
3. meta/episodes over hf:// with column projection (DROID's episode metadata is ~590 MB with stats columns; only a
   few columns are read): real video files per camera, frames per video file (median, p10, p90), real parquet data
   files (from the episode table and from the repo tree), hf num_shards of a streaming load_dataset (what
   StreamingLeRobotDataset sees).
4. Slice: whole consecutive episodes from the first episode, while every camera's episode stays in that camera's
   first video file (cameras roll over to new files independently), capped at R4_MAX_FRAMES (cap recorded).
5. Download only the first video file of every camera (hf_hub_download at the pinned sha), plus meta/info.json.
6. LeRobot's default encoder settings from the installed code (VideoEncoderConfig().get_codec_options()).

Usage: python r4_slice.py <work_dir> <out_dir>      Env: R4_DATASET, R4_MAX_FRAMES (60000), R4_CHECK_N (20), SMOKE
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

import common
from r4_common import DATASETS, env, env_int, key, log, stack_layout, write_json


def col(k, x):
    return f"videos/{k}/{x}"


def read_episodes(repo, sha, keys):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{repo}@{sha}/meta/episodes/*/*.parquet"))
    cols = ["episode_index", "length", "data/chunk_index", "data/file_index"]
    cols += [col(k, x) for k in keys for x in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
    t0 = time.perf_counter()
    tabs = []
    for f in files:
        with fs.open(f, block_size=1 << 20) as fh:
            tabs.append(pq.read_table(fh, columns=cols).to_pandas())
    import pandas as pd

    df = pd.concat(tabs).sort_values("episode_index").reset_index(drop=True)
    return df, {"files": len(files), "s": round(time.perf_counter() - t0, 1)}


def main():
    work, out = Path(sys.argv[1]), Path(sys.argv[2])
    D = out / "slice"
    D.mkdir(parents=True, exist_ok=True)
    ds = env("R4_DATASET", "libero")
    repo, sha = DATASETS[ds]
    smoke = env("SMOKE", "0") == "1"
    cap = env_int("R4_MAX_FRAMES", "1500" if smoke else "60000")
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    pinned = api.dataset_info(repo, revision=sha)
    try:
        cur = api.dataset_info(repo).sha
    except Exception as exc:  # noqa: BLE001
        cur = f"error: {exc}"
    res = {"dataset": ds, "repo": repo, "revision": pinned.sha, "pinned_sha": sha, "main_sha_now": cur,
           "main_moved": cur != sha, "private": pinned.private, "max_frames_cap": cap, "smoke": smoke}
    assert pinned.sha == sha, (pinned.sha, sha)
    info = json.loads(Path(hf_hub_download(repo, "meta/info.json", repo_type="dataset", revision=sha, local_dir=str(work / "src"))).read_text())
    vkeys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    cams = [k[len("observation.images."):] for k in vkeys]
    assert all(key(c) == k for c, k in zip(cams, vkeys)), vkeys
    shapes = {c: info["features"][key(c)]["shape"][:2] for c in cams}
    fps = info["fps"]
    assert abs(fps - round(fps)) < 1e-6, fps
    fps = int(round(fps))

    def dim(name):
        f = info["features"].get(name)
        return int(f["shape"][0]) if f and f.get("shape") else 14

    res.update(cams=cams, video_keys=vkeys, shapes=shapes, fps=fps, state_dim=dim("observation.state"), action_dim=dim("action"),
               robot_type=info.get("robot_type"), source_info={k: info.get(k) for k in ("codebase_version", "total_episodes", "total_frames",
                                                                                            "data_files_size_in_mb", "video_files_size_in_mb", "chunks_size")},
               source_video_info={c: info["features"][key(c)].get("info") for c in cams},
               stack_layout=stack_layout(shapes, cams))

    df, rd = read_episodes(repo, sha, vkeys)
    res["episodes_meta_read"] = rd
    log(f"{ds}: {len(df)} episodes, cameras {cams}, shapes {shapes}, fps {fps}")
    real = {"episodes": int(len(df)), "frames": int(df.length.sum()), "episode_length_median": float(df.length.median()), "per_camera": {}}
    pooled = []
    for c in cams:
        k = key(c)
        pair = list(zip(df[col(k, "chunk_index")].astype(int), df[col(k, "file_index")].astype(int)))
        df[f"f_{c}"] = pair
        fr = df.groupby(f"f_{c}").length.sum()
        pooled += fr.tolist()
        real["per_camera"][c] = {"video_files": int(len(fr)), "frames_per_file_median": float(fr.median()), "mean": float(fr.mean()),
                                 "p10": float(fr.quantile(0.1)), "p90": float(fr.quantile(0.9)), "max": int(fr.max())}
    real["video_files_total"] = int(sum(v["video_files"] for v in real["per_camera"].values()))
    real["video_files_per_camera"] = {c: real["per_camera"][c]["video_files"] for c in cams}
    real["frames_per_file_median_all_cameras"] = float(np.median(pooled))
    dpairs = set(zip(df["data/chunk_index"].astype(int), df["data/file_index"].astype(int)))
    real["parquet_data_files_from_episodes"] = len(dpairs)
    try:
        tree = api.list_repo_tree(repo, path_in_repo="data", repo_type="dataset", revision=sha, recursive=True)
        real["parquet_data_files_in_repo"] = sum(1 for t in tree if getattr(t, "path", "").endswith(".parquet"))
    except Exception as exc:  # noqa: BLE001
        real["parquet_data_files_in_repo"] = f"error: {exc}"
    try:  # what StreamingLeRobotDataset gets: load_dataset(..., streaming=True, data_files="data/*/*.parquet").num_shards
        import datasets

        t0 = time.perf_counter()
        hd = datasets.load_dataset(repo, split="train", streaming=True, data_files="data/*/*.parquet", revision=sha)
        real["hf_streaming_num_shards"] = int(hd.num_shards)
        real["hf_streaming_num_shards_s"] = round(time.perf_counter() - t0, 1)
    except Exception as exc:  # noqa: BLE001
        real["hf_streaming_num_shards"] = f"error: {type(exc).__name__}: {exc}"[:300]
    res["real"] = real
    log(f"real: {json.dumps(real)[:600]}")

    # ---- slice
    first = {c: min(df[f"f_{c}"]) for c in cams}
    in_first = np.all([df[f"f_{c}"].map(lambda p, c=c: p == first[c]).to_numpy() for c in cams], axis=0)
    eps, total, cap_applied = [], 0, False
    for i in range(len(df)):
        if not in_first[i]:
            break
        L = int(df.length.iloc[i])
        if eps and total + L > cap:
            cap_applied = True
            break
        r = df.iloc[i]
        eps.append({"episode": int(r.episode_index), "length": L,
                    "src": {c: {"path": f"videos/{key(c)}/chunk-{first[c][0]:03d}/file-{first[c][1]:03d}.mp4",
                                "from_timestamp": float(r[col(key(c), "from_timestamp")]), "to_timestamp": float(r[col(key(c), "to_timestamp")])}
                            for c in cams}})
        total += L
    run_all = 0  # consecutive episodes in the first files of all cameras (without the cap)
    for i in range(len(df)):
        if not in_first[i]:
            break
        run_all += int(df.length.iloc[i])
    for e in eps:
        for c in cams:
            s = e["src"][c]
            assert abs((s["to_timestamp"] - s["from_timestamp"]) * fps - e["length"]) < 0.5, (e["episode"], c, s)
    res["slice"] = {"episodes": eps, "n_episodes": len(eps), "frames": total, "cap_applied": cap_applied,
                    "consecutive_frames_in_first_files_all_cameras": run_all,
                    "first_file_frames": {c: int(df.length[df[f"f_{c}"].map(lambda p, c=c: p == first[c])].sum()) for c in cams},
                    "first_file": {c: list(first[c]) for c in cams},
                    "slice_over_real_median": total / real["frames_per_file_median_all_cameras"]}
    log(f"slice: {len(eps)} episodes, {total} frames (cap {cap}, applied {cap_applied}); real median frames/file "
        f"{real['frames_per_file_median_all_cameras']:.0f}")
    rng = random.Random(4)
    res["check_indices"] = sorted(rng.sample(range(total), min(env_int("R4_CHECK_N", "20"), total)))

    # ---- downloads
    t0 = time.perf_counter()
    src = {}
    for c in cams:
        rel = eps[0]["src"][c]["path"]
        p = hf_hub_download(repo, rel, repo_type="dataset", revision=sha, local_dir=str(work / "src"))
        src[c] = {"rel": rel, "path": p, "bytes": os.path.getsize(p), "moov": common.moov_info(p)}
    res["source_files"] = src
    res["download"] = {"files": len(src), "bytes": sum(v["bytes"] for v in src.values()), "s": round(time.perf_counter() - t0, 1)}
    log(f"downloaded {len(src)} source files, {res['download']['bytes'] / 1e6:.0f} MB in {res['download']['s']}s")

    # ---- LeRobot default encoder (installed code)
    from lerobot.configs.video import VideoEncoderConfig

    vc = VideoEncoderConfig()
    res["encoder_defaults"] = {"repr": repr(vc), "vcodec": vc.vcodec, "pix_fmt": vc.pix_fmt, "g": vc.g, "crf": vc.crf, "preset": vc.preset,
                               "fast_decode": vc.fast_decode, "codec_options": {k: str(v) for k, v in vc.get_codec_options().items()}}
    import lerobot

    res["lerobot_file"] = lerobot.__file__
    res["lerobot_sha"] = os.environ.get("LEROBOT_SHA", "?")
    log(f"encoder defaults: {res['encoder_defaults']}")
    write_json(D / "slice.json", res)


if __name__ == "__main__":
    main()
