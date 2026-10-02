"""Stage prep2 (round 2): build what the new arms read, from the round-1 variants (no re-encode).

1. Download the round-1 ckpt_full SEP (3 groups x 3 cameras) and STACK (3 groups) files at the pinned revision.
2. Option C, multi-track MP4: per group, ffmpeg stream copy of the 3 SEP files into ONE MP4 with 3 video tracks
   (`-map 0:v -map 1:v -map 2:v -c copy -movflags +faststart`). Same packets, no re-encode.
   Byte interleaving from ffprobe (packet pos per stream): for each timestamp, the byte distance between the 3
   cameras' packets. Muxer variants on group 0: default mov muxer, `-max_interleave_delta 0`, fragmented
   (`-frag_duration` 2 frames). Camera identity: TorchCodec VideoDecoder(file, stream_index=i) vs the SEP file of
   camera i, bit-exact, on random frames of every group. The default-muxer files are uploaded next to the other
   variants (variants/<round-1 run>/ckpt_full/MULTI/), new commit recorded.
3. Real LeRobot v3 datasets for the streaming arm (private repos): SEP = 3 video keys, STACK = 1 video key
   (224x672 stacked frame). PREP_NCOPY video files per camera, each a copy of one ckpt_full group (hard links
   locally; the Hub dedups the bytes), one parquet data file per video file, episode lengths from selection.json,
   observation.state / action float32[14] (random values), timestamps episode-relative, videos/*/from_timestamp
   file-relative. Uploaded with upload_folder; commit recorded.

Usage: python s10_prep.py <selection.json> <work_dir> <out_dir>
Env: R1_RUN (default 20260925T025221Z), RESULTS_REPO, PREP_NCOPY (8), PREP_SEP_REPO, PREP_STACK_REPO, PREP_ID_N (40)
"""

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from common import pin_threads, write_json, pct

pin_threads(2)

CAMS = ["top", "left_wrist", "right_wrist"]


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd[:4]}... failed: {r.stderr[-2000:]}")
    return r.stdout


def r1_upload(repo, run):
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(repo, f"results/{run}/upload/upload.json", repo_type="dataset")).read_text())


def download_variants(up, work, fam):
    from huggingface_hub import hf_hub_download

    t0 = time.perf_counter()
    got = []
    for f in up["files"]:
        if f["family"] != fam or f["layout"] not in ("SEP", "STACK"):
            continue
        p = hf_hub_download(up["repo"], f["path_in_repo"], repo_type="dataset", revision=up["revision"], local_dir=str(work / "dl"))
        assert os.path.getsize(p) == f["bytes"], (p, os.path.getsize(p), f["bytes"])
        got.append({"rel": f["rel"], "path": p, "bytes": f["bytes"], "frames": f["frames"], "sha256": f["sha256"]})
    return got, time.perf_counter() - t0


def interleave(path):
    """ffprobe packets -> per timestamp: span (min pos .. max pos+size) and gap (max pos - min pos) across streams."""
    out = sh(["ffprobe", "-v", "error", "-show_entries", "packet=stream_index,pts,pos,size", "-of", "json", path])
    by = {}
    order = []
    for pk in json.loads(out)["packets"]:  # json: field order of ffprobe csv output is not the requested order
        if "pos" not in pk or "pts" not in pk:
            continue
        si, pts, pos, size = int(pk["stream_index"]), int(pk["pts"]), int(pk["pos"]), int(pk["size"])
        by.setdefault(pts, {})[si] = (pos, size)
        order.append(si)
    spans, gaps = [], []
    for d in by.values():
        if len(d) < 3:
            continue
        lo = min(p for p, _ in d.values())
        hi = max(p + s for p, s in d.values())
        spans.append(hi - lo)
        gaps.append(max(p for p, _ in d.values()) - lo)
    runs = 1 + sum(1 for a, b in zip(order, order[1:]) if a != b)
    return {"timestamps": len(spans), "span_bytes_p50": pct(spans, 0.5), "span_bytes_p95": pct(spans, 0.95),
            "span_bytes_max": max(spans) if spans else None, "gap_bytes_p50": pct(gaps, 0.5), "gap_bytes_p95": pct(gaps, 0.95),
            "packets": len(order), "same_stream_runs": runs, "mean_packets_per_run": len(order) / runs if runs else None}


def build_multi(seps, dst, extra):
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for p in seps:
        cmd += ["-i", p]
    cmd += ["-map", "0:v", "-map", "1:v", "-map", "2:v", "-c", "copy", *extra, dst]
    sh(cmd)
    return os.path.getsize(dst)


def identity_check(multi, seps, n, seed):
    """TorchCodec per-track decode of the multi-track file == decode of the camera's SEP file (bit-exact)."""
    import torch
    from torchcodec.decoders import VideoDecoder

    ref = [VideoDecoder(p, seek_mode="approximate") for p in seps]
    m = [VideoDecoder(multi, stream_index=i, seek_mode="approximate") for i in range(3)]
    nf = ref[0].metadata.num_frames
    meta_ok = all(d.metadata.num_frames == nf for d in m + ref)
    rng = random.Random(seed)
    checked = bad = 0
    cross = 0  # frames of track i equal to a DIFFERENT camera's SEP frame (camera swap check)
    for _ in range(n):
        f = rng.randrange(nf)
        fr = [r.get_frames_at(indices=[f]).data for r in ref]
        for i in range(3):
            x = m[i].get_frames_at(indices=[f]).data
            checked += 1
            bad += int(not torch.equal(x, fr[i]))
            cross += sum(int(torch.equal(x, fr[j])) for j in range(3) if j != i)
    return {"frames_checked": checked, "mismatch": bad, "equal_to_other_camera": cross, "num_frames_ok": meta_ok,
            "num_frames": nf, "pass": bad == 0 and meta_ok}


# ------------------------------------------------------------------ v3 datasets
def build_dataset(kind, groups, files, ncopy, root, fps):
    """kind SEP | STACK. files: {(gi, cam or None): local path}. Returns stats."""
    import datasets
    import numpy as np
    import pandas as pd

    root = Path(root)
    if root.exists():
        subprocess.run(["rm", "-rf", str(root)], check=True)
    keys = [f"observation.images.{c}" for c in CAMS] if kind == "SEP" else ["observation.images.stack"]
    rng = np.random.default_rng(0)
    ep_rows = []
    gidx = 0
    ep_idx = 0
    feats = datasets.Features({"observation.state": datasets.Sequence(datasets.Value("float32"), length=14),
                               "action": datasets.Sequence(datasets.Value("float32"), length=14),
                               "timestamp": datasets.Value("float32"), "frame_index": datasets.Value("int64"),
                               "episode_index": datasets.Value("int64"), "index": datasets.Value("int64"),
                               "task_index": datasets.Value("int64")})
    for fi in range(ncopy):
        gi = fi % len(groups)
        g = groups[gi]
        lens = [e["length"] for e in g["episodes"]]
        assert sum(lens) == g["frames"], (g["name"], sum(lens), g["frames"])
        for k, cam in zip(keys, CAMS if kind == "SEP" else [None]):
            dst = root / "videos" / k / "chunk-000" / f"file-{fi:03d}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(files[(gi, cam)], dst)
        n = g["frames"]
        cols = {"observation.state": rng.standard_normal((n, 14), dtype=np.float32).tolist(),
                "action": rng.standard_normal((n, 14), dtype=np.float32).tolist(),
                "timestamp": [], "frame_index": [], "episode_index": [], "index": list(range(gidx, gidx + n)), "task_index": [0] * n}
        vfrom = 0
        for L in lens:
            cols["timestamp"] += [j / fps for j in range(L)]
            cols["frame_index"] += list(range(L))
            cols["episode_index"] += [ep_idx] * L
            row = {"episode_index": ep_idx, "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
                   "data/chunk_index": 0, "data/file_index": fi, "dataset_from_index": gidx, "dataset_to_index": gidx + L,
                   "tasks": ["pick and place"], "length": L}
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
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["pick and place"], name="task")).to_parquet(root / "meta" / "tasks.parquet")
    vinfo = {"video.height": 224 if kind == "SEP" else 672, "video.width": 224, "video.codec": "av1", "video.pix_fmt": "yuv420p",
             "video.is_depth_map": False, "video.fps": fps, "video.channels": 3, "has_audio": False, "video.g": 2, "video.crf": 30,
             "video.preset": 12}
    features = {"observation.state": {"dtype": "float32", "shape": [14], "names": None},
                "action": {"dtype": "float32", "shape": [14], "names": None}}
    for k in keys:
        features[k] = {"dtype": "video", "shape": [vinfo["video.height"], 224, 3], "names": ["height", "width", "channels"], "info": vinfo}
    for k, dt in (("timestamp", "float32"), ("frame_index", "int64"), ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64")):
        features[k] = {"dtype": dt, "shape": [1], "names": None}
    info = {"codebase_version": "v3.0", "robot_type": "yam_bimanual", "total_episodes": ep_idx, "total_frames": gidx, "total_tasks": 1,
            "chunks_size": 1000, "data_files_size_in_mb": 100, "video_files_size_in_mb": 200, "fps": fps,
            "splits": {"train": f"0:{ep_idx}"}, "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4", "features": features}
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    return {"kind": kind, "video_keys": keys, "files_per_key": ncopy, "episodes": ep_idx, "frames": gidx}


def upload_dataset(repo, root, kind):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    t0 = time.perf_counter()
    ci = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(root), delete_patterns=["data/**", "videos/**", "meta/**"],
                           commit_message=f"DATA-11 round 2 streaming test dataset ({kind}, copies of round-1 ckpt_full variants)")
    info = api.dataset_info(repo, revision=ci.oid, files_metadata=True)
    remote = {s.rfilename: s.size for s in info.siblings}
    local = {str(p.relative_to(root)): p.stat().st_size for p in Path(root).rglob("*") if p.is_file()}
    bad = [k for k, v in local.items() if remote.get(k) != v]
    return {"repo": repo, "commit": ci.oid, "upload_s": time.perf_counter() - t0, "files": len(local), "readback_mismatch": bad,
            "private": info.private}


def fetch_mode(run, repo, work, D):
    """Reuse a previous prep2 (uploads already done): download its SEP / STACK / MULTI files for local reference
    decodes, check sizes, write prep.json with this job's local paths."""
    from huggingface_hub import hf_hub_download

    res = json.loads(Path(hf_hub_download(repo, f"results/{run}/prep/prep.json", repo_type="dataset")).read_text())
    res["prep_from"] = run
    root = work / "dl"
    t0 = time.perf_counter()
    bad = []
    for rel, size in res["sizes"].items():
        rev = res["multi"]["upload"]["revision"] if "/MULTI/" in rel else res["r1_revision"]
        p = hf_hub_download(repo, f"{res['r1_prefix']}/{rel}", repo_type="dataset", revision=rev, local_dir=str(root))
        if os.path.getsize(p) != size:
            bad.append(rel)
    res["local_root"] = str(root / res["r1_prefix"])
    res["fetch_mode"] = {"files": len(res["sizes"]), "s": time.perf_counter() - t0, "size_mismatch": bad}
    write_json(D / "prep.json", res)
    print(f"prep reused from {run}: {len(res['sizes'])} files in {res['fetch_mode']['s']:.0f}s, mismatches {bad}", flush=True)
    return 0 if not bad else 1


def main():
    sel_path, work, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    D = out / "prep"
    D.mkdir(parents=True, exist_ok=True)
    repo = os.environ.get("RESULTS_REPO", "hadriencornier/lerobot-data11-bench")
    r1 = os.environ.get("R1_RUN", "20260925T025221Z")
    ncopy = int(os.environ.get("PREP_NCOPY", "8"))
    idn = int(os.environ.get("PREP_ID_N", "40"))
    sel = json.loads(Path(sel_path).read_text())
    if os.environ.get("PREP_FROM"):
        sys.exit(fetch_mode(os.environ["PREP_FROM"], repo, work, D))
    fam = os.environ.get("PREP_FAMILY", "ckpt_full")  # ckpt_1k for local tests
    groups = sel["families"][fam]
    up = r1_upload(repo, r1)
    res = {"family": fam, "r1_run": r1, "r1_revision": up["revision"], "r1_prefix": up["prefix"]}
    files, dl_s = download_variants(up, work, fam)
    res["download"] = {"files": len(files), "bytes": sum(f["bytes"] for f in files), "s": dl_s}
    print(f"downloaded {len(files)} files, {res['download']['bytes'] / 1e9:.2f} GB in {dl_s:.0f}s", flush=True)
    for f in files:  # integrity: LFS sha256 recorded in round 1
        h = hashlib.sha256()
        with open(f["path"], "rb") as fh:
            while b := fh.read(1 << 24):
                h.update(b)
        assert h.hexdigest() == f["sha256"], f["rel"]
    loc = {}
    for f in files:
        name = Path(f["rel"]).name[:-4]
        g = int(name.split("_g")[1][:3])
        cam = name.split("__")[1] if "__" in name else None
        loc[(g, cam)] = f["path"]

    res.update(repo=repo, groups=groups, local_root=str(work / "dl" / up["prefix"]),
               sizes={f["rel"]: f["bytes"] for f in files})

    # ---- option C
    mdir = work / "multi" / fam / "MULTI"
    mdir.mkdir(parents=True, exist_ok=True)
    res["multi"] = {"groups": [], "muxer_variants": {}}
    for gi in range(len(groups)):
        seps = [loc[(gi, c)] for c in CAMS]
        dst = str(mdir / f"{fam}_g{gi:03d}.mp4")
        size = build_multi(seps, dst, ["-movflags", "+faststart"])
        il = interleave(dst)
        idc = identity_check(dst, seps, idn, 100 + gi)
        sep_bytes = sum(os.path.getsize(p) for p in seps)
        row = {"group": gi, "file": dst, "bytes": size, "sep_bytes_sum": sep_bytes, "overhead_bytes": size - sep_bytes,
               "interleave": il, "identity": idc}
        res["multi"]["groups"].append(row)
        print(f"MULTI g{gi}: {size / 1e6:.1f} MB (SEP sum {sep_bytes / 1e6:.1f}), interleave span p50 {il['span_bytes_p50']:.0f} "
              f"p95 {il['span_bytes_p95']:.0f} B, identity {idc}", flush=True)
    seps0 = [loc[(0, c)] for c in CAMS]
    tmp = work / "multi_variants"
    tmp.mkdir(exist_ok=True)
    for name, extra in (("max_interleave_delta_0", ["-max_interleave_delta", "0", "-movflags", "+faststart"]),
                        ("fragmented_2frames", ["-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-frag_duration", "66667"])):
        p = str(tmp / f"{name}.mp4")
        try:
            size = build_multi(seps0, p, extra)
            v = {"args": extra, "bytes": size, "interleave": interleave(p)}
            try:
                v["identity"] = identity_check(p, seps0, 10, 7)
            except Exception as exc:  # noqa: BLE001
                v["identity"] = f"decode failed: {type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001
            v = {"args": extra, "error": str(exc)[-500:]}
        res["multi"]["muxer_variants"][name] = v
        print(f"muxer variant {name}: {json.dumps(v, default=str)[:400]}", flush=True)
    write_json(D / "prep.json", res)

    from huggingface_hub import HfApi

    api = HfApi()
    t0 = time.perf_counter()
    if os.environ.get("PREP_SKIP_MULTI_UPLOAD") == "1":
        res["multi"]["upload"] = {"skipped": True, "readback_ok": True, "revision": up["revision"], "hf_url_base": up["hf_url_base"]}
    else:
        ci = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(work / "multi"), path_in_repo=up["prefix"],
                               allow_patterns=[f"{fam}/MULTI/*.mp4"],
                               commit_message=f"DATA-11 round 2: option C multi-track MP4 (stream copy of {fam} SEP), job {os.environ.get('RUN_NAME')}")
        rev2 = ci.oid
        info = api.dataset_info(repo, revision=rev2, files_metadata=True)
        remote = {s.rfilename: s.size for s in info.siblings}
        mf = []
        for row in res["multi"]["groups"]:
            pir = f"{up['prefix']}/{fam}/MULTI/{Path(row['file']).name}"
            mf.append({"path_in_repo": pir, "bytes": row["bytes"], "remote_bytes": remote.get(pir)})
        res["multi"]["upload"] = {"revision": rev2, "s": time.perf_counter() - t0, "files": mf,
                                  "readback_ok": all(x["bytes"] == x["remote_bytes"] for x in mf),
                                  "hf_url_base": f"hf://datasets/{repo}@{rev2}/{up['prefix']}"}
        print(f"MULTI uploaded at {rev2}: {res['multi']['upload']['readback_ok']}", flush=True)
    write_json(D / "prep.json", res)

    ldir = Path(res["local_root"]) / fam / "MULTI"
    ldir.mkdir(parents=True, exist_ok=True)
    for row in res["multi"]["groups"]:
        name = Path(row["file"]).name
        if not (ldir / name).exists():
            os.link(row["file"], ldir / name)
        res["sizes"][f"{fam}/MULTI/{name}"] = row["bytes"]
    write_json(D / "prep.json", res)
    if os.environ.get("PREP_SKIP_STREAM") == "1":
        res["ok"] = all(g["identity"]["pass"] for g in res["multi"]["groups"])
        write_json(D / "prep.json", res)
        sys.exit(0 if res["ok"] else 1)

    # ---- streaming datasets
    res["stream"] = {}
    for kind, env, default in (("SEP", "PREP_SEP_REPO", "hadriencornier/lerobot-data11-stream-sep"),
                               ("STACK", "PREP_STACK_REPO", "hadriencornier/lerobot-data11-stream-stack")):
        root = work / "stream" / kind
        st = build_dataset(kind, groups, loc, ncopy, root, sel["fps"])
        st.update(upload_dataset(os.environ.get(env, default), root, kind))
        res["stream"][kind] = st
        print(f"stream dataset {kind}: {json.dumps(st)[:400]}", flush=True)
        write_json(D / "prep.json", res)
    ok = all(g["identity"]["pass"] for g in res["multi"]["groups"]) and res["multi"]["upload"]["readback_ok"] and all(
        not s["readback_mismatch"] for s in res["stream"].values())
    res["ok"] = ok
    write_json(D / "prep.json", res)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
