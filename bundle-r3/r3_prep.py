"""Round 3 prep: equal-size STACK and MULTI files, cut from the round-1 STACK and round-2 MULTI files.

Why: round 2's STACK / MULTI files held the same frames as one SEP file, so they were about 3x bigger in bytes, and a
laptop test showed that request latency grows with file size. LeRobot rolls files over at 200 MB, so real stacked
files would be about SEP-sized.

1. Download (at the round-2 MULTI revision, which also holds the round-1 files) the ckpt_full SEP (3 groups x 3 cams),
   STACK (3) and MULTI (3) files. Sizes checked; SEP / STACK sha256 checked against round 1.
2. Cut points per group: R3_PARTS (3) frame ranges with about equal STACK bytes. A cut starts only at a frame that is a
   keyframe in STACK and in all 3 MULTI tracks (GOP is 2), so a stream copy is possible.
3. Cut = packet stream copy with PyAV/libavformat (no re-encode), in a subprocess: packets [a, b) of every video
   stream, pts / dts shifted by the pts of packet a, `movflags +faststart`. Checks per cut: packet count == b - a per
   stream, pts == i / fps exactly, keyframe flags equal to the source's, every packet's bytes equal to the source
   packet (sha256 of the concatenation), and a TorchCodec decode of R3_ID_N random frames (plus first and last)
   bit-exact against the source file at frame a + i (every MULTI track against its SEP camera too).
4. Upload the cuts to variants/r3-<run>/ckpt_full/{STACK,MULTI}/ with cuts.json; read back sizes at the commit.
5. v3 datasets for the #3917 stage (private): SEP = 3 video keys over the SEP group files, STACK = 1 video key over the
   STACK cuts. Both have the SAME episodes: the source episodes of each group, split at the cut points (an episode
   crossing a cut becomes two). R3_NCOPY copies of the 3 groups (hard links; the Hub dedups the bytes). A third local
   dataset (not uploaded) points 3 video keys at the MULTI cuts, to test #3917 on a multi-track file.
6. prep_eq.json (equal-size layouts) and prep_orig.json (round-2 files, control) for r3_access.py.

Usage: python r3_prep.py <selection.json> <work_dir> <out_dir>      |  python r3_prep.py --cut <json args>  (internal)
Env: RESULTS_REPO, R1_RUN (20260925T025221Z), R2_MULTI_REV (a3d31680...), R3_PARTS (3), R3_ID_N (30), R3_NCOPY (24),
     R3_FAMILY (ckpt_full; ckpt_1k for local tests), R3_BUILD_MULTI=1 (build MULTI locally from SEP, local tests),
     NO_UPLOAD=1, R3_SEP_REPO, R3_STACK_REPO
"""

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

CAMS = ["top", "left_wrist", "right_wrist"]
R2_MULTI_REV = "a3d31680dba3ed626ee7c5afca85753b23941b52"


# ------------------------------------------------------------------ PyAV side (subprocess: no torchcodec here)
def av_packets(path):
    """[(stream_pos, [(pts, dts, size, key, sha256-able bytes len)], time_base)] per video stream, in decode order."""
    import av

    out = {}
    with av.open(path) as c:
        vs = list(c.streams.video)
        for s in vs:
            out[s.index] = {"tb": [s.time_base.numerator, s.time_base.denominator], "pts": [], "size": [], "key": []}
        for p in c.demux(*vs):
            if p.size == 0 or p.pts is None:
                continue
            d = out[p.stream.index]
            d["pts"].append(int(p.pts))
            d["size"].append(p.size)
            d["key"].append(bool(p.is_keyframe))
    return [out[k] for k in sorted(out)]


def av_cut(src, dst, a, b):
    """Stream copy of packets [a, b) of every video stream of src into dst; returns per-stream packet sha256."""
    import av

    hs, n, off = {}, {}, {}
    with av.open(src) as inp:
        vs = list(inp.streams.video)
        out = av.open(dst, "w", format="mp4", options={"movflags": "+faststart"})
        omap = {s.index: out.add_stream_from_template(s, opaque=True) for s in vs}
        for s in vs:
            n[s.index], hs[s.index] = 0, hashlib.sha256()
        for p in inp.demux(*vs):
            if p.size == 0 or p.pts is None:
                continue
            si = p.stream.index
            k = n[si]
            n[si] += 1
            if k < a:
                continue
            if k >= b:
                if all(n[x] > b for x in n):
                    break
                continue
            if k == a:
                off[si] = p.pts
                assert p.is_keyframe, (src, si, a)
            hs[si].update(bytes(p))
            p.pts -= off[si]
            p.dts -= off[si]
            p.stream = omap[si]
            out.mux(p)
        out.close()
    return [hs[k].hexdigest() for k in sorted(hs)]


def av_src_hash(src, a, b):
    """sha256 of the concatenated packet bytes [a, b) per video stream of src (for the cut check)."""
    import av

    hs, n = {}, {}
    with av.open(src) as inp:
        vs = list(inp.streams.video)
        for s in vs:
            n[s.index], hs[s.index] = 0, hashlib.sha256()
        for p in inp.demux(*vs):
            if p.size == 0 or p.pts is None:
                continue
            si = p.stream.index
            k = n[si]
            n[si] += 1
            if a <= k < b:
                hs[si].update(bytes(p))
    return [hs[k].hexdigest() for k in sorted(hs)]


def av_main(args):
    op = args["op"]
    if op == "packets":
        r = av_packets(args["src"])
    elif op == "cut":
        r = av_cut(args["src"], args["dst"], args["a"], args["b"])
    elif op == "cut_packets":  # packet tables of the cut, and of the source packets, in one process
        r = {"cut": av_packets(args["dst"]), "cut_hash": av_src_hash(args["dst"], 0, 10**9)}
    else:
        raise ValueError(op)
    sys.stdout.write(json.dumps(r))


def av_call(**args):
    r = subprocess.run([sys.executable, __file__, "--av", json.dumps(args)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"av {args.get('op')} failed: {r.stderr[-3000:]}")
    return json.loads(r.stdout)


# ------------------------------------------------------------------ main process (TorchCodec for checks)
def choose_cuts(stack_pk, multi_pk, parts):
    """Frame index cut points: about equal STACK bytes per part, each start a keyframe in STACK and all MULTI tracks."""
    n = len(stack_pk["size"])
    assert all(len(t["size"]) == n for t in multi_pk), "frame counts differ between STACK and MULTI"
    ok = [i for i in range(n) if stack_pk["key"][i] and all(t["key"][i] for t in multi_pk)]
    cum = [0]
    for s in stack_pk["size"]:
        cum.append(cum[-1] + s)
    starts = [0]
    for j in range(1, parts):
        target = cum[-1] * j / parts
        best = min(ok, key=lambda i: abs(cum[i] - target))
        starts.append(best)
    starts = sorted(set(starts))
    return [[s, e] for s, e in zip(starts, starts[1:] + [n])], cum


def verify_cut(src, dst, a, b, fps, src_pk, src_hash, id_n, seed, sep_refs=None):
    """Checks listed in the module docstring. sep_refs: [SEP path per track] for MULTI (camera identity)."""
    import torch
    from torchcodec.decoders import VideoDecoder

    cp = av_call(op="cut_packets", dst=dst)
    res = {"packets_ok": True, "pts_ok": True, "key_ok": True, "bytes_ok": cp["cut_hash"] == src_hash}
    for t, (c, s) in enumerate(zip(cp["cut"], src_pk)):
        if len(c["pts"]) != b - a:
            res["packets_ok"] = False
        tb = c["tb"][0] / c["tb"][1]
        if any(abs(p * tb - i / fps) > 1e-6 for i, p in enumerate(c["pts"])):
            res["pts_ok"] = False
        if c["key"] != s["key"][a:b]:
            res["key_ok"] = False
        res[f"track{t}_time_base"] = c["tb"]
    nt = len(src_pk)
    rng = random.Random(seed)
    frames = sorted({0, b - a - 1, *[rng.randrange(b - a) for _ in range(id_n)]})
    bad = cross = checked = 0
    for t in range(nt):
        dc = VideoDecoder(dst, stream_index=t if nt > 1 else None, seek_mode="exact")
        ds = VideoDecoder(src, stream_index=t if nt > 1 else None, seek_mode="exact")
        dr = VideoDecoder(sep_refs[t], seek_mode="exact") if sep_refs else None
        if dc.metadata.num_frames != b - a:
            res["packets_ok"] = False
        x = dc.get_frames_at(indices=frames)
        y = ds.get_frames_at(indices=[a + f for f in frames])
        for i in range(len(frames)):
            checked += 1
            bad += int(not torch.equal(x.data[i], y.data[i]))
            if abs(float(x.pts_seconds[i]) - frames[i] / fps) > 1e-6:
                res["pts_ok"] = False
        if dr is not None:
            z = dr.get_frames_at(indices=[a + f for f in frames])
            bad += sum(int(not torch.equal(x.data[i], z.data[i])) for i in range(len(frames)))
            checked += len(frames)
    if sep_refs:  # camera swap check: track t never equals another camera's SEP frame
        for t in range(nt):
            dc = VideoDecoder(dst, stream_index=t, seek_mode="exact")
            x = dc.get_frames_at(indices=frames[:5])
            for u in range(nt):
                if u != t:
                    z = VideoDecoder(sep_refs[u], seek_mode="exact").get_frames_at(indices=[a + f for f in frames[:5]])
                    cross += sum(int(torch.equal(x.data[i], z.data[i])) for i in range(len(frames[:5])))
    res.update(frames_checked=checked, frame_mismatch=bad, equal_to_other_camera=cross)
    res["pass"] = all(res[k] for k in ("packets_ok", "pts_ok", "key_ok", "bytes_ok")) and bad == 0 and cross == 0
    return res


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def split_episodes(lens, cuts):
    """Source episode lengths of one group -> [(start, length, part)] split at the cut starts."""
    bounds = sorted({c[0] for c in cuts} | {c[1] for c in cuts})
    out, s = [], 0
    for L in lens:
        e = s + L
        cur = s
        for x in bounds:
            if cur < x < e:
                out.append((cur, x - cur))
                cur = x
        out.append((cur, e - cur))
        s = e
    res = []
    for st, L in out:
        p = next(i for i, (a, b) in enumerate(cuts) if a <= st < b)
        assert st + L <= cuts[p][1]
        res.append((st, L, p))
    return res


def build_dataset(kind, groups, cuts, files, ncopy, root, fps):
    """kind SEP | STACK | MULTI (MULTI: local only). files: SEP {(g, cam): path}, STACK / MULTI {(g, part): path}."""
    import datasets
    import numpy as np
    import pandas as pd

    root = Path(root)
    if root.exists():
        subprocess.run(["rm", "-rf", str(root)], check=True)
    keys = [f"observation.images.{c}" for c in CAMS] if kind in ("SEP", "MULTI") else ["observation.images.stack"]
    rng = np.random.default_rng(0)
    ep_rows, gidx, ep_idx = [], 0, 0
    feats = datasets.Features({"observation.state": datasets.Sequence(datasets.Value("float32"), length=14),
                               "action": datasets.Sequence(datasets.Value("float32"), length=14),
                               "timestamp": datasets.Value("float32"), "frame_index": datasets.Value("int64"),
                               "episode_index": datasets.Value("int64"), "index": datasets.Value("int64"),
                               "task_index": datasets.Value("int64")})
    ep_map = []  # episode -> (group, start frame in group, length)
    nparts = len(cuts[0])
    for ci in range(ncopy):
        gi = ci % len(groups)
        g = groups[gi]
        lens = [e["length"] for e in g["episodes"]]
        eps = split_episodes(lens, cuts[gi])
        if kind == "SEP":
            for k, cam in zip(keys, CAMS):
                dst = root / "videos" / k / "chunk-000" / f"file-{ci:03d}.mp4"
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.link(files[(gi, cam)], dst)
        else:
            for p in range(nparts):
                for k in keys:
                    dst = root / "videos" / k / "chunk-000" / f"file-{ci * nparts + p:03d}.mp4"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    os.link(files[(gi, p)], dst)
        n = g["frames"]
        cols = {"observation.state": rng.standard_normal((n, 14), dtype=np.float32).tolist(),
                "action": rng.standard_normal((n, 14), dtype=np.float32).tolist(),
                "timestamp": [], "frame_index": [], "episode_index": [], "index": list(range(gidx, gidx + n)), "task_index": [0] * n}
        for st, L, p in eps:
            cols["timestamp"] += [j / fps for j in range(L)]
            cols["frame_index"] += list(range(L))
            cols["episode_index"] += [ep_idx] * L
            row = {"episode_index": ep_idx, "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
                   "data/chunk_index": 0, "data/file_index": ci, "dataset_from_index": gidx, "dataset_to_index": gidx + L,
                   "tasks": ["pick and place"], "length": L}
            vfile, vfrom = (ci, st) if kind == "SEP" else (ci * nparts + p, st - cuts[gi][p][0])
            for k in keys:
                row[f"videos/{k}/chunk_index"] = 0
                row[f"videos/{k}/file_index"] = vfile
                row[f"videos/{k}/from_timestamp"] = vfrom / fps
                row[f"videos/{k}/to_timestamp"] = (vfrom + L) / fps
            ep_rows.append(row)
            ep_map.append((gi, st, L))
            gidx += L
            ep_idx += 1
        dp = root / "data" / "chunk-000" / f"file-{ci:03d}.parquet"
        dp.parent.mkdir(parents=True, exist_ok=True)
        datasets.Dataset.from_dict(cols, features=feats).to_parquet(str(dp))
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ep_rows).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["pick and place"], name="task")).to_parquet(root / "meta" / "tasks.parquet")
    h = 672 if kind == "STACK" else 224
    vinfo = {"video.height": h, "video.width": 224, "video.codec": "av1", "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
             "video.fps": fps, "video.channels": 3, "has_audio": False, "video.g": 2, "video.crf": 30, "video.preset": 12}
    features = {"observation.state": {"dtype": "float32", "shape": [14], "names": None},
                "action": {"dtype": "float32", "shape": [14], "names": None}}
    for k in keys:
        features[k] = {"dtype": "video", "shape": [h, 224, 3], "names": ["height", "width", "channels"], "info": vinfo}
    for k, dt in (("timestamp", "float32"), ("frame_index", "int64"), ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64")):
        features[k] = {"dtype": dt, "shape": [1], "names": None}
    info = {"codebase_version": "v3.0", "robot_type": "yam_bimanual", "total_episodes": ep_idx, "total_frames": gidx, "total_tasks": 1,
            "chunks_size": 1000, "data_files_size_in_mb": 100, "video_files_size_in_mb": 200, "fps": fps,
            "splits": {"train": f"0:{ep_idx}"}, "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4", "features": features}
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    lens = [m[2] for m in ep_map]
    return {"kind": kind, "video_keys": keys, "video_files_per_key": ncopy * (1 if kind == "SEP" else nparts), "copies": ncopy,
            "episodes": ep_idx, "frames": gidx, "ep_len_min": min(lens), "ep_len_max": max(lens),
            "ep_len_mean": sum(lens) / len(lens), "episode_map": ep_map, "root": str(root)}


def upload_dataset(repo, root, kind):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    t0 = time.perf_counter()
    ci = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(root), delete_patterns=["data/**", "videos/**", "meta/**"],
                           commit_message=f"DATA-11 round 3 #3917 test dataset ({kind}, equal-size files)")
    info = api.dataset_info(repo, revision=ci.oid, files_metadata=True)
    remote = {s.rfilename: s.size for s in info.siblings}
    local = {str(p.relative_to(root)): p.stat().st_size for p in Path(root).rglob("*") if p.is_file()}
    bad = [k for k, v in local.items() if remote.get(k) != v]
    return {"repo": repo, "commit": ci.oid, "upload_s": time.perf_counter() - t0, "files": len(local), "readback_mismatch": bad,
            "private": info.private}


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def main():
    sel_path, work, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    D = out / "prep"
    D.mkdir(parents=True, exist_ok=True)
    from common import pin_threads

    pin_threads(2)
    from huggingface_hub import HfApi, hf_hub_download

    repo = os.environ.get("RESULTS_REPO", "hadriencornier/lerobot-data11-bench")
    r1 = os.environ.get("R1_RUN", "20260925T025221Z")
    rev = os.environ.get("R2_MULTI_REV", R2_MULTI_REV)
    parts = int(os.environ.get("R3_PARTS", "3"))
    id_n = int(os.environ.get("R3_ID_N", "30"))
    ncopy = int(os.environ.get("R3_NCOPY", "24"))
    fam = os.environ.get("R3_FAMILY", "ckpt_full")
    no_upload = os.environ.get("NO_UPLOAD") == "1"
    run = os.environ.get("RUN_NAME", time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    r3pre = run if run.startswith("r3-") else f"r3-{run}"
    sel = json.loads(Path(sel_path).read_text())
    fps = sel["fps"]
    groups = sel["families"][fam]
    up = json.loads(Path(hf_hub_download(repo, f"results/{r1}/upload/upload.json", repo_type="dataset")).read_text())
    prefix = up["prefix"]  # variants/<r1>
    vroot = work / "dl"  # mirrors the repo: vroot/variants/...
    res = {"family": fam, "repo": repo, "r1_run": r1, "r1_revision": up["revision"], "multi_revision": rev, "parts": parts,
           "r3_prefix": f"variants/{r3pre}", "groups": groups, "fps": fps}

    # ---- 1. download
    t0 = time.perf_counter()
    loc = {}
    want = [f for f in up["files"] if f["family"] == fam and f["layout"] in ("SEP", "STACK")]
    for f in want:
        p = hf_hub_download(repo, f["path_in_repo"], repo_type="dataset", revision=up["revision"], local_dir=str(vroot))
        assert os.path.getsize(p) == f["bytes"], f["rel"]
        assert sha256(p) == f["sha256"], f["rel"]
        name = Path(f["rel"]).name[:-4]
        g = int(name.split("_g")[1][:3])
        loc[(f["layout"], g, name.split("__")[1] if "__" in name else None)] = p
    for gi in range(len(groups)):
        rel = f"{fam}/MULTI/{fam}_g{gi:03d}.mp4"
        if os.environ.get("R3_BUILD_MULTI") == "1":
            p = vroot / prefix / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", *sum([["-i", loc[("SEP", gi, c)]] for c in CAMS], []),
                            "-map", "0:v", "-map", "1:v", "-map", "2:v", "-c", "copy", "-movflags", "+faststart", str(p)], check=True)
            p = str(p)
        else:
            p = hf_hub_download(repo, f"{prefix}/{rel}", repo_type="dataset", revision=rev, local_dir=str(vroot))
        loc[("MULTI", gi, None)] = p
    res["download"] = {"files": len(loc), "bytes": sum(os.path.getsize(p) for p in loc.values()), "s": time.perf_counter() - t0}
    print(f"downloaded {len(loc)} files, {res['download']['bytes'] / 1e9:.2f} GB in {res['download']['s']:.0f}s", flush=True)

    # ---- 2-3. cuts
    cdir = vroot / "variants" / r3pre / fam
    res["cuts"], res["checks"] = [], []
    all_cuts = []
    for gi, g in enumerate(groups):
        spk = av_call(op="packets", src=loc[("STACK", gi, None)])[0]
        mpk = av_call(op="packets", src=loc[("MULTI", gi, None)])
        assert len(spk["size"]) == g["frames"], (gi, len(spk["size"]), g["frames"])
        cuts, cum = choose_cuts(spk, mpk, parts)
        all_cuts.append(cuts)
        key_every = sorted({j - i for i, j in zip([k for k, x in enumerate(spk["key"]) if x], [k for k, x in enumerate(spk["key"]) if x][1:])})
        for p, (a, b) in enumerate(cuts):
            for lay in ("STACK", "MULTI"):
                src = loc[(lay, gi, None)]
                dst = cdir / lay / f"{fam}_g{gi:03d}_p{p}.mp4"
                dst.parent.mkdir(parents=True, exist_ok=True)
                t1 = time.perf_counter()
                h = av_call(op="cut", src=src, dst=str(dst), a=a, b=b)
                src_pk = [spk] if lay == "STACK" else mpk
                chk = verify_cut(src, str(dst), a, b, fps, src_pk, h, id_n, 1000 * gi + p,
                                 sep_refs=[loc[("SEP", gi, c)] for c in CAMS] if lay == "MULTI" else None)
                row = {"layout": lay, "group": gi, "part": p, "from_frame": a, "to_frame": b, "frames": b - a,
                       "from_ts_in_source": a / fps, "source": str(Path(src).relative_to(vroot)),
                       "rel": str(dst.relative_to(vroot / "variants")), "bytes": os.path.getsize(dst), "sha256": sha256(dst),
                       "stack_bytes_in_range": cum[b] - cum[a], "cut_s": time.perf_counter() - t1, "check": chk}
                res["cuts"].append(row)
                print(f"cut {lay} g{gi} p{p} [{a},{b}) {row['bytes'] / 1e6:.1f} MB check pass={chk['pass']} "
                      f"({chk['frames_checked']} frames, mismatch {chk['frame_mismatch']}, bytes_ok {chk['bytes_ok']})", flush=True)
        res.setdefault("keyframe_spacing", {})[gi] = key_every
        write_json(D / "prep.json", res)
    # MULTI interleave of the cuts (same measure as round 2)
    from s10_prep import interleave

    res["multi_interleave"] = {r["rel"]: interleave(str(vroot / "variants" / r["rel"])) for r in res["cuts"] if r["layout"] == "MULTI"}
    sep_sizes = {f"{Path(prefix).name}/{fam}/SEP/{Path(loc[('SEP', g, c)]).name}": os.path.getsize(loc[("SEP", g, c)])
                 for g in range(len(groups)) for c in CAMS}
    res["sep_sizes"] = sep_sizes
    ok_cuts = all(r["check"]["pass"] for r in res["cuts"])
    res["cuts_ok"] = ok_cuts
    write_json(D / "prep.json", res)
    write_json(cdir.parent / "cuts.json", {k: res[k] for k in ("family", "r1_run", "multi_revision", "parts", "cuts", "fps")})

    # ---- 4. upload cuts
    if no_upload:
        rev3 = rev
        res["upload"] = {"skipped": True}
    else:
        api = HfApi()
        t1 = time.perf_counter()
        ci = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(vroot / "variants" / r3pre), path_in_repo=f"variants/{r3pre}",
                               allow_patterns=["*.mp4", "cuts.json"],
                               commit_message=f"DATA-11 round 3: equal-size STACK / MULTI cuts (stream copy), job {run}")
        rev3 = ci.oid
        info = api.dataset_info(repo, revision=rev3, files_metadata=True)
        remote = {s.rfilename: s.size for s in info.siblings}
        bad = [r["rel"] for r in res["cuts"] if remote.get(f"variants/{r['rel']}") != r["bytes"]]
        res["upload"] = {"revision": rev3, "s": time.perf_counter() - t1, "readback_mismatch": bad}
        print(f"cuts uploaded at {rev3}, readback mismatches {bad}", flush=True)
    write_json(D / "prep.json", res)

    # ---- 6. access prep files (paths relative to variants/)
    base = f"hf://datasets/{repo}@{rev3}/variants"
    r1n = Path(prefix).name

    def acc_prep(mode):
        gs = []
        for gi, g in enumerate(groups):
            e = {"frames": g["frames"], "name": g["name"], "sep": {c: f"{r1n}/{fam}/SEP/{fam}_g{gi:03d}__{c}.mp4" for c in CAMS}}
            if mode == "eq":
                e["cuts"] = all_cuts[gi]
                e["STACK"] = [f"{r3pre}/{fam}/STACK/{fam}_g{gi:03d}_p{p}.mp4" for p in range(len(all_cuts[gi]))]
                e["MULTI"] = [f"{r3pre}/{fam}/MULTI/{fam}_g{gi:03d}_p{p}.mp4" for p in range(len(all_cuts[gi]))]
            else:
                e["cuts"] = [[0, g["frames"]]]
                e["STACK"] = [f"{r1n}/{fam}/STACK/{fam}_g{gi:03d}.mp4"]
                e["MULTI"] = [f"{r1n}/{fam}/MULTI/{fam}_g{gi:03d}.mp4"]
            gs.append(e)
        rels = {x for e in gs for x in [*e["sep"].values(), *e["STACK"], *e["MULTI"]]}
        sizes = {x: os.path.getsize(vroot / "variants" / x) for x in rels}
        return {"repo": repo, "family": fam, "r1_prefix": "variants", "groups": gs, "sizes": sizes, "local_root": str(vroot / "variants"),
                "layout_mode": mode, "multi": {"upload": {"revision": rev3, "hf_url_base": base}}}

    write_json(D / "prep_eq.json", acc_prep("eq"))
    write_json(D / "prep_orig.json", acc_prep("orig"))

    # ---- 5. v3 datasets
    res["datasets"] = {}
    sep_files = {(g, c): loc[("SEP", g, c)] for g in range(len(groups)) for c in CAMS}
    cut_files = {lay: {(r["group"], r["part"]): str(vroot / "variants" / r["rel"]) for r in res["cuts"] if r["layout"] == lay}
                 for lay in ("STACK", "MULTI")}
    for kind, files, env, default in (("SEP", sep_files, "R3_SEP_REPO", "hadriencornier/lerobot-data11-r3-sep"),
                                      ("STACK", cut_files["STACK"], "R3_STACK_REPO", "hadriencornier/lerobot-data11-r3-stack"),
                                      ("MULTI", cut_files["MULTI"], None, None)):
        st = build_dataset(kind, groups, all_cuts, files, ncopy if kind != "MULTI" else 1, work / "r3ds" / kind, fps)
        if kind != "MULTI" and not no_upload:
            st.update(upload_dataset(os.environ.get(env, default), work / "r3ds" / kind, kind))
        else:
            st["upload"] = "local only" if kind == "MULTI" else "skipped (NO_UPLOAD)"
        res["datasets"][kind] = st
        print(f"dataset {kind}: {json.dumps({k: v for k, v in st.items() if k != 'episode_map'})[:400]}", flush=True)
        write_json(D / "prep.json", res)
    same_eps = [m[2] for m in res["datasets"]["SEP"]["episode_map"]] == [m[2] for m in res["datasets"]["STACK"]["episode_map"]]
    res["datasets_same_episodes"] = same_eps
    res["ok"] = ok_cuts and same_eps and (no_upload or (not res["upload"]["readback_mismatch"]
                                                        and all(not res["datasets"][k]["readback_mismatch"] for k in ("SEP", "STACK"))))
    write_json(D / "prep.json", res)
    print(f"prep ok={res['ok']}", flush=True)
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--av":
        av_main(json.loads(sys.argv[2]))
    else:
        main()
