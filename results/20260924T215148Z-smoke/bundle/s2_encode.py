"""Stage encode (PyAV-only process; never import torchcodec here).

For every group of every active family: decode each camera's source episodes ONCE (the dataset's
own AV1 files, located by meta/episodes chunk/file index and from/to timestamps), in lockstep
across cameras, and feed the SAME RGB frames to 4 encoders:
  SEP/<group>__<cam>.mp4  x3  (224x224)
  STACK/<group>.mp4           (224x672, cameras top-to-bottom in selection["cams"] order)
Encoder = LeRobot VideoEncoderConfig() defaults via get_codec_options(): libsvtav1, g=2, crf=30,
preset=12, svtav1-params fast-decode=0, pix_fmt yuv420p; RGB frames in, encoder converts; pts=i,
time_base=1/fps (mirrors StreamingVideoEncoder). No B-frame options are added.

SVT-AV1 logs its effective config (e.g. "Preset M12 is mapped to M10") to stderr; each pool worker
sends fd 2 to logs/encode_worker_<pid>.log so the actual preset can be recorded.

Usage: python s2_encode.py <selection.json> <src_root> <enc_root> <out_dir>   (env SMOKE, ENC_PROCS)
"""

import json
import multiprocessing as mp
import os
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from common import Family, load_selection, memlog, write_json

VCODEC = "libsvtav1"
OPTS = {"g": "2", "crf": "30", "preset": "12", "svtav1-params": "fast-decode=0"}
LOG_DIR = None


def _init(log_dir):
    fd = os.open(os.path.join(log_dir, f"encode_worker_{os.getpid()}.log"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 2)
    sys.stderr = os.fdopen(2, "w", buffering=1)


def source_frames(path, t_from, n, fps):
    """Yield n RGB frames starting at t_from; asserts every pts is exactly on the 1/fps grid."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(t_from / s.time_base), backward=True, any_frame=False, stream=s)
        i = 0
        for f in c.decode(s):
            t = float(f.pts * s.time_base)
            if t < t_from - 0.5 / fps:
                continue
            exp = t_from + i / fps
            assert abs(t - exp) < 0.5 / fps, (str(path), i, t, exp)
            yield f.to_ndarray(format="rgb24")
            i += 1
            if i == n:
                return
    raise RuntimeError(f"{path}: only {i} of {n} frames from t={t_from}")


class Enc:
    def __init__(self, path, fps):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.c = av.open(str(path), "w")
        self.fps, self.s, self.i = fps, None, 0

    def add(self, arr):
        if self.s is None:
            self.s = self.c.add_stream(VCODEC, self.fps, options=OPTS)
            self.s.pix_fmt = "yuv420p"
            self.s.height, self.s.width = arr.shape[:2]
            self.s.time_base = Fraction(1, self.fps)
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")
        vf.pts = self.i
        vf.time_base = Fraction(1, self.fps)
        self.i += 1
        for p in self.s.encode(vf):
            self.c.mux(p)

    def close(self):
        for p in self.s.encode():
            self.c.mux(p)
        self.c.close()


def encode_group(args):
    src_root, enc_root, fam, groups, gi, cams, fps = args
    F = Family(enc_root, fam, groups, cams, fps)
    g = groups[gi]
    if all(F.sep(gi, c).exists() for c in cams) and F.stack(gi).exists() and (F.root / "STACK" / f"{g['name']}.done").exists():
        return {"group": g["name"], "skipped": True}
    t0 = time.perf_counter()
    sep = {c: Enc(F.sep(gi, c), fps) for c in cams}
    stk = Enc(F.stack(gi), fps)
    n = 0
    for ep in g["episodes"]:
        L = ep["length"]
        for c in cams:
            s = ep["src"][c]
            assert abs((s["to_timestamp"] - s["from_timestamp"]) * fps - L) < 0.5, (ep["episode"], c, s, L)
        gens = {c: source_frames(Path(src_root) / ep["src"][c]["path"], ep["src"][c]["from_timestamp"], L, fps) for c in cams}
        for _ in range(L):
            fr = {c: next(gens[c]) for c in cams}
            for c in cams:
                assert fr[c].shape == (224, 224, 3), fr[c].shape
                sep[c].add(fr[c])
            stk.add(np.concatenate([fr[c] for c in cams], axis=0))
            n += 1
        for gen in gens.values():
            gen.close()
    for e in (*sep.values(), stk):
        e.close()
    assert n == g["frames"] and all(e.i == n for e in (*sep.values(), stk)), (g["name"], n)
    (F.root / "STACK" / f"{g['name']}.done").write_text(str(n))
    return {"group": g["name"], "family": fam, "frames": n, "encode_s": round(time.perf_counter() - t0, 2), "pid": os.getpid()}


def main():
    sel_path, src_root, enc_root, out_dir = sys.argv[1:5]
    smoke = os.environ.get("SMOKE", "0") == "1"
    procs = int(os.environ.get("ENC_PROCS", "2"))
    sel = load_selection(sel_path, smoke)
    cams, fps = sel["cams"], sel["fps"]
    log_dir = Path(out_dir) / "logs" / "encode"
    log_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for fam, groups in sel["active"].items():
        # longest groups first so the pool drains evenly
        for gi in sorted(range(len(groups)), key=lambda i: -groups[i]["frames"]):
            tasks.append((src_root, enc_root, fam, groups, gi, cams, fps))
    tasks.sort(key=lambda t: -t[3][t[4]]["frames"])
    t0 = time.perf_counter()
    res = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(procs, initializer=_init, initargs=(str(log_dir),), maxtasksperchild=8) as pool:
        for r in pool.imap_unordered(encode_group, tasks):
            res.append(r)
            print(f"[encode {len(res)}/{len(tasks)} {time.perf_counter() - t0:.0f}s] {r}", flush=True)
            if len(res) % 8 == 0:
                memlog(out_dir, f"encode_{len(res)}")
    wall = time.perf_counter() - t0
    svt = {"preset_lines": set(), "mapped": set(), "version": set(), "pred_struct": set(), "lp": set()}
    for f in log_dir.glob("*.log"):
        for line in f.read_text(errors="replace").splitlines():
            if "mapped to" in line:
                svt["mapped"].add(line.split("Svt[warn]:")[-1].strip())
            if "preset / tune / pred struct" in line:
                svt["pred_struct"].add(re.sub(r"\s+", " ", line.split(":", 2)[-1]).strip())
            if "SVT [version]" in line:
                svt["version"].add(line.split("]:")[-1].strip())
            if "Level of Parallelism" in line:
                svt["lp"].add(line.split("]:")[-1].strip())
    total_frames = sum(r.get("frames", 0) for r in res)
    meta = {
        "encoder": {"vcodec": VCODEC, "options": OPTS, "pix_fmt": "yuv420p", "input": "rgb24 ndarray, pts=i, time_base=1/fps"},
        "svt_reported": {k: sorted(v) for k, v in svt.items() if k != "preset_lines"},
        "enc_procs": procs, "wall_s": round(wall, 1), "groups": len(res), "frames_per_camera": total_frames,
        "camera_frame_sets_per_s": round(total_frames / wall, 1) if wall else None,
        "pyav": av.__version__, "pyav_ffmpeg": {k: ".".join(map(str, v)) for k, v in av.library_versions.items()},
        "per_group": sorted(res, key=lambda r: r["group"]),
        "stack_order_top_to_bottom": cams,
    }
    write_json(Path(out_dir) / "encode" / "encode_meta.json", meta)
    memlog(out_dir, "encode_end")
    print(json.dumps({k: v for k, v in meta.items() if k != "per_group"}, indent=1))


if __name__ == "__main__":
    main()
