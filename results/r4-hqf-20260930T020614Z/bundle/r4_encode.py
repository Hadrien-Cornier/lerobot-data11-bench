"""Round 4 stage encode (PyAV-only process; never import torchcodec or lerobot here).

Decode each camera's slice frames ONCE (the dataset's own AV1 file, one container per camera, all episodes in one
forward pass), in lockstep across cameras, and feed the SAME RGB frames to:
  SEP/<cam>.mp4       one encoder per camera, native size
  STACK/stacked.mp4   one encoder, cameras top to bottom in slice["cams"] order; narrower cameras right-padded
                      with black to the widest width, odd heights padded by one black row (r4_common.stack_layout)
Encoder = LeRobot's defaults read from the installed code in stage slice (VideoEncoderConfig().get_codec_options():
libsvtav1, g, crf, preset, svtav1-params fast-decode), pix_fmt yuv420p, RGB frames in (the encoder converts), pts=i,
time_base=1/fps (as s2_encode.py). SVT-AV1 `lp` is set to the CPU quota (the container reports 64 CPUs).
Mux: +faststart when the source file has its moov before mdat (mirrors the source).

The source frames at slice["check_indices"] are saved to <work>/check_src.npz for the PSNR check.
SVT-AV1 logs its effective config (e.g. "Preset M12 is mapped to M10") on stderr; fd 2 goes to logs/encode_svt.log.

Usage: python r4_encode.py <slice.json> <work_dir> <out_dir>      Env: R4_DEC_THREADS (2), R4_ENC_LP (cpu quota)
"""

import os
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from common import cpu_quota, memlog, moov_info, pin_threads
from r4_common import log, read_json, write_json


def frames_of(path, episodes, cam, fps, dec_threads, stats):
    """One forward pass over the slice's episodes in one source file; asserts every pts on the 1/fps grid."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.codec_context.thread_count = dec_threads
        t_first = episodes[0]["src"][cam]["from_timestamp"]
        c.seek(int(t_first / s.time_base), backward=True, any_frame=False, stream=s)
        stats["decoder"] = s.codec_context.name
        it = c.decode(s)
        for ep in episodes:
            t_from, L = ep["src"][cam]["from_timestamp"], ep["length"]
            i = 0
            while i < L:
                f = next(it)
                t = float(f.pts * s.time_base)
                if t < t_from + i / fps - 0.5 / fps:
                    stats["skipped"] += 1
                    continue
                exp = t_from + i / fps
                assert abs(t - exp) < 0.5 / fps, (str(path), ep["episode"], i, t, exp)
                yield f.to_ndarray(format="rgb24")
                i += 1


class Enc:
    def __init__(self, path, fps, codec, pix_fmt, opts, mux_opts, lp):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.c = av.open(self.path, "w", options=mux_opts)
        self.fps, self.s, self.i = fps, None, 0
        self.codec, self.pix_fmt, self.opts, self.lp = codec, pix_fmt, dict(opts), lp
        self.feed_s = 0.0
        self.flush_s = 0.0

    def add(self, arr):
        t0 = time.perf_counter()
        if self.s is None:
            opts = dict(self.opts)
            if self.lp > 0:
                opts["svtav1-params"] = f"{opts['svtav1-params']}:lp={self.lp}" if opts.get("svtav1-params") else f"lp={self.lp}"
            self.used_opts = opts
            self.s = self.c.add_stream(self.codec, self.fps, options=opts)
            self.s.pix_fmt = self.pix_fmt
            self.s.height, self.s.width = arr.shape[:2]
            self.s.time_base = Fraction(1, self.fps)
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")
        vf.pts = self.i
        vf.time_base = Fraction(1, self.fps)
        self.i += 1
        for p in self.s.encode(vf):
            self.c.mux(p)
        self.feed_s += time.perf_counter() - t0

    def close(self):
        t0 = time.perf_counter()
        for p in self.s.encode():
            self.c.mux(p)
        self.c.close()
        self.flush_s = time.perf_counter() - t0


def main():
    sl = read_json(sys.argv[1])
    work, out = Path(sys.argv[2]), Path(sys.argv[3])
    D = out / "encode"
    D.mkdir(parents=True, exist_ok=True)
    quota = cpu_quota()
    pin_threads(1)
    lp = int(os.environ.get("R4_ENC_LP", "0")) or quota
    dec_threads = int(os.environ.get("R4_DEC_THREADS", "2"))
    cams, fps, eps = sl["cams"], sl["fps"], sl["slice"]["episodes"]
    lay = sl["stack_layout"]
    ed = sl["encoder_defaults"]
    opts = dict(ed["codec_options"])
    src_faststart = all(v["moov"]["moov_before_mdat"] for v in sl["source_files"].values())
    mux = {"movflags": "faststart"} if src_faststart else {}
    enc_root = work / "enc"
    log_path = out / "logs" / "encode_svt.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    saved_fd2 = os.dup(2)
    fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 2)
    sep = {c: Enc(enc_root / "SEP" / f"{c}.mp4", fps, ed["vcodec"], ed["pix_fmt"], opts, mux, lp) for c in cams}
    stk = Enc(enc_root / "STACK" / "stacked.mp4", fps, ed["vcodec"], ed["pix_fmt"], opts, mux, lp)
    H, W, crops = lay["height"], lay["width"], lay["crops"]
    canvas = np.zeros((H, W, 3), dtype=np.uint8)  # pad stays black
    dstats = {c: {"skipped": 0} for c in cams}
    gens = {c: frames_of(sl["source_files"][c]["path"], eps, c, fps, dec_threads, dstats[c]) for c in cams}
    check = set(sl["check_indices"])
    saved = {c: [] for c in cams}
    n_total = sl["slice"]["frames"]
    dec_s = stack_s = 0.0
    t0 = time.perf_counter()
    cpu0 = sum(os.times()[:2])
    for i in range(n_total):
        ta = time.perf_counter()
        fr = {c: next(gens[c]) for c in cams}
        dec_s += time.perf_counter() - ta
        for c in cams:
            h, w = sl["shapes"][c]
            assert fr[c].shape == (h, w, 3), (c, fr[c].shape)
            sep[c].add(fr[c])
        tb = time.perf_counter()
        for c in cams:
            y0, h, w = crops[c]
            canvas[y0: y0 + h, :w] = fr[c]
        stack_s += time.perf_counter() - tb
        stk.add(canvas)
        if i in check:
            for c in cams:
                saved[c].append(fr[c].copy())
        if i % 5000 == 4999:
            memlog(out, f"encode_{i + 1}")
            os.write(saved_fd2, f"[encode] {i + 1}/{n_total} frames, {time.perf_counter() - t0:.0f}s\n".encode())
    for g in gens.values():
        g.close()
    for e in (*sep.values(), stk):
        e.close()
    wall = time.perf_counter() - t0
    cpu = sum(os.times()[:2]) - cpu0
    os.dup2(saved_fd2, 2)
    os.close(fd)
    assert all(e.i == n_total for e in (*sep.values(), stk))
    np.savez(work / "check_src.npz", indices=np.array(sorted(check)), **{c: np.stack(saved[c]) for c in cams})

    svt = {"mapped": set(), "version": set(), "pred_struct": set(), "lp": set()}
    for line in log_path.read_text(errors="replace").splitlines():
        if "mapped to" in line:
            svt["mapped"].add(line.split("Svt[warn]:")[-1].strip())
        if "preset / tune / pred struct" in line:
            svt["pred_struct"].add(re.sub(r"\s+", " ", line.split(":", 2)[-1]).strip())
        if "SVT [version]" in line:
            svt["version"].add(line.split("]:")[-1].strip())
        if "Level of Parallelism" in line:
            svt["lp"].add(line.split("]:")[-1].strip())
    files = {f"SEP/{c}": {"path": sep[c].path, "bytes": os.path.getsize(sep[c].path), "shape": sl["shapes"][c], "feed_s": round(sep[c].feed_s, 1),
                          "flush_s": round(sep[c].flush_s, 1), "moov": moov_info(sep[c].path)} for c in cams}
    files["STACK"] = {"path": stk.path, "bytes": os.path.getsize(stk.path), "shape": [H, W], "feed_s": round(stk.feed_s, 1),
                      "flush_s": round(stk.flush_s, 1), "moov": moov_info(stk.path)}
    sep_bytes = sum(files[f"SEP/{c}"]["bytes"] for c in cams)
    meta = {"frames": n_total, "wall_s": round(wall, 1), "process_cpu_s": round(cpu, 1), "decode_s": round(dec_s, 1), "stack_copy_s": round(stack_s, 1),
            "frames_per_s": round(n_total / wall, 1), "decoder": {c: dstats[c] for c in cams}, "dec_threads": dec_threads,
            "encoder": {"vcodec": ed["vcodec"], "pix_fmt": ed["pix_fmt"], "options_from_lerobot": opts, "options_used": stk.used_opts,
                        "input": "rgb24 ndarray, pts=i, time_base=1/fps"},
            "svt_lp": lp, "cpu_quota": quota, "mux_options": mux, "source_faststart": src_faststart,
            "svt_reported": {k: sorted(v) for k, v in svt.items()},
            "files": files, "sep_bytes_total": sep_bytes, "stack_bytes": files["STACK"]["bytes"],
            "stack_over_sep_bytes": files["STACK"]["bytes"] / sep_bytes,
            "source_bytes_first_files": sum(v["bytes"] for v in sl["source_files"].values()),
            "stack_layout": lay, "stack_order_top_to_bottom": cams,
            "pyav": av.__version__, "pyav_ffmpeg": {k: ".".join(map(str, v)) for k, v in av.library_versions.items()},
            "note": "SEP and STACK encoders run in the same loop; feed_s is time spent in each encoder's encode() calls (SVT-AV1 "
                    "encodes on its own threads), so per-layout encode cost is not separable from this wall time."}
    write_json(D / "encode.json", meta)
    memlog(out, "encode_end")
    log(f"encode: {n_total} frames in {wall:.0f}s; SEP {sep_bytes / 1e6:.1f} MB, STACK {files['STACK']['bytes'] / 1e6:.1f} MB; svt {meta['svt_reported']}")


if __name__ == "__main__":
    main()
