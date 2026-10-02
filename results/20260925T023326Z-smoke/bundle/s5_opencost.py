"""Stage opencost (TorchCodec process, no DataLoader): per-open cost vs frames per file.

For every family (ckpt_* lengths and thrash) and layout (SEP, STACK, BIG):
  open    VideoDecoder(path, seek_mode="approximate") construction (PR #4556: local files by PATH)
  first   first get_frames_at([random index]) on the new decoder
  close   del decoder (what an LRU eviction costs; handle closes on GC)
  decode  get_frames_at([random index]) on an already-open decoder (steady-state decode cost)
in two page-cache states: warm (files prewarmed) and cold (posix_fadvise DONTNEED on the file
right before each open). Per open: rchar / read_bytes deltas from /proc/self/io.
Index size: moov box bytes per file. Memory: a fresh subprocess per (family, layout) opens N
decoders and keeps them; the RssAnon slope (least squares over N) is MB per open decoder, before
and after its first decoded frame.

Eviction check: resident fraction (mincore) of the family's files after prewarm and after evict,
plus read_bytes of a full read after evict.

Usage: python s5_opencost.py <selection.json> <enc_root> <out_dir>   (env SMOKE, OC_WARM, OC_COLD, OC_RSS_N)
       python s5_opencost.py --rss <fam> <layout> <n> <selection.json> <enc_root>  (internal)
"""

import gc
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from torchcodec.decoders import VideoDecoder

from common import (LAYOUTS, Family, evict, load_selection, memlog, moov_info, pct, pin_threads, prewarm, proc_io, proc_status,
                    resident_fraction_many, thread_report, write_json)


def slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")


def file_frames(F, lay):
    return [n for n in F.frames_per_file(lay) for _ in (F.cams if lay != "STACK" else [0])]


def rss_mode(fam, lay, n, sel_path, enc_root):
    pin_threads(1)  # quota-aware: HF containers report 64 CPUs for an 8-CPU quota
    sel = load_selection(sel_path, os.environ.get("SMOKE", "0") == "1")
    F = Family(enc_root, fam, sel["active"][fam], sel["cams"], sel["fps"])
    files, frames = F.files(lay), file_frames(F, lay)
    rng = random.Random(1)
    # warm-up: load codec libraries and allocator arenas once
    d = VideoDecoder(str(files[0]), seek_mode="approximate")
    d.get_frames_at(indices=[0])
    del d
    gc.collect()
    keep, a_open, a_first = [], [], []
    base = proc_status()
    for i in range(n):
        k = i % len(files)
        d = VideoDecoder(str(files[k]), seek_mode="approximate")
        keep.append(d)
        a_open.append(proc_status().get("RssAnon", proc_status().get("VmRSS", float("nan"))))
        d.get_frames_at(indices=[rng.randrange(frames[k])])
        a_first.append(proc_status().get("RssAnon", proc_status().get("VmRSS", float("nan"))))
    xs = list(range(1, n + 1))
    print(json.dumps({"n": n, "base": base, "rss_anon_after_open_mb": a_open, "rss_anon_after_first_mb": a_first,
                      "mb_per_decoder_open": slope(xs, a_open), "mb_per_decoder_after_first_frame": slope(xs, a_first),
                      "end": proc_status()}))


def measure(files, frames, reps, cold, rng):
    rows = []
    for r in range(reps):
        k = r % len(files)
        p = str(files[k])
        if cold:
            evict([p])
        io0 = proc_io()
        t0 = time.perf_counter()
        d = VideoDecoder(p, seek_mode="approximate")
        t1 = time.perf_counter()
        io1 = proc_io()
        d.get_frames_at(indices=[rng.randrange(frames[k])])
        t2 = time.perf_counter()
        dec = []
        for _ in range(3):
            i = rng.randrange(frames[k])
            ta = time.perf_counter()
            d.get_frames_at(indices=[i])
            dec.append(time.perf_counter() - ta)
        io2 = proc_io()
        t3 = time.perf_counter()
        del d
        t4 = time.perf_counter()
        rows.append({"file": k, "open_ms": (t1 - t0) * 1e3, "first_ms": (t2 - t1) * 1e3, "close_ms": (t4 - t3) * 1e3,
                     "decode_ms": statistics.mean(dec) * 1e3,
                     "open_rchar": io1[0] - io0[0], "open_read_bytes": io1[1] - io0[1],
                     "total_read_bytes": io2[1] - io0[1]})
    return rows


def summarize(rows):
    out = {}
    for k in ("open_ms", "first_ms", "close_ms", "decode_ms", "open_rchar", "open_read_bytes", "total_read_bytes"):
        v = [r[k] for r in rows]
        out[k] = {"median": statistics.median(v), "mean": statistics.mean(v), "p95": pct(v, 0.95)}
    v = [r["open_ms"] + r["first_ms"] + r["close_ms"] for r in rows]
    out["open_first_close_ms"] = {"median": statistics.median(v), "mean": statistics.mean(v), "p95": pct(v, 0.95)}
    return out


def main():
    if sys.argv[1] == "--rss":
        return rss_mode(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6])
    sel_path, enc_root, out_dir = sys.argv[1:4]
    smoke = os.environ.get("SMOKE", "0") == "1"
    warm_n = int(os.environ.get("OC_WARM", "10" if smoke else "40"))
    cold_n = int(os.environ.get("OC_COLD", "6" if smoke else "24"))
    rss_n = int(os.environ.get("OC_RSS_N", "8" if smoke else "30"))
    pin_threads(1)  # quota-aware: HF containers report 64 CPUs for an 8-CPU quota
    sel = load_selection(sel_path, smoke)
    rng = random.Random(0)
    res = {"cells": {}, "eviction_check": {}, "threads": thread_report()}
    for fam, groups in sel["active"].items():
        F = Family(enc_root, fam, groups, sel["cams"], sel["fps"])
        allf = [p for lay in LAYOUTS for p in F.files(lay)]
        # eviction check on this family's files
        prewarm(allf)
        r_warm = resident_fraction_many(allf)
        evict(allf)
        r_cold = resident_fraction_many(allf)
        probe = str(F.files("SEP")[0])
        io0 = proc_io()
        t0 = time.perf_counter()
        prewarm([probe])
        t_cold_read = time.perf_counter() - t0
        io1 = proc_io()
        t0 = time.perf_counter()
        prewarm([probe])
        t_warm_read = time.perf_counter() - t0
        io2 = proc_io()
        res["eviction_check"][fam] = {
            "resident_after_prewarm": r_warm, "resident_after_evict": r_cold, "probe_file_bytes": os.path.getsize(probe),
            "read_bytes_after_evict": io1[1] - io0[1], "read_bytes_warm_reread": io2[1] - io1[1],
            "read_s_after_evict": t_cold_read, "read_s_warm": t_warm_read,
            "worked": (r_cold < 0.05) or (io1[1] - io0[1] > 0.5 * os.path.getsize(probe)),
        }
        print(fam, "eviction", res["eviction_check"][fam], flush=True)
        for lay in LAYOUTS:
            files, frames = F.files(lay), file_frames(F, lay)
            mi = [moov_info(p) for p in files]
            prewarm(files)
            measure(files, frames, 2, False, rng)  # warm-up code paths
            w = measure(files, frames, warm_n, False, rng)
            c = measure(files, frames, cold_n, True, rng)
            prewarm(files)
            rss = subprocess.run([sys.executable, __file__, "--rss", fam, lay, str(rss_n), sel_path, enc_root],
                                 capture_output=True, text=True)
            rss_res = json.loads(rss.stdout.strip().splitlines()[-1]) if rss.returncode == 0 else {"error": rss.stderr[-2000:]}
            cell = {
                "family": fam, "layout": lay, "files": len(files), "frames_per_file_mean": statistics.mean(frames),
                "frames_per_file": sorted(set(frames)), "pixels_per_frame": 224 * 224 * (3 if lay == "STACK" else 1),
                "moov_bytes_mean": statistics.mean(m["moov_bytes"] for m in mi), "file_bytes_mean": statistics.mean(m["file_bytes"] for m in mi),
                "moov_before_mdat": sorted({m["moov_before_mdat"] for m in mi}),
                "warm": summarize(w), "cold": summarize(c), "rss": {k: v for k, v in rss_res.items() if not k.startswith("rss_anon_after")},
                "rss_series": {k: v for k, v in rss_res.items() if k.startswith("rss_anon_after")},
                "raw": {"warm": w, "cold": c},
            }
            res["cells"][f"{fam}|{lay}"] = cell
            print(f"{fam}|{lay}: frames/file {cell['frames_per_file_mean']:.0f} moov {cell['moov_bytes_mean']:.0f}B "
                  f"warm open {cell['warm']['open_ms']['median']:.2f}ms first {cell['warm']['first_ms']['median']:.2f}ms "
                  f"decode {cell['warm']['decode_ms']['median']:.2f}ms | cold open {cell['cold']['open_ms']['median']:.2f}ms "
                  f"read {cell['cold']['open_read_bytes']['median']:.0f}B | MB/decoder {rss_res.get('mb_per_decoder_after_first_frame')}", flush=True)
        memlog(out_dir, f"opencost_{fam}")
    write_json(Path(out_dir) / "opencost" / "opencost.json", res)


if __name__ == "__main__":
    main()
