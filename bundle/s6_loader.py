"""Stage loader: DataLoader workload over SEP / STACK / BIG with a per-worker LRU decoder cache.

Decoder cache = PR #4556 TorchcodecCache semantics for LOCAL files: an OrderedDict LRU keyed by
path, VideoDecoder(path, seek_mode="approximate") on a miss (opened BY PATH, no fsspec handle),
evict least-recently-used while len > max_size (the evicted decoder is dropped and freed), hit ->
move_to_end. One cache per DataLoader worker process (each worker holds its own dataset copy).

Frame lookup mirrors decode_video_frames_torchcodec: ts = frame / fps, index = round(ts *
metadata.average_fps), get_frames_at(indices=[index]); the returned pts is checked against ts.

Workload: map-style dataset over every frame of a family; global uniform shuffle (seeded
randperm per arm run), batch 32, spawn workers, prefetch_factor 2. One DataLoader per worker count
serves all arms (workers are spawned once); items are (arm_id, frame) and a worker that sees a new
arm_id drops its cache and starts a fresh LRU, so every arm starts from an empty cache as a fresh
worker would. Batches of the previous arm still in flight are received and discarded. Each sample = one timestamp, all 3 cameras
("all") or only top ("top"). STACK decodes the stacked frame and crops rows (copy included); SEP /
BIG decode one file per camera. Output per sample is [k, 3, 224, 224] uint8 for every layout.

Arms: family x layout x pattern x cache size x workers x page-cache state. Thrash family: cache
= round(ratio x SEP file count) for each ratio (same decoder count for every layout; the ratio vs
each layout's own file count is recorded too). Validation checkpoint family: cache 1 and "all".
Cold: posix_fadvise(DONTNEED) of every file of the family before the run, after warmup and then
every EVICT_INTERVAL_S from a thread in the main process (emulates a dataset far larger than RAM).

Timing: warmup batches (enough for every worker to fill its cache), then blocks of BLOCK_BATCHES;
stop when the 95% CI half-width of block samples/s <= CI_TARGET x mean (after MIN_BLOCKS), or
MAX_BLOCKS, or the per-run time cap. Repeats interleave layouts ABAB (order flips per repeat).

Usage: python s6_loader.py <selection.json> <enc_root> <out_dir>   (env: see CFG below)
"""

import ctypes
import ctypes.util
import gc
import json
import math
import os
import random
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from common import (H, LAYOUTS, Family, ci_halfwidth, cgroup_mem, env_int, env_list, evict, load_selection, memlog,
                    pct, pin_threads, prewarm, proc_io, proc_status, resident_fraction_many, thread_report, write_json)

STAT_KEYS = ["hits", "misses", "open_s", "decode_s", "crop_s", "cpu_s", "rchar", "read_bytes", "rss_anon_mb",
             "worker", "pts_err", "wall_s", "arm"]
PATTERN_CAMS = {"all": None, "top": ["top"]}
try:
    _malloc_trim = ctypes.CDLL(ctypes.util.find_library("c")).malloc_trim
except Exception:
    _malloc_trim = None


class LRU:
    """PR #4556 TorchcodecCache, local-path branch."""

    def __init__(self, max_size):
        from torchcodec.decoders import VideoDecoder

        self.VideoDecoder = VideoDecoder
        self.max_size = max_size
        self.d = OrderedDict()

    def get(self, path):
        e = self.d.get(path)
        if e is not None:
            self.d.move_to_end(path)
            return e, True
        dec = self.VideoDecoder(path, seek_mode="approximate")
        self.d[path] = dec
        while len(self.d) > 1 and self.max_size is not None and len(self.d) > self.max_size:
            self.d.popitem(last=False)
        return dec, False


class FrameDS(Dataset):
    """Items are (arm_id, frame_index). A worker switching to a new arm drops its whole cache
    (fresh LRU, like a fresh worker) and trims the heap, so arms do not share decoders."""

    def __init__(self, enc_root, sel, arms):
        self.enc_root, self.sel, self.arms = enc_root, sel, arms  # arms: list of dict(family, layout, pattern, cache)
        self.cur = None
        self.cache = None
        self.F = {}

    def _switch(self, aid):
        self.cache = None
        gc.collect()
        if _malloc_trim is not None:
            _malloc_trim(0)
        a = self.arms[aid]
        if a["family"] not in self.F:
            self.F[a["family"]] = Family(self.enc_root, a["family"], self.sel["active"][a["family"]], self.sel["cams"], self.sel["fps"])
        self.cur = aid
        self.fam = self.F[a["family"]]
        self.layout = a["layout"]
        self.want = PATTERN_CAMS[a["pattern"]] or self.sel["cams"]
        self.cache = LRU(a["cache"])
        self.fps = self.sel["fps"]

    def __getitem__(self, key):
        aid, i = key
        if aid != self.cur:
            self._switch(aid)
        c0, w0 = time.process_time(), time.perf_counter()
        io0 = proc_io()
        hits = misses = 0
        t_open = t_dec = t_crop = 0.0
        err = 0.0
        decoded = {}
        plan = self.fam.plan(self.layout, i, self.want)
        for path, fi, _ in plan:
            if path in decoded:
                continue
            ta = time.perf_counter()
            d, hit = self.cache.get(path)
            tb = time.perf_counter()
            hits += hit
            misses += not hit
            t_open += tb - ta
            ts = fi / self.fps
            fb = d.get_frames_at(indices=[round(ts * d.metadata.average_fps)])
            t_dec += time.perf_counter() - tb
            err = max(err, abs(float(fb.pts_seconds[0]) - ts))
            decoded[path] = fb.data[0]
        tc = time.perf_counter()
        imgs = []
        for path, _, crop in plan:
            x = decoded[path]
            if crop is not None:
                x = x[:, crop * H : (crop + 1) * H, :]
            imgs.append(x)
        out = torch.stack(imgs)  # copies: crop materialization for STACK, the same stacking copy for SEP/BIG
        t_crop = time.perf_counter() - tc
        io1 = proc_io()
        wi = torch.utils.data.get_worker_info()
        st = proc_status()
        stats = torch.tensor([hits, misses, t_open, t_dec, t_crop, time.process_time() - c0, io1[0] - io0[0],
                              io1[1] - io0[1], st.get("RssAnon", st.get("VmRSS", -1)), wi.id if wi else -1, err,
                              time.perf_counter() - w0, aid], dtype=torch.float64)
        return {"frames": out, "stats": stats}


class Driver:
    """Batch sampler fed by the main loop: yields batches of (arm_id, index) for the current arm,
    global uniform shuffle over that arm's frames (seeded per arm run). current=None ends it."""

    def __init__(self, bs):
        self.bs, self.current, self.perm, self.pos, self.n, self.gen = bs, None, None, 0, 0, None

    def set(self, aid, n, seed):
        self.gen = torch.Generator().manual_seed(seed)
        self.n, self.perm, self.pos = n, torch.randperm(n, generator=self.gen), 0
        self.current = aid

    def __iter__(self):
        while self.current is not None:
            if self.pos + self.bs > self.n:
                self.perm, self.pos = torch.randperm(self.n, generator=self.gen), 0
            idx = self.perm[self.pos : self.pos + self.bs].tolist()
            self.pos += self.bs
            yield [(self.current, i) for i in idx]


class Evictor(threading.Thread):
    def __init__(self, files, interval):
        super().__init__(daemon=True)
        self.files, self.interval, self.stop_ev, self.passes = files, interval, threading.Event(), 0

    def run(self):
        while not self.stop_ev.wait(self.interval):
            evict(self.files)
            self.passes += 1


def units(F, layout, pattern):
    """(units a sample chooses among, keys per unit) for the hit-rate model and warmup length."""
    k = len(PATTERN_CAMS[pattern] or F.cams)
    G = len(F.groups)
    if layout == "SEP":
        return G, k
    if layout == "STACK":
        return G, 1
    return F.n_big(), k


def harmonic(n):
    return sum(1.0 / i for i in range(1, n + 1))


def warmup_batches(U, k, C, W, bs, cap):
    m = min(U, max(1, C // k))
    per_worker = U * (harmonic(U) - harmonic(U - m)) if m < U else U * harmonic(U)
    return int(min(cap, math.ceil(W * per_worker / bs) + 2 * W))


class Session:
    """One persistent DataLoader (spawn workers started once) serving every arm of one worker count."""

    def __init__(self, cfg, sel, arms, W):
        self.cfg, self.sel, self.arms, self.W = cfg, sel, arms, W
        self.driver = Driver(cfg["batch"])
        self.ds = FrameDS(cfg["enc_root"], sel, arms)
        self.dl = DataLoader(self.ds, batch_sampler=self.driver, num_workers=W, multiprocessing_context="spawn",
                             prefetch_factor=2, persistent_workers=False)
        self.it = None
        self.spawn_s = None
        self.discarded = 0

    def next_for(self, aid):
        """Next batch belonging to arm aid (batches of the previous arm still in flight are dropped)."""
        while True:
            x = next(self.it)
            if int(x["stats"][0, -1]) == aid:
                return x
            self.discarded += 1

    def close(self):
        self.driver.current = None
        if self.it is not None:
            try:
                for _ in self.it:
                    pass
            except Exception:
                pass
        del self.it, self.dl


def run_one(sess, aid, arm, seed, cap_s):
    cfg, sel = sess.cfg, sess.sel
    fam, layout, pattern, C, W, state = arm["family"], arm["layout"], arm["pattern"], arm["cache"], arm["workers"], arm["state"]
    groups = sel["active"][fam]
    F = Family(cfg["enc_root"], fam, groups, sel["cams"], sel["fps"])
    files = [str(p) for p in F.files(layout)]
    U, k = units(F, layout, pattern)
    bs = cfg["batch"]
    wb = warmup_batches(U, k, C, W, bs, cfg["warmup_cap"])
    block = cfg["block_batches"] or 4 * W
    ev = None
    res_before = None
    if state == "cold":
        evict(files)
        res_before = resident_fraction_many(files)
    else:
        prewarm(files)  # earlier cold arms / the opencost stage may have evicted them
        res_before = resident_fraction_many(files)
    sess.driver.set(aid, F.total, seed)
    t0 = time.perf_counter()
    if sess.it is None:
        sess.it = iter(sess.dl)
    stats_warm, stats_meas, waits, blocks = [], [], [], []
    for b in range(wb):
        stats_warm.append(sess.next_for(aid)["stats"])
    warm_s = time.perf_counter() - t0
    if state == "cold":
        evict(files)
        ev = Evictor(files, cfg["evict_interval"])
        ev.start()
    stop = "max_blocks"
    t_meas0 = time.perf_counter()
    while len(blocks) < cfg["max_blocks"]:
        tb0 = time.perf_counter()
        n = 0
        for _ in range(block):
            ta = time.perf_counter()
            x = sess.next_for(aid)
            waits.append(time.perf_counter() - ta)
            stats_meas.append(x["stats"])
            n += x["frames"].shape[0]
        dt = time.perf_counter() - tb0
        blocks.append({"samples": n, "wall_s": dt, "sps": n / dt})
        sps = [bl["sps"] for bl in blocks]
        if len(blocks) >= cfg["min_blocks"]:
            if cfg["max_blocks"] == 1:
                stop = "single_block"
                break
            m = sum(sps) / len(sps)
            if ci_halfwidth(sps) <= cfg["ci_target"] * m:
                stop = "ci"
                break
        if time.perf_counter() - t_meas0 > cap_s:
            stop = "time_cap"
            break
    meas_s = time.perf_counter() - t_meas0
    if ev:
        ev.stop_ev.set()
        ev.join()
    res_end = resident_fraction_many(files) if state == "cold" else None
    S = torch.cat(stats_meas).reshape(-1, len(STAT_KEYS))
    col = {kk: S[:, j] for j, kk in enumerate(STAT_KEYS)}
    SW = torch.cat(stats_warm).reshape(-1, len(STAT_KEYS))
    ns = S.shape[0]
    sps = [bl["sps"] for bl in blocks]
    mean_sps = sum(sps) / len(sps)
    per_worker_rss = {}
    for w, r in zip(col["worker"].tolist(), col["rss_anon_mb"].tolist()):
        per_worker_rss[int(w)] = max(per_worker_rss.get(int(w), 0), r)
    hits, misses = col["hits"].sum().item(), col["misses"].sum().item()
    wh, wm = SW[:, 0].sum().item(), SW[:, 1].sum().item()
    return {
        **arm, "seed": seed, "files": len(files), "units": U, "keys_per_unit": k,
        "cache_over_own_files": C / len(files), "cache_over_sep_files": C / (3 * len(groups)),
        "frames_per_file_mean": sum(F.frames_per_file(layout)) / len(F.frames_per_file(layout)),
        "warmup_batches": wb, "warmup_s": warm_s, "block_batches": block, "blocks": blocks, "stop_reason": stop,
        "measured_s": meas_s, "samples_per_s": mean_sps, "sps_ci95_halfwidth": ci_halfwidth(sps),
        "batch_wait_p50_ms": pct(waits, 0.5) * 1e3, "batch_wait_p95_ms": pct(waits, 0.95) * 1e3,
        "samples": ns, "hit_rate": hits / (hits + misses) if hits + misses else float("nan"),
        "opens_per_sample": misses / ns,
        "worker_cpu_s_per_sample": col["cpu_s"].mean().item(), "worker_wall_s_per_sample": col["wall_s"].mean().item(),
        "open_s_per_sample": col["open_s"].mean().item(), "decode_s_per_sample": col["decode_s"].mean().item(),
        "crop_s_per_sample": col["crop_s"].mean().item(),
        "rchar_per_sample": col["rchar"].mean().item(), "read_bytes_per_sample": col["read_bytes"].mean().item(),
        "worker_rss_anon_max_mb": max(per_worker_rss.values()), "worker_rss_anon_mean_of_max_mb": sum(per_worker_rss.values()) / len(per_worker_rss),
        "workers_seen": len(per_worker_rss),
        "pts_err_max_s": col["pts_err"].max().item(), "pts_violations_1e-4": int((col["pts_err"] > 1e-4).sum()),
        "hit_rate_during_warmup": wh / max(1, wh + wm),
        "resident_before": res_before, "cold_resident_end": res_end, "evict_passes": ev.passes if ev else 0,
        "discarded_batches_total": sess.discarded, "cgroup": cgroup_mem(),
    }


def build_cells(cfg, sel):
    cells = []
    th = len(sel["active"]["thrash"])
    ck = cfg["ckpt_family"]
    ck_sep = 3 * len(sel["active"][ck])
    for W in cfg["workers"]:
        for state in cfg["states"]:
            for pattern in cfg["patterns"]:
                for r in cfg["ratios"]:
                    cells.append({"family": "thrash", "pattern": pattern, "cache_label": f"r{r}", "ratio": float(r),
                                  "cache": max(1, round(float(r) * 3 * th)), "workers": W, "state": state})
                if pattern in cfg["ckpt_patterns"]:
                    for lab in cfg["ckpt_caches"]:
                        C = ck_sep if lab == "all" else int(lab)
                        cells.append({"family": ck, "pattern": pattern, "cache_label": f"c{lab}", "ratio": C / ck_sep,
                                      "cache": C, "workers": W, "state": state})
    return cells


def aggregate(runs):
    arms = {}
    for r in runs:
        if r.get("skipped"):
            continue
        key = "|".join(str(r[k]) for k in ("family", "layout", "pattern", "cache_label", "workers", "state"))
        arms.setdefault(key, []).append(r)
    out = {}
    for key, rs in arms.items():
        allb = [b["sps"] for r in rs for b in r["blocks"]]
        m = sum(allb) / len(allb)
        tot_s = sum(r["samples"] for r in rs)

        def wmean(f):
            return sum(r[f] * r["samples"] for r in rs) / tot_s

        out[key] = {k: rs[0][k] for k in ("family", "layout", "pattern", "cache_label", "cache", "ratio", "workers", "state", "files",
                                           "units", "keys_per_unit", "cache_over_own_files", "frames_per_file_mean")}
        out[key].update({
            "runs": len(rs), "blocks": len(allb), "samples_per_s": m, "ci95_halfwidth": ci_halfwidth(allb),
            "ci95_rel": ci_halfwidth(allb) / m if len(allb) > 1 else None,
            "per_run_sps": [r["samples_per_s"] for r in rs], "stop_reasons": [r["stop_reason"] for r in rs],
            "batch_wait_p50_ms": max(r["batch_wait_p50_ms"] for r in rs), "batch_wait_p95_ms": max(r["batch_wait_p95_ms"] for r in rs),
            "hit_rate": wmean("hit_rate"), "opens_per_sample": wmean("opens_per_sample"),
            "worker_cpu_s_per_sample": wmean("worker_cpu_s_per_sample"), "worker_wall_s_per_sample": wmean("worker_wall_s_per_sample"),
            "open_s_per_sample": wmean("open_s_per_sample"), "decode_s_per_sample": wmean("decode_s_per_sample"),
            "crop_s_per_sample": wmean("crop_s_per_sample"), "rchar_per_sample": wmean("rchar_per_sample"),
            "read_bytes_per_sample": wmean("read_bytes_per_sample"),
            "worker_rss_anon_max_mb": max(r["worker_rss_anon_max_mb"] for r in rs),
            "pts_violations_1e-4": sum(r["pts_violations_1e-4"] for r in rs), "pts_err_max_s": max(r["pts_err_max_s"] for r in rs),
        })
    return out


def main():
    sel_path, enc_root, out_dir = sys.argv[1:4]
    smoke = os.environ.get("SMOKE", "0") == "1"
    # Main process: 1 thread. The env vars are inherited by the spawn workers (set before torch
    # starts there); DataLoader workers also call torch.set_num_threads(1) themselves.
    pin_threads(1)
    cfg = {
        "enc_root": enc_root, "batch": env_int("DL_BATCH", 32), "threads_main": thread_report(),
        "workers": [int(w) for w in env_list("DL_WORKERS", "4" if smoke else "4 8")],
        "patterns": env_list("DL_PATTERNS", "all top"),
        "ratios": env_list("DL_RATIOS", "0.1 1.0" if smoke else "0.05 0.1 0.22 0.5 1.0"),
        "states": env_list("DL_STATES", "warm cold"),
        "ckpt_family": os.environ.get("DL_CKPT_FAMILY", "ckpt_5k" if smoke else "ckpt_full"),
        "ckpt_caches": env_list("DL_CKPT_CACHES", "1 all"),
        "ckpt_patterns": env_list("DL_CKPT_PATTERNS", "all" if smoke else "all top"),
        "repeats": env_int("DL_REPEATS", 1 if smoke else 2),
        "block_batches": env_int("DL_BLOCK_BATCHES", 0),  # 0 -> 4 x workers
        "min_blocks": env_int("DL_MIN_BLOCKS", 1 if smoke else 4), "max_blocks": env_int("DL_MAX_BLOCKS", 1 if smoke else 25),
        "ci_target": float(os.environ.get("DL_CI_TARGET", "0.05")),
        "arm_cap_s": float(os.environ.get("DL_ARM_CAP_S", "40" if smoke else "180")),
        "warmup_cap": env_int("DL_WARMUP_CAP", 60 if smoke else 400),
        "evict_interval": float(os.environ.get("DL_EVICT_INTERVAL_S", "0.5")),
        "budget_s": float(os.environ.get("DL_BUDGET_S", "1200" if smoke else "30000")),
    }
    sel = load_selection(sel_path, smoke)
    cells = build_cells(cfg, sel)
    D = Path(out_dir) / "loader"
    D.mkdir(parents=True, exist_ok=True)
    write_json(D / "config.json", {**cfg, "cells": cells, "n_runs_planned": len(cells) * len(LAYOUTS) * cfg["repeats"]})
    print(f"loader: {len(cells)} cells x {len(LAYOUTS)} layouts x {cfg['repeats']} repeats", flush=True)
    runs = []
    t0 = time.perf_counter()
    jl = (D / "runs.jsonl").open("a")
    for W in cfg["workers"]:
        wcells = [c for c in cells if c["workers"] == W]
        arms = [{"family": c["family"], "layout": lay, "pattern": c["pattern"], "cache": c["cache"]} for c in wcells for lay in LAYOUTS]
        aid_of = {(c["family"], lay, c["pattern"], c["cache"]): i for i, (c, lay) in enumerate((c, lay) for c in wcells for lay in LAYOUTS)}
        sess = Session(cfg, sel, arms, W)
        try:
            for rep in range(cfg["repeats"]):
                for ci, cell in enumerate(wcells):
                    order = LAYOUTS if (rep + ci) % 2 == 0 else LAYOUTS[::-1]
                    for lay in order:
                        arm = {k: cell[k] for k in ("family", "pattern", "cache_label", "ratio", "cache", "workers", "state")}
                        arm.update(layout=lay, repeat=rep)
                        aid = aid_of[(cell["family"], lay, cell["pattern"], cell["cache"])]
                        if time.perf_counter() - t0 > cfg["budget_s"]:
                            r = {**arm, "skipped": "budget"}
                        else:
                            seed = 1000 * rep + 10 * ci + LAYOUTS.index(lay)
                            try:
                                r = run_one(sess, aid, arm, seed, cfg["arm_cap_s"] / cfg["repeats"])
                            except Exception as exc:  # keep going; record the failure (a dead worker also lands here)
                                r = {**arm, "skipped": f"error: {type(exc).__name__}: {exc}"}
                                sess.close()
                                sess = Session(cfg, sel, arms, W)
                        runs.append(r)
                        jl.write(json.dumps(r, default=str) + "\n")
                        jl.flush()
                        if r.get("skipped"):
                            print(f"[{time.perf_counter() - t0:.0f}s] {lay} {cell['family']} {cell['pattern']} {cell['cache_label']} W{W} {cell['state']}: SKIPPED {r['skipped']}", flush=True)
                        else:
                            print(f"[{time.perf_counter() - t0:.0f}s] r{rep} {lay:5s} {cell['family']} {cell['pattern']} {cell['cache_label']}(C={cell['cache']}) "
                                  f"W{W} {cell['state']}: {r['samples_per_s']:.0f} sps +-{r['sps_ci95_halfwidth']:.0f}, hit {r['hit_rate']:.3f}, opens/smp {r['opens_per_sample']:.2f}, "
                                  f"cpu {r['worker_cpu_s_per_sample'] * 1e3:.2f}ms, rd {r['read_bytes_per_sample'] / 1e3:.1f}kB, {len(r['blocks'])} blk {r['stop_reason']}, "
                                  f"warm {r['warmup_s']:.1f}s, wait p95 {r['batch_wait_p95_ms']:.0f}ms, rss {r['worker_rss_anon_max_mb']:.0f}MB, cg {r['cgroup']['cgroup_current_mb']:.0f}MB", flush=True)
                    write_json(D / "arms.json", aggregate(runs))
                memlog(out_dir, f"loader_W{W}_repeat_{rep}")
        finally:
            sess.close()
    write_json(D / "arms.json", aggregate(runs))
    errs = [r for r in runs if str(r.get("skipped", "")).startswith("error")]
    print(f"loader done in {time.perf_counter() - t0:.0f}s, {len(runs)} runs, {len(errs)} errors")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
