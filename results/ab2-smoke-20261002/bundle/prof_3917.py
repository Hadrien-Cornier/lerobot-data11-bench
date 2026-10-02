"""Profile PR #3917 (episode-pool streaming) at a pinned head: where does each sample's CPU go?

Two parts:
  probe        startup index probe on a real dataset (ABC): current 4 MiB probe vs a small probe + exact moov read.
               Same #3917 fetcher, interleaved per file, index equality checked. Plus parallel throughput, moov sizes,
               and the in-RAM size of each Mp4Index (to extrapolate the full-catalog build).
  run / arms   one fresh process per run on hadriencornier/lerobot-data11-r3-sep (3 cameras, 224x224, 232 episodes).
               Per-thread CPU (from /proc/self/task, grouped by Python thread name), timed wrappers around the
               main stages, cache timing_summary, optional stack sampler, optional py-spy.

Arms (all 1 rank, return_uint8=True like lerobot-train, fetch concurrency 4 = lerobot-train's default num_workers):
  D2    DataLoader(num_workers=1, batch 8, spawn, prefetch 4) and decode_threads=2  (lerobot-train defaults)
  D6    same, decode_threads=6
  D6T   D6 + ImageTransforms(enable=True) (default lerobot augmentation set)
  R6    no DataLoader, iterate in this process, decode_threads=6, torch default threads
  R6t1  R6 with torch.set_num_threads(1) (what a DataLoader worker does)

Usage: python prof_3917.py orchestrate <out_dir>             (stdlib only)
       python prof_3917.py run --arm D6 --rep 0 --out X.json   (inside the #3917 venv)
       python prof_3917.py probe --out X.json                  (inside the #3917 venv)
Env: VENV_PR3917, PF_ARMS, PF_REPEATS (3), PF_CAP_S (150), PF_WARM_S (45), PF_SAMPLED (1), PF_PYSPY (1),
     PF_PROBE_N (100), PF_PROBE_PAR_N (400), HF_LEROBOT_HOME
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = "hadriencornier/lerobot-data11-r3-sep"
REV = "31fd7b6605cd41c6c6c7c059ccd261924d74f033"
ABC = "lerobot/abc_130k_v3_train"
ABC_REV = "68651e4929d9fb00f798937b2d62617cab5c771d"
ARMS = {
    "D2": {"loader": True, "decode_threads": 2, "transforms": False, "torch_threads": None},
    "D6": {"loader": True, "decode_threads": 6, "transforms": False, "torch_threads": None},
    "D6T": {"loader": True, "decode_threads": 6, "transforms": True, "torch_threads": None},
    "D2T": {"loader": True, "decode_threads": 2, "transforms": True, "torch_threads": None},
    "R6": {"loader": False, "decode_threads": 6, "transforms": False, "torch_threads": None},
    "R6t1": {"loader": False, "decode_threads": 6, "transforms": False, "torch_threads": 1},
    # pool-swap test: a cohort of 8 episodes admitted together drains together; prefetch 2 covers 2 of the 8 replacements
    "D6P8": {"loader": True, "decode_threads": 6, "transforms": False, "torch_threads": None, "pool": 8, "prefetch": 2, "long": True},
    "D6P8f": {"loader": True, "decode_threads": 6, "transforms": False, "torch_threads": None, "pool": 8, "prefetch": 8, "long": True},
}
BATCH, FETCH_CONC, PREFETCH = 8, 4, 4
TICK = os.sysconf("SC_CLK_TCK")


# ====================================================================== profiler (runs in the process that does the work)
def _group(name):
    return re.sub(r"[_-]?\d+$", "", name)


def thread_cpu():
    """{tid: (python name or 'native:<comm>', cpu seconds)} for every thread of this process."""
    names = {t.native_id: t.name for t in threading.enumerate() if t.native_id is not None}
    out = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            st = Path(f"/proc/self/task/{tid}/stat").read_text()
            comm = st[st.index("(") + 1 : st.rindex(")")]
            f = st[st.rindex(")") + 2 :].split()
            cpu = (int(f[11]) + int(f[12])) / TICK
        except (OSError, ValueError):
            continue
        n = names.get(int(tid))
        out[int(tid)] = (n if n is not None else f"native:{comm}", cpu)
    return out


TIMERS = {}
TLOCK = threading.Lock()
EVENTS = {}  # name -> [(unix start, seconds)] for rare events (fetches, admissions) and long waits
EVENT_ALWAYS = {"fetch_and_synthesize(clip)", "load_episode_parquet(episode)", "planner_admit"}
EVENT_SLOW_S = 0.05


def timed(name, fn):
    def w(*a, **k):
        t0, c0 = time.perf_counter(), time.thread_time()
        try:
            return fn(*a, **k)
        finally:
            dt, dc = time.perf_counter() - t0, time.thread_time() - c0
            with TLOCK:
                r = TIMERS.setdefault(name, [0, 0.0, 0.0])
                r[0] += 1
                r[1] += dt
                r[2] += dc
                if name in EVENT_ALWAYS or dt > EVENT_SLOW_S:
                    ev = EVENTS.setdefault(name, [])
                    if len(ev) < 20000:
                        ev.append((round(time.time() - dt, 3), round(dt, 4)))

    return w


CACHES = []
IDLE_LEAVES = {"wait", "get", "accept", "_recv", "select", "poll", "sleep", "_wait_for_tstate_lock", "recv_into", "_worker"}


def install_patches():
    from lerobot.datasets import streaming_dataset as sd
    from lerobot.streaming import episode_cache as ec

    C = ec.EpisodeByteCache
    orig_init = C.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        self._pool._thread_name_prefix = "cache-fetch"  # threads are created lazily with this prefix
        CACHES.append(self)

    C.__init__ = init
    C._fetch_and_synthesize = timed("fetch_and_synthesize(clip)", C._fetch_and_synthesize)
    C._open_decoder = timed("open_decoder(clip)", C._open_decoder)
    C._get_frames = timed("get_frames(camera)", C._get_frames)
    C._get_entry = timed("wait_clip_bytes(camera)", C._get_entry)
    from lerobot.streaming import episode_pool as ep

    ep.ExactCoveragePool._admit_available = timed("planner_admit", ep.ExactCoveragePool._admit_available)
    S = sd.StreamingLeRobotDataset
    S._make_episode_item = timed("make_episode_item(sample)", S._make_episode_item)
    S._apply_image_transforms = timed("apply_image_transforms(sample)", S._apply_image_transforms)
    S._load_episode_dataset = timed("load_episode_parquet(episode)", S._load_episode_dataset)
    import lerobot.streaming.mp4 as m4

    ec.synthesize_mp4 = timed("synthesize_mp4(clip)", ec.synthesize_mp4)
    _ = m4


class Prof(threading.Thread):
    """Every 5 s: write per-thread CPU, timers and cache timings. Optional stack sampler after warm_s."""

    def __init__(self, out_dir, tag, sample, warm_s, hz=100):
        super().__init__(daemon=True, name="prof")
        self.dir, self.tag, self.sample, self.warm_s, self.hz = Path(out_dir), tag, sample, warm_s, hz
        self.t0 = time.time()
        self.leaf, self.incl, self.nsamp = {}, {}, Counter()

    def _stacks(self):
        me = threading.get_ident()
        names = {t.ident: t.name for t in threading.enumerate()}
        for ident, fr in sys._current_frames().items():
            if ident == me:
                continue
            g = _group(names.get(ident, "?"))
            self.nsamp[g] += 1
            seen, depth, leaf = set(), 0, None
            while fr is not None and depth < 80:
                co = fr.f_code
                fn = co.co_filename
                short = fn.split("site-packages/")[-1] if "site-packages/" in fn else fn.rsplit("/", 1)[-1]
                key = f"{co.co_name} ({short})"
                if leaf is None:
                    leaf = f"{co.co_name} ({short}:{fr.f_lineno})"
                seen.add(key)
                fr, depth = fr.f_back, depth + 1
            self.leaf.setdefault(g, Counter())[leaf] += 1
            if leaf.split(" ", 1)[0] in IDLE_LEAVES:
                self.nsamp[g + " (idle)"] += 1
                continue
            c = self.incl.setdefault(g, Counter())
            for k in seen:
                c[k] += 1

    def _dump(self):
        snap = {"t": time.time(), "threads": {str(k): v for k, v in thread_cpu().items()},
                "timers": {k: list(v) for k, v in TIMERS.items()}}
        if CACHES:
            try:
                snap["cache"] = CACHES[-1].timing_summary()
                snap["cache_reserved_gb"] = CACHES[-1].reserved_bytes / 2**30
                snap["open_decoders"] = CACHES[-1].open_decoder_count
            except Exception as e:  # noqa: BLE001
                snap["cache_err"] = repr(e)
        try:
            import psutil

            mi = psutil.Process().memory_full_info()
            snap["pss_gb"] = getattr(mi, "pss", mi.rss) / 2**30
        except Exception:  # noqa: BLE001
            pass
        with open(self.dir / f"{self.tag}.snaps.jsonl", "a") as f:
            f.write(json.dumps(snap) + "\n")
        if self.sample and self.nsamp:
            top = {g: {"samples": self.nsamp[g], "leaf": self.leaf.get(g, Counter()).most_common(25),
                       "inclusive": self.incl.get(g, Counter()).most_common(40)} for g in self.nsamp}
            tmp = self.dir / f"{self.tag}.stacks.tmp"
            tmp.write_text(json.dumps({"sampling_since_s": self.warm_s, "hz": self.hz, "groups": top}))
            tmp.replace(self.dir / f"{self.tag}.stacks.json")
        with TLOCK:
            ev = {k: list(v) for k, v in EVENTS.items()}
        tmp = self.dir / f"{self.tag}.events.tmp"
        tmp.write_text(json.dumps(ev))
        tmp.replace(self.dir / f"{self.tag}.events.json")

    def run(self):
        last, active = 0.0, None  # sampling starts warm_s after the first assembled sample
        while True:
            now = time.time()
            if now - last >= 5:
                self._dump()
                last = now
            if active is None and TIMERS.get("make_episode_item(sample)"):
                active = now
            if self.sample and active is not None and now - active >= self.warm_s:
                self._stacks()
            time.sleep(1 / self.hz)


def worker_init(_wid):
    d = os.environ["PF_RUN_DIR"]
    tag = os.environ["PF_TAG"] + ".worker"
    install_patches()
    Prof(d, tag, os.environ.get("PF_SAMPLE") == "1", float(os.environ["PF_WARM_S"])).start()


# ====================================================================== run (inside the #3917 venv)
def cmd_run(a):
    arm = ARMS[a.arm]
    out = Path(a.out)
    rd = out.parent
    tag = out.stem
    res = {"arm": a.arm, "cfg": arm, "rep": a.rep, "sample": a.sample, "status": "started", "utc_start": time.strftime("%FT%TZ", time.gmtime())}

    def dump(status):
        res["status"] = status
        out.with_suffix(".tmp").write_text(json.dumps(res, indent=1, default=str))
        out.with_suffix(".tmp").replace(out)

    import torch

    if arm["torch_threads"]:
        torch.set_num_threads(arm["torch_threads"])
    res["torch_threads"] = torch.get_num_threads()
    install_patches()
    os.environ.update(PF_RUN_DIR=str(rd), PF_TAG=tag, PF_SAMPLE="1" if a.sample else "0", PF_WARM_S=str(a.warm_s))
    if not arm["loader"]:
        Prof(rd, tag + ".main", a.sample, a.warm_s).start()
    else:
        Prof(rd, tag + ".main", False, a.warm_s).start()  # main process: only CPU snapshots
    t_start = time.perf_counter()
    try:
        from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

        tf = None
        if arm["transforms"]:
            from lerobot.transforms.transforms import ImageTransforms, ImageTransformsConfig

            tf = ImageTransforms(ImageTransformsConfig(enable=True))
        extra = {}
        if arm.get("pool"):
            extra = {"episode_pool_size": arm["pool"], "prefetch_episodes": arm["prefetch"]}
        ds = StreamingLeRobotDataset(REPO, revision=REV, image_transforms=tf, max_num_shards=FETCH_CONC, decode_threads=arm["decode_threads"],
                                     return_uint8=True, repeat=True, seed=100 + a.rep, **extra)
        res["build_s"] = time.perf_counter() - t_start
        if arm["loader"]:
            from torch.utils.data import DataLoader

            it = iter(DataLoader(ds, batch_size=BATCH, num_workers=1, prefetch_factor=PREFETCH, persistent_workers=True,
                                 multiprocessing_context="spawn", worker_init_fn=worker_init))
            per = BATCH
        else:
            it = iter(ds)
            per = 1
        times = []
        t_iter = time.perf_counter()
        n = 0
        spy = None
        for _ in it:
            now = time.perf_counter()
            n += per
            times.append(now)
            if len(times) == 1:
                res["first_s"] = now - t_iter
                dump("running")
            if a.pyspy and spy is None and now - times[0] >= a.warm_s:
                import psutil

                kids = [c for c in psutil.Process().children(recursive=True) if "spawn_main" in " ".join(c.cmdline())]  # the DataLoader worker, not the resource tracker
                pid = kids[0].pid if (arm["loader"] and kids) else os.getpid()
                flags = ["--gil"] if a.pyspy == "gil" else ["--idle"]
                spy = subprocess.Popen([os.path.join(os.path.dirname(sys.executable), "py-spy"), "record", "--pid", str(pid), "--duration",
                                        str(int(a.pyspy_s)), "-r", "100", "--nonblocking", *flags, "-f", "raw", "-o", str(rd / f"{tag}.pyspy.txt")],
                                       stdout=open(rd / f"{tag}.pyspy.log", "w"), stderr=subprocess.STDOUT)
                res["pyspy"] = {"pid": pid, "mode": a.pyspy, "start_after_first_s": now - times[0], "seconds": a.pyspy_s}
            if now - times[0] > a.cap_s:
                break
        t_first = times[0]
        w0 = t_first + a.warm_s
        win = [t for t in times if t >= w0]
        res["n"] = n
        res["steady_window_s"] = (win[-1] - win[0]) if len(win) > 1 else None
        res["sps"] = ((len(win) - 1) * per / res["steady_window_s"]) if res["steady_window_s"] else None
        res["sps_all"] = (len(times) - 1) * per / (times[-1] - t_first) if len(times) > 1 else None
        gaps = sorted(b - c for b, c in zip(win[1:], win[:-1], strict=False))
        if gaps:
            res["wait_ms_p50"] = gaps[len(gaps) // 2] * 1e3
            res["wait_ms_p95"] = gaps[int(len(gaps) * 0.95)] * 1e3
        # unix-time window for snapshot analysis
        off = time.time() - time.perf_counter()
        res["window_unix"] = [w0 + off, (win[-1] if win else times[-1]) + off]
        res["samples_in_window"] = (len(win) - 1) * per if len(win) > 1 else 0
        # 2 s bins of delivered samples over the whole run, and the longest gaps (stalls)
        bins = Counter(int((t - t_first) // 2) for t in times[1:])
        res["timeline_2s"] = [bins.get(i, 0) * per for i in range(int((times[-1] - t_first) // 2) + 1)]
        res["t_first_unix"] = t_first + off
        allg = sorted(((b - c), c - t_first) for b, c in zip(times[1:], times[:-1], strict=False))
        res["longest_gaps"] = [(round(g, 3), round(at, 1)) for g, at in allg[-10:]]
        if spy is not None:
            try:
                spy.wait(timeout=a.pyspy_s + 30)
            except subprocess.TimeoutExpired:
                spy.kill()
            res["pyspy"]["rc"] = spy.returncode
        time.sleep(1)
        dump("ok")
    except Exception as exc:  # noqa: BLE001
        import traceback

        res["error"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-4000:]
        dump("error")
    try:
        import psutil

        for c in psutil.Process().children(recursive=True):
            c.kill()
    except Exception:  # noqa: BLE001
        pass
    os._exit(0)


# ====================================================================== probe (inside the #3917 venv)
def cmd_probe(a):
    import random
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from huggingface_hub import HfApi

    from lerobot.datasets.streaming_sidecar import range_backend_for_root
    from lerobot.streaming import mp4
    from lerobot.streaming.range_fetch import make_range_fetcher

    res = {"dataset": ABC, "rev": ABC_REV, "status": "started"}
    out = Path(a.out)

    def save(status):
        res["status"] = status
        out.write_text(json.dumps(res, indent=1, default=str))

    t0 = time.perf_counter()
    files = [(f.path, f.size) for f in HfApi().list_repo_tree(ABC, repo_type="dataset", revision=ABC_REV, path_in_repo="videos", recursive=True)
             if getattr(f, "size", None) is not None and f.path.endswith(".mp4")]
    res["list_s"] = time.perf_counter() - t0
    sizes = np.array([s for _, s in files], dtype=np.int64)
    probe_default = 4 * 1024 * 1024
    res["catalog"] = {"mp4_files": len(files), "total_gb": float(sizes.sum() / 1e9), "size_mb_p50": float(np.median(sizes) / 1e6),
                      "size_mb_min": float(sizes.min() / 1e6), "size_mb_max": float(sizes.max() / 1e6),
                      "default_probe_bytes_total_gb": float(np.minimum(sizes, probe_default).sum() / 1e9)}
    save("listed")
    root = f"hf://datasets/{ABC}@{ABC_REV}"
    fetcher = make_range_fetcher(root, range_backend=range_backend_for_root(root), workers=16)
    CNT = threading.local()

    def rr(path, off, ln):
        b = fetcher.read_range(path, off, ln)
        CNT.calls = getattr(CNT, "calls", 0) + 1
        CNT.bytes = getattr(CNT, "bytes", 0) + len(b)
        return b

    def small_probe(path, size, first=64 * 1024):
        data = rr(path, 0, min(first, size))
        top = list(mp4.iter_boxes(data, 0, len(data), absolute_base=0, allow_truncated=True))
        moov = next((b for b in top if b.type == b"moov"), None)
        if moov is None:
            return None, "no moov in first 64 KiB"
        need = moov.end + 16  # moov plus the following mdat header
        if need > len(data):
            data = data + rr(path, len(data), min(need, size) - len(data))
        return mp4.parse_mp4_index(path, data, file_size=size), None

    def one(method, path, size):
        CNT.calls, CNT.bytes = 0, 0
        t = time.perf_counter()
        err = None
        if method == "default":
            idx = mp4.fetch_mp4_index(path, rr, file_size=size)
        else:
            idx, err = small_probe(path, size)
        return idx, {"s": time.perf_counter() - t, "calls": CNT.calls, "bytes": CNT.bytes, "err": err}

    def same(x, y):
        if x is None or y is None:
            return False
        for k in ("moov_offset", "mdat_offset", "mdat_payload_offset", "mdat_payload_size", "faststart", "codec", "timescale", "track_id", "width",
                  "height", "stsd_body"):
            if getattr(x, k) != getattr(y, k):
                return False
        return all(np.array_equal(getattr(x, k), getattr(y, k)) for k in ("sample_pts", "sample_durations", "sample_composition_offsets",
                                                                           "sample_sizes", "sample_offsets", "sync_samples"))

    rng = random.Random(0)
    pick = rng.sample(files, min(a.n, len(files)))
    rows = []
    for i, (path, size) in enumerate(pick):
        order = ["default", "small"] if i % 2 == 0 else ["small", "default"]
        got = {}
        for m in order:
            try:
                got[m] = one(m, path, size)
            except Exception as e:  # noqa: BLE001
                got[m] = (None, {"err": f"{type(e).__name__}: {e}"})
        d, s = got["default"][0], got["small"][0]
        arr_bytes = sum(getattr(d, k).nbytes for k in ("sample_pts", "sample_durations", "sample_composition_offsets", "sample_sizes",
                                                        "sample_offsets", "sync_samples")) if d is not None else None
        moov_size = (d.mdat_offset - d.moov_offset) if d is not None and d.faststart else None
        rows.append({"path": path, "size": size, "frames": int(len(d.sample_sizes)) if d is not None else None, "faststart": d.faststart if d else None,
                     "moov_to_mdat_bytes": moov_size, "index_array_bytes": arr_bytes, "same_index": same(d, s),
                     "default": got["default"][1], "small": got["small"][1]})
        if i % 20 == 0:
            res["rows"] = rows
            save("sequential")
    res["rows"] = rows
    ok = [r for r in rows if r["same_index"]]

    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else None

    res["sequential"] = {
        "files": len(rows), "same_index": len(ok),
        "default_ms_p50": med([r["default"]["s"] * 1e3 for r in ok]), "small_ms_p50": med([r["small"]["s"] * 1e3 for r in ok]),
        "default_kb_p50": med([r["default"]["bytes"] / 1024 for r in ok]), "small_kb_p50": med([r["small"]["bytes"] / 1024 for r in ok]),
        "small_calls_mean": sum(r["small"]["calls"] for r in ok) / max(1, len(ok)),
        "moov_kb_p50": med([r["moov_to_mdat_bytes"] / 1024 for r in ok if r["moov_to_mdat_bytes"]]),
        "moov_kb_max": max([r["moov_to_mdat_bytes"] / 1024 for r in ok if r["moov_to_mdat_bytes"]] or [0]),
        "frames_p50": med([r["frames"] for r in ok]), "index_array_kb_p50": med([r["index_array_bytes"] / 1024 for r in ok]),
        "index_array_bytes_per_frame": sum(r["index_array_bytes"] for r in ok) / max(1, sum(r["frames"] for r in ok)),
    }
    save("sequential_done")
    # parallel throughput, 16 threads like StreamingLeRobotDataset's default max_num_shards
    par = {}
    pick2 = rng.sample(files, min(a.par_n, len(files)))
    for m in (["default", "small"] if a.par_n % 2 == 0 else ["small", "default"]):
        t = time.perf_counter()
        with ThreadPoolExecutor(16) as ex:
            rs = list(ex.map(lambda ps, m=m: one(m, *ps)[1], pick2))
        wall = time.perf_counter() - t
        par[m] = {"files": len(pick2), "wall_s": wall, "files_per_s": len(pick2) / wall, "gb": sum(r.get("bytes", 0) for r in rs) / 1e9,
                  "errors": sum(1 for r in rs if r.get("err"))}
    res["parallel16"] = par
    n = res["catalog"]["mp4_files"]
    res["extrapolate_full_catalog"] = {m: {"minutes_at_16_threads": n / par[m]["files_per_s"] / 60, "gb_read": par[m]["gb"] / len(pick2) * n}
                                       for m in par}
    res["extrapolate_full_catalog"]["index_arrays_gb"] = res["sequential"]["index_array_bytes_per_frame"] * (
        res["sequential"]["frames_p50"] or 0) * n / 1e9
    try:
        res["fetcher_timing"] = fetcher.timing_summary()
    except Exception:  # noqa: BLE001
        pass
    fetcher.close()
    save("ok")
    os._exit(0)


# ====================================================================== analysis (stdlib)
def load_snaps(p):
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def window_delta(snaps, w0, w1):
    """Per-group CPU seconds and timer deltas between the snapshots closest inside [w0, w1]."""
    inside = [s for s in snaps if w0 <= s["t"] <= w1]
    if len(inside) < 2:
        return None
    a, b = inside[0], inside[-1]
    g = Counter()
    for tid, (name, cpu) in b["threads"].items():
        prev = a["threads"].get(tid, [name, 0.0])[1]
        g[_group(name)] += cpu - prev
    tm = {}
    for k, v in b["timers"].items():
        p = a["timers"].get(k, [0, 0.0, 0.0])
        tm[k] = [v[0] - p[0], v[1] - p[1], v[2] - p[2]]
    ca = {k: b.get("cache", {}).get(k, 0) - a.get("cache", {}).get(k, 0) for k in b.get("cache", {})}
    return {"span_s": b["t"] - a["t"], "cpu_by_group_s": dict(g), "timers": tm, "cache": ca, "pss_gb": max(s.get("pss_gb", 0) for s in snaps),
            "open_decoders": b.get("open_decoders"), "reserved_gb": b.get("cache_reserved_gb")}


def analyze_run(run_json):
    r = json.loads(Path(run_json).read_text())
    if r.get("status") != "ok" or not r.get("window_unix"):
        return r
    d = Path(run_json).parent
    tag = Path(run_json).stem
    w0, w1 = r["window_unix"]
    work = d / (f"{tag}.worker.snaps.jsonl" if r["cfg"]["loader"] else f"{tag}.main.snaps.jsonl")
    wd = window_delta(load_snaps(work), w0, w1)
    md = window_delta(load_snaps(d / f"{tag}.main.snaps.jsonl"), w0, w1) if r["cfg"]["loader"] else None
    r["worker_window"] = wd
    r["main_window"] = md
    if wd and r.get("sps"):
        n = r["sps"] * wd["span_s"]  # samples in the snapshot span, from the measured rate
        r["per_sample_ms"] = {
            "cpu_by_group": {k: v / n * 1e3 for k, v in sorted(wd["cpu_by_group_s"].items(), key=lambda x: -x[1])},
            "cpu_total": sum(v for k, v in wd["cpu_by_group_s"].items() if k != "prof") / n * 1e3,  # profiler thread excluded
            "timers": {k: {"calls_per_sample": v[0] / n, "wall_ms": v[1] / n * 1e3, "cpu_ms": v[2] / n * 1e3} for k, v in wd["timers"].items()},
        }
        if md:
            r["per_sample_ms"]["main_process_cpu"] = sum(md["cpu_by_group_s"].values()) / n * 1e3
    return r


# ====================================================================== orchestrate (stdlib only)
def cmd_orchestrate(a):
    out = Path(a.out) / "prof"
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    py = os.environ["VENV_PR3917"] + "/bin/python"
    arms = os.environ.get("PF_ARMS", "D2 D6 D6T R6 R6t1 D6P8 D6P8f").split()
    reps = int(os.environ.get("PF_REPEATS", "3"))
    cap = float(os.environ.get("PF_CAP_S", "150"))
    warm = float(os.environ.get("PF_WARM_S", "45"))
    sampled = os.environ.get("PF_SAMPLED", "1") == "1"
    pyspy = os.environ.get("PF_PYSPY", "1") == "1"
    res = {"config": {"arms": arms, "reps": reps, "cap_s": cap, "warm_s": warm, "batch": BATCH, "fetch_concurrency": FETCH_CONC, "repo": REPO, "rev": REV,
                      "pr3917_sha": os.environ.get("PR3917_SHA")}, "runs": []}

    def save():
        (out / "prof.json").write_text(json.dumps(res, indent=1, default=str))

    def go(arm, rep, sample, extra_prefix=None, tag=None, cap_override=None, extra_args=()):
        tag = tag or f"{arm}_r{rep}{'_s' if sample else ''}"
        o = runs / f"{tag}.json"
        c = cap_override or cap
        cmd = [py, str(HERE / "prof_3917.py"), "run", "--arm", arm, "--rep", str(rep), "--cap-s", str(c), "--warm-s", str(warm), "--out", str(o)]
        if sample:
            cmd.append("--sample")
        cmd += list(extra_args)
        if extra_prefix:
            cmd = [*extra_prefix(tag), *cmd]
        t = time.time()
        with open(runs / f"{tag}.log", "w") as lg:
            try:
                subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, timeout=c + 600)
            except subprocess.TimeoutExpired:
                pass
        subprocess.run(["pkill", "-f", "--", f"--out {o}"], check=False)
        subprocess.run(["pkill", "-f", "multiprocessing.spawn"], check=False)  # orphaned DataLoader workers, if any
        time.sleep(2)
        try:
            r = analyze_run(o)
        except Exception as e:  # noqa: BLE001
            r = {"status": "no_result", "err": repr(e)}
        r["tag"], r["wall_s"] = tag, time.time() - t
        res["runs"].append(r)
        ps = r.get("per_sample_ms") or {}
        print(f"[{time.strftime('%H:%M:%S')}] {tag}: {r.get('status')} sps {r.get('sps')} first {r.get('first_s')} cpu/sample {ps.get('cpu_total')} "
              f"groups {json.dumps({k: round(v, 2) for k, v in (ps.get('cpu_by_group') or {}).items()})[:300]} {r.get('error', '')}", flush=True)
        save()

    # clean repeats, arms interleaved, order reversed every other repeat
    long_cap = float(os.environ.get("PF_CAP_LONG_S", "600"))
    long_reps = int(os.environ.get("PF_LONG_REPEATS", "2"))
    short = [x for x in arms if not ARMS[x].get("long")]
    longs = [x for x in arms if ARMS[x].get("long")]
    for rep in range(reps):
        order = short if rep % 2 == 0 else short[::-1]
        for arm in order:
            go(arm, rep, False)
    for rep in range(long_reps):
        order = longs if rep % 2 == 0 else longs[::-1]
        for arm in order:
            go(arm, rep, False, cap_override=long_cap)
    if sampled:
        for arm in short:
            go(arm, 90, True)
    if pyspy:
        ok = subprocess.run([os.environ["VENV_PR3917"] + "/bin/py-spy", "dump", "--pid", str(os.getpid())], capture_output=True, text=True)
        res["pyspy_selftest"] = {"rc": ok.returncode, "err": ok.stderr[-500:]}
        if ok.returncode == 0:
            for arm in [x for x in os.environ.get("PF_PYSPY_ARMS", "D2 D6").split() if x in arms]:
                for mode in ("gil", "all"):
                    go(arm, 91, False, tag=f"{arm}_pyspy_{mode}", extra_args=["--pyspy", mode, "--pyspy-s", str(min(60, cap - warm - 10))])
    res["summary"] = summarize(res)
    save()
    write_md(out, res)


def summarize(res):
    by = {}
    for r in res["runs"]:
        if r.get("sample") or "pyspy" in r.get("tag", "") or not r.get("sps"):
            continue
        by.setdefault(r["arm"], []).append(r)
    s = {}
    for arm, rs in by.items():
        v = [r["sps"] for r in rs]
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
        groups = Counter()
        timers = {}
        tot = []
        for r in rs:
            ps = r.get("per_sample_ms") or {}
            tot.append(ps.get("cpu_total") or 0)
            for k, x in (ps.get("cpu_by_group") or {}).items():
                groups[k] += x / len(rs)
            for k, x in (ps.get("timers") or {}).items():
                t = timers.setdefault(k, Counter())
                for kk, xx in x.items():
                    t[kk] += xx / len(rs)
        tl = [r.get("timeline_2s") for r in rs if r.get("timeline_2s")]
        s[arm] = {"timeline_2s": tl, "longest_gaps": [r.get("longest_gaps") for r in rs], "sps": v, "sps_mean": m, "sps_sd": sd, "first_s": [r.get("first_s") for r in rs], "build_s": [r.get("build_s") for r in rs],
                  "cpu_ms_per_sample": sum(tot) / len(tot), "cpu_ms_by_group": dict(groups.most_common()), "timers": {k: dict(v) for k, v in timers.items()},
                  "wait_ms_p95": [r.get("wait_ms_p95") for r in rs], "pss_gb": [(r.get("worker_window") or {}).get("pss_gb") for r in rs]}
    return s


def write_md(out, res):
    L = [f"# PR #3917 profile ({res['config'].get('pr3917_sha')})", "", f"Config: {json.dumps(res['config'])}", ""]
    L += ["| arm | samples/s (mean +- sd, n) | CPU ms/sample | top thread groups (ms/sample) | first sample s | PSS GB |", "|---|---|---|---|---|---|"]
    for arm, s in res.get("summary", {}).items():
        top = ", ".join(f"{k} {v:.1f}" for k, v in list(s["cpu_ms_by_group"].items())[:5])
        L.append(f"| {arm} | {s['sps_mean']:.1f} +- {s['sps_sd']:.1f} (n={len(s['sps'])}) | {s['cpu_ms_per_sample']:.1f} | {top} | "
                 f"{', '.join(f'{x:.0f}' for x in s['first_s'] if x)} | {', '.join(f'{x:.1f}' for x in s['pss_gb'] if x)} |")
    L += ["", "## Timers (per sample, averaged over clean repeats)", ""]
    for arm, s in res.get("summary", {}).items():
        L.append(f"- **{arm}**: " + "; ".join(f"{k}: {v.get('calls_per_sample', 0):.3f} calls, {v.get('wall_ms', 0):.2f} ms wall, {v.get('cpu_ms', 0):.2f} ms CPU"
                                             for k, v in s["timers"].items()))
    L += ["", "## Stack samples (sampled runs, top inclusive per thread group)", ""]
    for r in res["runs"]:
        if not r.get("sample"):
            continue
        p = out / "runs" / f"{r['tag']}.{'worker' if r['cfg']['loader'] else 'main'}.stacks.json"
        if p.exists():
            st = json.loads(p.read_text())
            L.append(f"### {r['tag']}")
            for g, v in sorted(st["groups"].items(), key=lambda x: -x[1]["samples"]):
                L.append(f"- {g} ({v['samples']} samples): " + "; ".join(f"{k} {c}" for k, c in v["inclusive"][:12]))
    if "pyspy_selftest" in res:
        L += ["", f"py-spy self test: {res['pyspy_selftest']}"]
    (out / "prof.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("orchestrate")
    o.add_argument("out")
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True, choices=list(ARMS))
    r.add_argument("--rep", type=int, default=0)
    r.add_argument("--cap-s", type=float, default=150)
    r.add_argument("--warm-s", type=float, default=45)
    r.add_argument("--sample", action="store_true")
    r.add_argument("--pyspy", choices=["all", "gil"], default=None)
    r.add_argument("--pyspy-s", type=float, default=60)
    r.add_argument("--out", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--n", type=int, default=int(os.environ.get("PF_PROBE_N", "100")))
    p.add_argument("--par-n", type=int, default=int(os.environ.get("PF_PROBE_PAR_N", "400")))
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"orchestrate": cmd_orchestrate, "run": cmd_run, "probe": cmd_probe}[a.cmd](a)
