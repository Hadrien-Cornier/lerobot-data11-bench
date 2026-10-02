"""Streaming camera-parallelism benchmark: StreamingLeRobotDataset over hf://.

Arms (source trees prepared by entrypoint.sh):
  base   main + #4702 (cameras decoded one after another)
  p1     base + parallel camera threads
  p1p2   p1 + decoder-cache opens outside the global lock
  pr3917 PR #3917 head (episode-scoped range reads + MP4 sidecar), decode_threads 2 and 8

Subcommands:
  orchestrate OUT          plan and run every (arm, regime, repeat) in a fresh child process (stdlib only)
  run ARGS                 one measured run (inside the arm's venv), writes one JSON
  mapcheck IN.npz OUT.json direct decode of saved samples with the base install (no dataset download)
  summarize OUT            results.json + summary.md

Never print HF_TOKEN.
"""

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REPO = "lerobot/abc_130k_v3_train"
REV = "68651e4929d9fb00f798937b2d62617cab5c771d"
COUNTERS = ("requests", "resolve_requests", "cdn_requests", "api_requests", "http_bytes", "http_s", "status_429",
            "status_other_err", "cdn_hit", "cdn_miss", "fetch_calls", "fetch_bytes", "fetch_s")


def env_int(name, default):
    return int(os.environ.get(name, default))


# ====================================================================== run (inside an arm venv)
class MemSampler(threading.Thread):
    """Peak PSS of this process tree plus peak cgroup memory.current, sampled every second."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak_pss = 0
        self.peak_rss = 0
        self.peak_cg = 0
        self.last_pss = 0
        self.timeline = []  # (t_s, pss_gb, n_procs) every 5 s
        self.stop = False

    def run(self):
        import psutil

        me = psutil.Process()
        self._t0 = time.perf_counter()
        while not self.stop:
            pss = rss = 0
            try:
                procs = [me] + me.children(recursive=True)
            except Exception:
                procs = [me]
            for p in procs:
                try:
                    mi = p.memory_full_info()
                    pss += getattr(mi, "pss", mi.rss)
                    rss += mi.rss
                except Exception:
                    pass
            self.last_pss = pss
            if not self.timeline or time.perf_counter() - self._t0 - self.timeline[-1][0] >= 5:
                self.timeline.append((round(time.perf_counter() - self._t0, 1), round(pss / 2**30, 2), len(procs)))
            self.peak_pss = max(self.peak_pss, pss)
            self.peak_rss = max(self.peak_rss, rss)
            try:
                self.peak_cg = max(self.peak_cg, int(Path("/sys/fs/cgroup/memory.current").read_text()))
            except Exception:
                pass
            time.sleep(1.0)


def _snapshot():
    import common

    s = {k: common.REMOTE[k] for k in COUNTERS}
    s["resolver_min_seen"] = common.REMOTE["resolver_min_seen"]
    s["pid"] = os.getpid()
    return s


def _zero_counters():
    import common

    for k in COUNTERS:
        common.REMOTE[k] = 0 if isinstance(common.REMOTE[k], int) else 0.0


def _worker_init(_wid):
    _zero_counters()  # forked workers inherit the parent's counts


def _collate(batch):
    """Stack same-shape tensors per key, keep everything else as a list (default_collate fails on some
    keys of these samples)."""
    import torch

    snap = _snapshot()  # taken where the batch is assembled (worker, or main when num_workers=0)
    out = {}
    for k in batch[0]:
        vals = [b.get(k) for b in batch]
        if all(isinstance(v, torch.Tensor) for v in vals) and len({tuple(v.shape) for v in vals}) == 1:
            out[k] = torch.stack(vals)
        else:
            out[k] = vals
    out["_spy"] = snap
    return out


def _subset_episodes(total, n, seed=0):
    import numpy as np

    return sorted(int(x) for x in np.random.default_rng(seed).choice(total, size=n, replace=False))


def _patch_pr3917_sidecar(episodes):
    """#3917 indexes EVERY MP4 of the dataset at construction (4 MiB header probe per file, all sample
    arrays in RAM). For abc_130k (~38k files, ~1e9 samples) that is ~150 GB of reads and tens of GB of
    RAM, so the benchmark limits the sidecar to the files of the selected episodes. Reads at run time
    are unchanged."""
    import lerobot.datasets.streaming_sidecar as sc

    orig = sc.make_sidecar_spec

    class SubsetMeta:
        def __init__(self, meta):
            self._m = meta

        @property
        def total_episodes(self):
            return len(episodes)

        def get_video_file_path(self, i, key):
            return self._m.get_video_file_path(episodes[i], key)

        def __getattr__(self, name):
            return getattr(self._m, name)

    def make_spec(meta, data_root, *, token=None):
        return orig(SubsetMeta(meta), data_root, token=token)

    sc.make_sidecar_spec = make_spec


def build_dataset(a):
    from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

    if a.arm == "pr3917":
        n_eps = env_int("PR3917_EPISODES", 400)
        episodes = _subset_episodes(129225, n_eps, seed=0)
        _patch_pr3917_sidecar(episodes)
        return StreamingLeRobotDataset(REPO, revision=REV, episodes=episodes, seed=a.seed,
                                       decode_threads=a.decode_threads)
    return StreamingLeRobotDataset(REPO, revision=REV, buffer_size=a.buffer_size, seed=a.seed)


def cmd_run(a):
    import common

    res = {"arm": a.arm, "regime": a.regime, "rep": a.rep, "num_workers": a.num_workers, "seed": a.seed,
           "decode_threads": a.decode_threads if a.arm == "pr3917" else None,
           "buffer_size": a.buffer_size if a.arm != "pr3917" else None,
           "decoder_cache_env": os.environ.get("LEROBOT_VIDEO_DECODER_CACHE_SIZE"), "n_target": a.n_steady,
           "cap_s": a.cap_s, "status": "started", "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out_path = Path(a.out)
    lock = threading.Lock()

    def dump(status):
        with lock:
            res["status"] = status
            tmp = out_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(res, indent=1, default=str))
            tmp.replace(out_path)

    quota = common.cpu_quota()
    common.pin_threads(quota)
    import torch

    common.pin_threads(quota)
    common.install_http_spy()
    import lerobot

    res["lerobot_file"] = lerobot.__file__
    res["cpu_quota"] = quota
    mem = MemSampler()
    mem.start()
    t_start = time.perf_counter()

    def finalize_mem():
        res["peak_pss_gb"] = mem.peak_pss / 2**30
        res["peak_rss_sum_gb"] = mem.peak_rss / 2**30
        res["peak_cgroup_current_gb"] = mem.peak_cg / 2**30
        res["mem_timeline"] = mem.timeline

    def watchdog():
        time.sleep(a.cap_s + 20)  # the loop checks the cap between batches; this catches a blocked next()
        finalize_mem()
        res["watchdog"] = True
        dump("capped_blocked")
        os._exit(0)

    threading.Thread(target=watchdog, daemon=True).start()

    try:
        ds = build_dataset(a)
        res["build_s"] = time.perf_counter() - t_start
        import psutil

        res["rss_after_build_gb"] = psutil.Process().memory_info().rss / 2**30
        res["build_http"] = _snapshot()
        _zero_counters()
        from torch.utils.data import DataLoader

        kw = dict(batch_size=a.batch_size, num_workers=a.num_workers, collate_fn=_collate)
        if a.num_workers > 0:
            kw.update(worker_init_fn=_worker_init, persistent_workers=False)
        dl = DataLoader(ds, **kw)
        latest = {}  # pid -> latest cumulative snapshot

        def totals():
            t = {k: sum(s[k] for s in latest.values()) for k in COUNTERS}
            t["resolver_min_seen"] = min([s["resolver_min_seen"] for s in latest.values()] or [-1])
            return t

        hashes, saved, n = [], [], 0
        t_iter = time.perf_counter()
        t_first = None
        first_tot = None
        batch_times = []
        for batch in dl:
            now = time.perf_counter()
            snap = batch.pop("_spy")
            latest[snap["pid"]] = snap
            bs = len(batch["episode_index"]) if "episode_index" in batch else len(batch.get("_list", []))
            n += bs
            batch_times.append(round(now - t_iter, 4))
            if a.hash_n and len(hashes) < a.hash_n:
                cams = sorted(k for k in batch if k.startswith("observation.images."))
                for i in range(bs):
                    if len(hashes) >= a.hash_n:
                        break
                    h = {k: (float(batch[k][i]) if k == "timestamp" else int(batch[k][i])) if k in batch else None
                         for k in ("episode_index", "frame_index", "index", "timestamp")}
                    for c in cams:
                        x = batch[c][i].contiguous()
                        h[c] = hashlib.sha256(x.numpy().tobytes()).hexdigest()[:16]
                        h[c + ".dtype"] = str(x.dtype)
                        h[c + ".shape"] = list(x.shape)
                    hashes.append(h)
                    if len(saved) < a.save_n:
                        saved.append({**{k: h[k] for k in ("episode_index", "frame_index", "index", "timestamp")
                                         if h[k] is not None},
                                      **{c: batch[c][i].clone() for c in cams}})
            if t_first is None:
                res["pss_at_first_batch_gb"] = mem.last_pss / 2**30
                t_first = now
                first_tot = totals()
                res["first_batch_s"] = now - t_iter
                res["first_batch_s_incl_build"] = now - t_start
                res["first_batch_http"] = first_tot
                dump("running")
            res["n_samples"] = n
            steady_n = n - res.get("first_batch_n", bs)
            if "first_batch_n" not in res:
                res["first_batch_n"] = bs
            if steady_n >= a.n_steady or now - t_start > a.cap_s:
                if now - t_start > a.cap_s and steady_n < a.n_steady:
                    res["capped"] = True
                break
        t_end = time.perf_counter()
        tot = totals()
        steady_n = n - res.get("first_batch_n", 0)
        steady_s = t_end - t_first if t_first else None
        res.update(steady_n=steady_n, steady_s=steady_s,
                   sps=(steady_n / steady_s) if steady_s and steady_n > 0 else None,
                   batch_times=batch_times, total_http=tot)
        if first_tot is not None and steady_n > 0:
            d = {k: tot[k] - first_tot[k] for k in COUNTERS}
            res["steady_http"] = d
            res["per_sample"] = {"requests": d["requests"] / steady_n, "resolve": d["resolve_requests"] / steady_n,
                                 "cdn": d["cdn_requests"] / steady_n, "api": d["api_requests"] / steady_n,
                                 "mb": d["http_bytes"] / steady_n / 1e6, "status_429": d["status_429"],
                                 "cdn_hit": d["cdn_hit"], "cdn_miss": d["cdn_miss"]}
        res["status_429_total"] = res["build_http"]["status_429"] + tot["status_429"]
        if hashes:
            res["hashes"] = hashes
        if saved:
            import numpy as np

            arrs = {}
            for j, s in enumerate(saved):
                for k, v in s.items():
                    arrs[f"{j}/{k}"] = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
            np.savez(out_path.with_suffix(".npz"), **arrs)
            res["saved_npz"] = out_path.with_suffix(".npz").name
        finalize_mem()
        dump("capped" if res.get("capped") else "ok")
        del dl
    except Exception as exc:
        import traceback

        res["error"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-4000:]
        finalize_mem()
        dump("error")
    mem.stop = True
    os._exit(0)  # do not wait for DataLoader / decoder threads


# ====================================================================== mapcheck (base venv)
def cmd_mapcheck(a):
    import numpy as np
    import torch
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata as Meta
    from lerobot.datasets.video_utils import VideoDecoderCache, decode_video_frames_torchcodec

    meta = Meta(REPO, revision=REV)
    data = np.load(a.npz)
    idx = sorted({int(k.split("/")[0]) for k in data.files})
    cache = VideoDecoderCache()
    rows, n_eq, n_cmp = [], 0, 0
    for j in idx:
        ep = int(data[f"{j}/episode_index"])
        fi = int(data[f"{j}/frame_index"])
        ts = float(data[f"{j}/timestamp"])
        row = {"episode_index": ep, "frame_index": fi, "timestamp": ts, "ts_minus_fi_over_fps": ts - fi / meta.fps}
        for k in [x.split("/", 1)[1] for x in data.files if x.startswith(f"{j}/observation.images.")]:
            got = torch.from_numpy(data[f"{j}/{k}"])
            path = f"{meta.url_root}/{meta.get_video_file_path(ep, k)}"
            shifted = float(meta.episodes[ep][f"videos/{k}/from_timestamp"]) + ts
            ref = decode_video_frames_torchcodec(path, [shifted], 1e-4, decoder_cache=cache,
                                                 return_uint8=got.dtype == torch.uint8).squeeze(0)
            same_shape = tuple(ref.shape) == tuple(got.shape)
            eq = bool(same_shape and torch.equal(ref.to(got.dtype), got))
            maxdiff = float((ref.float() - got.float()).abs().max()) if same_shape else None
            row[k] = {"equal": eq, "max_abs_diff": maxdiff, "ref_shape": list(ref.shape), "got_shape": list(got.shape)}
            n_cmp += 1
            n_eq += eq
        rows.append(row)
    out = {"n_samples": len(idx), "n_camera_frames": n_cmp, "n_equal": n_eq, "rows": rows}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"mapcheck {a.npz}: {n_eq}/{n_cmp} camera frames bit-identical")


# ====================================================================== orchestrate (stdlib)
def arm_env(arm):
    env = dict(os.environ)
    env.pop("LEROBOT_VIDEO_DECODER_CACHE_SIZE", None)
    src = os.environ[f"SRC_{arm.upper()}"]
    env["PYTHONPATH"] = f"{src}/src"
    return env, os.environ["VENV_PR3917" if arm == "pr3917" else "VENV_MAIN"] + "/bin/python"


def regimes(arm):
    if arm == "pr3917":
        return [("w0_dt2", 0, 2, None), ("w1_dt2", 1, 2, None), ("w0_dt8", 0, 8, None), ("w1_dt8", 1, 8, None)]
    # num_workers=8 needs ~10 GB per worker on this dataset (each worker streams all 16 parquet shards) and
    # is stopped by the memory guard on a 32 GB machine; num_workers=2 is the largest count that fits
    return [("w0_cachedef", 0, None, None), ("w2_cachedef", 2, None, None), ("w8_cachedef", 8, None, None),
            ("w0_cache3", 0, None, "3"), ("w2_cache3", 2, None, "3"), ("w8_cache3", 8, None, "3")]


def _cgroup_anon_mb():
    try:
        for line in open("/sys/fs/cgroup/memory.stat"):
            if line.startswith("anon "):
                return int(line.split()[1]) // 2**20
    except Exception:
        pass
    return 0


_UPLOADER = {"p": None}


def upload_async(out, msg):
    """Push results after every run (an OOM kill of the container would lose everything else)."""
    hf, repo = os.environ.get("HF_CLI"), os.environ.get("RESULTS_REPO")
    if not hf or not repo:
        return
    prev = _UPLOADER["p"]
    if prev is not None and prev.poll() is None:
        try:
            prev.wait(timeout=300)
        except subprocess.TimeoutExpired:
            prev.kill()
    _UPLOADER["p"] = subprocess.Popen(
        [hf, "upload", repo, str(out), f"results/{out.name}", "--repo-type", "dataset", "--private",
         "--exclude", "*.npz", "--exclude", "*.tmp", "--commit-message", f"streaming-par {out.name}: {msg}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def child(arm, regime, nw, dt, cache, rep, out, n_steady, cap_s, hash_n=0, save_n=0, seed=0):
    env, py = arm_env(arm)
    if cache is not None:
        env["LEROBOT_VIDEO_DECODER_CACHE_SIZE"] = cache
    cmd = [py, str(HERE / "bench_stream.py"), "run", "--arm", arm, "--regime", regime, "--rep", str(rep),
           "--num-workers", str(nw), "--decode-threads", str(dt or 2), "--n-steady", str(n_steady),
           "--cap-s", str(cap_s), "--seed", str(seed), "--hash-n", str(hash_n), "--save-n", str(save_n),
           "--out", str(out)]
    log = open(str(out).replace(".json", ".log"), "w")
    t0 = time.time()
    p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    guard = env_int("MEM_GUARD_MB", 27000)  # kill the run before the container's OOM killer ends the job
    peak_anon, guard_hit = 0, False
    while p.poll() is None and time.time() - t0 < cap_s + 120:
        anon = _cgroup_anon_mb()
        peak_anon = max(peak_anon, anon)
        if anon > guard:
            guard_hit = True
            break
        time.sleep(0.25)
    try:
        os.killpg(p.pid, signal.SIGKILL)  # reap DataLoader workers left by os._exit
    except ProcessLookupError:
        pass
    p.wait()
    log.close()
    try:
        r = json.loads(Path(out).read_text())
    except Exception:
        r = {"arm": arm, "regime": regime, "rep": rep, "status": "no_result"}
        Path(out).write_text(json.dumps(r))
    r["wall_s"] = time.time() - t0
    r["peak_cgroup_anon_mb_seen_by_orchestrator"] = peak_anon
    if guard_hit:
        r["status"] = "mem_guard_killed"
        r["error"] = f"cgroup anon memory above {guard} MB; run killed to protect the job"
    Path(out).write_text(json.dumps(r, indent=1, default=str))
    upload_async(Path(out).parent.parent, f"{arm} {regime} rep{rep}")
    sps = r.get("sps")
    print(f"[{time.strftime('%H:%M:%S')}] {arm:7s} {regime:12s} rep{rep} {r.get('status')}: "
          f"sps={sps if sps is None else round(sps, 2)} first={_r2(r.get('first_batch_s'))} n={r.get('n_samples')} "
          f"429={r.get('status_429_total')} pss={_r2(r.get('peak_pss_gb'))} build_rss={_r2(r.get('rss_after_build_gb'))} "
          f"anon_mb={peak_anon} wall={r['wall_s']:.0f}s "
          f"{r.get('error', '')}", flush=True)
    return r


def _r2(x):
    return None if x is None else round(x, 2)


def cmd_orchestrate(a):
    out = Path(a.out)
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    arms = os.environ.get("ARMS", "base p1 p1p2 pr3917").split()
    reps = env_int("REPEATS", 3)
    n_steady = env_int("N_STEADY", 300)
    cap_s = env_int("CAP_S", 240)
    deadline = time.time() + env_int("BENCH_BUDGET_S", 9000)
    hash_n, save_n = env_int("HASH_N", 32), env_int("SAVE_N", 16)
    only = os.environ.get("REGIMES", "").split()
    plan = {"arms": arms, "repeats": reps, "n_steady": n_steady, "cap_s": cap_s, "hash_n": hash_n,
            "save_n": save_n, "pr3917_episodes": env_int("PR3917_EPISODES", 400),
            "regimes_filter": only}
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    print("plan", plan, flush=True)

    # prep + correctness run per arm, discarded for timing: fills the metadata cache, builds the #3917
    # sidecar, puts every arm's first files in the same CDN state, and hashes the first HASH_N samples
    # (num_workers=0, seed 0) with SAVE_N samples saved for the map-style check
    for arm in arms:
        rg = regimes(arm)[0]
        child(arm, "hash", rg[1], rg[2], rg[3], -1, runs / f"{arm}__hash.json", hash_n,
              env_int("WARMUP_CAP_S", 1500), hash_n, save_n)

    # map-style cross-check with the base install
    env, py = arm_env("base")
    for arm in ("pr3917", "base"):
        npz = runs / f"{arm}__hash.npz"
        if arm in arms and npz.exists():
            r = subprocess.run([py, str(HERE / "bench_stream.py"), "mapcheck", str(npz), str(out / f"mapcheck_{arm}.json")],
                               env=env, capture_output=True, text=True, timeout=1800)
            print(r.stdout[-2000:], r.stderr[-3000:], flush=True)
    for rep in range(reps):
        for arm in arms[rep % len(arms):] + arms[:rep % len(arms)]:  # rotate arm order across repeats
            for name, nw, dt, cache in regimes(arm):
                if only and name not in only:
                    continue
                if time.time() > deadline:
                    print(f"budget reached, skipping {arm} {name} rep{rep}", flush=True)
                    Path(runs / f"{arm}__{name}__r{rep}.json").write_text(
                        json.dumps({"arm": arm, "regime": name, "rep": rep, "status": "skipped_budget"}))
                    continue
                child(arm, name, nw, dt, cache, rep, runs / f"{arm}__{name}__r{rep}.json", n_steady, cap_s)

    cmd_summarize(argparse.Namespace(out=str(out)))
    upload_async(out, "summary")
    if _UPLOADER["p"] is not None:
        _UPLOADER["p"].wait()


# ====================================================================== summarize (stdlib)
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def mci(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None, 1
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m, T975.get(len(xs) - 1, 1.96) * sd / math.sqrt(len(xs)), len(xs)


def cmd_summarize(a):
    out = Path(a.out)
    runs = [json.loads(p.read_text()) for p in sorted((out / "runs").glob("*.json"))]
    cells = {}
    for r in runs:
        if r.get("rep", -1) < 0:
            continue
        cells.setdefault((r["arm"], r["regime"]), []).append(r)
    order = ["base", "p1", "p1p2", "pr3917"]
    table = []
    for (arm, regime), rs in sorted(cells.items(), key=lambda kv: (order.index(kv[0][0]) if kv[0][0] in order else 9, kv[0][1])):
        ok = [r for r in rs if r.get("sps")]
        sps = mci([r["sps"] for r in ok])
        row = {"arm": arm, "regime": regime, "n_ok": len(ok), "n_runs": len(rs),
               "statuses": [r.get("status") for r in rs],
               "sps_mean": sps[0], "sps_ci95": sps[1],
               "first_batch_s": mci([r.get("first_batch_s") for r in rs])[0],
               "build_s": mci([r.get("build_s") for r in rs])[0],
               "req_per_sample": mci([r.get("per_sample", {}).get("requests") for r in ok])[0],
               "resolve_per_sample": mci([r.get("per_sample", {}).get("resolve") for r in ok])[0],
               "cdn_per_sample": mci([r.get("per_sample", {}).get("cdn") for r in ok])[0],
               "mb_per_sample": mci([r.get("per_sample", {}).get("mb") for r in ok])[0],
               "status_429": sum(r.get("status_429_total") or 0 for r in rs),
               "peak_pss_gb": max([r.get("peak_pss_gb") or 0 for r in rs] or [0]),
               "peak_cgroup_gb": max([r.get("peak_cgroup_current_gb") or 0 for r in rs] or [0]),
               "steady_n": [r.get("steady_n") for r in rs], "errors": [r.get("error") for r in rs if r.get("error")]}
        table.append(row)

    # correctness: arms base/p1/p1p2 hash runs must match
    hashes = {}
    for arm in ("base", "p1", "p1p2", "pr3917"):
        p = out / "runs" / f"{arm}__hash.json"
        if p.exists():
            hashes[arm] = json.loads(p.read_text()).get("hashes")
    corr = {"n_hashed": {k: len(v or []) for k, v in hashes.items()}}
    if hashes.get("base"):
        for arm in ("p1", "p1p2"):
            if hashes.get(arm):
                corr[f"{arm}_vs_base_identical"] = hashes[arm] == hashes["base"]
                corr[f"{arm}_vs_base_first_mismatch"] = next(
                    (i for i, (x, y) in enumerate(zip(hashes[arm], hashes["base"])) if x != y), None)
    for arm in ("pr3917", "base"):
        p = out / f"mapcheck_{arm}.json"
        if p.exists():
            m = json.loads(p.read_text())
            corr[f"mapcheck_{arm}"] = {k: m[k] for k in ("n_samples", "n_camera_frames", "n_equal")}
            corr[f"mapcheck_{arm}"]["max_abs_diff"] = max(
                (v["max_abs_diff"] or 0) for row in m["rows"] for k, v in row.items() if isinstance(v, dict))
    prep = {r["arm"]: {k: r.get(k) for k in ("status", "build_s", "first_batch_s", "peak_pss_gb", "error",
                                             "status_429_total", "build_http")}
            for r in runs if r.get("regime") == "hash"}
    results = {"repo": REPO, "revision": REV, "plan": json.loads((out / "plan.json").read_text()) if (out / "plan.json").exists() else None,
               "table": table, "correctness": corr, "warmup": prep, "runs": runs}
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))

    def f(x, d=2):
        return "-" if x is None else f"{x:.{d}f}"

    lines = [f"# Streaming camera-parallelism bench ({out.name})", "",
             f"`{REPO}@{REV[:8]}`, StreamingLeRobotDataset over hf://, batch 8, 3 cameras, no delta_timestamps.", "",
             "| arm | regime | ok/runs | samples/s (mean +- 95% CI) | first batch s | build s | req/sample (resolve, cdn) | MB/sample | 429s | peak PSS GB | peak cgroup GB |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        ci = f" +- {f(r['sps_ci95'])}" if r["sps_ci95"] is not None else ""
        lines.append(f"| {r['arm']} | {r['regime']} | {r['n_ok']}/{r['n_runs']} | {f(r['sps_mean'])}{ci} | "
                     f"{f(r['first_batch_s'], 1)} | {f(r['build_s'], 1)} | {f(r['req_per_sample'], 3)} "
                     f"({f(r['resolve_per_sample'], 3)}, {f(r['cdn_per_sample'], 3)}) | {f(r['mb_per_sample'], 3)} | "
                     f"{r['status_429']} | {f(r['peak_pss_gb'])} | {f(r['peak_cgroup_gb'])} |")
    lines += ["", "## Correctness", "", "```", json.dumps(corr, indent=1), "```", "", "## Prep / hash runs (num_workers=0, discarded for timing)", "",
              "```", json.dumps(prep, indent=1, default=str), "```"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("orchestrate")
    o.add_argument("out")
    s = sub.add_parser("summarize")
    s.add_argument("out")
    m = sub.add_parser("mapcheck")
    m.add_argument("npz")
    m.add_argument("out")
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True)
    r.add_argument("--regime", required=True)
    r.add_argument("--rep", type=int, default=0)
    r.add_argument("--num-workers", type=int, default=0)
    r.add_argument("--decode-threads", type=int, default=2)
    r.add_argument("--buffer-size", type=int, default=16)
    r.add_argument("--batch-size", type=int, default=8)
    r.add_argument("--n-steady", type=int, default=300)
    r.add_argument("--cap-s", type=int, default=240)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--hash-n", type=int, default=0)
    r.add_argument("--save-n", type=int, default=0)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    {"orchestrate": cmd_orchestrate, "summarize": cmd_summarize, "mapcheck": cmd_mapcheck, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
