"""Round 4 stage map: map-style LeRobotDataset (the usual lerobot-train path) on the LOCAL copies, no network.

DataLoader as lerobot-train builds it at the installed sha (configs/train.py + scripts/lerobot_train.py):
batch_size 8, EpisodeAwareSampler(shuffle=True) (a global random permutation of frames), prefetch_factor 4,
persistent_workers, multiprocessing_context "spawn", return_uint8=True, no delta timestamps (one frame per camera).
num_workers in {0, 4} (4 is the lerobot-train default).
Layouts: SEP (N keys; LeRobot decodes the cameras of one sample in a ThreadPoolExecutor), STACK (1 key, cropped back
into per-camera tensors inside __getitem__, so the crop is timed), SEP1 (first camera only).
Cache regimes (LEROBOT_VIDEO_DECODER_CACHE_SIZE, read when lerobot.datasets.video_utils creates its module-level
_default_decoder_cache at import, so each arm runs in its own process and the spawned workers inherit the env):
  c100  default
  cemu  max(1, round(100 * R4_NCOPY / real video files per camera)): the slice gets the same cache-to-files ratio as
        the full dataset (same value for SEP and STACK)
Per arm and repeat: one child process; samples/s over R4_MAP_CAP_S seconds (or R4_MAP_N batches, or the end of the
epoch) after R4_MAP_WARM batches, CPU ms / sample of the process tree
(main + workers, psutil), decoder hit rate and open ms counted inside the workers. R4_MAP_REPEATS (3) repeats, arms
interleaved (order reversed on odd repeats).

Usage: python r4_map.py <slice.json> <build.json> <out_dir>
       python r4_map.py child <arm json>      (internal)
Env: R4_MAP_REPEATS (3), R4_MAP_WARM (20 batches), R4_MAP_N (1000000 batches), R4_MAP_CAP_S (30; smoke 10), R4_MAP_WORKERS ("0 4"),
     R4_MAP_BUDGET_S (3000)
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from r4_common import DEFAULT_CACHE, env, env_int, log, mean_ci, read_json, write_json


def arms(sl, bd):
    real = sl["real"]["video_files_per_camera"]
    ncopy = bd["ncopy"]
    files = max(real.values())
    cemu = max(1, round(DEFAULT_CACHE * ncopy / files))
    out = {}
    for w in [int(x) for x in env("R4_MAP_WORKERS", "0 4").split()]:
        for layout in ("sep", "stack", "sep1"):
            for cname, c in (("c100", DEFAULT_CACHE), ("cemu", cemu)):
                out[f"{layout.upper()}_w{w}_{cname}"] = {"layout": layout, "workers": w, "cache": c, "cache_regime": cname}
    return out, cemu


def child(a):
    """One arm, one repeat. Env LEROBOT_VIDEO_DECODER_CACHE_SIZE is set by the parent before this process starts."""
    import psutil
    import torch

    from common import pin_threads

    pin_threads(1)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.sampler import EpisodeAwareSampler
    from lerobot.datasets.video_utils import _default_decoder_cache
    from r4_common import MapWrap

    assert _default_decoder_cache.max_size == a["cache"], (_default_decoder_cache.max_size, a["cache"])
    t0 = time.perf_counter()
    base = LeRobotDataset(f"local/r4-{a['layout']}", root=a["root"], return_uint8=True)
    ds = MapWrap(base, a["crops"], a["order"])
    sampler = EpisodeAwareSampler(base.meta.episodes["dataset_from_index"], base.meta.episodes["dataset_to_index"],
                                  episode_indices_to_use=base.episodes, shuffle=True, seed=a["seed"],
                                  absolute_to_relative_idx=base.absolute_to_relative_idx)
    w = a["workers"]
    dl = torch.utils.data.DataLoader(ds, batch_size=a["batch_size"], sampler=sampler, num_workers=w, pin_memory=False, drop_last=False,
                                     prefetch_factor=4 if w > 0 else None, persistent_workers=w > 0,
                                     multiprocessing_context="spawn" if w > 0 else None)
    construct_s = time.perf_counter() - t0
    me = psutil.Process()

    def cpu():
        tot = sum(me.cpu_times()[:2])
        for ch in me.children(recursive=True):
            try:
                tot += sum(ch.cpu_times()[:2])
            except psutil.Error:
                pass
        return tot

    t_it = time.perf_counter()
    it = iter(dl)
    iter_s = time.perf_counter() - t_it
    tot = {"samples": 0, "hits": 0, "misses": 0, "open_s": 0.0}
    t_start = time.perf_counter()
    first_batch_s = None
    cpu_a = t_a = None
    nb = 0
    done_reason = "n"
    while True:
        try:
            b = next(it)
        except StopIteration:
            done_reason = "epoch end"
            break
        nb += 1
        if first_batch_s is None:
            first_batch_s = time.perf_counter() - t_start
        if nb == a["warm"]:
            cpu_a, t_a = cpu(), time.perf_counter()
            continue
        if t_a is None:
            continue
        n = len(b["_r4_hits"])
        tot["samples"] += n
        tot["hits"] += int(b["_r4_hits"].sum())
        tot["misses"] += int(b["_r4_misses"].sum())
        tot["open_s"] += float(b["_r4_open_s"].sum())
        if nb - a["warm"] >= a["n"]:
            break
        if time.perf_counter() - t_a >= a["cap_s"]:
            done_reason = "cap"
            break
    res = {**{k: a[k] for k in ("arm", "layout", "workers", "cache", "cache_regime", "seed", "repeat")}, "construct_s": construct_s, "iter_s": iter_s,
           "first_batch_s": first_batch_s, "batches": nb, "done": done_reason, "dataset_len": len(ds)}
    if t_a is None or tot["samples"] == 0:
        res["error"] = "stopped before the warm-up ended"
        return res
    wall = time.perf_counter() - t_a
    dc = cpu() - cpu_a
    acc = tot["hits"] + tot["misses"]
    res.update({"samples": tot["samples"], "wall_s": wall, "samples_per_s": tot["samples"] / wall, "cpu_ms_per_sample": dc / tot["samples"] * 1e3,
                "dec_hit_rate": tot["hits"] / max(1, acc), "decoder_calls_per_sample": acc / tot["samples"],
                "open_ms_per_open": tot["open_s"] / max(1, tot["misses"]) * 1e3, "totals": tot})
    t_sd = time.perf_counter()
    # stop the workers now (measurement is done), so none keeps running into the next arm; a clean
    # _shutdown_workers() or terminate() waited ~5 s per worker here, so they are killed (SIGKILL)
    import signal

    from torch.utils.data import _utils as dl_utils

    if getattr(it, "_worker_pids_set", False):  # PyTorch's SIGCHLD handler would raise on the kill
        dl_utils.signal_handling._remove_worker_pids(id(it))
        it._worker_pids_set = False
    if hasattr(signal, "SIGCHLD"):
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    for wp in getattr(it, "_workers", []):
        wp.kill()
    left = {wp.pid for wp in getattr(it, "_workers", [])}
    t_end = time.perf_counter() + 5
    while left and time.perf_counter() < t_end:  # reap (Process.join() waited 5 s per killed worker on macOS)
        for pid in list(left):
            try:
                if os.waitpid(pid, os.WNOHANG)[0] == pid:
                    left.discard(pid)
            except ChildProcessError:
                left.discard(pid)
        time.sleep(0.01)
    res["workers_not_reaped"] = len(left)
    it._shutdown = True  # the iterator's __del__ would otherwise join the killed workers again (~30 s)
    res["shutdown_s"] = time.perf_counter() - t_sd
    return res


def summarize(runs):
    by = {}
    for r in runs:
        if "error" not in r:
            by.setdefault(r["arm"], []).append(r)
    out = {}
    for k, rs in by.items():
        agg = {kk: rs[0][kk] for kk in ("layout", "workers", "cache", "cache_regime")}
        agg["repeats"] = len(rs)
        for m in ("samples_per_s", "cpu_ms_per_sample", "dec_hit_rate", "open_ms_per_open"):
            mm, h = mean_ci([r[m] for r in rs])
            agg[m], agg[f"{m}_ci95"], agg[f"{m}_per_repeat"] = mm, h, [r[m] for r in rs]
        out[k] = agg
    return out


def main():
    if sys.argv[1] == "child":
        a = json.loads(sys.argv[2])
        r = child(a)
        Path(a["result"]).write_text(json.dumps(r, default=str))
        sys.stdout.flush()
        os._exit(0)  # skip interpreter teardown (workers are already stopped)
    sl, bd = read_json(sys.argv[1]), read_json(sys.argv[2])
    out = Path(sys.argv[3])
    D = out / "map"
    D.mkdir(parents=True, exist_ok=True)
    smoke = env("SMOKE", "0") == "1"
    A, cemu = arms(sl, bd)
    names = env("R4_MAP_ARMS", " ".join(A)).split()
    cfg = {"arms": names, "cemu": cemu, "repeats": env_int("R4_MAP_REPEATS", "3"), "warm": env_int("R4_MAP_WARM", "20"),
           "n": env_int("R4_MAP_N", "1000000"), "cap_s": float(env("R4_MAP_CAP_S", "10" if smoke else "30")), "batch_size": 8,
           "budget_s": float(env("R4_MAP_BUDGET_S", "3000")),
           "dataloader": "batch_size 8, EpisodeAwareSampler(shuffle=True), prefetch_factor 4, persistent_workers, spawn, return_uint8=True",
           "real_video_files_per_camera": sl["real"]["video_files_per_camera"], "ncopy": bd["ncopy"]}
    res = {"config": cfg, "runs": [], "skipped": []}
    t_start = time.perf_counter()
    for rep in range(cfg["repeats"]):
        order = names if rep % 2 == 0 else names[::-1]
        for name in order:
            if time.perf_counter() - t_start > cfg["budget_s"]:
                res["skipped"].append({"arm": name, "rep": rep, "why": "budget"})
                continue
            spec = A[name]
            layout = spec["layout"]
            a = {**spec, "arm": name, "seed": 100 + rep, "repeat": rep, "root": bd["datasets"][layout]["root"],
                 "crops": sl["stack_layout"]["crops"] if layout == "stack" else None, "order": sl["cams"],
                 "warm": cfg["warm"], "n": cfg["n"], "cap_s": cfg["cap_s"], "batch_size": cfg["batch_size"],
                 "result": str(D / f"_run_{name}_r{rep}.json")}
            e = {**os.environ, "LEROBOT_VIDEO_DECODER_CACHE_SIZE": str(spec["cache"]), "HF_HUB_OFFLINE": "1"}
            t0 = time.perf_counter()
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "child", json.dumps(a)], env=e, capture_output=True, text=True,
                               timeout=cfg["cap_s"] + 600)
            if p.returncode == 0 and Path(a["result"]).exists():
                r = json.loads(Path(a["result"]).read_text())
            else:
                r = {"arm": name, "repeat": rep, "error": f"rc {p.returncode}: {p.stderr[-1500:]}"}
            r["process_s"] = time.perf_counter() - t0
            res["runs"].append(r)
            if "error" in r:
                log(f"{name} r{rep}: ERROR {r['error'][-600:]}")
            else:
                log(f"{name} r{rep}: {r['samples_per_s']:.1f} samples/s, CPU {r['cpu_ms_per_sample']:.1f} ms/sample, hit {r['dec_hit_rate']:.3f}, "
                    f"open {r['open_ms_per_open']:.1f} ms, {r['samples']} samples in {r['wall_s']:.0f}s ({r['done']}), process {r['process_s']:.0f}s")
            write_json(D / "map.json", res)
    res["summary"] = summarize(res["runs"])
    res["wall_s"] = time.perf_counter() - t_start
    write_json(D / "map.json", res)
    sys.exit(0 if res["summary"] else 1)


if __name__ == "__main__":
    main()
