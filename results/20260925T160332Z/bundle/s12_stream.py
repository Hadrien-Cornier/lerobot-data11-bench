"""Stage stream (round 2 HEADLINE): the real StreamingLeRobotDataset (lerobot PR #4702 branch) over hf://.

Datasets from stage prep2: SEP (3 video keys, one MP4 per camera) and STACK (1 video key, the 224x672 stacked
frame; after decode the frame is split into 3 views with a slice, which costs nothing measurable). N video files per
key, one parquet data file per video file. Nothing is emulated: iteration, shard choice (random shard, NEXT frame of
that shard), 1000-sample shuffle buffer, VideoDecoderCache (LRU, default 100 entries), fsspec.open defaults
(5 MiB readahead) are LeRobot's own code.

Arms (single process, as a DataLoader worker would run it):
  shards  S = max_num_shards in {1, 8}: number of parquet shards the iterator interleaves (each one its own file)
  cache   LEROBOT_VIDEO_DECODER_CACHE_SIZE in {100 (default), 12}. 12 with S = 8 reproduces the decoder-cache
          pressure of S = 64 with the default 100: SEP needs 3 S decoders (24 vs 12 -> hit ~ 12/24; 192 vs 100 ->
          hit ~ 100/192), STACK needs S (8 <= 12, 64 <= 100: no misses). Under uniform random shard choice the LRU hit
          rate is cache / needed, so the two cases match to ~2 points.
  cams    seq = LeRobot as shipped (plain loop, _query_videos) | par = ThreadPoolExecutor over video keys, the
          pattern of dataset_reader.py:419 (SEP only)
Per arm: made frames/s after a warm-up (every loop step makes one frame, whether it is yielded or buffered),
fetches / MB per frame for .mp4 and .parquet separately, resolve / CDN requests, decoder-cache hit rate and open time,
CPU seconds per frame, time to the first yielded sample (buffer fill). Repeats with different seeds.
Concurrency: STREAM_CONC processes (default = CPU quota), each its own dataset + seed, measured together.

Usage: python s12_stream.py <prep.json> <out_dir>
Env: STREAM_ARMS (all), STREAM_REPEATS (2), STREAM_N (1200), STREAM_WARM (200), STREAM_WARM_THRASH (40), STREAM_CAP_S (90),
     STREAM_CONC, STREAM_CONC_S (60), STREAM_BUDGET_S (2400), STREAM_MB_CAP (stop when HTTP MB exceed it; laptop)
"""

import json
import os
import sys
import threading
import time

from common import pin_threads

pin_threads(1)

from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

from common import REMOTE, ci_halfwidth, cpu_quota, install_http_spy, memlog, pct, proc_status, write_json  # noqa: E402

FT = {"mp4_fetches": 0, "mp4_bytes": 0, "parquet_fetches": 0, "parquet_bytes": 0, "other_fetches": 0, "other_bytes": 0,
      "dec_hits": 0, "dec_misses": 0, "open_s": 0.0, "frames": 0, "video_s": 0.0}
_L = threading.Lock()
KEYS = ["requests", "resolve_requests", "cdn_requests", "api_requests", "http_bytes", "status_429", "status_other_err"]

# name: (layout, S, cache, par)
ARMS = {"SEP_S1_c100_seq": ("SEP", 1, 100, False), "SEP_S1_c100_par": ("SEP", 1, 100, True),
        "SEP_S8_c100_seq": ("SEP", 8, 100, False), "SEP_S8_c100_par": ("SEP", 8, 100, True),
        "SEP_S8_c12_seq": ("SEP", 8, 12, False), "SEP_S8_c12_par": ("SEP", 8, 12, True),
        "STACK_S1_c100": ("STACK", 1, 100, False), "STACK_S8_c100": ("STACK", 8, 100, False), "STACK_S8_c12": ("STACK", 8, 12, False)}
CONC_ARMS = ["SEP_S8_c100_seq", "SEP_S8_c100_par", "STACK_S8_c100", "SEP_S8_c12_seq", "STACK_S8_c12"]


class StopArm(Exception):
    pass


def instrument():
    install_http_spy()
    from huggingface_hub import hf_file_system

    from lerobot.datasets import video_utils

    orig = hf_file_system.HfFileSystemFile._fetch_range

    def fetch(self, start, end):
        b = orig(self, start, end)
        p = str(getattr(self, "path", ""))
        k = "mp4" if p.endswith(".mp4") else ("parquet" if p.endswith(".parquet") else "other")
        with _L:
            FT[f"{k}_fetches"] += 1
            FT[f"{k}_bytes"] += len(b)
        return b

    hf_file_system.HfFileSystemFile._fetch_range = fetch
    orig_get = video_utils.VideoDecoderCache.get_decoder

    def get_decoder(self, video_path):
        hit = str(video_path) in self
        t0 = time.perf_counter()
        d = orig_get(self, video_path)
        with _L:
            if hit:
                FT["dec_hits"] += 1
            else:
                FT["dec_misses"] += 1
                FT["open_s"] += time.perf_counter() - t0
        return d

    video_utils.VideoDecoderCache.get_decoder = get_decoder


def snap():
    s = {k: REMOTE[k] for k in KEYS}
    s.update(FT)
    s["cpu_s"] = sum(os.times()[:2])
    s["t"] = time.perf_counter()
    return s


def classes():
    from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
    from lerobot.datasets.video_utils import decode_video_frames_torchcodec

    class Probe(StreamingLeRobotDataset):
        """LeRobot's class; _query_videos wrapped only to count / time made frames and to stop the arm."""

        hook = None
        split_stack = False

        def _query_videos(self, query_timestamps, ep_idx):
            t0 = time.perf_counter()
            out = self._query_videos_impl(query_timestamps, ep_idx)
            if self.split_stack:  # STACK: 3 camera views of the decoded stacked frame (slices, no copy)
                x = out["observation.images.stack"]
                h = x.shape[-2] // 3
                out = {**out, "_views": [x[..., i * h : (i + 1) * h, :] for i in range(3)]}
            with _L:
                FT["frames"] += 1
                FT["video_s"] += time.perf_counter() - t0
            if Probe.hook is not None:
                Probe.hook()
            return out

        def _query_videos_impl(self, query_timestamps, ep_idx):
            return StreamingLeRobotDataset._query_videos(self, query_timestamps, ep_idx)

    class ProbePar(Probe):
        """Cameras decoded in parallel threads, as dataset_reader.py:419 does for the map-style reader."""

        def _query_videos_impl(self, query_timestamps, ep_idx):
            root = self.meta.url_root if self.streaming and not self.streaming_from_local else self.root

            def one(kv):
                k, ts = kv
                fr = decode_video_frames_torchcodec(f"{root}/{self.meta.get_video_file_path(ep_idx, k)}", ts, self.tolerance_s,
                                                    decoder_cache=self.video_decoder_cache, return_uint8=self._return_uint8)
                return k, fr.squeeze(0) if len(ts) == 1 else fr

            items = list(query_timestamps.items())
            if len(items) <= 1:
                return dict(map(one, items))
            with ThreadPoolExecutor(max_workers=len(items)) as pool:
                return dict(pool.map(one, items))

    return Probe, ProbePar


def warm_for(name, warm, warm_thrash):
    layout, S, cache, _ = ARMS[name]
    need = S * (3 if layout == "SEP" else 1)
    return warm if cache >= need else warm_thrash  # thrash arms: the cache is full after a few shards


def run_arm(prep, name, seed, n, warm, cap_s, mb_cap=None, start_at=None, dur_s=None):
    layout, S, cache, par = ARMS[name]
    ds_info = prep["stream"][layout]
    os.environ["LEROBOT_VIDEO_DECODER_CACHE_SIZE"] = str(cache)
    Probe, ProbePar = classes()
    cls = ProbePar if par else Probe
    t_c0 = time.perf_counter()
    ds = cls(ds_info["repo"], revision=ds_info["commit"], max_num_shards=S, seed=seed, return_uint8=True)
    ds.split_stack = layout == "STACK"
    construct_s = time.perf_counter() - t_c0
    st = {"t0": None, "a": None, "b": None, "first_yield_s": None, "frame_times": []}
    base = snap()

    def hook():
        k = FT["frames"] - base["frames"]
        now = time.perf_counter()
        if k == warm:
            if start_at is not None:  # concurrency: all processes start measuring together
                while time.time() < start_at:
                    time.sleep(0.005)
            st["a"] = snap()
        if st["a"] is not None:
            st["frame_times"].append(time.perf_counter())
            m = FT["frames"] - st["a"]["frames"]
            el = time.perf_counter() - st["a"]["t"]
            if (dur_s is not None and el >= dur_s) or (dur_s is None and (m >= n or el >= cap_s)):
                st["b"] = snap()
                raise StopArm
        if mb_cap is not None and REMOTE["http_bytes"] / 1e6 > mb_cap:
            st["b"] = snap()
            st["mb_cap_hit"] = True
            raise StopArm
        if dur_s is None and now - st["t0"] > cap_s + 600:
            st["b"] = snap()
            raise StopArm

    Probe.hook = hook
    st["t0"] = time.perf_counter()
    yielded = 0
    try:
        for _item in ds:
            yielded += 1
            if st["first_yield_s"] is None:
                st["first_yield_s"] = time.perf_counter() - st["t0"]
    except StopArm:
        pass
    finally:
        Probe.hook = None
        if ds.video_decoder_cache is not None:
            cache_size_end = ds.video_decoder_cache.size()
            ds.video_decoder_cache.clear()
        else:
            cache_size_end = None
    a, b = st["a"], st["b"]
    res = {"arm": name, "layout": layout, "shards": S, "cache": cache, "cams": "par" if par else "seq", "seed": seed,
           "construct_s": construct_s, "num_shards_used": ds.num_shards, "hf_num_shards": ds.hf_dataset.num_shards,
           "first_yield_s": st["first_yield_s"], "yielded": yielded, "cache_size_end": cache_size_end,
           "mb_cap_hit": st.get("mb_cap_hit", False)}
    if a is None or b is None:
        res["error"] = "arm stopped before the warm-up ended"
        res["warmup_frames_made"] = FT["frames"] - base["frames"]
        return res
    d = {k: b[k] - a[k] for k in a}
    m = d["frames"]
    res.update({"frames": m, "wall_s": d["t"], "frames_per_s": m / d["t"] if d["t"] > 0 else None,
                "cpu_s_per_frame": d["cpu_s"] / m if m else None, "video_ms_per_frame": d["video_s"] / m * 1e3 if m else None,
                "dec_hit_rate": d["dec_hits"] / max(1, d["dec_hits"] + d["dec_misses"]),
                "opens_per_frame": d["dec_misses"] / max(1, m), "open_ms_per_open": d["open_s"] / max(1, d["dec_misses"]) * 1e3,
                "totals": d, "warmup": {k: a[k] - base[k] for k in ("frames", "mp4_fetches", "mp4_bytes", "dec_misses", "http_bytes")}})
    for k in ("mp4_fetches", "mp4_bytes", "parquet_fetches", "parquet_bytes", "resolve_requests", "cdn_requests", "api_requests", "http_bytes"):
        res[f"{k}_per_frame"] = d[k] / max(1, m)
    gaps = [y - x for x, y in zip(st["frame_times"], st["frame_times"][1:])]
    res["frame_gap_p50_ms"] = pct(gaps, 0.5) * 1e3 if gaps else None
    res["frame_gap_p95_ms"] = pct(gaps, 0.95) * 1e3 if gaps else None
    return res


def conc_worker(a):
    import random

    time.sleep(random.random() * 0.2)
    instrument()
    prep = json.loads(Path(a["prep"]).read_text())
    return run_arm(prep, a["arm"], a["seed"], 10**9, a["warm"], 10**9, start_at=a["start_at"], dur_s=a["dur_s"])


def summarize(runs):
    arms = {}
    for r in runs:
        if "error" not in r:
            arms.setdefault(r["arm"], []).append(r)
    out = {}
    for k, rs in arms.items():
        fps = [r["frames_per_s"] for r in rs]
        agg = {"arm": k, "layout": rs[0]["layout"], "shards": rs[0]["shards"], "cache": rs[0]["cache"], "cams": rs[0]["cams"],
               "repeats": len(rs), "frames": sum(r["frames"] for r in rs), "fps_mean": sum(fps) / len(fps), "fps_per_repeat": fps,
               "fps_ci95": ci_halfwidth(fps) if len(fps) > 1 else None,
               "first_yield_s": [r["first_yield_s"] for r in rs]}
        tot = {kk: sum(r["totals"][kk] for r in rs) for kk in rs[0]["totals"]}
        m = tot["frames"]
        for kk in ("mp4_fetches", "mp4_bytes", "parquet_fetches", "parquet_bytes", "resolve_requests", "cdn_requests", "api_requests", "http_bytes"):
            agg[f"{kk}_per_frame"] = tot[kk] / max(1, m)
        agg["dec_hit_rate"] = tot["dec_hits"] / max(1, tot["dec_hits"] + tot["dec_misses"])
        agg["opens_per_frame"] = tot["dec_misses"] / max(1, m)
        agg["open_ms_per_open"] = tot["open_s"] / max(1, tot["dec_misses"]) * 1e3
        agg["cpu_ms_per_frame"] = tot["cpu_s"] / max(1, m) * 1e3
        agg["video_ms_per_frame"] = tot["video_s"] / max(1, m) * 1e3
        agg["status_429"] = tot["status_429"]
        out[k] = agg
    return out


def write_summary(D, res):
    L = ["# DATA-11 round 2: real StreamingLeRobotDataset over hf:// (PR #4702 branch)", "",
         f"lerobot {res['config'].get('lerobot_ref')}; datasets SEP `{res['config']['sep_repo']}`, STACK `{res['config']['stack_repo']}` "
         f"({res['config']['files_per_key']} video files per key, copies of the round-1 ckpt_full variants). Single process = one "
         "DataLoader worker. fps = frames made per second after warm-up (every step makes one frame).", "",
         "| arm | S | cache | cams | fps | +-CI | mp4 fetch/frame | mp4 MB/frame | parquet fetch/frame | resolve/frame | decoder hit | opens/frame | open ms | CPU ms/frame | first yield s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, a in res["summary"].items():
        ci = f"{a['fps_ci95']:.1f}" if a["fps_ci95"] is not None else "-"
        fy = ", ".join(f"{x:.0f}" if x is not None else "-" for x in a["first_yield_s"])
        L.append(f"| {k} | {a['shards']} | {a['cache']} | {a['cams']} | {a['fps_mean']:.1f} | {ci} | {a['mp4_fetches_per_frame']:.4f} | "
                 f"{a['mp4_bytes_per_frame'] / 1e6:.4f} | {a['parquet_fetches_per_frame']:.4f} | {a['resolve_requests_per_frame']:.4f} | "
                 f"{a['dec_hit_rate']:.3f} | {a['opens_per_frame']:.3f} | {a['open_ms_per_open']:.0f} | {a['cpu_ms_per_frame']:.1f} | {fy} |")
    if res["conc"]:
        L += ["", f"## {res['config']['conc']} concurrent processes (own dataset + seed each)", "",
              "| arm | procs | fps total | resolves/s | CDN req/s | MB/s | decoder hit | 429 |", "|---|---|---|---|---|---|---|---|"]
        for c in res["conc"]:
            if "error" in c:
                L.append(f"| {c['arm']} | error: {c['error']} |")
                continue
            L.append(f"| {c['arm']} | {c['procs']} | {c['fps_total']:.1f} | {c['resolves_per_s']:.1f} | {c['cdn_per_s']:.1f} | "
                     f"{c['mb_per_s']:.1f} | {c['dec_hit_rate']:.3f} | {c['status_429']} |")
    (D / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def main():
    prep_path, out_dir = sys.argv[1], Path(sys.argv[2])
    D = out_dir / "stream"
    D.mkdir(parents=True, exist_ok=True)
    prep = json.loads(Path(prep_path).read_text())
    instrument()
    import lerobot

    names = os.environ.get("STREAM_ARMS", " ".join(ARMS)).split()
    cfg = {"arms": names, "repeats": int(os.environ.get("STREAM_REPEATS", "2")), "n": int(os.environ.get("STREAM_N", "1200")),
           "warm": int(os.environ.get("STREAM_WARM", "200")),
           "warm_thrash": int(os.environ.get("STREAM_WARM_THRASH", "40")), "cap_s": float(os.environ.get("STREAM_CAP_S", "90")),
           "conc": int(os.environ.get("STREAM_CONC", str(cpu_quota()))), "conc_s": float(os.environ.get("STREAM_CONC_S", "60")),
           "conc_arms": os.environ.get("STREAM_CONC_ARMS", " ".join(CONC_ARMS)).split(),
           "budget_s": float(os.environ.get("STREAM_BUDGET_S", "2400")), "mb_cap": float(os.environ["STREAM_MB_CAP"]) if os.environ.get("STREAM_MB_CAP") else None,
           "lerobot_file": lerobot.__file__, "lerobot_ref": os.environ.get("LEROBOT_REF", "?"),
           "sep_repo": prep["stream"]["SEP"]["repo"], "stack_repo": prep["stream"]["STACK"]["repo"],
           "files_per_key": prep["stream"]["SEP"]["files_per_key"]}
    res = {"config": cfg, "runs": [], "conc": [], "skipped": []}
    t_start = time.perf_counter()
    for rep in range(cfg["repeats"]):
        order = names if rep % 2 == 0 else names[::-1]
        for name in order:
            if time.perf_counter() - t_start > cfg["budget_s"]:
                res["skipped"].append({"arm": name, "rep": rep, "why": "budget"})
                continue
            try:
                r = run_arm(prep, name, 1000 + rep, cfg["n"], warm_for(name, cfg["warm"], cfg["warm_thrash"]), cfg["cap_s"], mb_cap=cfg["mb_cap"])
            except Exception as exc:  # noqa: BLE001
                r = {"arm": name, "seed": 1000 + rep, "error": f"{type(exc).__name__}: {exc}"}
            r["repeat"] = rep
            res["runs"].append(r)
            if "error" in r:
                print(f"[{time.perf_counter() - t_start:.0f}s] {name} r{rep}: ERROR {r['error']}", flush=True)
            else:
                print(f"[{time.perf_counter() - t_start:.0f}s] {name} r{rep}: {r['frames_per_s']:.1f} fps, mp4 fetch/frame {r['mp4_fetches_per_frame']:.4f} "
                      f"({r['mp4_bytes_per_frame'] / 1e6:.3f} MB), parquet fetch/frame {r['parquet_fetches_per_frame']:.4f}, hit {r['dec_hit_rate']:.3f}, "
                      f"opens/frame {r['opens_per_frame']:.3f} ({r['open_ms_per_open']:.0f} ms), CPU {r['cpu_s_per_frame'] * 1e3:.1f} ms/frame, "
                      f"first yield {r['first_yield_s']}, shards {r['num_shards_used']}/{r['hf_num_shards']}, HTTP MB so far {REMOTE['http_bytes'] / 1e6:.0f}", flush=True)
            write_json(D / "stream.json", res)
            if cfg["mb_cap"] is not None and REMOTE["http_bytes"] / 1e6 > cfg["mb_cap"]:
                res["skipped"].append({"why": f"MB cap {cfg['mb_cap']} reached"})
                break
        memlog(out_dir, f"stream_rep{rep}")
    if cfg["conc"] > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        for name in cfg["conc_arms"]:
            if time.perf_counter() - t_start > cfg["budget_s"]:
                res["skipped"].append({"conc": name, "why": "budget"})
                continue
            start_at = time.time() + 75
            args = [{"prep": prep_path, "arm": name, "seed": 5000 + w, "warm": warm_for(name, cfg["warm"], cfg["warm_thrash"]), "start_at": start_at,
                     "dur_s": cfg["conc_s"]}
                    for w in range(cfg["conc"])]
            try:
                with ctx.Pool(cfg["conc"]) as pool:
                    outs = pool.map(conc_worker, args)
            except Exception as exc:  # noqa: BLE001
                res["conc"].append({"arm": name, "error": f"{type(exc).__name__}: {exc}"})
                continue
            good = [o for o in outs if "error" not in o]
            if not good:
                res["conc"].append({"arm": name, "error": "all workers failed: " + str([o.get("error") for o in outs])[:300]})
                continue
            wall = max(o["wall_s"] for o in good)
            tot = {k: sum(o["totals"][k] for o in good) for k in good[0]["totals"]}
            c = {"arm": name, "procs": len(good), "failed": len(outs) - len(good), "wall_s": wall, "fps_total": tot["frames"] / wall,
                 "resolves_per_s": tot["resolve_requests"] / wall, "cdn_per_s": tot["cdn_requests"] / wall, "mb_per_s": tot["http_bytes"] / wall / 1e6,
                 "dec_hit_rate": tot["dec_hits"] / max(1, tot["dec_hits"] + tot["dec_misses"]), "status_429": tot["status_429"],
                 "cpu_ms_per_frame": tot["cpu_s"] / max(1, tot["frames"]) * 1e3, "per_proc_fps": [o["frames_per_s"] for o in good], "totals": tot}
            res["conc"].append(c)
            print(f"[{time.perf_counter() - t_start:.0f}s] conc {name} x{len(good)}: {c['fps_total']:.1f} fps, {c['resolves_per_s']:.1f} resolves/s, "
                  f"{c['mb_per_s']:.1f} MB/s, hit {c['dec_hit_rate']:.3f}, 429 {c['status_429']}", flush=True)
            write_json(D / "stream.json", res)
    res["summary"] = summarize(res["runs"])
    res["wall_s"] = time.perf_counter() - t_start
    res["rss"] = proc_status()
    res["remote_totals"] = dict(REMOTE)
    write_json(D / "stream.json", res)
    write_summary(D, res)
    sys.exit(0 if res["summary"] else 1)


if __name__ == "__main__":
    main()
