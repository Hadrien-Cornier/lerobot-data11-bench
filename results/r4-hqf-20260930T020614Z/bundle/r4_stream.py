"""Round 4 stage stream: the real StreamingLeRobotDataset (lerobot PR #4702 head) over hf://, SEP vs STACK vs SEP1.

Same measurement as s12_stream.py (single process = one DataLoader worker; iteration, random shard choice, shuffle
buffer, VideoDecoderCache, fsspec defaults are LeRobot's own code). Arms (layout, S = max_num_shards, cache =
LEROBOT_VIDEO_DECODER_CACHE_SIZE):
  SEP_S8_c100, STACK_S8_c100, SEP1_S8_c100   default cache
  SEP_S8_c12, STACK_S8_c12                   cache 12 with S = 8 = the cache hit rate of 64 shards with cache 100
STACK arms crop the decoded stacked frame back into per-camera tensors (slices) inside the timed region.
Metrics per arm: frames made per second after warm-up, CPU ms / frame, mp4 fetches and MB / frame, decoder hit
rate, open ms, first yield. R4_STREAM_REPEATS repeats (3), arms interleaved inside each repeat (order reversed on odd
repeats), 95% Student-t CIs over repeats.

Audit (local copy, no network): StreamingLeRobotDataset in a torch DataLoader with num_workers = max_num_shards =
R4_AUDIT_WORKERS (4, the lerobot-train default) on the SEP dataset: frames, distinct data files and max decoder cache
size per worker, vs the prediction of r4_common.streaming_active_shards (datasets splits each shard's files across
workers again).

Usage: python r4_stream.py <slice.json> <build.json> <out_dir>
Env: R4_STREAM_ARMS, R4_STREAM_REPEATS (3), R4_STREAM_N (1200), R4_STREAM_WARM (200), R4_STREAM_WARM_THRASH (40),
     R4_STREAM_CAP_S (60), R4_STREAM_BUDGET_S (3000), R4_STREAM_LOCAL (0; 1 = read the local copies, laptop test),
     R4_AUDIT (1), R4_AUDIT_WORKERS (4), R4_AUDIT_N (400)
"""

import json
import os
import sys
import threading
import time

from common import pin_threads

pin_threads(1)

from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from common import REMOTE, install_http_spy, memlog, pct, proc_status  # noqa: E402
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset  # noqa: E402
from r4_common import STACK_KEY, crop_views, env, env_int, log, mean_ci, read_json, streaming_active_shards, write_json  # noqa: E402

FT = {"mp4_fetches": 0, "mp4_bytes": 0, "parquet_fetches": 0, "parquet_bytes": 0, "other_fetches": 0, "other_bytes": 0,
      "dec_hits": 0, "dec_misses": 0, "open_s": 0.0, "frames": 0, "video_s": 0.0}
_L = threading.Lock()
KEYS = ["requests", "resolve_requests", "cdn_requests", "api_requests", "http_bytes", "status_429", "status_other_err"]
ARMS = {"SEP_S8_c100": ("sep", 8, 100), "STACK_S8_c100": ("stack", 8, 100), "SEP1_S8_c100": ("sep1", 8, 100),
        "SEP_S8_c12": ("sep", 8, 12), "STACK_S8_c12": ("stack", 8, 12)}


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


class Probe(StreamingLeRobotDataset):
    """LeRobot's class; _query_videos wrapped to crop STACK, count / time made frames, and stop the arm."""

    hook = None
    crops = None  # STACK: {cam: [y0, h, w]}
    order = None
    audit = False

    def _query_videos(self, query_timestamps, ep_idx):
        t0 = time.perf_counter()
        out = StreamingLeRobotDataset._query_videos(self, query_timestamps, ep_idx)
        if self.crops is not None:
            x = out.pop(STACK_KEY)
            out.update({f"observation.images.{c}": v for c, v in crop_views(x, self.crops, self.order).items()})
        with _L:
            FT["frames"] += 1
            FT["video_s"] += time.perf_counter() - t0
        if self.audit:
            wi = torch.utils.data.get_worker_info()
            out["_r4_worker"] = torch.tensor(wi.id if wi else -1)
            out["_r4_cache"] = torch.tensor(self.video_decoder_cache.size())
        if Probe.hook is not None:
            Probe.hook()
        return out


def warm_for(layout, S, cache, ncam, warm, warm_thrash):
    need = S * (ncam if layout == "sep" else 1)
    return warm if cache >= need else warm_thrash


def make_ds(d, S, seed, local, **kw):
    if local:
        return Probe(f"local/r4-{d['kind']}", root=d["root"], max_num_shards=S, seed=seed, return_uint8=True, **kw)
    return Probe(d["repo"], revision=d["commit"], max_num_shards=S, seed=seed, return_uint8=True, **kw)


def run_arm(sl, bd, name, seed, cfg):
    layout, S, cache = ARMS[name]
    d = bd["datasets"][layout]
    os.environ["LEROBOT_VIDEO_DECODER_CACHE_SIZE"] = str(cache)
    ncam = len(sl["cams"])
    warm = warm_for(layout, S, cache, ncam, cfg["warm"], cfg["warm_thrash"])
    t_c0 = time.perf_counter()
    ds = make_ds(d, S, seed, cfg["local"])
    construct_s = time.perf_counter() - t_c0
    if layout == "stack":
        ds.crops, ds.order = sl["stack_layout"]["crops"], sl["cams"]
    st = {"t0": time.perf_counter(), "a": None, "b": None, "first_yield_s": None}
    base = snap()
    n, cap_s = cfg["n"], cfg["cap_s"]

    def hook():
        k = FT["frames"] - base["frames"]
        if k == warm:
            st["a"] = snap()
        if st["a"] is not None:
            m = FT["frames"] - st["a"]["frames"]
            el = time.perf_counter() - st["a"]["t"]
            if m >= n or el >= cap_s:
                st["b"] = snap()
                raise StopArm
        if time.perf_counter() - st["t0"] > cap_s + 600:
            st["b"] = snap()
            raise StopArm

    Probe.hook = hook
    yielded = 0
    cache_size_end = None
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
    exhausted = st["a"] is not None and st["b"] is None  # every shard read to the end before n / cap_s
    if exhausted:
        st["b"] = snap()
    a, b = st["a"], st["b"]
    res = {"arm": name, "exhausted": exhausted, "layout": layout, "shards": S, "cache": cache, "seed": seed, "warm": warm, "construct_s": construct_s,
           "num_shards_used": ds.num_shards, "hf_num_shards": ds.hf_dataset.num_shards, "first_yield_s": st["first_yield_s"],
           "yielded": yielded, "cache_size_end": cache_size_end}
    if a is None or b is None:
        res["error"] = "arm stopped before the warm-up ended (dataset exhausted?)"
        res["warmup_frames_made"] = FT["frames"] - base["frames"]
        return res
    dd = {k: b[k] - a[k] for k in a}
    m = dd["frames"]
    res.update({"frames": m, "wall_s": dd["t"], "frames_per_s": m / dd["t"] if dd["t"] > 0 else None,
                "cpu_ms_per_frame": dd["cpu_s"] / m * 1e3 if m else None, "video_ms_per_frame": dd["video_s"] / m * 1e3 if m else None,
                "dec_hit_rate": dd["dec_hits"] / max(1, dd["dec_hits"] + dd["dec_misses"]),
                "opens_per_frame": dd["dec_misses"] / max(1, m), "open_ms_per_open": dd["open_s"] / max(1, dd["dec_misses"]) * 1e3,
                "totals": dd})
    for k in ("mp4_fetches", "mp4_bytes", "parquet_fetches", "parquet_bytes", "resolve_requests", "cdn_requests", "http_bytes"):
        res[f"{k}_per_frame"] = dd[k] / max(1, m)
    return res


def summarize(runs):
    arms = {}
    for r in runs:
        if "error" not in r:
            arms.setdefault(r["arm"], []).append(r)
    out = {}
    for k, rs in arms.items():
        agg = {"arm": k, "layout": rs[0]["layout"], "shards": rs[0]["shards"], "cache": rs[0]["cache"], "repeats": len(rs)}
        for m in ("frames_per_s", "cpu_ms_per_frame", "dec_hit_rate", "open_ms_per_open", "mp4_fetches_per_frame", "mp4_bytes_per_frame",
                  "first_yield_s", "opens_per_frame"):
            mm, h = mean_ci([r[m] for r in rs])
            agg[m] = mm
            agg[f"{m}_ci95"] = h
            agg[f"{m}_per_repeat"] = [r[m] for r in rs]
        agg["status_429"] = sum(r["totals"]["status_429"] for r in rs)
        out[k] = agg
    return out


def audit(sl, bd, cfg):
    """DataLoader workers on the local SEP copy: who reads which files."""
    W = cfg["audit_workers"]
    d = bd["datasets"]["sep"]
    os.environ["LEROBOT_VIDEO_DECODER_CACHE_SIZE"] = "100"
    ds = Probe(f"local/r4-sep", root=d["root"], max_num_shards=W, seed=3, return_uint8=True, buffer_size=16)
    ds.audit = True
    epf = d["episodes_per_file"]
    dl = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=W, multiprocessing_context="spawn" if W > 0 else None)
    per = {}
    t0 = time.perf_counter()
    n = 0
    for batch in dl:
        for w, e, c in zip(batch["_r4_worker"].tolist(), batch["episode_index"].tolist(), batch["_r4_cache"].tolist()):
            p = per.setdefault(w, {"frames": 0, "files": set(), "max_cache": 0})
            p["frames"] += 1
            p["files"].add(e // epf)
            p["max_cache"] = max(p["max_cache"], c)
        n += len(batch["_r4_worker"])
        if n >= cfg["audit_n"] or time.perf_counter() - t0 > 300:
            break
    P = d["files_per_key"]
    pred = streaming_active_shards(P, W)
    return {"workers": W, "parquet_files": P, "max_num_shards": W, "samples": n, "buffer_size": 16,
            "per_worker": {str(w): {"frames": v["frames"], "distinct_files": sorted(v["files"]), "max_decoder_cache_size": v["max_cache"]}
                           for w, v in sorted(per.items())},
            "workers_with_frames": len(per), "predicted_active_shards_per_worker": pred,
            "predicted_decoders_per_worker": [a * len(sl["cams"]) for a in pred], "s": round(time.perf_counter() - t0, 1)}


def main():
    sl, bd = read_json(sys.argv[1]), read_json(sys.argv[2])
    out = Path(sys.argv[3])
    D = out / "stream"
    D.mkdir(parents=True, exist_ok=True)
    for kind, d in bd["datasets"].items():
        d["kind"] = kind
    smoke = env("SMOKE", "0") == "1"
    cfg = {"arms": env("R4_STREAM_ARMS", " ".join(ARMS)).split(), "repeats": env_int("R4_STREAM_REPEATS", "3"),
           "n": env_int("R4_STREAM_N", "1200"), "warm": env_int("R4_STREAM_WARM", "60" if smoke else "200"),
           "warm_thrash": env_int("R4_STREAM_WARM_THRASH", "40"), "cap_s": float(env("R4_STREAM_CAP_S", "20" if smoke else "60")),
           "budget_s": float(env("R4_STREAM_BUDGET_S", "3000")), "local": env("R4_STREAM_LOCAL", "0") == "1",
           "audit": env("R4_AUDIT", "1") == "1", "audit_workers": env_int("R4_AUDIT_WORKERS", "4"), "audit_n": env_int("R4_AUDIT_N", "400"),
           "lerobot_sha": env("LEROBOT_SHA", "?"), "repos": {k: (d["repo"], d.get("commit")) for k, d in bd["datasets"].items()}}
    import lerobot

    cfg["lerobot_file"] = lerobot.__file__
    res = {"config": cfg, "runs": [], "skipped": []}
    if cfg["audit"]:
        try:
            res["audit"] = audit(sl, bd, cfg)
            log(f"audit: {json.dumps(res['audit'])}")
        except Exception as exc:  # noqa: BLE001
            res["audit"] = {"error": f"{type(exc).__name__}: {exc}"}
            log(f"audit ERROR {res['audit']}")
        write_json(D / "stream.json", res)
    instrument()
    t_start = time.perf_counter()
    for rep in range(cfg["repeats"]):
        order = cfg["arms"] if rep % 2 == 0 else cfg["arms"][::-1]
        for name in order:
            if time.perf_counter() - t_start > cfg["budget_s"]:
                res["skipped"].append({"arm": name, "rep": rep, "why": "budget"})
                continue
            try:
                r = run_arm(sl, bd, name, 1000 + rep, cfg)
            except Exception as exc:  # noqa: BLE001
                r = {"arm": name, "seed": 1000 + rep, "error": f"{type(exc).__name__}: {exc}"}
            r["repeat"] = rep
            res["runs"].append(r)
            if "error" in r:
                log(f"{name} r{rep}: ERROR {r['error']}")
            else:
                log(f"{name} r{rep}: {r['frames_per_s']:.1f} fps, CPU {r['cpu_ms_per_frame']:.1f} ms/frame, mp4 fetch/frame "
                    f"{r['mp4_fetches_per_frame']:.4f} ({r['mp4_bytes_per_frame'] / 1e6:.4f} MB), hit {r['dec_hit_rate']:.3f}, open "
                    f"{r['open_ms_per_open']:.0f} ms, shards {r['num_shards_used']}/{r['hf_num_shards']}, first yield {r['first_yield_s']}")
            write_json(D / "stream.json", res)
        memlog(out, f"stream_rep{rep}")
    res["summary"] = summarize(res["runs"])
    res["wall_s"] = time.perf_counter() - t_start
    res["rss"] = proc_status()
    res["remote_totals"] = dict(REMOTE)
    write_json(D / "stream.json", res)
    sys.exit(0 if res["summary"] else 1)


if __name__ == "__main__":
    main()
