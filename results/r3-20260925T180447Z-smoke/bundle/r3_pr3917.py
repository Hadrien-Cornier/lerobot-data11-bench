"""Round 3: PR #3917 (episode-pool streaming, exact byte range per (episode, camera)) on SEP vs STACK layouts.

Datasets (r3_prep.py): hadriencornier/lerobot-data11-r3-sep (3 video keys, SEP group files) and -r3-stack (1 video key,
equal-size STACK cuts); both have the same episodes. #3917 raises with more than one DataLoader worker per rank, so
"8 workers" here = 8 processes as ranks 0..7 of WORLD_SIZE 8 (its only way to use 8 readers); "1 worker" = 1 process.
Every run iterates the dataset directly (no DataLoader), in a fresh process.
Per run: samples/s after the first sample (one sample = one time step with all 3 camera images), time to first sample,
range GETs, HTTP requests and MB per episode fetched, peak PSS of the process and cgroup peak, CPU decode time
(thread CPU inside EpisodeByteCache._get_frames + _open_decoder), process CPU per sample.
Checks (return_uint8=True): decoded frames == local TorchCodec decode of the same source frame, for SEP / STACK over the
Hub, and for a LOCAL dataset whose 3 video keys point at the multi-track (MULTI) cuts.

Usage: python r3_pr3917.py orchestrate <prep.json> <out_dir>        (any python; stdlib only)
       python r3_pr3917.py run --kind SEP|STACK|MULTI ... --out X.json    (inside the #3917 venv)
Env: VENV_PR3917, P_REPEATS (3), P_N1 (6000), P_N8 (1500 per rank), P_CAP_S (120), P_CHECK_N (40), P_MODES ("p1 p8"),
     P_KINDS ("SEP STACK"), HF_LEROBOT_HOME
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CAMS = ["top", "left_wrist", "right_wrist"]


# ====================================================================== run (inside the #3917 venv)
class Mem(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak_pss = self.peak_cg = self.last = 0
        self.stop = False

    def run(self):
        import psutil

        me = psutil.Process()
        while not self.stop:
            try:
                mi = me.memory_full_info()
                self.last = getattr(mi, "pss", mi.rss)
                self.peak_pss = max(self.peak_pss, self.last)
            except Exception:  # noqa: BLE001
                pass
            try:
                self.peak_cg = max(self.peak_cg, int(Path("/sys/fs/cgroup/memory.current").read_text()))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)


ST = {"fetch_calls": 0, "fetch_bytes": 0, "fetch_s": 0.0, "episodes": set(), "dec_calls": 0, "dec_wall_s": 0.0, "dec_cpu_s": 0.0,
      "open_calls": 0, "open_wall_s": 0.0, "open_cpu_s": 0.0}
LOCK = threading.Lock()


def patch_pr3917():
    from lerobot.streaming import episode_cache as ec

    C = ec.EpisodeByteCache
    of, oo, og = C._fetch_and_synthesize, C._open_decoder, C._get_frames

    def fetch(self, episode_index, camera_key):
        t0 = time.perf_counter()
        e = of(self, episode_index, camera_key)
        span = self.manifest.lookup(episode_index, camera_key)
        with LOCK:
            ST["fetch_calls"] += 1
            ST["fetch_bytes"] += int(span.mdat_length)
            ST["fetch_s"] += time.perf_counter() - t0
            ST["episodes"].add(int(episode_index))
        return e

    def opend(self, key, data):
        t0, c0 = time.perf_counter(), time.thread_time()
        d = oo(self, key, data)
        with LOCK:
            ST["open_calls"] += 1
            ST["open_wall_s"] += time.perf_counter() - t0
            ST["open_cpu_s"] += time.thread_time() - c0
        return d

    def frames(self, episode_index, camera_key, timestamps):
        t0, c0 = time.perf_counter(), time.thread_time()
        x = og(self, episode_index, camera_key, timestamps)
        with LOCK:
            ST["dec_calls"] += 1
            ST["dec_wall_s"] += time.perf_counter() - t0
            ST["dec_cpu_s"] += time.thread_time() - c0
        return x

    C._fetch_and_synthesize, C._open_decoder, C._get_frames = fetch, opend, frames


def snap():
    import common

    s = {k: common.REMOTE[k] for k in ("requests", "resolve_requests", "cdn_requests", "api_requests", "http_bytes", "status_429",
                                        "status_other_err")}
    with LOCK:
        s.update({k: (len(v) if isinstance(v, set) else v) for k, v in ST.items()})
    ru = __import__("resource").getrusage(__import__("resource").RUSAGE_SELF)
    s["proc_cpu_s"] = ru.ru_utime + ru.ru_stime
    return s


def cmd_run(a):
    import common

    res = {"kind": a.kind, "mode": a.mode, "rep": a.rep, "rank": a.rank, "world": a.world, "seed": a.seed, "status": "started",
           "utc_start": time.strftime("%FT%TZ", time.gmtime())}
    out = Path(a.out)

    def dump(status):
        res["status"] = status
        out.with_suffix(".tmp").write_text(json.dumps(res, indent=1, default=str))
        out.with_suffix(".tmp").replace(out)

    if a.world > 1:
        os.environ["RANK"], os.environ["WORLD_SIZE"] = str(a.rank), str(a.world)
    common.pin_threads(a.threads)
    import torch  # noqa: F401

    common.pin_threads(a.threads)
    common.install_http_spy()
    patch_pr3917()
    mem = Mem()
    mem.start()
    t_start = time.perf_counter()
    try:
        from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

        kw = {"seed": a.seed, "return_uint8": a.check_n > 0}
        if a.root:
            ds = StreamingLeRobotDataset(a.repo, root=a.root, **kw)
        else:
            ds = StreamingLeRobotDataset(a.repo, revision=a.revision, **kw)
        res["build_s"] = time.perf_counter() - t_start
        res["build"] = snap()
        res["pool"] = {k: getattr(ds, k, None) for k in ("episode_pool_size", "prefetch_episodes", "decode_threads", "max_num_shards")}
        s0 = snap()
        n, t_first, s_first = 0, None, None
        chk = []
        ep_map = json.loads(Path(a.ep_map).read_text()) if a.ep_map else None
        t_iter = time.perf_counter()
        for item in ds:
            now = time.perf_counter()
            if t_first is None:
                t_first, s_first = now, snap()
                res["first_sample_s"] = now - t_iter
                dump("running")
            else:
                n += 1
            if a.check_n and len(chk) < a.check_n:
                keys = sorted(k for k in item if k.startswith("observation.images.") and not k.endswith("_is_pad"))
                chk.append({"episode_index": int(item["episode_index"]), "frame_index": int(item["frame_index"]),
                            "frames": {k: item[k].clone() for k in keys}})
                if len(chk) >= a.check_n:
                    break
            elif a.check_n:
                break
            if n >= a.n or now - t_start > a.cap_s:
                res["capped"] = n < a.n
                break
        t_end = time.perf_counter()
        s1 = snap()
        steady_s = (t_end - t_first) if t_first else None
        d = {k: s1[k] - s_first[k] for k in s1} if s_first else {}
        tot = {k: s1[k] - s0[k] for k in s1}
        eps = max(1, tot["episodes"])
        res.update(n=n, steady_s=steady_s, sps=(n / steady_s) if steady_s and n else None, total=tot, steady=d,
                   per_episode={"episodes_fetched": tot["episodes"], "range_gets": tot["fetch_calls"] / eps, "mb": tot["fetch_bytes"] / eps / 1e6,
                                "http_requests": tot["requests"] / eps, "cdn_requests": tot["cdn_requests"] / eps,
                                "http_mb": tot["http_bytes"] / eps / 1e6, "fetch_s_mean": tot["fetch_s"] / max(1, tot["fetch_calls"])},
                   per_sample=({"dec_cpu_ms": d["dec_cpu_s"] / n * 1e3, "dec_wall_ms": d["dec_wall_s"] / n * 1e3, "dec_calls": d["dec_calls"] / n,
                                "open_cpu_ms": d["open_cpu_s"] / n * 1e3, "proc_cpu_ms": d["proc_cpu_s"] / n * 1e3,
                                "mb": d["http_bytes"] / n / 1e6} if n else None),
                   decode_totals={"dec_cpu_s": tot["dec_cpu_s"], "dec_wall_s": tot["dec_wall_s"], "open_cpu_s": tot["open_cpu_s"],
                                  "open_calls": tot["open_calls"], "dec_calls": tot["dec_calls"]})
        if chk:
            res["check"] = check_frames(chk, ep_map, a)
        res["peak_pss_gb"] = mem.peak_pss / 2**30
        res["peak_cgroup_gb"] = mem.peak_cg / 2**30
        dump("capped" if res.get("capped") else "ok")
    except Exception as exc:  # noqa: BLE001
        import traceback

        res["error"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-4000:]
        res["peak_pss_gb"] = mem.peak_pss / 2**30
        dump("error")
    mem.stop = True
    os._exit(0)


def check_frames(chk, ep_map, a):
    """Compare each sample's camera images with a local TorchCodec decode of the source frame."""
    import torch
    from torchcodec.decoders import VideoDecoder

    loc = json.loads(Path(a.local_files).read_text())  # {"sep": {g: {cam: path}}}
    decs = {}

    def dec(p, f):
        if p not in decs:
            decs[p] = VideoDecoder(p, seek_mode="exact")
        return decs[p].get_frames_at(indices=[f]).data[0]

    def ref(g, cam, f):
        return dec(loc["sep"][str(g)][cam], f)

    def ref_stack(g, f):  # the STACK cut holding group frame f (STACK is its own encode: compare with its own decode)
        p = next(i for i, (a0, b0) in enumerate(loc["cuts"][str(g)]) if a0 <= f < b0)
        return dec(loc["stack"][str(g)][p], f - loc["cuts"][str(g)][p][0])

    out = {"samples": len(chk), "keys": {}, "shape": None}
    for s in chk:
        g, st, L = ep_map[s["episode_index"]]
        f = st + s["frame_index"]
        refs = [ref(g, c, f) for c in CAMS]
        for k, x in s["frames"].items():
            out["shape"] = list(x.shape)
            r = out["keys"].setdefault(k, {"equal_own": 0, "equal_cam": [0, 0, 0], "n": 0})
            r["n"] += 1
            if k.endswith(".stack"):
                r["equal_own"] += int(torch.equal(x, ref_stack(g, f)))
                for i in range(3):  # a stacked frame is a separate encode: its crops are NOT expected to equal SEP
                    r["equal_cam"][i] += int(torch.equal(x[:, i * 224:(i + 1) * 224, :], refs[i]))
            else:
                own = CAMS.index(k.split(".")[-1])
                for i in range(3):
                    r["equal_cam"][i] += int(torch.equal(x, refs[i]))
                r["equal_own"] += int(torch.equal(x, refs[own]))
    out["pass"] = all(v["equal_own"] == v["n"] for v in out["keys"].values())
    return out


# ====================================================================== orchestrate (stdlib only)
def cgroup_mb(name="memory.current"):
    try:
        return int(Path(f"/sys/fs/cgroup/{name}").read_text()) // 2**20
    except Exception:  # noqa: BLE001
        return 0


def launch(py, args, out):
    cmd = [py, str(HERE / "r3_pr3917.py"), "run", *args, "--out", str(out)]
    log = open(str(out).replace(".json", ".log"), "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True), log


def group(py, jobs, cap_s, guard_mb):
    """Run a list of (args, out) concurrently; return results, cgroup peak MB seen."""
    t0 = time.time()
    procs = [launch(py, a, o) for a, o in jobs]
    peak, killed = 0, False
    while any(p.poll() is None for p, _ in procs) and time.time() - t0 < cap_s + 240:
        cur = cgroup_mb()
        peak = max(peak, cur)
        if cur > guard_mb:
            killed = True
            break
        time.sleep(0.5)
    import signal

    for p, lg in procs:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()
        lg.close()
    rs = []
    for _, o in jobs:
        try:
            rs.append(json.loads(Path(o).read_text()))
        except Exception:  # noqa: BLE001
            rs.append({"status": "no_result", "out": str(o)})
    return rs, {"cgroup_peak_mb_seen": peak, "mem_guard_killed": killed, "wall_s": time.time() - t0}


def cmd_orchestrate(a):
    prep = json.loads(Path(a.prep).read_text())
    out = Path(a.out) / "pr3917"
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    py = os.environ["VENV_PR3917"] + "/bin/python"
    reps = int(os.environ.get("P_REPEATS", "3"))
    n1, n8 = int(os.environ.get("P_N1", "6000")), int(os.environ.get("P_N8", "1500"))
    cap = float(os.environ.get("P_CAP_S", "120"))
    check_n = int(os.environ.get("P_CHECK_N", "40"))
    guard = int(os.environ.get("MEM_GUARD_MB", "27000"))
    modes = os.environ.get("P_MODES", "p1 p8").split()
    kinds = os.environ.get("P_KINDS", "SEP STACK").split()
    quota = int(os.environ.get("CPU_QUOTA", "8"))
    ds = prep["datasets"]
    loc = {"sep": {}, "stack": {}, "cuts": {}}
    vroot = Path(prep["datasets"]["SEP"]["root"]).parent.parent / "dl"
    for g in range(len(prep["groups"])):
        loc["sep"][str(g)] = {c: str(vroot / "variants" / f"{prep['r1_run']}/{prep['family']}/SEP/{prep['family']}_g{g:03d}__{c}.mp4") for c in CAMS}
        rows = sorted((r for r in prep["cuts"] if r["layout"] == "STACK" and r["group"] == g), key=lambda r: r["part"])
        loc["stack"][str(g)] = [str(vroot / "variants" / r["rel"]) for r in rows]
        loc["cuts"][str(g)] = [[r["from_frame"], r["to_frame"]] for r in rows]
    (out / "local_files.json").write_text(json.dumps(loc))
    for k in ("SEP", "STACK", "MULTI"):
        (out / f"ep_map_{k}.json").write_text(json.dumps(ds[k]["episode_map"]))
    res = {"config": {"repeats": reps, "n1": n1, "n8": n8, "cap_s": cap, "modes": modes, "kinds": kinds, "quota": quota,
                      "datasets": {k: {kk: v.get(kk) for kk in ("repo", "commit", "episodes", "frames", "video_files_per_key", "ep_len_mean")}
                                   for k, v in ds.items()}},
           "prep_runs": [], "checks": [], "runs": [], "groups": []}

    def base(kind, extra):
        d = ds[kind]
        args = ["--kind", kind, "--local-files", str(out / "local_files.json"), "--ep-map", str(out / f"ep_map_{kind}.json")]
        if kind == "MULTI":
            args += ["--repo", "local/r3-multi", "--root", d["root"]]
        else:
            args += ["--repo", d["repo"], "--revision", d["commit"]]
        return args + extra

    def save():
        (out / "pr3917.json").write_text(json.dumps(res, indent=1, default=str))

    # 1. prep run per dataset (sidecar build, metadata cache) + correctness check; not timed
    for kind in [*kinds, "MULTI"]:
        o = runs / f"check_{kind}.json"
        rs, g = group(py, [(base(kind, ["--mode", "check", "--check-n", str(check_n), "--n", str(check_n), "--cap-s", str(cap),
                                        "--threads", str(quota), "--seed", "7"]), o)], cap + 300, guard)
        r = rs[0]
        r["group"] = g
        res["checks"].append(r)
        print(f"[{time.strftime('%H:%M:%S')}] check {kind}: {r.get('status')} build {r.get('build_s')} check {json.dumps(r.get('check'))[:300]} "
              f"{r.get('error', '')}", flush=True)
        save()
    # 2. timed runs
    for rep in range(reps):
        order = [(k, m) for m in modes for k in kinds]
        if rep % 2:
            order = order[::-1]
        for kind, mode in order:
            seed = 100 + rep
            if mode == "p1":
                jobs = [(base(kind, ["--mode", mode, "--rep", str(rep), "--n", str(n1), "--cap-s", str(cap), "--threads", str(quota),
                                     "--seed", str(seed)]), runs / f"{kind}_{mode}_r{rep}.json")]
            else:
                jobs = [(base(kind, ["--mode", mode, "--rep", str(rep), "--n", str(n8), "--cap-s", str(cap), "--threads", "1", "--seed", str(seed),
                                     "--rank", str(i), "--world", "8"]), runs / f"{kind}_{mode}_r{rep}_rank{i}.json") for i in range(8)]
            rs, g = group(py, jobs, cap, guard)
            agg = aggregate(kind, mode, rep, rs, g)
            res["runs"] += rs
            res["groups"].append(agg)
            print(f"[{time.strftime('%H:%M:%S')}] {kind} {mode} r{rep}: sps {agg.get('sps')} first {agg.get('first_sample_s_max')} "
                  f"gets/ep {agg.get('range_gets_per_ep')} MB/ep {agg.get('mb_per_ep')} dec_cpu_ms/smp {agg.get('dec_cpu_ms')} "
                  f"pss {agg.get('peak_pss_gb_sum')} cg {g['cgroup_peak_mb_seen']} status {agg['statuses']}", flush=True)
            save()
    res["summary"] = summarize(res)
    save()
    write_md(out, res)


def aggregate(kind, mode, rep, rs, g):
    ok = [r for r in rs if r.get("sps")]
    a = {"kind": kind, "mode": mode, "rep": rep, "procs": len(rs), "statuses": sorted({r.get("status") for r in rs}), **g}
    if not ok:
        a["errors"] = [r.get("error") for r in rs][:2]
        return a
    tot = lambda k: sum(r["total"][k] for r in ok)  # noqa: E731
    eps = max(1, tot("episodes"))
    a.update(sps=sum(r["sps"] for r in ok), n=sum(r["n"] for r in ok), first_sample_s_max=max(r["first_sample_s"] for r in ok),
             range_gets_per_ep=tot("fetch_calls") / eps, mb_per_ep=tot("fetch_bytes") / eps / 1e6, http_req_per_ep=tot("requests") / eps,
             episodes_fetched=tot("episodes"), peak_pss_gb_sum=sum(r["peak_pss_gb"] for r in ok), peak_pss_gb_max=max(r["peak_pss_gb"] for r in ok),
             dec_cpu_ms=sum(r["steady"]["dec_cpu_s"] for r in ok) / max(1, sum(r["n"] for r in ok)) * 1e3,
             proc_cpu_ms=sum(r["steady"]["proc_cpu_s"] for r in ok) / max(1, sum(r["n"] for r in ok)) * 1e3,
             status_429=tot("status_429"), fetch_s_mean=sum(r["total"]["fetch_s"] for r in ok) / max(1, tot("fetch_calls")))
    return a


def summarize(res):
    import statistics as stt

    out = {}
    for g in res["groups"]:
        if g.get("sps") is None:
            continue
        out.setdefault(f"{g['kind']}|{g['mode']}", []).append(g)
    s = {}
    for k, gs in out.items():
        def m(key):
            xs = [g[key] for g in gs]
            return {"mean": stt.mean(xs), "sd": stt.stdev(xs) if len(xs) > 1 else None, "all": xs}
        s[k] = {key: m(key) for key in ("sps", "first_sample_s_max", "range_gets_per_ep", "mb_per_ep", "http_req_per_ep", "peak_pss_gb_sum",
                                         "peak_pss_gb_max", "dec_cpu_ms", "proc_cpu_ms", "cgroup_peak_mb_seen", "fetch_s_mean")}
        s[k]["repeats"] = len(gs)
    return s


def write_md(out, res):
    s = res["summary"]
    L = ["# DATA-11 round 3: PR #3917 on SEP vs STACK (equal-size files)", "",
         "p1 = 1 process; p8 = 8 processes as ranks 0..7 of WORLD_SIZE 8 (#3917 allows at most 1 DataLoader worker per rank). "
         "sps = samples/s after the first sample, summed over processes (1 sample = 1 time step, 3 camera images). "
         "mean +- sd over repeats. dec CPU = thread CPU inside #3917's decode calls, per sample; proc CPU = process CPU per sample.", "",
         "| layout | mode | reps | sps | first sample s | range GETs/episode | MB/episode | HTTP req/episode | mean GET s | dec CPU ms/smp | proc CPU ms/smp | peak PSS GB (sum) | cgroup peak MB |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def f(v, d=1):
        if v["sd"] is None:
            return f"{v['mean']:.{d}f}"
        return f"{v['mean']:.{d}f} +- {v['sd']:.{d}f}"
    for k in sorted(s):
        v = s[k]
        L.append(f"| {k.split('|')[0]} | {k.split('|')[1]} | {v['repeats']} | {f(v['sps'])} | {f(v['first_sample_s_max'], 2)} | "
                 f"{f(v['range_gets_per_ep'], 2)} | {f(v['mb_per_ep'])} | {f(v['http_req_per_ep'], 1)} | {f(v['fetch_s_mean'], 2)} | "
                 f"{f(v['dec_cpu_ms'], 2)} | {f(v['proc_cpu_ms'], 2)} | {f(v['peak_pss_gb_sum'], 2)} | {f(v['cgroup_peak_mb_seen'], 0)} |")
    L += ["", "## Correctness checks (return_uint8=True, decoded frame == local TorchCodec decode of the source frame)", ""]
    for c in res["checks"]:
        ck = c.get("check") or {}
        L.append(f"- {c.get('kind')}: status {c.get('status')}, build {c.get('build_s') and round(c['build_s'], 1)} s, "
                 f"pass {ck.get('pass')}, shape {ck.get('shape')}, per key {json.dumps(ck.get('keys'))} {c.get('error', '')}")
    L += ["", "MULTI = local dataset (not on the Hub) whose 3 video keys point at the multi-track cuts. equal_cam[i] counts samples "
          "equal to SEP camera i; a correct reader gives equal_own == n for every key."]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("orchestrate")
    o.add_argument("prep")
    o.add_argument("out")
    r = sub.add_parser("run")
    for k, t, d in (("--kind", str, "SEP"), ("--mode", str, "p1"), ("--rep", int, 0), ("--rank", int, 0), ("--world", int, 1),
                    ("--seed", int, 0), ("--n", int, 1000), ("--cap-s", float, 120.0), ("--threads", int, 8), ("--check-n", int, 0),
                    ("--repo", str, None), ("--revision", str, None), ("--root", str, None), ("--ep-map", str, None),
                    ("--local-files", str, None), ("--out", str, None)):
        r.add_argument(k, type=t, default=d)
    a = ap.parse_args()
    if a.cmd == "orchestrate":
        cmd_orchestrate(a)
    else:
        cmd_run(a)


if __name__ == "__main__":
    main()
