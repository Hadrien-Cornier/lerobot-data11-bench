"""Stage access (round 2): random access over hf:// (map-style / Lance-like remote case), cheaper reader fixes.

Family ckpt_full (3 groups, ~24.7k frames per file, real ABC median), all 3 cameras per sample, same random
indices for every arm of a repeat. Layouts: SEP (3 files), STACK (1 file, crop), MULTI (option C: 1 file with 3
video tracks, one TorchCodec decoder per track via stream_index).
Readers (readers.py):
  fs64K fs256K fs1M fs5M  HfFileSystem().open(url, block_size=B) (fs5M = LeRobot today: fsspec.open defaults)
  rc256K                  signed CDN URL resolved once per file (cached), Range GETs, readahead 256 KiB
  exact                   Lance-like: moov once, keyframe byte window per frame (+64 KiB slack), signed-URL transport
MULTI modes: naive = one reader per decoder (3 handles on one file); shared = one block cache (or one exact store)
per file for the 3 decoders.
Regimes: cold = fresh open per file per sample (decoder-cache miss); warm = every file opened once up front (untimed).
Cameras: seq (one after another, StreamingLeRobotDataset) | par (thread per camera, dataset_reader.py:419).
Signed URLs are resolved for every file before the arms (untimed); their resolve cost is 1 request per file per
~1 h, reported separately.
Concurrency: ACC_CONC processes of the best configs (warm and cold), aggregate sps, requests/s, MB/s, 429s.
Checks: every arm's first samples bit-exact vs a local decode of the same file.

Usage: python s11_access.py <prep.json> <out_dir>
Env: ACC_N_COLD (16), ACC_N_WARM (30), ACC_REPEATS (3), ACC_ARMS (filter substring list), ACC_CONC, ACC_CONC_S (45),
     ACC_BUDGET_S (3600), ACC_MB_CAP (stop when HTTP MB exceed it; laptop), ACC_CHECK_N (2)
"""

import json
import os
import random
import sys
import time

from common import pin_threads

pin_threads(1)

from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from torchcodec.decoders import VideoDecoder  # noqa: E402

import readers as RD  # noqa: E402
from common import (H, REMOTE, CountingFile, ci_halfwidth, cpu_quota, install_http_spy, memlog, pct, proc_status,  # noqa: E402
                    write_json)

pin_threads(1)
CAMS = ["top", "left_wrist", "right_wrist"]
READERS = {"fs64K": ("fs", 64 * RD.KiB), "fs256K": ("fs", 256 * RD.KiB), "fs1M": ("fs", RD.MiB), "fs5M": ("fs", 5 * RD.MiB),
           "rc256K": ("rc", 256 * RD.KiB), "exact": ("exact", None)}
KEYS = ["requests", "resolve_requests", "cdn_requests", "api_requests", "http_bytes", "fetch_calls", "status_429", "status_other_err"]


def all_arms():
    arms = []
    for rd in READERS:
        for regime in ("cold", "warm"):
            arms += [("SEP", "seq", rd, "-", regime), ("SEP", "par", rd, "-", regime), ("STACK", "-", rd, "-", regime)]
    for rd in ("fs256K", "fs5M"):
        for mode in ("naive", "shared"):
            for acc in ("seq", "par"):
                for regime in ("cold", "warm"):
                    arms.append(("MULTI", acc, rd, mode, regime))
    for acc in ("seq", "par"):
        for regime in ("cold", "warm"):
            arms.append(("MULTI", acc, "exact", "shared", regime))
    return arms


def arm_name(a):
    return "|".join(a)


class Layout:
    """Sample index -> [(rel path, frame, tracks, crop rows)] for one layout of the ckpt_full family."""

    def __init__(self, groups, fam="ckpt_full"):
        self.fam, self.groups = fam, groups
        self.cum = [0]
        for g in groups:
            self.cum.append(self.cum[-1] + g["frames"])
        self.total = self.cum[-1]

    def locate(self, idx):
        import bisect

        gi = bisect.bisect_right(self.cum, idx) - 1
        return gi, idx - self.cum[gi]

    def files(self, layout):
        n = len(self.groups)
        if layout == "SEP":
            return [f"{self.fam}/SEP/{self.fam}_g{g:03d}__{c}.mp4" for g in range(n) for c in CAMS]
        return [f"{self.fam}/{layout}/{self.fam}_g{g:03d}.mp4" for g in range(n)]

    def plan(self, layout, idx):
        gi, f = self.locate(idx)
        if layout == "SEP":
            return [(f"{self.fam}/SEP/{self.fam}_g{gi:03d}__{c}.mp4", f) for c in CAMS]
        return [(f"{self.fam}/{layout}/{self.fam}_g{gi:03d}.mp4", f)]


class Opened:
    def __init__(self, decs, pre=None, closers=()):
        self.decs, self.pre, self.closers = decs, pre, closers

    def close(self):
        self.decs = []
        for c in self.closers:
            try:
                c()
            except Exception:  # noqa: BLE001
                pass


class Remote:
    def __init__(self, prep):
        from huggingface_hub import HfFileSystem

        m = prep["multi"]["upload"]
        self.repo, self.rev = prep["repo"], m["revision"]
        self.prefix = prep["r1_prefix"]
        self.base = m["hf_url_base"]  # hf://datasets/<repo>@<rev2>/<prefix>: SEP / STACK unchanged from round 1, plus MULTI
        self.fs = HfFileSystem()
        self.resolver = RD.Resolver(self.repo, self.rev)
        self.sizes = prep["sizes"]
        # Lance-like byte index known in advance: moov offset / size and first packet, from the local copies
        root = Path(prep["local_root"])
        self.hints = {rel: RD.moov_hint(root / rel) for rel in self.sizes if os.environ.get("ACC_EXACT_HINT", "1") == "1"}

    def pir(self, rel):
        return f"{self.prefix}/{rel}"

    def open(self, rel, reader, tracks, mode, cnt):
        kind, B = READERS[reader]
        url = f"{self.base}/{rel}"
        if kind == "fs":
            if mode == "shared":
                fe = RD.HfRawFetcher(self.fs, url)
                sb = RD.SharedBlocks(fe, B)
                return Opened([VideoDecoder(CountingFile(RD.View(sb), cnt), stream_index=t, seek_mode="approximate") for t in tracks],
                              closers=[fe.close])
            fhs = [self.fs.open(url, "rb", block_size=B) for _ in tracks]
            return Opened([VideoDecoder(CountingFile(fh, cnt), stream_index=t, seek_mode="approximate") for fh, t in zip(fhs, tracks)],
                          closers=[fh.close for fh in fhs])
        fe = RD.RangeFetcher(self.resolver, self.pir(rel), size=self.sizes[rel])
        if kind == "rc":
            if mode == "shared":
                sb = RD.SharedBlocks(fe, B)
                return Opened([VideoDecoder(CountingFile(RD.View(sb), cnt), stream_index=t, seek_mode="approximate") for t in tracks])
            return Opened([VideoDecoder(CountingFile(RD.ReadaheadFile(fe, B), cnt), stream_index=t, seek_mode="approximate") for t in tracks])
        ef = RD.ExactFile(fe, hint=self.hints.get(rel))
        k = len(tracks)
        return Opened([VideoDecoder(CountingFile(RD.View(ef.store), cnt), stream_index=t, seek_mode="approximate") for t in tracks],
                      pre=lambda fr: ef.store.fetch_spans([ef.window(i, fr) for i in range(k)]))


def tracks_of(layout):
    return [0, 1, 2] if layout == "MULTI" else [None]


def snap():
    s = {k: REMOTE[k] for k in KEYS}
    s.update({f"rd_{k}": v for k, v in RD.STATS.items()})
    return s


def delta(a, b):
    return {k: b[k] - a[k] for k in a}


def decode_one(dec, fr):
    fb = dec.get_frames_at(indices=[fr])
    return fb.data[0], abs(float(fb.pts_seconds[0]) - fr / 30.0)


def sample(R, L, arm, idx, opened, cnt, pool):
    """-> (tensor [3, 3, H, W], pts err, open s). opened: dict rel -> Opened (warm) or None (cold)."""
    layout, acc, reader, mode, regime = arm
    tracks = tracks_of(layout)
    plan = L.plan(layout, idx)
    t_open = [0.0]

    def get(rel):
        if opened is not None:
            return opened[rel]
        t0 = time.perf_counter()
        o = R.open(rel, reader, tracks, mode, cnt)
        t_open[0] += time.perf_counter() - t0
        return o

    if layout == "SEP":
        def cam(pr):
            rel, fr = pr
            o = get(rel)
            if o.pre:
                o.pre(fr)
            x, e = decode_one(o.decs[0], fr)
            if opened is None:
                o.close()
            return x, e

        outs = list(pool.map(cam, plan)) if acc == "par" else [cam(p) for p in plan]
        return torch.stack([x for x, _ in outs]), max(e for _, e in outs), t_open[0]
    rel, fr = plan[0]
    o = get(rel)
    if o.pre:
        o.pre(fr)
    if layout == "STACK":
        x, e = decode_one(o.decs[0], fr)
        out = torch.stack([x[:, c * H : (c + 1) * H, :] for c in range(3)]), e
    else:
        outs = list(pool.map(lambda d: decode_one(d, fr), o.decs)) if acc == "par" else [decode_one(d, fr) for d in o.decs]
        out = torch.stack([x for x, _ in outs]), max(e for _, e in outs)
    if opened is None:
        o.close()
    return out[0], out[1], t_open[0]


def open_all(R, L, arm, cnt):
    layout, acc, reader, mode, regime = arm
    t0 = time.perf_counter()
    s0 = snap()
    op = {rel: R.open(rel, reader, tracks_of(layout), mode, cnt) for rel in L.files(layout)}
    return op, {"files": len(op), "open_all_s": time.perf_counter() - t0, **delta(s0, snap())}


def new_cnt():
    return {"reads": 0, "seeks": 0, "read_bytes": 0}


def run_arm(R, L, arm, idxs, warm_idx, pool, mb_cap=None):
    regime = arm[4]
    cnt = new_cnt()
    opened, open_stats = (open_all(R, L, arm, cnt) if regime == "warm" else (None, None))
    sample(R, L, arm, warm_idx, opened, cnt, pool)  # warm-up sample, not recorded
    lat, opn, errs, outs = [], [], [], []
    c0 = dict(cnt)
    s0 = snap()
    t0 = time.perf_counter()
    for i in idxs:
        if mb_cap is not None and REMOTE["http_bytes"] / 1e6 > mb_cap:
            break
        ta = time.perf_counter()
        x, e, o = sample(R, L, arm, i, opened, cnt, pool)
        lat.append(time.perf_counter() - ta)
        opn.append(o)
        errs.append(e)
        if len(outs) < 4:
            outs.append((i, x))
    wall = time.perf_counter() - t0
    d = delta(s0, snap())
    dc = {k: cnt[k] - c0[k] for k in cnt}
    if opened is not None:
        for o in opened.values():
            o.close()
    n = len(lat)
    if n == 0:
        return {"arm": arm_name(arm), "error": "no samples (MB cap)"}, []
    res = {"arm": arm_name(arm), "layout": arm[0], "access": arm[1], "reader": arm[2], "mode": arm[3], "regime": regime, "n": n,
           "wall_s": wall, "samples_per_s": n / wall, "lat_p50_ms": pct(lat, 0.5) * 1e3, "lat_p95_ms": pct(lat, 0.95) * 1e3,
           "open_ms_per_sample": sum(opn) / n * 1e3, "pts_err_max_s": max(errs), "steady_open": open_stats,
           "totals": {**d, **{f"dec_{k}": v for k, v in dc.items()}}}
    for k, v in res["totals"].items():
        res[f"{k}_per_sample"] = v / n
    # every HTTP request that moved video bytes: CDN requests (fsspec: 1 per block fetch; rc / exact: 1 per range)
    res["fetches_per_sample"] = d["cdn_requests"] / n
    return res, outs


def local_ref(enc, cache, rel, fr, layout):
    """Local decode of the same frame(s): [3, 3, H, W]."""
    if layout == "SEP":
        raise AssertionError
    if rel not in cache:
        cache[rel] = [VideoDecoder(str(enc / rel), stream_index=t, seek_mode="approximate") for t in tracks_of(layout)]
    ds = cache[rel]
    if layout == "STACK":
        x = ds[0].get_frames_at(indices=[fr]).data[0]
        return torch.stack([x[:, c * H : (c + 1) * H, :] for c in range(3)])
    return torch.stack([d.get_frames_at(indices=[fr]).data[0] for d in ds])


def check(enc, L, arm, outs, cache):
    layout = arm[0]
    bad = 0
    for i, x in outs:
        plan = L.plan(layout, i)
        if layout == "SEP":
            ys = []
            for rel, fr in plan:
                if rel not in cache:
                    cache[rel] = [VideoDecoder(str(enc / rel), seek_mode="approximate")]
                ys.append(cache[rel][0].get_frames_at(indices=[fr]).data[0])
            y = torch.stack(ys)
        else:
            y = local_ref(enc, cache, plan[0][0], plan[0][1], layout)
        bad += int(not torch.equal(x, y))
    return len(outs), bad


# ------------------------------------------------------------------ concurrency
def conc_worker(a):
    install_http_spy()
    prep = json.loads(Path(a["prep"]).read_text())
    R = Remote(prep)
    L = Layout(prep["groups"], prep.get("family", "ckpt_full"))
    arm = tuple(a["arm"])
    for rel in L.files(arm[0]):  # signed URLs once per process (untimed)
        R.resolver.signed(R.pir(rel))
    pool = ThreadPoolExecutor(max_workers=3)
    cnt = new_cnt()
    opened = open_all(R, L, arm, cnt)[0] if arm[4] == "warm" else None
    rng = random.Random(a["seed"])
    sample(R, L, arm, rng.randrange(L.total), opened, cnt, pool)
    while time.time() < a["start_at"]:
        time.sleep(0.01)
    s0 = snap()
    lat = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < a["dur_s"]:
        ta = time.perf_counter()
        sample(R, L, arm, rng.randrange(L.total), opened, cnt, pool)
        lat.append(time.perf_counter() - ta)
    wall = time.perf_counter() - t0
    return {"n": len(lat), "wall_s": wall, "lat_ms": [x * 1e3 for x in lat], "d": delta(s0, snap())}


DEFAULT_CONC = ["SEP|seq|fs5M|-|warm", "STACK|-|fs5M|-|warm", "SEP|par|rc256K|-|warm", "SEP|par|exact|-|warm", "STACK|-|exact|-|warm",
                "MULTI|par|exact|shared|warm", "MULTI|par|fs5M|shared|warm", "SEP|par|exact|-|cold", "STACK|-|exact|-|cold",
                "MULTI|par|exact|shared|cold"]


def summarize(runs):
    arms = {}
    for r in runs:
        if "error" not in r:
            arms.setdefault(r["arm"], []).append(r)
    out = {}
    for k, rs in arms.items():
        n = sum(r["n"] for r in rs)
        sps = [r["samples_per_s"] for r in rs]
        tot = {kk: sum(r["totals"][kk] for r in rs) for kk in rs[0]["totals"]}
        a = {"arm": k, "layout": rs[0]["layout"], "access": rs[0]["access"], "reader": rs[0]["reader"], "mode": rs[0]["mode"],
             "regime": rs[0]["regime"], "repeats": len(rs), "n": n, "sps": n / sum(r["wall_s"] for r in rs), "sps_per_repeat": sps,
             "sps_ci95": ci_halfwidth(sps) if len(sps) > 1 else None,
             "lat_p50_ms": sorted(r["lat_p50_ms"] for r in rs)[len(rs) // 2], "lat_p95_ms": max(r["lat_p95_ms"] for r in rs),
             "open_ms_per_sample": sum(r["open_ms_per_sample"] * r["n"] for r in rs) / n, "status_429": tot["status_429"]}
        for kk, v in tot.items():
            a[f"{kk}_per_sample"] = v / n
        a["fetches_per_sample"] = tot["cdn_requests"] / n
        out[k] = a
    return out


def write_summary(D, res):
    L = ["# DATA-11 round 2: random access over hf:// (family ckpt_full, 3 cameras per sample)", "",
         f"Repo `{res['config']['repo']}@{res['config']['revision'][:10]}`. sps = samples/s, one reader. fetch = one HTTP request that moves "
         "video bytes (CDN). fs* readers also pay 1 resolve per fetch; rc / exact resolve once per file (untimed, shown in "
         f"`presigned`: {res['presign']['resolves']} resolves for {res['presign']['files']} files).", "",
         "| regime | layout | cams | reader | mode | sps | +-CI | p50 ms | p95 ms | fetch/smp | resolve/smp | API/smp | MB/smp | dec MB/smp | open ms/smp | fallback/smp |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {k: i for i, k in enumerate(READERS)}
    for k in sorted(res["summary"], key=lambda k: (k.split("|")[4], k.split("|")[0], k.split("|")[1], order[k.split("|")[2]], k.split("|")[3])):
        a = res["summary"][k]
        ci = f"{a['sps_ci95']:.2f}" if a["sps_ci95"] is not None else "-"
        L.append(f"| {a['regime']} | {a['layout']} | {a['access']} | {a['reader']} | {a['mode']} | {a['sps']:.2f} | {ci} | {a['lat_p50_ms']:.0f} | "
                 f"{a['lat_p95_ms']:.0f} | {a['fetches_per_sample']:.2f} | {a['resolve_requests_per_sample']:.2f} | {a['api_requests_per_sample']:.2f} | "
                 f"{a['http_bytes_per_sample'] / 1e6:.3f} | {a['dec_read_bytes_per_sample'] / 1e6:.3f} | {a['open_ms_per_sample']:.0f} | "
                 f"{a['rd_exact_fallback_fetches_per_sample']:.2f} |")
    if res["conc"]:
        L += ["", f"## {res['config']['conc']} concurrent reader processes", "",
              "| config | sps total | fetch/s | resolves/s | MB/s | p50 ms | p95 ms | 429 | other err |", "|---|---|---|---|---|---|---|---|---|"]
        for c in res["conc"]:
            if "error" in c:
                L.append(f"| {c['arm']} | error: {c['error'][:200]} |")
                continue
            L.append(f"| {c['arm']} | {c['sps']:.1f} | {c['cdn_per_s']:.1f} | {c['resolves_per_s']:.1f} | {c['mb_per_s']:.1f} | {c['lat_p50_ms']:.0f} | "
                     f"{c['lat_p95_ms']:.0f} | {c['status_429']} | {c['status_other_err']} |")
    ch = res["checks"]
    L += ["", f"Checks: remote == local decode (bit-exact) {ch['checked'] - ch['mismatch']}/{ch['checked']}; pts violations > 1e-4 s: "
          f"{res['pts_violations']}; skipped arms {len(res['skipped'])}; 429 total {res['remote_totals']['status_429']}."]
    (D / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def main():
    prep_path, out_dir = sys.argv[1], Path(sys.argv[2])
    D = out_dir / "access"
    D.mkdir(parents=True, exist_ok=True)
    prep = json.loads(Path(prep_path).read_text())
    enc = Path(prep["local_root"])
    filt = os.environ.get("ACC_ARMS", "").split()
    arms = [a for a in all_arms() if not filt or any(f in arm_name(a) for f in filt)]
    cfg = {"n_cold": int(os.environ.get("ACC_N_COLD", "16")), "n_warm": int(os.environ.get("ACC_N_WARM", "30")),
           "repeats": int(os.environ.get("ACC_REPEATS", "3")), "conc": int(os.environ.get("ACC_CONC", str(cpu_quota()))),
           "conc_s": float(os.environ.get("ACC_CONC_S", "45")), "budget_s": float(os.environ.get("ACC_BUDGET_S", "3600")),
           "mb_cap": float(os.environ["ACC_MB_CAP"]) if os.environ.get("ACC_MB_CAP") else None,
           "check_n": int(os.environ.get("ACC_CHECK_N", "2")),
           "conc_arms": os.environ.get("ACC_CONC_ARMS", " ".join(DEFAULT_CONC)).split(), "arms": [arm_name(a) for a in arms],
           "repo": prep["repo"], "revision": prep["multi"]["upload"]["revision"]}
    install_http_spy()
    R = Remote(prep)
    L = Layout(prep["groups"], prep.get("family", "ckpt_full"))
    t_start = time.perf_counter()
    s0 = snap()
    lays = sorted({a[0] for a in arms})
    for lay in lays:
        for rel in L.files(lay):
            R.resolver.signed(R.pir(rel))
    ps = delta(s0, snap())
    res = {"config": cfg, "presign": {"files": sum(len(L.files(x)) for x in lays), "resolves": ps["resolve_requests"], "requests": ps["requests"],
                                      "s": time.perf_counter() - t_start}, "runs": [], "conc": [], "skipped": []}
    # warm-up: TLS, dircache
    wc = new_cnt()
    o = R.open(L.files("SEP")[0], "fs5M", [None], "-", wc)
    decode_one(o.decs[0], 0)
    o.close()
    pool = ThreadPoolExecutor(max_workers=3)
    cache = {}
    checks = {"checked": 0, "mismatch": 0}
    for rep in range(cfg["repeats"]):
        rr = random.Random(2000 + rep)
        idx_cold = [rr.randrange(L.total) for _ in range(cfg["n_cold"])]
        idx_warm = [rr.randrange(L.total) for _ in range(cfg["n_warm"])]
        warm_idx = rr.randrange(L.total)
        order = arms if rep % 2 == 0 else arms[::-1]
        for arm in order:
            if time.perf_counter() - t_start > cfg["budget_s"]:
                res["skipped"].append({"arm": arm_name(arm), "rep": rep, "why": "budget"})
                continue
            if cfg["mb_cap"] is not None and REMOTE["http_bytes"] / 1e6 > cfg["mb_cap"]:
                res["skipped"].append({"arm": arm_name(arm), "rep": rep, "why": "MB cap"})
                continue
            try:
                r, outs = run_arm(R, L, arm, idx_cold if arm[4] == "cold" else idx_warm, warm_idx, pool, cfg["mb_cap"])
            except Exception as exc:  # noqa: BLE001
                r, outs = {"arm": arm_name(arm), "error": f"{type(exc).__name__}: {exc}"}, []
            r["repeat"] = rep
            if rep == 0 and outs:
                c, b = check(enc, L, arm, outs[: cfg["check_n"]], cache)
                checks["checked"] += c
                checks["mismatch"] += b
                r["check"] = {"checked": c, "mismatch": b}
            res["runs"].append(r)
            if "error" in r:
                print(f"[{time.perf_counter() - t_start:.0f}s] {r['arm']} r{rep}: ERROR {r['error']}", flush=True)
            else:
                print(f"[{time.perf_counter() - t_start:.0f}s] {r['arm']} r{rep}: {r['samples_per_s']:.2f} sps, fetch/smp {r['fetches_per_sample']:.2f}, "
                      f"res/smp {r['resolve_requests_per_sample']:.2f}, {r['http_bytes_per_sample'] / 1e6:.3f} MB/smp (dec {r['dec_read_bytes_per_sample'] / 1e6:.3f}), "
                      f"p50 {r['lat_p50_ms']:.0f} ms, open {r['open_ms_per_sample']:.0f} ms/smp, fallback/smp {r['rd_exact_fallback_fetches_per_sample']:.2f}, "
                      f"check {r.get('check')}, HTTP MB so far {REMOTE['http_bytes'] / 1e6:.0f}", flush=True)
        write_json(D / "access.json", res)
        memlog(out_dir, f"access_rep{rep}")
    pool.shutdown()
    cache.clear()
    res["checks"] = checks
    res["pts_violations"] = sum(int(r.get("pts_err_max_s", 0) > 1e-4) for r in res["runs"])
    if cfg["conc"] > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        for name in cfg["conc_arms"]:
            if time.perf_counter() - t_start > cfg["budget_s"] + 1200:
                res["skipped"].append({"conc": name, "why": "budget"})
                continue
            start_at = time.time() + (40 if name.endswith("warm") else 20)
            args = [{"prep": prep_path, "arm": name.split("|"), "seed": 70_000 + w, "start_at": start_at, "dur_s": cfg["conc_s"]}
                    for w in range(cfg["conc"])]
            try:
                with ctx.Pool(cfg["conc"]) as p:
                    outs = p.map(conc_worker, args)
            except Exception as exc:  # noqa: BLE001
                res["conc"].append({"arm": name, "error": f"{type(exc).__name__}: {exc}"})
                continue
            n = sum(o["n"] for o in outs)
            wall = max(o["wall_s"] for o in outs)
            tot = {k: sum(o["d"][k] for o in outs) for k in outs[0]["d"]}
            lat = [x for o in outs for x in o["lat_ms"]]
            c = {"arm": name, "procs": len(outs), "n": n, "wall_s": wall, "sps": n / wall, "cdn_per_s": tot["cdn_requests"] / wall,
                 "resolves_per_s": tot["resolve_requests"] / wall, "mb_per_s": tot["http_bytes"] / wall / 1e6, "lat_p50_ms": pct(lat, 0.5),
                 "lat_p95_ms": pct(lat, 0.95), "status_429": tot["status_429"], "status_other_err": tot["status_other_err"], "totals": tot}
            res["conc"].append(c)
            print(f"[{time.perf_counter() - t_start:.0f}s] conc{cfg['conc']} {name}: {c['sps']:.1f} sps, {c['cdn_per_s']:.1f} fetch/s, "
                  f"{c['resolves_per_s']:.1f} resolves/s, {c['mb_per_s']:.0f} MB/s, p50 {c['lat_p50_ms']:.0f} ms, 429 {c['status_429']}", flush=True)
            write_json(D / "access.json", res)
    res["summary"] = summarize(res["runs"])
    res["remote_totals"] = dict(REMOTE)
    res["reader_stats"] = dict(RD.STATS)
    res["wall_s"] = time.perf_counter() - t_start
    res["rss"] = proc_status()
    write_json(D / "access.json", res)
    write_summary(D, res)
    ok = checks["mismatch"] == 0 and res["pts_violations"] == 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
