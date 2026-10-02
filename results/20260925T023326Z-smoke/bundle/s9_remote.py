"""Stage remote (PRIMARY result): read SEP / STACK / BIG over hf:// the way LeRobot reads remote video.

Open path = LeRobot VideoDecoderCache.get_decoder for a remote root: an fsspec file handle
(HfFileSystem().open(url, "rb"), fsspec default block size 5 MiB, readahead cache) wrapped in
CountingFile, passed to VideoDecoder(fh, seek_mode="approximate"). url = hf://datasets/<repo>@<rev>/
variants/<run>/<family>/<layout>/<file>.mp4 from stage upload (pinned commit). Frame lookup as in
decode_video_frames_torchcodec: index = round(ts * average_fps), pts checked against ts.

Arms, per layout (SEP, STACK, BIG), same random sample indices for every layout (seed per repeat):
  access  seq   all 3 cameras, one file after another (StreamingLeRobotDataset today)
          par   all 3 cameras, one thread per file (map-style dataset_reader.py ThreadPoolExecutor)
          one   one camera ("top") only; STACK still decodes the whole stacked frame
  regime  first   fresh handle + decoder per file per sample, closed after (decoder-cache miss / thrash)
          steady  every file of the family opened once up front (not timed per sample), then samples
Per arm: samples/s (one stream), p50 / p95 / mean latency per sample, HTTP requests per sample
(resolve, CDN, API separately), block fetches, bytes over HTTP, bytes the decoder consumed, open time,
429 count, the "resolvers" rate-limit budget left (min seen in the ratelimit headers).
Concurrency arm: RM_CONC spawn processes, each a single stream of seq/all samples for RM_CONC_S
seconds (DataLoader-like), per layout and regime: aggregate samples/s and resolve calls/s vs the limit.
Open-cost curve: per uploaded family and layout, fresh opens: open time, requests and bytes per
open vs frames per file (for stage model).
Checks: remote-decoded samples equal the local decode of the same file (bit-exact); SEP == BIG
remote samples; STACK crop vs SEP PSNR (lossy, informational).

Usage: python s9_remote.py <selection.json> <enc_root> <out_dir>
       env: SMOKE, RM_FAMILY, RM_N_FIRST, RM_N_STEADY, RM_REPEATS, RM_OPEN_N, RM_CONC, RM_CONC_S, RM_BUDGET_S
"""

import json
import os
import random
import sys
import time

from common import pin_threads

pin_threads(1)  # before torch is imported: env for OpenMP; torch pools pinned again below

from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from torchcodec.decoders import VideoDecoder  # noqa: E402

from common import (H, LAYOUTS, REMOTE, CountingFile, Family, ci_halfwidth, cpu_quota, install_http_spy,  # noqa: E402
                    load_selection, memlog, pct, proc_status, resolver_budget, thread_report, write_json)

pin_threads(1)
KEYS = ["requests", "resolve_requests", "cdn_requests", "api_requests", "http_bytes", "fetch_calls", "fetch_bytes",
        "status_429", "status_other_err", "cdn_hit", "cdn_miss", "http_s", "fetch_s"]
ACCESS = {"seq": None, "par": None, "one": ["top"]}


def snap(cnt=None):
    s = {k: REMOTE[k] for k in KEYS}
    if cnt is not None:
        s.update({f"dec_{k}": v for k, v in cnt.items()})
    return s


def delta(a, b):
    return {k: b[k] - a[k] for k in a}


class Remote:
    def __init__(self, base, enc_root):
        from huggingface_hub import HfFileSystem

        self.fs, self.base, self.enc_root = HfFileSystem(), base, enc_root

    def url(self, local_path):
        return f"{self.base}/{os.path.relpath(str(local_path), str(self.enc_root))}"

    def open(self, local_path, cnt):
        fh = self.fs.open(self.url(local_path), "rb")
        try:
            dec = VideoDecoder(CountingFile(fh, cnt), seek_mode="approximate")
        except Exception:
            fh.close()
            raise
        return dec, fh


def decode(dec, fi, fps):
    ts = fi / fps
    fb = dec.get_frames_at(indices=[round(ts * dec.metadata.average_fps)])
    return fb.data[0], abs(float(fb.pts_seconds[0]) - ts)


def one_sample(R, F, layout, idx, cams, regime, par_pool, decs, cnt):
    """-> (tensor [k,3,H,W], pts_err, open_s, decode_s, close_s) for one timestamp."""
    plan = F.plan(layout, idx, cams)
    fi_of = {}
    for p, fi, _ in plan:
        fi_of.setdefault(p, fi)

    def work(p):
        t0 = time.perf_counter()
        if regime == "first":
            dec, fh = R.open(p, cnt)
        else:
            dec, fh = decs[p], None
        t1 = time.perf_counter()
        x, err = decode(dec, fi_of[p], F.fps)
        t2 = time.perf_counter()
        if fh is not None:
            del dec
            fh.close()
        return p, x, err, t1 - t0, t2 - t1, time.perf_counter() - t2

    rs = list(par_pool.map(work, fi_of)) if (par_pool is not None and len(fi_of) > 1) else [work(p) for p in fi_of]
    got = {r[0]: r[1] for r in rs}
    imgs = [got[p][:, c * H : (c + 1) * H, :] if c is not None else got[p] for p, _, c in plan]
    return (torch.stack(imgs), max(r[2] for r in rs), sum(r[3] for r in rs), sum(r[4] for r in rs), sum(r[5] for r in rs))


def open_all(R, F, layout, cnt):
    t0 = time.perf_counter()
    s0 = snap()
    decs, fhs = {}, []
    for p in F.files(layout):
        d, fh = R.open(p, cnt)
        decs[str(p)] = d
        fhs.append(fh)
    return decs, fhs, {"files": len(fhs), "open_all_s": time.perf_counter() - t0, **delta(s0, snap())}


def close_all(decs, fhs):
    decs.clear()
    for fh in fhs:
        fh.close()


def new_cnt():
    return {"reads": 0, "seeks": 0, "read_bytes": 0}


def run_arm(R, F, layout, access, regime, idxs, warm_idx, decs, pool):
    cams = ACCESS[access] or F.cams
    cnt = new_cnt() if decs is None else STEADY_CNT
    par = pool if access == "par" else None
    one_sample(R, F, layout, warm_idx, cams, regime, par, decs, cnt)  # warm-up, not recorded
    lat, opn, dcd, cls, errs, outs = [], [], [], [], [], []
    c0 = dict(cnt)
    s0 = snap()
    t0 = time.perf_counter()
    for i in idxs:
        ta = time.perf_counter()
        x, err, o, d, c = one_sample(R, F, layout, i, cams, regime, par, decs, cnt)
        lat.append(time.perf_counter() - ta)
        opn.append(o)
        dcd.append(d)
        cls.append(c)
        errs.append(err)
        if len(outs) < 3:
            outs.append(x)
    wall = time.perf_counter() - t0
    d = delta(s0, snap())
    dc = {k: cnt[k] - c0[k] for k in cnt}
    n = len(idxs)
    res = {"layout": layout, "access": access, "regime": regime, "n": n, "wall_s": wall, "samples_per_s": n / wall,
           "lat_p50_ms": pct(lat, 0.5) * 1e3, "lat_p95_ms": pct(lat, 0.95) * 1e3, "lat_mean_ms": sum(lat) / n * 1e3,
           "lat_mean_ci95_ms": ci_halfwidth(lat) * 1e3, "open_ms_per_sample": sum(opn) / n * 1e3,
           "decode_ms_per_sample": sum(dcd) / n * 1e3, "close_ms_per_sample": sum(cls) / n * 1e3,
           "files_per_sample": len({p for p, _, _ in F.plan(layout, idxs[0], cams)}),
           "pts_err_max_s": max(errs), "resolver_remaining_end": REMOTE["resolver_remaining"],
           "resolver_min_seen": REMOTE["resolver_min_seen"], "resolver_reset_s_end": REMOTE["resolver_reset_s"],
           "totals": {**d, **{f"dec_{k}": v for k, v in dc.items()}}, "lat_ms": [x * 1e3 for x in lat]}
    for k, v in d.items():
        res[f"{k}_per_sample"] = v / n
    for k, v in dc.items():
        res[f"dec_{k}_per_sample"] = v / n
    res["resolves_per_s"] = d["resolve_requests"] / wall
    return res, outs


STEADY_CNT = new_cnt()


# ------------------------------------------------------------------ concurrency worker (own process)
def conc_worker(a):
    install_http_spy()
    sel = load_selection(a["sel"], a["smoke"])
    F = Family(a["enc_root"], a["family"], sel["active"][a["family"]], sel["cams"], sel["fps"])
    R = Remote(a["base"], a["enc_root"])
    cnt = new_cnt()
    decs = fhs = None
    if a["regime"] == "steady":
        decs, fhs, _ = open_all(R, F, a["layout"], cnt)
    rng = random.Random(a["seed"])
    one_sample(R, F, a["layout"], rng.randrange(F.total), F.cams, a["regime"], None, decs, cnt)
    # start together: wait until the shared start time so the processes overlap
    while time.time() < a["start_at"]:
        time.sleep(0.01)
    s0 = snap()
    lat = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < a["dur_s"]:
        ta = time.perf_counter()
        one_sample(R, F, a["layout"], rng.randrange(F.total), F.cams, a["regime"], None, decs, cnt)
        lat.append(time.perf_counter() - ta)
    wall = time.perf_counter() - t0
    if decs is not None:
        close_all(decs, fhs)
    return {"n": len(lat), "wall_s": wall, "lat_ms": [x * 1e3 for x in lat], "d": delta(s0, snap()),
            "resolver_min_seen": REMOTE["resolver_min_seen"]}


def run_conc(args_list):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(len(args_list)) as pool:
        return pool.map(conc_worker, args_list)


# ------------------------------------------------------------------ main
def main():
    sel_path, enc_root, out_dir = sys.argv[1:4]
    smoke = os.environ.get("SMOKE", "0") == "1"
    D = Path(out_dir) / "remote"
    D.mkdir(parents=True, exist_ok=True)
    up = json.loads((Path(out_dir) / "upload" / "upload.json").read_text())
    assert not up["readback_problems"], "upload read-back failed; not measuring"
    sel = load_selection(sel_path, smoke)
    fam = os.environ.get("RM_FAMILY") or ("ckpt_5k" if smoke else "ckpt_full")
    assert fam in up["families"], (fam, up["families"])
    cfg = {"family": fam, "n_first": int(os.environ.get("RM_N_FIRST", "6" if smoke else "40")),
           "n_steady": int(os.environ.get("RM_N_STEADY", "12" if smoke else "80")),
           "repeats": int(os.environ.get("RM_REPEATS", "1" if smoke else "3")),
           "open_n": int(os.environ.get("RM_OPEN_N", "3" if smoke else "8")),
           "conc": int(os.environ.get("RM_CONC", str(cpu_quota()))), "conc_s": float(os.environ.get("RM_CONC_S", "15" if smoke else "60")),
           "budget_s": float(os.environ.get("RM_BUDGET_S", "900" if smoke else "4800")),
           "repo": up["repo"], "revision": up["revision"], "prefix": up["prefix"], "hf_url_base": up["hf_url_base"],
           "fsspec_block_size": None, "threads": thread_report()}
    install_http_spy()
    base = up["hf_url_base"]
    R = Remote(base, enc_root)
    import fsspec
    import huggingface_hub

    probe = up["files"][0]["path_in_repo"]
    budget0 = resolver_budget(up["repo"], up["revision"], probe)
    cfg.update(fsspec=fsspec.__version__, huggingface_hub=huggingface_hub.__version__,
               resolver_budget_start={"remaining": budget0[0], "reset_s": budget0[1], "limit": budget0[2], "window_s": budget0[3]})
    limit_per_s = budget0[2] / budget0[3] if budget0[2] > 0 and budget0[3] > 0 else float("nan")
    print(f"remote: family {fam}, base {base}, resolver budget {budget0}", flush=True)
    F = Family(enc_root, fam, sel["active"][fam], sel["cams"], sel["fps"])
    t_start = time.perf_counter()
    res = {"config": cfg, "open_curve": {}, "runs": [], "conc": [], "checks": {}, "skipped": []}

    # ---- warm-up: TLS connections, repo/revision lookup, dircache (not recorded)
    wc = new_cnt()
    d, fh = R.open(F.files("SEP")[0], wc)
    decode(d, 0, F.fps)
    del d
    fh.close()
    fh_block = R.fs.open(R.url(F.files("SEP")[0]), "rb")
    cfg["fsspec_block_size"], cfg["fsspec_cache"] = fh_block.blocksize, type(fh_block.cache).__name__
    fh_block.close()

    # ---- open-cost curve over every uploaded family
    rng = random.Random(7)
    for ofam in up["families"]:
        OF = Family(enc_root, ofam, sel["active"][ofam], sel["cams"], sel["fps"])
        for lay in LAYOUTS:
            files = OF.files(lay)
            nfs = [n for n in OF.frames_per_file(lay) for _ in (OF.cams if lay != "STACK" else [0])]
            rows = []
            for r in range(cfg["open_n"]):
                k = r % len(files)
                cnt = new_cnt()
                s0 = snap(cnt)
                t0 = time.perf_counter()
                dec, fh = R.open(files[k], cnt)
                t1 = time.perf_counter()
                s1 = snap(cnt)
                dec.get_frames_at(indices=[rng.randrange(nfs[k])])
                t2 = time.perf_counter()
                s2 = snap(cnt)
                dec.get_frames_at(indices=[rng.randrange(nfs[k])])
                t3 = time.perf_counter()
                s3 = snap(cnt)
                del dec
                fh.close()
                rows.append({"file": k, "open_ms": (t1 - t0) * 1e3, "first_ms": (t2 - t1) * 1e3, "next_ms": (t3 - t2) * 1e3,
                             "open": delta(s0, s1), "first": delta(s1, s2), "next": delta(s2, s3)})
            med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
            cell = {"family": ofam, "layout": lay, "files": len(files), "frames_per_file_mean": sum(nfs) / len(nfs),
                    "file_bytes_mean": sum(os.path.getsize(p) for p in files) / len(files),
                    "open_ms_median": med([x["open_ms"] for x in rows]), "first_ms_median": med([x["first_ms"] for x in rows]),
                    "next_ms_median": med([x["next_ms"] for x in rows]), "rows": rows}
            for ph in ("open", "first", "next"):
                for kk in ("requests", "resolve_requests", "cdn_requests", "api_requests", "fetch_calls", "http_bytes", "dec_read_bytes", "dec_reads", "dec_seeks"):
                    cell[f"{ph}_{kk}_mean"] = sum(x[ph][kk] for x in rows) / len(rows)
            res["open_curve"][f"{ofam}|{lay}"] = cell
            print(f"open {ofam}|{lay}: F {cell['frames_per_file_mean']:.0f} size {cell['file_bytes_mean'] / 1e6:.1f}MB open {cell['open_ms_median']:.0f}ms "
                  f"(req {cell['open_requests_mean']:.1f}, res {cell['open_resolve_requests_mean']:.1f}, api {cell['open_api_requests_mean']:.1f}, "
                  f"{cell['open_http_bytes_mean'] / 1e6:.2f}MB) first {cell['first_ms_median']:.0f}ms (fetch {cell['first_fetch_calls_mean']:.1f}) "
                  f"next {cell['next_ms_median']:.0f}ms (fetch {cell['next_fetch_calls_mean']:.1f})", flush=True)
    write_json(D / "remote.json", res)

    # ---- single-stream arms
    pool = ThreadPoolExecutor(max_workers=len(F.cams))
    local_decs = {}
    ref = {}
    mism = {"local_vs_remote_checked": 0, "local_vs_remote_mismatch": 0, "sep_vs_big_checked": 0, "sep_vs_big_mismatch": 0,
            "stack_vs_sep_psnr_db": []}
    for rep in range(cfg["repeats"]):
        rr = random.Random(1000 + rep)
        idx_first = [rr.randrange(F.total) for _ in range(cfg["n_first"])]
        idx_steady = [rr.randrange(F.total) for _ in range(cfg["n_steady"])]
        warm_idx = rr.randrange(F.total)
        order = LAYOUTS if rep % 2 == 0 else LAYOUTS[::-1]
        for regime in ("first", "steady"):
            idxs = idx_first if regime == "first" else idx_steady
            for lay in order:
                if time.perf_counter() - t_start > cfg["budget_s"]:
                    res["skipped"].append({"rep": rep, "regime": regime, "layout": lay, "why": "budget"})
                    continue
                decs = fhs = None
                open_stats = None
                if regime == "steady":
                    decs, fhs, open_stats = open_all(R, F, lay, STEADY_CNT)
                for access in ("seq", "par", "one"):
                    r, outs = run_arm(R, F, lay, access, regime, idxs, warm_idx, decs, pool)
                    r.update(repeat=rep, seed=1000 + rep, steady_open=open_stats)
                    res["runs"].append(r)
                    # correctness on the first samples of the first repeat
                    if rep == 0 and access in ("seq", "one"):
                        cams = ACCESS[access] or F.cams
                        for j, x in enumerate(outs):
                            loc = []
                            for p, fi, c in F.plan(lay, idxs[j], cams):
                                if p not in local_decs:
                                    local_decs[p] = VideoDecoder(p, seek_mode="approximate")
                                y, _ = decode(local_decs[p], fi, F.fps)
                                loc.append(y[:, c * H : (c + 1) * H, :] if c is not None else y)
                            mism["local_vs_remote_checked"] += 1
                            mism["local_vs_remote_mismatch"] += int(not torch.equal(x, torch.stack(loc)))
                            ref[(regime, access, lay, j)] = x
                    print(f"[{time.perf_counter() - t_start:.0f}s] r{rep} {regime:6s} {lay:5s} {access}: {r['samples_per_s']:.2f} sps, "
                          f"p50 {r['lat_p50_ms']:.0f} p95 {r['lat_p95_ms']:.0f} ms, res/smp {r['resolve_requests_per_sample']:.2f} "
                          f"cdn/smp {r['cdn_requests_per_sample']:.2f} api/smp {r['api_requests_per_sample']:.2f}, "
                          f"{r['http_bytes_per_sample'] / 1e6:.2f} MB/smp (decoder used {r['dec_read_bytes_per_sample'] / 1e6:.2f}), "
                          f"open {r['open_ms_per_sample']:.0f} ms/smp, 429 {r['totals']['status_429']}, budget min {r['resolver_min_seen']}", flush=True)
                if decs is not None:
                    close_all(decs, fhs)
            write_json(D / "remote.json", res)
        memlog(out_dir, f"remote_rep{rep}")
    pool.shutdown()
    local_decs.clear()
    for (regime, access, lay, j), x in ref.items():
        if lay == "SEP" and (regime, access, "BIG", j) in ref:
            mism["sep_vs_big_checked"] += 1
            mism["sep_vs_big_mismatch"] += int(not torch.equal(x, ref[(regime, access, "BIG", j)]))
        if lay == "SEP" and (regime, access, "STACK", j) in ref:
            y = ref[(regime, access, "STACK", j)]
            mse = ((x.double() - y.double()) ** 2).mean().item()
            mism["stack_vs_sep_psnr_db"].append(99.0 if mse == 0 else 10 * torch.log10(torch.tensor(255.0**2 / mse)).item())
    res["checks"] = mism
    res["pts_violations_1e-4"] = sum(int(r["pts_err_max_s"] > 1e-4) for r in res["runs"])

    # ---- concurrency arms (DataLoader-like: independent processes, seq / all cameras)
    if cfg["conc"] > 0:
        for regime in ("first", "steady"):
            for li, lay in enumerate(LAYOUTS):
                if time.perf_counter() - t_start > cfg["budget_s"]:
                    res["skipped"].append({"conc": True, "regime": regime, "layout": lay, "why": "budget"})
                    continue
                lead = 25.0 if regime == "steady" else 12.0
                start_at = time.time() + lead
                args = [{"sel": sel_path, "smoke": smoke, "enc_root": enc_root, "family": fam, "base": base, "layout": lay,
                         "regime": regime, "seed": 50_000 + 100 * li + w, "dur_s": cfg["conc_s"], "start_at": start_at}
                        for w in range(cfg["conc"])]
                t0 = time.perf_counter()
                try:
                    outs = run_conc(args)
                except Exception as exc:
                    res["conc"].append({"layout": lay, "regime": regime, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                n = sum(o["n"] for o in outs)
                wall = max(o["wall_s"] for o in outs)
                tot = {k: sum(o["d"][k] for o in outs) for k in outs[0]["d"]}
                lat = [x for o in outs for x in o["lat_ms"]]
                c = {"layout": lay, "regime": regime, "procs": cfg["conc"], "dur_s": cfg["conc_s"], "n": n, "wall_s": wall,
                     "samples_per_s": n / wall, "lat_p50_ms": pct(lat, 0.5), "lat_p95_ms": pct(lat, 0.95),
                     "resolves_per_s": tot["resolve_requests"] / wall, "requests_per_s": tot["requests"] / wall,
                     "resolve_per_sample": tot["resolve_requests"] / max(1, n), "cdn_per_sample": tot["cdn_requests"] / max(1, n),
                     "api_per_sample": tot["api_requests"] / max(1, n), "http_mb_per_s": tot["http_bytes"] / wall / 1e6,
                     "status_429": tot["status_429"], "status_other_err": tot["status_other_err"],
                     "resolver_min_seen": min((o["resolver_min_seen"] for o in outs if o["resolver_min_seen"] < 1 << 30), default=None),
                     "resolver_limit_per_s": limit_per_s,
                     "stage_wall_s": time.perf_counter() - t0, "totals": tot}
                res["conc"].append(c)
                print(f"[{time.perf_counter() - t_start:.0f}s] conc{cfg['conc']} {regime:6s} {lay:5s}: {c['samples_per_s']:.1f} sps, "
                      f"{c['resolves_per_s']:.1f} resolves/s (limit {limit_per_s:.0f}/s), {c['http_mb_per_s']:.0f} MB/s, "
                      f"p50 {c['lat_p50_ms']:.0f} p95 {c['lat_p95_ms']:.0f} ms, 429 {c['status_429']}, budget min {c['resolver_min_seen']}", flush=True)
                write_json(D / "remote.json", res)

    budget1 = resolver_budget(up["repo"], up["revision"], probe)
    res["resolver_budget_end"] = {"remaining": budget1[0], "reset_s": budget1[1], "limit": budget1[2], "window_s": budget1[3]}
    res["resolver_limit_per_s"] = limit_per_s
    res["totals_stage"] = snap()
    res["aggregate"] = aggregate(res["runs"], limit_per_s)
    res["wall_s"] = time.perf_counter() - t_start
    res["rss"] = proc_status()
    write_json(D / "remote.json", res)
    write_summary(D, res)
    ok = (mism["local_vs_remote_mismatch"] == 0 and mism["sep_vs_big_mismatch"] == 0 and res["pts_violations_1e-4"] == 0)
    print(f"remote done in {res['wall_s']:.0f}s; checks {json.dumps({k: v for k, v in mism.items() if k != 'stack_vs_sep_psnr_db'})}; "
          f"429 total {REMOTE['status_429']}; budget end {budget1}", flush=True)
    sys.exit(0 if ok else 1)


def aggregate(runs, limit_per_s):
    arms = {}
    for r in runs:
        arms.setdefault(f"{r['regime']}|{r['layout']}|{r['access']}", []).append(r)
    out = {}
    for key, rs in arms.items():
        n = sum(r["n"] for r in rs)
        lat = [x for r in rs for x in r["lat_ms"]]
        tot = {k: sum(r["totals"][k] for r in rs) for k in rs[0]["totals"]}
        sps = [r["samples_per_s"] for r in rs]
        a = {"regime": rs[0]["regime"], "layout": rs[0]["layout"], "access": rs[0]["access"], "repeats": len(rs), "n": n,
             "samples_per_s": n / sum(r["wall_s"] for r in rs), "sps_per_repeat": sps,
             "sps_ci95_halfwidth": ci_halfwidth(sps) if len(sps) > 1 else None,
             "lat_p50_ms": pct(lat, 0.5), "lat_p95_ms": pct(lat, 0.95), "lat_mean_ms": sum(lat) / len(lat),
             "lat_mean_ci95_ms": ci_halfwidth([x for x in lat]),
             "open_ms_per_sample": sum(r["open_ms_per_sample"] * r["n"] for r in rs) / n,
             "decode_ms_per_sample": sum(r["decode_ms_per_sample"] * r["n"] for r in rs) / n,
             "files_per_sample": rs[0]["files_per_sample"], "status_429": tot["status_429"],
             "resolver_min_seen": min(r["resolver_min_seen"] for r in rs), "pts_err_max_s": max(r["pts_err_max_s"] for r in rs)}
        for k, v in tot.items():
            a[f"{k}_per_sample"] = v / n
        a["resolves_per_s_single_stream"] = tot["resolve_requests"] / sum(r["wall_s"] for r in rs)
        a["max_sps_under_resolver_limit"] = limit_per_s / a["resolve_requests_per_sample"] if a["resolve_requests_per_sample"] else None
        out[key] = a
    return out


def write_summary(D, res):
    lim = res["resolver_limit_per_s"]
    b0 = res["config"]["resolver_budget_start"]
    L = [f"# DATA-11 remote (hf://) reads: family {res['config']['family']}", "",
         f"Repo `{res['config']['repo']}@{res['config']['revision'][:10]}` (private). fsspec block {res['config']['fsspec_block_size']} B, cache "
         f"{res['config'].get('fsspec_cache')}. Resolver limit {b0['limit']} per {b0['window_s']} s = {lim:.1f}/s per token.", "",
         "## Single stream (one reader)", "",
         "| regime | layout | access | sps | +-CI | p50 ms | p95 ms | resolve/smp | CDN/smp | API/smp | MB/smp | dec MB/smp | open ms/smp | 429 | sps cap at resolver limit |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(res["aggregate"], key=lambda k: (k.split("|")[0], k.split("|")[2], LAYOUTS.index(k.split("|")[1]))):
        a = res["aggregate"][k]
        ci = f"{a['sps_ci95_halfwidth']:.2f}" if a["sps_ci95_halfwidth"] is not None else "-"
        cap = f"{a['max_sps_under_resolver_limit']:.1f}" if a["max_sps_under_resolver_limit"] else "-"
        L.append(f"| {a['regime']} | {a['layout']} | {a['access']} | {a['samples_per_s']:.2f} | {ci} | {a['lat_p50_ms']:.0f} | {a['lat_p95_ms']:.0f} | "
                 f"{a['resolve_requests_per_sample']:.2f} | {a['cdn_requests_per_sample']:.2f} | {a['api_requests_per_sample']:.2f} | "
                 f"{a['http_bytes_per_sample'] / 1e6:.2f} | {a['dec_read_bytes_per_sample'] / 1e6:.2f} | {a['open_ms_per_sample']:.0f} | {a['status_429']} | {cap} |")
    if res["conc"]:
        L += ["", "## Concurrent readers (processes, seq / all cameras)", "",
              "| regime | layout | procs | sps | resolves/s | limit/s | MB/s | p50 ms | p95 ms | 429 | budget min seen |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in res["conc"]:
            if "error" in c:
                L.append(f"| {c['regime']} | {c['layout']} | error: {c['error']} |")
                continue
            L.append(f"| {c['regime']} | {c['layout']} | {c['procs']} | {c['samples_per_s']:.1f} | {c['resolves_per_s']:.1f} | {lim:.0f} | "
                     f"{c['http_mb_per_s']:.0f} | {c['lat_p50_ms']:.0f} | {c['lat_p95_ms']:.0f} | {c['status_429']} | {c['resolver_min_seen']} |")
    L += ["", "## Open cost over hf:// (fresh handle + decoder)", "",
          "| family | layout | frames/file | MB/file | open ms | req/open | resolve/open | API/open | MB/open | first ms | fetch first | next ms | fetch next |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in res["open_curve"].values():
        L.append(f"| {c['family']} | {c['layout']} | {c['frames_per_file_mean']:.0f} | {c['file_bytes_mean'] / 1e6:.1f} | {c['open_ms_median']:.0f} | "
                 f"{c['open_requests_mean']:.1f} | {c['open_resolve_requests_mean']:.1f} | {c['open_api_requests_mean']:.1f} | {c['open_http_bytes_mean'] / 1e6:.2f} | "
                 f"{c['first_ms_median']:.0f} | {c['first_fetch_calls_mean']:.1f} | {c['next_ms_median']:.0f} | {c['next_fetch_calls_mean']:.1f} |")
    ch = res["checks"]
    ps = ch["stack_vs_sep_psnr_db"]
    L += ["", "## Checks", "",
          f"- remote decode == local decode (bit-exact): {ch['local_vs_remote_checked'] - ch['local_vs_remote_mismatch']}/{ch['local_vs_remote_checked']}",
          f"- SEP == BIG remote samples: {ch['sep_vs_big_checked'] - ch['sep_vs_big_mismatch']}/{ch['sep_vs_big_checked']}",
          f"- STACK crop vs SEP PSNR (separate lossy encodes): min {min(ps) if ps else float('nan'):.1f} dB over {len(ps)} samples",
          f"- pts violations > 1e-4 s: {res['pts_violations_1e-4']} arms",
          f"- 429 responses in stage: {res['totals_stage']['status_429']}; resolver budget start {b0['remaining']}, end {res['resolver_budget_end']['remaining']}, "
          f"min seen {res['totals_stage'].get('resolver_min_seen', REMOTE['resolver_min_seen'])}",
          f"- skipped arms: {len(res['skipped'])}"]
    (D / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
