"""Round 3: random access over hf:// on EQUAL-SIZE files (reuses the round-2 arms of s11_access.py).

Layouts (family ckpt_full, 3 cameras per sample, same random sample indices for every arm of a repeat):
  SEP    3 files per group (one per camera), ~57-105 MB each (unchanged from rounds 1-2)
  STACK  equal-size cuts of the stacked file: 3 cuts per group, ~59-87 MB each (one decoder per cut)
  MULTI  equal-size cuts of the multi-track file (3 video tracks), one shared cache per file for the 3 track decoders
Readers: fs5M (LeRobot today), rc256K (signed link reused, 256 KiB readahead), exact (Lance-like exact byte ranges).
Regimes: warm (every file opened once up front) and cold (fresh open per file per sample). One reader, then
ACC_CONC processes (default = CPU quota) on the best warm and best cold config of each layout (picked from this run).
Control (prep_orig.json): the same arms on the round-2 files (STACK / MULTI ~3x bigger), exact reader only, so the
file-size effect shows up inside one job.

Usage: python r3_access.py <prep dir> <out_dir>
Env: ACC_N_COLD (16), ACC_N_WARM (30), ACC_REPEATS (3), ACC_CONC, ACC_CONC_S (45), ACC_BUDGET_S (3600), R3_CONTROL (1),
     ACC_ARMS (filter substrings)
"""

import bisect
import json
import os
import random
import sys
import time
from pathlib import Path

import s11_access as S  # pins threads, imports torch + torchcodec
from common import REMOTE, cpu_quota, install_http_spy, memlog, pct, proc_status, write_json

CAMS = S.CAMS


class Layout3:
    """Same interface as s11_access.Layout, over prep_eq / prep_orig groups (cuts of STACK / MULTI)."""

    def __init__(self, groups, fam="ckpt_full"):
        self.fam, self.groups = fam, groups
        self.cum = [0]
        for g in groups:
            self.cum.append(self.cum[-1] + g["frames"])
        self.total = self.cum[-1]

    def locate(self, idx):
        gi = bisect.bisect_right(self.cum, idx) - 1
        return gi, idx - self.cum[gi]

    def files(self, layout):
        if layout == "SEP":
            return [g["sep"][c] for g in self.groups for c in CAMS]
        return [x for g in self.groups for x in g[layout]]

    def plan(self, layout, idx):
        gi, f = self.locate(idx)
        g = self.groups[gi]
        if layout == "SEP":
            return [(g["sep"][c], f) for c in CAMS]
        p = next(i for i, (a, b) in enumerate(g["cuts"]) if a <= f < b)
        return [(g[layout][p], f - g["cuts"][p][0])]


S.Layout = Layout3  # s11_access.conc_worker / check build their Layout from the module global

ARMS_EQ = [("SEP", "seq", "fs5M", "-"), ("SEP", "par", "fs5M", "-"), ("SEP", "par", "rc256K", "-"), ("SEP", "par", "exact", "-"),
           ("STACK", "-", "fs5M", "-"), ("STACK", "-", "rc256K", "-"), ("STACK", "-", "exact", "-"),
           ("MULTI", "par", "fs5M", "shared"), ("MULTI", "par", "rc256K", "shared"), ("MULTI", "par", "exact", "shared")]
ARMS_ORIG = [("STACK", "-", "exact", "-"), ("STACK", "-", "rc256K", "-"), ("MULTI", "par", "exact", "shared")]


def conc_worker3(a):
    S.Layout = Layout3
    return S.conc_worker(a)


def run_set(tag, prep_path, arms, out, cfg, t_start, conc=True):
    D = out / tag
    D.mkdir(parents=True, exist_ok=True)
    prep = json.loads(Path(prep_path).read_text())
    enc = Path(prep["local_root"])
    R = S.Remote(prep)
    L = Layout3(prep["groups"], prep["family"])
    s0 = S.snap()
    lays = sorted({a[0] for a in arms})
    for lay in lays:
        for rel in L.files(lay):
            R.resolver.signed(R.pir(rel))
    ps = S.delta(s0, S.snap())
    files_mb = {lay: sorted(round(prep["sizes"][r] / 1e6, 1) for r in L.files(lay)) for lay in lays}
    res = {"config": {**cfg, "repo": prep["repo"], "revision": prep["multi"]["upload"]["revision"], "arms": ["|".join(a) for a in arms],
                      "layout_mode": prep["layout_mode"], "files_mb": files_mb, "conc": cfg["conc"] if conc else 0},
           "presign": {"files": sum(len(L.files(x)) for x in lays), "resolves": ps["resolve_requests"], "requests": ps["requests"]},
           "runs": [], "conc": [], "skipped": []}
    wc = S.new_cnt()
    o = R.open(L.files("SEP")[0] if "SEP" in lays else L.files(lays[0])[0], "fs5M", [None], "-", wc)
    S.decode_one(o.decs[0], 0)
    o.close()
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=3)
    cache, checks = {}, {"checked": 0, "mismatch": 0}
    full = [(*a, rg) for rg in ("cold", "warm") for a in arms]
    filt = os.environ.get("ACC_ARMS", "").split()
    full = [a for a in full if not filt or any(f in "|".join(a) for f in filt)]
    for rep in range(cfg["repeats"]):
        rr = random.Random(3000 + rep)
        idx_cold = [rr.randrange(L.total) for _ in range(cfg["n_cold"])]
        idx_warm = [rr.randrange(L.total) for _ in range(cfg["n_warm"])]
        warm_idx = rr.randrange(L.total)
        for arm in (full if rep % 2 == 0 else full[::-1]):
            name = "|".join(arm)
            if time.perf_counter() - t_start > cfg["budget_s"]:
                res["skipped"].append({"arm": name, "rep": rep, "why": "budget"})
                continue
            try:
                r, outs = S.run_arm(R, L, arm, idx_cold if arm[4] == "cold" else idx_warm, warm_idx, pool)
            except Exception as exc:  # noqa: BLE001
                r, outs = {"arm": name, "error": f"{type(exc).__name__}: {exc}"}, []
            r["repeat"] = rep
            r["utc"] = time.strftime("%H:%M:%S", time.gmtime())
            if rep == 0 and outs:
                c, b = S.check(enc, L, arm, outs[: cfg["check_n"]], cache)
                checks["checked"] += c
                checks["mismatch"] += b
                r["check"] = {"checked": c, "mismatch": b}
            res["runs"].append(r)
            if "error" in r:
                print(f"[{tag} {time.perf_counter() - t_start:.0f}s] {name} r{rep}: ERROR {r['error']}", flush=True)
            else:
                print(f"[{tag} {time.perf_counter() - t_start:.0f}s] {name} r{rep}: {r['samples_per_s']:.2f} sps, fetch/smp "
                      f"{r['fetches_per_sample']:.2f}, {r['http_bytes_per_sample'] / 1e6:.3f} MB/smp, p50 {r['lat_p50_ms']:.0f} ms, "
                      f"check {r.get('check')}", flush=True)
        write_json(D / "access.json", res)
        memlog(out, f"{tag}_rep{rep}")
    pool.shutdown()
    cache.clear()
    res["checks"] = checks
    res["pts_violations"] = sum(int(r.get("pts_err_max_s", 0) > 1e-4) for r in res["runs"])
    res["summary"] = S.summarize(res["runs"])
    if conc and cfg["conc"] > 1:
        best = []
        for lay in lays:
            for rg in ("warm", "cold"):
                cand = [v for v in res["summary"].values() if v["layout"] == lay and v["regime"] == rg]
                if cand:
                    best.append(max(cand, key=lambda v: v["sps"])["arm"])
        res["config"]["conc_arms"] = best
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        for name in best:
            if time.perf_counter() - t_start > cfg["budget_s"] + 1200:
                res["skipped"].append({"conc": name, "why": "budget"})
                continue
            start_at = time.time() + (40 if name.endswith("warm") else 20)
            args = [{"prep": str(prep_path), "arm": name.split("|"), "seed": 80_000 + w, "start_at": start_at, "dur_s": cfg["conc_s"]}
                    for w in range(cfg["conc"])]
            try:
                with ctx.Pool(cfg["conc"]) as p:
                    outs = p.map(conc_worker3, args)
            except Exception as exc:  # noqa: BLE001
                res["conc"].append({"arm": name, "error": f"{type(exc).__name__}: {exc}"})
                continue
            n = sum(o["n"] for o in outs)
            wall = max(o["wall_s"] for o in outs)
            tot = {k: sum(o["d"][k] for o in outs) for k in outs[0]["d"]}
            lat = [x for o in outs for x in o["lat_ms"]]
            c = {"arm": name, "procs": len(outs), "n": n, "wall_s": wall, "sps": n / wall, "cdn_per_s": tot["cdn_requests"] / wall,
                 "resolves_per_s": tot["resolve_requests"] / wall, "mb_per_s": tot["http_bytes"] / wall / 1e6, "lat_p50_ms": pct(lat, 0.5),
                 "lat_p95_ms": pct(lat, 0.95), "status_429": tot["status_429"], "status_other_err": tot["status_other_err"], "totals": tot,
                 "utc": time.strftime("%H:%M:%S", time.gmtime())}
            res["conc"].append(c)
            print(f"[{tag}] conc{cfg['conc']} {name}: {c['sps']:.1f} sps, {c['cdn_per_s']:.1f} fetch/s, {c['mb_per_s']:.0f} MB/s, "
                  f"p50 {c['lat_p50_ms']:.0f} ms, 429 {c['status_429']}", flush=True)
            write_json(D / "access.json", res)
    res["remote_totals"] = dict(REMOTE)
    res["reader_stats"] = dict(S.RD.STATS)
    res["rss"] = proc_status()
    write_json(D / "access.json", res)
    S.write_summary(D, res)
    return res


def main():
    pdir, out = Path(sys.argv[1]), Path(sys.argv[2])
    cfg = {"n_cold": int(os.environ.get("ACC_N_COLD", "16")), "n_warm": int(os.environ.get("ACC_N_WARM", "30")),
           "repeats": int(os.environ.get("ACC_REPEATS", "3")), "conc": int(os.environ.get("ACC_CONC", str(cpu_quota()))),
           "conc_s": float(os.environ.get("ACC_CONC_S", "45")), "budget_s": float(os.environ.get("ACC_BUDGET_S", "3600")),
           "check_n": int(os.environ.get("ACC_CHECK_N", "2"))}
    install_http_spy()
    t0 = time.perf_counter()
    a = run_set("access_eq", pdir / "prep_eq.json", ARMS_EQ, out, cfg, t0)
    ok = a["checks"]["mismatch"] == 0 and a["pts_violations"] == 0
    if os.environ.get("R3_CONTROL", "1") == "1":
        b = run_set("access_orig", pdir / "prep_orig.json", ARMS_ORIG, out, {**cfg, "budget_s": cfg["budget_s"] + 900}, t0, conc=False)
        ok = ok and b["checks"]["mismatch"] == 0 and b["pts_violations"] == 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
