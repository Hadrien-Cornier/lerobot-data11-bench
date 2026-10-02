"""Round 3: does request latency grow with file size inside HF? (laptop: 80 MB ~182 ms, 210-265 MB 268-332 ms)

Files (all in the private bench repo): SEP 80 MB (ckpt_full g000 top), BIG 254 MB (same layout as SEP, 3x longer),
equal-size STACK cut (the one closest to 80 MB), plus controls: round-2 STACK 261 MB and an equal-size MULTI cut.
Signed CDN link resolved ONCE per file (1 resolve request, untimed), then SZ_N random 256 KiB range GETs per file,
interleaved (each round visits every file once, in a new random order). Per request: time to response headers,
time to first body byte (TTFB), total time, x-cache, status. One httpx client (keep-alive), one request at a time.

Usage: python r3_size.py <prep.json of r3_prep> <out_dir> [tag]
Env: SZ_N (40), SZ_BYTES (262144), SZ_SEED (0)
"""

import json
import os
import random
import sys
import time
from pathlib import Path

from common import pct, write_json


def main():
    prep_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "size"
    D = out / tag
    D.mkdir(parents=True, exist_ok=True)
    prep = json.loads(prep_path.read_text())
    eq = json.loads((prep_path.parent / "prep_eq.json").read_text())
    import httpx
    from huggingface_hub import HfApi, hf_hub_url

    repo, rev = eq["repo"], eq["multi"]["upload"]["revision"]
    fam = eq["family"]
    r1n = prep["r1_run"]
    sizes = eq["sizes"]
    stack_cuts = [r for r in prep["cuts"] if r["layout"] == "STACK"]
    multi_cuts = [r for r in prep["cuts"] if r["layout"] == "MULTI"]
    sep_rel = f"{r1n}/{fam}/SEP/{fam}_g000__top.mp4"
    pick = lambda rows: min(rows, key=lambda r: abs(r["bytes"] - sizes[sep_rel]))["rel"]  # noqa: E731
    files = {"SEP": sep_rel, "BIG": f"{r1n}/{fam}/BIG/{fam}_b000__top.mp4", "STACK_eq": pick(stack_cuts),
             "STACK_r2": f"{r1n}/{fam}/STACK/{fam}_g001.mp4", "MULTI_eq": pick(multi_cuts)}
    n = int(os.environ.get("SZ_N", "40"))
    nb = int(os.environ.get("SZ_BYTES", str(256 * 1024)))
    rng = random.Random(int(os.environ.get("SZ_SEED", "0")))
    hdr = HfApi()._build_hf_headers()
    cl = httpx.Client(timeout=60, follow_redirects=False)
    signed = {}
    t0 = time.time()
    for k, rel in files.items():
        r = cl.get(hf_hub_url(repo, f"variants/{rel}", repo_type="dataset", revision=rev), headers={**hdr, "range": "bytes=0-0"})
        if r.status_code not in (301, 302, 303, 307, 308):
            print(f"skip {k} {rel}: resolve HTTP {r.status_code}", flush=True)
            continue
        size = int(r.headers.get("x-linked-size") or 0)
        if not size:
            size = int(cl.get(r.headers["location"], headers={"range": "bytes=0-0"}).headers["content-range"].rsplit("/", 1)[1])
        signed[k] = (r.headers["location"], size)
    hosts = {k: httpx.URL(v[0]).host for k, v in signed.items()}
    # warm-up: one request per file (TLS, connection), not recorded
    for k, (url, size) in signed.items():
        cl.get(url, headers={"range": f"bytes=0-{nb - 1}"})
    rows = []
    for rnd in range(n):
        order = list(signed)
        rng.shuffle(order)
        for k in order:
            url, size = signed[k]
            off = rng.randrange(0, size - nb)
            ta = time.perf_counter()
            with cl.stream("GET", url, headers={"range": f"bytes={off}-{off + nb - 1}"}) as r:
                th = time.perf_counter()
                it = r.iter_raw()
                first = next(it, b"")
                tf = time.perf_counter()
                got = len(first) + sum(len(c) for c in it)
            te = time.perf_counter()
            rows.append({"file": k, "round": rnd, "offset": off, "status": r.status_code, "bytes": got, "headers_ms": (th - ta) * 1e3,
                         "ttfb_ms": (tf - ta) * 1e3, "total_ms": (te - ta) * 1e3, "x_cache": r.headers.get("x-cache", ""),
                         "utc": time.strftime("%H:%M:%S", time.gmtime())})
    summ = {}
    for k in signed:
        rs = [r for r in rows if r["file"] == k and r["status"] == 206]
        for m in ("headers_ms", "ttfb_ms", "total_ms"):
            xs = [r[m] for r in rs]
            summ.setdefault(k, {})[m] = {"p25": pct(xs, 0.25), "p50": pct(xs, 0.5), "p75": pct(xs, 0.75), "p90": pct(xs, 0.9)}
        summ[k].update(n=len(rs), bad=sum(1 for r in rows if r["file"] == k and r["status"] != 206), bytes=signed[k][1], rel=files[k],
                       host=hosts[k], x_cache={x: sum(1 for r in rs if r["x_cache"] == x) for x in {r["x_cache"] for r in rs}})
    res = {"files": files, "n_per_file": n, "range_bytes": nb, "summary": summ, "rows": rows, "start_utc": time.strftime("%FT%TZ", time.gmtime(t0)),
           "wall_s": time.time() - t0}
    write_json(D / "size.json", res)
    L = [f"# File size vs range-request latency inside HF ({tag})", "",
         f"Signed CDN link resolved once per file; {n} random {nb // 1024} KiB range GETs per file, interleaved, one at a time. "
         "TTFB = time from sending the request to the first body byte. ms.", "",
         "| file | MB | n | TTFB p50 | IQR (p25-p75) | headers p50 | total p50 | total p90 | x-cache |", "|---|---|---|---|---|---|---|---|---|"]
    for k, s in summ.items():
        t, h, to = s["ttfb_ms"], s["headers_ms"], s["total_ms"]
        L.append(f"| {k} (`{s['rel'].split('/')[-1]}`) | {s['bytes'] / 1e6:.0f} | {s['n']} | {t['p50']:.0f} | {t['p25']:.0f}-{t['p75']:.0f} | "
                 f"{h['p50']:.0f} | {to['p50']:.0f} | {to['p90']:.0f} | {s['x_cache']} |")
    L += ["", f"Hosts: {sorted(set(hosts.values()))}. Wall {res['wall_s']:.0f} s, start {res['start_utc']}."]
    (D / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
