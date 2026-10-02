"""PR #3917 sidecar header probe, round 3: option A (read the file start and the file end at the same time).

Arms (all on the real sidecar build path, EpisodeVideoManifest._build_file_records, native-http, 16 workers):
  BASE     head 6d945985: 4 MiB first read, the prefix doubles from byte 0, then a tail read
  PATCH    head + patches-3917/probe-exact-reads.diff: 512 KiB first read, exact follow-up read, tail read
  A1M      PATCH, plus a parallel read of the last 1 MiB that starts with the first read. Later reads that fall
           inside the last 1 MiB come from that buffer. The tail read is always waited for (a real implementation
           could cancel it, so this is pessimistic for A).
  A3M      as A1M with the last 3 MiB (covers the largest moov seen in the survey, 2.2 MiB)
  A1M-c32  A1M with 32 HTTP connections in place of max(8, workers) = 16, so the extra tail reads do not queue

Option A changes only which bytes come over the network, not the parser: the index is the same by construction.
The equality stage checks this on real files (BASE, PATCH and A1M on the same files).

Subcommands:
  probe-run    (inside a variant venv) index a list of files once; per-file seconds, requests, bytes, round trips
  orchestrate  (main venv) equality + interleaved speed runs for each dataset in ab3-files.json; writes ab3.json

Env: VENV_BASE, VENV_PROBE, AB3_JOB (0..n-1), AB3_REPS (2), AB3_N (64), AB3_EQ_N (8), AB3_PIN (0-7), AB3_DATASETS
"""

import argparse
import functools
import hashlib
import json
import os
import random
import statistics as st
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIB, MIB = 1024, 1024 * 1024
# name, venv, tail bytes (0 = no parallel tail read), HTTP connections (0 = default)
ARMS = [("BASE", "base", 0, 0), ("PATCH", "probe", 0, 0), ("A1M", "probe", MIB, 0), ("A3M", "probe", 3 * MIB, 0),
        ("A1M-c32", "probe", MIB, 32)]


def _index_digest(idx):
    h = hashlib.sha1()
    for k in ("file_size", "moov_offset", "mdat_offset", "mdat_payload_offset", "mdat_payload_size", "faststart", "codec",
              "timescale", "duration", "track_id", "width", "height"):
        h.update(repr(getattr(idx, k)).encode())
    h.update(idx.ftyp)
    h.update(idx.stsd_body)
    for k in ("sample_pts", "sample_durations", "sample_composition_offsets", "sample_sizes", "sample_offsets", "sync_samples"):
        arr = getattr(idx, k)
        h.update(f"{arr.dtype}{arr.shape}".encode())
        h.update(arr.tobytes())
    return h.hexdigest()


# ====================================================================== probe-run (inside a variant venv)
def cmd_probe_run(a):
    from lerobot.streaming import manifest as mf
    from lerobot.streaming import mp4

    if a.conns:
        mf.make_range_fetcher = functools.partial(mf.make_range_fetcher, native_http_connections=a.conns)
    files = json.loads(Path(a.files).read_text())
    per_file, lock = {}, threading.Lock()
    orig = mf.fetch_mp4_index
    tail_pool = ThreadPoolExecutor(a.workers)
    header = getattr(mp4, "DEFAULT_HEADER_PROBE_BYTES", 4 * MIB)  # the base code has no constant: 4 MiB literals

    def wrapped(path, read_range, *, file_size, **kw):
        reads, served = [], []
        tail_fut, tail_start = None, None
        t = time.perf_counter()
        if a.tail and file_size > kw["header_probe_bytes"]:
            tail_start = max(kw["header_probe_bytes"], file_size - a.tail)
            tail_fut = tail_pool.submit(read_range, path, tail_start, file_size - tail_start)

        def rr(p, off, ln):
            if tail_fut is not None and off >= tail_start and off + ln <= file_size:
                buf = tail_fut.result()
                served.append((off, ln))
                return buf[off - tail_start: off - tail_start + ln]
            b = read_range(p, off, ln)
            reads.append((off, len(b)))
            return b

        idx = orig(path, rr, file_size=file_size, **kw)
        tail_bytes = len(tail_fut.result()) if tail_fut is not None else 0
        with lock:
            per_file[path] = {"s": time.perf_counter() - t, "round_trips": len(reads),
                              "requests": len(reads) + (tail_fut is not None),
                              "bytes": sum(n for _, n in reads) + tail_bytes, "tail_hit": bool(served),
                              "faststart": bool(idx.faststart), "digest": _index_digest(idx)}
        return idx

    mf.fetch_mp4_index = wrapped
    root = f"hf://datasets/{a.repo}@{a.rev}"
    res = {"arm": a.arm, "repo": a.repo, "header": header, "tail": a.tail, "conns": a.conns, "workers": a.workers,
           "files": len(files)}
    t = time.perf_counter()
    try:
        mf.EpisodeVideoManifest._build_file_records(files, root, range_backend="native-http", workers=a.workers,
                                                    header_probe_bytes=header, max_probe_bytes=64 * MIB, token=None)
        res["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        res["status"] = "error"
        res["error"] = f"{type(e).__name__}: {e}"
    res["wall_s"] = time.perf_counter() - t
    res["per_file"] = per_file
    Path(a.out).write_text(json.dumps(res))
    print(f"probe-run {a.repo} {a.arm} {len(files)} files {res['status']} {res['wall_s']:.1f}s", flush=True)


# ====================================================================== orchestrate (main venv)
def cmd_orchestrate(a):
    out = Path(a.out) / "ab3"
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    job = int(os.environ.get("AB3_JOB", "0"))
    reps = int(os.environ.get("AB3_REPS", "2"))
    n_run = int(os.environ.get("AB3_N", "64"))
    eq_n = int(os.environ.get("AB3_EQ_N", "8"))
    pin = os.environ.get("AB3_PIN", "0-7")
    venvs = {"base": os.environ["VENV_BASE"], "probe": os.environ["VENV_PROBE"]}
    catalog = json.loads((HERE / "ab3-files.json").read_text())
    only = os.environ.get("AB3_DATASETS")
    repos = only.split() if only else list(catalog)
    res = {"job": job, "reps": reps, "n_run": n_run, "arms": [x[0] for x in ARMS], "equality": {}, "runs": []}

    def save():
        (out / "ab3.json").write_text(json.dumps(res, indent=1))

    def probe(repo, arm, files, tag):
        name, venv, tail, conns = next(x for x in ARMS if x[0] == arm)
        fl = runs_dir / f"{tag}.files.json"
        fl.write_text(json.dumps(files))
        o = runs_dir / f"{tag}.json"
        cmd = ["taskset", "-c", pin, f"{venvs[venv]}/bin/python", str(HERE / "ab3_3917.py"), "probe-run", "--repo", repo,
               "--rev", catalog[repo]["revision"], "--arm", arm, "--files", str(fl), "--tail", str(tail),
               "--conns", str(conns), "--out", str(o)]
        try:
            subprocess.run(cmd, check=False, timeout=1200)
        except subprocess.TimeoutExpired:
            return {"arm": arm, "status": "timeout"}
        return json.loads(o.read_text()) if o.exists() else {"arm": arm, "status": "no-output"}

    for di, repo in enumerate(repos):
        paths = [p for p, _ in catalog[repo]["files"]]
        rng = random.Random(f"ab3:{job}:{repo}")
        perm = paths[:]
        rng.shuffle(perm)
        # Shrink the run size so that each (rep, arm) gets new files inside this job, down to 16 files per run.
        fit = (len(perm) - min(eq_n, len(perm))) // (reps * len(ARMS))
        n = min(n_run, fit) if fit >= 16 else min(n_run, len(perm))
        # Equality: the same files in BASE, PATCH and A1M.
        eq_files = perm[:min(eq_n, len(perm))]
        digests = {}
        for arm in ("PATCH", "A1M", "BASE"):
            r = probe(repo, arm, eq_files, f"eq-{di}-{arm}")
            digests[arm] = {p: f["digest"] for p, f in r.get("per_file", {}).items()}
        same = sum(len({digests[arm].get(p) for arm in digests} - {None}) == 1 and all(p in digests[arm] for arm in digests)
                   for p in eq_files)
        res["equality"][repo] = {"files": len(eq_files), "same": same}
        save()
        # Speed: disjoint files for each (rep, arm) inside this job where the dataset is large enough.
        offset = len(eq_files) if len(perm) >= len(eq_files) + reps * len(ARMS) * n else 0
        for rep in range(reps):
            k0 = (rep + job + di) % len(ARMS)
            for k in range(len(ARMS)):
                arm = ARMS[(k0 + k) % len(ARMS)][0]
                start = offset + ((rep * len(ARMS) + k) * n) % max(1, len(perm) - offset)
                files = [perm[(start + i) % len(perm)] for i in range(n)]
                if len(perm) < len(eq_files) + reps * len(ARMS) * n:
                    files = list(dict.fromkeys(files))
                r = probe(repo, arm, files, f"sp-{di}-r{rep}-{arm}")
                pf = list(r.get("per_file", {}).values())
                row = {"repo": repo, "rep": rep, "order": k, "arm": arm, "status": r.get("status"), "error": r.get("error"),
                       "files": len(files), "wall_s": r.get("wall_s")}
                if pf:
                    row.update({
                        "files_per_s": len(pf) / r["wall_s"],
                        "kib_per_file": st.mean(f["bytes"] for f in pf) / KIB,
                        "requests_per_file": st.mean(f["requests"] for f in pf),
                        "round_trips_per_file": st.mean(f["round_trips"] for f in pf),
                        "tail_hit": sum(f["tail_hit"] for f in pf),
                        "faststart": sum(f["faststart"] for f in pf),
                        "file_s_median": st.median(f["s"] for f in pf),
                    })
                res["runs"].append(row)
                print(json.dumps(row), flush=True)
                save()
    save()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe-run")
    p.add_argument("--repo", required=True)
    p.add_argument("--rev", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--files", required=True)
    p.add_argument("--tail", type=int, default=0)
    p.add_argument("--conns", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", required=True)
    o = sub.add_parser("orchestrate")
    o.add_argument("out")
    a = ap.parse_args()
    {"probe-run": cmd_probe_run, "orchestrate": cmd_orchestrate}[a.cmd](a)


if __name__ == "__main__":
    main()
