"""Shared helpers for the DATA-11 benchmark (no torchcodec / PyAV import here).

Layouts, built from the SAME decoded source pixels with the SAME encoder settings:
  SEP    one 224x224 file per camera per group
  STACK  one 224x672 file per group, cameras stacked vertically in selection["cams"] order
  BIG    per camera, stream-copy concat of 3 consecutive SEP groups (3x frames per file)

Paths (under $WORK/enc):  <family>/SEP/<group>__<cam>.mp4   <family>/STACK/<group>.mp4
                          <family>/BIG/<family>_b<j>__<cam>.mp4
"""

import bisect
import ctypes
import ctypes.util
import json
import mmap
import os
import resource
import struct
import sys
import time
from pathlib import Path

H = W = 224
LAYOUTS = ["SEP", "STACK", "BIG"]


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_list(name, default):
    return os.environ.get(name, default).split()


def load_selection(path, smoke):
    sel = json.loads(Path(path).read_text())
    fams = sel["families"]
    if smoke:
        fams = {k: fams[k][: sel["smoke"][k]] for k in sel["smoke"]}
    for k, gs in fams.items():
        if k == "thrash" or k.startswith("ckpt"):
            assert len(gs) % 3 == 0, (k, len(gs))
    sel["active"] = fams
    return sel


class Family:
    """Index math for one family of groups in one layout."""

    def __init__(self, enc_root, fam, groups, cams, fps):
        self.fam, self.groups, self.cams, self.fps = fam, groups, cams, fps
        self.root = Path(enc_root) / fam
        self.n = [g["frames"] for g in groups]
        self.cum = [0]
        for n in self.n:
            self.cum.append(self.cum[-1] + n)
        self.total = self.cum[-1]

    def sep(self, gi, cam):
        return self.root / "SEP" / f"{self.groups[gi]['name']}__{cam}.mp4"

    def stack(self, gi):
        return self.root / "STACK" / f"{self.groups[gi]['name']}.mp4"

    def big(self, bj, cam):
        return self.root / "BIG" / f"{self.fam}_b{bj:03d}__{cam}.mp4"

    def n_big(self):
        return len(self.groups) // 3

    def big_frames(self, bj):
        return sum(self.n[3 * bj : 3 * bj + 3])

    def locate(self, idx):
        gi = bisect.bisect_right(self.cum, idx) - 1
        return gi, idx - self.cum[gi]

    def plan(self, layout, idx, cams):
        """-> list of (path, frame_index_in_file, crop_row or None) for the requested cams.
        STACK returns one entry per cam with the same path (decoded once by the caller)."""
        gi, f = self.locate(idx)
        if layout == "SEP":
            return [(str(self.sep(gi, c)), f, None) for c in cams]
        if layout == "STACK":
            return [(str(self.stack(gi)), f, self.cams.index(c)) for c in cams]
        if layout == "BIG":
            bj = gi // 3
            off = self.cum[gi] - self.cum[3 * bj]
            return [(str(self.big(bj, c)), off + f, None) for c in cams]
        raise ValueError(layout)

    def files(self, layout):
        if layout == "SEP":
            return [self.sep(g, c) for g in range(len(self.groups)) for c in self.cams]
        if layout == "STACK":
            return [self.stack(g) for g in range(len(self.groups))]
        return [self.big(b, c) for b in range(self.n_big()) for c in self.cams]

    def frames_per_file(self, layout):
        if layout == "BIG":
            return [self.big_frames(b) for b in range(self.n_big())]
        return list(self.n)


def mp4_boxes(path):
    """Top-level MP4 boxes: [(type, offset, size)]."""
    out = []
    with open(path, "rb") as f:
        off = 0
        end = os.fstat(f.fileno()).st_size
        while off < end:
            f.seek(off)
            hdr = f.read(16)
            if len(hdr) < 8:
                break
            size, typ = struct.unpack(">I4s", hdr[:8])
            if size == 1:
                size = struct.unpack(">Q", hdr[8:16])[0]
            elif size == 0:
                size = end - off
            out.append((typ.decode("latin1"), off, size))
            off += size
    return out


def moov_info(path):
    b = mp4_boxes(path)
    moov = [x for x in b if x[0] == "moov"]
    mdat = [x for x in b if x[0] == "mdat"]
    return {"file_bytes": os.path.getsize(path), "moov_bytes": moov[0][2] if moov else None,
            "moov_before_mdat": bool(moov and mdat and moov[0][1] < mdat[0][1]), "boxes": [x[0] for x in b]}


# ------------------------------------------------------------------ page cache control
_libc = ctypes.CDLL(ctypes.util.find_library("c") or None, use_errno=True)


def evict(paths):
    """posix_fadvise(DONTNEED) every file. Unprivileged; drops clean cached pages of these files."""
    n = 0
    if not hasattr(os, "posix_fadvise"):
        return 0
    for p in paths:
        try:
            fd = os.open(str(p), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            finally:
                os.close(fd)
        except FileNotFoundError:
            pass
    return n


def resident_fraction(path):
    """Fraction of a file's pages in the page cache: libc mmap(PROT_READ) + mincore + munmap.
    Returns nan if the calls fail."""
    size = os.path.getsize(path)
    if size == 0:
        return 1.0
    page = mmap.PAGESIZE
    npages = (size + page - 1) // page
    _libc.mmap.restype = ctypes.c_void_p
    _libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    _libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    fd = os.open(str(path), os.O_RDONLY)
    try:
        addr = _libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            return float("nan")
        try:
            vec = (ctypes.c_ubyte * npages)()
            if _libc.mincore(addr, size, vec) != 0:
                return float("nan")
            return sum(v & 1 for v in vec) / npages
        finally:
            _libc.munmap(addr, size)
    finally:
        os.close(fd)


def resident_fraction_many(paths):
    tot = res = 0.0
    for p in paths:
        s = os.path.getsize(p)
        r = resident_fraction(p)
        tot += s
        res += s * r
    return res / tot if tot else float("nan")


def prewarm(paths):
    for p in paths:
        with open(p, "rb") as f:
            while f.read(1 << 22):
                pass


# ------------------------------------------------------------------ process / cgroup memory
def cgroup_mem():
    def rd(n):
        try:
            return int(Path(f"/sys/fs/cgroup/{n}").read_text().split()[0])
        except Exception:
            return -1
    return {"cgroup_current_mb": rd("memory.current") / 2**20, "cgroup_peak_mb": rd("memory.peak") / 2**20}


def proc_status():
    """VmRSS / RssAnon in MB (Linux); falls back to ru_maxrss."""
    out = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            k, _, v = line.partition(":")
            if k in ("VmRSS", "RssAnon", "RssFile", "VmHWM"):
                out[k] = int(v.split()[0]) / 1024
    except Exception:
        out["ru_maxrss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 1024)
    return out


def proc_io():
    """(rchar, read_bytes) of this process; (-1, -1) where unavailable."""
    try:
        d = dict(line.split(": ") for line in Path("/proc/self/io").read_text().splitlines())
        return int(d["rchar"]), int(d["read_bytes"])
    except Exception:
        return -1, -1


def memlog(out_dir, stage):
    p = Path(out_dir) / "logs" / "mem_stages.tsv"
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    m = cgroup_mem()
    with p.open("a") as f:
        if new:
            f.write("utc\tstage\tcgroup_current_mb\tcgroup_peak_mb\n")
        f.write(f"{time.strftime('%H:%M:%S', time.gmtime())}\t{stage}\t{m['cgroup_current_mb']:.0f}\t{m['cgroup_peak_mb']:.0f}\n")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}


def t975(df):
    if df <= 0:
        return float("inf")
    keys = sorted(T975)
    for k in keys:
        if df <= k:
            return T975[k]
    return 1.96


def ci_halfwidth(xs):
    n = len(xs)
    if n < 2:
        return float("inf")
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    return t975(n - 1) * sd / n**0.5


# ------------------------------------------------------------------ CPU quota and thread pools
def cpu_quota():
    """Usable CPUs: cgroup v2 cpu.max quota/period (ceil), else sched_getaffinity, else os.cpu_count."""
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if q != "max":
            return max(1, -(-int(q) // int(p)))
    except Exception:
        pass
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


def pin_threads(n):
    """Pin math-library thread pools of THIS process to n (env for children, torch for this one)."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = str(n)
    if "torch" not in sys.modules:
        return
    try:
        import torch

        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(n)
        except RuntimeError:
            pass  # already set once in this process
    except ImportError:
        pass


def thread_report():
    out = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CPU_QUOTA")}
    out["cpu_quota"] = cpu_quota()
    try:
        import torch

        out["torch_num_threads"] = torch.get_num_threads()
        out["torch_interop_threads"] = torch.get_num_interop_threads()
    except ImportError:
        pass
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("Threads:"):
                out["os_threads"] = int(line.split()[1])
    except Exception:
        pass
    return out


# ------------------------------------------------------------------ remote (hf://) instrumentation
REMOTE = {"requests": 0, "resolve_requests": 0, "cdn_requests": 0, "api_requests": 0, "http_bytes": 0, "http_s": 0.0,
          "status_429": 0, "status_other_err": 0, "cdn_hit": 0, "cdn_miss": 0, "fetch_calls": 0, "fetch_bytes": 0,
          "fetch_s": 0.0, "resolver_remaining": -1, "resolver_reset_s": -1, "api_remaining": -1, "api_reset_s": -1,
          "rl_seen_utc": 0.0, "resolver_min_seen": 1 << 30}
_SPY = {"on": False}


def _parse_rl(h):
    """'"resolvers";r=11999;t=292' -> ("resolvers", 11999, 292)"""
    try:
        name, *kv = [x.strip() for x in h.split(";")]
        d = dict(x.split("=") for x in kv)
        return name.strip('"'), int(d.get("r", -1)), int(d.get("t", -1))
    except Exception:
        return None, -1, -1


def install_http_spy():
    """Count every HTTP request huggingface_hub sends (httpx), by kind; record ratelimit headers.
    Also time HfFileSystemFile._fetch_range (one fsspec block fetch = 1 resolve + 1 CDN request)."""
    if _SPY["on"]:
        return
    import threading

    import httpx
    from huggingface_hub import hf_file_system

    orig = httpx.Client._send_single_request
    lock = threading.Lock()  # the parallel-camera arm sends requests from several threads

    def spy(self, request):
        t0 = time.perf_counter()
        r = orig(self, request)
        with lock:
            _count(request, r, time.perf_counter() - t0)
        return r

    def _count(request, r, dt):
        REMOTE["http_s"] += dt
        REMOTE["requests"] += 1
        host = request.url.host or ""
        if "/resolve/" in request.url.path:
            REMOTE["resolve_requests"] += 1
        elif "cdn" in host or "xet" in host or "cas" in host:
            REMOTE["cdn_requests"] += 1
            xc = r.headers.get("x-cache", "")
            if "Hit" in xc:
                REMOTE["cdn_hit"] += 1
            elif "Miss" in xc:
                REMOTE["cdn_miss"] += 1
        else:
            REMOTE["api_requests"] += 1
        if r.status_code == 429:
            REMOTE["status_429"] += 1
        elif r.status_code >= 400:
            REMOTE["status_other_err"] += 1
        try:
            REMOTE["http_bytes"] += int(r.headers.get("content-length", 0))
        except ValueError:
            pass
        rl = r.headers.get("ratelimit")
        if rl:
            name, rem, reset = _parse_rl(rl)
            if name == "resolvers":
                REMOTE["resolver_remaining"], REMOTE["resolver_reset_s"] = rem, reset
                REMOTE["resolver_min_seen"] = min(REMOTE["resolver_min_seen"], rem)
            elif name == "api":
                REMOTE["api_remaining"], REMOTE["api_reset_s"] = rem, reset
            REMOTE["rl_seen_utc"] = time.time()

    httpx.Client._send_single_request = spy
    orig_fetch = hf_file_system.HfFileSystemFile._fetch_range

    def fetch(self, start, end):
        t0 = time.perf_counter()
        b = orig_fetch(self, start, end)
        with lock:
            REMOTE["fetch_s"] += time.perf_counter() - t0
            REMOTE["fetch_calls"] += 1
            REMOTE["fetch_bytes"] += len(b)
        return b

    hf_file_system.HfFileSystemFile._fetch_range = fetch
    _SPY["on"] = True


class CountingFile:
    """File-like proxy (what VideoDecoder sees) counting read/seek calls and bytes returned."""

    def __init__(self, fh, c):
        self._fh, self._c = fh, c

    def read(self, n=-1):
        b = self._fh.read(n)
        self._c["reads"] += 1
        self._c["read_bytes"] += len(b)
        return b

    def seek(self, off, whence=0):
        self._c["seeks"] += 1
        return self._fh.seek(off, whence)

    def tell(self):
        return self._fh.tell()

    def close(self):
        return self._fh.close()


def remote_url(base, enc_root, path):
    return f"{base}/{os.path.relpath(str(path), str(enc_root))}"


def resolver_budget(repo, rev, probe_path):
    """One resolve (costs 1 of the budget) -> (remaining, reset_s, limit, window_s) from the ratelimit headers."""
    import httpx
    from huggingface_hub import HfApi, hf_hub_url

    api = HfApi()
    r = httpx.get(hf_hub_url(repo, probe_path, repo_type="dataset", revision=rev), headers={**api._build_hf_headers(), "range": "bytes=0-0"},
                  follow_redirects=False, timeout=30)
    _, rem, reset = _parse_rl(r.headers.get("ratelimit", ""))
    pol = r.headers.get("ratelimit-policy", "")
    q = w = -1
    for part in pol.split(";"):
        part = part.strip()
        if part.startswith("q="):
            q = int(part[2:])
        if part.startswith("w="):
            w = int(part[2:])
    return rem, reset, q, w
