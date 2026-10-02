"""Round-2 readers for hf:// video files (no torchcodec import here).

Every reader gives TorchCodec a file-like object (read / seek / tell). Transports:
  fsspec     HfFileSystem().open(url, "rb", block_size=B): readahead cache, every block fetch = 1 resolve + 1 CDN request
  rc         resolve once per file (signed CDN URL, cached until it expires), then plain HTTP Range GETs:
             every fetch = 1 CDN request. Readahead semantics copied from fsspec (fetch [pos, pos + n + B)).
  exact      like LeRobot's Lance backend (lance_backend.py _prepare_files / _window_byte_range, lance_utils.py
             _SparseBlobSource): read the moov once, compute each frame's keyframe byte window from the sample
             tables, fetch only those spans (64 KiB slack), serve the decoder from the fetched bytes; any read
             outside the spans is a counted fallback fetch.
  shared     one block cache per file shared by several decoders (multi-track MP4: one decoder per track).

All HTTP goes through httpx, so common.install_http_spy() counts it (resolve vs CDN requests, bytes).
"""

import bisect
import io
import os
import struct
import threading
import time
from collections import OrderedDict

KiB, MiB = 1 << 10, 1 << 20
OPEN_PROBE = 256 * KiB  # lance_utils._OPEN_PROBE_BYTES
SLACK = 64 * KiB  # lance_utils._RANGE_SLACK

STATS = {"rc_fetches": 0, "rc_bytes": 0, "rc_resolves": 0, "exact_span_fetches": 0, "exact_span_bytes": 0,
         "exact_fallback_fetches": 0, "exact_fallback_bytes": 0, "shared_fetches": 0, "shared_bytes": 0}
_LOCK = threading.Lock()


def _add(**kw):
    with _LOCK:
        for k, v in kw.items():
            STATS[k] += v


# ------------------------------------------------------------------ transports
class Resolver:
    """hf repo path -> signed CDN URL. One resolve request (no redirect follow) per file per TTL."""

    def __init__(self, repo, rev, repo_type="dataset", ttl_s=1800):
        import httpx
        from huggingface_hub import HfApi, hf_hub_url

        self._url = lambda p: hf_hub_url(repo, p, repo_type=repo_type, revision=rev)
        self._hdr = HfApi()._build_hf_headers()
        self.client = httpx.Client(timeout=60, follow_redirects=False, limits=httpx.Limits(max_connections=64, max_keepalive_connections=64))
        self.ttl_s = ttl_s
        self._cache = {}
        self._lock = threading.Lock()

    def signed(self, path, force=False):
        with self._lock:
            hit = self._cache.get(path)
            if hit and not force and time.time() - hit[1] < self.ttl_s:
                return hit[0], hit[2]
        r = self.client.get(self._url(path), headers={**self._hdr, "range": "bytes=0-0"})
        _add(rc_resolves=1)
        if r.status_code in (301, 302, 303, 307, 308):
            loc = r.headers["location"]
            size = None
        elif r.status_code in (200, 206):  # served directly (small / non-LFS file): no signed URL
            loc = str(r.request.url)
            size = None
        else:
            raise OSError(f"resolve {path}: HTTP {r.status_code}")
        # size from the resolve response when present (x-linked-size), else from a 1-byte range on the CDN
        size = int(r.headers.get("x-linked-size") or 0) or None
        with self._lock:
            self._cache[path] = (loc, time.time(), size)
        return loc, size


class RangeFetcher:
    """fetch(start, end) -> bytes of [start, end) from one file, via the cached signed URL (1 CDN request)."""

    def __init__(self, resolver, path, size=None):
        self.resolver, self.path = resolver, path
        self.url, sz = resolver.signed(path)
        self.size = size or sz or self._probe_size()

    def _probe_size(self):
        r = self.resolver.client.get(self.url, headers={"range": "bytes=0-0"})
        cr = r.headers.get("content-range", "")
        return int(cr.rsplit("/", 1)[1])

    def fetch(self, start, end):
        end = min(end, self.size)
        if end <= start:
            return b""
        for attempt in range(3):
            r = self.resolver.client.get(self.url, headers={"range": f"bytes={start}-{end - 1}"})
            if r.status_code == 206 or (r.status_code == 200 and start == 0):
                b = r.content[: end - start]
                _add(rc_fetches=1, rc_bytes=len(b))
                return b
            if r.status_code in (401, 403) and attempt == 0:  # signed URL expired: resolve again
                self.url, _ = self.resolver.signed(self.path, force=True)
                continue
            time.sleep(0.5 * (attempt + 1))
        raise OSError(f"range fetch {self.path} [{start},{end}): HTTP {r.status_code}")


class HfRawFetcher:
    """fetch(start, end) through HfFileSystemFile._fetch_range (1 resolve + 1 CDN request, what fsspec does)."""

    def __init__(self, fs, url):
        self.fh = fs.open(url, "rb", block_size=5 * MiB, cache_type="none")  # block_size=0 would give a size-less stream file
        self.size = self.fh.size

    def fetch(self, start, end):
        end = min(end, self.size)
        return self.fh._fetch_range(start, end) if end > start else b""

    def close(self):
        self.fh.close()


# ------------------------------------------------------------------ file-like objects
class ReadaheadFile:
    """fsspec ReadAheadCache semantics over a fetcher: one contiguous cached region; a miss fetches [pos, pos+n+B)."""

    def __init__(self, fetcher, block):
        self.f, self.block, self.size = fetcher, block, fetcher.size
        self.pos, self.c0, self.cache = 0, 0, b""

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        s, e = self.pos, self.pos + n
        if self.c0 <= s and e <= self.c0 + len(self.cache):
            out = self.cache[s - self.c0 : e - self.c0]
        else:
            self.cache = self.f.fetch(s, min(self.size, e + self.block))
            self.c0 = s
            out = self.cache[:n]
        self.pos += len(out)
        return out

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def close(self):
        self.cache = b""


class SharedBlocks:
    """Aligned-block LRU shared by several readers of ONE file (multi-track MP4). Missing consecutive blocks of one
    read are fetched in one request; concurrent readers wait on an in-flight block instead of fetching it twice."""

    def __init__(self, fetcher, block, capacity_blocks=16):
        self.f, self.block, self.size, self.cap = fetcher, block, fetcher.size, capacity_blocks
        self.blocks = OrderedDict()
        self.inflight = {}
        self.lock = threading.Lock()

    def read_at(self, s, n):
        e = min(self.size, s + n)
        if e <= s:
            return b""
        b0, b1 = s // self.block, (e - 1) // self.block
        while True:
            with self.lock:
                missing = [b for b in range(b0, b1 + 1) if b not in self.blocks]
                waits = [self.inflight[b] for b in missing if b in self.inflight]
                todo = [b for b in missing if b not in self.inflight]
                ev = None
                if todo:
                    ev = threading.Event()
                    for b in todo:
                        self.inflight[b] = ev
            if ev is not None:
                runs = []
                for b in todo:
                    if runs and b == runs[-1][1] + 1:
                        runs[-1][1] = b
                    else:
                        runs.append([b, b])
                got = {}
                try:
                    for lo, hi in runs:
                        data = self.f.fetch(lo * self.block, min(self.size, (hi + 1) * self.block))
                        _add(shared_fetches=1, shared_bytes=len(data))
                        for b in range(lo, hi + 1):
                            got[b] = data[(b - lo) * self.block : (b - lo + 1) * self.block]
                finally:
                    with self.lock:
                        for b, d in got.items():
                            self.blocks[b] = d
                        for b in todo:
                            self.inflight.pop(b, None)
                        while len(self.blocks) > self.cap:
                            self.blocks.popitem(last=False)
                    ev.set()
            for w in waits:
                w.wait()
            with self.lock:
                if all(b in self.blocks for b in range(b0, b1 + 1)):
                    for b in range(b0, b1 + 1):
                        self.blocks.move_to_end(b)
                    parts = [self.blocks[b] for b in range(b0, b1 + 1)]
                    break
        buf = b"".join(parts)
        off = s - b0 * self.block
        return buf[off : off + (e - s)]


class SparseStore:
    """Fetched byte spans of one file (lance_utils._SparseBlobSource storage), shared by any number of views.
    Reads outside the spans go to the fetcher (counted as fallback)."""

    def __init__(self, fetcher):
        self.f, self.size = fetcher, fetcher.size
        self.starts, self.chunks = [], []
        self.lock = threading.Lock()
        self.buffered = 0

    def add(self, offset, data):
        end = offset + len(data)
        lo = bisect.bisect_left(self.starts, offset)
        if lo > 0 and self.starts[lo - 1] + len(self.chunks[lo - 1]) >= offset:
            lo -= 1
        hi = bisect.bisect_right(self.starts, end)
        if lo == hi:
            self.starts.insert(lo, offset)
            self.chunks.insert(lo, data)
        else:
            ms = min(offset, self.starts[lo])
            me = max(end, max(self.starts[i] + len(self.chunks[i]) for i in range(lo, hi)))
            m = bytearray(me - ms)
            for i in range(lo, hi):
                a = self.starts[i] - ms
                m[a : a + len(self.chunks[i])] = self.chunks[i]
            m[offset - ms : offset - ms + len(data)] = data
            del self.starts[lo:hi]
            del self.chunks[lo:hi]
            self.starts.insert(lo, ms)
            self.chunks.insert(lo, bytes(m))
        self.buffered = sum(len(c) for c in self.chunks)

    def covers(self, s, e):
        i = bisect.bisect_right(self.starts, s) - 1
        return i >= 0 and self.starts[i] + len(self.chunks[i]) >= e

    def fetch_spans(self, spans):
        """spans [(s, e)] -> merge (64 KiB gap), skip covered, fetch each merged span with one request."""
        merged = []
        for s, e in sorted((max(0, s), min(self.size, e)) for s, e in spans):
            if e <= s:
                continue
            if merged and s <= merged[-1][1] + SLACK:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        n = 0
        for s, e in merged:
            with self.lock:
                if self.covers(s, e):
                    continue
                i = bisect.bisect_right(self.starts, s) - 1  # skip an already-held prefix
                if i >= 0 and self.starts[i] + len(self.chunks[i]) > s:
                    s = self.starts[i] + len(self.chunks[i])
            data = self.f.fetch(s, e)
            _add(exact_span_fetches=1, exact_span_bytes=len(data))
            n += 1
            with self.lock:
                self.add(s, data)
        return n

    def read_at(self, pos, want):
        with self.lock:
            i = bisect.bisect_right(self.starts, pos) - 1
            if i >= 0:
                inside = pos - self.starts[i]
                if inside < len(self.chunks[i]):
                    return self.chunks[i][inside : inside + want]
            j = bisect.bisect_right(self.starts, pos)
            if j < len(self.starts):
                want = min(want, self.starts[j] - pos)
        data = self.f.fetch(pos, pos + want)
        _add(exact_fallback_fetches=1, exact_fallback_bytes=len(data))
        with self.lock:
            self.add(pos, data)
        return data


class View:
    """File-like view with its own position over a shared store (SparseStore or SharedBlocks)."""

    def __init__(self, store):
        self.st, self.size, self.pos = store, store.size, 0

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        out = self.st.read_at(self.pos, n)
        self.pos += len(out)
        return out

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def close(self):
        pass


# ------------------------------------------------------------------ MP4 sample tables (from the moov only)
def _boxes(buf, off, end):
    while off + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[off : off + 8])
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", buf[off + 8 : off + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - off
        yield typ.decode("latin1"), off + hdr, off + size
        off += size


def top_boxes(read_at, size, stop=None):
    """[(type, offset, size)] of top-level boxes using 16-byte header reads (stops after box type `stop`)."""
    out, off = [], 0
    while off < size:
        h = read_at(off, 16)
        if len(h) < 8:
            break
        s, t = struct.unpack(">I4s", h[:8])
        if s == 1:
            s = struct.unpack(">Q", h[8:16])[0]
        elif s == 0:
            s = size - off
        out.append((t.decode("latin1"), off, s))
        if stop is not None and out[-1][0] == stop:
            break
        off += s
    return out


def parse_moov(moov):
    """moov box bytes (header included) -> {track_id: {"offsets", "sizes", "sync" (0-based sample idx), "timescale",
    "handler"}} for every track. Sample i in decode order; CFR without reordering assumed (checked by the caller)."""
    tracks = {}
    for t, a, e in _boxes(moov, 8, len(moov)):
        if t != "trak":
            continue
        tr = {"sync": None}
        stack = [(a, e)]
        while stack:
            a2, e2 = stack.pop()
            for t2, b, f in _boxes(moov, a2, e2):
                if t2 in ("mdia", "minf", "stbl"):
                    stack.append((b, f))
                elif t2 == "tkhd":
                    v = moov[b]
                    tr["id"] = struct.unpack(">I", moov[b + (20 if v == 1 else 12) : b + (24 if v == 1 else 16)])[0]
                elif t2 == "mdhd":
                    v = moov[b]
                    tr["timescale"] = struct.unpack(">I", moov[b + (20 if v == 1 else 12) : b + (24 if v == 1 else 16)])[0]
                elif t2 == "hdlr":
                    tr["handler"] = moov[b + 8 : b + 12].decode("latin1")
                elif t2 == "stsz":
                    fixed, n = struct.unpack(">II", moov[b + 4 : b + 12])
                    tr["sizes"] = [fixed] * n if fixed else list(struct.unpack(f">{n}I", moov[b + 12 : b + 12 + 4 * n]))
                elif t2 == "stco":
                    n = struct.unpack(">I", moov[b + 4 : b + 8])[0]
                    tr["chunks"] = list(struct.unpack(f">{n}I", moov[b + 8 : b + 8 + 4 * n]))
                elif t2 == "co64":
                    n = struct.unpack(">I", moov[b + 4 : b + 8])[0]
                    tr["chunks"] = list(struct.unpack(f">{n}Q", moov[b + 8 : b + 8 + 8 * n]))
                elif t2 == "stsc":
                    n = struct.unpack(">I", moov[b + 4 : b + 8])[0]
                    tr["stsc"] = [struct.unpack(">III", moov[b + 8 + 12 * i : b + 20 + 12 * i]) for i in range(n)]
                elif t2 == "stss":
                    n = struct.unpack(">I", moov[b + 4 : b + 8])[0]
                    tr["sync"] = [x - 1 for x in struct.unpack(f">{n}I", moov[b + 8 : b + 8 + 4 * n])]
                elif t2 == "ctts":
                    tr["has_ctts"] = True
        sizes, chunks, stsc = tr["sizes"], tr["chunks"], tr["stsc"]
        offs, si = [0] * len(sizes), 0
        for ci, co in enumerate(chunks):
            c1 = ci + 1
            k = 0
            while k + 1 < len(stsc) and stsc[k + 1][0] <= c1:
                k += 1
            o = co
            for _ in range(stsc[k][1]):
                if si >= len(sizes):
                    break
                offs[si] = o
                o += sizes[si]
                si += 1
        tr["offsets"] = offs
        if tr["sync"] is None:
            tr["sync"] = list(range(len(sizes)))
        del tr["chunks"], tr["stsc"]
        tracks[tr["id"]] = tr
    return tracks


class ExactFile:
    """Exact-range state of one remote MP4: store + moov-derived index. open() fetches the open spans
    (first 256 KiB, the moov + 64 KiB, the first packet + 256 KiB, as in lance_backend._prepare_files)."""

    def __init__(self, fetcher, hint=None):
        """hint = (moov_offset, moov_size, first_packet_offset) known in advance (Lance stores them in its videos table):
        the three open spans are then merged and fetched together (1 request when the moov is at the front)."""
        self.store = SparseStore(fetcher)
        size = fetcher.size
        head = min(OPEN_PROBE, size)
        if hint is not None:
            mo, ms, f0 = hint
            self.store.fetch_spans([(0, head), (mo, min(size, mo + ms + SLACK)), (f0, min(size, f0 + OPEN_PROBE))])
        else:
            self.store.fetch_spans([(0, head)])
            boxes = top_boxes(lambda o, n: self._peek(o, n), size, stop="moov")
            moov = [b for b in boxes if b[0] == "moov"]
            if not moov:
                raise ValueError("no moov")
            _, mo, ms = moov[0]
            self.store.fetch_spans([(mo, min(size, mo + ms + SLACK))])
        self.moov_offset, self.moov_size = mo, ms
        self.tracks = parse_moov(self.store.read_at(mo, ms))
        vids = [t for t in self.tracks.values() if t.get("handler") == "vide"]
        if hint is None:
            firsts = [t["offsets"][0] for t in vids if t["offsets"]]
            if firsts:
                f0 = min(firsts)
                self.store.fetch_spans([(f0, min(size, f0 + OPEN_PROBE))])
        self.video_tracks = sorted(t["id"] for t in vids)

    def _peek(self, o, n):
        """Header reads during the box walk: served from the store when covered, else one small fallback fetch."""
        return self.store.read_at(o, n)

    def window(self, track_pos, frame):
        """Byte span covering `frame` of the track_pos-th video track: preceding keyframe to the next keyframe + slack
        (lance_backend._window_byte_range)."""
        t = self.tracks[self.video_tracks[track_pos]]
        sync = t["sync"]
        i = max(bisect.bisect_right(sync, frame) - 1, 0)
        j = bisect.bisect_right(sync, frame)
        start = t["offsets"][sync[i]]
        end = t["offsets"][sync[j]] if j < len(sync) else self.store.size
        return start, min(end + SLACK, self.store.size)


def moov_hint(path):
    """(moov_offset, moov_size, first_packet_offset) of a local MP4 (what Lance precomputes at conversion)."""
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size

        def ra(o, n):
            f.seek(o)
            return f.read(n)

        boxes = top_boxes(ra, size, stop="moov")
        _, mo, ms = [b for b in boxes if b[0] == "moov"][0]
        tr = parse_moov(ra(mo, ms))
    f0 = min(t["offsets"][0] for t in tr.values() if t.get("handler") == "vide" and t["offsets"])
    return mo, ms, f0


def file_size_hint(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return None


class LocalFetcher:
    """For local tests: the same interface over a local file, counting like a remote fetch."""

    def __init__(self, path):
        self.path, self.size = path, os.path.getsize(path)
        self.calls = 0
        self.bytes = 0

    def fetch(self, s, e):
        with open(self.path, "rb") as f:
            f.seek(s)
            b = f.read(max(0, min(e, self.size) - s))
        self.calls += 1
        self.bytes += len(b)
        return b


def as_bytesio(b):
    return io.BytesIO(b)
