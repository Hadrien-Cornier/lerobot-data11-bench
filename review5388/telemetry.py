"""Low-overhead observation for the single-process streaming comparison."""

from collections import Counter, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter, thread_time
from urllib.parse import unquote, urlsplit
import gzip
import json
import os
import re

_PURPOSE = ContextVar("io_purpose", default=None)
_FILE = ContextVar("io_file", default=None)
_STORAGE_PATH = re.compile(r"(?:^|/)((?:meta|data|videos)/.+\.(?:json|jsonl|parquet|mp4|npz))$")


@contextmanager
def io_context(purpose=None, path=None):
    purpose_token = _PURPOSE.set(purpose or _PURPOSE.get())
    file_token = _FILE.set(path or _FILE.get())
    try:
        yield
    finally:
        _FILE.reset(file_token)
        _PURPOSE.reset(purpose_token)


def source_path(url):
    if not isinstance(url, str):
        return None
    path = unquote(urlsplit(url.rsplit("::", 1)[-1]).path)
    match = _STORAGE_PATH.search(path)
    return match[1] if match else None


def file_purpose(path):
    if not path:
        return None
    if path.startswith("meta/mp4-sidecars/"):
        return "sidecar_download"
    if path.startswith("meta/"):
        return "metadata"
    if path.startswith("data/"):
        return "parquet"
    if path.startswith("videos/"):
        return "video"


class Trace:
    def __init__(self, path, t0):
        self.file = path.open("w", buffering=1)
        self.lock = Lock()
        self.t0 = t0

    def write(self, kind, **fields):
        event = {"event": kind, "elapsed_s": round(perf_counter() - self.t0, 6), **fields}
        with self.lock:
            self.file.write(json.dumps(event, separators=(",", ":")) + "\n")

    def close(self):
        self.file.close()


class IORecorder:
    """Count raw HTTP response bodies and record identifiable source files."""

    def __init__(self, trace):
        self.trace = trace
        self.lock = Lock()
        self.by_purpose = defaultdict(Counter)
        self.files = defaultdict(Counter)
        self.redirects = {}
        self.sidecar = {}
        self.pinned_video_opens = 0
        self.probing = False
        self.cache_stats = None
        self.stages = defaultdict(Counter)

    def touch(self, purpose, path, kind):
        if path:
            with self.lock:
                counter = self.files[purpose, path]
                if not counter:
                    self.by_purpose[purpose]["files_touched"] += 1
                counter[kind] += 1

    def response_source(self, response):
        path = source_path(str(response.request.url))
        purpose = _PURPOSE.get()
        if not path:
            path = _FILE.get()
        if not path:
            with self.lock:
                path = self.redirects.get(str(response.request.url))
        if self.probing and file_purpose(path) == "video":
            purpose = "sidecar_probe"
        purpose = purpose or file_purpose(path) or "unclassified_http"
        location = response.headers.get("location")
        if path and location and response.is_redirect:
            destination = str(response.request.url.join(location))
            with self.lock:
                self.redirects[destination] = path
        return purpose, path

    def received(self, response, body_bytes, purpose, path):
        with self.lock:
            totals = self.by_purpose[purpose]
            totals["http_responses"] += 1
            totals["http_body_bytes"] += body_bytes
            totals[f"status_{response.status_code}"] += 1
            if path:
                file = self.files[purpose, path]
                if not file:
                    totals["files_touched"] += 1
                file["http_responses"] += 1
                file["http_body_bytes"] += body_bytes

    @contextmanager
    def measure(self, stage):
        wall_start = perf_counter()
        cpu_start = thread_time()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            wall_s = perf_counter() - wall_start
            cpu_s = thread_time() - cpu_start
            with self.lock:
                metric = self.stages[stage]
                metric["calls"] += 1
                metric["errors"] += failed
                metric["wall_s_sum"] += wall_s
                metric["calling_thread_cpu_s_sum"] += cpu_s

    def wrap_stage(self, module, name, stage):
        original = getattr(module, name)

        @wraps(original)
        def measured(*args, **kwargs):
            with self.measure(stage):
                return original(*args, **kwargs)

        setattr(module, name, measured)

    def reset(self):
        with self.lock:
            self.by_purpose.clear()
            self.stages.clear()
            self.files.clear()
            self.sidecar.clear()
            self.pinned_video_opens = 0

    def snapshot(self):
        cache = None if self.cache_stats is None else self.cache_stats()
        with self.lock:
            return {
                "by_purpose": {name: dict(counts) for name, counts in self.by_purpose.items()},
                "sidecar": dict(self.sidecar),
                "stage_durations": {stage: dict(value) for stage, value in self.stages.items()},
                "candidate_block_cache_totals": cache,
                "candidate_revision_pinned_opens": self.pinned_video_opens,
            }

    def write_files(self, path):
        with self.lock:
            files = sorted(self.files.items())
        with gzip.open(path, "wt", encoding="utf-8") as out:
            for (purpose, path), counts in files:
                out.write(json.dumps({"purpose": purpose, "file": path, **counts}, separators=(",", ":")) + "\n")

    def install_http(self):
        from huggingface_hub.utils import httpx

        sync_raw = httpx.Response.iter_raw
        async_raw = httpx.Response.aiter_raw

        @wraps(sync_raw)
        def iter_raw(response, *args, **kwargs):
            purpose, path = self.response_source(response)
            total = 0
            try:
                for chunk in sync_raw(response, *args, **kwargs):
                    total += len(chunk)
                    yield chunk
            finally:
                self.received(response, total, purpose, path)

        @wraps(async_raw)
        async def aiter_raw(response, *args, **kwargs):
            purpose, path = self.response_source(response)
            total = 0
            try:
                async for chunk in async_raw(response, *args, **kwargs):
                    total += len(chunk)
                    yield chunk
            finally:
                self.received(response, total, purpose, path)

        httpx.Response.iter_raw = iter_raw
        httpx.Response.aiter_raw = aiter_raw

    def install_workload(self, reader, repo, revision):
        import fsspec
        from huggingface_hub.hf_file_system import HfFileSystem, HfFileSystemFile

        # Keep the source name when HTTP requests are redirected to an opaque storage URL.
        original_range = HfFileSystemFile._fetch_range

        @wraps(original_range)
        def fetch_range(obj, *args, **kwargs):
            path = source_path(obj.path)
            purpose = _PURPOSE.get() or file_purpose(path)
            with io_context(purpose, path), self.measure(f"{purpose or 'other'}_fsspec_range_fetch"):
                return original_range(obj, *args, **kwargs)

        HfFileSystemFile._fetch_range = fetch_range

        original_hf_open = HfFileSystem.open

        @wraps(original_hf_open)
        def hf_open(obj, path, *args, **kwargs):
            relative_path = source_path(path)
            purpose = _PURPOSE.get() or file_purpose(relative_path)
            if purpose:
                self.touch(purpose, relative_path, "filesystem_opens")
            with io_context(purpose, relative_path):
                return original_hf_open(obj, path, *args, **kwargs)

        HfFileSystem.open = hf_open

        if reader == "candidate" and revision is not None:
            # Candidate metadata and Parquet use the supplied SHA. Its video URL omits it.
            original_fsspec_open = fsspec.open
            prefix = f"sharedblockcache::hf://datasets/{repo}"
            original_video_prefix = f"{prefix}/videos/"

            @wraps(original_fsspec_open)
            def pinned_open(urlpath, *args, **kwargs):
                if isinstance(urlpath, str) and urlpath.startswith(original_video_prefix):
                    urlpath = f"{prefix}@{revision}{urlpath[len(prefix):]}"
                    with self.lock:
                        self.pinned_video_opens += 1
                return original_fsspec_open(urlpath, *args, **kwargs)

            fsspec.open = pinned_open

        if reader == "candidate":
            from datasets.filesystems.caching import SharedBlockCacheFileSystem
            from lerobot.datasets import streaming_dataset

            self.wrap_stage(
                streaming_dataset, "decode_video_frames_torchcodec", "candidate_rgb_decode_inclusive_of_io"
            )
            self.wrap_stage(
                streaming_dataset.StreamingLeRobotDataset, "_sample_pool_with_delta", "candidate_pool_assembly"
            )
            original_cache_open = SharedBlockCacheFileSystem._open
            self.cache_stats = SharedBlockCacheFileSystem.default_store().stats

            @wraps(original_cache_open)
            def cache_open(obj, path, *args, **kwargs):
                relative_path = source_path(path)
                purpose = file_purpose(relative_path)
                if purpose:
                    self.touch(purpose, relative_path, "block_cache_opens")
                return original_cache_open(obj, path, *args, **kwargs)

            SharedBlockCacheFileSystem._open = cache_open
            return
        from lerobot.datasets import streaming_sidecar
        from lerobot.streaming.episode_cache import EpisodeByteCache
        from lerobot.streaming.episode_parquet import EpisodeParquetReader
        from lerobot.streaming.range_fetch import NativeHTTPRangeFetcher

        self.wrap_stage(EpisodeByteCache, "_open_decoder", "baseline_rgb_decoder_open")
        self.wrap_stage(EpisodeByteCache, "_get_frames", "baseline_rgb_decode_inclusive_of_cache_wait")
        self.wrap_stage(EpisodeByteCache, "_fetch_and_synthesize", "baseline_video_fetch_and_synthesize")
        self.wrap_stage(EpisodeParquetReader, "read_episode", "baseline_parquet_read_and_assemble")

        build = streaming_sidecar.build_mp4_sidecar
        download = streaming_sidecar.download_published_sidecar
        read_range = NativeHTTPRangeFetcher._read_range_single

        def observe_sidecar(kind, func):
            @wraps(func)
            def measured(*args, **kwargs):
                started = perf_counter()
                self.probing = kind == "sidecar_probe"
                self.trace.write("sidecar_start", phase=kind)
                try:
                    with io_context(kind):
                        result = func(*args, **kwargs)
                except BaseException:
                    with self.lock:
                        self.sidecar[kind] = {"status": "error", "s": perf_counter() - started}
                    raise
                finally:
                    self.probing = False
                status = "missing" if result is False else "ok"
                with self.lock:
                    self.sidecar[kind] = {"status": status, "s": perf_counter() - started}
                self.trace.write("sidecar_end", phase=kind, status=status, phase_s=perf_counter() - started)
                return result

            return measured

        streaming_sidecar.build_mp4_sidecar = observe_sidecar("sidecar_probe", build)
        streaming_sidecar.download_published_sidecar = observe_sidecar("sidecar_download", download)

        @wraps(read_range)
        def baseline_range(obj, relative_path, *args, **kwargs):
            # The baseline finishes sidecar work before it starts training reads.
            purpose = "sidecar_probe" if self.probing else (file_purpose(relative_path) or "video")
            self.touch(purpose, relative_path, "range_reads")
            with io_context(purpose, relative_path), self.measure(f"{purpose}_native_http_range_fetch"):
                return read_range(obj, relative_path, *args, **kwargs)

        NativeHTTPRangeFetcher._read_range_single = baseline_range


def net_rx_bytes():
    try:
        rows = Path("/proc/net/dev").read_text().splitlines()[2:]
        return sum(int(line.split(":")[1].split()[0]) for line in rows if line.split(":")[0].strip() != "lo")
    except (OSError, ValueError, IndexError):
        return None


def cgroup_value(name):
    try:
        return int((Path("/sys/fs/cgroup") / name).read_text().strip())
    except (OSError, ValueError):
        return None


class ResourceSampler:
    """Sample total process-tree RSS and available Linux container memory counters."""

    def __init__(self, trace, io, interval=1.0):
        self.trace = trace
        self.io = io
        self.interval = interval
        self.done = Event()
        self.thread = Thread(target=self.run, name="review5388-metrics", daemon=True)
        self.rx_start = net_rx_bytes()
        self.latest = {}
        self.rss_peak = None
        self.cgroup_peak = None

    def sample(self):
        import psutil

        process = psutil.Process()
        rss = {}
        for proc in [process, *process.children(recursive=True)]:
            try:
                rss[proc.pid] = proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        total_rss = sum(rss.values())
        self.rss_peak = max(total_rss, self.rss_peak or 0)
        cgroup_current = cgroup_value("memory.current")
        if cgroup_current is not None:
            self.cgroup_peak = max(cgroup_current, self.cgroup_peak or 0)
        rx = net_rx_bytes()
        self.latest = {
            "pid_rss_bytes": rss,
            "rss_sum_bytes": total_rss,
            "rss_sampled_peak_bytes": self.rss_peak,
            "cgroup_current_bytes": cgroup_current,
            "cgroup_sampled_peak_bytes": self.cgroup_peak,
            "cgroup_lifetime_peak_bytes": cgroup_value("memory.peak"),
            "net_namespace_rx_bytes": None if rx is None or self.rx_start is None else rx - self.rx_start,
        }

    def run(self):
        while not self.done.wait(self.interval):
            self.sample()
            self.trace.write("resources", **self.latest, io=self.io.snapshot())

    def reset(self):
        self.rx_start = net_rx_bytes()
        self.rss_peak = None
        self.cgroup_peak = None
        self.sample()

    def start(self):
        self.sample()
        self.thread.start()

    def stop(self):
        self.done.set()
        self.thread.join()
        self.sample()
