"""Benchmark-only RGB decode variants for the pinned #5388 reader."""

import asyncio
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from itertools import chain
from threading import Condition

import fsspec
import torch
from lerobot.datasets.video_utils import FrameTimestampError
from torchcodec.decoders import VideoDecoder

DECODE_WORKERS = 32
RGB_STAGE = "candidate_rgb_decode_inclusive_of_io"
_VIDEO_QUERY = ContextVar("internvideo_video_query", default=None)


class DecoderPool:
    """Keep a bounded number of decoder and file pairs with exclusive leases."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.condition = Condition()
        self.idle = OrderedDict()
        self.allocated = 0
        self.peak = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def snapshot(self):
        with self.condition:
            return {
                "capacity": self.capacity,
                "allocated": self.allocated,
                "idle": len(self.idle),
                "peak_allocated": self.peak,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }

    def acquire(self, path):
        evicted = None
        with self.condition:
            self.condition.wait_for(lambda: self.idle or self.allocated < self.capacity)
            for key, (idle_path, _, _) in reversed(self.idle.items()):
                if idle_path == path:
                    self.hits += 1
                    return self.idle.pop(key)

            self.misses += 1
            if self.allocated == self.capacity:
                self.evictions += 1
                _, evicted = self.idle.popitem(last=False)
            else:
                self.allocated += 1
                self.peak = max(self.peak, self.allocated)

        resources = ExitStack()
        try:
            if evicted is not None:
                previous_resources = evicted[1]
                del evicted
                previous_resources.close()
            source = resources.enter_context(fsspec.open(path))
            decoder = VideoDecoder(source, seek_mode="approximate")
        except BaseException:
            try:
                resources.close()
            finally:
                with self.condition:
                    self.allocated -= 1
                    self.condition.notify()
            raise
        return path, resources, decoder

    @contextmanager
    def lease(self, path):
        entry = self.acquire(path)
        try:
            yield entry[2]
        except BaseException:
            try:
                resources = entry[1]
                del entry
                resources.close()
            finally:
                with self.condition:
                    self.allocated -= 1
                    self.condition.notify()
            raise
        else:
            with self.condition:
                self.idle[id(entry)] = entry
                self.condition.notify()

    def close(self):
        with self.condition, ExitStack() as resources_to_close:
            assert len(self.idle) == self.allocated
            while self.idle:
                _, entry = self.idle.popitem()
                resources_to_close.callback(entry[1].close)
                del entry
            self.allocated = 0


@contextmanager
def _record_decode(io):
    if io is None:
        yield
        return
    with io.measure(RGB_STAGE):
        token = io._in_rgb_request.set(True)
        try:
            yield
        finally:
            io._in_rgb_request.reset(token)


def _match_frames(timestamps, data, pts_seconds, tolerance, path, return_uint8):
    # Keep the pinned reader's nearest timestamp check within each camera request.
    query = torch.tensor(timestamps, dtype=torch.float64)
    loaded = torch.tensor([pts.item() for pts in pts_seconds], dtype=torch.float64)
    distances, indices = torch.cdist(query[:, None], loaded[:, None], p=1).min(1)
    outside = ~(distances <= tolerance)
    if outside.any():
        raise FrameTimestampError(
            f"Query timestamps violate {tolerance=}: {distances[outside]}"
            f"\nqueried timestamps: {query}\nloaded timestamps: {loaded}\nvideo: {path}"
        )
    frames = torch.stack([data[index] for index in indices])
    if not return_uint8:
        frames = (frames / 255.0).type(torch.float32)
    return frames.squeeze(0) if len(timestamps) == 1 else frames


def _read_video(pool, path, timestamps, tolerance, return_uint8, io, *, group_reads):
    with _record_decode(io), pool.lease(path) as decoder:
        fps = decoder.metadata.average_fps
        indices = [[round(ts * fps) for ts in request] for request in timestamps]
        if len(indices) == 1 or not group_reads:
            results = []
            for request, request_indices in zip(timestamps, indices, strict=True):
                frames = decoder.get_frames_at(indices=request_indices)
                results.append(
                    _match_frames(
                        request,
                        frames.data,
                        frames.pts_seconds,
                        tolerance,
                        path,
                        return_uint8,
                    )
                )
            return results

        combined = sorted(set(chain.from_iterable(indices)))
        frames = decoder.get_frames_at(indices=combined)
        positions = {index: offset for offset, index in enumerate(combined)}
        return [
            _match_frames(
                request,
                [frames.data[positions[index]] for index in request_indices],
                [frames.pts_seconds[positions[index]] for index in request_indices],
                tolerance,
                path,
                return_uint8,
            )
            for request, request_indices in zip(timestamps, indices, strict=True)
        ]


def _close_mapped_tasks(mapped_iterable):
    # The pinned Datasets map cancels on an exception, but not on generator close.
    for loop, tasks in mapped_iterable._owned_loops_and_tasks:
        for task in tasks:
            task.cancel()
        if tasks:
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        loop.close()
    mapped_iterable._owned_loops_and_tasks.clear()


def make_candidate_variant(
    dataset_class, variant, *, batch_size, decoder_cache_size=32, io=None
):
    """Return a candidate subclass for the single-process, current-frame RGB benchmark."""
    if variant not in {"reuse", "batch_singletons", "grouped"}:
        raise ValueError(f"Unknown #5388 decode variant: {variant}")
    if not 1 <= batch_size <= DECODE_WORKERS or decoder_cache_size < 1:
        raise ValueError(
            "Expected batch size between 1 and 32 and a positive decoder cache size"
        )

    class CandidateDecodeVariant(dataset_class):
        def _query_videos(self, query_timestamps, ep_idx):
            query = _VIDEO_QUERY.get()
            if query is None:
                return super()._query_videos(query_timestamps, ep_idx)
            return query(query_timestamps, ep_idx)

        def _video_path(self, episode, camera):
            root = self.root if self.streaming_from_local else self.meta.url_root
            path = f"{root}/{self.meta.get_video_file_path(episode, camera)}"
            return path if self.streaming_from_local else f"sharedblockcache::{path}"

        def decoder_cache_stats(self):
            pool = getattr(self, "_bench_decoder_pool", None)
            return pool.snapshot() if pool is not None else None

        def __iter__(self):
            if (
                self.delta_indices is not None
                or self.meta.depth_keys
                or self.image_transforms is not None
            ):
                raise ValueError(
                    "Decode variants require current-frame RGB without image transforms"
                )
            if torch.utils.data.get_worker_info() is not None:
                raise ValueError("Decode variants require num_workers=0")

            # Keep the pinned reader's Parquet, episode pool, shuffle, and row order.
            ds = self.hf_dataset.batch(by_column="episode_index")
            ds = ds.shuffle(seed=42, buffer_size=self.buffer_size)
            ds = ds.map(
                self._sample_pool_with_delta, batched=True, batch_size=self.buffer_size
            )
            pool = self._bench_decoder_pool = DecoderPool(decoder_cache_size)

            def read_video(path, requests):
                return _read_video(
                    pool,
                    path,
                    requests,
                    self.tolerance_s,
                    self._return_uint8,
                    io,
                    group_reads=variant == "grouped",
                )

            def decode_item(item):
                def query_video(requests, episode):
                    return {
                        camera: read_video(
                            self._video_path(episode, camera), [timestamps]
                        )[0]
                        for camera, timestamps in requests.items()
                    }

                token = _VIDEO_QUERY.set(query_video)
                try:
                    return self.query_frames(item)
                finally:
                    _VIDEO_QUERY.reset(token)

            def prepare_batch(batch):
                samples = []
                requests_by_path = defaultdict(list)

                def defer_video(requests, episode):
                    position = len(samples)
                    for camera, timestamps in requests.items():
                        path = self._video_path(episode, camera)
                        requests_by_path[path].append((position, camera, timestamps))
                    return dict.fromkeys(requests)

                token = _VIDEO_QUERY.set(defer_video)
                try:
                    for values in zip(*batch.values(), strict=True):
                        samples.append(
                            self.query_frames(dict(zip(batch, values, strict=True)))
                        )
                finally:
                    _VIDEO_QUERY.reset(token)
                return samples, requests_by_path

            group_slots = asyncio.Semaphore(DECODE_WORKERS // batch_size)
            sample_slots = asyncio.Semaphore(DECODE_WORKERS)
            try:
                with ThreadPoolExecutor(DECODE_WORKERS) as executor:

                    async def decode_sample(item):
                        async with sample_slots:
                            loop = asyncio.get_running_loop()
                            return await loop.run_in_executor(
                                executor, decode_item, item
                            )

                    async def decode_group(batch):
                        async with group_slots:
                            loop = asyncio.get_running_loop()
                            samples, requests_by_path = prepare_batch(batch)
                            futures = [
                                loop.run_in_executor(
                                    executor,
                                    read_video,
                                    path,
                                    [timestamps for _, _, timestamps in requests],
                                )
                                for path, requests in requests_by_path.items()
                            ]
                            replies = await asyncio.gather(*futures)
                            for requests, reply in zip(
                                requests_by_path.values(), replies, strict=True
                            ):
                                for (position, camera, _), value in zip(
                                    requests, reply, strict=True
                                ):
                                    samples[position][camera] = value
                            return {
                                key: [sample[key] for sample in samples]
                                for key in samples[0]
                            }

                    if variant == "reuse":
                        mapped = ds.map(decode_sample)
                    else:
                        mapped = ds.map(
                            decode_group, batched=True, batch_size=batch_size
                        )
                        # Match the stock outstanding-input limit in anchors, not map tasks.
                        iterable = mapped._ex_iterable
                        iterable.max_num_running_async_map_functions_in_parallel = max(
                            1,
                            iterable.max_num_running_async_map_functions_in_parallel
                            // batch_size,
                        )
                    try:
                        yield from mapped
                    finally:
                        _close_mapped_tasks(mapped._ex_iterable)
            finally:
                pool.close()

    return CandidateDecodeVariant
