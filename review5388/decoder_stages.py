"""Measure decoder creation and frame retrieval on identical local RGB requests."""

import argparse
import json
import random
from contextlib import ExitStack, contextmanager, nullcontext
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from benchmark import NATIVE_TOLERANCE_S, provenance
from correctness.fixture import CAMERAS, FPS
from lerobot.datasets import streaming_dataset
from telemetry import IORecorder


@contextmanager
def reuse_candidate_decoders():
    """Control that keeps one decoder and its input handle per local source MP4."""
    import fsspec
    from torchcodec import decoders

    original = decoders.VideoDecoder
    cached = {}
    with ExitStack() as handles:

        def get_decoder(source, *args, **kwargs):
            if source.name not in cached:
                retained = handles.enter_context(fsspec.open(source.name))
                cached[source.name] = original(retained, *args, **kwargs)
            return cached[source.name]

        decoders.VideoDecoder = get_decoder
        try:
            yield
        finally:
            decoders.VideoDecoder = original
            cached.clear()


def fixture_data(root):
    episodes = json.loads((root / "oracle/episodes.json").read_text())
    anchors = [(row, frame) for row in episodes for frame in range(row["length"])]
    random.Random(42).shuffle(anchors)
    video_hash = sha256()
    video_files = sorted(root.glob("videos/*/*/*.mp4"))
    for path in video_files:
        # Read the same source files before each arm so local disk cache can warm.
        video_hash.update(sha256(path.read_bytes()).digest())
    return episodes, anchors, video_files, video_hash.hexdigest()


def exercise(args, dataset, episodes, anchors, offsets):
    cache_context = nullcontext()
    if args.reader == "baseline":
        episode_ids = [row["episode_index"] for row in episodes]
        share, _ = dataset._worker_share(episode_ids)
        cache_context = dataset._make_video_cache(episode_ids, 1, share)
    with cache_context as cache:
        stage_start = perf_counter()
        if cache is not None:
            for row in episodes:
                for camera in CAMERAS:
                    cache.get_bytes(row["episode_index"], camera)
        prepare_s = perf_counter() - stage_start

        outputs = []
        started = perf_counter()
        for row, anchor in anchors:
            episode = row["episode_index"]
            frames = [max(0, anchor + offset) for offset in offsets]
            requests = {
                camera: [
                    row[f"videos/{camera}/from_timestamp"] + frame / FPS
                    for frame in frames
                ]
                for camera in CAMERAS
            }
            if cache is None:
                images = dataset._query_videos(requests, episode)
            else:
                images = {
                    key: cache.get_frames(episode, key, ts)
                    for key, ts in requests.items()
                }
            outputs.append(
                {
                    key: value.reshape(len(offsets), *value.shape[-3:])
                    for key, value in images.items()
                }
            )
        wall_s = perf_counter() - started
        reuse = (
            {
                "retained_decoders": cache.open_decoder_count,
                "fallbacks": cache.decoder_fallback_count,
            }
            if cache is not None
            else None
        )
    return outputs, prepare_s, wall_s, reuse


def check_video(root, anchors, offsets, outputs):
    references = {}
    maximum_error = 0
    for (row, anchor), output in zip(anchors, outputs, strict=True):
        for camera_index, camera in enumerate(CAMERAS):
            file_index = row[f"videos/{camera}/file_index"]
            key = (file_index, camera_index)
            if key not in references:
                references[key] = np.load(
                    root / f"oracle/{file_index:03}-{camera_index}.npy", mmap_mode="r"
                )
            source_start = round(row[f"videos/{camera}/from_timestamp"] * FPS)
            indices = [source_start + max(0, anchor + offset) for offset in offsets]
            expected = np.moveaxis(references[key][indices], -1, 1)
            actual = output[camera].numpy()
            if actual.dtype != expected.dtype or actual.shape != expected.shape:
                raise RuntimeError(
                    f"Unexpected RGB shape or dtype for {key}, anchor {anchor}"
                )
            maximum_error = max(
                maximum_error, int(np.abs(actual.astype(np.int16) - expected).max())
            )
    return maximum_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader", required=True, choices=("baseline", "candidate"))
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--history",
        action="store_true",
        help="Request five camera frames for each anchor.",
    )
    parser.add_argument(
        "--candidate-reuse-control",
        action="store_true",
        help="Retain candidate decoders per MP4.",
    )
    args = parser.parse_args()
    if args.candidate_reuse_control and args.reader != "candidate":
        parser.error("--candidate-reuse-control requires --reader candidate")
    args.fixture = args.fixture.resolve()
    torch.set_num_threads(1)
    environment = provenance(args.reader, streaming_dataset)
    episodes, anchors, video_files, video_hash = fixture_data(args.fixture)
    offsets = [-20, -15, -10, -5, 0] if args.history else [0]
    if args.reader == "baseline":
        settings = {
            "repo_id": "fixture/stream5388",
            "episode_pool_size": len(episodes),
            "video_backend": "torchcodec",
        }
    else:
        settings = {"repo_id": str(args.fixture), "buffer_size": len(episodes)}
    dataset = streaming_dataset.StreamingLeRobotDataset(
        root=args.fixture,
        repo_type="dataset",
        tolerance_s=NATIVE_TOLERANCE_S,
        return_uint8=True,
        **settings,
    )
    recorder = IORecorder(trace=None)
    recorder.install_rgb_decode(args.reader)
    control = (
        reuse_candidate_decoders() if args.candidate_reuse_control else nullcontext()
    )
    with control:
        outputs, prepared_s, elapsed_s, reuse = exercise(
            args, dataset, episodes, anchors, offsets
        )
    stages = recorder.snapshot()["stage_durations"]
    rgb_error = check_video(args.fixture, anchors, offsets, outputs)
    requests = len(anchors) * len(CAMERAS)
    result = {
        "reader": args.reader,
        "candidate_reuse_control": args.candidate_reuse_control,
        "environment": environment,
        "episodes": len(episodes),
        "source_mp4s": len(video_files),
        "source_mp4_bytes": sum(path.stat().st_size for path in video_files),
        "source_mp4_inventory_sha256": video_hash,
        "anchors": len(anchors),
        "request_shuffle_seed": 42,
        "camera_requests": requests,
        "frame_offsets": offsets,
        "requested_frames": requests * len(offsets),
        "tolerance_s": NATIVE_TOLERANCE_S,
        "max_absolute_rgb8_difference_from_ffmpeg": rgb_error,
        "baseline_video_payload_prestage_s": prepared_s
        if args.reader == "baseline"
        else None,
        "reader_request_loop_s": elapsed_s,
        "baseline_decoder_reuse": reuse,
        "stage_durations": stages,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    reads = stages[f"{args.reader}_rgb_torchcodec_get_frames_at"]
    creates = stages[f"{args.reader}_rgb_torchcodec_create"]
    expected_creates = len(episodes) * len(CAMERAS)
    if args.reader == "candidate":
        expected_creates = (
            len(video_files) if args.candidate_reuse_control else requests
        )
    if (
        rgb_error > 4
        or reads["calls"] != requests
        or creates["calls"] != expected_creates
    ):
        raise SystemExit("Unexpected frame or decoder count. See the output JSON.")
    if any(stage["errors"] for stage in stages.values()):
        raise SystemExit(
            "At least one timed stage raised an error. See the output JSON."
        )


if __name__ == "__main__":
    main()
