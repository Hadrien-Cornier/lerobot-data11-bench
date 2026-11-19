"""Check relaxed timestamp tolerance on a full fixture and identifiable late MP4 frames."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import BATCH_SIZE, NATIVE_TOLERANCE_S, build_reader, provenance
from check_stream import Oracle
from fixture import CAMERAS, FPS
from lerobot.datasets import streaming_dataset
from lerobot.datasets.video_utils import FrameTimestampError, decode_video_frames_torchcodec

READER = streaming_dataset.StreamingLeRobotDataset
LATE_FPS = 30
LATE_CASES = ((54901, 8), (136684, 2))


def fixture_oracle(args):
    oracle = Oracle(args.fixture, "anchors", None)

    def local_reader(repo_id, **kwargs):
        if args.reader == "candidate":
            repo_id = str(args.fixture)
        return READER(repo_id, root=args.fixture, **(kwargs | {"repo_type": "dataset"}))

    config = SimpleNamespace(
        reader=args.reader, tolerance_s=args.tolerance_s, single_frame=True, history=False, seed=42, pool_size=4,
    )
    dataset = build_reader(config, local_reader, "fixture/stream5388", None, FPS, CAMERAS)
    ids = []
    batches = 0
    for batch in torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0):
        batches += 1
        for offset in range(len(batch["index"])):
            ids.append(oracle.sample({key: value[offset] for key, value in batch.items()}))
    return {
        "batches": batches,
        "expected": len(oracle.ids),
        "delivered": len(ids),
        "unique": len(set(ids)),
        "coverage_matches": sorted(ids) == oracle.ids,
        "checks_failed": dict(oracle.errors),
        "first_failure": oracle.examples,
        "max_decoded_rgb8_difference": oracle.max_pixel_error,
    }


def create_late_frame_video(path):
    enabled = "+".join(f"eq(n,{frame})" for frame, _ in LATE_CASES)
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=blue:size=16x16:rate={LATE_FPS}",
        "-vf", f"drawbox=w=iw:h=ih:color=red:t=fill:enable='{enabled}'",
        "-frames:v", str(LATE_CASES[-1][0] + 2), "-c:v", "libx264", "-preset", "ultrafast",
        "-threads", "1", "-g", str(LATE_FPS), str(path),
    ], check=True)


def late_queries(args, path):
    create_late_frame_video(path)

    def decode(timestamp, tolerance):
        if args.reader == "candidate":
            with path.open("rb") as video:
                return decode_video_frames_torchcodec(video, [timestamp], tolerance, return_uint8=True)
        return decode_video_frames_torchcodec(path, [timestamp], tolerance, return_uint8=True)

    results = []
    for frame, anchor in LATE_CASES:
        ideal_s = frame / LATE_FPS
        start_s = (frame - anchor) / LATE_FPS
        # The candidate adds the episode's source start to the current frame in float32.
        quantized_query_s = (torch.tensor(anchor / LATE_FPS, dtype=torch.float32) + start_s).item()
        try:
            decode(quantized_query_s, NATIVE_TOLERANCE_S)
        except FrameTimestampError:
            strict_rejects = True
        else:
            strict_rejects = False
        exact = decode(ideal_s, NATIVE_TOLERANCE_S)
        relaxed = decode(quantized_query_s, args.tolerance_s)
        neighbor = decode((frame - 1) / LATE_FPS, NATIVE_TOLERANCE_S)
        target_rgb = relaxed[0, :, 8, 8].tolist()
        neighbor_rgb = neighbor[0, :, 8, 8].tolist()
        results.append({
            "frame_index": frame,
            "episode_anchor": anchor,
            "ideal_s": ideal_s,
            "quantized_query_s": quantized_query_s,
            "error_s": abs(quantized_query_s - ideal_s),
            "strict_tolerance_s": NATIVE_TOLERANCE_S,
            "strict_rejects_quantized_timestamp": strict_rejects,
            "relaxed_matches_exact": torch.equal(relaxed, exact),
            "relaxed_selects_red_target": target_rgb[0] > 200 and target_rgb[2] < 30,
            "neighbor_is_blue": neighbor_rgb[2] > 200 and neighbor_rgb[0] < 30,
            "relaxed_center_rgb": target_rgb,
            "neighbor_center_rgb": neighbor_rgb,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader", required=True, choices=("baseline", "candidate"))
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance-s", type=float, default=1e-3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {
        "reader": args.reader,
        "workload": "single_frame",
        "tolerance_s": args.tolerance_s,
        "environment": provenance(args.reader, streaming_dataset),
        "fixture": fixture_oracle(args),
    }
    with TemporaryDirectory() as directory:
        result["late_timestamp_decode_helper"] = late_queries(args, Path(directory) / "late.mp4")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    cases = result["late_timestamp_decode_helper"]
    if not (
        result["fixture"]["coverage_matches"]
        and not result["fixture"]["checks_failed"]
        and all(
            case["strict_rejects_quantized_timestamp"] and case["relaxed_matches_exact"]
            and case["relaxed_selects_red_target"] and case["neighbor_is_blue"] for case in cases
        )
    ):
        raise SystemExit("One or more tolerance checks failed.")


if __name__ == "__main__":
    main()
