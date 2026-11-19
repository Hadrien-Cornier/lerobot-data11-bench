"""Probe timestamp precision on real video and Parquet through current LeRobot readers."""

import argparse
from collections import Counter
from contextlib import closing
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.video_utils import FrameTimestampError

FPS = 30
CAMERA = "observation.images.front"
PACKED_LENGTHS = (54893, 16, 40886, 40887, 16)
PACKED_CASES = ((1, 8), (4, 2))
LONG_FRAME = 61441
REPO = "fixture/long-video"
MAIN = "8ec1a474da5f3f90a670d9343386995ec9f81264"


def fixture(root):
    root.mkdir(parents=True)
    for subdir in ("meta/episodes/chunk-000", "data/chunk-000", f"videos/{CAMERA}/chunk-000"):
        (root / subdir).mkdir(parents=True)
    source_frames = sum(PACKED_LENGTHS)
    source_red = {
        sum(PACKED_LENGTHS[:episode]) + frame for episode, frame in PACKED_CASES
    } | {LONG_FRAME}
    enabled = "+".join(f"eq(n,{frame})" for frame in sorted(source_red))
    video = root / f"videos/{CAMERA}/chunk-000/file-000.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=blue:size=16x16:rate={FPS}",
        "-vf", f"drawbox=w=iw:h=ih:color=red:t=fill:enable='{enabled}'",
        "-frames:v", str(source_frames), "-c:v", "libx264", "-preset", "ultrafast",
        "-threads", "1", "-g", str(FPS), str(video),
    ], check=True)
    os.link(video, video.with_name("file-001.mp4"))

    lengths = (*PACKED_LENGTHS, source_frames)
    frames = np.concatenate([np.arange(length) for length in lengths])
    total = len(frames)
    episodes = np.repeat(np.arange(len(lengths)), lengths)
    base_columns = {
        "timestamp": pa.array(frames.astype(np.float64) / FPS, type=pa.float32()),
        "frame_index": pa.array(frames, type=pa.int64()),
        "episode_index": pa.array(episodes, type=pa.int64()),
        "index": pa.array(np.arange(total), type=pa.int64()),
        "task_index": pa.array(np.zeros(total, dtype=np.int64)),
    }
    pq.write_table(pa.table(base_columns), root / "data/chunk-000/file-000.parquet", row_group_size=4096)
    rows = []
    index = 0
    for episode, length in enumerate(lengths):
        single_episode_video = episode == len(PACKED_LENGTHS)
        file_start = 0 if single_episode_video else index
        rows.append({
            "episode_index": episode, "dataset_from_index": index, "dataset_to_index": index + length,
            "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
            "data/chunk_index": 0, "data/file_index": 0, "length": length, "tasks": ["test video"],
            f"videos/{CAMERA}/chunk_index": 0, f"videos/{CAMERA}/file_index": int(single_episode_video),
            f"videos/{CAMERA}/from_timestamp": file_start / FPS,
            f"videos/{CAMERA}/to_timestamp": (file_start + length) / FPS,
        })
        index += length
    pd.DataFrame(rows).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["test video"], name="task")).to_parquet(root / "meta/tasks.parquet")
    features = {key: {"dtype": str(col.type), "shape": [1], "names": None} for key, col in base_columns.items()}
    features["timestamp"]["dtype"] = "float32"
    features[CAMERA] = {
        "dtype": "video", "shape": [16, 16, 3], "names": ["height", "width", "channels"],
        "info": {"video.height": 16, "video.width": 16, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                 "video.is_depth_map": False, "video.fps": FPS, "video.channels": 3, "has_audio": False},
    }
    info = {
        "codebase_version": "v3.0", "robot_type": "fixture", "total_episodes": len(rows), "total_frames": total,
        "total_tasks": 1, "chunks_size": 1000, "data_files_size_in_mb": 100, "video_files_size_in_mb": 500,
        "fps": FPS, "splits": {"train": f"0:{len(rows)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4", "features": features,
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/stats.json").write_text("{}")


def color(item):
    rgb = item[CAMERA][:, 8, 8].tolist()
    if not isinstance(rgb[0], int):
        rgb = [round(value * 255) for value in rgb]
    if rgb[0] > 200 and rgb[2] < 30:
        return "red"
    if rgb[2] > 200 and rgb[0] < 30:
        return "blue"
    return rgb


def probe_map(root):
    dataset = LeRobotDataset(REPO, root=root, video_backend="torchcodec")
    long_start = sum(PACKED_LENGTHS)
    cases = [("packed", sum(PACKED_LENGTHS[:ep]) + frame, ep, frame) for ep, frame in PACKED_CASES]
    cases += [("long_episode", long_start + LONG_FRAME, len(PACKED_LENGTHS), LONG_FRAME)]
    rows = []
    for case, index, ep, frame in cases:
        local_exact = frame / FPS
        file_start = 0.0 if case == "long_episode" else sum(PACKED_LENGTHS[:ep]) / FPS
        item = {
            "case": case, "index": index, "episode": ep, "frame_index": frame, "episode_from_s": file_start,
            "ideal_pts_s": file_start + local_exact,
            "map_query_s": file_start + float(np.float32(local_exact)),
            "float32_sum_query_s": float(np.float32(file_start) + np.float32(local_exact)),
        }
        item["earlier_frame"] = color(dataset[index - 1])
        try:
            item["default"] = color(dataset[index])
        except FrameTimestampError as exc:
            item["error_type"] = type(exc).__name__
            item["error_message"] = str(exc).split("\n", 1)[0]
            relaxed = LeRobotDataset(REPO, root=root, video_backend="torchcodec", tolerance_s=0.001)
            item["relaxed"] = color(relaxed[index])
        rows.append(item)
    return rows


def probe_stream(root):
    selected = [episode for episode, _ in PACKED_CASES]
    dataset = StreamingLeRobotDataset(REPO, root=root, episodes=selected, video_backend="torchcodec")
    observed = Counter()
    with closing(iter(dataset)) as stream:
        for item in stream:
            observed[(int(item["episode_index"]), int(item["frame_index"]), color(item))] += 1
    return {
        "samples": sum(observed.values()),
        "unique": len(observed),
        "matches_full_oracle": observed == {
            (episode, frame, "red" if (episode, frame) in PACKED_CASES else "blue"): 1
            for episode in selected for frame in range(PACKED_LENGTHS[episode])
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    fixture(args.fixture)
    rows = probe_map(args.fixture)
    streaming = probe_stream(args.fixture)
    result = {"main": MAIN, "fps": FPS, "torch": torch.__version__, "map_style": rows, "packed_streaming": streaming}
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    assert all(row["default"] == "red" and row["earlier_frame"] == "blue" for row in rows[:2])
    assert rows[-1]["error_type"] == "FrameTimestampError" and rows[-1]["relaxed"] == "red"
    assert streaming["matches_full_oracle"]


if __name__ == "__main__":
    main()
