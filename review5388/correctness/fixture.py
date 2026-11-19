"""Generate a small LeRobot v3 dataset and a decoder-independent reference."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

FPS = 25
HEIGHT, WIDTH = 48, 64
CAMERAS = ("observation.images.front", "observation.images.left_wrist", "observation.images.right_wrist")
LENGTHS_BY_FILE = ((107, 11), (17, 109), (43, 7), (13, 131), (29, 19), (101, 8), (57, 6), (15, 113))
SHORT_LENGTHS_BY_FILE = ((17, 2), (3, 6), (3, 1), (3, 9), (2, 1), (31, 1), (2, 1), (1, 3))
STATE = "observation.state"
ACTION = "action"
TASK = "move the arm"


def vectors(ids, key):
    ids = np.asarray(ids, dtype=np.float32)
    if key == STATE:
        return np.stack((ids / 32, -ids / 16), axis=-1)
    return np.stack((ids / 64, 4 - ids / 32, ids / 16 - 2), axis=-1)


def stats(values):
    return {name: operation(values, axis=0).tolist() for name, operation in (
        ("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max),
    )} | {"count": [len(values)]}


def camera_video(path, frame_count, file_index, camera_index):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(6388 + file_index * 3 + camera_index)
    patches = rng.integers(24, 232, (frame_count, HEIGHT // 8, WIDTH // 8, 3), dtype=np.uint8)
    frames = np.repeat(np.repeat(patches, 8, axis=1), 8, axis=2)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0", "-vf", "format=yuv420p",
         "-c:v", "libx264", "-threads", "1", "-g", "16", "-bf", "2", "-crf", "12", str(path)],
        input=frames.tobytes(), check=True,
    )
    decoded = subprocess.check_output(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "1", "-i", str(path),
         "-f", "rawvideo", "-c:v", "rawvideo", "-threads", "1", "-pix_fmt", "rgb24", "pipe:1"]
    )
    return np.frombuffer(decoded, np.uint8).reshape(frame_count, HEIGHT, WIDTH, 3)


def make(root, lengths_by_file=LENGTHS_BY_FILE):
    root.mkdir(parents=True, exist_ok=False)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "oracle").mkdir(parents=True)
    rows = []
    index = 0
    scalar_types = {"timestamp": pa.float32(), "frame_index": pa.int64(), "episode_index": pa.int64(),
                    "index": pa.int64(), "task_index": pa.int64()}
    video_offsets = (0, 4, 9)

    for file_index, lengths in enumerate(lengths_by_file):
        count = sum(lengths)
        for camera_index, camera in enumerate(CAMERAS):
            path = root / f"videos/{camera}/chunk-000/file-{file_index:03}.mp4"
            video = camera_video(path, count + video_offsets[camera_index], file_index, camera_index)
            np.save(root / f"oracle/{file_index:03}-{camera_index}.npy", video)
        arrays = {key: [] for key in scalar_types}
        offset = 0
        for length in lengths:
            episode = len(rows)
            arrays["timestamp"].extend(frame / FPS for frame in range(length))
            arrays["frame_index"].extend(range(length))
            arrays["episode_index"].extend([episode] * length)
            arrays["index"].extend(range(index, index + length))
            arrays["task_index"].extend([0] * length)
            row = {
                "episode_index": episode, "dataset_from_index": index, "dataset_to_index": index + length,
                "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
                "data/chunk_index": 0, "data/file_index": file_index, "length": length, "tasks": [TASK],
            }
            for camera, video_offset in zip(CAMERAS, video_offsets, strict=True):
                row[f"videos/{camera}/chunk_index"] = 0
                row[f"videos/{camera}/file_index"] = file_index
                row[f"videos/{camera}/from_timestamp"] = (offset + video_offset) / FPS
                row[f"videos/{camera}/to_timestamp"] = (offset + video_offset + length) / FPS
            rows.append(row)
            offset += length
            index += length
        table = {key: pa.array(values, type=scalar_types[key]) for key, values in arrays.items()}
        for key, width in ((STATE, 2), (ACTION, 3)):
            table[key] = pa.array(vectors(arrays["index"], key).tolist(), type=pa.list_(pa.float32(), width))
        pq.write_table(pa.table(table), root / f"data/chunk-000/file-{file_index:03}.parquet", row_group_size=47)

    pd.DataFrame(rows).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index([TASK], name="task")).to_parquet(root / "meta/tasks.parquet")
    features = {key: {"dtype": str(dtype), "shape": [1], "names": None} for key, dtype in scalar_types.items()}
    features["timestamp"]["dtype"] = "float32"
    features[STATE] = {"dtype": "float32", "shape": [2], "names": ["a", "b"]}
    features[ACTION] = {"dtype": "float32", "shape": [3], "names": ["a", "b", "c"]}
    for camera in CAMERAS:
        features[camera] = {
            "dtype": "video", "shape": [HEIGHT, WIDTH, 3], "names": ["height", "width", "channels"],
            "info": {"video.height": HEIGHT, "video.width": WIDTH, "video.codec": "h264",
                     "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "video.fps": FPS,
                     "video.channels": 3, "has_audio": False},
        }
    info = {
        "codebase_version": "v3.0", "robot_type": "fixture", "total_episodes": len(rows),
        "total_frames": index, "total_tasks": 1, "chunks_size": 1000, "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500, "fps": FPS, "splits": {"train": f"0:{len(rows)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (root / "meta/info.json").write_text(json.dumps(info, indent=2))
    statistics = {key: stats(vectors(range(index), key)) for key in (STATE, ACTION)}
    for camera_index, camera in enumerate(CAMERAS):
        all_frames = np.concatenate([
            np.load(root / f"oracle/{file_index:03}-{camera_index}.npy")[video_offsets[camera_index]:]
            for file_index in range(len(lengths_by_file))
        ])
        camera_stats = stats(all_frames.reshape(-1, 3).astype(np.float32) / 255)
        for name in ("mean", "std", "min", "max"):
            camera_stats[name] = [[ [channel] ] for channel in camera_stats[name]]
        camera_stats["count"] = [index]
        statistics[camera] = camera_stats
    (root / "meta/stats.json").write_text(json.dumps(statistics, indent=2))
    (root / "oracle/episodes.json").write_text(json.dumps(rows, indent=2))
    print(f"{index} frames in {len(rows)} episodes and {len(lengths_by_file)} parquet files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--short", action="store_true")
    args = parser.parse_args()
    make(args.root.resolve(), SHORT_LENGTHS_BY_FILE if args.short else LENGTHS_BY_FILE)
