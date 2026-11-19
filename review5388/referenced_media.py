"""Count video files actually named by episodes in the pinned metadata."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq

from catalog import SOURCES


def count_files(dataset):
    repo, revision = SOURCES[dataset]
    root = f"datasets/{repo}@{revision}" if revision else f"buckets/{repo}"
    fs = HfFileSystem()
    with fs.open(f"{root}/meta/info.json", "rb") as stream:
        info = json.load(stream)
    cameras = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    references = set()
    episode_count = 0
    for path in sorted(fs.glob(f"{root}/meta/episodes/*/*.parquet")):
        with fs.open(path, "rb", block_size=131072, cache_type="readahead") as stream:
            columns = [f"videos/{key}/{field}" for key in cameras for field in ("chunk_index", "file_index")]
            table = pq.ParquetFile(stream).read(columns=columns)
            episode_count += len(table)
            for key in cameras:
                chunks = table.column(f"videos/{key}/chunk_index").to_pylist()
                files = table.column(f"videos/{key}/file_index").to_pylist()
                for chunk, file in set(zip(chunks, files, strict=True)):
                    references.add(info["video_path"].format(video_key=key, chunk_index=chunk, file_index=file))
    if episode_count != info["total_episodes"]:
        raise ValueError(f"Expected {info['total_episodes']} episodes, scanned {episode_count}")
    physical = {
        path.removeprefix(root + "/") for path in fs.glob(f"{root}/videos/*/*/*.mp4")
    }
    return {"dataset": dataset, "episode_count": episode_count, "physical": len(physical),
            "referenced": len(references), "missing_references": sorted(references - physical),
            "unreferenced": sorted(physical - references)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=SOURCES)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stats = count_files(args.dataset)
    args.output.write_text(json.dumps(stats, indent=2) + "\n")
    print(args.dataset, stats["referenced"], stats["physical"], len(stats["missing_references"]))
