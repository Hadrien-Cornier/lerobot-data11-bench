"""Read #5377's saved pytest fixtures with the proposed streaming reader."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import DepthEncoderConfig
from lerobot.datasets.depth_utils import dequantize_depth, quantize_depth
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset


def check(root):
    key = "observation.images.depth"
    dataset = StreamingLeRobotDataset("local/depth-seek", root=root, shuffle=True)
    encoder = DepthEncoderConfig.from_video_info(dataset.meta.features[key]["info"])
    quantizer = {k: getattr(encoder, k) for k in ("depth_min", "depth_max", "shift", "use_log")}
    expected_ids = {
        (episode, frame)
        for episode in range(dataset.meta.total_episodes)
        for frame in range(int(dataset.meta.episodes[episode]["length"]))
    }
    seen = set()
    for row in dataset:
        episode, frame = int(row["episode_index"]), int(row["frame_index"])
        assert (episode, frame) not in seen
        seen.add((episode, frame))
        # Same input raster and quantization as the writer PR's test.
        original = np.linspace(0.5, 1.5, 32 * 48, dtype=np.float32).reshape(32, 48, 1) + episode + frame / 10
        codes = quantize_depth(original, **quantizer, video_backend=None)
        expected = dequantize_depth(torch.from_numpy(codes.copy()), **quantizer)
        np.testing.assert_array_equal(row[key].numpy(), expected.numpy())
    assert seen == expected_ids
    return len(seen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("basetemp", type=Path)
    args = parser.parse_args()
    roots = sorted(p / "dataset" for p in args.basetemp.glob("test_recorded_depth_random_acc*") if not p.is_symlink())
    assert len(roots) == 16
    print(json.dumps({root.parent.name: check(root) for root in roots}, indent=2))
