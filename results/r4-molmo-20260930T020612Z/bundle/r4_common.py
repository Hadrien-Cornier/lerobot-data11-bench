"""Round 4 shared helpers (no torch / torchcodec / PyAV import at module level).

Round 4 repeats the SEP vs STACK comparison on 4 more public LeRobot v3 datasets (one HF job per dataset).
Layouts built from the SAME decoded RGB frames with LeRobot's default encoder:
  SEP    one MP4 per camera at native size (N video keys)
  STACK  one MP4, cameras stacked top to bottom in the recorded camera order (1 video key); narrower cameras are
         right-padded with black to the widest width; odd heights / widths get one black row / column
  SEP1   only the first camera key, same files as SEP for that camera
"""

import json
import os
import time
from pathlib import Path

DATASETS = {
    # short: (repo, full sha pinned 2026-09-29)
    "droid": ("lerobot/droid_1.0.1", "0eabc778f959c54b8c5aa3626cc1128d2d2e54d4"),
    "molmo": ("allenai/MolmoAct2-BimanualYAM-Dataset", "e9f21ae15074330839f2ac25ed4b49d76dfa1f9c"),
    "libero": ("lerobot/libero", "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"),
    "hqf": ("lerobot/high_quality_folding", "c9eb858d4b84e520edecbda84a3534c3c1e78436"),
}
LAYOUTS = ["sep", "stack", "sep1"]
STACK_KEY = "observation.images.stacked"
DEFAULT_CACHE = 100


def env(name, default):
    return os.environ.get(name, default)


def env_int(name, default):
    return int(os.environ.get(name, default))


def read_json(p):
    return json.loads(Path(p).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def key(cam):
    return f"observation.images.{cam}"


def stack_layout(shapes, order):
    """shapes {cam: [h, w]} -> geometry of the stacked frame.
    Each camera block starts at y0 and has an even height (h + h % 2); the canvas width is the widest width rounded
    up to even. crops[cam] = [y0, h, w] recovers the camera's exact frame with a slice."""
    W = max(shapes[c][1] for c in order)
    W += W % 2
    y, crops, odd_pad = 0, {}, {}
    for c in order:
        h, w = shapes[c]
        crops[c] = [y, h, w]
        odd_pad[c] = h % 2
        y += h + h % 2
    H = y
    cam_px = sum(shapes[c][0] * shapes[c][1] for c in order)
    return {"order": order, "height": H, "width": W, "crops": crops, "odd_height_pad_rows": odd_pad,
            "pad_pixels_per_frame": H * W - cam_px, "pad_pixel_share": (H * W - cam_px) / (H * W),
            "pad_raw_rgb_bytes_per_frame": 3 * (H * W - cam_px)}


def crop_views(x, crops, order):
    """x: (..., C, H, W) stacked tensor -> {cam: view} (slices, no copy)."""
    return {c: x[..., crops[c][0]: crops[c][0] + crops[c][1], : crops[c][2]] for c in order}


# ------------------------------------------------------------------ map-style wrapper (picklable for spawn workers)
_MAPCOUNT = {"on": False, "hits": 0, "misses": 0, "open_s": 0.0}


def _install_map_counter():
    """Count VideoDecoderCache hits / misses in THIS process (each DataLoader worker installs its own)."""
    if _MAPCOUNT["on"]:
        return
    import threading

    from lerobot.datasets import video_utils

    lock = threading.Lock()
    orig = video_utils.VideoDecoderCache.get_decoder

    def get_decoder(self, video_path):
        hit = str(video_path) in self
        t0 = time.perf_counter()
        d = orig(self, video_path)
        with lock:
            if hit:
                _MAPCOUNT["hits"] += 1
            else:
                _MAPCOUNT["misses"] += 1
                _MAPCOUNT["open_s"] += time.perf_counter() - t0
        return d

    video_utils.VideoDecoderCache.get_decoder = get_decoder
    _MAPCOUNT["on"] = True


class MapWrap:
    """Map-style dataset around a LeRobotDataset: STACK -> per-camera crops (slices) inside __getitem__, plus the
    decoder-cache counts of this sample (_r4_hits, _r4_misses, _r4_open_s)."""

    def __init__(self, base, crops=None, order=None):
        self.base, self.crops, self.order = base, crops, order

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        _install_map_counter()
        h0, m0, o0 = _MAPCOUNT["hits"], _MAPCOUNT["misses"], _MAPCOUNT["open_s"]
        item = self.base[idx]
        if self.crops is not None:
            x = item.pop(STACK_KEY)
            item.update({key(c): v for c, v in crop_views(x, self.crops, self.order).items()})
        item["_r4_hits"] = _MAPCOUNT["hits"] - h0
        item["_r4_misses"] = _MAPCOUNT["misses"] - m0
        item["_r4_open_s"] = _MAPCOUNT["open_s"] - o0
        return item


# ------------------------------------------------------------------ statistics
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def mean_ci(xs):
    """-> (mean, 95% Student-t half width or None)."""
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return None, None
    m = sum(xs) / n
    if n < 2:
        return m, None
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    return m, T975.get(n - 1, 1.96) * sd / n ** 0.5


def fmt_ci(m, h, nd=1):
    if m is None:
        return "-"
    return f"{m:.{nd}f} +- {h:.{nd}f}" if h is not None else f"{m:.{nd}f}"


# ------------------------------------------------------------------ extrapolation (streaming shard split)
def contiguous_split(n, k):
    """datasets' contiguous shard sizes: n items into k shards, the first n % k shards get one extra."""
    return [n // k + (1 if i < n % k else 0) for i in range(k)]


def streaming_active_shards(parquet_files, workers):
    """StreamingLeRobotDataset in a DataLoader with num_workers=W (lerobot-train passes max_num_shards=W):
    num_shards = min(P, W) shards (contiguous file blocks); inside each worker, datasets' _iter_pytorch splits each
    shard's files again across the W workers (non contiguous), so worker w only reads shard k if the shard has more
    than w files. Returns active shards per worker, i.e. files a worker reads at the same time."""
    W = max(1, workers)
    S = min(parquet_files, W)
    sizes = contiguous_split(parquet_files, S)
    if workers <= 0:  # num_workers=0: the main process reads every shard
        return [S]
    return [sum(1 for n in sizes if n > w) for w in range(W)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)
