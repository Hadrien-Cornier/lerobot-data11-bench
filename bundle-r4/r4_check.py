"""Round 4 stage check: quality, camera order, remote == local, and the datasets read back through LeRobot.

1. PSNR at slice["check_indices"] (source RGB saved by stage encode): SEP vs source, STACK crop vs source, STACK crop
   vs SEP, per camera (TorchCodec decode of the local files, as LeRobot decodes).
2. Camera order: PSNR of each STACK crop against every SEP camera of the same shape; the own camera must be best.
3. Remote == local: TorchCodec VideoDecoder on fsspec hf://datasets/<repo>@<commit>/... (as VideoDecoderCache opens
   it) vs the local file, bit-exact, on R4_REMOTE_N frames per layout (split across the first and last video file of
   every key). Skipped with NO_UPLOAD=1.
4. Through LeRobot (map-style LeRobotDataset on the local copies, return_uint8=True): at R4_DS_N random global indices,
   SEP item == direct decode of (file, frame) bit-exact; SEP1 item == SEP item of the first camera bit-exact;
   STACK crop vs SEP item PSNR.

Usage: python r4_check.py <slice.json> <build.json> <work_dir> <out_dir>
"""

import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import pin_threads  # noqa: E402
from r4_common import STACK_KEY, crop_views, env, env_int, key, log, read_json, write_json  # noqa: E402

pin_threads(2)


def psnr(a, b):
    a = a.astype(np.float64) if isinstance(a, np.ndarray) else a.double().numpy()
    b = b.astype(np.float64) if isinstance(b, np.ndarray) else b.double().numpy()
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def hwc(t):
    """torchcodec (C,H,W) uint8 -> (H,W,C) numpy"""
    return t.permute(1, 2, 0).numpy()


def main():
    from torchcodec.decoders import VideoDecoder

    sl, bd = read_json(sys.argv[1]), read_json(sys.argv[2])
    work, out = Path(sys.argv[3]), Path(sys.argv[4])
    D = out / "check"
    D.mkdir(parents=True, exist_ok=True)
    cams, lay = sl["cams"], sl["stack_layout"]
    crops = lay["crops"]
    res = {}
    # ---- 1 + 2
    src = np.load(work / "check_src.npz")
    idx = [int(i) for i in src["indices"]]
    sep_root, stack_root, sep1_root = (Path(bd["datasets"][k]["root"]) for k in ("sep", "stack", "sep1"))
    sep_dec = {c: VideoDecoder(str(sep_root / "videos" / key(c) / "chunk-000" / "file-000.mp4"), seek_mode="approximate") for c in cams}
    stk_dec = VideoDecoder(str(stack_root / "videos" / STACK_KEY / "chunk-000" / "file-000.mp4"), seek_mode="approximate")
    rows = {c: {"sep_vs_src": [], "stack_vs_src": [], "stack_vs_sep": []} for c in cams}
    order = {c: {c2: [] for c2 in cams if sl["shapes"][c2] == sl["shapes"][c]} for c in cams}
    for j, i in enumerate(idx):
        st = stk_dec.get_frames_at(indices=[i]).data[0]
        views = crop_views(st, crops, cams)
        sp = {c: sep_dec[c].get_frames_at(indices=[i]).data[0] for c in cams}
        for c in cams:
            s = src[c][j]
            v, p = hwc(views[c]), hwc(sp[c])
            assert v.shape == s.shape == p.shape, (c, v.shape, s.shape, p.shape)
            rows[c]["sep_vs_src"].append(psnr(p, s))
            rows[c]["stack_vs_src"].append(psnr(v, s))
            rows[c]["stack_vs_sep"].append(psnr(v, p))
            for c2 in order[c]:
                order[c][c2].append(psnr(v, hwc(sp[c2])))
    res["psnr"] = {c: {k: {"mean": float(np.mean(v)), "min": float(np.min(v))} for k, v in r.items()} for c, r in rows.items()}
    res["psnr_frames"] = len(idx)
    om = {c: {c2: float(np.mean(v)) for c2, v in d.items()} for c, d in order.items()}
    res["camera_order_psnr_matrix"] = om
    res["camera_order_ok"] = all(max(d, key=d.get) == c for c, d in om.items())
    res["camera_order_comparable_pairs"] = sum(len(d) - 1 for d in om.values())
    res["psnr_min_stack_vs_src"] = min(v["stack_vs_src"]["min"] for v in res["psnr"].values())
    res["psnr_min_stack_vs_sep"] = min(v["stack_vs_sep"]["min"] for v in res["psnr"].values())
    log(f"PSNR {res['psnr']}; camera order ok {res['camera_order_ok']} ({om})")
    write_json(D / "check.json", res)

    # ---- 3 remote == local
    if env("NO_UPLOAD", "0") == "1":
        res["remote_equal"] = {"skipped": "NO_UPLOAD=1"}
    else:
        import fsspec

        n = env_int("R4_REMOTE_N", "20")
        rng = random.Random(7)
        rem = {}
        for kind, d in bd["datasets"].items():
            checked = bad = 0
            nf = d["frames_per_file"]
            files = sorted({0, d["files_per_key"] - 1})
            per = max(1, n // (len(files) * len(d["video_keys"])))
            for k in d["video_keys"]:
                for fi in files:
                    rel = f"videos/{k}/chunk-000/file-{fi:03d}.mp4"
                    loc = VideoDecoder(str(Path(d["root"]) / rel), seek_mode="approximate")
                    fh = fsspec.open(f"hf://datasets/{d['repo']}@{d['commit']}/{rel}").__enter__()
                    try:
                        r = VideoDecoder(fh, seek_mode="approximate")
                        for i in sorted(rng.sample(range(nf), min(per, nf))):
                            checked += 1
                            bad += int(not torch.equal(r.get_frames_at(indices=[i]).data, loc.get_frames_at(indices=[i]).data))
                    finally:
                        fh.close()
            rem[kind] = {"frames_checked": checked, "mismatch": bad, "pass": bad == 0 and checked > 0}
        res["remote_equal"] = rem
        log(f"remote == local: {rem}")
    write_json(D / "check.json", res)

    # ---- 4 through LeRobot
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dsd = {kind: LeRobotDataset(f"local/r4-{kind}", root=bd["datasets"][kind]["root"], return_uint8=True) for kind in ("sep", "stack", "sep1")}
    N = len(dsd["sep"])
    nf = bd["datasets"]["sep"]["frames_per_file"]
    rng = random.Random(11)
    gi = sorted(rng.sample(range(N), min(env_int("R4_DS_N", "20"), N)))
    exact_sep = exact_sep1 = 0
    ps = {c: [] for c in cams}
    direct = {c: {} for c in cams}
    for g in gi:
        fi, j = divmod(g, nf)
        a, b, s1 = dsd["sep"][g], dsd["stack"][g], dsd["sep1"][g]
        assert int(a["index"]) == g and int(b["index"]) == g
        views = crop_views(b[STACK_KEY], crops, cams)
        for c in cams:
            if fi not in direct[c]:
                direct[c][fi] = VideoDecoder(str(sep_root / "videos" / key(c) / "chunk-000" / f"file-{fi:03d}.mp4"), seek_mode="approximate")
            ref = direct[c][fi].get_frames_at(indices=[j]).data[0]
            exact_sep += int(torch.equal(a[key(c)], ref))
            ps[c].append(psnr(hwc(views[c]), hwc(a[key(c)])))
        exact_sep1 += int(torch.equal(s1[key(cams[0])], a[key(cams[0])]))
    res["through_lerobot"] = {"samples": len(gi), "sep_item_equals_file_frame": f"{exact_sep}/{len(gi) * len(cams)}",
                              "sep1_equals_sep_first_camera": f"{exact_sep1}/{len(gi)}",
                              "stack_crop_vs_sep_item_psnr_mean": {c: float(np.mean(v)) for c, v in ps.items()},
                              "stack_crop_vs_sep_item_psnr_min": {c: float(np.min(v)) for c, v in ps.items()},
                              "pass": exact_sep == len(gi) * len(cams) and exact_sep1 == len(gi)}
    log(f"through LeRobot: {res['through_lerobot']}")
    thr = float(env("R4_PSNR_MIN", "35"))
    res["psnr_threshold"] = thr
    res["ok"] = (res["camera_order_ok"] and res["psnr_min_stack_vs_src"] >= thr and res["through_lerobot"]["pass"]
                 and all(v.get("pass", True) for v in res["remote_equal"].values() if isinstance(v, dict)))
    write_json(D / "check.json", res)
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
