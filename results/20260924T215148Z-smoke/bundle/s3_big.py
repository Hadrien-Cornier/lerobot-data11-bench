"""Stage big: per camera, stream-copy concat of 3 consecutive SEP groups into one BIG file
(ffmpeg concat demuxer, -c copy, no re-encode). Also records ffprobe packet checks for every
layout file: packet count == frames, pts strictly increasing, pts == dts (no reordering).

Usage: python s3_big.py <selection.json> <enc_root> <out_dir>   (env SMOKE)
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from common import LAYOUTS, Family, load_selection, moov_info, write_json


def packets(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts,dts,flags",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.split()
    rows = [r.split(",") for r in out]
    return [(int(p), int(d) if d not in ("", "N/A") else None, "K" in f) for p, d, f in rows]


def probe_stream(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,width,height,pix_fmt,time_base,avg_frame_rate,nb_frames,has_b_frames",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


def main():
    sel_path, enc_root, out_dir = sys.argv[1:4]
    sel = load_selection(sel_path, os.environ.get("SMOKE", "0") == "1")
    cams, fps = sel["cams"], sel["fps"]
    t0 = time.perf_counter()
    report = {"ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0], "files": {}}
    for fam, groups in sel["active"].items():
        F = Family(enc_root, fam, groups, cams, fps)
        for bj in range(F.n_big()):
            for c in cams:
                dst = F.big(bj, c)
                dst.parent.mkdir(parents=True, exist_ok=True)
                lst = dst.with_suffix(".txt")
                lst.write_text("".join(f"file '{F.sep(3 * bj + k, c)}'\n" for k in range(3)))
                subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                                "-map", "0:v:0", "-c", "copy", str(dst)], check=True)
        for lay in LAYOUTS:
            for p, n in zip(F.files(lay), [n for n in F.frames_per_file(lay) for _ in (cams if lay != "STACK" else [0])], strict=True):
                pk = packets(p)
                pts = [x[0] for x in pk]
                st = probe_stream(p)
                report["files"][str(p)] = {
                    "family": fam, "layout": lay, "frames_expected": n, "packets": len(pk),
                    "pts_strictly_increasing": all(b > a for a, b in zip(pts, pts[1:])),
                    "pts_eq_dts": all(d is None or d == q for q, d, _ in pk),
                    "keyframes": sum(k for *_, k in pk), "stream": st, **moov_info(p),
                }
    bad = {k: v for k, v in report["files"].items()
           if v["packets"] != v["frames_expected"] or not v["pts_strictly_increasing"] or not v["pts_eq_dts"]}
    report["problems"] = bad
    report["wall_s"] = round(time.perf_counter() - t0, 1)
    write_json(Path(out_dir) / "encode" / "files.json", report)
    summ = {}
    for v in report["files"].values():
        k = f"{v['family']}|{v['layout']}"
        s = summ.setdefault(k, {"files": 0, "frames": 0, "bytes": 0, "moov_bytes": 0, "keyframes": 0, "has_b_frames": set()})
        s["files"] += 1
        s["frames"] += v["frames_expected"]
        s["bytes"] += v["file_bytes"]
        s["moov_bytes"] += v["moov_bytes"] or 0
        s["keyframes"] += v["keyframes"]
        s["has_b_frames"].add(v["stream"].get("has_b_frames"))
    for s in summ.values():
        s["bytes_per_frame"] = round(s["bytes"] / s["frames"], 1)
        s["moov_bytes_per_frame"] = round(s["moov_bytes"] / s["frames"], 2)
        s["has_b_frames"] = sorted(s["has_b_frames"])
    write_json(Path(out_dir) / "encode" / "layout_sizes.json", summ)
    print(json.dumps(summ, indent=1, default=str))
    assert not bad, f"{len(bad)} files failed packet checks"


if __name__ == "__main__":
    main()
