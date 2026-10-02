"""Stage verify (TorchCodec process): correctness of every layout before anything is timed.

  timestamps  for sampled frames of every file: LeRobot's ts -> round(ts * average_fps) index maps
              back to the same frame, and the returned pts equals index / fps (max abs error)
  cam order   STACK crop row r vs the SEP decode of camera c at the same frame: PSNR matrix; the
              diagonal must be high and every off-diagonal low (a swapped order fails)
  BIG         every checked BIG frame bit-identical to the SEP frame it was copied from
              (VERIFY_BIG=all: every frame; otherwise group boundaries + VERIFY_BIG random frames per file)

Usage: python s4_verify.py <selection.json> <enc_root> <out_dir>   (env SMOKE, VERIFY_BIG, VERIFY_N)
"""

import os
import random
import sys
import time
from pathlib import Path

import torch
from torchcodec.decoders import VideoDecoder

from common import H, LAYOUTS, Family, load_selection, write_json


def psnr(a, b):
    mse = ((a.double() - b.double()) ** 2).mean().item()
    return 99.0 if mse == 0 else 10 * torch.log10(torch.tensor(255.0**2 / mse)).item()


def dec(p):
    return VideoDecoder(str(p), seek_mode="approximate")


def main():
    sel_path, enc_root, out_dir = sys.argv[1:4]
    smoke = os.environ.get("SMOKE", "0") == "1"
    n_ts = int(os.environ.get("VERIFY_N", "16"))
    big_mode = os.environ.get("VERIFY_BIG", "all" if smoke else "300")
    torch.set_num_threads(1)
    sel = load_selection(sel_path, smoke)
    cams, fps = sel["cams"], sel["fps"]
    rng = random.Random(0)
    t0 = time.perf_counter()
    rep = {"timestamps": {}, "cam_order": {}, "big_identical": {}, "problems": []}
    for fam, groups in sel["active"].items():
        F = Family(enc_root, fam, groups, cams, fps)
        # --- timestamps and metadata, every file of every layout
        worst = {lay: 0.0 for lay in LAYOUTS}
        for lay in LAYOUTS:
            nfs = F.frames_per_file(lay)
            per_file_n = [n for n in nfs for _ in (cams if lay != "STACK" else [0])]
            for p, n in zip(F.files(lay), per_file_n, strict=True):
                d = dec(p)
                md = d.metadata
                exp_hw = (H * 3, H) if lay == "STACK" else (H, H)
                if md.num_frames != n or abs(md.average_fps - fps) > 1e-6 or (md.height, md.width) != exp_hw:
                    rep["problems"].append(f"metadata {p}: frames {md.num_frames}/{n} fps {md.average_fps} hw {(md.height, md.width)}")
                idx = sorted({0, n - 1, *rng.sample(range(n), min(n_ts, n))})
                mapped = [round((i / fps) * md.average_fps) for i in idx]
                if mapped != idx:
                    rep["problems"].append(f"index mapping {p}")
                fb = d.get_frames_at(indices=mapped)
                err = max(abs(float(t) - i / fps) for t, i in zip(fb.pts_seconds, idx))
                worst[lay] = max(worst[lay], err)
                del d
        rep["timestamps"][fam] = {lay: {"max_abs_pts_error_s": e, "ok": e < 1e-6} for lay, e in worst.items()}
        for lay, e in worst.items():
            if e >= 1e-6:
                rep["problems"].append(f"pts error {fam}/{lay}: {e}")
        # --- camera order: STACK crops vs SEP decodes
        mats = []
        for gi in range(len(groups)):
            n = F.n[gi]
            idx = sorted(rng.sample(range(n), min(4, n)))
            st = dec(F.stack(gi)).get_frames_at(indices=idx).data  # [k,3,672,224]
            sep = {c: dec(F.sep(gi, c)).get_frames_at(indices=idx).data for c in cams}
            m = [[psnr(st[:, :, r * H : (r + 1) * H, :], sep[c]) for c in cams] for r in range(len(cams))]
            mats.append(m)
        k = len(cams)
        diag = [m[i][i] for m in mats for i in range(k)]
        off = [m[i][j] for m in mats for i in range(k) for j in range(k) if i != j]
        ok = min(diag) > 30 and min(diag) > max(off) + 10
        rep["cam_order"][fam] = {"diag_psnr_min": min(diag), "diag_psnr_mean": sum(diag) / len(diag),
                                 "offdiag_psnr_max": max(off), "offdiag_psnr_mean": sum(off) / len(off), "ok": ok,
                                 "first_group_matrix_rows_stack_cols_sep": mats[0], "order": cams}
        if not ok:
            rep["problems"].append(f"camera order check failed for {fam}")
        # --- BIG bit-identical to SEP
        checked = mism = 0
        for bj in range(F.n_big()):
            for c in cams:
                big = dec(F.big(bj, c))
                off0 = 0
                for k3 in range(3):
                    gi = 3 * bj + k3
                    n = F.n[gi]
                    sep = dec(F.sep(gi, c))
                    if big_mode == "all":
                        for a in range(0, n, 256):
                            b = min(n, a + 256)
                            x = big.get_frames_in_range(start=off0 + a, stop=off0 + b).data
                            y = sep.get_frames_in_range(start=a, stop=b).data
                            checked += b - a
                            mism += int((x != y).flatten(1).any(1).sum())
                    else:
                        m = int(big_mode) // 3
                        idx = sorted({0, 1, n - 2, n - 1, *rng.sample(range(n), min(m, n))})
                        x = big.get_frames_at(indices=[off0 + i for i in idx])
                        y = sep.get_frames_at(indices=idx)
                        checked += len(idx)
                        mism += int((x.data != y.data).flatten(1).any(1).sum())
                        perr = max(abs(float(t) - (off0 + i) / fps) for t, i in zip(x.pts_seconds, idx))
                        if perr >= 1e-6:
                            rep["problems"].append(f"BIG pts error {F.big(bj, c)}: {perr}")
                    off0 += n
                    del sep
                del big
        rep["big_identical"][fam] = {"frames_checked": checked, "mismatched_frames": mism, "mode": big_mode, "ok": mism == 0}
        if mism:
            rep["problems"].append(f"BIG mismatch {fam}: {mism}/{checked}")
        print(fam, rep["timestamps"][fam], rep["cam_order"][fam]["diag_psnr_min"], rep["cam_order"][fam]["offdiag_psnr_max"],
              rep["big_identical"][fam], flush=True)
    rep["wall_s"] = round(time.perf_counter() - t0, 1)
    rep["ok"] = not rep["problems"]
    write_json(Path(out_dir) / "verify" / "verify.json", rep)
    print("verify ok" if rep["ok"] else f"verify PROBLEMS: {rep['problems']}")
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
