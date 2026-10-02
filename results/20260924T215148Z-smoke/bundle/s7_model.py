"""Stage model: fit, validate, extrapolate. Everything this script predicts is labelled EXTRAPOLATED.

Model (per layout, pattern, workers, page-cache state):
  worker_s_per_sample = opens_per_sample x open_cost(frames_per_file) + decode_cost
  opens_per_sample    = k x (1 - h),   h = min(1, floor(C / k) / U)
    C = decoder cache size per worker, k = keys per unit (cameras decoded per sample for SEP/BIG,
    1 for STACK), U = units a uniform sample chooses among (files per camera for SEP/BIG, stacked
    files for STACK). This is the LRU hit rate for independent uniform references.
  worker_s_per_sample is W / samples_per_s (worker-seconds per sample, from the DataLoader runs).

Fits:
  open_cost(F)  least squares a + b F on the isolated open-cost curve (stage opencost), where the
                cost of one miss = construct + close + (first-frame decode - steady decode)
  decode_cost   intercept, and kappa = slope / open_cost(F_thrash), of a line through the thrash
                sweep points (opens_per_sample, worker_s_per_sample)
Validation: predict the checkpoint family's DataLoader arms (predicted and measured opens) and
compare with the measurements. Predictions: real datasets from their file counts and frames per
file, cache 100 (LeRobot default), the measured worker counts.

Usage: python s7_model.py <selection.json> <out_dir>   (env SMOKE)
"""

import json
import os
import sys
from pathlib import Path

from common import write_json

PIX = 224 * 224


def lsq(xs, ys):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1 or len(set(xs)) == 1:
        return sum(ys) / n, 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def h_model(C, k, U):
    """LRU hit rate, uniform independent references over U units of k keys each (a unit is only
    cached whole, so C < k holds nothing)."""
    return min(1.0, (C // k) / U)


def main():
    sel_path, out_dir = sys.argv[1:3]
    smoke = os.environ.get("SMOKE", "0") == "1"
    out = Path(out_dir)
    sel = json.loads(Path(sel_path).read_text())
    oc = json.loads((out / "opencost/opencost.json").read_text())
    arms = json.loads((out / "loader/arms.json").read_text())
    sizes = json.loads((out / "encode/layout_sizes.json").read_text())
    lcfg = json.loads((out / "loader/config.json").read_text())
    ck_fam = lcfg["ckpt_family"]
    res = {"label": "EXTRAPOLATED from a scaled-down benchmark" + (" (SMOKE: numbers are not results)" if smoke else ""),
           "open_cost_fit": {}, "open_cost_points": {}, "thrash_fit": {}, "hit_rate_check": [], "validation": [], "predictions": []}

    # ---- 1. open-cost curve
    for state in ("warm", "cold"):
        for lay in ("SEP", "STACK", "BIG", "SEP+BIG"):
            pts = []
            for key, c in oc["cells"].items():
                if c["layout"] not in lay.split("+"):
                    continue
                s = c[state]
                cost = s["open_ms"]["median"] + s["close_ms"]["median"] + max(0.0, s["first_ms"]["median"] - s["decode_ms"]["median"])
                pts.append((c["frames_per_file_mean"], cost, key, c["moov_bytes_mean"], c["rss"].get("mb_per_decoder_after_first_frame")))
            a, b = lsq([p[0] for p in pts], [p[1] for p in pts])
            res["open_cost_fit"][f"{lay}|{state}"] = {"a_ms": a, "b_ms_per_frame": b, "F_range": [min(p[0] for p in pts), max(p[0] for p in pts)] if pts else None}
            res["open_cost_points"][f"{lay}|{state}"] = [{"frames_per_file": p[0], "miss_cost_ms": p[1], "cell": p[2], "moov_bytes": p[3], "mb_per_decoder": p[4]} for p in pts]

    def open_cost_ms(lay, state, F):
        key = f"{'SEP+BIG' if lay in ('SEP', 'BIG') else 'STACK'}|{state}"
        f = res["open_cost_fit"][key]
        return f["a_ms"] + f["b_ms_per_frame"] * F

    # ---- 2. thrash sweep fit + hit-rate check
    groups = {}
    for key, a in arms.items():
        a["worker_s_per_sample"] = a["workers"] / a["samples_per_s"]
        a["h_model"] = h_model(a["cache"], a["keys_per_unit"], a["units"])
        if a["family"] == "thrash":
            groups.setdefault((a["layout"], a["pattern"], a["workers"], a["state"]), []).append(a)
            res["hit_rate_check"].append({"arm": key, "cache": a["cache"], "ratio_vs_sep_files": a["ratio"], "cache_over_own_files": a["cache_over_own_files"],
                                          "h_measured": a["hit_rate"], "h_model": a["h_model"], "abs_err": abs(a["hit_rate"] - a["h_model"])})
    for (lay, pat, W, state), rs in groups.items():
        d, slope = lsq([r["opens_per_sample"] for r in rs], [r["worker_s_per_sample"] for r in rs])
        d_cpu, slope_cpu = lsq([r["opens_per_sample"] for r in rs], [r["worker_cpu_s_per_sample"] for r in rs])
        F = rs[0]["frames_per_file_mean"]
        iso = open_cost_ms(lay, state, F) / 1e3
        res["thrash_fit"][f"{lay}|{pat}|W{W}|{state}"] = {
            "points": len(rs), "decode_cost_ms": d * 1e3, "loader_open_cost_ms": slope * 1e3, "isolated_open_cost_ms": iso * 1e3,
            "kappa": slope / iso if iso > 0 and len(rs) > 1 else None, "cpu_decode_ms": d_cpu * 1e3, "cpu_open_ms": slope_cpu * 1e3,
            "frames_per_file": F, "opens_range": [min(r["opens_per_sample"] for r in rs), max(r["opens_per_sample"] for r in rs)]}

    def params(lay, pat, W, state):
        f = res["thrash_fit"].get(f"{lay}|{pat}|W{W}|{state}")
        if not f:
            return None
        kappa = f["kappa"] if f["kappa"] and f["kappa"] > 0 else 1.0
        return f["decode_cost_ms"] / 1e3, kappa

    def predict(lay, pat, W, state, F, opens, pix_scale=1.0):
        p = params(lay, pat, W, state)
        if p is None:
            return None, None
        d, kappa = p
        t = opens * kappa * open_cost_ms(lay, state, F) / 1e3 + d * pix_scale
        t_nocal = opens * open_cost_ms(lay, state, F) / 1e3 + d * pix_scale
        return t, t_nocal

    # ---- 3. validation on the checkpoint family
    for key, a in arms.items():
        if a["family"] != ck_fam:
            continue
        k, U = a["keys_per_unit"], a["units"]
        o_pred = k * (1 - a["h_model"])
        t_pred, t_nocal = predict(a["layout"], a["pattern"], a["workers"], a["state"], a["frames_per_file_mean"], o_pred)
        t_meas_opens, _ = predict(a["layout"], a["pattern"], a["workers"], a["state"], a["frames_per_file_mean"], a["opens_per_sample"])
        if t_pred is None:
            continue
        sps_pred = a["workers"] / t_pred
        res["validation"].append({
            "arm": key, "frames_per_file": a["frames_per_file_mean"], "cache": a["cache"],
            "sps_measured": a["samples_per_s"], "sps_ci95_rel": a["ci95_rel"], "sps_pred": sps_pred,
            "sps_pred_uncalibrated": a["workers"] / t_nocal, "sps_pred_with_measured_opens": a["workers"] / t_meas_opens,
            "err_pct": 100 * (sps_pred / a["samples_per_s"] - 1),
            "err_pct_measured_opens": 100 * (a["workers"] / t_meas_opens / a["samples_per_s"] - 1),
            "opens_measured": a["opens_per_sample"], "opens_pred": o_pred, "h_measured": a["hit_rate"], "h_model": a["h_model"]})

    # ---- 4. predictions for real datasets
    rs = sel["real_file_stats"]
    bpf = {lay: sizes[f"thrash|{lay}"]["bytes_per_frame"] for lay in ("SEP", "STACK")}
    stack_frames_ratio = bpf["STACK"] / bpf["SEP"]  # bytes per stacked frame / bytes per camera frame (~3)
    abc_files, abc_F = rs["all_cameras"]["files"], rs["all_cameras"]["frames_per_file_median"]
    scen = [
        {"dataset": "lerobot/abc_130k_v3_train", "files_total": abc_files, "cams": 3, "frames_per_file": abc_F, "h": 224, "w": 224,
         "source": f"meta/episodes @ {sel['revision'][:8]}: {abc_files} files, median frames/file {abc_F:.0f}"},
        {"dataset": "lerobot/droid_1.0.1", "files_total": 813, "cams": 3, "frames_per_file": 27630375 * 3 / 813, "h": 180, "w": 320,
         "source": "info.json total_frames 27,630,375; 813 files (312/310/191 per camera); mean frames/file"},
        {"dataset": "cadene/droid_1.0.1_v30", "files_total": 6144, "cams": 3, "frames_per_file": 27607757 * 3 / 6144, "h": 180, "w": 320,
         "source": "info.json total_frames 27,607,757; 6144 files (2048 per camera); mean frames/file"},
        {"dataset": "control: 2 files per camera", "files_total": 6, "cams": 3, "frames_per_file": abc_F, "h": 224, "w": 224,
         "source": "synthetic control (everything fits in the cache)"},
    ]
    C = 100
    Fmax_meas = max(p["frames_per_file"] for p in res["open_cost_points"]["SEP+BIG|warm"]) if res["open_cost_points"]["SEP+BIG|warm"] else 0
    for sc in scen:
        per_cam = sc["files_total"] / sc["cams"]
        variants = {
            "SEP (as stored)": ("SEP", per_cam, sc["frames_per_file"]),
            "STACK, same frames/file (files / 3)": ("STACK", per_cam, sc["frames_per_file"]),
            "STACK, same bytes/file (LeRobot size rollover)": ("STACK", per_cam * stack_frames_ratio, sc["frames_per_file"] / stack_frames_ratio),
            "BIG (3x frames/file, files / 3)": ("BIG", per_cam / 3, sc["frames_per_file"] * 3),
        }
        pix_scale = sc["h"] * sc["w"] / PIX
        for W in lcfg["workers"]:
            for state in lcfg["states"]:
                for pat in lcfg["patterns"]:
                    k_cams = sc["cams"] if pat == "all" else 1
                    for vname, (lay, U, F) in variants.items():
                        k = 1 if lay == "STACK" else k_cams
                        h = h_model(C, k, max(1.0, U))
                        t, t_nocal = predict(lay, pat, W, state, F, k * (1 - h), pix_scale)
                        if t is None:
                            continue
                        res["predictions"].append({
                            "label": "EXTRAPOLATED", "dataset": sc["dataset"], "variant": vname, "layout_measured": lay, "pattern": pat,
                            "workers": W, "state": state, "cache": C, "units": U, "frames_per_file": F, "hit_rate_model": h,
                            "opens_per_sample": k * (1 - h), "sps_pred": W / t, "sps_pred_uncalibrated": W / t_nocal,
                            "beyond_measured_frames_range": F > Fmax_meas * 1.05, "pixel_scale_on_decode": pix_scale, "source": sc["source"]})
    res["assumptions"] = [
        "LRU hit rate for independent uniform frame references: h = min(1, floor(C/k)/U); checked against the thrash sweep (hit_rate_check).",
        "Workers are CPU-bound and independent: samples/s = W / worker_s_per_sample; measured only for W in the loader config, on this flavor.",
        "open_cost(F) linear in frames per file, fitted on the isolated curve; scaled by kappa (in-loader / isolated open cost at the thrash length).",
        "decode_cost independent of frames per file (validated on the checkpoint family) and proportional to pixels when applied to 180x320 DROID.",
        "Open cost for DROID uses the 224x224 curve (index size depends on frames, not pixels); frames/file beyond the measured range is a linear extrapolation (flagged).",
        f"STACK same-bytes variant: frames per file divided by (STACK bytes per stacked frame / SEP bytes per camera frame) = {stack_frames_ratio:.3f} (thrash family); file count multiplied by it.",
        "cold = posix_fadvise DONTNEED eviction on this container's local disk; real training storage (network FS, Hub streaming) is not modeled.",
        "Same encoder settings as the source (libsvtav1, crf 30, g 2); other codecs or GOPs are not covered.",
    ]
    write_json(out / "model" / "model.json", res)

    # ---- markdown summary
    L = [f"# DATA-11 benchmark model ({res['label']})", ""]
    L += ["## Open-cost fits (ms per miss = construct + close + extra first-frame decode)", "", "| layout | state | a ms | b us/frame | F range |", "|---|---|---|---|---|"]
    for k, f in res["open_cost_fit"].items():
        lay, st = k.split("|")
        L.append(f"| {lay} | {st} | {f['a_ms']:.3f} | {f['b_ms_per_frame'] * 1e3:.4f} | {f['F_range']} |")
    L += ["", "## Thrash sweep fits (worker-s per sample vs opens per sample)", "", "| arm | pts | decode ms | loader open ms | isolated open ms | kappa |", "|---|---|---|---|---|---|"]
    for k, f in res["thrash_fit"].items():
        L.append(f"| {k} | {f['points']} | {f['decode_cost_ms']:.3f} | {f['loader_open_cost_ms']:.3f} | {f['isolated_open_cost_ms']:.3f} | {f['kappa'] if f['kappa'] is None else round(f['kappa'], 3)} |")
    if res["hit_rate_check"]:
        e = [x["abs_err"] for x in res["hit_rate_check"]]
        L += ["", f"Hit-rate model vs measured (thrash arms): max abs err {max(e):.3f}, mean {sum(e) / len(e):.3f} over {len(e)} arms."]
    L += ["", f"## Validation on {ck_fam}", "", "| arm | measured sps | pred sps | err % | err % (measured opens) | h meas / model |", "|---|---|---|---|---|---|"]
    for v in res["validation"]:
        L.append(f"| {v['arm']} | {v['sps_measured']:.0f} | {v['sps_pred']:.0f} | {v['err_pct']:+.1f} | {v['err_pct_measured_opens']:+.1f} | {v['h_measured']:.3f} / {v['h_model']:.3f} |")
    L += ["", "## Predictions (EXTRAPOLATED; cache 100 per worker)", "", "| dataset | variant | pattern | W | state | h | opens/sample | pred sps | beyond F range |", "|---|---|---|---|---|---|---|---|---|"]
    for p in res["predictions"]:
        L.append(f"| {p['dataset']} | {p['variant']} | {p['pattern']} | {p['workers']} | {p['state']} | {p['hit_rate_model']:.3f} | {p['opens_per_sample']:.2f} | {p['sps_pred']:.0f} | {'yes' if p['beyond_measured_frames_range'] else ''} |")
    L += ["", "## Assumptions", ""] + [f"- {a}" for a in res["assumptions"]]
    (out / "model" / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
