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

Remote (hf://) model: see remote_model() below; fitted from stage remote, written to
model/remote_model.json and appended to model/summary.md. Local and remote parts run
independently; a missing input is reported as a MODEL PROBLEM (exit 1) without hiding the other.

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


def local_model(sel, out, smoke):
    """Local-disk model (stages opencost + loader). Writes model/model.json and model/summary.md."""
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


# ------------------------------------------------------------------ remote (hf://) model, from stage remote
def r2(xs, ys, a, b):
    my = sum(ys) / len(ys)
    ss = sum((y - my) ** 2 for y in ys)
    return 1 - sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)) / ss if ss else float("nan")


def p_block_miss(file_bytes, block):
    """Chance that a uniformly random frame is outside the one block the fsspec readahead cache holds."""
    return max(0.0, 1.0 - block / file_bytes) if file_bytes > 0 else 0.0


def remote_model(sel, out, smoke):
    """Remote cost model. Terms (all fitted from stage remote, single reader):
      per-fetch cost    lat_ms per sample = a + b x block fetches per sample   (seq / one arms, both regimes)
      per-open cost     open_ms = a + b x frames per file                       (open curve, per layout group)
      requests per open resolve / CDN / API calls in fs.open + VideoDecoder()   (open curve)
      fetches per access  open: 1 + p_miss(S), cached decoder: p_miss(S), p_miss = 1 - block / file bytes
    Predictions (EXTRAPOLATED) for uniform random map-style access with a 100-decoder LRU per worker."""
    rj = json.loads((out / "remote/remote.json").read_text())
    sizes = json.loads((out / "encode/layout_sizes.json").read_text())
    agg, oc, cfg = rj["aggregate"], rj["open_curve"], rj["config"]
    B = cfg["fsspec_block_size"] or 5 * 2**20
    lim = rj.get("resolver_limit_per_s") or float("nan")
    rm = {"label": "EXTRAPOLATED from a scaled-down benchmark" + (" (SMOKE: numbers are not results)" if smoke else ""),
          "family": cfg["family"], "block_bytes": B, "resolver_limit_per_s": lim, "resolver_limit_source": cfg.get("resolver_limit_source")}

    # 1. per-fetch cost
    pts = [(a["fetch_calls_per_sample"], a["lat_mean_ms"], k) for k, a in agg.items() if a["access"] in ("seq", "one")]
    fa, fb = lsq([p[0] for p in pts], [p[1] for p in pts])
    rm["fetch_fit"] = {"a_ms": fa, "b_ms_per_fetch": fb, "r2": r2([p[0] for p in pts], [p[1] for p in pts], fa, fb) if len(pts) > 2 else None,
                       "points": [{"arm": k, "fetches_per_sample": x, "lat_mean_ms": y, "pred_ms": fa + fb * x} for x, y, k in pts]}
    req_per_fetch = [a["requests_per_sample"] / a["fetch_calls_per_sample"] for a in agg.values() if a["fetch_calls_per_sample"] > 0]
    rm["requests_per_fetch"] = sum(req_per_fetch) / len(req_per_fetch) if req_per_fetch else None
    rm["resolves_per_fetch"] = (sum(a["resolve_requests_per_sample"] for a in agg.values()) /
                                max(1e-9, sum(a["fetch_calls_per_sample"] for a in agg.values())))
    rm["ms_per_request"] = fb / rm["requests_per_fetch"] if rm["requests_per_fetch"] else None

    # 2. per-open cost and requests per open
    rm["open_fit"] = {}
    for grp in ("SEP+BIG", "STACK"):
        cs = [c for c in oc.values() if c["layout"] in grp.split("+")]
        a, b = lsq([c["frames_per_file_mean"] for c in cs], [c["open_ms_median"] for c in cs])
        rm["open_fit"][grp] = {"a_ms": a, "b_ms_per_frame": b, "F_range": [min(c["frames_per_file_mean"] for c in cs), max(c["frames_per_file_mean"] for c in cs)] if cs else None,
                               "requests_per_open": sum(c["open_requests_mean"] for c in cs) / len(cs) if cs else None,
                               "resolves_per_open": sum(c["open_resolve_requests_mean"] for c in cs) / len(cs) if cs else None,
                               "api_per_open": sum(c["open_api_requests_mean"] for c in cs) / len(cs) if cs else None,
                               "fetches_per_open": sum(c["open_fetch_calls_mean"] for c in cs) / len(cs) if cs else None}
    # 3. check the fetch-count model: open curve (open + first decode) and steady arms (per file decode)
    chk = []
    for c in oc.values():
        chk.append({"where": f"open {c['family']}|{c['layout']}", "file_mb": c["file_bytes_mean"] / 1e6,
                    "measured": c["open_fetch_calls_mean"] + c["first_fetch_calls_mean"], "model": 1 + p_block_miss(c["file_bytes_mean"], B)})
    fam_bytes = {lay: oc[f"{cfg['family']}|{lay}"]["file_bytes_mean"] for lay in ("SEP", "STACK", "BIG") if f"{cfg['family']}|{lay}" in oc}
    for k, a in agg.items():
        if a["regime"] == "steady" and a["access"] in ("seq", "one") and a["layout"] in fam_bytes:
            chk.append({"where": f"steady {k}", "file_mb": fam_bytes[a["layout"]] / 1e6,
                        "measured": a["fetch_calls_per_sample"] / a["files_per_sample"], "model": p_block_miss(fam_bytes[a["layout"]], B)})
    rm["fetch_count_check"] = chk

    # 4. EXTRAPOLATED predictions
    bpf = {lay: sizes[f"{cfg['family']}|{lay}"]["bytes_per_frame"] for lay in ("SEP", "STACK") if f"{cfg['family']}|{lay}" in sizes}
    rs = sel["real_file_stats"]["all_cameras"]
    scen = [{"dataset": "lerobot/abc_130k_v3_train", "files_total": rs["files"], "cams": 3, "frames_per_file": rs["frames_per_file_median"], "pix": 1.0,
             "source": f"meta/episodes @ {sel['revision'][:8]}: {rs['files']} files, median frames/file {rs['frames_per_file_median']:.0f}"},
            {"dataset": "cadene/droid_1.0.1_v30", "files_total": 6144, "cams": 3, "frames_per_file": 27607757 * 3 / 6144, "pix": 180 * 320 / PIX,
             "source": "info.json total_frames 27,607,757; 6144 files (2048 per camera); mean frames/file; bytes/frame scaled by pixels (ASSUMED)"}]
    C = 100
    fmax = max((c["frames_per_file_mean"] for c in oc.values()), default=0)
    preds = []
    for sc in scen:
        per_cam = sc["files_total"] / sc["cams"]
        variants = {"SEP (as stored)": ("SEP", per_cam, sc["frames_per_file"], bpf.get("SEP", 0) * sc["pix"]),
                    "STACK, same frames/file (files / 3)": ("STACK", per_cam, sc["frames_per_file"], bpf.get("STACK", 0) * sc["pix"]),
                    "BIG (3x frames/file, files / 3)": ("BIG", per_cam / 3, sc["frames_per_file"] * 3, bpf.get("SEP", 0) * sc["pix"])}
        for pat, k_cams in (("all", sc["cams"]), ("one", 1)):
            for vname, (lay, U, F, bytes_pf) in variants.items():
                k = 1 if lay == "STACK" else k_cams
                h = h_model(C, k, max(1.0, U))
                S = F * bytes_pf
                pm = p_block_miss(S, B)
                opens = k * (1 - h)
                fetches = opens * (1 + pm) + (k - opens) * pm
                grp = "STACK" if lay == "STACK" else "SEP+BIG"
                of = rm["open_fit"].get(grp, {})
                api = opens * (of.get("api_per_open") or 0.0)
                resolves = fetches * rm["resolves_per_fetch"]
                lat = fa + fb * fetches
                preds.append({"label": "EXTRAPOLATED", "dataset": sc["dataset"], "variant": vname, "pattern": pat, "cache": C, "units": U,
                              "frames_per_file": F, "file_mb": S / 1e6, "hit_rate_model": h, "opens_per_sample": opens,
                              "fetches_per_sample": fetches, "resolves_per_sample": resolves, "api_per_sample_upper": api,
                              "mb_per_sample": fetches * min(S, B) / 1e6, "lat_ms_single_stream_seq": lat, "sps_single_stream": 1e3 / lat if lat > 0 else None,
                              "sps_cap_resolver_limit": lim / resolves if resolves else None,
                              "streams_to_hit_resolver_limit": (lim / resolves) / (1e3 / lat) if resolves and lat > 0 else None,
                              "beyond_measured_frames_range": F > fmax * 1.05, "source": sc["source"]})
        # sequential streaming read (analytic only, NOT measured): one fetch per block of consecutive frames
        for vname, lay, keys, bytes_pf in (("SEP (as stored)", "SEP", sc["cams"], bpf.get("SEP", 0) * sc["pix"]),
                                           ("STACK", "STACK", 1, bpf.get("STACK", 0) * sc["pix"])):
            fetches = keys * bytes_pf / B if B else float("nan")
            preds.append({"label": "ANALYTIC (not measured): sequential in-order reads", "dataset": sc["dataset"], "variant": vname, "pattern": "all",
                          "fetches_per_sample": fetches, "resolves_per_sample": fetches * rm["resolves_per_fetch"],
                          "sps_cap_resolver_limit": lim / (fetches * rm["resolves_per_fetch"]) if fetches else None, "source": sc["source"]})
    rm["predictions"] = preds
    rm["assumptions"] = [
        "Access = uniform random frames (map-style shuffle) with a 100-decoder LRU per worker; LRU hit rate h = min(1, floor(C/k)/U).",
        f"fsspec readahead cache holds one block of {B} bytes per open handle; a random frame needs a new block fetch with chance 1 - block/file bytes; an open costs one fetch (moov is at the start, faststart).",
        "Each block fetch = 1 resolve call (counts against the per-token 'resolvers' budget) + 1 CDN call; measured, see resolves_per_fetch.",
        "Latency per sample is linear in block fetches per sample (single reader, cameras one after another); fitted on this job's network path to the Hub.",
        "sps_cap_resolver_limit = resolver budget per second / resolves per sample: a per-token ceiling that no amount of hardware removes.",
        "DROID bytes per frame = ABC bytes per frame x pixel ratio (ASSUMED; not measured). Frames/file beyond the measured range are flagged.",
        "API calls per open (paths-info when the fsspec dircache misses) are an upper bound: a long-lived worker caches them.",
        "Sequential rows are analytic only (no measurement of in-order streaming reads here).",
    ]
    write_json(out / "model" / "remote_model.json", rm)
    L = ["", f"# Remote (hf://) model ({rm['label']})", "",
         f"Per-fetch fit on {len(pts)} single-reader arms: lat_ms = {fa:.1f} + {fb:.1f} x fetches/sample (R^2 {rm['fetch_fit']['r2']}). "
         f"Requests per fetch {rm['requests_per_fetch']:.2f}, resolves per fetch {rm['resolves_per_fetch']:.2f}, ms per request {rm['ms_per_request']:.1f}. "
         f"Resolver limit {lim:.1f}/s per token ({rm['resolver_limit_source']}).", "",
         "| open fit | a ms | b us/frame | F range | req/open | resolve/open | API/open | fetch/open |", "|---|---|---|---|---|---|---|---|"]
    for g, f in rm["open_fit"].items():
        L.append(f"| {g} | {f['a_ms']:.1f} | {f['b_ms_per_frame'] * 1e3:.3f} | {f['F_range']} | {f['requests_per_open']} | {f['resolves_per_open']} | {f['api_per_open']} | {f['fetches_per_open']} |")
    L += ["", "| fetch-count check | file MB | measured fetches | model |", "|---|---|---|---|"]
    for c in chk:
        L.append(f"| {c['where']} | {c['file_mb']:.1f} | {c['measured']:.2f} | {c['model']:.2f} |")
    L += ["", "## Remote predictions (EXTRAPOLATED unless marked ANALYTIC)", "",
          "| dataset | variant | pattern | h | opens/smp | fetches/smp | resolves/smp | MB/smp | ms/smp 1 reader | sps 1 reader | sps cap (resolver limit) | beyond F range |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in preds:
        if p["label"] != "EXTRAPOLATED":
            continue
        L.append(f"| {p['dataset']} | {p['variant']} | {p['pattern']} | {p['hit_rate_model']:.4f} | {p['opens_per_sample']:.2f} | {p['fetches_per_sample']:.2f} | "
                 f"{p['resolves_per_sample']:.2f} | {p['mb_per_sample']:.1f} | {p['lat_ms_single_stream_seq']:.0f} | {p['sps_single_stream']:.2f} | "
                 f"{p['sps_cap_resolver_limit']:.1f} | {'yes' if p['beyond_measured_frames_range'] else ''} |")
    L += ["", "| ANALYTIC sequential | variant | fetches/smp | sps cap (resolver limit) |", "|---|---|---|---|"]
    for p in preds:
        if p["label"].startswith("ANALYTIC"):
            L.append(f"| {p['dataset']} | {p['variant']} | {p['fetches_per_sample']:.5f} | {p['sps_cap_resolver_limit']:.0f} |")
    L += ["", "## Remote assumptions", ""] + [f"- {a}" for a in rm["assumptions"]]
    return L


def main():
    sel_path, out_dir = sys.argv[1:3]
    smoke = os.environ.get("SMOKE", "0") == "1"
    out = Path(out_dir)
    sel = json.loads(Path(sel_path).read_text())
    failed = []
    local_ok = all((out / f).exists() for f in ("opencost/opencost.json", "loader/arms.json", "encode/layout_sizes.json", "loader/config.json"))
    if local_ok:
        try:
            local_model(sel, out, smoke)
        except Exception as exc:
            failed.append(f"local model: {type(exc).__name__}: {exc}")
    else:
        failed.append("local model: inputs missing (opencost / loader / encode)")
    lines = []
    if (out / "remote/remote.json").exists():
        try:
            lines = remote_model(sel, out, smoke)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            failed.append(f"remote model: {type(exc).__name__}: {exc}")
    else:
        failed.append("remote model: remote/remote.json missing")
    summ = out / "model" / "summary.md"
    summ.parent.mkdir(parents=True, exist_ok=True)
    prev = summ.read_text() if summ.exists() and local_ok else ""
    rsum = (out / "remote/summary.md").read_text() if (out / "remote/summary.md").exists() else ""
    summ.write_text(prev + "\n".join(lines) + "\n" + (f"\n---\n\n{rsum}" if rsum else "") + "".join(f"\n- MODEL PROBLEM: {f}" for f in failed) + "\n")
    print("\n".join(lines))
    for f in failed:
        print("MODEL PROBLEM:", f)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
