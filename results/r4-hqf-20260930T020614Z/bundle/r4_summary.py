"""Round 4 stage summary: results/<run>/summary.md from the stage JSON files (slice, encode, build, check, stream, map).

Usage: python r4_summary.py <out_dir>
"""

import sys
from pathlib import Path

from r4_common import DEFAULT_CACHE, fmt_ci, read_json, streaming_active_shards


def load(out, rel):
    p = out / rel
    return read_json(p) if p.exists() else None


def f(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    out = Path(sys.argv[1])
    sl, enc, bd, ck = (load(out, r) for r in ("slice/slice.json", "encode/encode.json", "build/build.json", "check/check.json"))
    st, mp = load(out, "stream/stream.json"), load(out, "map/map.json")
    L = []
    if sl is None:
        (out / "summary.md").write_text("# DATA-11 round 4\n\nNo slice.json: the job failed before the slice stage finished.\n")
        return
    cams, ncam = sl["cams"], len(sl["cams"])
    real, s = sl["real"], sl["slice"]
    L += [f"# DATA-11 round 4: {sl['dataset']} ({sl['repo']})", "",
          f"Source `{sl['repo']}@{sl['revision']}` (main now `{sl['main_sha_now']}`), {real['episodes']} episodes, {real['frames']} frames, "
          f"fps {sl['fps']}. Cameras (STACK order, top to bottom): " + ", ".join(f"`{c}` {sl['shapes'][c][0]}x{sl['shapes'][c][1]}" for c in cams) + ".",
          f"lerobot `{sl.get('lerobot_sha')}` (PR #4702 head). SMOKE={int(sl['smoke'])}.", "",
          "## Slice", "",
          f"- {s['n_episodes']} whole consecutive episodes, {s['frames']} frames per group (cap {sl['max_frames_cap']}, cap applied: {s['cap_applied']}). "
          f"Consecutive frames in the first file of every camera: {s['consecutive_frames_in_first_files_all_cameras']}.",
          f"- Real median frames per video file: {real['frames_per_file_median_all_cameras']:.0f} (slice / median = {s['slice_over_real_median']:.2f}).",
          f"- Real video files per camera: " + ", ".join(f"{c} {n}" for c, n in real["video_files_per_camera"].items()) + "; real parquet data files: "
          f"{real['parquet_data_files_in_repo']} (episode table: {real['parquet_data_files_from_episodes']}); streaming load_dataset num_shards: "
          f"{real['hf_streaming_num_shards']}.", ""]
    if enc:
        lay = enc["stack_layout"]
        L += ["## Encode", "",
              f"LeRobot defaults from the installed code: `{sl['encoder_defaults']['codec_options']}` ({sl['encoder_defaults']['vcodec']}, "
              f"{sl['encoder_defaults']['pix_fmt']}). SVT-AV1 reported: {enc['svt_reported'].get('mapped') or '-'}; version "
              f"{enc['svt_reported'].get('version') or '-'}; lp {enc['svt_lp']}.",
              f"Stacked frame {lay['height']}x{lay['width']}; pad pixels {lay['pad_pixels_per_frame']} per frame ({100 * lay['pad_pixel_share']:.1f}% of the "
              f"stacked frame, {lay['pad_raw_rgb_bytes_per_frame']} raw RGB bytes). Mux {enc['mux_options'] or 'default'} (source faststart: {enc['source_faststart']}).",
              "", "| file | shape | MB | encoder feed s | flush s |", "|---|---|---|---|---|"]
        for k, v in enc["files"].items():
            L.append(f"| {k} | {v['shape'][0]}x{v['shape'][1]} | {v['bytes'] / 1e6:.2f} | {v['feed_s']} | {v['flush_s']} |")
        L += ["", f"SEP total {enc['sep_bytes_total'] / 1e6:.2f} MB, STACK {enc['stack_bytes'] / 1e6:.2f} MB (STACK / SEP = {enc['stack_over_sep_bytes']:.3f}). "
              f"Encode wall {enc['wall_s']} s for {enc['frames']} frames (decode {enc['decode_s']} s, process CPU {enc['process_cpu_s']} s); "
              "SEP and STACK share one loop, so their encode costs are not separated.", ""]
    if bd:
        L += ["## Datasets", ""]
        for k, d in bd["datasets"].items():
            L.append(f"- {k}: `{d['repo']}` commit `{d.get('commit', 'not uploaded')}`, {d['files_per_key']} files per key, {d['episodes']} episodes, "
                     f"{d['frames']} frames, keys {d['video_keys']}, private {d.get('private')}.")
        L.append("")
    if ck:
        L += ["## Checks", "", "| camera | PSNR SEP vs source | STACK crop vs source | STACK crop vs SEP (min) |", "|---|---|---|---|"]
        for c, v in ck["psnr"].items():
            L.append(f"| {c} | {v['sep_vs_src']['mean']:.2f} | {v['stack_vs_src']['mean']:.2f} | {v['stack_vs_sep']['mean']:.2f} ({v['stack_vs_sep']['min']:.2f}) |")
        tl = ck.get("through_lerobot", {})
        L += ["", f"- Camera order ok: {ck['camera_order_ok']} ({ck['camera_order_comparable_pairs']} same-shape pairs compared).",
              f"- Remote (hf://) == local, bit-exact: {ck.get('remote_equal')}.",
              f"- Through LeRobotDataset: SEP item == file frame {tl.get('sep_item_equals_file_frame')}; SEP1 == SEP first camera "
              f"{tl.get('sep1_equals_sep_first_camera')}; STACK crop vs SEP item PSNR min {tl.get('stack_crop_vs_sep_item_psnr_min')}.",
              f"- Overall check ok: {ck.get('ok')} (PSNR threshold {ck.get('psnr_threshold')} dB, STACK vs source min {ck['psnr_min_stack_vs_src']:.2f}).", ""]
    if st:
        L += ["## StreamingLeRobotDataset over hf:// (one process = one DataLoader worker)", "",
              f"{st['config']['repeats']} repeats, arms interleaved, 95% CI over repeats. Local copies instead of hf://: {st['config']['local']}.", "",
              "| arm | frames/s | CPU ms/frame | mp4 fetch/frame | mp4 MB/frame | decoder hit | open ms | first yield s |", "|---|---|---|---|---|---|---|---|"]
        for k, a in st.get("summary", {}).items():
            L.append(f"| {k} | {fmt_ci(a['frames_per_s'], a['frames_per_s_ci95'])} | {fmt_ci(a['cpu_ms_per_frame'], a['cpu_ms_per_frame_ci95'])} | "
                     f"{f(a['mp4_fetches_per_frame'], 4)} | {f(a['mp4_bytes_per_frame'] / 1e6 if a['mp4_bytes_per_frame'] is not None else None, 4)} | "
                     f"{f(a['dec_hit_rate'], 3)} | {f(a['open_ms_per_open'], 0)} | {f(a['first_yield_s'], 1)} |")
        errs = [r for r in st["runs"] if "error" in r]
        if errs:
            L.append(f"\nErrors: {[(r['arm'], r['error'][:200]) for r in errs]}")
        au = st.get("audit")
        if au and "error" not in au:
            L += ["", f"Audit, DataLoader num_workers = max_num_shards = {au['workers']} on the local SEP copy ({au['parquet_files']} parquet files): "
                  f"workers with frames {au['workers_with_frames']}; per worker " +
                  "; ".join(f"w{w}: {v['frames']} frames, files {v['distinct_files']}, max cache {v['max_decoder_cache_size']}" for w, v in au["per_worker"].items()) +
                  f". Predicted active shards per worker {au['predicted_active_shards_per_worker']} (decoders {au['predicted_decoders_per_worker']})."]
        elif au:
            L += ["", f"Audit error: {au['error']}"]
        L.append("")
    if mp:
        L += ["## Map-style LeRobotDataset on the local copies (lerobot-train DataLoader settings)", "",
              f"cemu = max(1, round(100 x {mp['config']['ncopy']} / {max(mp['config']['real_video_files_per_camera'].values())})) = {mp['config']['cemu']}. "
              f"{mp['config']['dataloader']}. {mp['config']['repeats']} repeats, interleaved.", "",
              "| arm | workers | cache | samples/s | CPU ms/sample | decoder hit | open ms |", "|---|---|---|---|---|---|---|"]
        for k, a in mp.get("summary", {}).items():
            L.append(f"| {k} | {a['workers']} | {a['cache']} | {fmt_ci(a['samples_per_s'], a['samples_per_s_ci95'])} | "
                     f"{fmt_ci(a['cpu_ms_per_sample'], a['cpu_ms_per_sample_ci95'])} | {f(a['dec_hit_rate'], 3)} | {f(a['open_ms_per_open'], 1)} |")
        errs = [r for r in mp["runs"] if "error" in r]
        if errs:
            L.append(f"\nErrors: {[(r['arm'], r['error'][-300:]) for r in errs]}")
        L.append("")
    # ---- extrapolation
    F = max(real["video_files_per_camera"].values())
    P = real["parquet_data_files_in_repo"] if isinstance(real["parquet_data_files_in_repo"], int) else real["parquet_data_files_from_episodes"]
    L += ["## Extrapolation inputs", "",
          f"- Real video files per camera {F}; real parquet data files {P}; real median frames per file {real['frames_per_file_median_all_cameras']:.0f} vs slice {s['frames']}.",
          f"- Map-style, full size: each worker's cache holds {DEFAULT_CACHE} decoders; random access over SEP {F * ncam} files vs STACK {F} files "
          f"-> expected hit rate about {min(1, DEFAULT_CACHE / (F * ncam)):.3f} (SEP) vs {min(1, DEFAULT_CACHE / F):.3f} (STACK). "
          "The cemu arms measure this ratio on the slice.",
          "- Streaming, lerobot-train (max_num_shards = num_workers, every worker iterates num_shards = min(P, W) shards; datasets then splits each "
          "shard's files across the W workers again, so worker w reads only the shards with more than w files). Decoders needed per worker = "
          "active shards x cameras (SEP) or active shards (STACK):", "",
          "| num_workers W | workers that read | active shards: worker 0 / mean | SEP decoders (worker 0) | STACK decoders (worker 0) | SEP overflows 100 | STACK overflows 100 |",
          "|---|---|---|---|---|---|---|"]
    for W in (4, 8, 16, 32, 34, 64, 101, 128):
        act = streaming_active_shards(P, W)
        a0, am = act[0], sum(act) / len(act)
        L.append(f"| {W} | {sum(1 for x in act if x)} | {a0} / {am:.1f} | {a0 * ncam} | {a0} | {a0 * ncam > DEFAULT_CACHE} | {a0 > DEFAULT_CACHE} |")
    L += ["", "Worker 0 always reads min(P, W) shards, so the rule 'SEP overflows when min(P, W) x cameras > 100, STACK when min(P, W) > 100' "
          "holds for the busiest worker. Other workers read fewer shards, and all W workers carry the full load only when P >= W^2 "
          "(every shard has at least W files); with P < W some workers read nothing. The audit above tests this split on the SEP copy.", "",
          "## Limits", "",
          "- One group per dataset, copied 8 times: files are identical copies, so file-to-file variation (content, sizes) is not sampled.",
          "- Streaming arms are one process (one DataLoader worker); the overflow case is emulated with cache 12 at S = 8, not run with 64 workers.",
          "- Map-style arms read local disk (page cache warm after the first pass); they measure decode CPU, not network or cold-disk reads.",
          "- STACK and SEP encodes share one loop, so encode time per layout is not measured separately.",
          "- State / action values are random; only their shapes match the source.",
          f"- cpu-upgrade box (8 CPUs, 32 GB) shared with other stages' background uploads; compare layouts only inside this run.", ""]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
