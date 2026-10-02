import json, sys
from pathlib import Path
L = Path(sys.argv[1])
ld = lambda p: json.loads(p.read_text())
out = ["# DATA-11 round 2: laptop run (outside HF) + Option E check", "",
       "Run 2026-09-25 16:00-16:03 UTC from Hadrien's Mac (home internet, laptop possibly busy). One reader, warm regime, family ckpt_full at "
       "`hadriencornier/lerobot-data11-bench@a3d31680db`. HTTP bytes counted by the huggingface_hub/httpx spy: **369 MB total, under the 500 MB hard cap**. "
       "Setup and test traffic before this run (not in the 369 MB): about 0.5 GB (tiny test files and a tiny streaming test).", "",
       "## RTT from the laptop", "",
       "- ICMP ping huggingface.co: avg 49 ms (min 45); cas-bridge.xethub.hf.co: avg 47 ms",
       "- HTTP: resolve request (huggingface.co, 1 byte, no redirect) median 74 ms; 1-byte Range GET on the signed CDN URL (us.aws.cdn.hf.co, warm connection) median 169 ms, min 147 ms",
       "- 5 MiB range GET: 0.46-0.66 s (8-11 MB/s)", "",
       "## Random access, warm, one reader (sps = samples/s, 3 cameras per sample, cameras sequential)", "",
       "| layout | reader | sps per repeat | fetch/smp | resolve/smp | MB/smp |", "|---|---|---|---|---|---|"]
rows = {}
for sub in ("access_a", "access_b"):
    for r in ld(L / sub / "access" / "access.json")["runs"]:
        if "error" not in r:
            rows.setdefault((r["layout"], r["reader"]), []).append(r)
for (lay, rd), rs in rows.items():
    sps = ", ".join(f"{r['samples_per_s']:.2f}" for r in rs)
    m = lambda k: sum(r[k] for r in rs) / len(rs)
    out.append(f"| {lay} | {rd} | {sps} | {m('fetches_per_sample'):.2f} | {m('resolve_requests_per_sample'):.2f} | {m('http_bytes_per_sample') / 1e6:.3f} |")
out += ["", "exact on the laptop runs without the precomputed moov index (ACC_EXACT_HINT=0), so its open costs 2 requests; opens are untimed in the warm regime. "
        "fs5M: 6 samples, 1 repeat (budget). Local bit-exact checks were off here (no local copies of the 1.3 GB files); the HF job checks them.", "",
        "## Streaming order, real StreamingLeRobotDataset (#4702 branch), S = 1 shard, warm", ""]
for r in ld(L / "stream" / "stream.json")["runs"]:
    out.append(f"- {r['arm']}: {r['frames_per_s']:.0f} frames/s over {r['frames']} frames after 50 warm-up frames, mp4 fetches in the window "
               f"{r['totals']['mp4_fetches']}, CPU {r['cpu_s_per_frame'] * 1e3:.1f} ms/frame")
out += ["", "With one shard every camera file is read forward in order. One 5 MiB block holds about 1,700 SEP frames or 600 STACK frames, so 400 frames "
        "needed no fetch. The network cost is the file opens (5.3 MB each).", "",
        "## Option E: can TorchCodec 0.11 decode a frame without reading the moov?", "",
        "Local test on the ckpt_1k SEP `top` file (moov 6.2 KB at offset 32), torchcodec 0.11.1, FFmpeg 8. Script and raw output in optionE/.", "",
        "| test | result |", "|---|---|"]
t = (L.parent / "optionE" / "optionE_output.txt").read_text()
j = json.loads(t[t.find("{"):])
desc = {"A_default": "default open", "B_custom_frame_mappings": "custom_frame_mappings (ffprobe JSON)", "C_moov_zeroed": "moov bytes zeroed",
        "C_moov_zeroed_custom_mappings": "moov zeroed + custom_frame_mappings", "D_ftyp_mdat_only": "ftyp + mdat of the needed packets, no moov",
        "E_synthetic_mini_mp4": "synthetic mini-MP4: packets cut with an external index + 17-byte av1C, re-muxed in memory",
        "F_raw_obu_stream": "raw AV1 OBU stream, no container"}
for k, v in j.items():
    out.append(f"| {desc[k]} | {json.dumps(v)[:200]} |")
out += ["", "Answer: **No** for TorchCodec on its own. It reads the whole moov at open, also with custom_frame_mappings, and it fails when the moov is "
        "missing or zeroed. **Yes with a workaround**: keep the sample index (offset, size, keyframe flag, about 8 bytes per frame) and the codec config "
        "outside the MP4, cut the packet bytes, wrap them in a tiny in-memory MP4 (5.5 KB here); TorchCodec decodes it bit-exact. The re-mux cost per sample "
        "was not timed."]
(L.parent / "summary.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
