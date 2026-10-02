"""Stage summary2 (round 2): one summary.md at the run root from prep / stream / access results.

Usage: python s13_summary.py <out_dir>
"""

import json
import sys
from pathlib import Path


def main():
    out = Path(sys.argv[1])
    L = [f"# DATA-11 round 2 ({out.name})", "",
         "Question: does STACK (3 cameras in one video) beat SEP (one MP4 per camera) over hf://, and do cheaper reader fixes close the gap? "
         "Option C = one MP4 with 3 video tracks (stream copy of SEP). Raw numbers: prep/prep.json, stream/stream.json, access/access.json.", ""]
    p = out / "prep" / "prep.json"
    if p.exists():
        prep = json.loads(p.read_text())
        m = prep.get("multi", {})
        if m.get("groups"):
            L += ["## Option C build (multi-track MP4, stream copy)", "",
                  "| group | MB | SEP sum MB | same-timestamp span p50 B | p95 B | gap p50 B | p95 B | packets per run | identity (bit-exact) |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for g in m["groups"]:
                il, idc = g["interleave"], g["identity"]
                L.append(f"| {g['group']} | {g['bytes'] / 1e6:.1f} | {g['sep_bytes_sum'] / 1e6:.1f} | {il['span_bytes_p50']:.0f} | {il['span_bytes_p95']:.0f} | "
                         f"{il['gap_bytes_p50']:.0f} | {il['gap_bytes_p95']:.0f} | {il['mean_packets_per_run']:.2f} | "
                         f"{'PASS' if idc['pass'] else 'FAIL'} {idc['frames_checked'] - idc['mismatch']}/{idc['frames_checked']} |")
            for name, v in m.get("muxer_variants", {}).items():
                il = v.get("interleave")
                if il:
                    L.append(f"- muxer `{' '.join(v['args'])}`: span p50 {il['span_bytes_p50']:.0f} B, p95 {il['span_bytes_p95']:.0f} B, "
                             f"packets per run {il['mean_packets_per_run']:.2f}, decode {v.get('identity', {}).get('pass') if isinstance(v.get('identity'), dict) else v.get('identity')}")
                else:
                    L.append(f"- muxer `{' '.join(v['args'])}`: {v.get('error', '?')[:200]}")
            u = m.get("upload", {})
            L += ["", f"MULTI uploaded at `{u.get('revision', '?')}` (read-back ok: {u.get('readback_ok')}).", ""]
    for sub in ("stream", "access"):
        s = out / sub / "summary.md"
        if s.exists():
            L += ["---", "", s.read_text(), ""]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
