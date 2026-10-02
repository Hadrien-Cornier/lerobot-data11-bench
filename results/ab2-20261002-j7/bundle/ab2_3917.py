"""A/B test two more changes on PR #3917 at head 6d945985 (which already has the per-sample fixes).

  PROBE  patches-3917/probe-exact-reads.diff: small first header read, exact follow-up reads, tail read for a moov
         after the payload. Measured on the real sidecar build path (EpisodeVideoManifest._build_file_records,
         native-http, 16 workers = max_num_shards default) over random lerobot/abc_130k_v3_train source MP4s.
  TFX    patches-3917/transforms-on-decode-threads.diff: RGB image transforms on the decode threads, not the main thread.
         Measured with prof_3917.py run (D2T, D6T, and D6 without transforms as the A = A check).

Subcommands (all but orchestrate run inside one variant venv):
  probe-run    index a list of source MP4s once; per-file reads, bytes, seconds, faststart and an index digest
  tf-digest    stream 2 episodes once with a fixed or a random transform; sha1 of every camera tensor by index
  orchestrate  (main venv) every stage below; writes ab2/ab2.json and ab2/ab2.md

Env: VENV_BASE, VENV_PROBE, VENV_TFX, AB2_JOB (0..n-1, picks disjoint files), AB2_PROBE_REPS (3), AB2_PROBE_N (300),
     AB2_EQ_N (100), AB2_TF_ARMS ("D2T D6T D6"), AB2_TF_REPS (2), PF_CAP_S (150), PF_WARM_S (45), AB2_PIN ("0-7")
"""

import argparse
import hashlib
import json
import os
import random
import statistics as st
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prof_3917 import ABC, ABC_REV, REPO, REV, analyze_run  # noqa: E402

MIB = 1024 * 1024
# (variant, header_probe_bytes or None for the variant's own default)
PROBE_ARMS = [("BASE", None), ("PROBE", 64 * 1024), ("PROBE", 256 * 1024), ("PROBE", None)]


def arm_name(variant, header):
    return f"{variant}@{'default' if header is None else f'{header // 1024}K'}"


# ====================================================================== probe-run (inside a variant venv)
def _index_digest(idx):
    h = hashlib.sha1()
    for k in ("file_size", "moov_offset", "mdat_offset", "mdat_payload_offset", "mdat_payload_size", "faststart", "codec", "timescale",
              "duration", "track_id", "width", "height"):
        h.update(repr(getattr(idx, k)).encode())
    h.update(idx.ftyp)
    h.update(idx.stsd_body)
    for k in ("sample_pts", "sample_durations", "sample_composition_offsets", "sample_sizes", "sample_offsets", "sync_samples"):
        a = getattr(idx, k)
        h.update(f"{a.dtype}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def cmd_probe_run(a):
    from lerobot.streaming import manifest as mf
    from lerobot.streaming import mp4

    files = json.loads(Path(a.files).read_text())
    per_file, lock = {}, threading.Lock()
    orig = mf.fetch_mp4_index

    def wrapped(path, read_range, **kw):
        reads = []

        def rr(p, off, ln):
            b = read_range(p, off, ln)
            reads.append((off, len(b)))
            return b

        t = time.perf_counter()
        try:
            idx = orig(path, rr, **kw)
        except Exception as e:
            with lock:
                per_file[path] = {"s": time.perf_counter() - t, "reads": len(reads), "bytes": sum(n for _, n in reads),
                                  "error": f"{type(e).__name__}: {e}"}
            raise
        with lock:
            per_file[path] = {"s": time.perf_counter() - t, "reads": len(reads), "bytes": sum(n for _, n in reads),
                              "faststart": bool(idx.faststart), "moov_bytes": int(idx.mdat_offset - idx.moov_offset) if idx.faststart else None,
                              "digest": _index_digest(idx)}
        return idx

    mf.fetch_mp4_index = wrapped
    default_header = getattr(mp4, "DEFAULT_HEADER_PROBE_BYTES", 4 * MIB)  # the base code has no constant: 4 MiB literals
    root = f"hf://datasets/{ABC}@{ABC_REV}"
    res = {"variant": a.variant, "header": a.header, "default_header": default_header,
           "workers": a.workers, "files": len(files)}
    t = time.perf_counter()
    try:
        mf.EpisodeVideoManifest._build_file_records([p for p, _ in files], root, range_backend="native-http", workers=a.workers,
                                                    header_probe_bytes=a.header or default_header, max_probe_bytes=64 * MIB, token=None)
        res["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        res["status"] = "error"
        res["error"] = f"{type(e).__name__}: {e}"
    res["wall_s"] = time.perf_counter() - t
    res["per_file"] = per_file
    Path(a.out).write_text(json.dumps(res))
    print(f"probe-run {a.variant} header={a.header} {len(files)} files {res['status']} {res['wall_s']:.1f}s", flush=True)


# ====================================================================== tf-digest (inside a variant venv)
def _fixed_transform(img):
    import torch

    return (img.to(torch.int16).flip(-1) * 7 % 256).to(img.dtype)


def cmd_tf_digest(a):
    import torch

    from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

    if a.kind == "fixed":
        tf = _fixed_transform
    else:
        from lerobot.transforms.transforms import ImageTransforms, ImageTransformsConfig

        tf = ImageTransforms(ImageTransformsConfig(enable=True))
    torch.manual_seed(0)
    ds = StreamingLeRobotDataset(REPO, revision=REV, episodes=[0, 1], image_transforms=tf, return_uint8=True, repeat=False, seed=7,
                                 decode_threads=6, max_num_shards=4)
    cams = None
    out = {}
    for item in ds:
        cams = cams or [k for k in item if k.startswith("observation.images")]
        out[str(int(item["index"].item()))] = {k: hashlib.sha1(item[k].contiguous().numpy().tobytes()).hexdigest() for k in cams}
    Path(a.out).write_text(json.dumps({"kind": a.kind, "n": len(out), "cameras": cams, "digests": out}))
    print(f"tf-digest {a.kind}: {len(out)} samples", flush=True)


def compare(pa, pb):
    da, db = json.loads(Path(pa).read_text())["digests"], json.loads(Path(pb).read_text())["digests"]
    common = set(da) & set(db)
    same = sum(da[k] == db[k] for k in common)
    return {"a": len(da), "b": len(db), "common": len(common), "identical_samples": same, "all_identical": same == len(common) == len(da) == len(db)}


# ====================================================================== orchestrate (main venv)
def cmd_orchestrate(a):
    out = Path(a.out) / "ab2"
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    venvs = {"BASE": os.environ["VENV_BASE"], "PROBE": os.environ["VENV_PROBE"], "TFX": os.environ["VENV_TFX"]}
    job = int(os.environ.get("AB2_JOB", "0"))
    reps = int(os.environ.get("AB2_PROBE_REPS", "3"))
    n = int(os.environ.get("AB2_PROBE_N", "300"))
    eq_n = int(os.environ.get("AB2_EQ_N", "100"))
    tf_arms = os.environ.get("AB2_TF_ARMS", "D2T D6T D6").split()
    tf_reps = int(os.environ.get("AB2_TF_REPS", "2"))
    cap = float(os.environ.get("PF_CAP_S", "150"))
    warm = float(os.environ.get("PF_WARM_S", "45"))
    pin = os.environ.get("AB2_PIN", "0-7")
    pre = ["taskset", "-c", pin] if pin else []
    stages = os.environ.get("AB2_STAGES", "probe tfdigest tf").split()
    res = {"config": {"job": job, "probe_reps": reps, "probe_n": n, "eq_n": eq_n, "tf_arms": tf_arms, "tf_reps": tf_reps, "cap_s": cap,
                      "warm_s": warm, "pin": pin, "pr3917_sha": os.environ.get("PR3917_SHA")}, "probe": [], "probe_eq": {}, "tf_digest": {},
           "runs": []}

    def save():
        (out / "ab2.json").write_text(json.dumps(res, indent=1, default=str))

    def run(cmd, log, timeout):
        with open(log, "w") as lg:
            try:
                subprocess.run([*pre, *cmd], stdout=lg, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            except subprocess.TimeoutExpired:
                lg.write("TIMEOUT\n")

    # 1. probe: equality set read by every arm, then disjoint random file sets per (arm, repeat), arm order rotated
    if "probe" in stages:
        from huggingface_hub import HfApi

        files = sorted((f.path, f.size) for f in HfApi().list_repo_tree(ABC, repo_type="dataset", revision=ABC_REV, path_in_repo="videos",
                                                                         recursive=True) if getattr(f, "size", None) and f.path.endswith(".mp4"))
        random.Random(1000 + job).shuffle(files)
        res["config"]["catalog_files"] = len(files)
        cursor = 0

        def take(k):
            nonlocal cursor
            chunk = files[cursor : cursor + k]
            cursor += k
            return chunk

        def probe(variant, header, chunk, tag):
            fp = runs / f"{tag}.files.json"
            fp.write_text(json.dumps(chunk))
            o = runs / f"{tag}.json"
            cmd = [venvs[variant] + "/bin/python", str(HERE / "ab2_3917.py"), "probe-run", "--variant", variant, "--files", str(fp), "--out", str(o)]
            if header is not None:
                cmd += ["--header", str(header)]
            run(cmd, runs / f"{tag}.log", 1800)
            try:
                return json.loads(o.read_text())
            except Exception as e:  # noqa: BLE001
                return {"status": "no_result", "error": repr(e), "per_file": {}}

        eq = take(eq_n)
        eq_res = {}
        for variant, header in PROBE_ARMS:
            name = arm_name(variant, header)
            eq_res[name] = probe(variant, header, eq, f"eq_{name}")
            print(f"[{time.strftime('%H:%M:%S')}] eq {name}: {eq_res[name].get('status')} {eq_res[name].get('wall_s', 0):.1f}s", flush=True)
        ref = eq_res[arm_name("BASE", None)]["per_file"]
        for name, r in eq_res.items():
            pf = r["per_file"]
            res["probe_eq"][name] = {"status": r.get("status"), "files": len(pf), "same_digest_as_base": sum(1 for p, v in pf.items()
                                     if v.get("digest") and ref.get(p, {}).get("digest") == v["digest"]),
                                     "errors": sum(1 for v in pf.values() if v.get("error")), "non_faststart": sum(1 for v in pf.values()
                                     if v.get("faststart") is False)}
        save()
        for rep in range(reps):
            k = (rep + job) % len(PROBE_ARMS)
            for variant, header in PROBE_ARMS[k:] + PROBE_ARMS[:k]:
                name = arm_name(variant, header)
                r = probe(variant, header, take(n), f"speed_{name}_r{rep}")
                pf = r.get("per_file", {})
                ok = [v for v in pf.values() if not v.get("error")]
                row = {"arm": name, "rep": rep, "status": r.get("status"), "error": r.get("error"), "files": len(pf), "wall_s": r.get("wall_s"),
                       "files_per_s": len(pf) / r["wall_s"] if r.get("wall_s") else None, "mb": sum(v["bytes"] for v in pf.values()) / 1e6,
                       "reads_mean": st.mean(v["reads"] for v in pf.values()) if pf else None,
                       "file_s_p50": st.median(v["s"] for v in ok) if ok else None, "errors": len(pf) - len(ok),
                       "non_faststart": sum(1 for v in ok if v.get("faststart") is False),
                       "non_faststart_mb": sum(v["bytes"] for v in ok if v.get("faststart") is False) / 1e6}
                res["probe"].append(row)
                print(f"[{time.strftime('%H:%M:%S')}] probe {name} r{rep}: {row['status']} {row['files_per_s'] or 0:.1f} files/s "
                      f"{row['mb']:.0f} MB reads {row['reads_mean'] or 0:.2f} non-faststart {row['non_faststart']}", flush=True)
                save()

    # 2. same output with a fixed transform; reproducibility with the random default transforms
    if "tfdigest" in stages:
        dig = {}
        for tag, variant, kind in [("fixed_BASE", "BASE", "fixed"), ("fixed_TFX", "TFX", "fixed"), ("random_BASE_a", "BASE", "random"),
                                   ("random_BASE_b", "BASE", "random"), ("random_TFX_a", "TFX", "random"), ("random_TFX_b", "TFX", "random")]:
            o = runs / f"digest_{tag}.json"
            run([venvs[variant] + "/bin/python", str(HERE / "ab2_3917.py"), "tf-digest", "--kind", kind, "--out", str(o)],
                runs / f"digest_{tag}.log", 1800)
            dig[tag] = o
        for name, (x, y) in {"fixed_BASE_vs_TFX": ("fixed_BASE", "fixed_TFX"), "random_BASE_twice": ("random_BASE_a", "random_BASE_b"),
                             "random_TFX_twice": ("random_TFX_a", "random_TFX_b")}.items():
            try:
                res["tf_digest"][name] = compare(dig[x], dig[y])
            except Exception as e:  # noqa: BLE001
                res["tf_digest"][name] = {"error": repr(e)}
        print("tf_digest", json.dumps(res["tf_digest"]), flush=True)
        save()

    # 3. transforms speed: BASE vs TFX per arm, interleaved, order flipped each repeat
    if "tf" in stages:
        def go(variant, arm, rep):
            tag = f"{variant}_{arm}_r{rep}"
            o = runs / f"{tag}.json"
            t = time.time()
            run([venvs[variant] + "/bin/python", str(HERE / "prof_3917.py"), "run", "--arm", arm, "--rep", str(rep), "--cap-s", str(cap),
                 "--warm-s", str(warm), "--out", str(o)], runs / f"{tag}.log", cap + 600)
            subprocess.run(["pkill", "-f", "--", f"--out {o}"], check=False)
            subprocess.run(["pkill", "-f", "multiprocessing.spawn"], check=False)
            time.sleep(2)
            try:
                r = analyze_run(o)
            except Exception as e:  # noqa: BLE001
                r = {"status": "no_result", "err": repr(e)}
            r.update(tag=tag, variant=variant, arm=arm, rep=rep, wall_s=time.time() - t)
            res["runs"].append(r)
            ps = r.get("per_sample_ms") or {}
            print(f"[{time.strftime('%H:%M:%S')}] {tag}: {r.get('status')} sps {r.get('sps')} cpu/sample {ps.get('cpu_total')}", flush=True)
            save()

        for rep in range(tf_reps):
            order = tf_arms if rep % 2 == 0 else tf_arms[::-1]
            for arm in order:
                for v in (("BASE", "TFX") if (rep + job + tf_arms.index(arm)) % 2 == 0 else ("TFX", "BASE")):
                    go(v, arm, rep)
    res["summary"] = summarize(res)
    save()
    write_md(out, res)


def summarize(res):
    s = {"probe": {}, "tf": {}}
    for r in res["probe"]:
        if r.get("status") == "ok":
            s["probe"].setdefault(r["arm"], []).append(r)
    s["probe"] = {arm: {"files_per_s_mean": st.mean(x["files_per_s"] for x in rows), "mb_per_file": sum(x["mb"] for x in rows) / sum(x["files"] for x in rows),
                        "reads_mean": st.mean(x["reads_mean"] for x in rows), "file_s_p50_mean": st.mean(x["file_s_p50"] for x in rows),
                        "n": len(rows), "non_faststart": sum(x["non_faststart"] for x in rows)} for arm, rows in s["probe"].items()}
    for r in res["runs"]:
        if r.get("status") != "ok" or r.get("sps") is None:
            continue
        d = s["tf"].setdefault(r["arm"], {}).setdefault(r["variant"], {})
        d[r["rep"]] = r["sps"]
    for arm, vv in s["tf"].items():
        if "BASE" in vv and "TFX" in vv:
            ks = sorted(set(vv["BASE"]) & set(vv["TFX"]))
            vv["paired_ratio"] = [vv["TFX"][k] / vv["BASE"][k] for k in ks]
    return s


def write_md(out, res):
    s = res.get("summary", {})
    L = [f"# #3917 ab2 job {res['config']['job']} ({res['config'].get('pr3917_sha')})", "", "## Probe", "",
         "| arm | files/s | MB per file | reads per file | file s p50 | runs | non-faststart |", "|---|---|---|---|---|---|---|"]
    for arm, o in s.get("probe", {}).items():
        L.append(f"| {arm} | {o['files_per_s_mean']:.1f} | {o['mb_per_file']:.3f} | {o['reads_mean']:.2f} | {o['file_s_p50_mean']:.3f} | {o['n']} | {o['non_faststart']} |")
    L += ["", f"Equality: `{json.dumps(res.get('probe_eq'))}`", "", "## Transforms", "", f"Digests: `{json.dumps(res.get('tf_digest'))}`", "",
          "| arm | BASE sps | TFX sps | paired TFX/BASE |", "|---|---|---|---|"]
    for arm, vv in s.get("tf", {}).items():
        b, t = vv.get("BASE", {}), vv.get("TFX", {})
        L.append(f"| {arm} | {[round(x) for x in b.values()]} | {[round(x) for x in t.values()]} | {[round(x, 2) for x in vv.get('paired_ratio', [])]} |")
    (out / "ab2.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("probe-run")
    p.add_argument("--variant", required=True)
    p.add_argument("--files", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--header", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    d = sp.add_parser("tf-digest")
    d.add_argument("--kind", choices=["fixed", "random"], required=True)
    d.add_argument("--out", required=True)
    o = sp.add_parser("orchestrate")
    o.add_argument("out")
    a = ap.parse_args()
    {"probe-run": cmd_probe_run, "tf-digest": cmd_tf_digest, "orchestrate": cmd_orchestrate}[a.cmd](a)
