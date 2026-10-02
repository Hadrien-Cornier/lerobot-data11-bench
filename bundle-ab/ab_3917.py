"""A/B test two per-sample fixes on PR #3917: BASE (head) vs FIX (head + patches/3917-per-sample-fixes.diff).

Subcommands:
  digest       (inside one variant's venv) stream a few whole episodes once, hash every sample by absolute index
  orchestrate  (main venv) digest both variants and compare, then interleaved speed runs via prof_3917.py run,
               then py-spy --gil on D6 for both variants; writes ab/ab.json and ab/ab.md

Env: VENV_BASE, VENV_FIX, AB_ARMS ("D2 D6 R6"), AB_REPEATS (5), PF_CAP_S (150), PF_WARM_S (45), AB_DIGEST_EPS ("0 1 2 3"),
     AB_PYSPY (1)
"""

import argparse
import hashlib
import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prof_3917 import REPO, REV, analyze_run  # noqa: E402


# ====================================================================== digest (inside a variant venv)
def _digest_value(v):
    import torch

    if isinstance(v, torch.Tensor):
        t = v.detach().contiguous().cpu()
        return f"{t.dtype}{tuple(t.shape)}:" + hashlib.sha1(t.numpy().tobytes()).hexdigest()
    return repr(v)


def cmd_digest(a):
    from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

    eps = [int(x) for x in a.episodes.split()]
    t0 = time.perf_counter()
    ds = StreamingLeRobotDataset(REPO, revision=REV, episodes=eps, return_uint8=True, repeat=False, seed=7, decode_threads=2, max_num_shards=4)
    build_s = time.perf_counter() - t0
    out, dup = {}, 0
    t1 = time.perf_counter()
    for item in ds:
        key = str(int(item["index"].item()))
        dup += key in out
        out[key] = {k: _digest_value(item[k]) for k in sorted(item)}
    res = {"episodes": eps, "n": len(out), "dup": dup, "build_s": build_s, "iter_s": time.perf_counter() - t1, "digests": out}
    Path(a.out).write_text(json.dumps(res))
    print(f"digest {a.out}: {len(out)} samples, {dup} duplicates, iter {res['iter_s']:.1f}s", flush=True)


def compare_digests(pa, pb):
    a, b = json.loads(Path(pa).read_text()), json.loads(Path(pb).read_text())
    da, db = a["digests"], b["digests"]
    common = sorted(set(da) & set(db), key=int)
    diff = [k for k in common if da[k] != db[k]]
    fields = sorted({f for k in diff for f in da[k] if da[k].get(f) != db[k].get(f)})
    return {"n_base": len(da), "n_fix": len(db), "dup_base": a["dup"], "dup_fix": b["dup"], "common": len(common),
            "only_base": len(set(da) - set(db)), "only_fix": len(set(db) - set(da)), "differing": len(diff),
            "differing_fields": fields, "differing_examples": diff[:5], "identical": not diff and set(da) == set(db)}


# ====================================================================== orchestrate (main venv)
def cmd_orchestrate(a):
    out = Path(a.out) / "ab"
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    venvs = {"BASE": os.environ["VENV_BASE"], "FIX": os.environ["VENV_FIX"]}
    arms = os.environ.get("AB_ARMS", "D2 D6 R6").split()
    reps = int(os.environ.get("AB_REPEATS", "5"))
    cap = float(os.environ.get("PF_CAP_S", "150"))
    warm = float(os.environ.get("PF_WARM_S", "45"))
    eps = os.environ.get("AB_DIGEST_EPS", "0 1 2 3")
    res = {"config": {"arms": arms, "reps": reps, "cap_s": cap, "warm_s": warm, "repo": REPO, "rev": REV, "digest_episodes": eps,
                      "pr3917_sha": os.environ.get("PR3917_SHA")}, "runs": []}

    def save():
        (out / "ab.json").write_text(json.dumps(res, indent=1, default=str))

    # 1. same output
    dig = {}
    for v, venv in venvs.items():
        p = out / f"digest_{v}.json"
        with open(runs / f"digest_{v}.log", "w") as lg:
            subprocess.run([venv + "/bin/python", str(HERE / "ab_3917.py"), "digest", "--episodes", eps, "--out", str(p)], stdout=lg,
                           stderr=subprocess.STDOUT, timeout=3600, check=False)
        dig[v] = p
    try:
        res["digest"] = compare_digests(dig["BASE"], dig["FIX"])
    except Exception as e:  # noqa: BLE001
        res["digest"] = {"error": repr(e)}
    print("digest", json.dumps(res["digest"])[:600], flush=True)
    save()

    def go(variant, arm, rep, extra_args=(), tag=None):
        tag = tag or f"{variant}_{arm}_r{rep}"
        o = runs / f"{tag}.json"
        cmd = [venvs[variant] + "/bin/python", str(HERE / "prof_3917.py"), "run", "--arm", arm, "--rep", str(rep), "--cap-s", str(cap),
               "--warm-s", str(warm), "--out", str(o), *extra_args]
        t = time.time()
        with open(runs / f"{tag}.log", "w") as lg:
            try:
                subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, timeout=cap + 600)
            except subprocess.TimeoutExpired:
                pass
        subprocess.run(["pkill", "-f", "--", f"--out {o}"], check=False)
        subprocess.run(["pkill", "-f", "multiprocessing.spawn"], check=False)
        time.sleep(2)
        try:
            r = analyze_run(o)
        except Exception as e:  # noqa: BLE001
            r = {"status": "no_result", "err": repr(e)}
        r.update(tag=tag, variant=variant, wall_s=time.time() - t)
        res["runs"].append(r)
        ps = r.get("per_sample_ms") or {}
        print(f"[{time.strftime('%H:%M:%S')}] {tag}: {r.get('status')} sps {r.get('sps')} cpu/sample {ps.get('cpu_total')} {r.get('error', '')}", flush=True)
        save()

    # 2. speed: every (arm, variant) pair per repeat, interleaved; variant order flips each repeat, arm order flips every other
    for rep in range(reps):
        order = arms if rep % 2 == 0 else arms[::-1]
        for arm in order:
            for v in (("BASE", "FIX") if (rep + arms.index(arm)) % 2 == 0 else ("FIX", "BASE")):
                go(v, arm, rep)
    # 3. who holds the lock, after the fix
    if os.environ.get("AB_PYSPY", "1") == "1":
        for v in ("BASE", "FIX"):
            go(v, "D6", 91, extra_args=["--pyspy", "gil", "--pyspy-s", str(min(60, cap - warm - 10))], tag=f"{v}_D6_pyspy_gil")
    res["summary"] = summarize(res)
    save()
    write_md(out, res)


def summarize(res):
    s = {}
    for r in res["runs"]:
        if "pyspy" in r["tag"] or r.get("status") != "ok" or r.get("sps") is None:
            continue
        d = s.setdefault(r["arm"], {}).setdefault(r["variant"], {"sps": [], "cpu": [], "rep": []})
        d["sps"].append(r["sps"])
        d["cpu"].append((r.get("per_sample_ms") or {}).get("cpu_total"))
        d["rep"].append(r["rep"])
    out = {}
    for arm, vv in s.items():
        o = out[arm] = {}
        for v, d in vv.items():
            o[v] = {"sps": d["sps"], "sps_mean": st.mean(d["sps"]), "sps_sd": st.stdev(d["sps"]) if len(d["sps"]) > 1 else 0.0,
                    "cpu_ms": [c for c in d["cpu"] if c is not None]}
            o[v]["cpu_mean"] = st.mean(o[v]["cpu_ms"]) if o[v]["cpu_ms"] else None
        if "BASE" in vv and "FIX" in vv:  # paired by repeat: runs of one repeat sit next to each other in time
            pb = dict(zip(vv["BASE"]["rep"], vv["BASE"]["sps"], strict=True))
            pf = dict(zip(vv["FIX"]["rep"], vv["FIX"]["sps"], strict=True))
            ratios = [pf[k] / pb[k] for k in sorted(set(pb) & set(pf))]
            o["paired_ratio"] = ratios
            o["paired_ratio_mean"] = st.mean(ratios) if ratios else None
            o["paired_ratio_sd"] = st.stdev(ratios) if len(ratios) > 1 else None
            cb, cf = o["BASE"]["cpu_mean"], o["FIX"]["cpu_mean"]
            o["cpu_ratio"] = cf / cb if cb and cf else None
    return out


def write_md(out, res):
    L = [f"# #3917 per-sample fixes A/B ({res['config'].get('pr3917_sha')})", "", f"Digest: `{json.dumps(res.get('digest'))[:400]}`", "",
         "| arm | BASE sps | FIX sps | paired FIX/BASE | BASE cpu ms | FIX cpu ms |", "|---|---|---|---|---|---|"]
    for arm, o in res.get("summary", {}).items():
        b, f = o.get("BASE", {}), o.get("FIX", {})
        L.append(f"| {arm} | {b.get('sps_mean', 0):.0f} ± {b.get('sps_sd', 0):.0f} | {f.get('sps_mean', 0):.0f} ± {f.get('sps_sd', 0):.0f} | "
                 f"{(o.get('paired_ratio_mean') or 0):.3f} ± {(o.get('paired_ratio_sd') or 0):.3f} | {b.get('cpu_mean') or 0:.2f} | {f.get('cpu_mean') or 0:.2f} |")
    (out / "ab.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("digest")
    d.add_argument("--episodes", required=True)
    d.add_argument("--out", required=True)
    o = sp.add_parser("orchestrate")
    o.add_argument("out")
    a = ap.parse_args()
    {"digest": cmd_digest, "orchestrate": cmd_orchestrate}[a.cmd](a)
