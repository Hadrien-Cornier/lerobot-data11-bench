"""Pick the ABC v3 slice for the DATA-11 benchmark (runs LOCALLY, before the bundle is built).

Reads meta/episodes of lerobot/abc_130k_v3_train at a pinned revision, measures the real
frames-per-file distribution, and chooses whole episodes for every file group of the benchmark,
such that each chosen episode is fully contained in a downloaded source MP4 for all 3 cameras
(cameras roll over to new files independently, so this is checked per camera).

Families (group = K whole episodes, the same groups in every layout):
  ckpt  open-cost curve: GROUPS_PER_LEN groups at ~1k, ~5k and ~M frames, M = real median frames/file
  thrash many short groups (~2k frames, one episode each), count divisible by 3 (BIG = 3 groups)

Smoke uses a subset of the same groups (ckpt lengths 1k and 5k, 12 thrash groups), so it
downloads a subset of the same files.

Usage: python select_slice.py <meta_root> <out selection.json>
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi

REPO = "lerobot/abc_130k_v3_train"
REVISION = "68651e4929d9fb00f798937b2d62617cab5c771d"
CAMS = ["top", "left_wrist", "right_wrist"]
GROUPS_PER_LEN = 3
TOL = 0.15  # ~1k single episodes: length within +-15% of 1000
TOL_LONG = 0.06  # multi-episode groups (5k, full): within +-6% of the target
THRASH_RANGE = (1700, 2300)
N_THRASH_FULL, N_THRASH_SMOKE = 48, 12


def col(c, x):
    return f"videos/observation.images.{c}/{x}"


def load(meta_root):
    cols = ["episode_index", "length"] + [col(c, x) for c in CAMS for x in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
    files = sorted(glob.glob(f"{meta_root}/meta/episodes/*/*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files]).sort_values("episode_index").reset_index(drop=True)
    for c in CAMS:
        df[f"f_{c}"] = list(zip(df[col(c, "chunk_index")].astype(int), df[col(c, "file_index")].astype(int)))
    return df


def file_stats(df):
    out, pooled = {}, []
    for c in CAMS:
        fr = df.groupby(f"f_{c}").length.sum()
        pooled.extend(fr.tolist())
        out[c] = {"files": int(len(fr)), "frames_per_file_median": float(fr.median()), "mean": float(fr.mean()),
                  "p10": float(fr.quantile(0.1)), "p90": float(fr.quantile(0.9))}
    p = np.array(pooled)
    out["all_cameras"] = {"files": int(len(p)), "frames_per_file_median": float(np.median(p)), "mean": float(p.mean()),
                          "p10": float(np.quantile(p, 0.1)), "p90": float(np.quantile(p, 0.9)),
                          "episode_length_median": float(df.length.median()), "episodes": int(len(df))}
    return out


def greedy(pool, target, n, used, tol=TOL):
    """n groups of whole episodes, each summing to target +-TOL; consumes from pool (ordered)."""
    groups, cur, s = [], [], 0
    lo, hi = target * (1 - tol), target * (1 + tol)
    for e, L in pool:
        if e in used:
            continue
        if s + L > hi:
            continue
        cur.append(e)
        s += L
        if s >= lo:
            groups.append(cur)
            used.update(cur)
            cur, s = [], 0
            if len(groups) == n:
                return groups
    return None


def assign(sub, M):
    """-> dict family -> list of groups (lists of episode indices), or None if the pool is too small."""
    used = set()
    pool = list(zip(sub.episode_index.astype(int), sub.length.astype(int)))
    res = {}
    # rarest first: ~1k single episodes, then the thrash singles, then the long groups
    one_k = [(e, L) for e, L in pool if 1000 * (1 - TOL) <= L <= 1000 * (1 + TOL)]
    g = greedy(one_k, 1000, GROUPS_PER_LEN, used)
    if g is None:
        return None
    res["ckpt_1k"] = g
    th = [(e, L) for e, L in pool if THRASH_RANGE[0] <= L <= THRASH_RANGE[1] and e not in used]
    if len(th) < N_THRASH_FULL:
        return None
    res["thrash"] = [[e] for e, _ in th[:N_THRASH_FULL]]
    used.update(e for e, _ in th[:N_THRASH_FULL])
    for name, tgt in (("ckpt_5k", 5000), ("ckpt_full", M)):
        g = greedy(pool, tgt, GROUPS_PER_LEN, used, TOL_LONG)
        if g is None:
            return None
        res[name] = g
    return res


def files_for(df_idx, groups):
    eps = [e for fam in groups.values() for grp in fam for e in grp]
    rows = df_idx.loc[eps]
    return {c: sorted(set(rows[f"f_{c}"])) for c in CAMS}


def main():
    meta_root, out = sys.argv[1], Path(sys.argv[2])
    df = load(meta_root)
    stats = file_stats(df)
    M = int(round(stats["all_cameras"]["frames_per_file_median"]))
    print("frames/file", json.dumps(stats, indent=1))
    df_idx = df.set_index("episode_index", drop=False)

    ff = {c: df.groupby(f"f_{c}").length.sum().to_dict() for c in CAMS}
    best = None
    starts = list(range(0, len(df) - 400, 997))[:130]
    for s in starts:
        for span in range(40, 400, 10):
            rng = df.iloc[s : s + span]
            need = {c: set(rng[f"f_{c}"]) for c in CAMS}
            ok = np.all([df[f"f_{c}"].isin(need[c]) for c in CAMS], axis=0)
            sub = df[ok]
            res = assign(sub, M)
            if res is None:
                continue
            fl = files_for(df_idx, res)
            nfiles = sum(len(v) for v in fl.values())
            frames_dl = sum(int(sum(ff[c][f] for f in fl[c])) for c in CAMS)
            if best is None or frames_dl < best[0]:
                best = (frames_dl, s, span, res, fl, nfiles)
            break
    frames_dl, s, span, res, fl, nfiles = best
    print("best start", s, "span", span, "files", nfiles, "frames downloaded (3 cams)", frames_dl)

    smoke = {"ckpt_1k": res["ckpt_1k"], "ckpt_5k": res["ckpt_5k"], "thrash": res["thrash"][:N_THRASH_SMOKE]}
    fl_smoke = files_for(df_idx, smoke)

    api = HfApi()
    paths = sorted({f"videos/observation.images.{c}/chunk-{ch:03d}/file-{fi:03d}.mp4" for c in CAMS for ch, fi in fl[c]})
    sizes = {p.path: p.size for p in api.get_paths_info(REPO, paths, repo_type="dataset", revision=REVISION)}
    paths_smoke = sorted({f"videos/observation.images.{c}/chunk-{ch:03d}/file-{fi:03d}.mp4" for c in CAMS for ch, fi in fl_smoke[c]})

    def ep_row(e):
        r = df_idx.loc[e]
        return {"episode": int(e), "length": int(r.length),
                "src": {c: {"path": f"videos/observation.images.{c}/chunk-{int(r[col(c, 'chunk_index')]):03d}/file-{int(r[col(c, 'file_index')]):03d}.mp4",
                            "from_timestamp": float(r[col(c, "from_timestamp")]), "to_timestamp": float(r[col(c, "to_timestamp")])} for c in CAMS}}

    groups = {fam: [{"name": f"{fam}_g{i:03d}", "episodes": [ep_row(e) for e in grp], "frames": int(sum(df_idx.loc[e].length for e in grp))}
                    for i, grp in enumerate(gs)] for fam, gs in res.items()}
    sel = {
        "repo_id": REPO, "revision": REVISION, "cams": CAMS, "fps": 30, "stack_order_top_to_bottom": CAMS,
        "real_file_stats": stats, "ckpt_full_target_frames": M,
        "families": groups,
        "smoke": {"ckpt_1k": len(smoke["ckpt_1k"]), "ckpt_5k": len(smoke["ckpt_5k"]), "thrash": len(smoke["thrash"])},
        "download": {"full": {p: sizes[p] for p in paths}, "smoke": {p: sizes[p] for p in paths_smoke}},
        "download_bytes": {"full": int(sum(sizes[p] for p in paths)), "smoke": int(sum(sizes[p] for p in paths_smoke))},
        "search": {"start_episode_row": s, "span": span},
    }
    out.write_text(json.dumps(sel, indent=1))
    for fam, gs in groups.items():
        print(fam, [g["frames"] for g in gs][:12], "..." if len(gs) > 12 else "")
    print("download GB full", sel["download_bytes"]["full"] / 1e9, "files", len(paths), "| smoke", sel["download_bytes"]["smoke"] / 1e9, "files", len(paths_smoke))


if __name__ == "__main__":
    main()
