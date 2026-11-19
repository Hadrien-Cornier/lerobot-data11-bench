"""Audit public LeRobot MP4 index publication across declared Hub repository scopes.

Run with Python 3.10+ and requests. The output records visibility per branch,
including inaccessible repositories separately from confirmed missing directories.
"""
import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://huggingface.co/api/datasets"
INDEX_DIR = "meta/mp4-sidecars"
local = threading.local()
rate_lock = threading.Lock()
next_request = 0.0


def get(url, params=None, method="GET"):
    global next_request
    if not hasattr(local, "session"):
        local.session = requests.Session()
    for _ in range(6):
        with rate_lock:
            wait = max(0.0, next_request - time.monotonic())
        if wait:
            time.sleep(wait)
        try:
            response = local.session.request(method, url, params=params, timeout=30, allow_redirects=method != "HEAD")
        except requests.exceptions.RequestException:
            time.sleep(2)
            continue
        quota = re.search(r'\br=(\d+);t=(\d+)', response.headers.get("RateLimit", ""))
        if response.status_code == 429 or (quota and int(quota[1]) < 9):
            pause = int(quota[2]) + 2 if quota else int(response.headers.get("Retry-After", 60)) + 2
            with rate_lock:
                next_request = max(next_request, time.monotonic() + pause)
        if response.status_code != 429 and response.status_code < 500:
            return response
        time.sleep(1)
    raise RuntimeError(f"Hub request repeatedly failed for {url}")


def catalog(params, max_rows=None):
    result = []
    url = BASE
    while url:
        response = get(url, params)
        response.raise_for_status()
        result.extend(response.json())
        if max_rows and len(result) >= max_rows:
            return result[:max_rows]
        url = response.links.get("next", {}).get("url")
        params = None
    return result


def inspect(task):
    repo_id, branch, revision = task
    url = f"{BASE}/{repo_id}/tree/{revision}/{INDEX_DIR}"
    files = []
    while url:
        response = get(url)
        if response.status_code != 200:
            error = response.json().get("error", "")
            if response.status_code == 404 and "Invalid rev id" in error and branch == "lerobot-sidecars":
                return repo_id, branch, {"status": "branch_absent"}
            if response.status_code == 404 and "does not exist on" in error:
                return repo_id, branch, {"status": "directory_absent"}
            return repo_id, branch, {"status": "unobservable", "http": response.status_code, "reason": error}
        files.extend({"path": item["path"], "bytes": item["size"], "oid": item.get("oid")}
                     for item in response.json() if item["type"] == "file" and item["path"].endswith(".npz"))
        url = response.links.get("next", {}).get("url")
    return repo_id, branch, {"status": "checked", "archives": files}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017 (Python 3.10)


def inspect_branch(repo):
    url = f"https://huggingface.co/datasets/{repo}/resolve/lerobot-sidecars/.gitattributes"
    # The resolver reports a missing revision separately from a missing file.
    response = get(url, method="HEAD")
    if response.status_code == 404 and response.headers.get("X-Error-Code") == "RevisionNotFound":
        return repo, {"status": "branch_absent"}
    # Check the exact archive directory whenever a branch may exist or auth masks it.
    _, _, info = inspect((repo, "lerobot-sidecars", "lerobot-sidecars"))
    return repo, info


def all_tagged(args):
    started = now()
    first = json.loads(args.scoped.read_text())
    if not first["complete"]:
        raise ValueError("Scoped audit has not finished")
    groups = {}
    def add_group(repo, group):
        groups.setdefault(repo, set()).add(group)
    for repo, row in first["repos"].items():
        for group in row["groups"]:
            add_group(repo, group)
    for record in catalog({"filter": "lerobot", "limit": 1000, "expand": "likes"}):
        add_group(record["id"], "tagged")
    for record in catalog({"author": "pepijn223", "limit": 1000, "expand": "likes"}):
        add_group(record["id"], "bucket_owner")
    public = {}
    for repo, row in first["repos"].items():
        public[repo] = row["branches"]["lerobot-sidecars"]
    original = len(public)
    tasks = sorted(set(groups) - public.keys())
    def save(complete=False):
        doc = {"started_utc": started, "observed_utc": now(), "complete": complete,
               "scoped_audit": args.scoped.name, "scoped_observed_utc": first["observed_utc"],
               "metadata_scope": {group: sum(group in g for g in groups.values())
                                  for group in sorted({k for g in groups.values() for k in g})},
               "repos": {repo: {"groups": sorted(tags), "lookup": public.get(repo)} for repo, tags in groups.items()}}
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, separators=(",", ":"), sort_keys=True) + "\n")
        tmp.replace(args.output)
    save()
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, (repo, info) in enumerate(pool.map(inspect_branch, tasks), original + 1):
            public[repo] = info
            if i % 100 == 0 or info["status"] != "branch_absent":
                save()
                print(i, len(groups), repo, info["status"], flush=True)
    save(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--scoped", type=Path,
                        help="Use a completed scoped audit and sweep publication branches of all tagged repos.")
    args = parser.parse_args()
    if args.scoped:
        all_tagged(args)
        return
    started = now()
    official = catalog({"author": "lerobot", "limit": 1000})
    top = catalog({"filter": "lerobot", "sort": "likes", "direction": -1, "limit": args.top}, args.top)
    repos = {}
    for group, rows in (("official", official), ("most_liked_tagged", top)):
        for record in rows:
            item = repos.setdefault(record["id"], {"groups": [], "source_sha": record["sha"],
                                    "gated": record.get("gated"), "branches": {}})
            item["groups"].append(group)
    tasks = [(repo, label, revision) for label in ("lerobot-sidecars", "source")
             for repo, item in repos.items()
             for revision in [item["source_sha"] if label == "source" else label]]
    def save(finished=False):
        report = {"started_utc": started, "observed_utc": now(), "complete": finished,
                  "top_tagged_count_requested": args.top, "directory": INDEX_DIR, "repos": repos}
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        tmp.replace(args.output)
    save()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (repo, branch, info) in enumerate(pool.map(inspect, tasks), 1):
            repos[repo]["branches"][branch] = info
            if i % 20 == 0 or info.get("archives"):
                save()
                print(i, len(tasks), repo, branch, len(info.get("archives", [])), flush=True)
    save(True)


if __name__ == "__main__":
    main()
