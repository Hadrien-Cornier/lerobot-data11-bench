"""Fingerprint the source used for a paired run without reading the training payload."""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, hf_hub_download


SOURCES = {
    "bucket": ("pepijn223/MolmoAct2-BimanualYAM-Dataset-bucket", None),
    "abc": ("lerobot/abc_130k_v3_train", "68651e4929d9fb00f798937b2d62617cab5c771d"),
    "droid": ("lerobot/droid_1.0.1", "71da9dcba338f53889907ac5762a2ebc6c2e55ff"),
}


def snapshot(dataset):
    api = HfApi()
    repo, revision = SOURCES[dataset]
    if revision:
        items = api.list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True)
    else:
        items = api.list_bucket_tree(repo, recursive=True)

    objects = []
    for item in items:
        if not hasattr(item, "size"):
            continue
        digest = getattr(item, "xet_hash", None) or getattr(item, "blob_id", None)
        if not digest:
            raise ValueError(f"Missing object identifier for {item.path}")
        objects.append((item.path, item.size, digest))
    objects.sort()
    manifest = hashlib.sha256()
    for row in objects:
        manifest.update((json.dumps(row, separators=(",", ":")) + "\n").encode())

    with TemporaryDirectory() as directory:
        if revision:
            info_path = hf_hub_download(repo, "meta/info.json", repo_type="dataset", revision=revision)
        else:
            info_path = Path(directory) / "info.json"
            api.download_bucket_files(repo, [("meta/info.json", info_path)], raise_on_missing_files=True)
        info = json.loads(Path(info_path).read_text())

    media = [r for r in objects if r[0].startswith("videos/") and r[0].endswith(".mp4")]
    tables = [r for r in objects if r[0].startswith("data/") and r[0].endswith(".parquet")]
    sidecars = [r for r in objects if r[0].startswith("meta/mp4-sidecars/")]
    return {
        "recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": repo, "revision": revision, "manifest_sha256": manifest.hexdigest(),
        "files": len(objects), "video_files": len(media), "video_bytes": sum(r[1] for r in media),
        "data_parquets": len(tables), "data_parquet_bytes": sum(r[1] for r in tables),
        "sidecars": sidecars, "total_frames": info["total_frames"],
        "total_episodes": info["total_episodes"], "fps": info["fps"],
        "cameras": [key for key, spec in info["features"].items() if spec["dtype"] == "video"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SOURCES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expect", help="Fail if the source differs from this manifest fingerprint.")
    args = parser.parse_args()
    result = snapshot(args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(args.dataset, result["manifest_sha256"], result["video_files"], flush=True)
    if args.expect and result["manifest_sha256"] != args.expect:
        raise SystemExit("Source fingerprint changed. Discard this pair.")


if __name__ == "__main__":
    main()
