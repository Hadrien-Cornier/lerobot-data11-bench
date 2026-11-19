"""Audit source-published indexes and time optional private transfers."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.streaming_sidecar import (
    DEFAULT_SIDECAR_CACHE,
    make_sidecar_spec,
    published_sidecar_url,
    streaming_data_root,
)
from lerobot.streaming.sidecar import SidecarSpec, sidecar_cache_path
from lerobot.utils.constants import HF_LEROBOT_HOME

SOURCES = {
    "bucket": ("pepijn223/MolmoAct2-BimanualYAM-Dataset-bucket", None, "bucket"),
    "abc": ("lerobot/abc_130k_v3_train", "68651e4929d9fb00f798937b2d62617cab5c771d", "dataset"),
    "droid": ("lerobot/droid_1.0.1", "71da9dcba338f53889907ac5762a2ebc6c2e55ff", "dataset"),
}


def audit_published(fs, url, spec):
    start = time.monotonic()
    result = {"url": url}
    try:
        info = fs.info(url)
        result.update(size_bytes=info["size"], xet_hash=info.get("xet_hash"))
        with fs.open(url, "rb", cache_type="none") as stream, ZipFile(stream) as archive:
            if "manifest_json.npy" not in archive.namelist():
                result.update(status="incompatible", reason="manifest_json.npy missing")
                return result
            with archive.open("manifest_json.npy") as member:
                manifest = json.loads(bytes(np.lib.format.read_array(member, allow_pickle=False)))
        identity = manifest.get("sidecar", {})
        result.update(
            manifest_version=manifest.get("version"),
            schema_version=identity.get("schema_version"),
            source_repo=identity.get("repo_id"),
            source_revision=identity.get("revision"),
            source_root=identity.get("data_root"),
            source_files=len(identity.get("source_files", [])),
            source_fingerprints=len(identity.get("source_fingerprints", {})),
        )
        matches = manifest.get("version") == 3 and spec.matches(SidecarSpec.from_dict(identity))
        result["status"] = "matching_header" if matches else "incompatible"
    except FileNotFoundError:
        result["status"] = "absent"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result.update(status="unverified", error_type=type(exc).__name__)
    finally:
        result["audit_seconds"] = time.monotonic() - start
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=SOURCES, required=True)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--private-repo")
    parser.add_argument("--private-path")
    parser.add_argument("--private-revision")
    parser.add_argument("--private-destination", type=Path)
    parser.add_argument("--private-sha256")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.private_repo:
        if args.audit:
            parser.error("Audit and private transfer require independent caches.")
        if not all((args.private_path, args.private_revision, args.private_destination, args.private_sha256)):
            parser.error("Private transfer requires a path, immutable revision, cache destination and checksum.")
        relative = args.private_destination
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "streaming-sidecars":
            parser.error("The private destination must be relative to the isolated LeRobot sidecar cache.")
        start = time.monotonic()
        if HfApi().dataset_info(args.private_repo, revision=args.private_revision).private is not True:
            raise ValueError("Private transfer source must be private.")
        destination = HF_LEROBOT_HOME / relative
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        transfer_start = time.monotonic()
        archive = hf_hub_download(
            args.private_repo,
            args.private_path,
            repo_type="dataset",
            revision=args.private_revision,
            cache_dir=destination.parent / ".transfer",
            force_download=True,
        )
        os.link(Path(archive).resolve(), destination)
        transfer_seconds = time.monotonic() - transfer_start
        with destination.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksum != args.private_sha256:
            destination.unlink()
            raise ValueError("Transferred archive checksum differs from its publisher checksum")
        result = {
            "dataset": args.dataset,
            "private_transfer": {
                "network_transfer_and_link_seconds": transfer_seconds,
                "consumer_prep_seconds": time.monotonic() - start,
                "size_bytes": destination.stat().st_size,
                "sha256": checksum,
                "repo": args.private_repo,
                "path": args.private_path,
                "revision": args.private_revision,
                "relative_cache_path": str(relative),
            },
        }
    else:
        repo_id, revision, repo_type = SOURCES[args.dataset]
        meta = LeRobotDatasetMetadata(repo_id, revision=revision, repo_type=repo_type)
        root = streaming_data_root(meta, requested_root=None, configured_data_root=None)
        spec = make_sidecar_spec(meta, root)
        destination = sidecar_cache_path(DEFAULT_SIDECAR_CACHE, spec)
        expected_url = published_sidecar_url(spec)
        result = {
            "dataset": args.dataset,
            "data_root": spec.data_root,
            "metadata_revision": spec.revision,
            "schema_version": spec.schema_version,
            "source_files": len(spec.source_files),
            "source_fingerprints": len(spec.source_fingerprints),
            "expected_url": expected_url,
            "relative_cache_path": str(destination.relative_to(HF_LEROBOT_HOME)),
        }
        if args.audit:
            fs = HfFileSystem()
            urls = [expected_url]
            if repo_type == "bucket":
                urls.extend(
                    f"{root}/{item.path}"
                    for item in HfApi().list_bucket_tree(repo_id, prefix="meta/mp4-sidecars")
                    if item.type == "file" and item.path.endswith(".npz") and f"{root}/{item.path}" != expected_url
                )
            result["publisher_files"] = [audit_published(fs, url, spec) for url in urls]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
