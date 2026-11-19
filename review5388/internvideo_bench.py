"""Time the first 1,000 MolmoAct2 batches on the pinned PR and decode variants."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import random
import resource
import sys
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmark import BATCH_SIZE, INPUTS, atomic_json, build_reader, check_identity, provenance, redact
from catalog import SOURCES
from internvideo_audit import BatchAudit, load_reference
from internvideo_variants import make_candidate_variant
from lerobot.datasets import streaming_dataset
from lerobot.utils.collate import lerobot_collate_fn
from telemetry import IORecorder, Trace


def run(args):
    repo, revision = SOURCES["bucket"]
    fps, _, _, camera_suffixes = INPUTS["bucket"]
    cameras = [f"observation.images.{suffix}" for suffix in camera_suffixes]
    audit = BatchAudit(load_reference(args.reference), cameras)
    source = provenance("candidate", streaming_dataset)
    torch.set_num_threads(1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace = Trace(args.output.with_suffix(".events.jsonl"), perf_counter())
    io = IORecorder(trace)
    io.install_http()
    io.install_workload("candidate", repo, revision)
    dataset_class = streaming_dataset.StreamingLeRobotDataset
    if args.mode != "stock":
        dataset_class = make_candidate_variant(dataset_class, args.mode, batch_size=BATCH_SIZE,
                                               decoder_cache_size=96, io=io)
    result = dict(
        mode=args.mode, source=repo, environment=source, host=platform.node(),
        started_utc=datetime.now(timezone.utc).isoformat(), status="started", milestones={},
        settings=dict(batch_size=BATCH_SIZE, cameras=cameras, pool=32, tolerance_s=0.001,
                      workers=0, action_steps=1, decoder_cache_capacity=96 if args.mode != "stock" else 0),
    )
    atomic_json(args.output, result)
    iterator = None
    try:
        settings = argparse.Namespace(reader="candidate", pool_size=32, seed=42, single_frame=True, tolerance_s=0.001)
        start = perf_counter()
        dataset = build_reader(settings, dataset_class, repo, revision, fps, cameras)
        result["constructor_s"] = perf_counter() - start
        result["metadata"] = check_identity(dataset, revision, fps, cameras)
        collate = lerobot_collate_fn if dataset.meta.has_language_columns else None
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=collate)
        iterator = iter(loader)
        audit_s = 0.0
        for number in range(1, 1001):
            batch = next(iterator)
            delivered_s = perf_counter() - start
            if number in (1, 125, 1000):
                result["milestones"][str(number)] = {
                    "delivered_s": delivered_s, "prior_consumer_audit_s": audit_s, "io": io.snapshot(),
                    "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if sys.platform == "darwin" else 1024),
                }
                if args.mode != "stock":
                    result["milestones"][str(number)]["decoder_cache"] = dataset.decoder_cache_stats()
            before_audit = perf_counter()
            audit.check(number, batch)
            audit_s += perf_counter() - before_audit
            del batch
            if number in (1, 125, 1000):
                result["checked_batches"] = number
                atomic_json(args.output, result)
                print(f"{args.mode} verified batch {number} at {delivered_s:.2f}s", flush=True)
        result.update(audit=audit.summary(), total_consumer_audit_s=audit_s, status="complete")
        atomic_json(args.output, result)
    except Exception as exc:
        result.update(status="error", error=redact(repr(exc)))
        atomic_json(args.output, result)
        raise
    finally:
        del iterator
        trace.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("stock", "reuse", "grouped", "batch_singletons"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    run(parser.parse_args())
