#!/usr/bin/env python
"""DATA-11 round 4 (train side): how many samples per second do LeRobot policies consume in training?

Synthetic batches, no dataset download. For each (policy, camera setup, batch size) this measures one
`lerobot-train` step the way `src/lerobot/scripts/lerobot_train.py` does it, with the same building blocks:
TrainPipelineConfig (+ validate() for the policy optimizer/scheduler presets), make_accelerator,
make_policy, make_pre_post_processors (same kwargs/overrides as lerobot-train), make_optimizer_and_scheduler,
accelerator.prepare, then per step:
  prep   = lerobot_train._preprocess_dataset_batch  (uint8 -> float32 /255 on CPU, then the policy
           preprocessor: tokenizer for SmolVLA, device transfer, normalization)
  update = lerobot_train.update_policy             (forward + backward + grad clip + optimizer step
           + scheduler step, under accelerator.autocast / accumulate)
Each phase is bracketed by torch.cuda.synchronize(). samples/s = batch_size / median(update_s).

The fake dataset metadata is a real LeRobotDatasetMetadata.create(...) (info.json only, in a temp dir) with
stats set to plain zeros (mean/min/q01) and ones (std/max/q99), so no download is needed. Batch keys and
shapes follow LeRobotDataset + the policy's delta indices (resolve_delta_timestamps), with *_is_pad flags,
index columns and a `task` string, in pinned memory like the lerobot-train DataLoader on CUDA.

Subcommands:
  run          one measurement in this process; writes one JSON file
  orchestrate  one subprocess per run (clean CUDA context and peak memory), results.jsonl + summary.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

# name: (cameras as (H, W), state dim, action dim)
SETUPS: dict[str, tuple[list[tuple[int, int]], int, int]] = {
    "abc": ([(224, 224)] * 3, 14, 14),
    "droid": ([(180, 320)] * 3, 8, 8),
    "molmo": ([(360, 640)] * 3, 14, 14),
    "libero": ([(256, 256)] * 2, 8, 8),
    "hqf": ([(720, 1280), (720, 1280), (480, 640)], 14, 14),
}
SMOLVLA_REPO = "lerobot/smolvla_base"


def _quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _stats(xs: list[float]) -> dict:
    return {
        "median": statistics.median(xs),
        "p90": _quantile(xs, 0.9),
        "mean": statistics.fmean(xs),
        "min": min(xs),
        "max": max(xs),
    }


def _jsonable(x):
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(x).items()}
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


# --------------------------------------------------------------------------------------------- run
def build_meta(setup: str, root: Path):
    import numpy as np

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    cams, sdim, adim = SETUPS[setup]
    features = {}
    for i, (h, w) in enumerate(cams):
        features[f"observation.images.cam{i}"] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    features["observation.state"] = {"dtype": "float32", "shape": (sdim,), "names": [f"s{j}" for j in range(sdim)]}
    features["action"] = {"dtype": "float32", "shape": (adim,), "names": [f"a{j}" for j in range(adim)]}
    meta = LeRobotDatasetMetadata.create(
        repo_id=f"local/r4train-{setup}", fps=30, features=features, root=root / setup, use_videos=True
    )

    def st(shape):
        z = np.zeros(shape, dtype=np.float32)
        o = np.ones(shape, dtype=np.float32)
        return {"mean": z, "std": o, "min": z, "max": o, "q01": z, "q99": o, "count": np.array([1000])}

    stats = {f"observation.images.cam{i}": st((3, 1, 1)) for i in range(len(cams))}
    stats["observation.state"] = st((sdim,))
    stats["action"] = st((adim,))
    meta.stats = stats
    return meta


def make_cpu_batch(meta, policy_cfg, batch_size: int, pin: bool) -> dict:
    """One collated batch shaped like LeRobotDataset(delta_timestamps=...) + default collate."""
    import torch

    from lerobot.datasets.factory import resolve_delta_timestamps

    g = torch.Generator().manual_seed(0)
    delta = resolve_delta_timestamps(policy_cfg, meta) or {}
    b = batch_size
    batch: dict = {}
    for key, ft in meta.features.items():
        shape = tuple(ft["shape"])
        n = len(delta[key]) if key in delta else None
        if ft["dtype"] in ("video", "image"):
            h, w, c = shape
            chw = (c, h, w)
            # dataset_reader._query_videos squeezes dim 0, so a single delta step gives (C, H, W)
            full = (b, *chw) if n in (None, 1) else (b, n, *chw)
            batch[key] = torch.randint(0, 256, full, dtype=torch.uint8, generator=g)
        elif key in ("observation.state", "action"):
            full = (b, *shape) if n is None else (b, n, *shape)
            batch[key] = torch.randn(full, generator=g, dtype=torch.float32)
        if n is not None:
            batch[f"{key}_is_pad"] = torch.zeros((b, n), dtype=torch.bool)
    batch["timestamp"] = torch.zeros(b, dtype=torch.float32)
    batch["frame_index"] = torch.arange(b, dtype=torch.int64)
    batch["episode_index"] = torch.zeros(b, dtype=torch.int64)
    batch["index"] = torch.arange(b, dtype=torch.int64)
    batch["task_index"] = torch.zeros(b, dtype=torch.int64)
    batch["task"] = ["pick up the cube and put it in the bin"] * b
    if pin:
        batch = {k: (v.pin_memory() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    return batch


def run_one(args) -> dict:
    import torch

    rec: dict = {
        "policy": args.policy_name,
        "setup": args.setup,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "steps": args.steps,
        "device_requested": args.device,
        "smolvla_weights": args.smolvla_weights if args.policy_name == "smolvla" else None,
    }
    cams, sdim, adim = SETUPS[args.setup]
    rec.update(n_cameras=len(cams), camera_hw=cams, state_dim=sdim, action_dim=adim)
    t_setup = time.perf_counter()

    import lerobot
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.distributed import make_accelerator
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.policies.factory import ProcessorConfigKwargs
    from lerobot.processor.rename_processor import rename_stats
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch, update_policy
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
    from lerobot.utils.random_utils import set_seed

    rec["lerobot_file"] = lerobot.__file__
    rec["torch"] = torch.__version__
    rec["torch_cuda"] = torch.version.cuda
    rec["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    rec["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        rec["gpu_name"] = torch.cuda.get_device_name()
        rec["gpu_total_mem_gb"] = torch.cuda.get_device_properties(0).total_memory / 2**30

    tmp = Path(tempfile.mkdtemp(prefix="r4train_"))
    meta = build_meta(args.setup, tmp / "meta")

    # ---- policy config (what `--policy.type=act` / `--policy.path=lerobot/smolvla_base` would give)
    if args.policy_name == "act":
        from lerobot.policies.act.configuration_act import ACTConfig

        policy_cfg = ACTConfig(device=args.device)
    elif args.policy_name == "smolvla":
        policy_cfg = PreTrainedConfig.from_pretrained(SMOLVLA_REPO)
        if args.smolvla_weights == "pretrained":
            policy_cfg.pretrained_path = Path(SMOLVLA_REPO)
        else:
            # Local CPU check only: same architecture, random init, no 900 MB weight download.
            policy_cfg.load_vlm_weights = False
        policy_cfg.device = args.device
    else:
        raise ValueError(args.policy_name)
    # Features come from the (fake) dataset, as for a fine-tune on a new dataset: output_features are
    # always taken from the dataset by make_policy; input_features only when empty, so clear the
    # pretrained ones (smolvla_base ships camera1..3 at 256x256 and a 6-d state).
    policy_cfg.input_features = {}
    policy_cfg.push_to_hub = False  # as `--policy.push_to_hub=false`; otherwise validate() wants a repo_id

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=meta.repo_id),
        policy=policy_cfg,
        output_dir=tmp / "out",
        batch_size=args.batch_size,
    )
    cfg.validate()  # resolves the policy's optimizer / scheduler presets

    accelerator = make_accelerator(cfg)
    set_seed(cfg.seed, accelerator=accelerator)
    torch.backends.cudnn.benchmark = not cfg.cudnn_deterministic
    torch.backends.cuda.matmul.allow_tf32 = True
    device = accelerator.device

    policy = make_policy(cfg=cfg.trainable_config, ds_meta=meta, rename_map=cfg.rename_map)

    # ---- processors: same kwargs / overrides as lerobot_train.train()
    active_cfg = cfg.trainable_config
    processor_pretrained_path = active_cfg.pretrained_path
    if args.policy_name == "smolvla" and args.smolvla_weights == "none":
        processor_pretrained_path = Path(SMOLVLA_REPO)  # still exercise the pretrained processor path
    processor_kwargs = ProcessorConfigKwargs()
    processor_dataset_stats = rename_stats(meta.stats, cfg.rename_map)
    processor_kwargs["dataset_stats"] = processor_dataset_stats
    if processor_pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
                "stats": processor_dataset_stats,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        processor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
                "stats": processor_dataset_stats,
            },
        }
    preprocessor, _post = make_pre_post_processors(
        policy_cfg=active_cfg,
        pretrained_path=processor_pretrained_path,
        pretrained_revision=active_cfg.pretrained_revision,
        **processor_kwargs,
    )
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    policy, optimizer, lr_scheduler = accelerator.prepare(policy, optimizer, lr_scheduler)

    rec["defaults"] = {
        "train_batch_size_default": next(
            f.default for f in dataclasses.fields(TrainPipelineConfig) if f.name == "batch_size"
        ),
        "mixed_precision": cfg.accelerator.mixed_precision,
        "gradient_accumulation_steps": cfg.accelerator.gradient_accumulation.steps,
        "policy_use_amp_field": getattr(active_cfg, "use_amp", None),
        "use_policy_training_preset": cfg.use_policy_training_preset,
        "optimizer": {"class": type(cfg.optimizer).__name__, **_jsonable(cfg.optimizer)},
        "scheduler": None
        if cfg.scheduler is None
        else {"class": type(cfg.scheduler).__name__, **_jsonable(cfg.scheduler)},
        "torch_optimizer": type(optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer).__name__,
        "grad_clip_norm": cfg.optimizer.grad_clip_norm,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "chunk_size": getattr(active_cfg, "chunk_size", None),
        "seed": cfg.seed,
    }
    unwrapped = accelerator.unwrap_model(policy)
    rec["params_total"] = sum(p.numel() for p in unwrapped.parameters())
    rec["params_trainable"] = sum(p.numel() for p in unwrapped.parameters() if p.requires_grad)
    rec["input_features"] = {k: list(v.shape) for k, v in unwrapped.config.input_features.items()}
    rec["output_features"] = {k: list(v.shape) for k, v in unwrapped.config.output_features.items()}

    cpu_batch = make_cpu_batch(meta, active_cfg, args.batch_size, pin=device.type == "cuda")
    rec["batch_shapes"] = {
        k: list(v.shape) if isinstance(v, torch.Tensor) else f"list[{len(v)}]" for k, v in cpu_batch.items()
    }
    rec["batch_uint8_mb"] = sum(
        v.numel() for v in cpu_batch.values() if isinstance(v, torch.Tensor) and v.dtype == torch.uint8
    ) / 2**20
    rec["setup_s"] = time.perf_counter() - t_setup

    metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
    }
    if accelerator.scaler is not None:
        metrics["grad_scale"] = AverageMeter("scale", ":.0f")
    if torch.cuda.is_available():
        metrics["gpu_mem_gb"] = AverageMeter("mem_gb", ":.2f")
    tracker = MetricsTracker(args.batch_size, 1_000_000, 1000, metrics)

    cuda = device.type == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize()

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    prep_s, update_s, total_s, losses = [], [], [], []
    peak_alloc = 0
    policy.train()
    for i in range(args.warmup + args.steps):
        batch = dict(cpu_batch)  # shallow copy: _preprocess_dataset_batch replaces entries
        sync()
        t0 = time.perf_counter()
        batch = _preprocess_dataset_batch(batch, meta.camera_keys, cfg.rename_map, preprocessor)
        sync()
        t1 = time.perf_counter()
        tracker, _ = update_policy(
            tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )
        sync()
        t2 = time.perf_counter()
        if cuda:  # update_policy resets the peak at its start, so this is this step's peak
            peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated())
        del batch
        if i >= args.warmup:
            prep_s.append(t1 - t0)
            update_s.append(t2 - t1)
            total_s.append(t2 - t0)
            losses.append(float(tracker.loss.val))

    n_cam = len(cams)
    upd, tot = _stats(update_s), _stats(total_s)
    rec.update(
        status="ok",
        prep_s=_stats(prep_s),
        update_s=upd,
        total_s=tot,
        samples_per_s=args.batch_size / upd["median"],
        camera_frames_per_s=args.batch_size / upd["median"] * n_cam,
        samples_per_s_incl_prep=args.batch_size / tot["median"],
        camera_frames_per_s_incl_prep=args.batch_size / tot["median"] * n_cam,
        loss_first=losses[0],
        loss_last=losses[-1],
        update_s_raw=update_s,
        prep_s_raw=prep_s,
    )
    if cuda:
        rec["peak_mem_allocated_gb"] = peak_alloc / 2**30
        rec["peak_mem_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
    return rec


def cmd_run(args) -> int:
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        rec = run_one(args)
    except Exception as e:  # noqa: BLE001
        import torch

        oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
        rec = {
            "policy": args.policy_name,
            "setup": args.setup,
            "batch_size": args.batch_size,
            "status": "oom" if oom else "error",
            "error": f"{type(e).__name__}: {e}"[:2000],
            "traceback": traceback.format_exc()[-6000:],
        }
        if torch.cuda.is_available():
            rec["gpu_name"] = torch.cuda.get_device_name()
            rec["peak_mem_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
    rec["wall_s"] = time.time() - t0
    out.write_text(json.dumps(rec, indent=1))
    brief = {k: rec.get(k) for k in ("policy", "setup", "batch_size", "status", "samples_per_s", "error")}
    print("RESULT", json.dumps(brief), flush=True)
    return 0


# ------------------------------------------------------------------------------------- orchestrate
def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def write_summary(out: Path, recs: list[dict], env: dict) -> None:
    lines = ["# DATA-11 r4train: training step throughput with synthetic batches", ""]
    for k, v in env.items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "update = forward + backward + grad clip + optimizer + scheduler (`lerobot_train.update_policy`).",
        "prep = uint8->float on CPU + policy preprocessor (tokenizer, device copy, normalization).",
        "samples/s = batch / median update. Camera frames/s = samples/s x cameras.",
        "",
        "| policy | setup | cams | batch | status | update median ms | update p90 ms | prep median ms "
        "| samples/s | cam frames/s | samples/s incl prep | peak alloc GB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        ok = r.get("status") == "ok"
        lines.append(
            "| {p} | {s} | {c} | {b} | {st} | {um} | {u9} | {pm} | {sps} | {cfs} | {spp} | {pk} |".format(
                p=r.get("policy"),
                s=r.get("setup"),
                c=r.get("n_cameras", len(SETUPS[r["setup"]][0])),
                b=r.get("batch_size"),
                st=r.get("status"),
                um=fmt(r["update_s"]["median"] * 1e3) if ok else "-",
                u9=fmt(r["update_s"]["p90"] * 1e3) if ok else "-",
                pm=fmt(r["prep_s"]["median"] * 1e3) if ok else "-",
                sps=fmt(r.get("samples_per_s")) if ok else "-",
                cfs=fmt(r.get("camera_frames_per_s")) if ok else "-",
                spp=fmt(r.get("samples_per_s_incl_prep")) if ok else "-",
                pk=fmt(r.get("peak_mem_allocated_gb"), 2) if ok else fmt(r.get("peak_mem_reserved_gb"), 2),
            )
        )
    lines += ["", "## Defaults found (per policy)", ""]
    seen = set()
    for r in recs:
        if r.get("status") == "ok" and r["policy"] not in seen:
            seen.add(r["policy"])
            lines.append(f"- {r['policy']}: `{json.dumps(r['defaults'])}`")
            lines.append(f"  params total {r['params_total']:,}, trainable {r['params_trainable']:,}")
    bad = [r for r in recs if r.get("status") != "ok"]
    if bad:
        lines += ["", "## Failed runs", ""]
        for r in bad:
            lines.append(f"- {r['policy']} {r['setup']} bs{r['batch_size']}: {r.get('status')} {r.get('error', '')[:300]}")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def cmd_orchestrate(args) -> int:
    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    policies = [p for p in args.policies.split(",") if p]
    setups = [s for s in args.setups.split(",") if s]
    extra = {}
    for item in [x for x in args.extra_batch_sizes.split(",") if x]:
        pol, bs = item.split(":")
        extra.setdefault(pol, []).append(int(bs))
    plan = [(p, s, args.batch_size, False) for p in policies for s in setups]
    plan += [(p, s, bs, True) for p in policies for bs in extra.get(p, []) for s in setups]
    t_start = time.time()
    recs = []
    env = {
        "lerobot_sha": os.environ.get("LEROBOT_SHA", "?"),
        "python": platform.python_version(),
        "host": platform.node(),
        "warmup": args.warmup,
        "steps": args.steps,
        "device": args.device,
    }
    for i, (pol, setup, bs, is_extra) in enumerate(plan):
        elapsed = time.time() - t_start
        tag = f"{pol}_{setup}_bs{bs}"
        if is_extra and elapsed > args.budget_s:
            rec = {"policy": pol, "setup": setup, "batch_size": bs, "status": "skipped_budget",
                   "error": f"elapsed {elapsed:.0f}s > budget {args.budget_s}s"}
            print("SKIP", tag, rec["error"], flush=True)
        else:
            jpath = out / "runs" / f"{tag}.json"
            cmd = [sys.executable, os.path.abspath(__file__), "run", "--policy-name", pol, "--setup", setup,
                   "--batch-size", str(bs), "--warmup", str(args.warmup), "--steps", str(args.steps),
                   "--device", args.device, "--smolvla-weights", args.smolvla_weights, "--json-out", str(jpath)]
            print(f"== [{i + 1}/{len(plan)}] {tag} (elapsed {elapsed:.0f}s)", flush=True)
            t0 = time.time()
            try:
                with open(out / "runs" / f"{tag}.log", "w") as lf:
                    rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=args.run_timeout_s).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
            if jpath.exists():
                rec = json.loads(jpath.read_text())
            else:
                rec = {"policy": pol, "setup": setup, "batch_size": bs, "status": f"crash rc={rc}",
                       "error": (out / "runs" / f"{tag}.log").read_text()[-1500:]}
            rec["subprocess_wall_s"] = time.time() - t0
            print("   ", rec.get("status"), "samples/s", fmt(rec.get("samples_per_s")),
                  "update ms", fmt(rec["update_s"]["median"] * 1e3) if rec.get("status") == "ok" else "-",
                  "peak GB", fmt(rec.get("peak_mem_allocated_gb"), 2), f"({rec['subprocess_wall_s']:.0f}s)", flush=True)
            if "gpu_name" in rec:
                env["gpu"] = rec["gpu_name"]
            if "torch" in rec:
                env["torch"] = f"{rec['torch']} cuda {rec.get('torch_cuda')} cudnn {rec.get('cudnn')}"
        recs.append(rec)
        with open(out / "results.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps({k: v for k, v in r.items() if not k.endswith("_raw") and k != "traceback"}) + "\n")
        write_summary(out, recs, env)
    print(f"== orchestrate done in {time.time() - t_start:.0f}s -> {out}/summary.md", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--warmup", type=int, default=10)
    common.add_argument("--steps", type=int, default=50)
    common.add_argument("--device", default="cuda")
    common.add_argument("--smolvla-weights", choices=["pretrained", "none"], default="pretrained")
    r = sub.add_parser("run", parents=[common])
    # NOTE: never name an option `--policy.*`: TrainPipelineConfig.validate() reads `--policy.path` from sys.argv.
    r.add_argument("--policy-name", required=True, choices=["act", "smolvla"])
    r.add_argument("--setup", required=True, choices=sorted(SETUPS))
    r.add_argument("--batch-size", type=int, default=8)
    r.add_argument("--json-out", required=True)
    o = sub.add_parser("orchestrate", parents=[common])
    o.add_argument("--out", required=True)
    o.add_argument("--policies", default="act,smolvla")
    o.add_argument("--setups", default=",".join(SETUPS))
    o.add_argument("--batch-size", type=int, default=8, help="lerobot-train default (TrainPipelineConfig.batch_size)")
    o.add_argument("--extra-batch-sizes", default="", help="e.g. smolvla:64 (run after all default runs)")
    o.add_argument("--budget-s", type=float, default=1e9, help="skip extra runs once this much time has passed")
    o.add_argument("--run-timeout-s", type=float, default=900)
    args = ap.parse_args()
    return cmd_run(args) if args.cmd == "run" else cmd_orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
