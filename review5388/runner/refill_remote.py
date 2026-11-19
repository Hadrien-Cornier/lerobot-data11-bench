"""Run the two native readers on one host with a fresh application cache per reader."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from runner import remote as r


def run_refills():
    pins = json.loads(Path(__file__).with_name("native_molmo_pins.json").read_text())["grouped"]
    archive = r.WORK / "patched-lerobot.tar.gz"
    revision = pins["revision"]
    subprocess.run(["curl", "-fsSL", "--retry", "2",
                    f"https://codeload.github.com/Hadrien-Cornier/lerobot/tar.gz/{revision}",
                    "-o", archive], check=True)
    source = r.WORK / "src/candidate"
    source.rename(source.with_name("stock"))
    source.mkdir()
    subprocess.run(["tar", "-xzf", archive, "-C", source, "--strip-components=1"], check=True)
    for name, want in pins["module_sha256"].items():
        if hashlib.sha256((source / "src/lerobot/datasets" / name).read_bytes()).hexdigest() != want:
            raise RuntimeError(f"Patched reader mismatch in {name}")
    reference = json.loads(os.environ["PREPARED_INDEX"])
    order = ("candidate","baseline") if os.environ.get("REFILL_PATCHED_FIRST") == "1" else ("baseline","candidate")
    r.append({"event":"refills_setup", "patched":pins, "prepared":reference,
              "order":list(order), "max_batches":30000, "seed":42})
    if not r.verify_bucket("before-transfer"):
        raise RuntimeError("The index and source have different manifests")
    outcomes = {}
    for reader in order:
        cache = r.CACHE / f"native-refills-{reader}"
        if cache.exists():
            raise FileExistsError(cache)
        if not r.verify_bucket(f"before-{reader}"):
            raise RuntimeError("Source changed before " + reader)
        if reader == "baseline" and not r.download_prepared("bucket", cache, reference):
            raise RuntimeError("The publisher index did not validate")
        label = f"bucket-native-refills-{reader}"
        allowance = min(8400, int(r.DEADLINE - time.monotonic()) - 100)
        args = ["-m", "runner.refill_arm", "--dataset", "bucket", "--reader", reader,
                "--output", r.RESULTS / f"{label}.json", "--single-frame",
                "--tolerance-s", "0.001", "--seed", "42", "--pool-size", "32",
                "--cache-state", "cold" if reader == "candidate" else "unclassified",
                "--workers", "0", "--max-batches", "30000", "--continue-to-max-batches",
                "--max-seconds", str(allowance - 30)]
        code = r.run(label, args, allowance, reader, cache)
        if not r.verify_bucket(f"after-{reader}"):
            raise RuntimeError("Source changed during " + reader)
        report = r.RESULTS / f"{label}.json"
        result = json.loads(report.read_text()) if report.exists() else {}
        outcomes[reader] = {"code":code, "progress":result.get("progress", {}), "status":result.get("status")}
        r.write_json(r.RESULTS / "outcomes.json", outcomes)
        r.upload(f"closed-{reader}")
        shutil.rmtree(cache, ignore_errors=True)
    if any(row["code"] != 0 or row["progress"].get("batches") != 30000 for row in outcomes.values()):
        raise RuntimeError(f"Incomplete reader run: {outcomes}")


if __name__ == "__main__":
    r.DEADLINE -= max(0, time.time() - int(os.environ["JOB_BOOT_STARTED_AT"]))
    r.main()
