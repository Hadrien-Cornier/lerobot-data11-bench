"""Run the normal lerobot-train command with a trace of successful update batches."""

import json
import os
from pathlib import Path


if __name__ == "__main__":
    from lerobot.scripts import lerobot_train

    preprocess = lerobot_train._preprocess_dataset_batch
    update = lerobot_train.update_policy
    pending = []
    trace = Path(os.environ["STREAM5388_TRAIN_TRACE"])
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        trace = trace.with_suffix(f".rank{os.environ['RANK']}.jsonl")

    def capture_batch(batch, *args, **kwargs):
        pending[:] = batch["index"].detach().cpu().tolist()
        return preprocess(batch, *args, **kwargs)

    def log_update(*args, **kwargs):
        result = update(*args, **kwargs)
        with trace.open("a") as file:
            print(json.dumps(pending), file=file)
        return result

    lerobot_train._preprocess_dataset_batch = capture_batch
    lerobot_train.update_policy = log_update
    lerobot_train.main()
