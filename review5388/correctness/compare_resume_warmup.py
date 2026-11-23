"""Compare resumed sample IDs with an uninterrupted HF Datasets stream."""

import itertools
import json
import random

from datasets import IterableDataset


def generate_rows():
    for episode in range(2000):
        for frame in range(4):
            yield {"episode_index": episode, "frame_index": frame}


def mix_pool(block):
    # Use the draft reader's episode-pool shuffle, without decoding any media.
    slots = [
        (episode, frame)
        for episode, frames in enumerate(block["frame_index"])
        for frame in range(len(frames))
    ]
    seed = "|".join(str(episode[0]) for episode in block["episode_index"])
    random.Random(seed).shuffle(slots)
    return {key: [column[e][f] for e, f in slots] for key, column in block.items()}


def build(grouped):
    rows = IterableDataset.from_generator(generate_rows)
    if grouped:
        rows = rows.batch(by_column="episode_index")
    rows = rows.shuffle(seed=42, buffer_size=32)
    if grouped:
        rows = rows.map(mix_pool, batched=True, batch_size=32)
    return rows


def ids(stream):
    return [(row["episode_index"], row["frame_index"]) for row in stream]


def check(grouped, stop):
    dataset = build(grouped)
    stream = iter(dataset)
    prefix = ids(itertools.islice(stream, stop))
    state = dataset.state_dict()
    reference = ids(stream)
    restored = build(grouped)
    restored.load_state_dict(state)
    suffix = ids(restored)
    observed = prefix + suffix
    expected = set(prefix + reference)
    assert len(expected) == len(prefix) + len(reference) == 8000
    return {
        "grouped": grouped,
        "consumed": stop,
        "remaining": len(reference),
        "resumed": len(suffix),
        "position_matches": sum(a == b for a, b in zip(suffix, reference)),
        "duplicates": len(observed) - len(set(observed)),
        "missing": len(expected - set(observed)),
    }


if __name__ == "__main__":
    print(json.dumps([check(grouped, stop) for grouped in (False, True) for stop in (1001, 4096)], indent=2))
