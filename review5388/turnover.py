"""Verify pool turnover from native admission and emitted sample IDs."""

from collections import Counter, defaultdict
from functools import wraps
from threading import Lock


class PoolTurnover:
    def __init__(self, reader, trace, *, refill_limit=2):
        self.refill_limit = refill_limit
        self.reader = reader
        self.trace = trace
        self.lock = Lock()
        self.admission_count = 0
        self.initial_width = None
        self.cohorts = []
        self.by_episode = defaultdict(list)
        self.sizes = []
        self.delivered = Counter()
        self.fragments_seen = defaultdict(set)
        self.refill_batches = []

    def add(self, spans):
        with self.lock:
            offset = self.admission_count
            if self.initial_width is None:
                self.initial_width = len(spans)
            for episode, first, last, count in spans:
                if self.reader == "baseline":
                    cohort = self.admission_count // self.initial_width
                else:
                    cohort = len(self.cohorts) if self.admission_count == offset else len(self.cohorts) - 1
                if cohort == len(self.cohorts):
                    self.cohorts.append([])
                    self.sizes.append(0)
                fragment = len(self.cohorts[cohort])
                self.cohorts[cohort].append((episode, first, last, count))
                self.sizes[cohort] += count
                self.by_episode[episode].append((first, last, cohort, fragment))
                self.admission_count += 1
        self.trace.write("pool_admission", offset=offset, spans=spans)

    def reset(self):
        with self.lock:
            self.admission_count = 0
            self.initial_width = None
            self.cohorts.clear()
            self.by_episode.clear()
            self.sizes.clear()
            self.delivered.clear()
            self.fragments_seen.clear()
            self.refill_batches.clear()

    def batch(self, batch_index, new_ids):
        with self.lock:
            for episode, frame in new_ids:
                for first, last, cohort, fragment in self.by_episode.get(episode, ()):
                    if first <= frame <= last:
                        self.delivered[cohort] += 1
                        self.fragments_seen[cohort].add(fragment)
                        break
            self.verify_refills(batch_index)

    def verify_refills(self, batch_index):
        while self.refill_limit is None or len(self.refill_batches) < self.refill_limit:
            prior = len(self.refill_batches)
            current = prior + 1
            if current >= len(self.sizes):
                break
            if self.delivered[prior] != self.sizes[prior]:
                break
            if len(self.fragments_seen[current]) < self.initial_width:
                break
            self.refill_batches.append(batch_index)
            self.trace.write(
                "pool_refill_verified",
                batch=batch_index,
                refill=len(self.refill_batches),
                previous_cohort_samples=self.sizes[prior],
                replacement_episode_fragments=len(self.cohorts[current]),
            )

    def snapshot(self):
        return {
            "refills_verified": len(self.refill_batches),
            "refill_batches": list(self.refill_batches),
            "method": (
                "baseline groups consecutive native admissions by initial resident pool width"
                if self.reader == "baseline"
                else "candidate uses native mapped groups of episode spans"
            ),
            "verification": "Each prior cohort must emit all its unique anchors and every span of its replacement must emit an anchor.",
            "cohorts_observed": len(self.cohorts),
            "cohorts": [
                {
                    "episode_spans": len(group),
                    "planned_anchors": self.sizes[index],
                    "unique_anchors_emitted": self.delivered[index],
                    "episode_spans_emitted": len(self.fragments_seen[index]),
                }
                for index, group in enumerate(self.cohorts[:3])
            ],
        }

    def install(self, dataset_class):
        if self.reader == "candidate":
            original = dataset_class._sample_pool_with_delta

            @wraps(original)
            def sample_pool(dataset, episodes_batch):
                spans = [
                    (int(episode[0]), int(frames[0]), int(frames[-1]), len(frames))
                    for episode, frames in zip(
                        episodes_batch["episode_index"], episodes_batch["frame_index"], strict=True
                    )
                ]
                self.add(spans)
                return original(dataset, episodes_batch)

            dataset_class._sample_pool_with_delta = sample_pool
            return

        from lerobot.streaming.episode_pool import ExactCoveragePool

        original_init = ExactCoveragePool.__init__
        original_next = ExactCoveragePool.__next__

        def record(planner, episodes):
            self.add([(int(ep), 0, int(planner._counts[ep]) - 1, int(planner._counts[ep])) for ep in episodes])

        @wraps(original_init)
        def init_pool(planner, *args, **kwargs):
            original_init(planner, *args, **kwargs)
            record(planner, planner.resident)

        @wraps(original_next)
        def next_sample(planner):
            result = original_next(planner)
            if planner.newly_admitted:
                record(planner, planner.newly_admitted)
            return result

        ExactCoveragePool.__init__ = init_pool
        ExactCoveragePool.__next__ = next_sample
