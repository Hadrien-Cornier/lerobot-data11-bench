"""Compare observed batch waits where two streaming runs delivered the same episode mix."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
from statistics import mean, median

from summarize import read, run_events, source_verified


def read_prefix(path, limit, batch_size):
    trace = path.with_suffix('.events.jsonl')
    if not trace.is_file():
        trace = path.with_suffix('.events.jsonl.gz')
    open_trace = gzip.open if trace.suffix == '.gz' else open
    strata = defaultdict(list)
    issues = []
    with open_trace(trace, 'rt') as stream:
        for line in stream:
            event = json.loads(line)
            if event['event'] != 'batch':
                continue
            if event['batch'] > limit:
                break
            if event['usable_unique'] != batch_size or event['contract_errors']:
                issues.append(event['batch'])
                continue
            # Integer counts avoid grouping on rounded proportions.
            key = (event['distinct_episodes'], round(event['largest_episode_share'] * batch_size),
                   event['near_0p2s_pairs'])
            strata[key].append(event['wait_s'])
    return strata, issues


def compare(left, right, limit):
    paths = [left, right]
    rows = [read(p) for p in paths]
    if rows[0]['dataset'] != rows[1]['dataset'] or rows[0]['workload'] != rows[1]['workload']:
        raise ValueError('The two measurements must use the same dataset and workload.')
    if {r['reader'] for r in rows} != {'baseline', 'candidate'}:
        raise ValueError('Specify one result from each reader.')
    for key, default in [('tolerance_s', 0.0001), ('batch_size', None), ('workers', None), ('pool_size', None)]:
        if rows[0]['config'].get(key, default) != rows[1]['config'].get(key, default):
            raise ValueError(f'Reader settings differ for {key}.')
    if (rows[0]['source']['revision'] != rows[1]['source']['revision'] or
            any(not source_verified(p, r, run_events(p)) for p, r in zip(paths, rows))):
        raise ValueError('Verify both source revisions or both bucket manifests before comparison.')
    host_ids = [read(p.parent / 'host.json')['job_id'] for p in paths]
    if host_ids[0] != host_ids[1] or rows[0]['cache'] != rows[1]['cache']:
        raise ValueError('Use runs from the same job with the same initial cache state.')
    batch_size = rows[0]['config']['batch_size']
    a, b = [read_prefix(p, limit, batch_size) for p in paths]
    report = {'cells': [p.stem for p in paths], 'cutoff_batches': limit,
              'valid_unique_batches': [sum(map(len, v[0].values())) for v in (a, b)],
              'invalid_or_repeated_batches': [len(a[1]), len(b[1])]}
    if report['valid_unique_batches'] != [limit, limit] or a[1] or b[1]:
        return {**report, 'status': 'incomplete or invalid prefix'}
    strata = []
    weights = []
    for key in sorted(a[0].keys() & b[0].keys()):
        waits = [a[0][key], b[0][key]]
        counts = [len(s) for s in waits]
        weight = min(counts) / limit
        weights.append(weight)
        strata.append({'episodes': key[0], 'largest_episode_examples': key[1],
                       'same_episode_pairs_within_0p2s': key[2], 'batches': counts,
                       'mean_wait_s': [mean(s) for s in waits], 'median_wait_s': [median(s) for s in waits]})
    overlap = sum(weights)
    standardized = [sum(w * s['mean_wait_s'][i] for w, s in zip(weights, strata)) / overlap
                    for i in range(2)] if overlap else None
    return {**report, 'status': 'complete', 'distribution_overlap': overlap,
            'overlap_weighted_mean_consumer_wait_s': standardized, 'strata': strata}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--limit', type=int, default=1000)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error('The prefix must have a positive batch count.')
    print(json.dumps(compare(args.baseline, args.candidate, args.limit), indent=2))
