"""Read only known in-memory speculative-decoding counters; no scrape or probe."""
from __future__ import annotations

import math
import sys

from .collector import MISSING, stored

COUNTERS = {
    'rounds': 'vllm:spec_decode_num_drafts',
    'draft_tokens': 'vllm:spec_decode_num_draft_tokens',
    'accepted_tokens': 'vllm:spec_decode_num_accepted_tokens',
    'accepted_per_position': 'vllm:spec_decode_num_accepted_tokens_per_pos',
    'drafted_per_position': 'vllm:spec_decode_num_draft_tokens_per_pos',
}


def read_counters(*, modules=None, registry=None):
    modules = sys.modules if modules is None else modules
    prometheus = modules.get('prometheus_client')
    metrics_module = modules.get('prometheus_client.metrics')
    registry = stored(prometheus, 'REGISTRY') if registry is None else registry
    collectors = stored(registry, '_names_to_collectors')
    counter_type = stored(metrics_module, 'Counter')
    if type(collectors) is not dict or counter_type is MISSING:
        return None
    result = {}
    for key, name in COUNTERS.items():
        collector = collectors.get(name)
        if type(collector) is not counter_type:
            continue
        children = stored(collector, '_metrics')
        if type(children) is dict and len(children) > 256:
            continue
        total, positions = 0, {}
        # Only the installed Prometheus Counter class is admitted. Collecting
        # the whole registry could invoke unrelated custom collectors.
        for family in collector.collect():
            for sample in family.samples[:1024]:
                if sample.name != name + '_total':
                    continue
                value = sample.value
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    continue
                total += value
                position = sample.labels.get('position')
                if type(position) is str and position.isdigit() and 0 <= int(position) < 16:
                    positions[int(position)] = positions.get(int(position), 0) + value
        result[key] = positions if key.endswith('_position') else total
    return result if {'rounds', 'draft_tokens', 'accepted_tokens'} <= result.keys() else None


def summarize(counters):
    rounds, drafted, accepted = (counters[name] for name in ('rounds', 'draft_tokens', 'accepted_tokens'))
    if rounds <= 0 or drafted <= 0:
        return {'state': 'not_observed', 'reason': 'no_drafts_in_window'}
    if accepted > drafted:
        return {'state': 'unknown', 'reason': 'inconsistent_counter_snapshot'}
    positions = []
    for position, count in sorted(counters.get('drafted_per_position', {}).items()):
        accepted_count = counters.get('accepted_per_position', {}).get(position)
        if count > 0 and accepted_count is not None and 0 <= accepted_count <= count:
            positions.append({'position': position + 1, 'drafted': count, 'accepted': accepted_count,
                               'acceptance_rate': accepted_count / count})
    return {'state': 'known', 'rounds': rounds, 'draft_tokens': drafted, 'accepted_tokens': accepted,
            'acceptance_rate': accepted / drafted, 'accepted_drafts_per_round': accepted / rounds,
            'estimated_tokens_per_verification': 1 + accepted / rounds, 'positions': positions}


class CounterWindow:
    def __init__(self):
        self.previous = None

    def snapshot(self, counters, now):
        if counters is None:
            return {'state': 'not_observed', 'source': 'prometheus_speculation_counters',
                    'reason': 'counters_unavailable'}
        lifetime = summarize(counters)
        recent = {'state': 'not_observed', 'reason': 'needs_previous_snapshot'}
        if self.previous is not None:
            before, when = self.previous
            delta = {name: counters[name] - before[name] for name in ('rounds', 'draft_tokens', 'accepted_tokens')}
            if any(value < 0 for value in delta.values()):
                recent = {'state': 'not_observed', 'reason': 'counter_reset'}
            elif now > when:
                recent = {**summarize(delta), 'window_seconds': now - when}
        self.previous = (counters, now)
        return {'state': lifetime['state'], 'source': 'prometheus_speculation_counters',
                'scope': 'api_process_engine_counters', 'lifetime': lifetime, 'recent': recent,
                'note': 'Acceptance depends on prompts and sampling. Estimated tokens per verification includes one bonus token; it is not measured throughput.'}
