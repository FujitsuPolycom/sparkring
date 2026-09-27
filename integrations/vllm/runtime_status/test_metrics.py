from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from sparkring_runtime_status.metrics import CounterWindow, read_counters, summarize


def test_acceptance_counts_bonus_separately_and_handles_no_activity_or_reset():
    window = CounterWindow()
    first = window.snapshot({'rounds': 100, 'draft_tokens': 300, 'accepted_tokens': 150}, 10)
    assert first['lifetime']['acceptance_rate'] == 0.5
    assert first['lifetime']['estimated_tokens_per_verification'] == 2.5
    assert first['recent']['reason'] == 'needs_previous_snapshot'
    second = window.snapshot({'rounds': 110, 'draft_tokens': 330, 'accepted_tokens': 174}, 15)
    assert second['recent']['acceptance_rate'] == 0.8
    assert second['recent']['window_seconds'] == 5
    assert window.snapshot({'rounds': 110, 'draft_tokens': 330, 'accepted_tokens': 174}, 20)['recent']['reason'] == 'no_drafts_in_window'
    assert window.snapshot({'rounds': 1, 'draft_tokens': 3, 'accepted_tokens': 2}, 25)['recent']['reason'] == 'counter_reset'


def test_positional_acceptance_uses_each_drafted_denominator():
    value = summarize({'rounds': 2, 'draft_tokens': 6, 'accepted_tokens': 3,
                        'drafted_per_position': {0: 2, 1: 2, 2: 2},
                        'accepted_per_position': {0: 2, 1: 1, 2: 0}})
    assert [row['acceptance_rate'] for row in value['positions']] == [1, 0.5, 0]
    assert summarize({'rounds': 1, 'draft_tokens': 3, 'accepted_tokens': 4})['state'] == 'unknown'


def test_reader_ignores_unrelated_collectors():
    from prometheus_client import CollectorRegistry, Counter
    from sparkring_runtime_status.metrics import COUNTERS
    registry = CollectorRegistry()
    class Trap:
        def describe(self):
            return []
        def collect(self):
            raise AssertionError('unrelated collector executed')
    registry.register(Trap())
    for key, name in list(COUNTERS.items())[:3]:
        counter = Counter(name, name, registry=registry)
        counter.inc({'rounds': 2, 'draft_tokens': 6, 'accepted_tokens': 3}[key])
    assert read_counters(registry=registry)['accepted_tokens'] == 3
