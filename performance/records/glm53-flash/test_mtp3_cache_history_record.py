import gzip
import json
import statistics
from pathlib import Path


def test_mtp3_cache_history_numeric_observations():
    path = Path(__file__).with_name('mtp3-cache-history-observations.json.gz')
    record = json.loads(gzip.decompress(path.read_bytes()))
    turns = record['turns']
    assert len(turns) == 551 and all(t['valid'] for t in turns)
    assert record['summary']['errors'] == 0
    soak = [t for t in turns if t['phase'] == 'soak']
    assert len(soak) == 545
    assert sum(t['tokenized_prompt_tokens'] for t in soak) == 70074391
    assert sum(t['cached_tokens_reported'] > 0 for t in soak) == 525
    for phase in ('before', 'after'):
        probes = [t for t in turns if t['phase'] == phase]
        assert len(probes) == 3
        expected = record['summary']['probes'][phase]
        assert statistics.median(t['ttft_seconds'] for t in probes) == expected['median_ttft_seconds']
        assert statistics.median(t['decode_tokens_per_second_estimate'] for t in probes) == expected['median_decode_tokens_per_second_estimate']
