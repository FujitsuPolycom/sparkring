"""Every KV capacity measurement in performance/profile-capacity.json cites evidence that states it."""
import pytest

from runtime.common.profiles import ROOT, read_json
from scripts.generate_profiles import check_capacity_records


def test_capacity_records_and_checkpoint_measurements_need_evidence_that_states_them(tmp_path):
    records = read_json(ROOT / 'performance/profile-capacity.json')['profiles']
    check_capacity_records(records)
    assert any(record.get('checkpoints') for record in records.values())

    (tmp_path / 'evidence.md').write_text('GPU KV cache size: 2,000 tokens\nGPU KV cache size: 1,000 tokens\n',
                                          encoding='utf-8')
    measured = {'tokens': 1000, 'kv_bytes_per_rank': 2**30, 'conditions': 'one start',
                'source': 'evidence.md', 'witness': 'GPU KV cache size: 1,000 tokens'}
    record = {**measured, 'tokens': 2000, 'witness': 'GPU KV cache size: 2,000 tokens'}
    check_capacity_records({'profile': {**record, 'checkpoints': {'other': measured}}}, tmp_path)

    def changed(entry, change):
        return {key: value for key, value in {**entry, **change}.items() if value is not None}

    for change, message in (
        ({'source': None}, 'nonempty POSIX relative paths'),
        ({'source': ''}, 'nonempty POSIX relative paths'),
        ({'source': 'missing.md'}, 'Missing or escaping repository file'),
        ({'witness': None}, 'witness text'),
        ({'witness': ''}, 'witness text'),
        ({'witness': 'GPU KV cache size: 999 tokens'}, 'capacity evidence changed'),
        ({'conditions': ''}, 'measurement conditions'),
        ({'kv_bytes_per_rank': None}, 'KV bytes per rank'),
    ):
        with pytest.raises(ValueError, match=message):
            check_capacity_records({'profile': {**record, 'checkpoints': {'other': changed(measured, change)}}}, tmp_path)
        if 'kv_bytes_per_rank' in change:
            # A profile's own record may omit its KV bytes per rank.
            check_capacity_records({'profile': changed(record, change)}, tmp_path)
        else:
            with pytest.raises(ValueError, match=message):
                check_capacity_records({'profile': changed(record, change)}, tmp_path)
