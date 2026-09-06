"""Exercise startup memory thresholds, mutation guards, and cluster ordering."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import managed_cluster
import managed_memory as memory


def value(blocks=200, gib=96, page=4096):
    order = ((32 << 20) // page).bit_length() - 1
    counts = [0] * (order + 2)
    counts[order] = blocks
    return memory.evaluate(f'MemAvailable: {gib * (1 << 20)} kB',
                           'Node 0, zone Normal ' + ' '.join(map(str, counts)), page)


@pytest.mark.parametrize('page', [4096, 65536])
def test_thresholds(page):
    assert value(page=page)['passed']
    assert not value(blocks=199, page=page)['passed']
    assert not value(gib=95, page=page)['passed']


def test_larger_blocks_are_counted_proportionally():
    counts = [0] * 15
    counts[14] = 100
    actual = memory.evaluate('MemAvailable: 125829120 kB',
                             'Node 0, zone Normal ' + ' '.join(map(str, counts)), 4096)
    assert actual['equivalent_32mib_blocks'] == 200


@pytest.mark.parametrize('page', [0, -1, 12288])
def test_invalid_page_size_fails_closed(page):
    with pytest.raises(ValueError):
        memory.evaluate('MemAvailable: 125829120 kB', 'Node 0, zone Normal 0', page)


def test_missing_buddy_data_fails_closed():
    with pytest.raises(ValueError):
        memory.evaluate('MemAvailable: 125829120 kB', '', 4096)


@pytest.mark.parametrize('after_passes', [True, False])
def test_preparation_rechecks_and_never_reboots(monkeypatch, after_passes):
    snapshots = iter([value(blocks=20), value(blocks=200 if after_passes else 30)])
    actions = []
    monkeypatch.setattr(memory, 'start_lock', lambda config: nullcontext())
    monkeypatch.setattr(memory, 'snapshot', lambda: next(snapshots))
    monkeypatch.setattr(memory, 'assert_idle', lambda config: actions.append('idle'))
    monkeypatch.setattr(memory, 'reclaim', lambda: actions.append('reclaim'))
    result = memory.prepare({})
    assert actions == ['idle', 'reclaim']
    assert result['passed'] is after_passes
    assert result['status'] == ('recovered' if after_passes else 'reboot-required')


def test_healthy_memory_does_not_reclaim(monkeypatch):
    monkeypatch.setattr(memory, 'start_lock', lambda config: nullcontext())
    monkeypatch.setattr(memory, 'snapshot', value)
    monkeypatch.setattr(memory, 'reclaim', lambda: pytest.fail('Unexpected mutation'))
    assert not memory.prepare({})['reclaimed']


@pytest.mark.parametrize('obstacle', ['intent', 'container', 'gpu', 'port', 'identity'])
def test_idle_guard_rejects_active_or_unknown_state(monkeypatch, tmp_path, obstacle):
    config = {'state_dir': str(tmp_path), 'container_id': 'a'*64, 'container_image': 'sha256:'+'b'*64}
    if obstacle == 'intent':
        (tmp_path/'model-intent.json').write_text('{"active":true}')
    def run(argv, **kwargs):
        if argv[:2] == ['docker', 'ps']:
            return 'container' if obstacle == 'container' else ''
        if argv[0] == 'nvidia-smi':
            return '1234' if obstacle == 'gpu' else ''
        if argv[:2] == ['docker', 'inspect']:
            return json.dumps([{'Image': 'wrong' if obstacle == 'identity' else config['container_image'],
                'State': {'Running': False}, 'Config': {'Env': ['PORT=8015','SPARKRING_LIVENESS_PORT=8016'],
                'Cmd': ['--master-port','29775']}}])
        if argv[0] == 'ss':
            return 'LISTEN' if obstacle == 'port' else ''
        pytest.fail(str(argv))
    monkeypatch.setattr(memory, 'run', run)
    with pytest.raises(RuntimeError):
        memory.assert_idle(config)


def test_all_memory_barriers_precede_model_start():
    names = [name for name, _ in managed_cluster.phases('start-model', '/code', '/config')]
    assert names == ['model-stop-barrier','memory-idle-barrier','prepare-launch-memory',
                     'memory-ready-barrier','four-rank-ready','start-model-units']
