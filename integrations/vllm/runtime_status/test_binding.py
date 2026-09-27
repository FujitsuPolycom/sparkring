"""Installer/runtime joins reject mismatched, stale or unbound observations."""
import copy
import json
from pathlib import Path
import sys
import uuid

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from sparkring_runtime_status import binding

NODE = str(uuid.UUID(int=1))
BOOT = str(uuid.UUID(int=2))
DEPLOYMENT = 'a' * 64
CONTAINER = 'b' * 64
IMAGE = 'sha256:' + 'c' * 64


def fixtures():
    fields = dict(deployment_id=DEPLOYMENT, node_id=NODE, container_id=CONTAINER,
                  image_id=IMAGE, rank=0)
    facts = {key: {'state': 'known', 'value': value, 'source': 'fixture'} for key, value in fields.items()}
    facts.update(boot_id={'state': 'known', 'value': BOOT, 'source': 'linux_proc_boot_id'},
                 process_id={'state': 'known', 'value': 123, 'source': 'worker_process'})
    expected = dict(deployment_id=DEPLOYMENT, node_id=NODE, image_id=IMAGE, rank=0)
    host = dict(schema='sparkring-node-status/v1', source='host-agent', node_id=NODE,
                boot_id=BOOT, observed_at=1000, state='network-configured', identity_errors={})
    model = dict(schema='sparkring-model-observation/v1', source='installer-docker-inspect',
                 **fields, expected_node_id=NODE, expected_image_id=IMAGE, boot_id=BOOT,
                 node_identity_matches=True, present=True, running=True, observed_at=1000,
                 container_started_at='1970-01-01T00:15:00Z', identity_errors={})
    worker = dict(schema='sparkring-worker-status/v1', identity=facts,
                  collected_at_unix_ns=1000 * 10**9)
    runtime = {'workers': {'state': 'complete', 'stale': False, 'ranks': [worker]}}
    return expected, host, model, runtime


def test_matching_identity_is_not_a_model_or_fabric_qualification():
    result = binding.join(*fixtures(), now=1001)
    assert result['state'] == 'matched'
    assert result['scope'] == 'identity_and_freshness_only'
    assert result['model_ready'] is None and result['hardware_qualified'] is False


@pytest.mark.parametrize('field,value', [
    ('node_id', str(uuid.UUID(int=8))), ('boot_id', str(uuid.UUID(int=9))),
    ('container_id', 'd' * 64), ('image_id', 'sha256:' + 'd' * 64),
    ('deployment_id', 'd' * 64),
])
def test_replaced_node_boot_container_image_or_deployment_does_not_match(field, value):
    expected, host, model, runtime = fixtures()
    runtime['workers']['ranks'][0]['identity'][field]['value'] = value
    assert binding.join(expected, host, model, runtime, now=1001)['state'] == 'mismatch'


@pytest.mark.parametrize('source', ['host', 'model', 'worker'])
def test_each_source_has_its_own_freshness_gate(source):
    expected, host, model, runtime = fixtures()
    if source == 'host':
        host['observed_at'] = 800
    elif source == 'model':
        model['observed_at'] = 800
    else:
        runtime['workers']['ranks'][0]['collected_at_unix_ns'] = 980 * 10**9
    assert binding.join(expected, host, model, runtime, now=1001)['state'] == 'stale'


def test_container_restart_invalidates_an_earlier_worker_report():
    expected, host, model, runtime = fixtures()
    model['container_started_at'] = '1970-01-01T00:16:40.5Z'
    assert binding.join(expected, host, model, runtime, now=1001)['state'] == 'stale'


def test_duplicate_or_missing_worker_identity_never_uses_rank_alone():
    expected, host, model, runtime = fixtures()
    runtime['workers']['ranks'].append(copy.deepcopy(runtime['workers']['ranks'][0]))
    assert binding.join(expected, host, model, runtime, now=1001)['state'] == 'unbound'
    runtime['workers']['ranks'].pop()
    del runtime['workers']['ranks'][0]['identity']['container_id']
    assert binding.join(expected, host, model, runtime, now=1001)['state'] == 'unbound'


@pytest.mark.parametrize('time', [float('nan'), float('inf'), True, 1010])
def test_invalid_or_future_time_is_not_fresh(time):
    expected, host, model, runtime = fixtures()
    host['observed_at'] = time
    assert binding.join(expected, host, model, runtime, now=1001)['state'] != 'matched'


def test_binding_is_loaded_at_registration_not_each_snapshot(monkeypatch):
    payload = dict(schema='sparkring-runtime-binding/v1', deployment_id=DEPLOYMENT,
                   node_id=NODE, container_id=CONTAINER, image_id=IMAGE, rank=0)
    calls = []
    def read(path, limit):
        calls.append(str(path))
        return BOOT if str(path).endswith('boot_id') else json.dumps(payload)
    monkeypatch.setattr(binding, '_configured', False)
    binding.configure({'SPARKRING_RUNTIME_BINDING': '/run/sparkring/runtime-binding.json'}, read=read)
    for _ in range(3):
        result = binding.worker_identity(0)
        assert result['container_id']['value'] == CONTAINER
        assert result['boot_id']['value'] == BOOT
    assert len(calls) == 2
    assert binding.worker_identity(1)['container_id']['state'] == 'unknown'
    monkeypatch.setattr(binding, '_configured', False)
    binding.configure({}, read=read)
    assert binding.worker_identity(0)['container_id']['state'] == 'unknown'


def test_invalid_metadata_does_not_fail_model_serving_or_expose_contents(monkeypatch):
    monkeypatch.setattr(binding, '_configured', False)
    binding.configure({'SPARKRING_RUNTIME_BINDING': '/run/sparkring/runtime-binding.json'},
                      read=lambda *_: 'private invalid contents')
    result = binding.worker_identity(0)
    assert result['deployment_id']['state'] == 'unknown'
    assert 'private invalid contents' not in json.dumps(result)


def test_unbounded_timestamps_are_rejected_without_overflow():
    expected, host, model, runtime = fixtures()
    host['observed_at'] = 10**400
    assert binding.join(expected, host, model, runtime, now=1001)['state'] == 'unbound'
    host['observed_at'] = 1000
    runtime['workers']['ranks'][0]['collected_at_unix_ns'] = 10**400
    assert binding.join(expected, host, model, runtime, now=1001)['state'] == 'unbound'
