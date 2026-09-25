from types import SimpleNamespace

import pytest

from runtime.common import installer
from runtime.common.test_installer_image import ring_site
from runtime.host import glm_existing_mesh as lifecycle
from scripts import installer_host as host


def lock():
    site = ring_site()
    for row in site['hosts']:
        row['fabric_ip'] = row['management_ip']
    return installer.make_lock(installer.GLM_NO_CACHE[4], site, '1' * 40, '2' * 64)


def test_verified_existing_fabric_never_plans_service_install_or_teardown():
    value = lock()
    assert value['backend'] == 'glm-existing-mesh'
    assert value['selection']['sparkcache'] is False
    up = [p['id'] for p in installer.operation_plan(value, 'up')['phases']]
    assert 'managed-prepare' in up and 'preflight' in up
    assert not {'managed-install', 'managed-up', 'managed-native-check', 'mesh-replace', 'mesh-up'} & set(up)
    assert [p['id'] for p in installer.operation_plan(value, 'down')['phases']] == ['owned', 'stop']


@pytest.fixture
def environment(monkeypatch, tmp_path):
    value = lock()
    spec = SimpleNamespace(name=value['site']['name'] + '-r0')
    info = {'Id': 'a' * 64, 'State': {'Running': False}}
    events = []
    monkeypatch.setattr(lifecycle, 'resolve', lambda *a: (spec, {}, tmp_path, tmp_path / 'image.json'))
    monkeypatch.setattr(host, 'image_info', lambda _: {})
    monkeypatch.setattr(host, 'container', lambda _: info)
    monkeypatch.setattr(host, 'owned', lambda spec, current, image: current)
    monkeypatch.setattr(host, 'admit_image', lambda _: events.append('image'))
    monkeypatch.setattr(host, 'verify_model', lambda *a: events.append('model'))
    monkeypatch.setattr(host, 'require_idle', lambda: events.append('idle'))
    monkeypatch.setattr(host, 'run', lambda argv: events.append(argv))
    monkeypatch.setattr(lifecycle.qwen_mesh, 'check', lambda *a: events.append('mesh-verified'))
    return value, info, events, tmp_path


def test_start_checks_existing_mesh_image_and_weights_before_docker_start(environment):
    value, _, events, state = environment
    assert lifecycle.perform('start', value, 0, state)['ok']
    assert events == ['mesh-verified', 'image', 'model', 'idle', ['docker', 'start', 'a' * 64]]


def test_failed_fabric_verification_prevents_start(environment, monkeypatch):
    value, _, events, state = environment
    def broken(*a):
        raise ValueError('Mesh changed')
    monkeypatch.setattr(lifecycle.qwen_mesh, 'check', broken)
    with pytest.raises(ValueError, match='Mesh changed'):
        lifecycle.perform('start', value, 0, state)
    assert events == []


def test_stop_only_stops_the_owned_container_and_leaves_mesh_alone(environment):
    value, info, events, state = environment
    info['State']['Running'] = True
    assert lifecycle.perform('stop', value, 0, state)['ok']
    assert events == [['docker', 'stop', '--time', '60', 'a' * 64]]
