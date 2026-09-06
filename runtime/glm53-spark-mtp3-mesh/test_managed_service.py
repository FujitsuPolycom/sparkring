"""GPU-free authentication, readiness, and lifecycle contract checks."""
import importlib.util
from pathlib import Path
import threading
import time

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('managed_service_tests_subject', HERE / 'managed_service.py')
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)
units_spec = importlib.util.spec_from_file_location('managed_units_tests_subject', HERE / 'managed_units.py')
units = importlib.util.module_from_spec(units_spec)
units_spec.loader.exec_module(units)


def test_signature_is_canonical_and_authenticated():
    key = b'a' * 32
    assert service.sign(key, {'a': 1, 'b': 2}) == service.sign(key, {'b': 2, 'a': 1})
    assert service.sign(key, {'a': 1}) != service.sign(key, {'a': 2})
    assert service.sign(key, {'a': 1}) != service.sign(b'b' * 32, {'a': 1})


def rows():
    generations = {str(rank): f'{rank:032x}' for rank in range(4)}
    return [{'rank': rank, 'generation': generations[str(rank)], 'phase': 'armed',
             'view_digest': service.digest(generations)} for rank in range(4)]


def test_common_generation_view_required():
    assert service.validate_group(rows())
    changed = rows()
    changed[2]['generation'] = 'f' * 32
    with pytest.raises(RuntimeError, match='same process generation'):
        service.validate_group(changed)


def test_one_connection_timeout_does_not_destroy_a_healthy_generation():
    watch = service.PeerWatch()
    view = watch.observe(rows())
    watch.transport_error(10.0)
    assert watch.observe(rows()) == view
    assert watch.outage_started is None


def test_sustained_peer_connection_loss_latches_failure():
    watch = service.PeerWatch()
    watch.observe(rows())
    watch.transport_error(10.0)
    watch.transport_error(13.9)
    with pytest.raises(RuntimeError, match='grace'):
        watch.transport_error(14.0)


def test_generation_change_is_not_given_a_transport_grace():
    watch = service.PeerWatch()
    watch.observe(rows())
    changed = rows()
    changed[1]['generation'] = 'f' * 32
    with pytest.raises(RuntimeError, match='generation changed'):
        watch.observe(changed)


def test_degraded_peer_blocks_new_model_admission():
    changed = rows()
    changed[0]['peer_health_degraded'] = True
    with pytest.raises(RuntimeError):
        service.validate_group(changed)


@pytest.mark.parametrize('phase', ['starting', 'failed', 'stopping', 'stopped'])
def test_group_rejects_unarmed_peer(phase):
    changed = rows()
    changed[1]['phase'] = phase
    with pytest.raises(RuntimeError):
        service.validate_group(changed)


def test_group_rejects_missing_or_duplicate_rank():
    with pytest.raises(ValueError):
        service.validate_group(rows()[:3])
    with pytest.raises(ValueError):
        service.validate_group([rows()[0]] * 4)


class Child:
    def __init__(self, code=None):
        self.code = code

    def poll(self):
        return self.code


def owner():
    result = service.MeshService.__new__(service.MeshService)
    result.lock = threading.Lock()
    result.state = {'local_ready': True, 'phase': 'armed'}
    result.last_progress = time.monotonic()
    result.children = [Child(), Child()]
    return result


def test_health_invalidates_immediately_on_marker_exit():
    result = owner()
    assert result.health_body('nonce')['local_ready']
    result.children[0].code = 1
    assert not result.health_body('nonce')['local_ready']


def test_health_rejects_stalled_monitor_even_if_http_thread_is_alive():
    result = owner()
    result.last_progress -= service.HEALTH_MAX_AGE + 1
    assert not result.health_body('nonce')['local_ready']


def test_health_requires_exactly_two_children():
    result = owner()
    result.children = []
    assert not result.health_body('nonce')['local_ready']


def test_cleanup_status_write_is_best_effort(tmp_path, monkeypatch):
    result = owner()
    result.model, result.marker_records = 'a' * 64, []
    result.children = []
    result.state_dir = tmp_path
    monkeypatch.setattr(Path, 'write_bytes', lambda *args: (_ for _ in ()).throw(OSError('disk full')))
    result.publish(best_effort=True, local_ready=False)
    assert result.state['local_ready'] is False
    with pytest.raises(OSError):
        result.publish(local_ready=True)


def test_peer_response_nonce_is_checked(monkeypatch):
    key = b'a' * 32
    body = {'protocol': service.PROTOCOL, 'nonce': 'wrong', 'rank': 0,
            'identity': 'test', 'epoch': 'epoch', 'generation': 'a' * 32, 'local_ready': True}
    raw = service.canonical({'body': body, 'signature': service.sign(key, body)})
    class Response:
        status = 200

        def read(self, limit):
            return raw
    class Connection:
        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return Response()

        def close(self):
            pass
    monkeypatch.setattr(service.http.client, 'HTTPConnection', lambda *args, **kwargs: Connection())
    with pytest.raises(ValueError, match='freshness'):
        service.fetch_peer('127.0.0.1', 9975, key, 0, 'test', 'epoch')


def test_model_stop_checks_state_after_kill(monkeypatch):
    states = iter([True, False])
    monkeypatch.setattr(service, 'docker_running', lambda name: next(states))
    calls = []
    class Result:
        returncode = 0
    monkeypatch.setattr(service.subprocess, 'run', lambda argv, **kw: calls.append(argv) or Result())
    service.stop_model('a' * 64)
    assert calls == [['docker', 'kill', 'a' * 64]]


def test_units_bind_model_and_disable_automatic_recovery():
    rendered = units.unit_text('/opt/sparkring/managed-mesh', '/etc/sparkring/managed-mesh', 'a' * 64)
    mesh, model = rendered['sparkring-mesh.service'], rendered['sparkring-mesh-model.service']
    assert 'Restart=no' in mesh and 'Restart=no' in model
    assert 'BindsTo=sparkring-mesh.service' in model
    assert 'After=docker.service sparkring-mesh.service' in model
    assert ' gate --config ' in model
    assert 'KillMode=process' in mesh and 'TimeoutStopSec=infinity' in mesh
    assert 'RuntimeDirectoryPreserve=yes' in mesh


@pytest.mark.parametrize('path', ['/opt/space here', '/opt/%n', '/opt/../tmp', 'relative'])
def test_unit_paths_reject_injection(path):
    with pytest.raises(ValueError):
        units.systemd_path(path)


def test_key_rejects_short_material(tmp_path, monkeypatch):
    path = tmp_path / 'key'
    path.write_bytes(b'short')
    path.chmod(0o600)
    # Ownership is tested on Linux deployment; this fixture tests key length.
    class Info:
        st_mode = 0o100600
        st_uid = 0
    monkeypatch.setattr(Path, 'lstat', lambda self: Info())
    with pytest.raises(ValueError, match='32 random bytes'):
        service.read_key(path)


def test_old_marker_path_is_not_silently_overlapped(tmp_path):
    process = tmp_path / '123'
    process.mkdir()
    (process / 'cmdline').write_bytes(b'\0'.join([b'/old/diagnostic-helper', b'--attach',
        b'--device=rocep1s0f0', b'--source-port', b'65535']) + b'\0')
    assert service.conflicting_markers({'rocep1s0f0'}, tmp_path) == [{'pid': 123, 'device': 'rocep1s0f0'}]
    assert service.conflicting_markers({'rocep1s0f1'}, tmp_path) == []
