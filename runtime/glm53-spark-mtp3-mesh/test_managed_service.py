"""GPU-free authentication, readiness, and lifecycle contract checks."""
import importlib.util
from pathlib import Path
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

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
    watch.transport_error(309.9)
    with pytest.raises(RuntimeError, match='grace'):
        watch.transport_error(310.0)


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


@pytest.mark.parametrize("count", [0, 1, 3])
def test_health_requires_exactly_two_children(count):
    result = owner()
    result.children = [Child() for _ in range(count)]
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


def test_docker_inspect_timeout_is_unknown_without_stalling_fabric_monitor(monkeypatch):
    pending = Future()
    submitted = []

    class Executor:
        def submit(self, function, name):
            submitted.append((function, name))
            return pending

        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(service.concurrent.futures, 'ThreadPoolExecutor', lambda **kw: Executor())
    watcher = service.DockerStatePoll('a' * 64)
    assert watcher.poll() is None
    for _ in range(20):
        assert watcher.poll() is None
    assert len(submitted) == 1
    command = ['docker', 'inspect', '--format', '{{.State.Running}}', 'a' * 64]
    pending.set_exception(service.subprocess.TimeoutExpired(command, 3))
    assert watcher.poll() is None
    assert '3 seconds' in watcher.error
    assert len(submitted) == 2
    watcher.close()


def test_docker_status_recovers_without_reusing_stale_stopped_evidence(monkeypatch):
    futures = [Future(), Future(), Future()]
    pending = iter(futures)
    executor = SimpleNamespace(submit=lambda *args: next(pending), shutdown=lambda **kw: None)
    monkeypatch.setattr(service.concurrent.futures, 'ThreadPoolExecutor', lambda **kw: executor)
    watcher = service.DockerStatePoll('a' * 64)
    assert watcher.poll() is None
    futures[0].set_result(False)
    assert watcher.poll() is False
    assert watcher.poll() is None
    futures[1].set_result(True)
    assert watcher.poll() is True
    assert watcher.error is None
    watcher.close()


def test_unknown_docker_status_blocks_model_admission():
    changed = rows()
    changed[0]['docker_status_degraded'] = True
    with pytest.raises(RuntimeError):
        service.validate_group(changed)


def test_model_arm_rechecks_local_docker_status_after_group_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'load_config', lambda path: ({'state_dir': str(tmp_path)},))
    (tmp_path / 'status.json').write_bytes(service.canonical({
        'phase': 'armed', 'local_ready': True, 'generation': 'g', 'docker_status_degraded': True,
    }))
    with pytest.raises(RuntimeError, match='not armed'):
        service.model_intent('unused', True)
    assert not (tmp_path / 'model-intent.json').exists()


@pytest.mark.parametrize('message, missing', [
    ('Error: No such object: ' + 'a' * 64, True),
    ('error: no such object: ' + 'a' * 64, True),
    ('error: no such container: ' + 'a' * 64 + '\n', True),
    ('Error: No such object: ' + 'b' * 64, False),
    ('error: no such object: ' + 'b' * 64, False),
    ('No such file or directory: Docker socket', False),
])
def test_only_pinned_missing_container_proves_absence(monkeypatch, message, missing):
    monkeypatch.setattr(service.subprocess, 'run', lambda *args, **kw:
                        SimpleNamespace(returncode=1, stdout='', stderr=message))
    if missing:
        assert service.docker_running('a' * 64) is False
    else:
        with pytest.raises(RuntimeError):
            service.docker_running('a' * 64)


@pytest.mark.parametrize('output', ['', 'unexpected', 'False'])
def test_unknown_docker_reply_cannot_prove_model_stopped(monkeypatch, output):
    monkeypatch.setattr(service.subprocess, 'run', lambda *args, **kw:
                        SimpleNamespace(returncode=0, stdout=output, stderr=''))
    with pytest.raises(RuntimeError):
        service.stop_model('a' * 64)


def test_docker_timeout_cannot_pass_model_stop_barrier(monkeypatch):
    calls = []

    def timeout(argv, **kwargs):
        calls.append((argv, kwargs['timeout']))
        raise service.subprocess.TimeoutExpired(argv, kwargs['timeout'])

    monkeypatch.setattr(service.subprocess, 'run', timeout)
    with pytest.raises(service.subprocess.TimeoutExpired):
        service.stop_model('a' * 64)
    assert len(calls) == 1
    assert calls[0][1] == 3


@pytest.mark.parametrize('peer_failure', [False, True])
@pytest.mark.parametrize('marker_failure', [False, True])
def test_monitor_keeps_fabric_checks_live_when_docker_is_unknown(tmp_path, monkeypatch, peer_failure, marker_failure):
    result = owner()
    result.rank, result.generation, result.model = 0, 'g', 'a' * 64
    result.config = {'site_path': '/unused'}
    result.site, result.identity, result.key = {}, 'identity', b'k' * 32
    result.state_dir, result.network, result.server = tmp_path, None, None
    result.marker_records, result.logfiles = [{'pid': 101}, {'pid': 102}], []
    result.failed, result.owns_guard, result.model_seen = False, False, True
    events = []
    samples = iter([True, None, True])
    clock = [100.0]
    monitor_round = [0]

    class Stop:
        def is_set(self):
            return monitor_round[0] >= 3

        def wait(self, seconds):
            monitor_round[0] += 1
            clock[0] += 6

    result.stop = Stop()
    result.start_markers = lambda: None
    result.start_server = lambda: None
    result.publish = lambda **changes: result.state.update(changes)
    def marker_wait(**kwargs):
        if marker_failure:
            raise service.subprocess.TimeoutExpired(['marker'], kwargs['timeout'])
    result.children = [SimpleNamespace(
        pid=101 + index, poll=lambda: None,
        terminate=lambda: events.append('marker-stop'),
        kill=lambda: events.append('marker-kill'), wait=marker_wait,
    ) for index in range(2)]
    result.server = SimpleNamespace(shutdown=lambda: events.append('server-shutdown'),
                                    server_close=lambda: events.append('server-close'))
    lock_handles = []
    original_open = type(tmp_path).open
    def track_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path.name == 'service.lock':
            lock_handles.append(handle)
        return handle
    monkeypatch.setattr(type(tmp_path), 'open', track_open)
    (tmp_path / 'model-intent.json').write_bytes(service.canonical({
        'generation': 'g', 'active': True, 'deadline_monotonic': 0,
    }))
    monkeypatch.setattr(service.os, 'geteuid', lambda: 0, raising=False)
    original_lstat = type(tmp_path).lstat

    def fixture_lstat(path):
        info = original_lstat(path)
        if path == tmp_path:
            return SimpleNamespace(st_mode=info.st_mode, st_uid=0)
        return info

    # Model the root-owned service directory without changing host ownership.
    monkeypatch.setattr(type(tmp_path), 'lstat', fixture_lstat)
    monkeypatch.setattr(service.signal, 'signal', lambda *args: None)
    monkeypatch.setitem(service.sys.modules, 'fcntl', SimpleNamespace(
        flock=lambda *args: None, LOCK_EX=1, LOCK_NB=2,
    ))
    monkeypatch.setitem(service.sys.modules, 'managed_network', SimpleNamespace(
        NetworkManager=lambda *args: SimpleNamespace(
            up=lambda: None,
            check=lambda **kw: events.append('network-check'),
            down=lambda: events.append('network-down') or {'clean': True},
        )
    ))
    monkeypatch.setattr(service.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(service.time, 'sleep', lambda seconds: events.append('retain-markers'))
    monkeypatch.setattr(service, 'notify', lambda message: events.append(message))
    monkeypatch.setattr(service, 'docker_running', lambda name: events.append('startup-inspect') or False)
    monkeypatch.setattr(service, 'DockerStatePoll', lambda name: SimpleNamespace(
        poll=lambda: next(samples), error='Docker inspect timed out after 3 seconds', close=lambda: None,
    ))

    def group_check(*args):
        events.append('peer-check')
        if peer_failure and monitor_round[0] == 1:
            raise RuntimeError('Authenticated peer is not locally ready')
        return rows()

    stop_attempts = [0]

    def stop_model(name):
        stop_attempts[0] += 1
        if stop_attempts[0] == 1:
            assert 'marker-stop' not in events and 'network-down' not in events
            raise service.subprocess.TimeoutExpired(['docker', 'inspect', name], 3)
        events.append('model-stop-confirmed')

    monkeypatch.setattr(service, 'group_check', group_check)
    monkeypatch.setattr(service, 'stop_model', stop_model)
    assert result.run() == 1  # The first model-stop inspection timed out and required a retry.
    assert events.count('startup-inspect') == 1
    assert events.count('peer-check') == (2 if peer_failure else 3)
    assert 'network-check' in events
    assert events.index('retain-markers') < events.index('model-stop-confirmed')
    assert events.index('model-stop-confirmed') < events.index('marker-stop')
    if marker_failure:
        assert 'network-down' not in events
        assert result.marker_records == [{'pid': 101}, {'pid': 102}]
        assert result.state['marker_stop_unconfirmed'] == [101, 102]
    else:
        assert events.index('marker-stop') < events.index('network-down')
        assert result.marker_records == []
    assert events[-1] == 'server-close'
    assert lock_handles and all(handle.closed for handle in lock_handles)
    if peer_failure:
        assert 'not locally ready' in result.state['error']
    else:
        assert 'error' not in result.state
        assert result.state['docker_status_degraded'] is False


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


@pytest.mark.parametrize("code,message", [(0, "markers found"), (2, "Cannot enumerate"), (3, "Cannot enumerate")])
def test_pgrep_error_is_not_reported_as_marker_presence(code, message):
    with pytest.raises(RuntimeError, match=message):
        service.require_no_markers(SimpleNamespace(returncode=code), "markers found")
    service.require_no_markers(SimpleNamespace(returncode=1), "markers found")


def test_numeric_container_id_rejected_before_site_access(tmp_path, monkeypatch):
    import json
    config = {'schema': service.PROTOCOL, 'site_path': '/unused/site', 'rank': 0,
        'key_file': '/unused/key', 'epoch': 'a' * 32, 'health_port': 9975,
        'state_dir': '/unused/state', 'container_id': int('1' * 64),
        'container_image': 'sha256:' + 'a' * 64}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(service.mesh_profile, 'load_site', lambda *args: pytest.fail('site accessed before type rejection'))
    with pytest.raises(ValueError, match='full pre-created model container ID'):
        service.load_config(path)


def test_invalid_marker_count_rejected_before_process_launch(monkeypatch):
    result = owner()
    result.rank = 0
    result.plan = SimpleNamespace(markers=[SimpleNamespace(source_rank=0)])
    monkeypatch.setattr(service.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('marker spawned'))
    with pytest.raises(RuntimeError, match='before launch'):
        result.start_markers()


def test_unconfirmed_marker_exit_sets_failure_and_retains_identity():
    result = owner()
    result.failed = False
    result.marker_records = [{'pid': 123, 'start_ticks': 456, 'argv': ['marker']}]
    result.publish = lambda **changes: result.state.update(changes)
    def wait(**kwargs):
        raise service.subprocess.TimeoutExpired(['marker'], kwargs['timeout'])
    result.children = [SimpleNamespace(pid=123, poll=lambda: None, terminate=lambda: None,
                                       kill=lambda: None, wait=wait)]
    assert result.stop_markers() is False
    assert result.failed is True
    assert result.marker_records[0]['start_ticks'] == 456
    assert result.state['marker_stop_unconfirmed'] == [123]
