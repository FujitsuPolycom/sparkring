"""Own non-expiring hardware markers and stop serving when mesh readiness is lost."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import hmac
import http.client
import importlib.util
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import stat
import subprocess
import sys
import threading
import time

_spec = importlib.util.spec_from_file_location('managed_mesh_profile', Path(__file__).with_name('profile.py'))
mesh_profile = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mesh_profile
_spec.loader.exec_module(mesh_profile)

PROTOCOL = 'sparkring-managed-mesh/v1'
POLL_SECONDS = 1.0
PEER_TIMEOUT = 2.0
PEER_OUTAGE_GRACE = 4.0
NETWORK_POLL_SECONDS = 5.0
HEALTH_MAX_AGE = 10.0


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sign(key, value):
    return hmac.new(key, canonical(value), hashlib.sha256).hexdigest()


def read_key(path):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o077:
        raise ValueError('Mesh key must be a root-owned regular file readable only by root')
    value = path.read_bytes()
    if len(value) != 32:
        raise ValueError('Mesh key must contain exactly 32 random bytes')
    return value


def load_config(path):
    document = json.loads(Path(path).read_text())
    expected = {'schema', 'site_path', 'rank', 'key_file', 'epoch', 'health_port', 'state_dir',
                'container_id', 'container_image'}
    if set(document) != expected or document['schema'] != PROTOCOL:
        raise ValueError('Unsupported managed mesh configuration')
    if type(document['rank']) is not int or document['rank'] not in range(4):
        raise ValueError('Mesh rank must be an integer from zero through three')
    if type(document['health_port']) is not int or not 1024 <= document['health_port'] <= 65535:
        raise ValueError('Mesh health port must be an unprivileged TCP port')
    if not isinstance(document['epoch'], str) or not re.fullmatch('[0-9a-f]{32}', document['epoch']):
        raise ValueError('Mesh epoch must be a shared 128-bit hexadecimal identifier')
    if not re.fullmatch('[0-9a-f]{64}', str(document['container_id'])):
        raise ValueError('Pin the full pre-created model container ID')
    if not re.fullmatch('sha256:[0-9a-f]{64}', str(document['container_image'])):
        raise ValueError('Pin the immutable model image ID')
    for name in ('site_path', 'key_file', 'state_dir'):
        mesh_profile.absolute(document[name], name)
    site, topology, plan = mesh_profile.load_site(Path(document['site_path']))
    sources = {name: mesh_profile.sha(Path(__file__).with_name(name))
               for name in ('managed_service.py', 'managed_network.py', 'profile.py', 'inspect_fabric.py')}
    identity = digest({'protocol': PROTOCOL, 'site': site, 'topology': topology.sha256,
                       'epoch': document['epoch'], 'port': document['health_port'], 'sources': sources,
                       'image': document['container_image']})
    return document, site, topology, plan, identity


def fetch_peer(address, port, key, rank, identity, epoch):
    nonce = secrets.token_hex(16)
    challenge = {'protocol': PROTOCOL, 'nonce': nonce}
    # Direct HTTP avoids proxy/redirect behavior and unused TLS context creation.
    connection = http.client.HTTPConnection(address, port, timeout=PEER_TIMEOUT)
    try:
        connection.request('GET', f'/health?nonce={nonce}', headers={'X-Mesh-Auth': sign(key, challenge)})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError('Mesh health endpoint rejected the authenticated request')
        raw = response.read(8193)
    finally:
        connection.close()
    if len(raw) > 8192:
        raise ValueError('Oversized mesh health response')
    envelope = json.loads(raw)
    body = envelope['body']
    if not hmac.compare_digest(sign(key, body), str(envelope['signature'])):
        raise ValueError('Mesh health authentication failed')
    if (body.get('protocol') != PROTOCOL or body.get('nonce') != nonce or body.get('rank') != rank
            or body.get('identity') != identity or body.get('epoch') != epoch
            or not re.fullmatch('[0-9a-f]{32}', str(body.get('generation', '')))):
        raise ValueError('Mesh peer identity or freshness differs')
    if body.get('local_ready') is not True:
        raise RuntimeError(f'Mesh rank {rank} is not locally ready')
    return body


def group_check(site, config, key, identity):
    def call(rank):
        return fetch_peer(site['management_addresses'][rank], config['health_port'], key,
                          rank, identity, config['epoch'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(call, range(4)))


def validate_group(rows):
    if len(rows) != 4 or [row['rank'] for row in rows] != list(range(4)):
        raise ValueError('Expected exactly four ordered mesh ranks')
    generations = {str(row['rank']): row['generation'] for row in rows}
    view = digest(generations)
    if any(row.get('phase') != 'armed' or row.get('view_digest') != view
           or row.get('peer_health_degraded', False) for row in rows):
        raise RuntimeError('Mesh ranks have not armed the same process generation set')
    return view


class PeerWatch:
    """Tolerate short connection loss, never authenticated negative readiness or a new generation."""
    def __init__(self):
        self.generations = None
        self.outage_started = None

    def observe(self, rows):
        observed = {str(row['rank']): row['generation'] for row in rows}
        if self.generations is not None and observed != self.generations:
            raise RuntimeError('A mesh peer process generation changed')
        self.generations = observed
        self.outage_started = None
        return digest(observed)

    def transport_error(self, now):
        if self.outage_started is None:
            self.outage_started = now
        if now - self.outage_started >= PEER_OUTAGE_GRACE:
            raise RuntimeError('Authenticated peer transport remained unavailable beyond its grace interval')


def notify(message):
    address = os.environ.get('NOTIFY_SOCKET')
    if not address:
        return
    if address.startswith('@'):
        address = '\0' + address[1:]
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
        client.connect(address)
        client.sendall(message.encode())


def docker_running(name):
    result = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', name],
                            capture_output=True, text=True, timeout=3)
    if result.returncode:
        if 'No such' in result.stderr:
            return False
        raise RuntimeError('Cannot establish the dependent container state')
    return result.stdout.strip() == 'true'


def stop_model(name):
    """Fail closed; never stop containers outside this rank's configured profile."""
    if not docker_running(name):
        return
    result = subprocess.run(['docker', 'kill', name], capture_output=True, text=True, timeout=5)
    if result.returncode and docker_running(name):
        raise RuntimeError('Could not stop the dependent model container')
    if docker_running(name):
        raise RuntimeError('Dependent model container is still running')


def process_record(pid):
    root = Path('/proc') / str(pid)
    fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': int(fields[19]),
            'argv': (root / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')}


def conflicting_markers(devices, proc_root=Path('/proc')):
    """Reject leftover managed or diagnostic marker CLI processes on reserved devices."""
    found = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = (directory / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        except (OSError, UnicodeError):
            continue
        def option(name):
            for index, value in enumerate(argv):
                if value == name and index + 1 < len(argv):
                    return argv[index + 1]
                if value.startswith(name + '='):
                    return value.split('=', 1)[1]
            return None
        device = option('--device')
        if '--attach' in argv and option('--source-port') == '65535' and device in devices:
            found.append({'pid': int(directory.name), 'device': device})
    return found


def cleanup_orphans(config_path):
    """Reap only recorded child identities after the cluster's model-stop barrier."""
    config, site, _, plan, _ = load_config(config_path)
    if docker_running(config['container_id']):
        raise RuntimeError('Stop the dependent model before orphan cleanup')
    state_dir = Path(config['state_dir'])
    import fcntl
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (state_dir / 'service.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_file = state_dir / 'status.json'
        records = json.loads(status_file.read_text()).get('markers', []) if status_file.exists() else []
        expected_devices = {m.rdma_device for m in plan.markers if m.source_rank == config['rank']}
        for record in records:
            try:
                descriptor = os.pidfd_open(record['pid'])
            except ProcessLookupError:
                continue
            try:
                observed = process_record(record['pid'])
                if observed != record:
                    raise RuntimeError('Recorded marker PID identity changed; refusing to signal')
                argv = observed['argv']
                if (argv[0] != site['marker_binary'] or '--managed' not in argv
                        or argv.count('--device') != 1 or argv[argv.index('--device') + 1] not in expected_devices):
                    raise RuntimeError('Recorded child is not a configured managed marker')
                signal.pidfd_send_signal(descriptor, signal.SIGTERM)
                import select
                poll = select.poll()
                poll.register(descriptor, select.POLLIN)
                if not poll.poll(5000):
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                    if not poll.poll(2000):
                        raise RuntimeError('Recorded marker did not stop')
            finally:
                os.close(descriptor)
        remaining = subprocess.run(['pgrep', '-f', '^' + re.escape(site['marker_binary']) + ' '],
                                   capture_output=True, timeout=3)
        if remaining.returncode != 1:
            raise RuntimeError('Unrecorded marker processes remain; explicit operator inspection is required')
        from managed_network import NetworkManager
        result = NetworkManager(Path(config['site_path']), config['rank'], state_dir / 'network').down()
        print(json.dumps({'orphan_cleanup': result}))
        if not result['clean']:
            raise RuntimeError('Some owned network objects require operator inspection')


class MeshService:
    def __init__(self, config_path):
        self.config, self.site, self.topology, self.plan, self.identity = load_config(config_path)
        self.key = read_key(self.config['key_file'])
        self.rank = self.config['rank']
        self.generation = secrets.token_hex(16)
        self.model = self.config['container_id']
        self.state = {'protocol': PROTOCOL, 'rank': self.rank, 'epoch': self.config['epoch'],
                      'identity': self.identity, 'generation': self.generation,
                      'local_ready': False, 'phase': 'starting', 'view_digest': None}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.children = []
        self.marker_records = []
        self.logfiles = []
        self.network = None
        self.server = None
        self.failed = False
        self.owns_guard = False
        self.last_progress = time.monotonic()
        self.model_seen = False
        self.state_dir = Path(self.config['state_dir'])

    def publish(self, *, best_effort=False, **changes):
        with self.lock:
            self.state.update(changes)
            body = dict(self.state, model=self.model, pid=os.getpid(), updated_unix=time.time(),
                        marker_pids=[child.pid for child in self.children], markers=self.marker_records)
        try:
            target = self.state_dir / 'status.json'
            temporary = target.with_suffix('.tmp')
            temporary.write_bytes(canonical(body))
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
        except OSError:
            if not best_effort:
                raise

    def health_body(self, nonce):
        with self.lock:
            body = dict(self.state, nonce=nonce)
        if (time.monotonic() - self.last_progress > HEALTH_MAX_AGE
                or len(self.children) != 2 or any(child.poll() is not None for child in self.children)):
            body['local_ready'] = False
        return body

    def start_server(self):
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                match = re.fullmatch(r'/health\?nonce=([0-9a-f]{32})', self.path)
                if not match:
                    self.send_error(404)
                    return
                challenge = {'protocol': PROTOCOL, 'nonce': match[1]}
                if not hmac.compare_digest(sign(owner.key, challenge), self.headers.get('X-Mesh-Auth', '')):
                    self.send_error(403)
                    return
                body = owner.health_body(match[1])
                encoded = canonical({'body': body, 'signature': sign(owner.key, body)})
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(encoded)))
                self.end_headers()
                try:
                    self.wfile.write(encoded)
                except ConnectionError:
                    pass

            def log_message(self, *args):
                pass

        class Server(HTTPServer):
            allow_reuse_address = True

            def get_request(self):
                connection, address = super().get_request()
                connection.settimeout(0.5)
                return connection, address

        self.server = Server((self.site['management_addresses'][self.rank], self.config['health_port']), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def start_markers(self):
        binary = self.site['marker_binary']
        if mesh_profile.sha(Path(binary)) != self.site['marker_binary_sha256']:
            raise ValueError('Managed marker digest differs from the site')
        devices = {marker.rdma_device for marker in self.plan.markers if marker.source_rank == self.rank}
        conflicts = conflicting_markers(devices)
        if conflicts:
            raise RuntimeError(f'An existing marker uses reserved devices: {conflicts}')
        existing = subprocess.run(['pgrep', '-f', '^' + re.escape(binary) + ' '], capture_output=True, timeout=3)
        if existing.returncode != 1:
            raise RuntimeError('Configured marker already exists outside this service')
        for marker in self.plan.markers:
            if marker.source_rank != self.rank:
                continue
            path = self.state_dir / f'{marker.rdma_device}-{self.generation}.log'
            output = path.open('xb')
            self.logfiles.append(output)
            child = subprocess.Popen([binary, '--device', marker.rdma_device, '--source-port', '65535',
                                      '--replacement-ethertype', '0x88b5', '--attach', '--managed'],
                                     stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
            self.children.append(child)
            self.marker_records.append(process_record(child.pid))
            self.publish()
            deadline = time.monotonic() + 10
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f'Managed marker exited before readiness: {path.name}')
                try:
                    ready = json.loads(path.read_text())
                except (ValueError, OSError):
                    ready = None
                if ready is not None:
                    if (ready.get('attached') is not True or ready.get('managed') is not True
                            or ready.get('lifetime_seconds', 'missing') is not None):
                        raise RuntimeError('Native marker did not confirm managed attachment')
                    record = process_record(child.pid)
                    if record['argv'] != [binary, '--device', marker.rdma_device, '--source-port', '65535',
                                          '--replacement-ethertype', '0x88b5', '--attach', '--managed']:
                        raise RuntimeError('Ready marker process arguments differ from its launch')
                    self.marker_records[-1] = record
                    self.publish()
                    break
                if self.stop.wait(0.05) or time.monotonic() > deadline:
                    raise RuntimeError('Managed marker readiness deadline exceeded')
        if len(self.children) != 2:
            raise RuntimeError('Expected exactly two managed source markers')

    def run(self):
        if os.geteuid() != 0:
            raise PermissionError('Managed mesh service requires root')
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.state_dir.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0:
            raise ValueError('State directory must be a root-owned directory, not a symlink')
        os.chmod(self.state_dir, 0o700)
        # The lock lives for the entire service, not only individual network operations.
        import fcntl
        service_lock = (self.state_dir / 'service.lock').open('a')
        fcntl.flock(service_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: self.stop.set())
        try:
            if docker_running(self.model):
                raise RuntimeError('Stop the dependent model before starting mesh ownership')
            self.owns_guard = True
            from managed_network import NetworkManager
            self.network = NetworkManager(Path(self.config['site_path']), self.rank, self.state_dir / 'network')
            self.network.up()
            self.start_markers()
            self.start_server()
            self.last_progress = time.monotonic()
            self.publish(local_ready=True)
            notify('READY=1\nSTATUS=Local fabric ready; waiting for authenticated peers')
            peer_watch = PeerWatch()
            last_network = time.monotonic()
            last_full_network = last_network
            while not self.stop.is_set():
                if any(child.poll() is not None for child in self.children):
                    raise RuntimeError('A managed source marker exited')
                if time.monotonic() - last_network >= NETWORK_POLL_SECONDS:
                    full = time.monotonic() - last_full_network >= 60
                    self.network.check(verify_rdma_mtu=full)
                    if full:
                        last_full_network = time.monotonic()
                    last_network = time.monotonic()
                try:
                    rows = group_check(self.site, self.config, self.key, self.identity)
                    view = peer_watch.observe(rows)
                    self.publish(phase='armed', view_digest=view, peer_health_degraded=False)
                except OSError:
                    if peer_watch.generations is not None:
                        peer_watch.transport_error(time.monotonic())
                    elif docker_running(self.model):
                        raise
                    self.publish(peer_health_degraded=True)
                except Exception:
                    if peer_watch.generations is not None or docker_running(self.model):
                        raise
                if self.state['phase'] != 'armed' and docker_running(self.model):
                    raise RuntimeError('Dependent model started before four-rank readiness')
                intent_path = self.state_dir / 'model-intent.json'
                if intent_path.exists():
                    intent = json.loads(intent_path.read_text())
                    if intent.get('generation') == self.generation and intent.get('active') is True:
                        if docker_running(self.model):
                            self.model_seen = True
                        elif self.model_seen or time.monotonic() > intent['deadline_monotonic']:
                            raise RuntimeError('Dependent model exited or failed to start')
                    else:
                        self.model_seen = False
                self.last_progress = time.monotonic()
                notify('WATCHDOG=1')
                self.stop.wait(POLL_SECONDS)
        except Exception as error:
            self.failed = True
            self.publish(best_effort=True, local_ready=False, phase='failed', error=str(error))
            print(json.dumps({'event': 'mesh_failure', 'rank': self.rank, 'error': str(error)}), flush=True)
        finally:
            self.publish(best_effort=True, local_ready=False, phase='failed' if self.failed else 'stopping')
            while self.owns_guard:
                try:
                    stop_model(self.model)
                    break
                except Exception as error:
                    # Keep forwarding while Docker cannot confirm that serving stopped.
                    # The unit must not kill marker children on watchdog/main-process loss.
                    print(json.dumps({'event': 'model_stop_failed', 'error': str(error)}), flush=True)
                    self.failed = True
                    notify('WATCHDOG=1\nSTATUS=Failed; retaining forwarding until model stop is confirmed')
                    time.sleep(1)
            for child in self.children:
                if child.poll() is None:
                    child.terminate()
            for child in self.children:
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=2)
            for output in self.logfiles:
                output.close()
            self.marker_records = []
            if self.network is not None:
                try:
                    cleanup = self.network.down()
                    if cleanup.get('clean') is False:
                        self.failed = True
                        print(json.dumps({'event': 'network_cleanup_incomplete', 'result': cleanup}), flush=True)
                except Exception as error:
                    self.failed = True
                    print(json.dumps({'event': 'network_cleanup_incomplete', 'error': str(error)}), flush=True)
            if self.server is not None:
                self.server.shutdown()
                self.server.server_close()
            self.publish(best_effort=True, local_ready=False, phase='failed' if self.failed else 'stopped')
            service_lock.close()
        return 1 if self.failed else 0


def gate(config_path, timeout):
    config, site, _, _, identity = load_config(config_path)
    key = read_key(config['key_file'])
    inspected = subprocess.run(['docker', 'inspect', config['container_id']],
                               capture_output=True, text=True, check=True, timeout=5)
    container = json.loads(inspected.stdout)[0]
    if (container['Id'] != config['container_id'] or container['Image'] != config['container_image']
            or container['Name'].lstrip('/') != site['container_prefix'] + f"-r{config['rank']}"):
        raise ValueError('Prepared model container identity differs from the managed contract')
    deadline = time.monotonic() + timeout
    while True:
        try:
            rows = group_check(site, config, key, identity)
            view = validate_group(rows)
            print(json.dumps({'ready': True, 'identity': identity, 'view_digest': view, 'ranks': rows}))
            return 0
        except Exception as error:
            if time.monotonic() >= deadline:
                raise RuntimeError(f'Four-rank readiness failed: {error}') from error
            time.sleep(0.25)


def model_intent(config_path, active):
    config, *_ = load_config(config_path)
    state_dir = Path(config['state_dir'])
    status_path = state_dir / 'status.json'
    if not status_path.exists():
        if active:
            raise RuntimeError('Mesh service has no readiness state')
        return
    status = json.loads(status_path.read_text())
    if active and (status.get('phase') != 'armed' or status.get('local_ready') is not True):
        raise RuntimeError('Mesh is not armed for model startup')
    record = {'generation': status['generation'], 'active': active,
              'deadline_monotonic': time.monotonic() + 15}
    temporary = state_dir / 'model-intent.tmp'
    temporary.write_bytes(canonical(record))
    os.chmod(temporary, 0o600)
    os.replace(temporary, state_dir / 'model-intent.json')


def reset_units():
    for name in ('sparkring-mesh.service', 'sparkring-mesh-model.service'):
        result = subprocess.run(['systemctl', 'show', name, '--property=LoadState,ActiveState'],
                                capture_output=True, text=True, check=True, timeout=5)
        fields = dict(line.split('=', 1) for line in result.stdout.splitlines())
        if fields.get('LoadState') != 'loaded':
            raise RuntimeError('Managed systemd unit definition is missing')
        if fields.get('ActiveState') == 'failed':
            subprocess.run(['systemctl', 'reset-failed', name], check=True, timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'gate', 'stop-model', 'cleanup', 'model-arm',
                                         'model-quiesce', 'model-stopped', 'reset-units'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    if not 0 <= args.timeout <= 900:
        parser.error('Timeout must be between zero and 900 seconds')
    if args.action == 'run':
        raise SystemExit(MeshService(args.config).run())
    if args.action == 'gate':
        gate(args.config, args.timeout)
    elif args.action == 'cleanup':
        cleanup_orphans(args.config)
    elif args.action in ('model-arm', 'model-quiesce'):
        model_intent(args.config, args.action == 'model-arm')
    elif args.action == 'model-stopped':
        config, *_ = load_config(args.config)
        if docker_running(config['container_id']):
            raise RuntimeError('Dependent model is still running')
        print(json.dumps({'stopped': True, 'container_id': config['container_id']}))
    elif args.action == 'reset-units':
        reset_units()
    else:
        config, site, *_ = load_config(args.config)
        stop_model(config['container_id'])


if __name__ == '__main__':
    main()
