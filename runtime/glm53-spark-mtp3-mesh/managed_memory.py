"""Guard GLM startup with available-memory and contiguous-allocation checks."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time

MIN_AVAILABLE = 96 << 30
BLOCK_BYTES = 32 << 20
MIN_BLOCKS = 200


def evaluate(meminfo, buddyinfo, page_size):
    """Count Normal-zone blocks using the shared GLM preflight thresholds."""
    fields = dict(line.split(':', 1) for line in meminfo.splitlines() if ':' in line)
    available = int(fields['MemAvailable'].split()[0]) * 1024
    if page_size <= 0 or BLOCK_BYTES % page_size:
        raise ValueError('Unsupported host page size for the 32 MiB allocation check')
    pages = BLOCK_BYTES // page_size
    if pages < 1 or pages & (pages - 1):
        raise ValueError('Unsupported host page size for the 32 MiB allocation check')
    order = pages.bit_length() - 1
    blocks = 0
    found = False
    for line in buddyinfo.splitlines():
        words = line.split()
        if len(words) >= 5 and words[3] == 'Normal':
            found = True
            counts = [int(value) for value in words[4:]]
            if any(value < 0 for value in counts):
                raise ValueError('Invalid buddy allocator counters')
            blocks += sum(count << (index - order) for index, count in enumerate(counts) if index >= order)
    if not found:
        raise ValueError('Normal-zone buddy allocator counters are unavailable')
    return {'available_bytes': available, 'equivalent_32mib_blocks': blocks,
            'minimum_available_bytes': MIN_AVAILABLE, 'minimum_blocks': MIN_BLOCKS,
            'passed': available >= MIN_AVAILABLE and blocks >= MIN_BLOCKS}


def snapshot():
    return evaluate(Path('/proc/meminfo').read_text(), Path('/proc/buddyinfo').read_text(),
                    os.sysconf('SC_PAGE_SIZE'))


def require_ready(value):
    if not value['passed']:
        raise RuntimeError('Host memory is not ready for GLM startup: ' + json.dumps(value)
                           + '. Run the coordinated start-model command to prepare memory.')


@contextmanager
def start_lock(config):
    """Serialize compaction with managed model arming on this host."""
    import fcntl
    path = Path(config['state_dir']) / 'memory-start.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run(argv, timeout=10):
    return subprocess.run(argv, capture_output=True, text=True, check=True, timeout=timeout).stdout


def assert_idle(config):
    """Refuse host mutation while a container, GPU workload, or serving port is active."""
    intent = Path(config['state_dir']) / 'model-intent.json'
    if intent.exists() and json.loads(intent.read_text()).get('active') is True:
        raise RuntimeError('Model startup is armed; quiesce it before memory preparation')
    if run(['docker', 'ps', '--quiet']).strip():
        raise RuntimeError('Stop all running containers before host memory preparation')
    pids = run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']).strip()
    if pids:
        raise RuntimeError('GPU workloads are present or cannot be excluded: ' + pids)
    container = json.loads(run(['docker', 'inspect', config['container_id']]))[0]
    if container['Image'] != config['container_image'] or container['State']['Running']:
        raise RuntimeError('Expected stopped, source-pinned model container')
    env = dict(value.split('=', 1) for value in container['Config']['Env'] if '=' in value)
    argv = container['Config']['Cmd']
    if argv.count('--master-port') != 1:
        raise ValueError('Expected exactly one rendezvous port in model arguments')
    ports = [env['PORT'], env['SPARKRING_LIVENESS_PORT'], argv[argv.index('--master-port') + 1]]
    for value in ports:
        port = int(value)
        if not 1 <= port <= 65535:
            raise ValueError('Invalid serving port in model container')
        if run(['ss', '-ltnH', f'sport = :{port}']).strip():
            raise RuntimeError(f'Refusing memory preparation while TCP port {port} is listening')


def reclaim():
    """Flush writes, release clean page cache, and request bounded kernel compaction."""
    run(['sync'], timeout=60)
    run(['sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'], timeout=90)
    time.sleep(2)


def prepare(config):
    with start_lock(config):
        before = snapshot()
        if before['passed']:
            return {'status': 'ready', 'reclaimed': False, 'before': before, 'after': before, 'passed': True}
        assert_idle(config)
        reclaim()
        after = snapshot()
        return {'status': 'recovered' if after['passed'] else 'reboot-required',
                'reclaimed': True, 'before': before, 'after': after, 'passed': after['passed']}
