"""Exercise stream admission and shared staging order without CUDA hardware."""
import ast
import contextlib
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'third_party/b12x_roce/b12x/comm/roce/roce_oneshot.py'


def methods(names, namespace):
    tree = ast.parse(SOURCE.read_text())
    nodes = [node for cls in tree.body if isinstance(cls, ast.ClassDef)
             for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    nodes += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    if '_MAX_MESSAGE_BYTES' in namespace:
        constants = [node for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == '_MAX_MESSAGE_BYTES'
                             for target in node.targets)]
        assert len(constants) == 1
        nodes = constants + nodes
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('sequence', [0, 1, 0x80000000, 0xFFFFFFFF])
def test_timeout_health_uses_flag_and_unsigned_sequence(sequence):
    namespace = methods({'check_health', 'poisoned'}, {})
    signed = sequence if sequence < 0x80000000 else sequence - (1 << 32)
    runtime = SimpleNamespace(_lock=threading.RLock(), _proxy=None, rank=1,
                              _ctrl_np=[0, 0, signed, 2, 0, 0, 1])
    assert namespace['poisoned'].fget(runtime)
    with pytest.raises(RuntimeError) as error:
        namespace['check_health'](runtime)
    assert f'at sequence {sequence};' in str(error.value)
    assert f'epoch stopped at {(sequence - 1) & 0xFFFFFFFF},' in str(error.value)
    runtime._ctrl_np = [0] * 7
    assert not namespace['poisoned'].fget(runtime)
    namespace['check_health'](runtime)


def test_per_call_capture_context_cannot_reset_cuda_capture_identity():
    state = SimpleNamespace(stream='A', capture_id=123)
    namespace = methods({'capture', '_order_stream'}, {
        'contextmanager':contextmanager, 'Optional':__import__('typing').Optional,
        'torch':SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda device:state.stream)),
        '_capture_id':lambda stream:state.capture_id})
    runtime = SimpleNamespace(device=0, _capture_id=0, _capture_stream=None, _last_stream=None)
    with namespace['capture'](runtime):
        namespace['_order_stream'](runtime, True)
    state.stream = 'B'
    with pytest.raises(RuntimeError, match='one stream'):
        with namespace['capture'](runtime):
            namespace['_order_stream'](runtime, True)
    state.capture_id = 124
    with namespace['capture'](runtime):
        namespace['_order_stream'](runtime, True)
    assert runtime._capture_stream == 'B'


def test_misaligned_input_waits_before_shared_scratch_copy():
    events = []
    class Tensor:
        shape = (4, 4096)
        dtype = 'bf16'
        device = 0
        def __init__(self, address): self.address = address
        def is_contiguous(self): return True
        def data_ptr(self): return self.address
        def numel(self): return 4 * 4096
        def element_size(self): return 2
        def copy_(self, other): events.append('copy')
    cuda = SimpleNamespace(device=lambda _:nullcontext(), is_current_stream_capturing=lambda:False)
    namespace = methods({'all_reduce'}, {
        'torch':SimpleNamespace(Tensor=Tensor,cuda=cuda),
        'Optional':__import__('typing').Optional, 'Sequence':__import__('typing').Sequence,
        '_nullcontext':nullcontext, 'PACK_BYTES':16,
        'is_launcher_prepared':lambda *args:True,
        'get_launcher':lambda *args:lambda *a:events.append('launch')})
    runtime = SimpleNamespace(_lock=nullcontext(), device=0, check_health=lambda:None,
        should_allreduce=lambda inp:True, _launcher_key=lambda dtype:(),
        _aligned_scratch=lambda which,like:Tensor(32),
        _order_stream=lambda capturing:events.append('wait'),
        _mark_stream=lambda capturing:events.append('record'),
        _recv_base=0,_flag_base=0,_send_base=0,_ctrl_base=0,_slot_bytes=0,
        _epoch_address=0,spin_limit=1,_blocks=1)
    namespace['all_reduce'](runtime, Tensor(2), out=Tensor(18))
    assert events == ['wait','copy','launch','copy','record']


def test_padded_gather_orders_staging_and_records_after_output_copy():
    tree = ast.parse(SOURCE.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name=='all_gather')
    calls = [n for n in ast.walk(method) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    # Select the padded path after the aligned path's early return.
    stage = next(n for n in calls if n.func.attr=='copy_' and isinstance(n.func.value, ast.Subscript))
    waits = [n.lineno for n in calls if n.func.attr=='_order_stream']
    records = [n.lineno for n in calls if n.func.attr=='_mark_stream']
    output = max(n.lineno for n in calls if n.func.attr=='copy_')
    assert max(waits) < stage.lineno
    assert max(records) > output


def test_padded_gather_result_survives_the_next_gather(monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda, 'device', lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    namespace = methods({'all_gather'}, {
        'torch': torch, 'Optional': __import__('typing').Optional,
        '_nullcontext': nullcontext, 'PACK_BYTES': 16,
        '_align_up': lambda value, alignment: (value + alignment - 1) // alignment * alignment,
    })
    staged = torch.empty(32, dtype=torch.uint8)
    gathered = torch.empty(64, dtype=torch.uint8)

    def gather(input_address, output_address, nbytes, row_packs):
        assert (input_address, output_address, nbytes, row_packs) == (
            staged.data_ptr(), gathered.data_ptr(), 32, 2)
        gathered.view(2, 32).copy_(staged.expand(2, 32))

    runtime = SimpleNamespace(
        _lock=nullcontext(), device='cpu', world_size=2, check_health=lambda: None,
        should_all_gather=lambda inp, dim: True, _normalize_dim=lambda inp, dim: 0,
        _direct_gather_layout=lambda inp, dim: False,
        _gather_scratch=lambda padded: (staged, gathered), _launch_gather=gather,
        _order_stream=lambda capturing: None, _mark_stream=lambda capturing: None,
    )
    first_input = torch.arange(9, dtype=torch.float32)[1:]
    second_input = (torch.arange(9, dtype=torch.float32) + 100)[1:]
    assert first_input.data_ptr() % 16 and second_input.data_ptr() % 16
    first = namespace['all_gather'](runtime, first_input, dim=0)
    expected = torch.cat([first_input, first_input])
    assert torch.equal(first, expected)
    second = namespace['all_gather'](runtime, second_input, dim=0)
    assert torch.equal(second, torch.cat([second_input, second_input]))
    assert torch.equal(first, expected), 'a later gather overwrote the returned tensor'


@pytest.mark.parametrize('field,value', [('max_size', 1 << 31), ('max_gather_bytes', 1 << 31), ('max_gather_bytes', -1)])
def test_message_capacity_is_rejected_before_device_setup(field, value):
    def device_setup(_):
        raise AssertionError('device setup ran before capacity validation')
    namespace = methods({'__init__'}, {
        'torch': SimpleNamespace(device=object), 'ProcessGroup': object,
        'Sequence': __import__('typing').Sequence,
        'Optional': __import__('typing').Optional, 'DEFAULT_MAX_SIZE': 2 << 20,
        'DEFAULT_MAX_GATHER_BYTES': 16 << 20, 'DEFAULT_THREADS': 512,
        'DEFAULT_BLOCKS': 8, 'PACK_BYTES': 16, '_MAX_MESSAGE_BYTES': None,
        '_normalize_device': device_setup,
    })
    with pytest.raises(ValueError, match=field):
        namespace['__init__'](SimpleNamespace(), exchange_group=None, device='cuda:0', **{field: value})


def test_eligibility_rejects_unrepresentable_kernel_byte_counts():
    class Tensor:
        dtype = 'bf16'
        device = 0
        is_cuda = True
        is_sparse = False
        def numel(self): return 1 << 30
        def element_size(self): return 2
        def is_contiguous(self): return True
        def is_complex(self): return False
        def dim(self): return 1
    namespace = methods({'should_allreduce', 'should_all_gather'}, {
        'torch': SimpleNamespace(Tensor=Tensor, bool='bool'),
        'SUPPORTED_DTYPES': ('bf16',), 'PACK_BYTES': 16, '_MAX_MESSAGE_BYTES': None,
    })
    runtime = SimpleNamespace(_closed=False, _proxy=object(), device=0,
                              max_size=1 << 33, max_gather_bytes=1 << 33,
                              _normalize_dim=lambda inp, dim: 0)
    assert namespace['should_allreduce'](runtime, Tensor()) is False
    assert namespace['should_all_gather'](runtime, Tensor()) is False


@pytest.mark.parametrize('method', ['stats', 'benchmark_counters', 'check_health', 'poisoned'])
def test_diagnostic_read_cannot_overlap_proxy_destruction(method):
    entered, release, destroyed = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def read(value):
        entered.set()
        assert release.wait(3)
        assert not destroyed.is_set(), 'proxy was destroyed during a diagnostic read'
        return value

    class Proxy:
        def stats(self): return read({})
        def path_counters(self): return read([])
        def failed(self): return read(False)
        def peer_hca(self, peer): return (0, 1)
        def close(self): destroyed.set()

    namespace = methods({method, 'close'}, {
        'torch': SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device: None)),
        'Any': __import__('typing').Any, 'contextlib': contextlib,
    })
    scalar = SimpleNamespace(item=lambda: 0)
    runtime = SimpleNamespace(_lock=threading.RLock(), _closed=False, _proxy=Proxy(),
                              device=0, world_size=2, rank=0, hca_names=('a', 'b'),
                              max_size=16, max_gather_bytes=16, _slot_bytes=4096,
                              _counters=[scalar], _error_word=scalar, _ctrl_words=[scalar]*7,
                              _ctrl_np=[0]*7, spin_limit=1, _opposite_paths=2)
    operation = namespace[method]
    if isinstance(operation, property):
        operation = operation.fget

    def snapshot():
        try:
            operation(runtime)
        except BaseException as error:
            errors.append(error)

    reader = threading.Thread(target=snapshot)
    closer = threading.Thread(target=lambda: namespace['close'](runtime))
    reader.start()
    assert entered.wait(3)
    closer.start()
    try:
        assert not destroyed.wait(0.1), 'close overtook an active diagnostic read'
    finally:
        release.set()
        reader.join(3)
        closer.join(3)
    assert not reader.is_alive() and not closer.is_alive()
    assert not errors
    assert destroyed.is_set()


def test_lifecycle_lock_supports_nested_health_checks():
    tree = ast.parse(SOURCE.read_text())
    constructor = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    assignment = next(node for node in constructor.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Attribute) and target.attr == '_lock' for target in node.targets))
    lock = eval(compile(ast.Expression(assignment.value), str(SOURCE), 'eval'), {'threading': threading})
    namespace = methods({'check_health'}, {})
    runtime = SimpleNamespace(_lock=lock, _proxy=None, _ctrl_np=[0]*7)
    with lock:
        assert lock.acquire(blocking=False), 'health checks require a reentrant lifecycle lock'
        try:
            namespace['check_health'](runtime)
        finally:
            lock.release()


@pytest.mark.parametrize('name', ['B12X_ROCE_HCA', 'NCCL_IB_HCA'])
def test_hca_exclusions_are_not_converted_to_inclusions(monkeypatch, name):
    namespace = methods({'_env_list'}, {'os': __import__('os')})
    monkeypatch.setenv(name, '^rocep1s0f0')
    with pytest.raises(ValueError, match='exclusion'):
        namespace['_env_list'](name)
    monkeypatch.setenv(name, 'rocep1s0f0:2')
    with pytest.raises(ValueError, match='port 1'):
        namespace['_env_list'](name)
    monkeypatch.setenv(name, '=rocep1s0f0:1,roceP2p1s0f0:1')
    assert namespace['_env_list'](name) == ('rocep1s0f0', 'roceP2p1s0f0')
