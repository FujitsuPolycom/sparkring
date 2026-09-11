"""Exercise stream admission and shared staging order without CUDA hardware."""
import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'third_party/b12x_roce/b12x/comm/roce/roce_oneshot.py'


def methods(names, namespace):
    tree = ast.parse(SOURCE.read_text())
    nodes = [node for cls in tree.body if isinstance(cls, ast.ClassDef)
             for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


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
