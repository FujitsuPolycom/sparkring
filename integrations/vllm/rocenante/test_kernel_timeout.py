"""Execute kernel timeout control flow with CPU memory and modular uint32 values.

The simulator runs one waiting thread in a single block with no payload packs.
It checks protocol state, not CUDA compilation, memory ordering, or RDMA traffic.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = ROOT / "third_party/b12x_roce/b12x/comm/roce"
MASK = 0xFFFFFFFF
CTRL = 0x1000
EPOCH = 0x2000


class Uint32(int):
    def __new__(cls, value):
        return super().__new__(cls, int(value) & MASK)

    def __add__(self, other):
        return Uint32(int(self) + int(other))


def simulate_kernel(filename, paths, sequence, timeout):
    source = SOURCE_DIR / filename
    tree = ast.parse(source.read_text())
    kernels = [node for cls in tree.body if isinstance(cls, ast.ClassDef)
               for node in cls.body
               if isinstance(node, ast.FunctionDef) and node.name == "kernel"]
    assert len(kernels) == 1
    kernel = kernels[0]
    kernel.decorator_list = []
    # Annotations belong to the CuTe compiler; preserve the entire executable body.
    for argument in kernel.args.args:
        argument.annotation = None
    kernel.returns = None
    memory = {EPOCH: Uint32(sequence - 1)}
    waits = []

    def load(address):
        return Uint32(memory.get(address, 0))

    def store(address, value):
        memory[address] = Uint32(value)

    def atomic_add(address, value):
        previous = load(address)
        store(address, previous + value)
        return previous

    def spin(address, expected, limit):
        waits.append((address, int(expected), int(limit)))
        return Uint32(timeout)

    namespace = {
        "Uint32": Uint32, "Int32": int, "Int64": int,
        "PACK_BYTES": 16, "PATH_COUNT": 2,
        "cute": SimpleNamespace(arch=SimpleNamespace(
            thread_idx=lambda: (0, 0, 0), block_idx=lambda: (0, 0, 0),
            grid_dim=lambda: (1, 1, 1), sync_threads=lambda: None)),
        "cutlass": SimpleNamespace(const_expr=bool, range_constexpr=range),
        "ld_relaxed_gpu_u32": load, "ld_relaxed_sys_u32": load,
        "st_relaxed_sys_u32": store, "st_release_sys_u32": store,
        "st_release_gpu_u32": store, "atomic_add_relaxed_gpu_u32": atomic_add,
        "fence_sc_sys": lambda: None, "fence_sc_gpu": lambda: None,
        "spin_until_eq_acquire_sys": spin,
    }
    exec(compile(ast.Module(body=[kernel], type_ignores=[]), str(source), "exec"), namespace)
    runtime = SimpleNamespace(_threads=32, _world_size=4, _rank=1,
                              _slots=2, _opposite_paths=paths, _flag_stride=128)
    pointer = SimpleNamespace(toint=lambda: 0x3000)
    arguments = dict(self=runtime, input_ptr=pointer, output_ptr=pointer,
                     nbytes=0, recv_base=0x4000, flag_base=0x8000,
                     send_base=0xC000, ctrl_base=CTRL, slot_bytes=16,
                     epoch_ptr=EPOCH, spin_limit=Uint32(1))
    if filename == "_allgather_cute.py":
        arguments.update(shard_packs=0, row_packs=1)
    else:
        arguments.update(size_packs=0)
    namespace["kernel"](**arguments)
    assert waits, "The simulated thread must execute a peer wait"
    assert all(expected == sequence for _, expected, _ in waits)
    return memory


@pytest.mark.parametrize("filename", ["_allgather_cute.py", "_oneshot_cute.py"])
@pytest.mark.parametrize("paths", [2, 4])
@pytest.mark.parametrize("sequence", [1, 0x80000000, 0xFFFFFFFF, 0])
def test_timeout_remains_visible_at_uint32_boundaries(filename, paths, sequence):
    memory = simulate_kernel(filename, paths, sequence, timeout=True)
    assert memory.get(CTRL + 24, 0) == 1
    assert memory.get(EPOCH + 12, 0) == 1
    assert memory[CTRL + 8] == sequence
    assert memory[EPOCH] == (sequence - 1) & MASK


@pytest.mark.parametrize("filename", ["_allgather_cute.py", "_oneshot_cute.py"])
@pytest.mark.parametrize("paths", [2, 4])
def test_successful_sequence_wrap_advances_epoch(filename, paths):
    memory = simulate_kernel(filename, paths, 0, timeout=False)
    assert memory[CTRL] == 0
    assert memory[EPOCH] == 0
    assert memory.get(CTRL + 24, 0) == 0
    assert memory.get(EPOCH + 12, 0) == 0
