"""Check GDN auxiliary construction against the selected B12X source.

The shared recurrence constructor also serves KDA checkpoint exports. Execute
its real initializer and GDN's subclass/factory without importing CUDA DSLs;
reaching the native compiler is the CPU admission boundary, not GPU evidence.
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def source_root():
    return Path(os.environ["SPARKRING_B12X_SOURCE_ROOT"]).resolve() / "b12x/sequence"


def constructor_namespace():
    root = source_root()
    nodes = []
    selections = (
        (root / "_shared/delta_prefill/_cute_kernels.py", {"_RecurrenceKernel"}),
        (
            root / "gdn_prefill/_parallel_kernels.py",
            {
                "_BoundaryKernel",
                "_PartitionKernel",
                "_PackTransferKernel",
                "_CommitKernel",
            },
        ),
    )
    for path, names in selections:
        selected = set()
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ClassDef) and node.name in names:
                node.body = [
                    method
                    for method in node.body
                    if isinstance(method, ast.FunctionDef) and method.name == "__init__"
                ]
                assert len(node.body) == 1, "Kernel must define its constructor"
                nodes.append(node)
                selected.add(node.name)
        assert selected == names, "Selected source is missing required kernel classes"
    namespace = dict(_HEAD_DIM=128, REC=NS(K_TILDE=0), Int32=int)
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *nodes,
            ],
            type_ignores=[],
        )
    )
    exec(compile(module, "selected-gdn-constructors", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("heads", [3, 12, 24])
def test_gdn_boundary_declares_one_disabled_checkpoint(heads):
    kernel = constructor_namespace()["_BoundaryKernel"](
        heads=heads, max_seqs=2, max_segments=6, v_split=64, null_state_index=0
    )
    assert kernel.max_checkpoints == 1
    assert kernel.checkpoint_export is False
    assert kernel.max_segments == 6


@pytest.mark.parametrize("capacity", [1, 2, 4])
def test_kda_retains_explicit_checkpoint_capacity(capacity):
    kernel = constructor_namespace()["_RecurrenceKernel"](
        heads=16,
        tiles_capacity=8,
        window_tiles=8,
        rows=2,
        v_split=64,
        k_split=1,
        stages=2,
        checkpoint_export=True,
        max_checkpoints=capacity,
        null_state_index=0,
        index_type=int,
    )
    assert kernel.max_checkpoints == capacity
    assert kernel.checkpoint_export is True


def test_gdn_auxiliary_factory_reaches_native_compilation():
    namespace = constructor_namespace()
    path = source_root() / "gdn_prefill/_parallel_kernels.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    factory = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "compile_auxiliary"
    )

    class NativeCompilationReached(Exception):
        pass

    def compile_native(*args, **kwargs):
        raise NativeCompilationReached

    namespace.update(
        _CACHE={},
        _metadata=lambda *args, **kwargs: ((), ()),
        _fake_pointer=lambda dtype: dtype,
        Float32=float,
        BFloat16=float,
        Int64=int,
        raise_if_kernel_resolution_frozen=lambda *args, **kwargs: None,
        b12x_compile=compile_native,
        current_cuda_stream=lambda: None,
        KernelCompileSpec=NS(from_key=lambda *args: args),
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                factory,
            ],
            type_ignores=[],
        )
    )
    exec(compile(module, str(path), "exec"), namespace)
    binding = NS(
        _state=NS(caps=NS(heads=3, max_seqs=2, null_state_index=0), v_split=64),
        output=NS(device=NS(index=0)),
        initial_state_indices=NS(dtype="int32"),
        parallel=NS(_state=NS(max_segments=6, segment_tokens=256, reuse_outputs=True)),
    )
    with pytest.raises(NativeCompilationReached):
        namespace["compile_auxiliary"](binding)
