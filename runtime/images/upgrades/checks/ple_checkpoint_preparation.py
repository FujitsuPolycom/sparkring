"""Explicit-source CPU launch-contract checks for the PLE checkpoint port.

Executes production dispatch functions with test-only CUDA/program objects.
This validates preparation/dispatch ownership, not device checkpoint values.
"""

import ast
from contextlib import nullcontext
from dataclasses import dataclass
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def load_dispatch():
    root = Path(os.environ["SPARKRING_B12X_SOURCE_ROOT"])
    path = root / "b12x/sequence/ple/_checkpoint.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_export_checkpoint_op"
    )
    node.decorator_list = []
    calls = []

    class Program:
        def __getitem__(self, grid):
            def launch(*args):
                calls.append((grid, args))

            return launch

    state = NS(
        mixed=True,
        programs=(None,) * 5 + (Program(),),
        query=NS(
            max_seqs=8, state_strides=(72, 6, 1), max_tokens=4096, max_state_slots=1000
        ),
        channels=12,
        state_length=3,
        state_capacity=6,
    )
    resolutions = []

    def prepared(plan, component, device):
        resolutions.append((plan, component, device))
        return state

    env = {
        "torch": NS(Tensor=object, cuda=NS(device=lambda _: nullcontext())),
        "triton": NS(cdiv=lambda a, b: (a + b - 1) // b),
        "plan_from_handle": lambda handle: ("plan", handle),
        "require_prepared": prepared,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    return env["_export_checkpoint_op"], state, calls, resolutions


def test_dispatch_uses_prepared_program_with_live_device_metadata():
    function, state, calls, resolutions = load_dispatch()
    pool = NS(device="cuda:0", stride=lambda: state.query.state_strides)
    first = [object() for _ in range(7)]
    second = [*first[:3], object(), object(), *first[5:]]
    function(*first, pool, 123)
    function(*second, pool, 123)
    assert resolutions == [(("plan", 123), "sequence.ple", "cuda:0")] * 2
    assert len(calls) == 2 and calls[0][0] == calls[1][0] == (8, 1, 1)
    assert calls[0][1][:7] == tuple(first)
    assert calls[1][1][:7] == tuple(second)
    assert calls[0][1][7:] == (pool, 12, 3, 6, 72, 4096, 1000, 256)


@pytest.mark.parametrize("defect", ["not_mixed", "missing_program", "stride_drift"])
def test_dispatch_refuses_unprepared_or_incompatible_storage(defect):
    function, state, calls, _ = load_dispatch()
    if defect == "not_mixed":
        state.mixed = False
    if defect == "missing_program":
        state.programs = state.programs[:5]
    stride = (99, 6, 1) if defect == "stride_drift" else state.query.state_strides
    pool = NS(device="cuda:0", stride=lambda: stride)
    with pytest.raises(ValueError):
        function(*[object() for _ in range(7)], pool, 123)
    assert calls == []


def test_checkpoint_compiles_only_static_geometry_and_preloads_launch_handle():
    path = (
        Path(os.environ["SPARKRING_B12X_SOURCE_ROOT"])
        / "b12x/sequence/ple/_checkpoint.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
        and n.name in {"_CompilePointer", "compile_checkpoint"}
    ]
    calls = []

    class Program:
        def __getitem__(self, grid):
            calls.append(("preload", grid))

    program = Program()

    def warmup(*pointers, **options):
        calls.append(("compile", pointers, options))
        return program

    env = dict(
        dataclass=dataclass,
        torch=NS(
            dtype=object,
            bfloat16="bf16",
            int32="i32",
            int64="i64",
            bool="bool",
            cuda=NS(device=lambda _: nullcontext()),
        ),
        triton=NS(cdiv=lambda a, b: (a + b - 1) // b),
        _export_checkpoint_kernel=NS(warmup=warmup),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    query = NS(
        streams=4,
        hidden_size=16,
        dilation=1,
        kernel_size=4,
        max_speculative_tokens=3,
        state_strides=(384, 6, 1),
        max_tokens=8192,
        max_seqs=16,
        max_state_slots=100000,
    )
    assert env["compile_checkpoint"](query, 0) is program
    assert calls[-1] == ("preload", (1, 1, 1))
    _, pointers, options = calls[0]
    assert [p.alignment for p in pointers] == [2, 2, 4, 4, 8, 1, 4, 2]
    assert options == dict(
        CHANNELS=64,
        STATE_LENGTH=3,
        STATE_CAPACITY=6,
        STATE_STRIDE=384,
        MAX_TOKENS=8192,
        MAX_SLOTS=100000,
        BLOCK=256,
        num_warps=4,
        grid=(16, 2),
    )


def test_checkpoint_program_semantics_have_a_distinct_preparation_identity():
    root = Path(os.environ["SPARKRING_B12X_SOURCE_ROOT"]) / "b12x"
    tuning = ast.parse((root / "sequence/ple/_tuning.py").read_text())
    assignment = next(
        node
        for node in tuning.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "TUNING"
            for target in node.targets
        )
    )
    version = next(
        item.value.value
        for item in assignment.value.keywords
        if item.arg == "semantic_version" and isinstance(item.value, ast.Constant)
    )
    assert version == 2
    session = ast.parse((root / "preparation/session.py").read_text())
    declaration = next(
        node
        for node in session.body
        if isinstance(node, ast.FunctionDef) and node.name == "_declaration_key"
    )
    namespace = {
        "_CompositePlan": type("Composite", (), {}),
        "FrozenMapping": lambda value: tuple(sorted(value.items())),
    }
    exec(
        compile(
            ast.Module(body=[declaration], type_ignores=[]),
            "source preparation identity",
            "exec",
        ),
        namespace,
    )
    contract = NS(
        component_id="sequence.ple",
        query_schema_version=4,
        config_schema_version=1,
        semantic_version=version,
        candidate_contract_version=1,
        encode_query=lambda query: {"mode": query},
    )
    plan = NS(
        contract=contract,
        query="mixed",
        invocation="call",
        override=None,
        _device="cpu-proof",
    )
    checkpoint_key = namespace["_declaration_key"](plan)
    contract.semantic_version = 1
    assert namespace["_declaration_key"](plan) != checkpoint_key
