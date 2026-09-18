"""Explicit-source CPU launch-contract checks for the PLE checkpoint port.

Executes production dispatch functions with test-only CUDA/program objects.
This validates preparation/dispatch ownership, not device checkpoint values.
"""

import ast
from contextlib import nullcontext
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys
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


def test_checkpoint_discovery_is_metadata_only_with_deferred_triton(monkeypatch):
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

    module_path = path.parents[2] / "_lib/compile_plan.py"
    spec = importlib.util.spec_from_file_location("checkpoint_compile_plan_test", module_path)
    planner = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, planner)
    spec.loader.exec_module(planner)
    program = planner.DeferredTritonKernel(
        planner.ProgramKey("triton", "checkpoint-test", "_export_checkpoint_kernel"),
        source=NS(name="_export_checkpoint_kernel"),
    )

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
    token = planner._PLANNING.set(set())
    try:
        assert env["compile_checkpoint"](query, 0) is program
        assert planner.program_keys(program) == program.__b12x_programs__
        with pytest.raises(RuntimeError, match="compile planning attempted Triton"):
            program[(1, 1, 1)]
    finally:
        planner._PLANNING.reset(token)
    assert len(calls) == 1
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


def test_ple_materialization_loads_every_program_before_exposing_state():
    path = (Path(os.environ["SPARKRING_B12X_SOURCE_ROOT"])
            / "b12x/sequence/ple/_preparation.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    materialize = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.FunctionDef) and node.name == "materialize")
    programs = tuple(object() for _ in range(6))
    calls = []

    def compile_layer(*args):
        calls.append(("compile", args))
        return programs

    def load_programs(value):
        assert value is programs
        calls.append(("load", value))
        return value

    def state(query, layout, value):
        assert calls[-1] == ("load", programs)
        assert value is programs
        return NS(programs=value)

    env = dict(query="query", query_payload="payload", caps="caps",
               _materialize_layout=lambda *args: "layout", compile_layer=compile_layer,
               load_programs=load_programs, _PleState=state)
    exec(compile(ast.Module(body=[materialize], type_ignores=[]), str(path), "exec"), env)
    result = env["materialize"](NS(config=NS(to_dict=lambda: {})), NS(ordinal=0))
    assert result.programs is programs
    assert [call[0] for call in calls] == ["compile", "load"]


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
