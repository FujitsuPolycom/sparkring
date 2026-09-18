"""CPU checks of the explicit prepared B12X source; no GPU qualification."""

import ast
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import sys
import types

ROOT = Path(os.environ["SPARKRING_B12X_SOURCE_ROOT"]) / "b12x"
BASE = Path(os.environ["SPARKRING_ORACLE_BASELINE"]) / "b12x"


def load(path, names, env=None):
    path = ROOT / path
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    namespace = {} if env is None else env
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
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("count", [1, 2, 4])
def test_checkpoint_capacity_survives_query_and_materialization(count):
    env = load(
        "sequence/kda_prefill/_tuning.py",
        {"KdaPrefillQuery"},
        dict(dataclass=dataclass, field=field, capture_exhaustive_search=lambda: False),
    )
    querytype = env["KdaPrefillQuery"]
    env = load(
        "sequence/kda_prefill/_impl.py",
        {"Caps", "_query"},
        dict(
            dataclass=dataclass,
            torch=torch,
            KDA_HEAD_DIM=128,
            CHUNK_TOKENS=16,
            canonical_device=torch.device,
            positive=lambda n, v: int(v),
            KdaPrefillQuery=querytype,
        ),
    )
    caps = env["Caps"](
        device="cuda:0",
        max_tokens=8192,
        max_seqs=16,
        max_state_slots=100000,
        heads=4,
        checkpoint_export=True,
        max_checkpoints=count,
    )
    query = env["_query"](caps, {})
    assert query.to_dict()["max_checkpoints"] == count
    assert query.to_dict()["metadata_validation"] == "transactional"
    prepare = load(
        "sequence/_shared/delta_prefill/preparation.py",
        {"_caps_from_query"},
        dict(torch=torch),
    )
    restored = prepare["_caps_from_query"](NS(Caps=env["Caps"]), query, 0, is_gdn=False)
    assert restored == caps


@pytest.mark.parametrize(
    "settings",
    [
        dict(max_checkpoints=3),
        dict(max_checkpoints=True),
        dict(max_checkpoints=2, checkpoint_export=False),
        dict(max_checkpoints=4, metadata_validation="trusted"),
        dict(metadata_validation="none"),
    ],
)
def test_invalid_checkpoint_capacity_refused(settings):
    env = load(
        "sequence/kda_prefill/_impl.py",
        {"Caps"},
        dict(
            dataclass=dataclass,
            torch=torch,
            KDA_HEAD_DIM=128,
            CHUNK_TOKENS=16,
            canonical_device=torch.device,
            positive=lambda n, v: int(v),
        ),
    )
    args = dict(
        device="cuda:0",
        max_tokens=8192,
        max_seqs=16,
        max_state_slots=100000,
        heads=4,
        checkpoint_export=True,
    )
    args.update(settings)
    with pytest.raises(ValueError):
        env["Caps"](**args)


def metadata_case(count=4):
    return dict(
        cu_seqlens=[0, 64],
        initial_state_indices=[1],
        final_state_indices=[2],
        checkpoint_state_indices=[[3, 4, 5, 6]][:1] if count > 1 else [3],
        checkpoint_offsets=[[16, 32, 48, 64]][:1] if count > 1 else [16],
        num_seqs=1,
        num_tokens=64,
        token_capacity=64,
        seq_capacity=1,
        state_slots=10,
        max_checkpoints=count,
    )


@pytest.mark.parametrize(
    "case", ["valid", "duplicate", "alias", "offset", "range", "counts"]
)
def test_multicheckpoint_metadata_oracle(case):
    validate = load("sequence/kda_prefill/metadata.py", {"validate_metadata"})[
        "validate_metadata"
    ]
    args = metadata_case()
    if case == "duplicate":
        args["checkpoint_state_indices"][0][1] = 3
    if case == "alias":
        args["checkpoint_state_indices"][0][1] = 1
    if case == "offset":
        args["checkpoint_offsets"][0][1] = 17
    if case == "range":
        args["checkpoint_state_indices"][0][1] = 100000
    if case == "counts":
        args["num_tokens"] = 65
    if case == "valid":
        assert validate(**args) == [(0, 64)]
    else:
        with pytest.raises((ValueError, IndexError)):
            validate(**args)


def test_delta_launch_consumes_precompiled_programs_and_resets_no_resources():
    calls = []

    class Stream:
        def wait_event(self, event):
            calls.append(("wait", event))

    class Event:
        def record(self, stream):
            calls.append(("record", stream))

    main = Stream()
    side = Stream()
    env = dict(
        torch=NS(
            cuda=NS(
                device=lambda _: nullcontext(),
                current_stream=lambda _: main,
                stream=lambda _: nullcontext(),
            )
        ),
        _LOG2E=1.442695,
    )
    run = load("sequence/_shared/delta_prefill/_cute_kernels.py", {"run_prefill"}, env)[
        "run_prefill"
    ]

    def program(name):
        return lambda *args: calls.append((name, args))

    binding = NS(
        output=NS(device="cuda:0"),
        _state=NS(max_windows=8, window_tiles=64, caps=NS(tiles_capacity=480)),
    )
    resources = NS(stream=side, events=[Event() for _ in range(17)])
    for windows in [1, 3]:
        run(
            binding,
            programs=tuple(program(n) for n in ["prologue", "prepare", "recurrence"]),
            resources=resources,
            lower_bound=-5,
            scale=1,
            eps=1e-6,
            windows=windows,
        )
    assert [c[1][1] for c in calls if c[0] == "prologue"] == [64, 192]
    assert len([c for c in calls if c[0] == "prepare"]) == 4
    assert len([c for c in calls if c[0] == "recurrence"]) == 4


def test_qsa_retains_every_baseline_support_kernel_body():
    path = Path("attention/qsa/_kernels.py")
    old = ast.parse((BASE / path).read_text())
    new = ast.parse((ROOT / path).read_text())
    actual = {n.name: n for n in new.body if isinstance(n, ast.FunctionDef)}
    kernels = [
        n
        for n in old.body
        if isinstance(n, ast.FunctionDef) and n.name.endswith("_kernel")
    ]
    assert len(kernels) >= 20
    for n in kernels:
        assert ast.dump(n) == ast.dump(actual[n.name]), n.name


def test_qsa_transaction_retains_all_validation_before_state_updates():
    tree = ast.parse((ROOT / "attention/qsa/_contract.py").read_text())
    func = next(n for n in tree.body if getattr(n, "name", None) == "_qsa_decode_impl")
    calls = sorted(
        (n.lineno, n.func.id)
        for n in ast.walk(func)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id.startswith("launch_")
    )
    names = [name for _, name in calls]
    assert names[:5] == [
        "launch_validate_rows",
        "launch_validate_page_tables",
        "launch_validate_shared_pool_ownership",
        "launch_validate_completed_groups",
        "launch_propagate_request_errors",
    ]
    assert names.index("launch_commit_raw_ring") > names.index(
        "launch_validate_completed_groups"
    )
    assert names[-1] == "launch_poison_failed_rows"
    for n in ast.walk(func):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id.startswith("launch_")
        ):
            assert any(k.arg == "_prepared" for k in n.keywords), n.func.id


def test_qsa_paired_score_and_draft_bounds_still_match_baseline():
    for name in ["_RepresentativeScoreKernel", "_or_error"]:
        path = Path("attention/qsa/_score_cute.py")

        def get(root):
            return next(
                n
                for n in ast.parse((root / path).read_text()).body
                if getattr(n, "name", None) == name
            )

        assert ast.dump(get(ROOT)) == ast.dump(get(BASE))
    assert (ROOT / "attention/qsa/_score_pair_cute.py").read_text() == (
        BASE / "attention/qsa/_score_pair_cute.py"
    ).read_text()
    for name in ["_record_kernel", "_prepare_kernel"]:
        path = Path("attention/qsa/_draft_selection.py")

        def get(root):
            return next(
                n
                for n in ast.parse((root / path).read_text()).body
                if getattr(n, "name", None) == name
            )

        assert ast.dump(get(ROOT)) == ast.dump(get(BASE))


@pytest.mark.parametrize("rows", [1, 127, 128, 129, 8192])
def test_qsa_replay_consumes_prepared_score_with_runtime_rows_and_int64_pages(rows):
    calls = []

    class Tensor:
        def __init__(self, shape, dtype="bf16"):
            self.shape = shape
            self.dtype = dtype

        def stride(self, dim):
            return [1 << 35, 128][dim]

    query = Tensor((rows, 4, 128))
    cache = Tensor((1 << 33, 128, 128))
    table = Tensor((16, 1000), "i32")
    scores = Tensor((rows, 1000), "f32")
    fake_torch = NS(bfloat16="bf16", float32="f32", int32="i32", int64="i64")

    def forbidden(**kwargs):
        raise AssertionError("runtime compiler invoked")

    env = dict(
        torch=fake_torch,
        BFloat16="bf16",
        Float32="f32",
        Int32=int,
        Int64=int,
        _pointer=lambda t, dt: t,
        compile_score_representatives=forbidden,
        compile_only_launches_enabled=lambda: False,
        current_cuda_stream=lambda: "stream",
    )
    launch = load(
        "attention/qsa/_score_cute.py", {"launch_score_representatives"}, env
    )["launch_score_representatives"]
    t = Tensor((rows,), "i32")
    launch(
        prepared_query=query,
        query_positions=t,
        request_ids=t,
        sequence_lengths=t,
        compressed_cache=cache,
        compressed_block_table=table,
        state_errors=t,
        scores=scores,
        eligible_counts=t,
        merge_lengths=t,
        group_offset=0,
        group_count=1000,
        caps=object(),
        _prepared=lambda *a: calls.append(a),
    )
    assert calls[0][2:6] == (rows, 1 << 33, 0, 1000)
    assert calls[0][1][0] == 1 << 35


def test_all_reconciled_modules_parse():
    for directory in [
        "sequence/kda_prefill",
        "sequence/_shared/delta_prefill",
        "attention/qsa",
    ]:
        for path in (ROOT / directory).glob("*.py"):
            ast.parse(path.read_text())


def test_native_delta_compiler_and_replay_argument_counts_match_kernel_abi():
    tree = ast.parse(
        (ROOT / "sequence/_shared/delta_prefill/_cute_kernels.py").read_text()
    )
    defs = {n.name: n for n in tree.body if hasattr(n, "name")}
    for stage in ["prologue", "prepare", "recurrence"]:
        function = defs["_compile_" + stage]
        kernel = defs["_" + stage.title() + "Kernel"]
        signature = next(
            n for n in kernel.body if getattr(n, "name", None) == "__call__"
        )
        expected = len(signature.args.args) - 1
        compile_call = next(
            n
            for n in ast.walk(function)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "b12x_compile"
        )
        raw_call = next(
            n
            for n in ast.walk(function)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "raw"
        )
        assert len(compile_call.args) - 1 == expected, stage
        assert len(raw_call.args) == expected, stage


@pytest.mark.parametrize(
    "heads,dim,paired", [(4, 128, True), (8, 128, False), (4, 120, False)]
)
def test_score_prepares_once_for_static_geometry_not_live_row_count(
    monkeypatch, heads, dim, paired
):
    calls = []

    class Number:
        width = 16

        def __new__(cls, n):
            return n

    class Scalar:
        def __init__(self, *geometry):
            self.geometry = geometry

    class Paired(Scalar):
        pass

    module = types.ModuleType("test_qsa._score_pair_cute")
    module.PairedRepresentativeScoreKernel = Paired
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def tensor(dtype):
        return NS(dtype=dtype, device=NS(index=0))

    env = dict(
        __package__="test_qsa",
        torch=NS(
            bfloat16="bf16",
            float32="f32",
            int32="i32",
            int64="i64",
            cuda=NS(device=lambda _: nullcontext()),
        ),
        BFloat16=Number,
        Float32=Number,
        Int32=Number,
        Int64=Number,
        _CACHE={},
        _RepresentativeScoreKernel=Scalar,
        raise_if_kernel_resolution_frozen=lambda *a, **k: None,
        make_ptr=lambda *a, **k: "pointer",
        cute=NS(AddressSpace=NS(gmem=1)),
        current_cuda_stream=lambda: 1,
        KernelCompileSpec=NS(from_key=lambda *args: args),
        b12x_compile=lambda *a, **k: calls.append((a, k)) or object(),
    )
    compile_score = load(
        "attention/qsa/_score_cute.py", {"compile_score_representatives"}, env
    )["compile_score_representatives"]
    args = dict(
        prepared_query=tensor("bf16"),
        query_positions=tensor("i64"),
        request_ids=tensor("i32"),
        sequence_lengths=tensor("i32"),
        compressed_cache=tensor("bf16"),
        compressed_block_table=tensor("i32"),
        state_errors=tensor("i32"),
        scores=tensor("f32"),
        eligible_counts=tensor("i32"),
        merge_lengths=tensor("i32"),
        caps=NS(
            index_heads=heads,
            index_head_dim=dim,
            compress_ratio=4,
            compressed_page_size=128,
            max_groups=65536,
            group_budget=512,
        ),
    )
    first = compile_score(**args)
    args["prepared_query"].shape = (8192, heads, dim)
    assert compile_score(**args) is first
    assert len(calls) == 1
    assert isinstance(calls[0][0][0], Paired) == paired
    assert len(calls[0][0][1]) == 10  # Error state is a retained tensor argument.


def test_guarded_draft_compile_abi_and_replay_abi_match():
    kernels = ast.parse((ROOT / "attention/qsa/_draft_selection.py").read_text())
    names = {n.name: n for n in kernels.body if isinstance(n, ast.FunctionDef)}
    contract = ast.parse((ROOT / "attention/qsa/_contract.py").read_text())
    compiler = next(
        n for n in contract.body if getattr(n, "name", None) == "compile_qsa"
    )
    for kernel_name, host in [
        ("_record_kernel", "record_anchors"),
        ("_prepare_kernel", "prepare_selection"),
    ]:
        kernel = names[kernel_name]
        runtime_count = sum(
            not (
                isinstance(a.annotation, ast.Attribute)
                and a.annotation.attr == "constexpr"
            )
            for a in kernel.args.args
        )
        call = next(
            n
            for n in ast.walk(compiler)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "launch_triton"
            and isinstance(n.args[0], ast.Name)
            and n.args[0].id == kernel_name
        )
        assert len(call.args) - 2 == runtime_count, kernel_name
        launch = next(
            n
            for n in ast.walk(names[host])
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Subscript)
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "program"
        )
        assert len(launch.args) == runtime_count, host
