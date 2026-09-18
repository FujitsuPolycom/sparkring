"""Routing behavior and source admission for prepared RoCEnante hooks."""

import __future__
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def policy(monkeypatch):
    loaded = []

    def load(mode="both", trace=False, cutoff=20480):
        monkeypatch.setenv("QWEN_DISPATCH_MODE", mode)
        monkeypatch.setenv("QWEN_DISPATCH_AR_BYTES", str(cutoff))
        monkeypatch.setenv("QWEN_DISPATCH_TRACE", str(int(trace)))
        spec = importlib.util.spec_from_file_location(
            "prepared_collective_policy",
            Path(__file__).with_name("qwen38_collective_policy.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(module)
        return module

    yield load
    for module in loaded:
        sys.meta_path[:] = [
            finder for finder in sys.meta_path if not isinstance(finder, module.Finder)
        ]


@pytest.fixture
def source():
    root = os.environ.get("SPARKRING_VLLM_SOURCE_ROOT")
    if not root:
        pytest.skip("Set SPARKRING_VLLM_SOURCE_ROOT to the pinned reconciled source")
    return Path(root)


def definition(path, cls_name, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name
    )
    cls.decorator_list = []
    cls.bases = []
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in cls.body} == names
    return cls


def adapter_module(source):
    path = source / "vllm/distributed/device_communicators/b12x_roce_all_reduce.py"
    calls = []

    def vote(rows, value, group):
        rows[:] = [value] * len(rows)

    runtime = NS(
        should_allreduce=lambda tensor: tensor.allowed,
        should_all_gather=lambda tensor, dim: tensor.allowed and dim == 0,
        all_reduce=lambda inp, **kwargs: calls.append(("reduce", kwargs)) or inp,
        all_gather=lambda inp, **kwargs: calls.append(("gather", kwargs)) or inp,
    )
    ns = dict(
        dist=NS(all_gather_object=vote),
        envs=NS(VLLM_ROCE_ALLREDUCE_MAX_SIZE=2097152),
        PreparationResourceUnavailableError=RuntimeError,
        logger=NS(info=lambda *args: None, debug=lambda *args: None),
        torch=NS(cuda=NS(is_current_stream_capturing=lambda: False)),
    )
    cls = definition(
        path,
        "B12xRoceAllReduce",
        {
            "_exchange_vote",
            "should_custom_ar",
            "should_all_gather",
            "custom_all_reduce",
            "all_gather",
            "_prepared_plan",
            "get_b12x_preparation_units",
        },
    )
    exec(
        compile(
            ast.Module(body=[cls], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        ns,
    )
    obj = ns[cls.name]()
    obj.rank, obj.world_size, obj.group = 0, 4, object()
    obj.disabled, obj._runtime, obj._plan = False, runtime, object()
    obj._announced = obj._announced_gather = False
    return NS(**ns), obj, calls


def tensor(rows, allowed=True):
    return NS(
        shape=(rows, 2560),
        dtype="bfloat16",
        allowed=allowed,
        numel=lambda: rows * 2560,
        element_size=lambda: 2,
    )


@pytest.mark.parametrize(
    "mode,reduce,gather",
    [("both", True, True), ("reduce", True, False), ("nccl", False, False)],
)
def test_actual_adapter_preserves_cutoff_runtime_guards_and_modes(
    policy, source, mode, reduce, gather
):
    hook = policy(mode)
    module, adapter, calls = adapter_module(source)
    hook.patch_adapter(module)
    assert adapter.should_custom_ar(tensor(4)) is reduce
    assert not adapter.should_custom_ar(tensor(5))
    assert not adapter.should_custom_ar(tensor(4, False))
    assert adapter.should_all_gather(tensor(4), 0) is gather
    assert not adapter.should_all_gather(tensor(4), 1)
    assert not adapter.should_all_gather(tensor(4, False), 0)
    adapter.disabled = True
    assert not adapter.should_custom_ar(tensor(4))
    assert not adapter.should_all_gather(tensor(4), 0)


def test_actual_upstream_dispatch_keeps_the_prepared_plan(policy, source):
    hook = policy()
    module, adapter, calls = adapter_module(source)
    hook.patch_adapter(module)
    value = tensor(4)
    assert adapter.custom_all_reduce(value) is value
    assert adapter.all_gather(value, 0) is value
    assert calls == [
        ("reduce", {"plan": adapter._plan}),
        ("gather", {"dim": 0, "plan": adapter._plan}),
    ]
    adapter._plan = None
    with pytest.raises(RuntimeError, match="no declared plan"):
        adapter.custom_all_reduce(value)


def test_trace_disabled_does_not_query_capture_state(policy, source):
    hook = policy(trace=False)
    module, adapter, _ = adapter_module(source)

    def forbidden():
        raise AssertionError("Trace disabled must not inspect CUDA capture state")

    module.torch.cuda.is_current_stream_capturing = forbidden
    hook.patch_adapter(module)
    assert adapter.should_custom_ar(tensor(4))
    assert adapter.should_all_gather(tensor(4), 0)


def test_policy_vote_preserves_capability_failure_and_rejects_disagreement(
    policy, source
):
    hook = policy()
    module, adapter, calls = adapter_module(source)
    hook.patch_adapter(module)
    assert adapter._exchange_vote("unsupported device", (10, 20)).startswith(
        "rank 0: unsupported device"
    )
    module.dist.all_gather_object = lambda rows, value, group: rows.__setitem__(
        slice(None), [value, ("nccl", 0), value, value]
    )
    with pytest.raises(RuntimeError, match="differs across ranks"):
        adapter._exchange_vote(None, (10, 20))


@pytest.mark.parametrize("trace", [False, True])
def test_trace_selection_keeps_real_graph_call_signature(policy, source, trace):
    hook = policy(trace=trace)
    name = "vllm.compilation.cuda_graph"
    assert (name in hook.PATCHES) is trace
    if not trace:
        return
    path = source / "vllm/compilation/cuda_graph.py"
    ns = dict(is_forward_context_available=lambda: False)
    cls = definition(path, "CUDAGraphWrapper", {"__call__"})
    exec(
        compile(
            ast.Module(body=[cls], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        ns,
    )
    module = NS(**ns)
    hook.patch_graph(module)
    graph = module.CUDAGraphWrapper()
    graph.runtime_mode = "FULL"
    graph.runnable = lambda *args, **kwargs: (args, kwargs)
    assert graph(1, named=2) == ((1,), {"named": 2})
    context = NS(batch_descriptor="batch", cudagraph_runtime_mode="NONE")
    ns["is_forward_context_available"] = lambda: True
    ns["get_forward_context"] = lambda: context
    ns["CUDAGraphMode"] = NS(NONE="NONE")
    module.is_forward_context_available = lambda: True
    module.get_forward_context = lambda: context
    assert graph(3, named=4) == ((3,), {"named": 4})


def test_source_guards_match_reconciled_source_and_reject_drift(
    policy, source, monkeypatch, tmp_path
):
    hook = policy(trace=True)
    for name, expected in hook.SOURCE_HASHES.items():
        file = source / (name.replace(".", "/") + ".py")
        assert hashlib.sha256(file.read_bytes()).hexdigest() == expected
    changed = tmp_path / "changed.py"
    changed.write_text("# unrelated source\n")
    monkeypatch.setattr(
        hook.importlib.machinery.PathFinder,
        "find_spec",
        lambda *args: NS(origin=str(changed), loader=object()),
    )
    with pytest.raises(SystemExit, match="source identity mismatch"):
        hook.Finder().find_spec(
            "vllm.distributed.device_communicators.b12x_roce_all_reduce"
        )


def test_incompatible_api_is_rejected_before_patching(policy):
    hook = policy()

    class Wrong:
        def _exchange_vote(self, reason):
            return reason

    original = Wrong._exchange_vote
    with pytest.raises(RuntimeError, match="prepared RoCE API"):
        hook.patch_adapter(NS(B12xRoceAllReduce=Wrong))
    assert Wrong._exchange_vote is original
