from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "build_parallel_state_shared_capture_overlay",
    HERE / "build_parallel_state_shared_capture_overlay.py",
)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class _FakeStream:
    next_id = 1

    def __init__(self, *, device: _FakeDevice) -> None:
        self.device = device
        self.stream_id = _FakeStream.next_id
        _FakeStream.next_id += 1


@dataclass(frozen=True)
class _FakeDevice:
    index: int | None


@dataclass
class _FakeContext:
    stream: _FakeStream
    channel_id: str | None = None


class _FakeGroup:
    def __init__(self) -> None:
        self.contexts: list[_FakeContext] = []

    @contextmanager
    def graph_capture(self, context: _FakeContext):
        self.contexts.append(context)
        yield context


def _load_graph_capture(patched: str) -> dict[str, Any]:
    tree = ast.parse(patched)
    selected = [
        node
        for node in tree.body
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                name.startswith("_SPARK_")
                for name in (
                    [target.id for target in node.targets if isinstance(target, ast.Name)]
                    if isinstance(node, ast.Assign)
                    else [node.target.id]
                    if isinstance(node.target, ast.Name)
                    else []
                )
            )
        )
        or (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in {"_spark_graph_capture_context", "graph_capture"}
        )
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0),
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)

    group = _FakeGroup()
    fake_cuda = SimpleNamespace(Stream=_FakeStream, current_device=lambda: 0)
    namespace: dict[str, Any] = {
        "Any": Any,
        "GraphCaptureContext": _FakeContext,
        "_DCP": None,
        "contextlib": SimpleNamespace(AbstractContextManager=object),
        "contextmanager": contextmanager,
        "get_dcp_group": lambda: (_ for _ in ()).throw(AssertionError("DCP disabled")),
        "get_pp_group": lambda: group,
        "get_tp_group": lambda: group,
        "nullcontext": nullcontext,
        "os": os,
        "threading": threading,
        "torch": SimpleNamespace(
            cuda=fake_cuda,
            device=lambda device: device,
        ),
    }
    exec(compile(module, "parallel_state_overlay.py", "exec"), namespace)
    namespace["_fake_group"] = group
    return namespace


def test_exact_composed_source_generates_shared_stream_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = (
        HERE.parent.parent
        / ".sparkring/r7-build-context/vllm/vllm/distributed/parallel_state.py"
    )
    if not source_path.is_file():
        pytest.skip("the optional composed R7 vLLM source tree is absent")

    source_bytes = source_path.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == BUILDER.EXPECTED_SOURCE_SHA256
    patched = BUILDER.patch_source(source_bytes.decode("utf-8"))
    compile(patched, str(source_path), "exec")

    namespace = _load_graph_capture(patched)
    graph_capture = namespace["graph_capture"]
    device = _FakeDevice(index=0)
    monkeypatch.setenv("VLLM_SPARK_SHARED_CAPTURE_STREAM", "1")

    # These sequential calls model target and speculative-draft graph managers.
    with graph_capture(device, channel_id="vllm:target:production") as target:
        with pytest.raises(RuntimeError, match="overlapping Spark shared"):
            with graph_capture(device, channel_id="vllm:draft:prefill:production"):
                pass
    with graph_capture(device, channel_id="vllm:draft:prefill:production") as draft:
        pass

    assert target is not draft
    assert target.stream is draft.stream
    assert target.channel_id == "vllm:target:production"
    assert draft.channel_id == "vllm:draft:prefill:production"
    assert not namespace["_SPARK_ACTIVE_CAPTURE_STREAMS"]

    explicit = _FakeContext(_FakeStream(device=device), "explicit")
    with graph_capture(device, explicit) as observed:
        assert observed is explicit

    monkeypatch.setenv("VLLM_SPARK_SHARED_CAPTURE_STREAM", "0")
    with graph_capture(device, channel_id="stock-a") as stock_a:
        pass
    with graph_capture(device, channel_id="stock-b") as stock_b:
        pass
    assert stock_a.stream is not stock_b.stream

    assert "capture_b12x_dcp_a2a(" in patched
    assert "maybe_dcp_capture" in patched
    assert "graph capture context and argument specify different" in patched


def test_build_rejects_unpinned_source(tmp_path: Path) -> None:
    source = tmp_path / "parallel_state.py"
    source.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        BUILDER.build(source, tmp_path / "overlay.py")


def test_patch_rejects_source_drift() -> None:
    with pytest.raises(RuntimeError, match="import preimage"):
        BUILDER.patch_source("import contextlib\n")
