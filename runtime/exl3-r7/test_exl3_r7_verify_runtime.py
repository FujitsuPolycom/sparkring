"""Tests for fail-closed EXL3 R7 runtime verification."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_VERIFY_RUNTIME_PATH = Path(__file__).with_name("verify_runtime.py")
_SPEC = importlib.util.spec_from_file_location(
    "sparkring_r7_verify_runtime",
    _VERIFY_RUNTIME_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
verify_runtime = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verify_runtime)


def test_wrapper_chain_rejects_cycles() -> None:
    def first(self: object, input_: object) -> None:
        del self, input_

    def second(self: object, input_: object) -> None:
        del self, input_

    first._spark_original = second  # type: ignore[attr-defined]
    second._spark_original = first  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="wrapper chain contains a cycle"):
        verify_runtime.spark_original_chain(
            "CudaCommunicator.all_reduce",
            first,
        )


def test_stacked_wrapper_uses_terminal_signature_and_any_chain_marker() -> None:
    def terminal(self: object, input_: object, dim: int) -> None:
        del self, input_, dim

    def vocabulary_wrapper(
        self: object,
        input_tensor: object,
        dim: int,
    ) -> None:
        del self, input_tensor, dim

    def dcp_wrapper(
        self: object,
        input_tensor: object,
        dim: int,
    ) -> None:
        del self, input_tensor, dim

    vocabulary_wrapper._spark_original = terminal  # type: ignore[attr-defined]
    vocabulary_wrapper._spark_tp4_vocab_backend = True  # type: ignore[attr-defined]
    dcp_wrapper._spark_original = vocabulary_wrapper  # type: ignore[attr-defined]

    actual = verify_runtime.verify_hook_chain(
        "GroupCoordinator._all_gather_out_place",
        dcp_wrapper,
        ("self", "input_", "dim"),
        "_spark_tp4_vocab_backend",
        True,
    )

    assert actual == ("self", "input_", "dim")


def test_wrapper_chain_rejects_more_than_64_original_links() -> None:
    def make_wrapper(index: int) -> object:
        def wrapper(self: object, input_: object) -> None:
            del self, input_

        wrapper.__name__ = f"wrapper_{index}"
        return wrapper

    wrappers = [make_wrapper(index) for index in range(66)]
    for wrapper, original in zip(
        wrappers[:-1],
        wrappers[1:],
        strict=True,
    ):
        wrapper._spark_original = original  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="exceeds 64 original links"):
        verify_runtime.spark_original_chain(
            "CudaCommunicator.all_reduce",
            wrappers[0],
        )


def test_wrapper_chain_rejects_non_callable_target() -> None:
    with pytest.raises(RuntimeError, match="contains a non-callable target"):
        verify_runtime.spark_original_chain(
            "CudaCommunicator.all_reduce",
            object(),
        )


def test_wrapper_chain_rejects_non_callable_original_link() -> None:
    def wrapper(self: object, input_: object) -> None:
        del self, input_

    wrapper._spark_original = object()  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="contains a non-callable original"):
        verify_runtime.spark_original_chain(
            "CudaCommunicator.all_reduce",
            wrapper,
        )


def test_vocab_mode_requires_vocab_marker_in_group_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def all_reduce_terminal(self: object, input_: object) -> None:
        del self, input_

    def all_reduce_wrapper(self: object, input_: object) -> None:
        del self, input_

    all_reduce_wrapper._spark_original = all_reduce_terminal  # type: ignore[attr-defined]
    all_reduce_wrapper._spark_tp4_backend = True  # type: ignore[attr-defined]

    def all_gather_stock(
        self: object,
        output_tensor: object,
        input_tensor: object,
        stream: object,
    ) -> None:
        del self, output_tensor, input_tensor, stream

    def group_terminal(
        self: object,
        input_: object,
        dim: int,
    ) -> None:
        del self, input_, dim

    def vocabulary_wrapper(
        self: object,
        input_tensor: object,
        dim: int,
    ) -> None:
        del self, input_tensor, dim

    vocabulary_wrapper._spark_original = group_terminal  # type: ignore[attr-defined]

    cuda_module = types.ModuleType(
        "vllm.distributed.device_communicators.cuda_communicator"
    )
    cuda_module.CudaCommunicator = types.SimpleNamespace(
        all_reduce=all_reduce_wrapper
    )
    pynccl_module = types.ModuleType(
        "vllm.distributed.device_communicators.pynccl"
    )
    pynccl_module.PyNcclCommunicator = types.SimpleNamespace(
        all_gather=all_gather_stock
    )
    parallel_state = types.ModuleType("vllm.distributed.parallel_state")
    parallel_state.GroupCoordinator = types.SimpleNamespace(
        _all_gather_out_place=vocabulary_wrapper
    )
    for name, module in {
        cuda_module.__name__: cuda_module,
        pynccl_module.__name__: pynccl_module,
        parallel_state.__name__: parallel_state,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr(
        verify_runtime,
        "module_source",
        lambda name: (
            tmp_path / f"{name}.py",
            "\n".join(verify_runtime.SIRCL_HOOK_MARKERS[name]),
        ),
    )
    library = tmp_path / "libspark_transport_capi.so"
    library.write_bytes(b"test")
    for name in (
        "VLLM_SPARK_TP4_MODE",
        "VLLM_SPARK_TP4_VOCAB_MODE",
    ):
        monkeypatch.setenv(name, "custom")
    monkeypatch.setenv("SPARK_TP4_LIBRARY", str(library))

    with pytest.raises(
        RuntimeError,
        match="_spark_tp4_vocab_backend is absent",
    ):
        verify_runtime.verify_sircl_hooks({})

    vocabulary_wrapper._spark_tp4_vocab_backend = True  # type: ignore[attr-defined]
    evidence: dict[str, object] = {}
    verify_runtime.verify_sircl_hooks(evidence)
    assert evidence["sircl"]["targets"][  # type: ignore[index]
        "GroupCoordinator._all_gather_out_place"
    ] == ["self", "input_", "dim"]
