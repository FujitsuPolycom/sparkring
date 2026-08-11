from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("verify_runtime.py")
SPEC = importlib.util.spec_from_file_location("exl3_r7_verify_runtime", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
verify_runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_runtime)


def _terminal(self, output_tensor, input_tensor, stream):
    del self, output_tensor, input_tensor, stream


def _wrapper(inner):
    def wrapped(*args, **kwargs):
        return inner(*args, **kwargs)

    wrapped._spark_original = inner
    return wrapped


def test_hook_chain_accepts_marker_on_inner_wrapper() -> None:
    marked = _wrapper(_terminal)
    marked._spark_tp4_allgather_backend = True
    outer = _wrapper(marked)

    assert verify_runtime.verify_hook_chain(
        "PyNcclCommunicator.all_gather",
        outer,
        ("self", "output_tensor", "input_tensor", "stream"),
        "_spark_tp4_allgather_backend",
        True,
    ) == ("self", "output_tensor", "input_tensor", "stream")


def test_hook_chain_rejects_cycle() -> None:
    first = _wrapper(_terminal)
    second = _wrapper(first)
    first._spark_original = second

    with pytest.raises(RuntimeError, match="wrapper chain cycle"):
        verify_runtime.spark_original_chain("cyclic", second)


def test_hook_chain_rejects_non_callable_original() -> None:
    wrapper = _wrapper(_terminal)
    wrapper._spark_original = object()

    with pytest.raises(RuntimeError, match="non-callable original"):
        verify_runtime.spark_original_chain("invalid", wrapper)


def test_hook_chain_validates_terminal_signature() -> None:
    def wrong(self, input_):
        del self, input_

    wrapper = _wrapper(wrong)
    wrapper._spark_tp4_allgather_backend = True

    with pytest.raises(RuntimeError, match="signature drift"):
        verify_runtime.verify_hook_chain(
            "PyNcclCommunicator.all_gather",
            wrapper,
            ("self", "output_tensor", "input_tensor", "stream"),
            "_spark_tp4_allgather_backend",
            True,
        )


def test_hook_chain_requires_marker_when_mode_is_enabled() -> None:
    wrapper = _wrapper(_terminal)

    with pytest.raises(RuntimeError, match="hook marker.*is absent"):
        verify_runtime.verify_hook_chain(
            "PyNcclCommunicator.all_gather",
            wrapper,
            ("self", "output_tensor", "input_tensor", "stream"),
            "_spark_tp4_allgather_backend",
            True,
        )


def test_hook_chain_rejects_marker_only_on_terminal_callable() -> None:
    _terminal._spark_tp4_allgather_backend = True
    wrapper = _wrapper(_terminal)
    try:
        with pytest.raises(RuntimeError, match="hook marker.*is absent"):
            verify_runtime.verify_hook_chain(
                "PyNcclCommunicator.all_gather",
                wrapper,
                ("self", "output_tensor", "input_tensor", "stream"),
                "_spark_tp4_allgather_backend",
                True,
            )
    finally:
        del _terminal._spark_tp4_allgather_backend


def test_hook_chain_rejects_more_than_64_original_links() -> None:
    target = _terminal
    for _ in range(65):
        target = _wrapper(target)

    with pytest.raises(RuntimeError, match="exceeds 64 original links"):
        verify_runtime.spark_original_chain("too-deep", target)


def test_hook_chain_accepts_exactly_64_original_links() -> None:
    target = _terminal
    for _ in range(64):
        target = _wrapper(target)

    assert len(verify_runtime.spark_original_chain("bounded", target)) == 65


def test_hook_chain_allows_absent_marker_when_mode_is_disabled() -> None:
    wrapper = _wrapper(_terminal)

    assert verify_runtime.verify_hook_chain(
        "PyNcclCommunicator.all_gather",
        wrapper,
        ("self", "output_tensor", "input_tensor", "stream"),
        "_spark_tp4_allgather_backend",
        False,
    ) == ("self", "output_tensor", "input_tensor", "stream")
