from __future__ import annotations

import builtins
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest


SITECUSTOMIZE = Path(__file__).with_name("sitecustomize.py")
FEATURE_FLAGS = (
    "SPARK_ADAPTIVE_MTP_CONTROL",
    "SPARK_GLM52_MTP_INDEX_REUSE",
    "VLLM_SPARK_TRUE_ADAPTIVE_DRAFT",
    "SPARK_Q2R_PROBE",
    "SPARK_TP4_FLIGHT_RECORDER",
    "SPARK_CUDAGRAPH_REPLAY_TIMING",
    "VLLM_SPARK_TP4_MODE",
    "VLLM_SPARK_TP4_ALLGATHER_MODE",
    "VLLM_SPARK_TP4_VOCAB_MODE",
    "VLLM_SPARK_TP4_DCP_MODE",
    "SPARK_TP4_DCP_COLLECTIVE_AUDIT",
    "VLLM_SPARK_TRACE_ALLREDUCE",
)


class FatalExit(BaseException):
    def __init__(self, code: int) -> None:
        self.code = code


def _clear_features(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in FEATURE_FLAGS:
        monkeypatch.delenv(name, raising=False)


def _fake_controller_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    install,
    snapshot,
) -> None:
    package = ModuleType("adaptive_mtp_controller")
    package.__path__ = []  # type: ignore[attr-defined]
    runtime = ModuleType("adaptive_mtp_controller.runtime_installer")
    runtime.install = install  # type: ignore[attr-defined]
    runtime.runtime_installation_snapshot = snapshot  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)


def test_required_controller_install_is_attested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_features(monkeypatch)
    monkeypatch.setenv("SPARK_ADAPTIVE_MTP_CONTROL", "1")
    calls: list[str] = []
    _fake_controller_module(
        monkeypatch,
        install=lambda: calls.append("install"),
        snapshot=lambda: {
            "installed": True,
            "add_request_owned": True,
            "utility_owned": True,
        },
    )

    runpy.run_path(str(SITECUSTOMIZE), run_name="test_sitecustomize_success")

    assert calls == ["install"]


def test_required_controller_failure_exits_instead_of_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_features(monkeypatch)
    monkeypatch.setenv("SPARK_ADAPTIVE_MTP_CONTROL", "1")

    def fail_install() -> None:
        raise RuntimeError("synthetic source-attestation failure")

    _fake_controller_module(
        monkeypatch,
        install=fail_install,
        snapshot=lambda: {"installed": False},
    )
    monkeypatch.setattr(
        os,
        "_exit",
        lambda code: (_ for _ in ()).throw(FatalExit(code)),
    )

    with pytest.raises(FatalExit) as caught:
        runpy.run_path(str(SITECUSTOMIZE), run_name="test_sitecustomize_failure")

    assert caught.value.code == 78


def test_unowned_controller_hooks_are_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_features(monkeypatch)
    monkeypatch.setenv("SPARK_ADAPTIVE_MTP_CONTROL", "1")
    _fake_controller_module(
        monkeypatch,
        install=lambda: None,
        snapshot=lambda: {
            "installed": False,
            "add_request_owned": False,
            "utility_owned": False,
        },
    )
    monkeypatch.setattr(
        os,
        "_exit",
        lambda code: (_ for _ in ()).throw(FatalExit(code)),
    )

    with pytest.raises(FatalExit) as caught:
        runpy.run_path(str(SITECUSTOMIZE), run_name="test_sitecustomize_unowned")

    assert caught.value.code == 78


def test_broken_diagnostic_stream_still_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_features(monkeypatch)
    monkeypatch.setenv("SPARK_ADAPTIVE_MTP_CONTROL", "1")
    _fake_controller_module(
        monkeypatch,
        install=lambda: (_ for _ in ()).throw(RuntimeError("broken")),
        snapshot=lambda: {"installed": False},
    )
    monkeypatch.setattr(
        os,
        "_exit",
        lambda code: (_ for _ in ()).throw(FatalExit(code)),
    )
    monkeypatch.setattr(
        builtins,
        "print",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(BrokenPipeError()),
    )

    with pytest.raises(FatalExit) as caught:
        runpy.run_path(
            str(SITECUSTOMIZE), run_name="test_sitecustomize_broken_stderr"
        )

    assert caught.value.code == 78
