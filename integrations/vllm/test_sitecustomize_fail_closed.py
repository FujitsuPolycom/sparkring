"""Fail-closed startup tests for the supported vLLM adapters."""

from __future__ import annotations

import builtins
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

SITECUSTOMIZE = Path(__file__).with_name("sitecustomize.py")

SUPPORTED_HOOKS = (
    ("VLLM_SPARK_TP4_MODE", "custom", "spark_tp4_backend"),
    ("SPARK_TP4_HEALTH_GATE", "1", "spark_tp4_health_gate"),
    (
        "VLLM_SPARK_TP4_VOCAB_MODE",
        "custom",
        "spark_tp4_vocab_allgather_backend",
    ),
    (
        "SPARK_TP4_DCP_COLLECTIVE_AUDIT",
        "1",
        "spark_dcp_collective_audit",
    ),
    (
        "SPARK_CUDAGRAPH_REPLAY_TIMING",
        "1",
        "spark_cudagraph_replay_timing",
    ),
)

FEATURE_FLAGS = tuple(flag for flag, _value, _module in SUPPORTED_HOOKS)


class FatalExit(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


def _clear_features(monkeypatch: pytest.MonkeyPatch) -> None:
    for flag in FEATURE_FLAGS:
        monkeypatch.delenv(flag, raising=False)


def _fake_module(
    name: str,
    error: BaseException | None = None,
) -> ModuleType:
    module = ModuleType(name)
    module.calls = 0  # type: ignore[attr-defined]

    def install() -> bool:
        module.calls += 1  # type: ignore[attr-defined]
        if error is not None:
            raise error
        return True

    module.install = install  # type: ignore[attr-defined]
    return module


@pytest.mark.parametrize(("flag", "value", "module_name"), SUPPORTED_HOOKS)
def test_enabled_supported_hook_is_installed_once(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str,
    module_name: str,
) -> None:
    _clear_features(monkeypatch)
    monkeypatch.setenv(flag, value)
    module = _fake_module(module_name)
    monkeypatch.setitem(sys.modules, module_name, module)

    runpy.run_path(SITECUSTOMIZE, run_name=f"test_{module_name}_success")

    assert module.calls == 1  # type: ignore[attr-defined]


@pytest.mark.parametrize(("flag", "value", "module_name"), SUPPORTED_HOOKS)
def test_enabled_supported_hook_failure_is_fatal(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str,
    module_name: str,
) -> None:
    _clear_features(monkeypatch)
    monkeypatch.setenv(flag, value)
    monkeypatch.setitem(
        sys.modules,
        module_name,
        _fake_module(module_name, RuntimeError("synthetic install failure")),
    )
    monkeypatch.setattr(
        os,
        "_exit",
        lambda code: (_ for _ in ()).throw(FatalExit(code)),
    )

    with pytest.raises(FatalExit) as caught:
        runpy.run_path(SITECUSTOMIZE, run_name=f"test_{module_name}_failure")

    assert caught.value.code == 78


def test_disabled_hooks_are_not_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_features(monkeypatch)
    modules = {
        module_name: _fake_module(module_name)
        for _flag, _value, module_name in SUPPORTED_HOOKS
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    runpy.run_path(SITECUSTOMIZE, run_name="test_no_supported_hooks")

    assert all(module.calls == 0 for module in modules.values())  # type: ignore[attr-defined]


def test_broken_diagnostic_stream_still_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_features(monkeypatch)
    monkeypatch.setenv("VLLM_SPARK_TP4_MODE", "custom")
    monkeypatch.setitem(
        sys.modules,
        "spark_tp4_backend",
        _fake_module("spark_tp4_backend", RuntimeError("synthetic failure")),
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
        runpy.run_path(SITECUSTOMIZE, run_name="test_broken_diagnostic_stream")

    assert caught.value.code == 78


@pytest.mark.parametrize(("flag", "value", "module_name"), SUPPORTED_HOOKS)
@pytest.mark.parametrize("failure", ["import", "install", "missing"])
def test_required_hook_startup_failure_blocks_user_code(
    tmp_path: Path, flag: str, value: str, module_name: str, failure: str,
) -> None:
    (tmp_path / "sitecustomize.py").write_bytes(SITECUSTOMIZE.read_bytes())
    if failure != "missing":
        source = (
            "raise ImportError('synthetic adapter dependency missing')\n"
            if failure == "import" else
            "def install():\n    raise RuntimeError('synthetic install failure')\n"
        )
        (tmp_path / f"{module_name}.py").write_text(source, encoding="utf-8")
    environment = {k: v for k, v in os.environ.items() if k not in FEATURE_FLAGS}
    environment.update(PYTHONPATH=str(tmp_path), **{flag: value})
    result = subprocess.run(
        [sys.executable, "-c", "print('STARTUP_CONTINUED')"],
        env=environment, cwd=tmp_path, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 78, result.stderr
    assert "STARTUP_CONTINUED" not in result.stdout
    assert "required Spark startup hook failed" in result.stderr


@pytest.mark.parametrize("allreduce_mode", ["", "disabled", "DISABLED"])
def test_disabled_hooks_do_not_import_broken_modules_at_startup(
    tmp_path: Path, allreduce_mode: str,
) -> None:
    (tmp_path / "sitecustomize.py").write_bytes(SITECUSTOMIZE.read_bytes())
    for _flag, _value, module_name in SUPPORTED_HOOKS:
        (tmp_path / f"{module_name}.py").write_text(
            "raise ImportError('disabled adapter must not be imported')\n", encoding="utf-8",
        )
    environment = {k: v for k, v in os.environ.items() if k not in FEATURE_FLAGS}
    environment["PYTHONPATH"] = str(tmp_path)
    environment["VLLM_SPARK_TP4_MODE"] = allreduce_mode
    result = subprocess.run(
        [sys.executable, "-c", "print('STARTUP_CONTINUED')"],
        env=environment, cwd=tmp_path, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "STARTUP_CONTINUED"
    assert result.stderr == ""
