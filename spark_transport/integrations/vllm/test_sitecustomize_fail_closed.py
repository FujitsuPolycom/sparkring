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
    "VLLM_SPARK_NF3_PROFILE",
    "VLLM_SPARK_NF3_WORKSPACE_RESERVE_BYTES",
    "VLLM_SPARK_NF3_STARTUP_PROFILE_MAX_TOKENS",
    "VLLM_SPARK_NF3_SINGLE_COMPILE_RANGE",
    "VLLM_USE_V2_MODEL_RUNNER",
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


def _fake_nf3_profile_cap_module(
    *,
    install_error: BaseException | None = None,
    compile_warmup_installed: bool = True,
    runner_kind: str = "v2",
    v1_runner_owned: bool = False,
    v2_runner_owned: bool = True,
    memory_ownership_guard_installed: bool = True,
) -> ModuleType:
    module = ModuleType("spark_nf3_startup_profile_cap")
    state = {"installed": False, "calls": 0}

    def install() -> bool:
        state["calls"] += 1
        if install_error is not None:
            raise install_error
        state["installed"] = True
        return True

    def startup_profile_cap_snapshot() -> dict[str, object]:
        return {
            **state,
            "compile_warmup_installed": compile_warmup_installed,
            "runner_kind": runner_kind,
            "v1_runner_owned": v1_runner_owned,
            "v2_runner_owned": v2_runner_owned,
            "memory_ownership_guard_installed": (
                memory_ownership_guard_installed
            ),
        }

    module.install = install  # type: ignore[attr-defined]
    module.startup_profile_cap_snapshot = (  # type: ignore[attr-defined]
        startup_profile_cap_snapshot
    )
    module.state = state  # type: ignore[attr-defined]
    return module


def _fake_nf3_workspace_reserve_module(
    *,
    profile: str = "reference-four-spark",
    reserve_bytes: int = 768 * 1024**2,
    owned: bool = True,
    install_error: BaseException | None = None,
) -> ModuleType:
    module = ModuleType("spark_nf3_workspace_reserve")
    state = {"installed": False, "calls": 0}

    def install() -> bool:
        state["calls"] += 1
        if install_error is not None:
            raise install_error
        state["installed"] = True
        return True

    def workspace_reserve_snapshot() -> dict[str, object]:
        return {
            **state,
            "owned": owned,
            "profile": profile,
            "reserve_bytes": reserve_bytes,
        }

    module.install = install  # type: ignore[attr-defined]
    module.workspace_reserve_snapshot = (  # type: ignore[attr-defined]
        workspace_reserve_snapshot
    )
    module.state = state  # type: ignore[attr-defined]
    return module


def test_required_nf3_workspace_reserve_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_features(monkeypatch)
    monkeypatch.setenv("VLLM_SPARK_NF3_PROFILE", "reference-four-spark")
    monkeypatch.setenv(
        "VLLM_SPARK_NF3_WORKSPACE_RESERVE_BYTES",
        str(768 * 1024**2),
    )
    module = _fake_nf3_workspace_reserve_module()
    monkeypatch.setitem(
        sys.modules,
        "spark_nf3_workspace_reserve",
        module,
    )

    runpy.run_path(
        str(SITECUSTOMIZE),
        run_name="test_nf3_workspace_reserve_success",
    )

    assert module.state == {"installed": True, "calls": 1}  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "module",
    (
        _fake_nf3_workspace_reserve_module(owned=False),
        _fake_nf3_workspace_reserve_module(
            install_error=RuntimeError("synthetic reserve failure")
        ),
    ),
)
def test_required_nf3_workspace_reserve_failure_is_fatal(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> None:
    _clear_features(monkeypatch)
    monkeypatch.setenv("VLLM_SPARK_NF3_PROFILE", "reference-four-spark")
    monkeypatch.setenv(
        "VLLM_SPARK_NF3_WORKSPACE_RESERVE_BYTES",
        str(768 * 1024**2),
    )
    monkeypatch.setitem(
        sys.modules,
        "spark_nf3_workspace_reserve",
        module,
    )
    monkeypatch.setattr(
        os,
        "_exit",
        lambda code: (_ for _ in ()).throw(FatalExit(code)),
    )

    with pytest.raises(FatalExit) as caught:
        runpy.run_path(
            str(SITECUSTOMIZE),
            run_name="test_nf3_workspace_reserve_failure",
        )

    assert caught.value.code == 78


def test_reference_profile_without_workspace_reserve_is_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_features(monkeypatch)
    monkeypatch.setenv("VLLM_SPARK_NF3_PROFILE", "reference-four-spark")
    module = _fake_nf3_workspace_reserve_module()
    monkeypatch.setitem(
        sys.modules,
        "spark_nf3_workspace_reserve",
        module,
    )
    monkeypatch.setattr(
        os,
        "_exit",
        lambda code: (_ for _ in ()).throw(FatalExit(code)),
    )

    with pytest.raises(FatalExit) as caught:
        runpy.run_path(
            str(SITECUSTOMIZE),
            run_name="test_nf3_workspace_reserve_missing",
        )

    assert caught.value.code == 78


def test_required_nf3_profile_cap_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_features(monkeypatch)
    monkeypatch.setenv("VLLM_SPARK_NF3_STARTUP_PROFILE_MAX_TOKENS", "256")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    module = _fake_nf3_profile_cap_module()
    monkeypatch.setitem(
        sys.modules,
        "spark_nf3_startup_profile_cap",
        module,
    )

    runpy.run_path(str(SITECUSTOMIZE), run_name="test_nf3_profile_cap_success")

    assert module.state == {"installed": True, "calls": 1}  # type: ignore[attr-defined]


def test_required_nf3_profile_cap_failure_is_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_features(monkeypatch)
    monkeypatch.setenv("VLLM_SPARK_NF3_STARTUP_PROFILE_MAX_TOKENS", "256")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    module = _fake_nf3_profile_cap_module(
        install_error=RuntimeError("synthetic profile-cap failure")
    )
    monkeypatch.setitem(
        sys.modules,
        "spark_nf3_startup_profile_cap",
        module,
    )
    monkeypatch.setattr(
        os,
        "_exit",
        lambda code: (_ for _ in ()).throw(FatalExit(code)),
    )

    with pytest.raises(FatalExit) as caught:
        runpy.run_path(
            str(SITECUSTOMIZE),
            run_name="test_nf3_profile_cap_failure",
        )

    assert caught.value.code == 78
    assert module.state == {"installed": False, "calls": 1}  # type: ignore[attr-defined]


def test_required_nf3_single_range_worker_hook_is_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_features(monkeypatch)
    monkeypatch.setenv("VLLM_SPARK_NF3_STARTUP_PROFILE_MAX_TOKENS", "1")
    monkeypatch.setenv("VLLM_SPARK_NF3_SINGLE_COMPILE_RANGE", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    module = _fake_nf3_profile_cap_module(
        compile_warmup_installed=False
    )
    monkeypatch.setitem(
        sys.modules,
        "spark_nf3_startup_profile_cap",
        module,
    )
    monkeypatch.setattr(
        os,
        "_exit",
        lambda code: (_ for _ in ()).throw(FatalExit(code)),
    )

    with pytest.raises(FatalExit) as caught:
        runpy.run_path(
            str(SITECUSTOMIZE),
            run_name="test_nf3_single_range_worker_unowned",
        )

    assert caught.value.code == 78


def test_required_nf3_runner_ownership_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_features(monkeypatch)
    monkeypatch.setenv("VLLM_SPARK_NF3_STARTUP_PROFILE_MAX_TOKENS", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    module = _fake_nf3_profile_cap_module(
        v1_runner_owned=True,
    )
    monkeypatch.setitem(
        sys.modules,
        "spark_nf3_startup_profile_cap",
        module,
    )
    monkeypatch.setattr(
        os,
        "_exit",
        lambda code: (_ for _ in ()).throw(FatalExit(code)),
    )

    with pytest.raises(FatalExit) as caught:
        runpy.run_path(
            str(SITECUSTOMIZE),
            run_name="test_nf3_runner_ownership_unowned",
        )

    assert caught.value.code == 78


def test_required_nf3_runner_kind_matches_explicit_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_features(monkeypatch)
    monkeypatch.setenv("VLLM_SPARK_NF3_STARTUP_PROFILE_MAX_TOKENS", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    module = _fake_nf3_profile_cap_module(runner_kind="v1")
    monkeypatch.setitem(
        sys.modules,
        "spark_nf3_startup_profile_cap",
        module,
    )
    monkeypatch.setattr(
        os,
        "_exit",
        lambda code: (_ for _ in ()).throw(FatalExit(code)),
    )

    with pytest.raises(FatalExit) as caught:
        runpy.run_path(
            str(SITECUSTOMIZE),
            run_name="test_nf3_wrong_runner_kind",
        )

    assert caught.value.code == 78


def test_required_nf3_memory_guard_is_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_features(monkeypatch)
    monkeypatch.setenv("VLLM_SPARK_NF3_STARTUP_PROFILE_MAX_TOKENS", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    module = _fake_nf3_profile_cap_module(
        memory_ownership_guard_installed=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "spark_nf3_startup_profile_cap",
        module,
    )
    monkeypatch.setattr(
        os,
        "_exit",
        lambda code: (_ for _ in ()).throw(FatalExit(code)),
    )

    with pytest.raises(FatalExit) as caught:
        runpy.run_path(
            str(SITECUSTOMIZE),
            run_name="test_nf3_memory_guard_unowned",
        )

    assert caught.value.code == 78


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


def _fake_tp4_backend_module(name: str, install_error: Exception) -> ModuleType:
    module = ModuleType(name)

    def install() -> None:
        raise install_error

    module.install = install  # type: ignore[attr-defined]
    return module


TP4_FAMILIES = (
    ("VLLM_SPARK_TP4_MODE", "spark_tp4_backend"),
    ("VLLM_SPARK_TP4_ALLGATHER_MODE", "spark_tp4_allgather_backend"),
    ("VLLM_SPARK_TP4_VOCAB_MODE", "spark_tp4_vocab_allgather_backend"),
    ("VLLM_SPARK_TP4_DCP_MODE", "spark_tp4_dcp_backend"),
)


@pytest.mark.parametrize(("flag", "module_name"), TP4_FAMILIES)
def test_required_tp4_family_failure_is_fatal(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    module_name: str,
) -> None:
    """A failed TP4 install terminates rather than serving on stock NCCL.

    CPython's site module suppresses exceptions raised while importing
    sitecustomize. Without an explicit exit, an enabled TP4 family whose
    install fails would leave vLLM serving its own collectives, which is
    a silent transport substitution rather than a refusal.
    """

    _clear_features(monkeypatch)
    monkeypatch.setenv(flag, "custom")
    monkeypatch.setitem(
        sys.modules,
        module_name,
        _fake_tp4_backend_module(
            module_name, RuntimeError(f"synthetic {module_name} failure")
        ),
    )
    monkeypatch.setattr(
        os,
        "_exit",
        lambda code: (_ for _ in ()).throw(FatalExit(code)),
    )

    with pytest.raises(FatalExit) as caught:
        runpy.run_path(str(SITECUSTOMIZE), run_name=f"test_{module_name}")

    assert caught.value.code == 78


@pytest.mark.parametrize(("flag", "module_name"), TP4_FAMILIES)
def test_tp4_family_is_not_installed_when_its_flag_is_unset(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    module_name: str,
) -> None:
    """No TP4 family installs without its own flag, so the guard is scoped."""

    _clear_features(monkeypatch)
    monkeypatch.setitem(
        sys.modules,
        module_name,
        _fake_tp4_backend_module(
            module_name, AssertionError(f"{module_name} installed unbidden")
        ),
    )
    monkeypatch.setattr(
        os,
        "_exit",
        lambda code: (_ for _ in ()).throw(FatalExit(code)),
    )

    runpy.run_path(str(SITECUSTOMIZE), run_name=f"test_{module_name}_unset")
