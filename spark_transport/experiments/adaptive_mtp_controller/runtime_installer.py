"""Opt-in, source-pinned adaptive-MTP reset/status installer.

Importing this module is inert.  ``install()`` is the deployment entry point;
``RuntimeInstaller`` is the dependency-free monkeypatch used by unit tests.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from .controller_surface import AdaptiveMtpControlSurface, ControllerTelemetry


_EXPECTED_VLLM_VERSION = (
    "0.11.2.dev279+eldritch.final.fcc6141.b12x284a2ea."
    "fi25dd814.cu132.20260626"
)
_EXPECTED_FILE_SHA256 = {
    "vllm.v1.spec_decode.dynamic.acceptance_length": (
        "bbc5176e48827fee1412a7ce95ecd8ef7a57c60ad8b7c66e16447ae3cb133a0e"
    ),
    "vllm.v1.core.sched.scheduler": (
        "0887b3a649774e2189bbbc66f10761550a64f2666219c42ffad86fa69975a52e"
    ),
    "vllm.v1.core.sched.async_scheduler": (
        "da6343d7e7c394a1738cf72905cbecc208003ffa461ccb441268333a3eb9f884"
    ),
    "vllm.v1.engine.core": (
        "cd661d0356003026225a83293234acf5d8b668acea48bcfdbee2881b05aa452d"
    ),
}
_OPT_IN_ENV = "SPARK_ADAPTIVE_MTP_CONTROL"
_TELEMETRY_ATTR = "_spark_adaptive_mtp_telemetry"
_WRAPPER_MARKER = "_spark_adaptive_mtp_control_wrapper"


@dataclass(frozen=True)
class RuntimeTypes:
    controller: type
    scheduler: type
    engine_core: type


class RuntimeInstaller:
    """Install one idle-epoch wrapper and one EngineCore utility method."""

    def __init__(
        self,
        types: RuntimeTypes,
        *,
        depth_ladder: tuple[int, ...],
    ) -> None:
        depths = tuple(sorted(set(depth_ladder)))
        if not depths or any(
            not isinstance(depth, int)
            or isinstance(depth, bool)
            or depth <= 0
            for depth in depths
        ):
            raise ValueError("depth_ladder must contain positive integers")
        self._types = types
        self._depth_ladder = depths
        self._original_engine_add_request: Callable[..., Any] | None = None
        self._installed = False

    def _telemetry_for(self, controller: Any) -> ControllerTelemetry:
        telemetry = getattr(controller, _TELEMETRY_ATTR, None)
        if telemetry is not None:
            if not isinstance(telemetry, ControllerTelemetry):
                raise RuntimeError(
                    "adaptive-MTP telemetry attribute is owned by another patch"
                )
            return telemetry

        maximum = getattr(controller, "max_num_spec_tokens", None)
        if (
            not isinstance(maximum, int)
            or isinstance(maximum, bool)
            or maximum <= 0
        ):
            raise RuntimeError(
                "adaptive-MTP controller ABI mismatch: invalid maximum depth"
            )
        depths = tuple(sorted(set((*self._depth_ladder, maximum))))
        telemetry = ControllerTelemetry(depth_ladder=depths)
        setattr(controller, _TELEMETRY_ATTR, telemetry)
        return telemetry

    def _validate(self) -> Callable[..., Any]:
        engine_add_request = getattr(
            self._types.engine_core, "add_request", None
        )
        if not callable(engine_add_request):
            raise RuntimeError(
                f"{self._types.engine_core.__qualname__}.add_request is absent"
            )
        if getattr(engine_add_request, _WRAPPER_MARKER, False):
            raise RuntimeError(
                f"{self._types.engine_core.__qualname__}.add_request "
                "is already wrapped"
            )
        if hasattr(self._types.engine_core, "spark_adaptive_mtp_control"):
            raise RuntimeError(
                "EngineCore.spark_adaptive_mtp_control already exists"
            )
        return engine_add_request

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("adaptive-MTP runtime installer is already installed")
        engine_add_request = self._validate()
        installer = self

        def engine_control(engine_core: Any, action: str) -> dict[str, Any]:
            scheduler = getattr(engine_core, "scheduler", None)
            if scheduler is None:
                raise RuntimeError("EngineCore has no scheduler")
            controller = getattr(
                scheduler, "acceptance_length_controller", None
            )
            telemetry = (
                installer._telemetry_for(controller)
                if controller is not None
                else ControllerTelemetry(depth_ladder=installer._depth_ladder)
            )
            return AdaptiveMtpControlSurface(
                engine_core, telemetry=telemetry
            ).control(action)

        @functools.wraps(engine_add_request)
        def wrapped_engine_add_request(
            engine_core: Any,
            request: Any,
            request_wave: int = 0,
        ) -> Any:
            scheduler = getattr(engine_core, "scheduler", None)
            controller = (
                getattr(scheduler, "acceptance_length_controller", None)
                if scheduler is not None
                else None
            )
            if controller is not None:
                has_work = getattr(engine_core, "has_work", None)
                if not callable(has_work):
                    raise RuntimeError(
                        "EngineCore.has_work is required for idle-epoch reset"
                    )
                if not has_work():
                    telemetry = installer._telemetry_for(controller)
                    AdaptiveMtpControlSurface(
                        engine_core, telemetry=telemetry
                    ).reset_idle_epoch_if_safe()
            return engine_add_request(engine_core, request, request_wave)

        setattr(wrapped_engine_add_request, _WRAPPER_MARKER, True)

        # All validation and wrapper construction completes before mutation.
        self._types.engine_core.add_request = wrapped_engine_add_request
        self._types.engine_core.spark_adaptive_mtp_control = engine_control
        self._original_engine_add_request = engine_add_request
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        current_engine_add_request = self._types.engine_core.add_request
        if not getattr(current_engine_add_request, _WRAPPER_MARKER, False):
            raise RuntimeError("adaptive-MTP runtime hooks changed after install")
        assert self._original_engine_add_request is not None
        self._types.engine_core.add_request = self._original_engine_add_request
        delattr(self._types.engine_core, "spark_adaptive_mtp_control")
        self._original_engine_add_request = None
        self._installed = False


def _file_sha256(module: ModuleType) -> str:
    filename = getattr(module, "__file__", None)
    if not filename:
        raise RuntimeError(f"module {module.__name__!r} has no source file")
    path = Path(filename)
    if path.suffix == ".pyc":
        path = Path(importlib.util.source_from_cache(str(path)))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_depth_ladder() -> tuple[int, ...]:
    raw = os.getenv("VLLM_ADAPTIVE_SPEC_DEPTHS", "2,4")
    try:
        depths = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as error:
        raise RuntimeError(
            "VLLM_ADAPTIVE_SPEC_DEPTHS must be a comma-separated integer ladder"
        ) from error
    if not depths or any(depth <= 0 for depth in depths):
        raise RuntimeError("VLLM_ADAPTIVE_SPEC_DEPTHS must contain positive depths")
    return depths


def _load_and_attest() -> RuntimeTypes:
    import vllm

    if vllm.__version__ != _EXPECTED_VLLM_VERSION:
        raise RuntimeError(f"unsupported vLLM version: {vllm.__version__}")
    modules = {
        name: importlib.import_module(name)
        for name in _EXPECTED_FILE_SHA256
    }
    mismatches = []
    for name, expected in _EXPECTED_FILE_SHA256.items():
        actual = _file_sha256(modules[name])
        if actual != expected:
            mismatches.append(f"{name}: expected {expected}, got {actual}")
    if mismatches:
        raise RuntimeError(
            "adaptive-MTP source attestation failed: " + "; ".join(mismatches)
        )
    return RuntimeTypes(
        controller=modules[
            "vllm.v1.spec_decode.dynamic.acceptance_length"
        ].AcceptanceLengthController,
        scheduler=modules["vllm.v1.core.sched.scheduler"].Scheduler,
        engine_core=modules["vllm.v1.engine.core"].EngineCore,
    )


_installer: RuntimeInstaller | None = None


def install() -> None:
    """Attest the deployed ABI, then install scheduler-local control hooks."""
    global _installer
    if os.getenv(_OPT_IN_ENV) != "1":
        raise RuntimeError(f"{_OPT_IN_ENV}=1 is required")
    if _installer is not None:
        raise RuntimeError("adaptive-MTP control is already installed")
    candidate = RuntimeInstaller(
        _load_and_attest(),
        depth_ladder=_parse_depth_ladder(),
    )
    candidate.install()
    _installer = candidate


def runtime_installation_snapshot() -> dict[str, bool]:
    """Return hook-ownership attestation for startup and live gates."""

    if _installer is None:
        return {
            "installed": False,
            "add_request_owned": False,
            "utility_owned": False,
        }
    engine_core = _installer._types.engine_core
    add_request = getattr(engine_core, "add_request", None)
    utility = getattr(engine_core, "spark_adaptive_mtp_control", None)
    add_request_owned = bool(getattr(add_request, _WRAPPER_MARKER, False))
    utility_owned = callable(utility)
    return {
        "installed": bool(
            _installer._installed and add_request_owned and utility_owned
        ),
        "add_request_owned": add_request_owned,
        "utility_owned": utility_owned,
    }


def uninstall() -> None:
    """Restore original class methods (test/development only)."""
    global _installer
    if _installer is None:
        return
    _installer.uninstall()
    _installer = None
