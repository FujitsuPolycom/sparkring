"""Pre-reserve a pointer-stable workspace for NF3 CUDA graph capture.

The pinned vLLM V2 runner locks its global workspace after CUDA graph
capture.  The NF3 reference profiles later exercise a mixed Q4096
prefill/decode path whose 575.31 MiB requirement exceeds the observed
544 MiB capture high-water mark.

Growing at the lock boundary would be unsafe: it would replace storage after
CUDA graphs had captured pointers into it.  This adapter therefore wraps the
source-attested ``GPUModelRunner.capture_model`` method, reserves before the
first capture, and then proves that the same buffer survived through the
original capture and lock.  The current reference profiles pin 768 MiB.  The
adapter admits reserves of at least 640 MiB so future profiles can select a
larger value without changing this mechanism.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import logging
import os
import threading
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)

_PROFILE_ENV = "VLLM_SPARK_NF3_PROFILE"
_RESERVE_ENV = "VLLM_SPARK_NF3_WORKSPACE_RESERVE_BYTES"
_V2_RUNNER_ENV = "VLLM_USE_V2_MODEL_RUNNER"
_REFERENCE_PROFILES = frozenset(
    {
        "reference-four-spark",
        "reference-four-spark-adaptive-2-4",
        "reference-four-spark-adaptive-2-4-c8",
    }
)
_MIB = 1024**2
_MINIMUM_RESERVE_BYTES = 640 * _MIB
_REFERENCE_RESERVE_BYTES = 768 * _MIB
_PATCH_MARKER = "_spark_nf3_workspace_reserve"

_EXPECTED_VERSION = (
    "0.11.2.dev279+eldritch.final.fcc6141.b12x284a2ea."
    "fi25dd814.cu132.20260626"
)
_EXPECTED_CAPTURE_MODEL_SHA256 = (
    "f2259e9e42407d4332ee2726e1d98f3733ca764aeeec593586018593ed0d432a"
)
_EXPECTED_LOCK_WORKSPACE_SHA256 = (
    "4e1e7bfae2c2012f631825cd0d4a6f43482f900293a4aba6732c586a73b37f7b"
)
_EXPECTED_CURRENT_MANAGER_SHA256 = (
    "2ccab860b6614b3dc620a316e58b2c2e87232c034c881b4a077b8777bbcacdd9"
)
_EXPECTED_MANAGER_INIT_SHA256 = (
    "c2fe1e014eb2fd1002fd565f9494bae74e10d9d6664923d256acf505b4aacfdf"
)
_EXPECTED_ENSURE_WORKSPACE_SHA256 = (
    "2cf54f2386b3bf97b403cc75da644831632ab36a59e6157ec4db2fd544a03943"
)
_EXPECTED_IS_LOCKED_SHA256 = (
    "1d3b5186b72eaae16a57e7dcd7932fc0645135eeba16004ac2c60fddad028969"
)
_EXPECTED_WORKSPACE_SIZE_SHA256 = (
    "18f7cc8fc41cddbe85060bd27c9a74c7408dc3845f4ad54ad33b8d03193e8645"
)


@dataclass(frozen=True)
class _RuntimeBindings:
    version: str
    model_runner_module: ModuleType | Any
    runner_cls: type[Any]
    workspace_module: ModuleType | Any
    manager_cls: type[Any]


_LOCK = threading.RLock()
_INSTALLED = False
_OWNED = False
_CALLS = 0
_PROFILE: str | None = None
_RESERVE_BYTES: int | None = None
_LAST_WORKSPACE_BYTES: int | None = None
_PATCHED_RUNNER_CLS: type[Any] | None = None
_ORIGINAL_CAPTURE_MODEL: Callable[..., Any] | None = None


def _source_sha256(value: object) -> str:
    return hashlib.sha256(inspect.getsource(value).encode("utf-8")).hexdigest()


def _attest_method(
    value: object,
    *,
    expected_sha256: str,
    expected_parameters: tuple[str, ...],
    label: str,
) -> None:
    actual_sha256 = _source_sha256(value)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{label} source attestation failed: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    actual_parameters = tuple(inspect.signature(value).parameters)
    if actual_parameters != expected_parameters:
        raise RuntimeError(
            f"{label} ABI changed: expected {expected_parameters}, "
            f"got {actual_parameters}"
        )


def _parse_configuration() -> tuple[str, int]:
    profile = os.getenv(_PROFILE_ENV)
    if profile not in _REFERENCE_PROFILES:
        raise RuntimeError(
            f"{_PROFILE_ENV} must select an exact NF3 reference profile; "
            f"got {profile!r}"
        )
    if os.getenv(_V2_RUNNER_ENV) != "1":
        raise RuntimeError(
            f"{_V2_RUNNER_ENV}=1 is required for the NF3 workspace reserve"
        )
    raw_reserve = os.getenv(_RESERVE_ENV)
    if not raw_reserve or not raw_reserve.isdecimal():
        raise RuntimeError(
            f"{_RESERVE_ENV} must be an integer byte count; "
            f"got {raw_reserve!r}"
        )
    reserve_bytes = int(raw_reserve)
    if reserve_bytes < _MINIMUM_RESERVE_BYTES:
        raise RuntimeError(
            f"{_RESERVE_ENV} must be at least "
            f"{_MINIMUM_RESERVE_BYTES} bytes (640 MiB); got {reserve_bytes}"
        )
    return profile, reserve_bytes


def _load_bindings() -> _RuntimeBindings:
    vllm = importlib.import_module("vllm")
    model_runner_module = importlib.import_module(
        "vllm.v1.worker.gpu.model_runner"
    )
    workspace_module = importlib.import_module("vllm.v1.worker.workspace")
    return _RuntimeBindings(
        version=vllm.__version__,
        model_runner_module=model_runner_module,
        runner_cls=model_runner_module.GPUModelRunner,
        workspace_module=workspace_module,
        manager_cls=workspace_module.WorkspaceManager,
    )


def _attest(bindings: _RuntimeBindings) -> None:
    if bindings.version != _EXPECTED_VERSION:
        raise RuntimeError(
            "NF3 workspace reserve does not recognize vLLM version "
            f"{bindings.version!r}"
        )
    _attest_method(
        bindings.runner_cls.capture_model,
        expected_sha256=_EXPECTED_CAPTURE_MODEL_SHA256,
        expected_parameters=("self",),
        label="V2 GPUModelRunner.capture_model",
    )
    _attest_method(
        bindings.workspace_module.lock_workspace,
        expected_sha256=_EXPECTED_LOCK_WORKSPACE_SHA256,
        expected_parameters=(),
        label="workspace.lock_workspace",
    )
    _attest_method(
        bindings.workspace_module.current_workspace_manager,
        expected_sha256=_EXPECTED_CURRENT_MANAGER_SHA256,
        expected_parameters=(),
        label="workspace.current_workspace_manager",
    )
    _attest_method(
        bindings.manager_cls.__init__,
        expected_sha256=_EXPECTED_MANAGER_INIT_SHA256,
        expected_parameters=("self", "device", "num_ubatches"),
        label="WorkspaceManager.__init__",
    )
    _attest_method(
        bindings.manager_cls._ensure_workspace_size,
        expected_sha256=_EXPECTED_ENSURE_WORKSPACE_SHA256,
        expected_parameters=("self", "required_bytes"),
        label="WorkspaceManager._ensure_workspace_size",
    )
    _attest_method(
        bindings.manager_cls.is_locked,
        expected_sha256=_EXPECTED_IS_LOCKED_SHA256,
        expected_parameters=("self",),
        label="WorkspaceManager.is_locked",
    )
    _attest_method(
        bindings.manager_cls._workspace_size_bytes,
        expected_sha256=_EXPECTED_WORKSPACE_SIZE_SHA256,
        expected_parameters=("workspace",),
        label="WorkspaceManager._workspace_size_bytes",
    )
    if (
        bindings.model_runner_module.lock_workspace
        is not bindings.workspace_module.lock_workspace
    ):
        raise RuntimeError(
            "V2 model_runner.lock_workspace no longer owns the audited "
            "workspace.lock_workspace binding"
        )


def _install_on_class(
    runner_cls: type[Any],
    workspace_module: Any,
    *,
    reserve_bytes: int,
    profile: str,
) -> bool:
    global _INSTALLED, _OWNED, _PROFILE, _RESERVE_BYTES
    global _PATCHED_RUNNER_CLS, _ORIGINAL_CAPTURE_MODEL

    requested = (profile, reserve_bytes)
    with _LOCK:
        current = runner_cls.capture_model
        existing = getattr(current, _PATCH_MARKER, None)
        if existing is not None:
            if existing != requested:
                raise RuntimeError(
                    "NF3 workspace reserve is already installed with "
                    f"{existing}, refusing {requested}"
                )
            _INSTALLED = True
            _OWNED = True
            _PROFILE = profile
            _RESERVE_BYTES = reserve_bytes
            return False

        original_capture_model = current
        manager_cls = workspace_module.WorkspaceManager

        @functools.wraps(original_capture_model)
        def capture_with_reserved_workspace(self: Any) -> Any:
            global _CALLS, _LAST_WORKSPACE_BYTES

            with _LOCK:
                manager = workspace_module.current_workspace_manager()
                if type(manager) is not manager_cls:
                    raise RuntimeError(
                        "NF3 workspace reserve refuses unexpected manager "
                        f"type {type(manager)!r}"
                    )
                if getattr(manager, "_num_ubatches", None) != 1:
                    raise RuntimeError(
                        "NF3 workspace reserve requires exactly one "
                        "workspace ubatch lane"
                    )
                if manager.is_locked():
                    raise RuntimeError(
                        "NF3 workspace was locked before CUDA graph capture"
                    )

                reserved_workspace = manager._ensure_workspace_size(
                    reserve_bytes
                )
                actual_bytes = int(
                    manager._workspace_size_bytes(reserved_workspace)
                )
                if actual_bytes < reserve_bytes:
                    raise RuntimeError(
                        "NF3 workspace reservation returned only "
                        f"{actual_bytes} bytes; required {reserve_bytes}"
                    )

                result = original_capture_model(self)

                current_manager = workspace_module.current_workspace_manager()
                current_workspaces = getattr(
                    current_manager,
                    "_current_workspaces",
                    None,
                )
                if (
                    current_manager is not manager
                    or not isinstance(current_workspaces, list)
                    or len(current_workspaces) != 1
                    or current_workspaces[0] is not reserved_workspace
                ):
                    raise RuntimeError(
                        "NF3 workspace storage changed during CUDA graph "
                        "capture; captured pointers are unsafe"
                    )
                if not manager.is_locked():
                    raise RuntimeError(
                        "NF3 workspace remained unlocked after CUDA graph "
                        "capture"
                    )

                _CALLS += 1
                _LAST_WORKSPACE_BYTES = actual_bytes
                LOGGER.warning(
                    "Spark NF3 CUDA graph workspace is pointer-stable: "
                    "profile=%s reserve_bytes=%d actual_bytes=%d",
                    profile,
                    reserve_bytes,
                    actual_bytes,
                )
                return result

        setattr(capture_with_reserved_workspace, _PATCH_MARKER, requested)
        runner_cls.capture_model = capture_with_reserved_workspace
        _PATCHED_RUNNER_CLS = runner_cls
        _ORIGINAL_CAPTURE_MODEL = original_capture_model
        _INSTALLED = True
        _OWNED = (
            getattr(runner_cls.capture_model, _PATCH_MARKER, None)
            == requested
        )
        _PROFILE = profile
        _RESERVE_BYTES = reserve_bytes
        return True


def install() -> bool:
    """Attest and install the configured NF3 V2 workspace reserve."""

    profile, reserve_bytes = _parse_configuration()
    bindings = _load_bindings()
    _attest(bindings)
    return _install_on_class(
        bindings.runner_cls,
        bindings.workspace_module,
        reserve_bytes=reserve_bytes,
        profile=profile,
    )


def workspace_reserve_snapshot() -> dict[str, Any]:
    with _LOCK:
        owned = _OWNED
        if _PATCHED_RUNNER_CLS is not None:
            owned = (
                getattr(
                    _PATCHED_RUNNER_CLS.capture_model,
                    _PATCH_MARKER,
                    None,
                )
                == (_PROFILE, _RESERVE_BYTES)
            )
        return {
            "installed": _INSTALLED,
            "owned": owned,
            "calls": _CALLS,
            "profile": _PROFILE,
            "reserve_bytes": _RESERVE_BYTES,
            "minimum_reserve_bytes": _MINIMUM_RESERVE_BYTES,
            "reference_reserve_bytes": _REFERENCE_RESERVE_BYTES,
            "last_workspace_bytes": _LAST_WORKSPACE_BYTES,
            "version": _EXPECTED_VERSION,
            "capture_model_source_sha256": (
                _EXPECTED_CAPTURE_MODEL_SHA256
            ),
            "lock_workspace_source_sha256": (
                _EXPECTED_LOCK_WORKSPACE_SHA256
            ),
            "ensure_workspace_source_sha256": (
                _EXPECTED_ENSURE_WORKSPACE_SHA256
            ),
        }


def _reset_for_tests() -> None:
    global _INSTALLED, _OWNED, _CALLS, _PROFILE, _RESERVE_BYTES
    global _LAST_WORKSPACE_BYTES, _PATCHED_RUNNER_CLS
    global _ORIGINAL_CAPTURE_MODEL

    with _LOCK:
        if (
            _PATCHED_RUNNER_CLS is not None
            and _ORIGINAL_CAPTURE_MODEL is not None
        ):
            _PATCHED_RUNNER_CLS.capture_model = _ORIGINAL_CAPTURE_MODEL
        _INSTALLED = False
        _OWNED = False
        _CALLS = 0
        _PROFILE = None
        _RESERVE_BYTES = None
        _LAST_WORKSPACE_BYTES = None
        _PATCHED_RUNNER_CLS = None
        _ORIGINAL_CAPTURE_MODEL = None
