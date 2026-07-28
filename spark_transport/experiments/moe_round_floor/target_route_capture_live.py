"""Opt-in, source-pinned live installer for target expert-route capture.

This module deliberately does not install itself.  An experiment overlay must
call :func:`install_opt_in` before ``Worker.initialize_from_config``.  The
wrapper lets stock KV-cache initialization finish, then validates the target
runner, loads the CUDA extension, allocates the fixed capture arena, and binds
exactly 75 target ``BaseRouter.capture_fn`` callbacks before graph warmup.

The deployed GLM-5.2 stack has a separate speculator.  Only
``Worker.model_runner.static_forward_context`` is traversed; no draft object
or scheduler-wide routed-expert capturer is enabled.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import functools
import hashlib
import inspect
from pathlib import Path
import re
import threading
from typing import Any

try:
    from .target_route_capture import (
        CaptureConfig,
        CaptureProvenance,
        TargetRouteCapture,
        load_cuda_extension,
        salted_request_key,
    )
except ImportError:  # pragma: no cover - direct experiment-module deployment
    from target_route_capture import (  # type: ignore[no-redef]
        CaptureConfig,
        CaptureProvenance,
        TargetRouteCapture,
        load_cuda_extension,
        salted_request_key,
    )


# Exact deployed 2026-07-26 sources. A version drift is a hard refusal, not a
# warning, because the target/draft ownership and pre-graph timing are safety
# properties of this experiment.
DEPLOYED_WORKER_INITIALIZE_SHA256 = (
    "196bbe8208eb5ba56f0e2eb97c0d8922351f1963ac6dbd3466eae94378864ad9"
)
DEPLOYED_RUNNER_INITIALIZE_KV_SHA256 = (
    "c606851a60fef594fb231c7c68e695d3a1d52396d2e12a0304819bef8c21e808"
)
DEPLOYED_RUNNER_SAMPLE_SHA256 = (
    "4d5ce613197dfa32ab5cce9472ef966ce4bca45f8a41edc87b79527908e9b07d"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIRST_ROUTED_LAYER = 3
_LAST_ROUTED_LAYER = 77
_EXPECTED_LAYER_IDS = tuple(range(_FIRST_ROUTED_LAYER, _LAST_ROUTED_LAYER + 1))
_MTP_DRAFT_RUNTIME_LAYER_ID = 78
_MTP_DRAFT_PREFIX_LAYER_ID = 78
_LAYER_PREFIX = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


class LiveInstallError(RuntimeError):
    """The source or target-runner topology did not match the pinned seam."""


def source_sha256(function: Callable[..., Any]) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LiveInstallConfig:
    extension_name: str = "sparkring_target_route_capture_v2"
    worker_initialize_sha256: str = DEPLOYED_WORKER_INITIALIZE_SHA256
    runner_initialize_kv_sha256: str = DEPLOYED_RUNNER_INITIALIZE_KV_SHA256
    runner_sample_sha256: str = DEPLOYED_RUNNER_SAMPLE_SHA256
    capture: CaptureConfig = CaptureConfig()

    def validate(self) -> None:
        if not self.extension_name:
            raise ValueError("extension_name must be non-empty and immutable")
        if not _SHA256.fullmatch(self.worker_initialize_sha256):
            raise ValueError("worker_initialize_sha256 must be lowercase SHA-256")
        if not _SHA256.fullmatch(self.runner_initialize_kv_sha256):
            raise ValueError(
                "runner_initialize_kv_sha256 must be lowercase SHA-256"
            )
        if not _SHA256.fullmatch(self.runner_sample_sha256):
            raise ValueError("runner_sample_sha256 must be lowercase SHA-256")
        self.capture.validate()


@dataclass(frozen=True)
class LiveDependencies:
    torch_module: Any
    moe_runner_type: type
    base_router_type: type
    extension_loader: Callable[[Any, str], Any] = load_cuda_extension
    capture_factory: Callable[..., TargetRouteCapture] = TargetRouteCapture

    def validate(self) -> None:
        if not isinstance(self.moe_runner_type, type):
            raise ValueError("moe_runner_type must be a class")
        if not isinstance(self.base_router_type, type):
            raise ValueError("base_router_type must be a class")
        if not callable(self.extension_loader):
            raise ValueError("extension_loader must be callable")
        if not callable(self.capture_factory):
            raise ValueError("capture_factory must be callable")


@dataclass(frozen=True)
class _RouterBinding:
    layer_id: int
    router: Any
    previous_callback: Any


class LiveTargetRouteController:
    """Low-rate process-local control surface; never called from a layer."""

    def __init__(
        self,
        capture: TargetRouteCapture,
        bindings: tuple[_RouterBinding, ...],
        extension_handle: Any,
        runner: Any,
    ) -> None:
        self.capture = capture
        self._bindings = bindings
        self._extension_handle = extension_handle
        self._runner = runner
        self._closed = False
        self._armed = False

    @property
    def bound_layer_ids(self) -> tuple[int, ...]:
        return tuple(binding.layer_id for binding in self._bindings)

    @property
    def armed(self) -> bool:
        return self._armed

    def arm(
        self,
        *,
        request_slot: int,
        request_id: str,
        salt: bytes,
        stream_slot: int = 0,
    ) -> str:
        request_key = salted_request_key(request_id, salt)
        self.arm_salted(
            request_slot=request_slot,
            request_key=request_key,
            stream_slot=stream_slot,
        )
        return request_key

    def arm_salted(
        self,
        *,
        request_slot: int,
        request_key: str,
        stream_slot: int = 0,
    ) -> None:
        self._require_open()
        self.capture.begin_request(
            request_slot=request_slot,
            request_key=request_key,
            stream_slot=stream_slot,
            model_role="target",
        )
        self._armed = True

    def disarm(self, *, stream_slot: int = 0) -> None:
        self._require_open()
        self.capture.disarm(stream_slot=stream_slot)
        self._armed = False

    def counters(self) -> dict[str, int]:
        self._require_open()
        return self.capture.read_counters(timed_execution_complete=True)

    def record_rejection(
        self,
        num_sampled: Any,
        num_rejected: Any,
        *,
        stream_slot: int = 0,
    ) -> None:
        self._require_open()
        self.capture.record_rejection(
            num_sampled,
            num_rejected,
            stream_slot=stream_slot,
        )

    def drain(
        self,
        path: Path,
        provenance: CaptureProvenance,
        *,
        stream_slot: int = 0,
    ) -> dict[str, int]:
        self._require_open()
        self.capture.disarm(stream_slot=stream_slot)
        self._armed = False
        return self.capture.drain_jsonl(
            path, provenance, timed_execution_complete=True
        )

    def unbind(self) -> None:
        """Restore the pre-experiment callbacks after model execution stops."""

        if self._closed:
            return
        for binding in reversed(self._bindings):
            current = getattr(binding.router, "capture_fn", None)
            if not getattr(current, "_sparkring_target_route_capture", False):
                raise LiveInstallError(
                    f"layer {binding.layer_id} callback changed after binding"
                )
        for binding in reversed(self._bindings):
            binding.router.set_capture_fn(binding.previous_callback)
        self._armed = False
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise LiveInstallError("target-route controller is closed")


class SourcePinnedLiveInstaller:
    """Patch one exact Worker method and bind one exact target runner."""

    def __init__(
        self,
        *,
        worker_type: type,
        runner_type: type,
        dependencies: LiveDependencies,
        config: LiveInstallConfig,
    ) -> None:
        if not isinstance(worker_type, type) or not isinstance(runner_type, type):
            raise ValueError("worker_type and runner_type must be classes")
        config.validate()
        dependencies.validate()
        self.worker_type = worker_type
        self.runner_type = runner_type
        self.dependencies = dependencies
        self.config = config
        self._original_initialize: Callable[..., Any] | None = None
        self._original_sample: Callable[..., Any] | None = None
        self._controller: LiveTargetRouteController | None = None
        self._runner_identity: int | None = None
        self._installed = False

    @property
    def controller(self) -> LiveTargetRouteController:
        if self._controller is None:
            raise LiveInstallError(
                "capture controller is unavailable before Worker initialization"
            )
        return self._controller

    def validate_sources(
        self,
    ) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
        initialize = getattr(self.worker_type, "initialize_from_config", None)
        initialize_kv = getattr(
            self.runner_type, "initialize_kv_cache", None
        )
        sample = getattr(self.runner_type, "sample", None)
        if (
            not callable(initialize)
            or not callable(initialize_kv)
            or not callable(sample)
        ):
            raise LiveInstallError("required Worker/GPUModelRunner seam is absent")
        if getattr(initialize, "_sparkring_target_route_installer", False):
            raise LiveInstallError("Worker.initialize_from_config is already wrapped")
        if getattr(sample, "_sparkring_target_rejection_installer", False):
            raise LiveInstallError("GPUModelRunner.sample is already wrapped")
        try:
            initialize_hash = source_sha256(initialize)
            initialize_kv_hash = source_sha256(initialize_kv)
            sample_hash = source_sha256(sample)
        except (OSError, TypeError) as error:
            raise LiveInstallError("cannot inspect pinned vLLM source") from error
        if initialize_hash != self.config.worker_initialize_sha256:
            raise LiveInstallError(
                "Worker.initialize_from_config source mismatch: "
                f"expected {self.config.worker_initialize_sha256}, "
                f"got {initialize_hash}"
            )
        if initialize_kv_hash != self.config.runner_initialize_kv_sha256:
            raise LiveInstallError(
                "GPUModelRunner.initialize_kv_cache source mismatch: "
                f"expected {self.config.runner_initialize_kv_sha256}, "
                f"got {initialize_kv_hash}"
            )
        if sample_hash != self.config.runner_sample_sha256:
            raise LiveInstallError(
                "GPUModelRunner.sample source mismatch: "
                f"expected {self.config.runner_sample_sha256}, "
                f"got {sample_hash}"
            )
        return initialize, initialize_kv, sample

    def install(self) -> None:
        if self._installed:
            raise LiveInstallError("live installer is already installed")
        initialize, _, sample = self.validate_sources()

        @functools.wraps(sample)
        def wrapped_sample(runner: Any, *args: Any, **kwargs: Any) -> Any:
            result = sample(runner, *args, **kwargs)
            # Calls before Worker initialization attaches the arena are profile
            # or bootstrap work and intentionally remain inert.
            if self._controller is None:
                return result
            if self._runner_identity != id(runner):
                raise LiveInstallError(
                    "sample result came from an unbound target runner"
                )
            if not self._controller.armed:
                return result
            if len(args) >= 2:
                input_batch = args[1]
            else:
                input_batch = kwargs.get("input_batch")
            num_reqs = getattr(input_batch, "num_reqs", None)
            num_draft_tokens = getattr(
                input_batch, "num_draft_tokens", None
            )
            if (
                isinstance(num_reqs, bool)
                or not isinstance(num_reqs, int)
                or isinstance(num_draft_tokens, bool)
                or not isinstance(num_draft_tokens, int)
            ):
                raise LiveInstallError(
                    "GPUModelRunner.sample InputBatch contract changed"
                )
            if num_reqs != 1:
                raise LiveInstallError(
                    "target rejection capture requires exactly one request"
                )
            if num_draft_tokens == 0:
                return result
            if num_draft_tokens not in (4, 5):
                raise LiveInstallError(
                    "target rejection capture requires four or five draft tokens"
                )
            if not isinstance(result, tuple) or len(result) != 3:
                raise LiveInstallError(
                    "GPUModelRunner.sample no longer returns exactly three values"
                )
            self._controller.record_rejection(
                result[1],
                result[2],
                stream_slot=0,
            )
            return result

        wrapped_sample._sparkring_target_rejection_installer = True  # type: ignore[attr-defined]
        wrapped_sample._sparkring_original = sample  # type: ignore[attr-defined]

        @functools.wraps(initialize)
        def wrapped(worker: Any, *args: Any, **kwargs: Any) -> Any:
            runner_before = getattr(worker, "model_runner", None)
            if not isinstance(runner_before, self.runner_type):
                raise LiveInstallError(
                    "Worker.model_runner is not the pinned target runner type"
                )
            if bool(
                getattr(
                    getattr(runner_before, "model_config", None),
                    "enable_return_routed_experts",
                    False,
                )
            ):
                raise LiveInstallError(
                    "built-in scheduler-wide routed-expert capture must "
                    "remain disabled"
                )
            result = initialize(worker, *args, **kwargs)
            runner = getattr(worker, "model_runner", None)
            if runner is not runner_before:
                raise LiveInstallError(
                    "Worker.model_runner identity changed during initialization"
                )
            self._attach_target_runner(runner)
            return result

        wrapped._sparkring_target_route_installer = True  # type: ignore[attr-defined]
        wrapped._sparkring_original = initialize  # type: ignore[attr-defined]
        setattr(self.runner_type, "sample", wrapped_sample)
        try:
            setattr(self.worker_type, "initialize_from_config", wrapped)
        except Exception:
            setattr(self.runner_type, "sample", sample)
            raise
        self._original_initialize = initialize
        self._original_sample = sample
        self._installed = True

    def _attach_target_runner(self, runner: Any) -> None:
        if self._controller is not None:
            if self._runner_identity != id(runner):
                raise LiveInstallError(
                    "one process cannot bind two target model runners"
                )
            return
        torch = self.dependencies.torch_module
        if torch.cuda.is_current_stream_capturing():
            raise LiveInstallError("installer reached target runner during graph capture")
        modules = self._validate_target_modules(runner)

        # Loading before arena construction makes missing/invalid custom-op
        # registration fail before any callback is exposed to vLLM.
        extension = self.dependencies.extension_loader(
            torch, name=self.config.extension_name
        )
        if extension is None:
            raise LiveInstallError("CUDA extension loader returned no handle")
        if torch.cuda.is_current_stream_capturing():
            raise LiveInstallError("extension load entered CUDA graph capture")
        capture = self.dependencies.capture_factory(
            torch_module=torch,
            device=runner.device,
            config=self.config.capture,
        )
        if torch.cuda.is_current_stream_capturing():
            raise LiveInstallError("capture arena allocation entered graph capture")
        callbacks: list[tuple[Any, Any]] = []
        for layer_id, router in modules:
            callback = capture.make_base_router_callback(
                routed_layer_index=layer_id - _FIRST_ROUTED_LAYER,
                model_role="target",
                stream_slot=0,
            )
            callback._sparkring_target_route_capture = True
            callbacks.append((router, callback))

        installed: list[_RouterBinding] = []
        try:
            for (layer_id, router), (_, callback) in zip(modules, callbacks):
                previous = router.capture_fn
                binding = _RouterBinding(
                    layer_id=layer_id,
                    router=router,
                    previous_callback=previous,
                )
                installed.append(binding)
                router.set_capture_fn(callback)
        except Exception:
            for binding in reversed(installed):
                binding.router.set_capture_fn(binding.previous_callback)
            raise
        self._runner_identity = id(runner)
        self._controller = LiveTargetRouteController(
            capture, tuple(installed), extension, runner
        )

    def _validate_target_modules(self, runner: Any) -> tuple[tuple[int, Any], ...]:
        if bool(
            getattr(
                getattr(runner, "model_config", None),
                "enable_return_routed_experts",
                False,
            )
        ):
            raise LiveInstallError(
                "built-in scheduler-wide routed-expert capture must remain disabled"
            )
        compilation = getattr(runner, "compilation_config", None)
        context = getattr(compilation, "static_forward_context", None)
        if not isinstance(context, Mapping):
            raise LiveInstallError("target static_forward_context is absent")

        modules: list[tuple[int, Any]] = []
        draft_modules: list[tuple[str, int]] = []
        for prefix, module in context.items():
            if not isinstance(module, self.dependencies.moe_runner_type):
                continue
            if not isinstance(prefix, str):
                raise LiveInstallError("MoERunner context key is not a string")
            layer_id = getattr(module, "layer_id", None)
            router = getattr(module, "router", None)
            if isinstance(layer_id, bool) or not isinstance(layer_id, int):
                raise LiveInstallError("target MoERunner has a non-integer layer_id")
            prefix_match = _LAYER_PREFIX.search(prefix)
            if prefix_match is None:
                raise LiveInstallError(
                    f"MoERunner context key has no layer identity: {prefix!r}"
                )
            prefix_layer_id = int(prefix_match.group(1))
            if not isinstance(router, self.dependencies.base_router_type):
                raise LiveInstallError(
                    f"target MoERunner layer {layer_id} lacks BaseRouter"
                )
            if not hasattr(router, "capture_fn") or not callable(
                getattr(router, "set_capture_fn", None)
            ):
                raise LiveInstallError(
                    f"target BaseRouter layer {layer_id} has no reversible callback"
                )
            if router.capture_fn is not None:
                raise LiveInstallError(
                    f"target BaseRouter layer {layer_id} already has a callback"
                )
            if (
                layer_id == _MTP_DRAFT_RUNTIME_LAYER_ID
                and prefix_layer_id == _MTP_DRAFT_PREFIX_LAYER_ID
            ):
                draft_modules.append((prefix, layer_id))
                continue
            if (
                layer_id not in _EXPECTED_LAYER_IDS
                or prefix_layer_id != layer_id
            ):
                raise LiveInstallError(
                    "unexpected MoERunner ownership: "
                    f"prefix={prefix!r}, layer_id={layer_id}"
                )
            modules.append((layer_id, router))

        if len(draft_modules) != 1:
            raise LiveInstallError(
                "expected exactly one unbound MTP draft MoERunner at "
                f"prefix layer {_MTP_DRAFT_PREFIX_LAYER_ID} with runtime "
                f"layer {_MTP_DRAFT_RUNTIME_LAYER_ID}, found "
                f"{len(draft_modules)}"
            )
        observed = [layer_id for layer_id, _ in modules]
        if len(observed) != 75:
            raise LiveInstallError(
                f"expected exactly 75 target MoERunners, found {len(observed)}"
            )
        if len(set(observed)) != len(observed):
            duplicates = sorted(
                layer for layer in set(observed) if observed.count(layer) > 1
            )
            raise LiveInstallError(f"duplicate target layer IDs: {duplicates}")
        if set(observed) != set(_EXPECTED_LAYER_IDS):
            missing = sorted(set(_EXPECTED_LAYER_IDS) - set(observed))
            extra = sorted(set(observed) - set(_EXPECTED_LAYER_IDS))
            raise LiveInstallError(
                f"wrong target layer IDs; missing={missing}, extra={extra}"
            )
        return tuple(sorted(modules, key=lambda item: item[0]))

    def uninstall(self) -> None:
        if not self._installed:
            return
        current = getattr(self.worker_type, "initialize_from_config", None)
        if not getattr(current, "_sparkring_target_route_installer", False):
            raise LiveInstallError("Worker initializer changed after installation")
        current_sample = getattr(self.runner_type, "sample", None)
        if not getattr(
            current_sample, "_sparkring_target_rejection_installer", False
        ):
            raise LiveInstallError("GPUModelRunner.sample changed after installation")
        if self._controller is not None:
            self._controller.unbind()
        assert self._original_initialize is not None
        assert self._original_sample is not None
        setattr(
            self.worker_type,
            "initialize_from_config",
            self._original_initialize,
        )
        setattr(self.runner_type, "sample", self._original_sample)
        self._controller = None
        self._runner_identity = None
        self._original_initialize = None
        self._original_sample = None
        self._installed = False


_GLOBAL_LOCK = threading.Lock()
_GLOBAL_INSTALLER: SourcePinnedLiveInstaller | None = None


def install_opt_in(
    *,
    worker_type: type,
    runner_type: type,
    dependencies: LiveDependencies,
    config: LiveInstallConfig,
) -> SourcePinnedLiveInstaller:
    """Install once per worker process; never called automatically on import."""

    global _GLOBAL_INSTALLER
    with _GLOBAL_LOCK:
        if _GLOBAL_INSTALLER is not None:
            raise LiveInstallError("process-global live installer already exists")
        installer = SourcePinnedLiveInstaller(
            worker_type=worker_type,
            runner_type=runner_type,
            dependencies=dependencies,
            config=config,
        )
        installer.install()
        _GLOBAL_INSTALLER = installer
        return installer


def uninstall_opt_in() -> None:
    global _GLOBAL_INSTALLER
    with _GLOBAL_LOCK:
        if _GLOBAL_INSTALLER is None:
            return
        _GLOBAL_INSTALLER.uninstall()
        _GLOBAL_INSTALLER = None


def _global_controller() -> LiveTargetRouteController:
    if _GLOBAL_INSTALLER is None:
        raise LiveInstallError("process-global target-route capture is not installed")
    return _GLOBAL_INSTALLER.controller


def arm_capture(
    *,
    request_slot: int,
    request_id: str,
    salt: bytes,
    stream_slot: int = 0,
) -> str:
    return _global_controller().arm(
        request_slot=request_slot,
        request_id=request_id,
        salt=salt,
        stream_slot=stream_slot,
    )


def arm_capture_salted(
    *,
    request_slot: int,
    request_key: str,
    stream_slot: int = 0,
) -> None:
    _global_controller().arm_salted(
        request_slot=request_slot,
        request_key=request_key,
        stream_slot=stream_slot,
    )


def disarm_capture(*, stream_slot: int = 0) -> None:
    _global_controller().disarm(stream_slot=stream_slot)


def capture_counters() -> dict[str, int]:
    return _global_controller().counters()


def drain_capture(
    path: Path,
    provenance: CaptureProvenance,
    *,
    stream_slot: int = 0,
) -> dict[str, int]:
    return _global_controller().drain(
        path, provenance, stream_slot=stream_slot
    )
