"""Opt-in, source-pinned live installer for the Q-2R timing census.

Importing this module has no side effects. ``install()`` additionally requires
``SPARK_Q2R_PHASE_TIMING=1``. No launch or sitecustomize file imports it yet.
"""

from __future__ import annotations

import functools
import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .draft_step_timing import (
    DraftGenerationOrdinals,
    PinnedDraftLoopAdapter,
)
from .manager_roles import ManagerRole, ManagerRoleRegistry
from .phase_timing import (
    DrainResult,
    PhaseDescriptor,
    PhaseKind,
    PhaseTimingCollector,
)
from .vllm_adapter import (
    AdapterValidationError,
    FailClosedMethodAdapter,
    MethodHook,
    source_sha256,
)

_EXPECTED_VLLM_VERSION = (
    "0.11.2.dev279+eldritch.final.fcc6141.b12x284a2ea."
    "fi25dd814.cu132.20260626"
)
_CUDA_INIT_SHA256 = (
    "48af9e1aa167af1c0914dd96c2d686a73c2e95c29095b5ae2901356629f0d01d"
)
_RUN_FULLGRAPH_SHA256 = (
    "4d58b8ef1a5023af0c11eb7a659620faca15f8a0303b37774ed0d28f4a5919db"
)
_RUN_PW_GRAPH_SHA256 = (
    "d515cf6e2e5b9e9fd4516c93b9028b16788a213b5529d0304a3792a4d49833a8"
)
_MODEL_RUN_FULLGRAPH_SHA256 = (
    "9601425569906383209d353ff82e2fa49bafdd3778f9ee9cf094dec29f1a33ed"
)
_AUTOREGRESSIVE_INIT_SHA256 = (
    "f17ff547954777855bb33cd9796a1c82e6d412de859c5cd5dfd481af147304f6"
)
_INITIALIZE_KV_SHA256 = (
    "c606851a60fef594fb231c7c68e695d3a1d52396d2e12a0304819bef8c21e808"
)
_EXECUTE_MODEL_SHA256 = (
    "f04573f6477367594a977c54dc5048d10ad7aa4364fe85642fa56e657ea2e081"
)
_SAMPLE_TOKENS_SHA256 = (
    "7125f499709171ea88e141c72ec98330cac1ab82fd64876ccf683a28d7daf951"
)
_MULTI_STEP_DECODE_SHA256 = (
    "0e7c45ef90ae463db0a3bd4a9142301c9a2261d11837fac1215eb538d4e1ac26"
)
_GENERATE_DRAFT_SHA256 = (
    "0f0df160847853d7e359d86f9b4de05863eff5e596f0557aa03c2279a53afdda"
)
_SUPPORTED_SPECULATIVE_STEPS = frozenset((4, 5))
_DEFAULT_CAPACITY = 4096

_TARGET_FORWARD = PhaseDescriptor(PhaseKind.STEP_ENVELOPE, "execute_model")
_SAMPLE_AND_DRAFT = PhaseDescriptor(PhaseKind.STEP_ENVELOPE, "sample_tokens")
_UNBOUND_FULL = PhaseDescriptor(
    PhaseKind.OTHER_GRAPH, "unbound-manager,method=run_fullgraph"
)
_UNBOUND_PW = PhaseDescriptor(
    PhaseKind.OTHER_GRAPH, "unbound-manager,method=run_pw_graph"
)


@dataclass(frozen=True)
class LivePins:
    vllm_version: str = _EXPECTED_VLLM_VERSION
    cuda_init: str = _CUDA_INIT_SHA256
    run_fullgraph: str = _RUN_FULLGRAPH_SHA256
    run_pw_graph: str = _RUN_PW_GRAPH_SHA256
    model_run_fullgraph: str = _MODEL_RUN_FULLGRAPH_SHA256
    autoregressive_init: str = _AUTOREGRESSIVE_INIT_SHA256
    initialize_kv_cache: str = _INITIALIZE_KV_SHA256
    execute_model: str = _EXECUTE_MODEL_SHA256
    sample_tokens: str = _SAMPLE_TOKENS_SHA256
    multi_step_decode: str = _MULTI_STEP_DECODE_SHA256
    generate_draft: str = _GENERATE_DRAFT_SHA256


@dataclass(frozen=True)
class LiveTypes:
    cuda_graph_manager: type
    model_cuda_graph_manager: type
    autoregressive_speculator: type
    initialization_runner: type
    execution_runner: type


@dataclass(frozen=True)
class SpeculativeDepthAttestation:
    configured_speculative_steps: int
    attested_round_depths: tuple[int, ...]
    adaptive_window: int | None


@dataclass(frozen=True)
class _BindingHook:
    owner: type
    method_name: str
    expected_hash: str
    bind_after_call: Callable[
        [Any, tuple[Any, ...], dict[str, Any], Any], None
    ]


class _PinnedBindingAdapter:
    """One source-pinned post-call ownership seam."""

    def __init__(self, hook: _BindingHook) -> None:
        self._hook = hook
        self._original: Callable[..., Any] | None = None
        self._installed = False

    def validate(self) -> Callable[..., Any]:
        original = getattr(self._hook.owner, self._hook.method_name, None)
        if original is None or not callable(original):
            raise AdapterValidationError("binding owner method is absent")
        if getattr(original, "_spark_q2r_manager_binding", False):
            raise AdapterValidationError("binding owner is already wrapped")
        actual = source_sha256(original)
        if actual != self._hook.expected_hash:
            raise AdapterValidationError(
                f"source mismatch for {self._hook.owner.__qualname__}."
                f"{self._hook.method_name}: expected "
                f"{self._hook.expected_hash}, got {actual}"
            )
        return original

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("binding adapter is already installed")
        original = self.validate()
        hook = self._hook

        @functools.wraps(original)
        def wrapped(
            instance: Any, *args: Any, **kwargs: Any
        ) -> Any:
            result = original(instance, *args, **kwargs)
            hook.bind_after_call(instance, args, kwargs, result)
            return result

        wrapped._spark_q2r_manager_binding = True  # type: ignore[attr-defined]
        wrapped._spark_original = original  # type: ignore[attr-defined]
        setattr(hook.owner, hook.method_name, wrapped)
        self._original = original
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        current = getattr(self._hook.owner, self._hook.method_name)
        if not getattr(current, "_spark_q2r_manager_binding", False):
            raise AdapterValidationError(
                "binding owner changed after installation"
            )
        assert self._original is not None
        setattr(
            self._hook.owner, self._hook.method_name, self._original
        )
        self._original = None
        self._installed = False


def _assert_source(owner: type, method: str, expected: str) -> None:
    function = getattr(owner, method, None)
    if function is None or not callable(function):
        raise AdapterValidationError(
            f"{owner.__qualname__}.{method} is absent"
        )
    actual = source_sha256(function)
    if actual != expected:
        raise AdapterValidationError(
            f"source mismatch for {owner.__qualname__}.{method}: "
            f"expected {expected}, got {actual}"
        )


def _manager_query_len(manager: Any, fallback: Any = None) -> int:
    value = getattr(manager, "decode_query_len", fallback)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RuntimeError("graph manager has no valid decode_query_len")
    return value


class LiveQ2RSession:
    """Installed-but-unarmed live recorder with explicit lifecycle."""

    def __init__(
        self,
        *,
        types: LiveTypes,
        pins: LivePins,
        event_factory: Callable[[], Any],
        current_stream: Callable[[Any], Any],
        capacity: int,
        expected_speculative_steps: int = 5,
        attested_round_depths: tuple[int, ...] | None = None,
        adaptive_window: int | None = None,
        nvtx: Any = None,
    ) -> None:
        self._types = types
        self._pins = pins
        self._registry = ManagerRoleRegistry()
        self._draft_ordinals = DraftGenerationOrdinals(
            expected_speculative_steps,
            attested_round_depths=attested_round_depths,
            adaptive_window=adaptive_window,
        )
        # Event allocation is complete before any source is patched.
        self._collector = PhaseTimingCollector(
            event_factory=event_factory,
            capacity=capacity,
            descriptors=(
                _TARGET_FORWARD,
                _SAMPLE_AND_DRAFT,
                _UNBOUND_FULL,
                _UNBOUND_PW,
                *self._draft_ordinals.descriptors,
            ),
            nvtx=nvtx,
        )
        self._current_stream = current_stream
        self._descriptors_finalized = False
        self._installed = False

        def bind_managers(
            runner: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            result: Any,
        ) -> None:
            del args, kwargs, result
            target = getattr(runner, "cudagraph_manager", None)
            speculator = getattr(runner, "speculator", None)
            draft_prefill = getattr(
                speculator, "prefill_cudagraph_manager", None
            )
            draft_decode = getattr(
                speculator, "decode_cudagraph_manager", None
            )
            managers = (target, draft_prefill, draft_decode)
            if (
                any(manager is None for manager in managers)
                or len({id(manager) for manager in managers}) != 3
            ):
                raise RuntimeError(
                    "initialize_kv_cache did not expose distinct target, "
                    "MTP draft-prefill, and MTP draft-decode graph managers"
                )
            self._draft_ordinals.bind(speculator, draft_decode)
            self._registry.register(
                target,
                decode_query_len=_manager_query_len(target),
                role=ManagerRole.TARGET_VERIFY,
            )
            self._registry.register(
                draft_prefill,
                decode_query_len=_manager_query_len(draft_prefill),
                role=ManagerRole.DRAFT_PREFILL,
            )
            self._registry.register(
                draft_decode,
                decode_query_len=_manager_query_len(draft_decode),
                role=ManagerRole.DRAFT_DECODE,
            )

        self._binding_adapter = _PinnedBindingAdapter(
            _BindingHook(
                owner=types.initialization_runner,
                method_name="initialize_kv_cache",
                expected_hash=pins.initialize_kv_cache,
                bind_after_call=bind_managers,
            )
        )

        def graph_descriptor(
            method: str,
        ) -> Callable[
            [Any, tuple[Any, ...], dict[str, Any]], PhaseDescriptor
        ]:
            fallback = (
                _UNBOUND_FULL if method == "run_fullgraph" else _UNBOUND_PW
            )

            def resolve(
                instance: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
            ) -> PhaseDescriptor:
                del args, kwargs
                generation = self._draft_ordinals.graph_descriptor(instance)
                if generation is not None:
                    return generation
                try:
                    return self._registry.graph_descriptor(
                        instance, graph_method=method
                    )
                except RuntimeError:
                    return fallback

            return resolve

        def graph_stream(
            instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            del args, kwargs
            return self._current_stream(instance.device)

        def execution_stream(
            instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            del args, kwargs
            return self._current_stream(instance.device)

        self._timing_adapter = FailClosedMethodAdapter(
            self._collector,
            (
                MethodHook(
                    owner=types.cuda_graph_manager,
                    method_name="run_fullgraph",
                    expected_source_sha256=pins.run_fullgraph,
                    descriptor=graph_descriptor("run_fullgraph"),
                    stream_for_call=graph_stream,
                ),
                MethodHook(
                    owner=types.cuda_graph_manager,
                    method_name="run_pw_graph",
                    expected_source_sha256=pins.run_pw_graph,
                    descriptor=graph_descriptor("run_pw_graph"),
                    stream_for_call=graph_stream,
                ),
                MethodHook(
                    owner=types.execution_runner,
                    method_name="execute_model",
                    expected_source_sha256=pins.execute_model,
                    descriptor=_TARGET_FORWARD,
                    stream_for_call=execution_stream,
                ),
                MethodHook(
                    owner=types.execution_runner,
                    method_name="sample_tokens",
                    expected_source_sha256=pins.sample_tokens,
                    descriptor=_SAMPLE_AND_DRAFT,
                    stream_for_call=execution_stream,
                ),
                MethodHook(
                    owner=types.autoregressive_speculator,
                    method_name="_generate_draft",
                    expected_source_sha256=pins.generate_draft,
                    descriptor=self._draft_ordinals.eager_descriptor,
                    stream_for_call=execution_stream,
                ),
            ),
        )
        self._draft_loop_adapter = PinnedDraftLoopAdapter(
            owner=types.autoregressive_speculator,
            expected_source_sha256=pins.multi_step_decode,
            ordinals=self._draft_ordinals,
        )

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("live session is already installed")
        # Validate every observed or wrapped source before the first mutation.
        _assert_source(
            self._types.cuda_graph_manager,
            "__init__",
            self._pins.cuda_init,
        )
        _assert_source(
            self._types.autoregressive_speculator,
            "init_cudagraph_manager",
            self._pins.autoregressive_init,
        )
        # The deployed override calls ``super().run_fullgraph``.  Pin that
        # fact, but wrap only the base method: wrapping both methods records
        # two nested event pairs for one target replay.
        _assert_source(
            self._types.model_cuda_graph_manager,
            "run_fullgraph",
            self._pins.model_run_fullgraph,
        )
        self._binding_adapter.validate()
        self._timing_adapter.validate()
        self._draft_loop_adapter.validate()
        self._binding_adapter.install()
        try:
            self._timing_adapter.install()
            try:
                self._draft_loop_adapter.install()
            except Exception:
                self._timing_adapter.uninstall()
                raise
        except Exception:
            self._binding_adapter.uninstall()
            raise
        self._installed = True

    def _finalize_descriptors(self) -> None:
        if self._descriptors_finalized:
            return
        entries = self._registry.manager_entries()
        roles = [identity.role for _manager, identity in entries]
        if roles.count(ManagerRole.TARGET_VERIFY) != 1:
            raise RuntimeError(
                "expected exactly one explicitly bound target manager"
            )
        if roles.count(ManagerRole.DRAFT_PREFILL) != 1:
            raise RuntimeError(
                "expected exactly one explicitly bound MTP draft-prefill "
                "manager"
            )
        if roles.count(ManagerRole.DRAFT_DECODE) != 1:
            raise RuntimeError(
                "expected exactly one explicitly bound MTP draft-decode "
                "manager"
            )
        if any(role is ManagerRole.UNKNOWN for role in roles):
            raise RuntimeError("unknown graph-manager role blocks arming")
        descriptors = tuple(
            self._registry.graph_descriptor(
                manager, graph_method=method
            )
            for manager, _identity in entries
            for method in ("run_fullgraph", "run_pw_graph")
        )
        self._collector.register_descriptors(descriptors)
        self._descriptors_finalized = True

    def arm(self, epoch: str) -> None:
        if not self._installed:
            raise RuntimeError("install before arm")
        self._finalize_descriptors()
        self._collector.arm(epoch)
        try:
            self._draft_ordinals.arm()
        except Exception:
            self._collector.disarm()
            raise

    def disarm(self) -> None:
        self._collector.disarm()
        self._draft_ordinals.disarm()

    def drain(self) -> DrainResult:
        return self._collector.drain()

    def snapshot(self) -> dict[str, Any]:
        # Status reporting starts before the first explicit arm.  Once
        # initialize_kv_cache has bound all three managers, snapshots need the
        # same finite descriptor registry that arm() would create; otherwise
        # looking up graph counters below fails with a KeyError and makes the
        # control bridge appear disabled.
        self._finalize_descriptors()
        phase_timing = self._collector.snapshot()
        descriptor_metrics = phase_timing["descriptors"]
        target_forward_samples = int(
            descriptor_metrics[_TARGET_FORWARD.key]["count"]
        )
        sample_and_draft_samples = int(
            descriptor_metrics[_SAMPLE_AND_DRAFT.key]["count"]
        )

        graph_counts = {
            "target_verify": 0,
            "draft_prefill": 0,
            "draft_decode": 0,
        }
        for manager, identity in self._registry.manager_entries():
            if identity.role.value not in graph_counts:
                continue
            graph_counts[identity.role.value] += sum(
                int(
                    descriptor_metrics[
                        self._registry.graph_descriptor(
                            manager, graph_method=method
                        ).key
                    ]["count"]
                )
                for method in ("run_fullgraph", "run_pw_graph")
            )
        step_samples = (
            target_forward_samples + sample_and_draft_samples
        )
        return {
            "phase_timing": phase_timing,
            "manager_roles": self._registry.snapshot(),
            "coverage": {
                "step_envelope": {
                    "implemented": True,
                    "samples": step_samples,
                    "components": {
                        "execute_model": target_forward_samples,
                        "sample_tokens": sample_and_draft_samples,
                    },
                    "additive": True,
                },
                "graph_methods": {
                    "implemented": True,
                    "methods": ["run_fullgraph", "run_pw_graph"],
                    "counts": graph_counts,
                },
                "draft_decode_generation": {
                    "implemented": True,
                    **self._draft_ordinals.snapshot(),
                    "observed": {
                        descriptor.key: int(
                            descriptor_metrics[descriptor.key]["count"]
                        )
                        for descriptor in self._draft_ordinals.descriptors
                    },
                },
                "eager_transition": {
                    "implemented": False,
                    "samples": 0,
                },
                "collective_boundary": {
                    "implemented": False,
                    "samples": 0,
                },
            },
        }

    def reset(self) -> None:
        self._collector.reset()
        self._draft_ordinals.reset()

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._draft_loop_adapter.uninstall()
        self._timing_adapter.uninstall()
        self._binding_adapter.uninstall()
        self._installed = False


def _load_types() -> LiveTypes:
    cuda_module = importlib.import_module(
        "vllm.v1.worker.gpu.cudagraph_utils"
    )
    autoregressive_module = importlib.import_module(
        "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator"
    )
    initialization_module = importlib.import_module(
        "vllm.v1.worker.gpu.model_runner"
    )
    return LiveTypes(
        cuda_graph_manager=cuda_module.CudaGraphManager,
        model_cuda_graph_manager=cuda_module.ModelCudaGraphManager,
        autoregressive_speculator=(
            autoregressive_module.AutoRegressiveSpeculator
        ),
        initialization_runner=initialization_module.GPUModelRunner,
        execution_runner=initialization_module.GPUModelRunner,
    )


def _capacity() -> int:
    text = os.getenv("SPARK_Q2R_PHASE_TIMING_CAPACITY", str(_DEFAULT_CAPACITY))
    try:
        return int(text)
    except ValueError as error:
        raise RuntimeError(
            "SPARK_Q2R_PHASE_TIMING_CAPACITY must be an integer"
        ) from error


def _expected_speculative_steps() -> int:
    text = os.getenv("VLLM_SPARK_MTP_TOKENS", "")
    try:
        value = int(text)
    except ValueError as error:
        raise RuntimeError(
            "VLLM_SPARK_MTP_TOKENS must be 4 or 5"
        ) from error
    if value not in _SUPPORTED_SPECULATIVE_STEPS:
        raise RuntimeError("VLLM_SPARK_MTP_TOKENS must be 4 or 5")
    return value


def _depth_attestation() -> SpeculativeDepthAttestation:
    configured = _expected_speculative_steps()
    window_text = os.getenv("VLLM_SPARK_MTP_ADAPTIVE_WINDOW", "0")
    try:
        window = int(window_text)
    except ValueError as error:
        raise RuntimeError(
            "VLLM_SPARK_MTP_ADAPTIVE_WINDOW must be 0 or 32"
        ) from error
    if window == 0:
        return SpeculativeDepthAttestation(
            configured_speculative_steps=configured,
            attested_round_depths=(configured,),
            adaptive_window=None,
        )
    if window != 32:
        raise RuntimeError(
            "VLLM_SPARK_MTP_ADAPTIVE_WINDOW must be 0 or 32"
        )
    if configured != 4:
        raise RuntimeError(
            "adaptive window 32 requires VLLM_SPARK_MTP_TOKENS=4"
        )
    raw_depths = os.getenv("VLLM_ADAPTIVE_SPEC_DEPTHS", "")
    try:
        parsed = tuple(
            int(item.strip()) for item in raw_depths.split(",")
        )
    except ValueError as error:
        raise RuntimeError(
            "VLLM_ADAPTIVE_SPEC_DEPTHS must attest exactly 2,4"
        ) from error
    if parsed != (2, 4):
        raise RuntimeError(
            "VLLM_ADAPTIVE_SPEC_DEPTHS must attest exactly 2,4"
        )
    return SpeculativeDepthAttestation(
        configured_speculative_steps=configured,
        attested_round_depths=parsed,
        adaptive_window=window,
    )


_session: LiveQ2RSession | None = None


def install() -> None:
    """Validate pins, preallocate events, then install all live hooks."""
    global _session
    if os.getenv("SPARK_Q2R_PHASE_TIMING") != "1":
        raise RuntimeError("SPARK_Q2R_PHASE_TIMING=1 is required")
    if _session is not None:
        raise RuntimeError("Q-2R live timing is already installed")
    import torch
    import vllm

    if vllm.__version__ != _EXPECTED_VLLM_VERSION:
        raise RuntimeError(
            f"unsupported vLLM version: {vllm.__version__}"
        )
    nvtx = (
        torch.cuda.nvtx
        if os.getenv("SPARK_Q2R_PHASE_TIMING_NVTX") == "1"
        else None
    )
    depth = _depth_attestation()
    candidate = LiveQ2RSession(
        types=_load_types(),
        pins=LivePins(),
        event_factory=lambda: torch.cuda.Event(enable_timing=True),
        current_stream=torch.cuda.current_stream,
        capacity=_capacity(),
        expected_speculative_steps=depth.configured_speculative_steps,
        attested_round_depths=depth.attested_round_depths,
        adaptive_window=depth.adaptive_window,
        nvtx=nvtx,
    )
    candidate.install()
    _session = candidate


def _required_session() -> LiveQ2RSession:
    if _session is None:
        raise RuntimeError("Q-2R live timing is not installed")
    return _session


def arm(epoch: str) -> None:
    _required_session().arm(epoch)


def disarm() -> None:
    _required_session().disarm()


def drain() -> DrainResult:
    return _required_session().drain()


def snapshot() -> dict[str, Any]:
    return _required_session().snapshot()


def reset() -> None:
    _required_session().reset()
