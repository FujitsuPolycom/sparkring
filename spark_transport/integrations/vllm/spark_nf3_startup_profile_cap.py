"""Bound NF3 startup compilation without shrinking runtime batching.

vLLM normally profiles the language model at ``max_num_batched_tokens`` even
when ``kv_cache_memory_bytes`` is explicitly configured.  That is useful for
compilation, but the GLM-5.2 NF3 large-M path can make a Q4096 startup profile
starve GB10 unified-memory hosts for many minutes.

This opt-in adapter temporarily caps ``GPUModelRunner.max_num_tokens`` while
``profile_run`` executes.  When the separately gated single-range contract is
enabled, it also caps the scheduler's two token budgets at the explicitly
selected Q32 or Q40 graph ceiling while
``Worker._compile_or_warm_up_model_impl`` executes.  This covers kernel
warmups that read scheduler limits directly instead of using the profile
dummy width.  Every value is restored in a ``finally`` block before serving.

The cap is admitted only for the exact audited vLLM method and only when the
operator has explicitly sized the KV cache.  Using a smaller profile to infer
available KV memory would be unsafe and therefore fails closed.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)

_CAP_ENV = "VLLM_SPARK_NF3_STARTUP_PROFILE_MAX_TOKENS"
_SINGLE_RANGE_ENV = "VLLM_SPARK_NF3_SINGLE_COMPILE_RANGE"
_GRAPH_MAX_ENV = "VLLM_SPARK_MAX_CUDAGRAPH_CAPTURE_SIZE"
_V2_RUNNER_ENV = "VLLM_USE_V2_MODEL_RUNNER"
_PATCH_MARKER = "_spark_nf3_startup_profile_cap"
_WORKER_PATCH_MARKER = "_spark_nf3_compile_warmup_cap"
_MEMORY_GUARD_PATCH_MARKER = "_spark_nf3_profile_ownership_guard"
_REFERENCE_RUNTIME_MAX_TOKENS = 4096
_ALLOWED_RUNTIME_MAX_TOKENS = frozenset({3072, 4096})
_ALLOWED_GRAPH_MAX_TOKENS = frozenset({32, 40})
_EXPECTED_VERSION = (
    "0.11.2.dev279+eldritch.final.fcc6141.b12x284a2ea."
    "fi25dd814.cu132.20260626"
)
_EXPECTED_PROFILE_RUN_SHA256 = (
    "74c286296e4c2b5c759c00bb8823844ed3d857b0ec70bdf5407613c2c1f874e8"
)
_EXPECTED_V2_PROFILE_RUN_SHA256 = (
    "c291e403266f06fbbf01a1f41a8f7385365141f77bc98519962b6776ec31a0a4"
)
_EXPECTED_COMPILE_WARMUP_SHA256 = (
    "8ce16f1da9f32c8701631fba3d8413bddebd8c5e5a909ccc11c227a44bbe00f6"
)
_EXPECTED_DETERMINE_MEMORY_SHA256 = (
    "4bddf1b007123490cdd403ac3e6e52596e0d1d3ca20444b53a3962947a573bd7"
)


@dataclass(frozen=True)
class _RuntimeBindings:
    version: str
    runner_cls: type[Any]
    runner_kind: str
    v1_runner_cls: type[Any]
    v2_runner_cls: type[Any]
    worker_cls: type[Any]


_LOCK = threading.RLock()
_INSTALLED = False
_PROFILE_CALLS = 0
_CAPPED_CALLS = 0
_LAST_RUNTIME_MAX: int | None = None
_LAST_PROFILE_MAX: int | None = None
_LAST_ELAPSED_SECONDS: float | None = None
_ORIGINAL: Callable[..., Any] | None = None
_RUNNER_KIND: str | None = None
_V1_RUNNER_OWNED = False
_V2_RUNNER_OWNED = False
_WORKER_INSTALLED = False
_WORKER_CALLS = 0
_LAST_WORKER_RUNTIME_BATCHED: int | None = None
_LAST_WORKER_RUNTIME_SCHEDULED: int | None = None
_WORKER_ORIGINAL: Callable[..., Any] | None = None
_MEMORY_GUARD_INSTALLED = False
_MEMORY_GUARD_CALLS = 0
_MEMORY_GUARD_ORIGINAL: Callable[..., Any] | None = None


def _single_range_contract_enabled() -> bool:
    raw = os.getenv(_SINGLE_RANGE_ENV, "0")
    if raw not in {"0", "1"}:
        raise RuntimeError(f"{_SINGLE_RANGE_ENV} must be 0 or 1, got {raw!r}")
    return raw == "1"


def _graph_max_tokens() -> int:
    raw = os.getenv(_GRAPH_MAX_ENV, "32")
    if not raw.isdecimal():
        raise RuntimeError(
            f"{_GRAPH_MAX_ENV} must select Q32 or Q40, got {raw!r}"
        )
    value = int(raw)
    if value not in _ALLOWED_GRAPH_MAX_TOKENS:
        raise RuntimeError(
            f"{_GRAPH_MAX_ENV} must select Q32 or Q40, got Q{value}"
        )
    return value


def _range_bounds(value: Any) -> tuple[int, int]:
    try:
        return int(value.start), int(value.end)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"unexpected vLLM compile-range value: {value!r}"
        ) from error


def _validate_single_compile_range_contract(
    runner: Any,
    runtime_max: int,
    profile_max: int,
) -> None:
    graph_max = _graph_max_tokens()
    if profile_max < 2:
        raise RuntimeError(
            "NF3 single-range contract rejects a singleton Q1 startup "
            "profile: PyTorch specializes dynamic integers with value 0 or "
            "1, which removes the symbolic Q argument from the compiled "
            "artifact ABI. Use profile tokens >=2 for the Q1-Q4096 range."
        )
    if runtime_max not in _ALLOWED_RUNTIME_MAX_TOKENS:
        raise RuntimeError(
            "NF3 single-range contract requires runtime max tokens in "
            f"{sorted(_ALLOWED_RUNTIME_MAX_TOKENS)}, got {runtime_max}"
        )
    scheduler_max = int(runner.scheduler_config.max_num_batched_tokens)
    if scheduler_max != runtime_max:
        raise RuntimeError(
            "NF3 single-range contract requires scheduler and runner token "
            f"limits to match, got scheduler={scheduler_max}, "
            f"runner={runtime_max}"
        )

    compilation_config = runner.compilation_config
    endpoints = [
        int(value)
        for value in (compilation_config.compile_ranges_endpoints or [])
    ]
    if endpoints != [runtime_max]:
        raise RuntimeError(
            "NF3 single-range contract requires normalized Inductor endpoints "
            f"[{runtime_max}] with no interior split; got {endpoints}"
        )
    ranges = [
        _range_bounds(value)
        for value in compilation_config.get_compile_ranges()
    ]
    if ranges != [(1, runtime_max)]:
        raise RuntimeError(
            "NF3 single-range contract requires compile range "
            f"[(1, {runtime_max})]; got {ranges}"
        )

    capture_sizes = [
        int(value)
        for value in (compilation_config.cudagraph_capture_sizes or [])
    ]
    if (
        not capture_sizes
        or min(capture_sizes) < 1
        or max(capture_sizes) > graph_max
    ):
        raise RuntimeError(
            "NF3 single-range contract requires CUDA graph captures within "
            f"Q1-Q{graph_max}; got {capture_sizes}"
        )
    if (
        int(compilation_config.max_cudagraph_capture_size)
        != graph_max
    ):
        raise RuntimeError(
            "NF3 single-range contract requires max CUDA graph capture "
            f"Q{graph_max}"
        )
    oversized_compile_sizes = [
        int(value)
        for value in (compilation_config.compile_sizes or [])
        if isinstance(value, int)
        and int(value) > graph_max
    ]
    if oversized_compile_sizes:
        raise RuntimeError(
            "NF3 single-range contract forbids startup compile sizes above "
            f"Q{graph_max}; got {oversized_compile_sizes}"
        )


def _runner_kind_from_environment() -> str:
    raw = os.getenv(_V2_RUNNER_ENV)
    if raw not in {"0", "1"}:
        raise RuntimeError(
            f"{_V2_RUNNER_ENV} must be explicitly 0 or 1 for the NF3 "
            f"profile cap; got {raw!r}"
        )
    return "v2" if raw == "1" else "v1"


def _load_bindings() -> _RuntimeBindings:
    import vllm
    from vllm.v1.worker.gpu.model_runner import (
        GPUModelRunner as V2GPUModelRunner,
    )
    from vllm.v1.worker.gpu_worker import Worker
    from vllm.v1.worker.gpu_model_runner import (
        GPUModelRunner as V1GPUModelRunner,
    )

    runner_kind = _runner_kind_from_environment()
    runner_cls = (
        V2GPUModelRunner if runner_kind == "v2" else V1GPUModelRunner
    )

    return _RuntimeBindings(
        version=vllm.__version__,
        runner_cls=runner_cls,
        runner_kind=runner_kind,
        v1_runner_cls=V1GPUModelRunner,
        v2_runner_cls=V2GPUModelRunner,
        worker_cls=Worker,
    )


def _source_sha256(value: object) -> str:
    return hashlib.sha256(inspect.getsource(value).encode("utf-8")).hexdigest()


def _parse_cap() -> int:
    raw = os.getenv(_CAP_ENV)
    if raw is None or raw == "":
        raise RuntimeError(f"{_CAP_ENV} must be set when the hook is installed")
    if not raw.isdecimal():
        raise RuntimeError(f"{_CAP_ENV} must be a positive integer, got {raw!r}")
    cap = int(raw)
    if cap < 1 or cap > 4096:
        raise RuntimeError(f"{_CAP_ENV} must be in [1, 4096], got {cap}")
    return cap


def _attest_method(
    value: object,
    *,
    expected_sha256: str,
    expected_parameters: tuple[str, ...],
    label: str,
) -> None:
    actual = _source_sha256(value)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} source attestation failed: "
            f"expected {expected_sha256}, got {actual}"
        )
    parameters = tuple(inspect.signature(value).parameters)
    if parameters != expected_parameters:
        raise RuntimeError(
            f"{label} ABI changed: expected {expected_parameters}, "
            f"got {parameters}"
        )


def _attest(bindings: _RuntimeBindings) -> None:
    if bindings.version != _EXPECTED_VERSION:
        raise RuntimeError(
            "NF3 startup-profile cap does not recognize vLLM version "
            f"{bindings.version!r}"
        )
    _attest_method(
        bindings.runner_cls.profile_run,
        expected_sha256=(
            _EXPECTED_V2_PROFILE_RUN_SHA256
            if bindings.runner_kind == "v2"
            else _EXPECTED_PROFILE_RUN_SHA256
        ),
        expected_parameters=("self",),
        label=f"{bindings.runner_kind.upper()} GPUModelRunner.profile_run",
    )
    _attest_method(
        bindings.worker_cls.determine_available_memory,
        expected_sha256=_EXPECTED_DETERMINE_MEMORY_SHA256,
        expected_parameters=("self",),
        label="Worker.determine_available_memory",
    )
    if _single_range_contract_enabled():
        _attest_method(
            bindings.worker_cls._compile_or_warm_up_model_impl,
            expected_sha256=_EXPECTED_COMPILE_WARMUP_SHA256,
            expected_parameters=("self",),
            label="Worker._compile_or_warm_up_model_impl",
        )


def _install_on_class(
    runner_cls: type[Any],
    cap: int,
    *,
    runner_kind: str = "direct",
) -> bool:
    global _INSTALLED, _ORIGINAL

    with _LOCK:
        current = runner_cls.profile_run
        existing_cap = getattr(current, _PATCH_MARKER, None)
        if existing_cap is not None:
            if existing_cap != cap:
                raise RuntimeError(
                    "NF3 startup-profile cap is already installed with "
                    f"{existing_cap}, refusing requested {cap}"
                )
            _INSTALLED = True
            return False

        original = current

        @functools.wraps(original)
        def capped_profile_run(self: Any) -> Any:
            global _PROFILE_CALLS, _CAPPED_CALLS
            global _LAST_RUNTIME_MAX, _LAST_PROFILE_MAX
            global _LAST_ELAPSED_SECONDS

            runtime_max = int(self.max_num_tokens)
            if _single_range_contract_enabled():
                _validate_single_compile_range_contract(
                    self,
                    runtime_max,
                    cap,
                )
            with _LOCK:
                _PROFILE_CALLS += 1
                _LAST_RUNTIME_MAX = runtime_max

            if cap >= runtime_max:
                with _LOCK:
                    _LAST_PROFILE_MAX = runtime_max
                return original(self)

            cache_bytes = getattr(
                getattr(self, "cache_config", None),
                "kv_cache_memory_bytes",
                None,
            )
            if not cache_bytes or int(cache_bytes) <= 0:
                raise RuntimeError(
                    "NF3 startup-profile cap requires explicit positive "
                    "kv_cache_memory_bytes; refusing unsafe memory inference"
                )

            with _LOCK:
                _CAPPED_CALLS += 1
                _LAST_PROFILE_MAX = cap

            LOGGER.warning(
                "Spark NF3 startup profile capped: runner_kind=%s "
                "runtime_max_tokens=%d "
                "profile_tokens=%d",
                runner_kind,
                runtime_max,
                cap,
            )
            started = time.monotonic()
            self.max_num_tokens = cap
            try:
                return original(self)
            finally:
                self.max_num_tokens = runtime_max
                elapsed = time.monotonic() - started
                with _LOCK:
                    _LAST_ELAPSED_SECONDS = elapsed
                LOGGER.warning(
                    "Spark NF3 startup profile completed: runner_kind=%s "
                    "profile_tokens=%d "
                    "runtime_max_tokens_restored=%d elapsed_seconds=%.3f",
                    runner_kind,
                    cap,
                    runtime_max,
                    elapsed,
                )

        setattr(capped_profile_run, _PATCH_MARKER, cap)
        runner_cls.profile_run = capped_profile_run
        _ORIGINAL = original
        _INSTALLED = True
        return True


def _install_memory_ownership_guard(
    worker_cls: type[Any],
    *,
    runner_cls: type[Any],
    runner_kind: str,
    cap: int,
) -> bool:
    global _MEMORY_GUARD_INSTALLED, _MEMORY_GUARD_ORIGINAL

    with _LOCK:
        current = worker_cls.determine_available_memory
        existing = getattr(current, _MEMORY_GUARD_PATCH_MARKER, None)
        requested = (runner_kind, cap)
        if existing is not None:
            if existing != requested:
                raise RuntimeError(
                    "NF3 profile ownership guard is already installed with "
                    f"{existing}, refusing {requested}"
                )
            _MEMORY_GUARD_INSTALLED = True
            return False

        original = current

        @functools.wraps(original)
        def guarded_determine_available_memory(self: Any) -> Any:
            global _MEMORY_GUARD_CALLS

            configured_v2 = bool(self.use_v2_model_runner)
            expected_v2 = runner_kind == "v2"
            if configured_v2 != expected_v2:
                raise RuntimeError(
                    "NF3 profile cap runner selection disagrees with Worker: "
                    f"environment={runner_kind}, "
                    f"worker_use_v2={configured_v2}"
                )

            actual_runner = self.model_runner
            if type(actual_runner) is not runner_cls:
                raise RuntimeError(
                    "NF3 profile cap refuses unexpected model runner type: "
                    f"expected={runner_cls.__module__}.{runner_cls.__name__}, "
                    f"got={type(actual_runner).__module__}."
                    f"{type(actual_runner).__name__}"
                )
            owned_cap = getattr(
                type(actual_runner).profile_run,
                _PATCH_MARKER,
                None,
            )
            if owned_cap != cap:
                raise RuntimeError(
                    "NF3 profile cap does not own the selected runner at "
                    "memory-profile entry: "
                    f"runner={runner_kind}, expected_cap={cap}, "
                    f"owned_cap={owned_cap}"
                )

            with _LOCK:
                _MEMORY_GUARD_CALLS += 1
            return original(self)

        setattr(
            guarded_determine_available_memory,
            _MEMORY_GUARD_PATCH_MARKER,
            requested,
        )
        worker_cls.determine_available_memory = (
            guarded_determine_available_memory
        )
        _MEMORY_GUARD_ORIGINAL = original
        _MEMORY_GUARD_INSTALLED = True
        return True


def _install_on_worker_class(worker_cls: type[Any], cap: int) -> bool:
    global _WORKER_INSTALLED, _WORKER_ORIGINAL

    with _LOCK:
        current = worker_cls._compile_or_warm_up_model_impl
        existing_cap = getattr(current, _WORKER_PATCH_MARKER, None)
        if existing_cap is not None:
            if existing_cap != cap:
                raise RuntimeError(
                    "NF3 compile-warmup cap is already installed with "
                    f"{existing_cap}, refusing requested {cap}"
                )
            _WORKER_INSTALLED = True
            return False

        original = current

        @functools.wraps(original)
        def capped_compile_or_warm_up(self: Any) -> Any:
            global _WORKER_CALLS
            global _LAST_WORKER_RUNTIME_BATCHED
            global _LAST_WORKER_RUNTIME_SCHEDULED

            scheduler_config = self.vllm_config.scheduler_config
            for name in (
                "max_num_batched_tokens",
                "max_num_scheduled_tokens",
            ):
                if not hasattr(scheduler_config, name):
                    raise RuntimeError(
                        "NF3 compile-warmup cap requires scheduler field "
                        f"{name}"
                    )

            runtime_batched = int(
                scheduler_config.max_num_batched_tokens
            )
            runtime_scheduled_raw = (
                scheduler_config.max_num_scheduled_tokens
            )
            runtime_scheduled = (
                None
                if runtime_scheduled_raw is None
                else int(runtime_scheduled_raw)
            )
            if runtime_batched not in _ALLOWED_RUNTIME_MAX_TOKENS:
                raise RuntimeError(
                    "NF3 compile-warmup cap requires runtime "
                    "max_num_batched_tokens in "
                    f"{sorted(_ALLOWED_RUNTIME_MAX_TOKENS)}, "
                    f"got {runtime_batched}"
                )
            if (
                runtime_scheduled is not None
                and runtime_scheduled != runtime_batched
            ):
                raise RuntimeError(
                    "NF3 compile-warmup cap requires runtime "
                    "max_num_scheduled_tokens to be None or "
                    f"match max_num_batched_tokens={runtime_batched}, got "
                    f"{runtime_scheduled}"
                )

            with _LOCK:
                _WORKER_CALLS += 1
                _LAST_WORKER_RUNTIME_BATCHED = runtime_batched
                _LAST_WORKER_RUNTIME_SCHEDULED = runtime_scheduled

            LOGGER.warning(
                "Spark NF3 compile/kernel warmup capped: "
                "runtime_batched=%d runtime_scheduled=%s warmup_cap=%d",
                runtime_batched,
                runtime_scheduled,
                cap,
            )
            scheduler_config.max_num_batched_tokens = cap
            scheduler_config.max_num_scheduled_tokens = cap
            try:
                return original(self)
            finally:
                scheduler_config.max_num_batched_tokens = runtime_batched
                scheduler_config.max_num_scheduled_tokens = (
                    runtime_scheduled_raw
                )
                LOGGER.warning(
                    "Spark NF3 compile/kernel warmup completed: "
                    "runtime_batched_restored=%d "
                    "runtime_scheduled_restored=%s",
                    runtime_batched,
                    runtime_scheduled,
                )

        setattr(capped_compile_or_warm_up, _WORKER_PATCH_MARKER, cap)
        worker_cls._compile_or_warm_up_model_impl = capped_compile_or_warm_up
        _WORKER_ORIGINAL = original
        _WORKER_INSTALLED = True
        return True


def install() -> bool:
    """Attest and install the configured NF3 startup guards."""

    global _RUNNER_KIND, _V1_RUNNER_OWNED, _V2_RUNNER_OWNED
    global _MEMORY_GUARD_INSTALLED, _MEMORY_GUARD_CALLS
    global _MEMORY_GUARD_ORIGINAL

    profile_cap = _parse_cap()
    bindings = _load_bindings()
    _attest(bindings)
    installed = _install_on_class(
        bindings.runner_cls,
        profile_cap,
        runner_kind=bindings.runner_kind,
    )
    with _LOCK:
        _RUNNER_KIND = bindings.runner_kind
        _V1_RUNNER_OWNED = (
            getattr(
                bindings.v1_runner_cls.profile_run,
                _PATCH_MARKER,
                None,
            )
            == profile_cap
        )
        _V2_RUNNER_OWNED = (
            getattr(
                bindings.v2_runner_cls.profile_run,
                _PATCH_MARKER,
                None,
            )
            == profile_cap
        )
    if bindings.runner_kind == "v2":
        if not _V2_RUNNER_OWNED or _V1_RUNNER_OWNED:
            raise RuntimeError(
                "NF3 profile cap selected V2 but ownership is invalid: "
                f"v1_owned={_V1_RUNNER_OWNED}, "
                f"v2_owned={_V2_RUNNER_OWNED}"
            )
    elif not _V1_RUNNER_OWNED or _V2_RUNNER_OWNED:
        raise RuntimeError(
            "NF3 profile cap selected V1 but ownership is invalid: "
            f"v1_owned={_V1_RUNNER_OWNED}, "
            f"v2_owned={_V2_RUNNER_OWNED}"
        )

    installed = (
        _install_memory_ownership_guard(
            bindings.worker_cls,
            runner_cls=bindings.runner_cls,
            runner_kind=bindings.runner_kind,
            cap=profile_cap,
        )
        or installed
    )

    worker_cap = (
        _graph_max_tokens()
        if _single_range_contract_enabled()
        else None
    )
    if worker_cap is not None:
        installed = (
            _install_on_worker_class(
                bindings.worker_cls,
                worker_cap,
            )
            or installed
        )
    return installed


def startup_profile_cap_snapshot() -> dict[str, Any]:
    with _LOCK:
        selected_source_sha256 = (
            _EXPECTED_V2_PROFILE_RUN_SHA256
            if _RUNNER_KIND == "v2"
            else _EXPECTED_PROFILE_RUN_SHA256
        )
        return {
            "installed": _INSTALLED,
            "profile_calls": _PROFILE_CALLS,
            "capped_calls": _CAPPED_CALLS,
            "last_runtime_max_tokens": _LAST_RUNTIME_MAX,
            "last_profile_max_tokens": _LAST_PROFILE_MAX,
            "last_elapsed_seconds": _LAST_ELAPSED_SECONDS,
            "runner_kind": _RUNNER_KIND,
            "v1_runner_owned": _V1_RUNNER_OWNED,
            "v2_runner_owned": _V2_RUNNER_OWNED,
            "memory_ownership_guard_installed": (
                _MEMORY_GUARD_INSTALLED
            ),
            "memory_ownership_guard_calls": _MEMORY_GUARD_CALLS,
            "compile_warmup_installed": _WORKER_INSTALLED,
            "compile_warmup_calls": _WORKER_CALLS,
            "last_worker_runtime_batched_tokens": (
                _LAST_WORKER_RUNTIME_BATCHED
            ),
            "last_worker_runtime_scheduled_tokens": (
                _LAST_WORKER_RUNTIME_SCHEDULED
            ),
            "source_sha256": selected_source_sha256,
            "v1_profile_source_sha256": (
                _EXPECTED_PROFILE_RUN_SHA256
            ),
            "v2_profile_source_sha256": (
                _EXPECTED_V2_PROFILE_RUN_SHA256
            ),
            "compile_warmup_source_sha256": (
                _EXPECTED_COMPILE_WARMUP_SHA256
            ),
            "determine_memory_source_sha256": (
                _EXPECTED_DETERMINE_MEMORY_SHA256
            ),
            "version": _EXPECTED_VERSION,
        }


def _reset_for_tests(
    runner_cls: type[Any] | None = None,
    worker_cls: type[Any] | None = None,
) -> None:
    global _INSTALLED, _PROFILE_CALLS, _CAPPED_CALLS
    global _LAST_RUNTIME_MAX, _LAST_PROFILE_MAX
    global _LAST_ELAPSED_SECONDS, _ORIGINAL
    global _RUNNER_KIND, _V1_RUNNER_OWNED, _V2_RUNNER_OWNED
    global _WORKER_INSTALLED, _WORKER_CALLS
    global _LAST_WORKER_RUNTIME_BATCHED
    global _LAST_WORKER_RUNTIME_SCHEDULED, _WORKER_ORIGINAL
    global _MEMORY_GUARD_INSTALLED, _MEMORY_GUARD_CALLS
    global _MEMORY_GUARD_ORIGINAL

    with _LOCK:
        if runner_cls is not None and _ORIGINAL is not None:
            runner_cls.profile_run = _ORIGINAL
        if worker_cls is not None and _WORKER_ORIGINAL is not None:
            worker_cls._compile_or_warm_up_model_impl = _WORKER_ORIGINAL
        if (
            worker_cls is not None
            and _MEMORY_GUARD_ORIGINAL is not None
        ):
            worker_cls.determine_available_memory = (
                _MEMORY_GUARD_ORIGINAL
            )
        _INSTALLED = False
        _PROFILE_CALLS = 0
        _CAPPED_CALLS = 0
        _LAST_RUNTIME_MAX = None
        _LAST_PROFILE_MAX = None
        _LAST_ELAPSED_SECONDS = None
        _ORIGINAL = None
        _RUNNER_KIND = None
        _V1_RUNNER_OWNED = False
        _V2_RUNNER_OWNED = False
        _MEMORY_GUARD_INSTALLED = False
        _MEMORY_GUARD_CALLS = 0
        _MEMORY_GUARD_ORIGINAL = None
        _WORKER_INSTALLED = False
        _WORKER_CALLS = 0
        _LAST_WORKER_RUNTIME_BATCHED = None
        _LAST_WORKER_RUNTIME_SCHEDULED = None
        _WORKER_ORIGINAL = None
