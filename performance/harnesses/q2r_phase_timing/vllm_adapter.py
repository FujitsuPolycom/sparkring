"""All-or-nothing, source-pinned method adapter for Q-2R telemetry.

No vLLM method is patched until every requested owner/method/hash validates.
The concrete deployed integration points beyond ``run_fullgraph`` still need
one read-only source census; callers must supply them explicitly rather than
letting this experiment guess.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .phase_timing import PhaseDescriptor, PhaseTimingCollector

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
DescriptorResolver = Callable[
    [Any, tuple[Any, ...], dict[str, Any]], PhaseDescriptor
]


class AdapterValidationError(RuntimeError):
    """The adapter could not prove an exact, untouched integration seam."""


@dataclass(frozen=True)
class MethodHook:
    owner: type
    method_name: str
    expected_source_sha256: str
    descriptor: PhaseDescriptor | DescriptorResolver
    stream_for_call: Callable[[Any, tuple[Any, ...], dict[str, Any]], Any]

    def __post_init__(self) -> None:
        if not isinstance(self.owner, type):
            raise ValueError("owner must be a class")
        if not self.method_name:
            raise ValueError("method_name must be nonempty")
        if not _SHA256.fullmatch(self.expected_source_sha256):
            raise ValueError("expected_source_sha256 must be lowercase SHA-256")


@dataclass(frozen=True)
class _Validated:
    hook: MethodHook
    original: Callable[..., Any]


def source_sha256(function: Callable[..., Any]) -> str:
    return hashlib.sha256(
        inspect.getsource(function).encode("utf-8")
    ).hexdigest()


class FailClosedMethodAdapter:
    """Patch a set of exact methods only after the whole set validates."""

    def __init__(
        self,
        collector: PhaseTimingCollector,
        hooks: tuple[MethodHook, ...],
    ) -> None:
        if not hooks:
            raise ValueError("at least one hook is required")
        targets = {(hook.owner, hook.method_name) for hook in hooks}
        if len(targets) != len(hooks):
            raise ValueError("hook targets must be unique")
        self._collector = collector
        self._hooks = hooks
        self._validated: tuple[_Validated, ...] = ()
        self._installed = False

    def validate(self) -> tuple[_Validated, ...]:
        validated: list[_Validated] = []
        for hook in self._hooks:
            original = getattr(hook.owner, hook.method_name, None)
            if original is None or not callable(original):
                raise AdapterValidationError(
                    f"{hook.owner.__qualname__}.{hook.method_name} is absent"
                )
            if getattr(original, "_spark_q2r_phase_timing", False):
                raise AdapterValidationError(
                    f"{hook.owner.__qualname__}.{hook.method_name} is "
                    "already wrapped"
                )
            try:
                actual_hash = source_sha256(original)
            except (OSError, TypeError) as error:
                raise AdapterValidationError(
                    f"cannot inspect {hook.owner.__qualname__}."
                    f"{hook.method_name}"
                ) from error
            if actual_hash != hook.expected_source_sha256:
                raise AdapterValidationError(
                    f"source mismatch for {hook.owner.__qualname__}."
                    f"{hook.method_name}: expected "
                    f"{hook.expected_source_sha256}, got {actual_hash}"
                )
            validated.append(_Validated(hook=hook, original=original))
        return tuple(validated)

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("adapter is already installed")
        validated = self.validate()
        installed: list[_Validated] = []
        try:
            for item in validated:
                hook = item.hook
                original = item.original

                @functools.wraps(original)
                def wrapped(
                    instance: Any,
                    *args: Any,
                    __hook: MethodHook = hook,
                    __original: Callable[..., Any] = original,
                    **kwargs: Any,
                ) -> Any:
                    stream = __hook.stream_for_call(instance, args, kwargs)
                    descriptor = __hook.descriptor
                    if callable(descriptor):
                        descriptor = descriptor(instance, args, kwargs)
                    return self._collector.measure(
                        descriptor,
                        stream,
                        lambda: __original(instance, *args, **kwargs),
                    )

                wrapped._spark_q2r_phase_timing = True  # type: ignore[attr-defined]
                wrapped._spark_original = original  # type: ignore[attr-defined]
                setattr(hook.owner, hook.method_name, wrapped)
                installed.append(item)
        except Exception:
            for item in reversed(installed):
                setattr(
                    item.hook.owner, item.hook.method_name, item.original
                )
            raise
        self._validated = validated
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        for item in reversed(self._validated):
            current = getattr(item.hook.owner, item.hook.method_name)
            if not getattr(current, "_spark_q2r_phase_timing", False):
                raise AdapterValidationError(
                    f"{item.hook.owner.__qualname__}."
                    f"{item.hook.method_name} changed after installation"
                )
            setattr(item.hook.owner, item.hook.method_name, item.original)
        self._validated = ()
        self._installed = False
