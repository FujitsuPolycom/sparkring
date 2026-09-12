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
    locally_defined: bool


def _restore(item: _Validated) -> None:
    if item.locally_defined:
        setattr(item.hook.owner, item.hook.method_name, item.original)
    elif item.hook.method_name in vars(item.hook.owner):
        delattr(item.hook.owner, item.hook.method_name)


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
        self._wrappers: tuple[Callable[..., Any], ...] = ()

    def validate(self) -> tuple[_Validated, ...]:
        validated: list[_Validated] = []
        for hook in self._hooks:
            original = inspect.getattr_static(hook.owner, hook.method_name, None)
            if not inspect.isfunction(original):
                raise AdapterValidationError(
                    f"{hook.owner.__qualname__}.{hook.method_name} must be a plain instance method"
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
            validated.append(_Validated(hook=hook, original=original,
                                        locally_defined=hook.method_name in vars(hook.owner)))
        return tuple(validated)

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("adapter is already installed")
        validated = self.validate()
        installed: list[_Validated] = []
        wrappers: list[Callable[..., Any]] = []
        def make_wrapper(item: _Validated) -> Callable[..., Any]:
            hook, original = item.hook, item.original

            @functools.wraps(original)
            def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
                stream = hook.stream_for_call(instance, args, kwargs)
                descriptor = hook.descriptor
                if not isinstance(descriptor, PhaseDescriptor):
                    descriptor = descriptor(instance, args, kwargs)
                return self._collector.measure(
                    descriptor, stream, lambda: original(instance, *args, **kwargs),
                )

            wrapped._spark_q2r_phase_timing = True  # type: ignore[attr-defined]
            wrapped._spark_original = original  # type: ignore[attr-defined]
            return wrapped

        try:
            for item in validated:
                hook = item.hook
                wrapped = make_wrapper(item)
                # Record rollback ownership before the assignment: a signal or
                # metaclass can raise after the attribute has already changed.
                installed.append(item)
                wrappers.append(wrapped)
                setattr(hook.owner, hook.method_name, wrapped)
        except BaseException:
            for item in reversed(installed):
                _restore(item)
            raise
        self._validated = validated
        self._wrappers = tuple(wrappers)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        # Verify the whole set before removing any hook. Wrapper metadata can
        # be copied by functools.wraps, so it does not prove ownership.
        for item, wrapper in zip(self._validated, self._wrappers, strict=True):
            current = inspect.getattr_static(item.hook.owner, item.hook.method_name, None)
            if current is not wrapper:
                raise AdapterValidationError(
                    f"{item.hook.owner.__qualname__}."
                    f"{item.hook.method_name} changed after installation"
                )
        for item in reversed(self._validated):
            _restore(item)
        self._validated = ()
        self._wrappers = ()
        self._installed = False
