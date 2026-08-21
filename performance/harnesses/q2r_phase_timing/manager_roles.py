"""Stable startup-time identities for vLLM CUDA graph managers.

``CudaGraphManager`` descriptors and ``decode_query_len`` do not carry
semantic ownership. The deployed MTP path owns three managers: target
verification, draft prefill, and repeated draft decode. This registry accepts
roles only from explicit, source-pinned semantic seams.
"""

from __future__ import annotations

import threading
import functools
import re
from dataclasses import dataclass
from enum import Enum
from collections.abc import Callable
from typing import Any

from .vllm_adapter import AdapterValidationError, source_sha256

from .phase_timing import PhaseDescriptor, PhaseKind

_MAX_MANAGERS = 64


class ManagerRole(str, Enum):
    UNKNOWN = "unknown"
    TARGET_VERIFY = "target_verify"
    DRAFT_BLOCK = "draft_block"
    DRAFT_PREFILL = "draft_prefill"
    DRAFT_DECODE = "draft_decode"
    PLAIN_DECODE = "plain_decode"


_DRAFT_ROLES = frozenset(
    {
        ManagerRole.DRAFT_BLOCK,
        ManagerRole.DRAFT_PREFILL,
        ManagerRole.DRAFT_DECODE,
    }
)


@dataclass(frozen=True)
class ManagerIdentity:
    manager_id: int
    role: ManagerRole
    decode_query_len: int


class ManagerRoleRegistry:
    """Bounded mapping populated from a source-pinned ``__init__`` hook."""

    def __init__(
        self,
        *,
        maximum_managers: int = _MAX_MANAGERS,
    ) -> None:
        if not 1 <= maximum_managers <= _MAX_MANAGERS:
            raise ValueError(
                f"maximum_managers must be in [1, {_MAX_MANAGERS}]"
            )
        self._maximum_managers = maximum_managers
        self._lock = threading.Lock()
        # Keep a bounded strong reference. This avoids assuming the deployed
        # extension-backed manager is weak-referenceable or hashable, and it
        # prevents Python object-ID reuse during the process lifetime.
        self._identities: dict[
            int, tuple[object, ManagerIdentity]
        ] = {}
        self._next_id = 0

    def register(
        self,
        manager: object,
        *,
        decode_query_len: int,
        role: ManagerRole = ManagerRole.UNKNOWN,
    ) -> ManagerIdentity:
        if (
            not isinstance(decode_query_len, int)
            or isinstance(decode_query_len, bool)
            or decode_query_len < 1
        ):
            raise ValueError("decode_query_len must be a positive integer")
        if not isinstance(role, ManagerRole):
            raise ValueError("role must be a ManagerRole")
        object_id = id(manager)
        with self._lock:
            existing_entry = self._identities.get(object_id)
            if existing_entry is not None:
                existing_manager, existing = existing_entry
                if existing_manager is not manager:
                    raise RuntimeError("graph-manager object ID collision")
                if existing.decode_query_len != decode_query_len:
                    raise RuntimeError(
                        "manager was re-registered with a different "
                        "decode_query_len"
                    )
                if (
                    role is not ManagerRole.UNKNOWN
                    and existing.role is not role
                ):
                    if existing.role is ManagerRole.UNKNOWN:
                        updated = ManagerIdentity(
                            manager_id=existing.manager_id,
                            role=role,
                            decode_query_len=existing.decode_query_len,
                        )
                        self._identities[object_id] = (
                            manager,
                            updated,
                        )
                        return updated
                    raise RuntimeError(
                        "manager was re-registered with a different role"
                    )
                return existing
            if self._next_id >= self._maximum_managers:
                raise RuntimeError("graph-manager registry capacity exhausted")
            identity = ManagerIdentity(
                manager_id=self._next_id,
                role=role,
                decode_query_len=decode_query_len,
            )
            self._next_id += 1
            self._identities[object_id] = (manager, identity)
            return identity

    def assign_role(
        self, manager: object, role: ManagerRole
    ) -> ManagerIdentity:
        """Assign one explicit semantic role; never infer it from Q length."""
        if role is ManagerRole.UNKNOWN or not isinstance(role, ManagerRole):
            raise ValueError("an explicit non-UNKNOWN role is required")
        object_id = id(manager)
        with self._lock:
            entry = self._identities.get(object_id)
            if entry is None or entry[0] is not manager:
                raise RuntimeError("unregistered graph manager")
            identity = entry[1]
            if identity.role is role:
                return identity
            if identity.role is not ManagerRole.UNKNOWN:
                raise RuntimeError(
                    f"manager role is already {identity.role.value}"
                )
            updated = ManagerIdentity(
                manager_id=identity.manager_id,
                role=role,
                decode_query_len=identity.decode_query_len,
            )
            self._identities[object_id] = (manager, updated)
            return updated

    def identity(self, manager: object) -> ManagerIdentity:
        with self._lock:
            entry = self._identities.get(id(manager))
        if entry is None or entry[0] is not manager:
            raise RuntimeError("unregistered graph manager")
        return entry[1]

    def graph_descriptor(
        self,
        manager: object,
        *,
        graph_method: str,
        draft_step: int | None = None,
    ) -> PhaseDescriptor:
        """Resolve a finite descriptor without guessing Q1 ownership."""
        identity = self.identity(manager)
        suffix = (
            f"manager={identity.manager_id},qlen="
            f"{identity.decode_query_len},method={graph_method}"
        )
        if draft_step is not None:
            if draft_step < 0:
                raise ValueError("draft_step must be nonnegative")
            if identity.role not in _DRAFT_ROLES:
                raise RuntimeError(
                    "draft-step context applied to a non-draft manager"
                )
            return PhaseDescriptor(
                PhaseKind.DRAFT_MULTISTEP_GRAPH,
                f"step={draft_step},{suffix}",
            )
        if identity.role is ManagerRole.TARGET_VERIFY:
            kind = (
                PhaseKind.TARGET_FULL_GRAPH
                if graph_method == "run_fullgraph"
                else PhaseKind.OTHER_GRAPH
            )
            return PhaseDescriptor(kind, suffix)
        if identity.role in _DRAFT_ROLES:
            stage = {
                ManagerRole.DRAFT_BLOCK: "block",
                ManagerRole.DRAFT_PREFILL: "prefill",
                ManagerRole.DRAFT_DECODE: "decode",
            }[identity.role]
            return PhaseDescriptor(
                PhaseKind.DRAFT_MULTISTEP_GRAPH,
                f"stage={stage},{suffix}",
            )
        return PhaseDescriptor(
            PhaseKind.OTHER_GRAPH,
            f"role={identity.role.value},{suffix}",
        )

    def snapshot(self) -> dict[str, Any]:
        """Startup/debug evidence; never called by graph replay."""
        with self._lock:
            values = sorted(
                (
                    identity
                    for _manager, identity in self._identities.values()
                ),
                key=lambda item: item.manager_id,
            )
        return {
            "maximum_managers": self._maximum_managers,
            "registered": len(values),
            "identities": [
                {
                    "manager_id": item.manager_id,
                    "role": item.role.value,
                    "decode_query_len": item.decode_query_len,
                }
                for item in values
            ],
        }

    def manager_entries(
        self,
    ) -> tuple[tuple[object, ManagerIdentity], ...]:
        """Return stable startup entries; never called by graph replay."""
        with self._lock:
            return tuple(
                sorted(
                    self._identities.values(),
                    key=lambda item: item[1].manager_id,
                )
            )


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
RoleCallResolver = Callable[
    [Any, tuple[Any, ...], dict[str, Any], Any], Any
]


@dataclass(frozen=True)
class RoleAssignmentHook:
    """One exact semantic seam that owns a graph manager after returning."""

    owner: type
    method_name: str
    expected_source_sha256: str
    role: ManagerRole
    manager_after_call: RoleCallResolver
    decode_query_len_after_call: RoleCallResolver

    def __post_init__(self) -> None:
        if not isinstance(self.owner, type):
            raise ValueError("owner must be a class")
        if not self.method_name:
            raise ValueError("method_name must be nonempty")
        if not _SHA256.fullmatch(self.expected_source_sha256):
            raise ValueError("expected_source_sha256 must be lowercase SHA-256")


@dataclass(frozen=True)
class _ValidatedRoleHook:
    hook: RoleAssignmentHook
    original: Callable[..., Any]


class FailClosedRoleAssignmentAdapter:
    """Source-pin semantic manager creation before changing any method."""

    def __init__(
        self,
        registry: ManagerRoleRegistry,
        hooks: tuple[RoleAssignmentHook, ...],
    ) -> None:
        if not hooks:
            raise ValueError("at least one role hook is required")
        targets = {(hook.owner, hook.method_name) for hook in hooks}
        if len(targets) != len(hooks):
            raise ValueError("role hook targets must be unique")
        self._registry = registry
        self._hooks = hooks
        self._validated: tuple[_ValidatedRoleHook, ...] = ()
        self._installed = False

    def validate(self) -> tuple[_ValidatedRoleHook, ...]:
        validated: list[_ValidatedRoleHook] = []
        for hook in self._hooks:
            original = getattr(hook.owner, hook.method_name, None)
            if original is None or not callable(original):
                raise AdapterValidationError(
                    f"{hook.owner.__qualname__}.{hook.method_name} is absent"
                )
            if getattr(original, "_spark_q2r_role_assignment", False):
                raise AdapterValidationError(
                    f"{hook.owner.__qualname__}.{hook.method_name} is "
                    "already wrapped for role assignment"
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
            validated.append(
                _ValidatedRoleHook(hook=hook, original=original)
            )
        return tuple(validated)

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("role adapter is already installed")
        validated = self.validate()
        installed: list[_ValidatedRoleHook] = []
        try:
            for item in validated:
                hook = item.hook
                original = item.original

                @functools.wraps(original)
                def wrapped(
                    instance: Any,
                    *args: Any,
                    __hook: RoleAssignmentHook = hook,
                    __original: Callable[..., Any] = original,
                    **kwargs: Any,
                ) -> Any:
                    result = __original(instance, *args, **kwargs)
                    manager = __hook.manager_after_call(
                        instance, args, kwargs, result
                    )
                    decode_query_len = int(
                        __hook.decode_query_len_after_call(
                            instance, args, kwargs, result
                        )
                    )
                    self._registry.register(
                        manager,
                        decode_query_len=decode_query_len,
                        role=__hook.role,
                    )
                    return result

                wrapped._spark_q2r_role_assignment = True  # type: ignore[attr-defined]
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
            if not getattr(current, "_spark_q2r_role_assignment", False):
                raise AdapterValidationError(
                    f"{item.hook.owner.__qualname__}."
                    f"{item.hook.method_name} changed after installation"
                )
            setattr(item.hook.owner, item.hook.method_name, item.original)
        self._validated = ()
        self._installed = False
