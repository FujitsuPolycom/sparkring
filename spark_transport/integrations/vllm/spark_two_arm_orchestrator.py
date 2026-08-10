"""Tracer-bullet two-arm orchestrator for the SIRCL-vs-NCCL numerical probe.

This module renders and optionally executes a two-arm benchmark where each
arm selects exactly one transport implementation:

- **SIRCL arm** — native TP4 transport via ``VLLM_SPARK_TP4_MODE=custom``.
- **NCCL-IB control arm** — patched switchless NCCL via ``NCCL_NET=IB``,
  ``NCCL_IB_DISABLE=0`` (SETUP.md §8.3). This is the real comparison arm.
- **NCCL Socket diagnostic arm** — optional diagnostic-only arm via
  ``NCCL_NET=Socket``, ``NCCL_IB_DISABLE=1``. This is NOT a control arm
  and must never satisfy or replace the patched NCCL-IB control gate.

Key design decisions (per goal-4 blocker 1, goal-11):

- **One selector per arm.** The probe reads ``VLLM_SPARK_TP4_MODE`` before
  initializing any transport. Unknown/missing values fail closed.
- **Shared contract module.** All canonical env/argv/identity constants
  live in ``spark_transport_contract.py`` — the probe, plan builder, JSON
  parser, and validator all import from that one module. No duplicate
  ``_build_env_projection()`` functions.
- **Four-rank consensus before any collective.** A synchronized four-process
  control-plane preflight runs before any SIRCL or NCCL data collective.
- **Elements are consumed.** ``--elements`` is passed to the probe via
  the ``ELEMENTS`` environment variable; the probe no longer hardcodes 6144.
- **Four-node launch is truthful.** Each arm renders one rank per host
  (4 hosts, 1 rank each), NOT ``torchrun --nproc_per_node=4`` (one 4-GPU
  node). An explicit ``executor`` seam is required to actually launch
  across hosts; without it, ``--no-dry-run`` fails before mutation.
- **Observed counters from per-rank receipts.** Counters come from
  per-rank probe output, not caller-authored declarations.
- **Every collective classified exactly once.** Each collective is
  native, NCCL-IB, NCCL-Socket, unsupported-bypassed, unclassified,
  or fatal-after-native.
- **Arm invalidation.** SIRCL arm: any NCCL/unclassified event
  invalidates. NCCL-IB arm: any native/Socket event invalidates.
- **Totals reconcile.** Per-rank and global totals must match observed
  collective count.

Safety classes (per AGENTS.md):
  - OFFLINE — no native execution, plan rendering only.
  - READ-ONLY REMOTE — inspect generated plan, no host mutation.
  - MUTATES HOST — would launch processes / write evidence.
  - STOPS SERVING — would terminate serving workers.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

from spark_transport_contract import (
    AUTHORITATIVE_RANKS,
    AUTHORITATIVE_WORLD_SIZE,
    ARM_BINDING,
    ARM_NAME_NCCL_IB,
    ARM_NAME_NCCL_SOCKET_DIAGNOSTIC,
    ARM_NAME_SIRCL,
    COUNTER_NATIVE as _COUNTER_NATIVE,
    COUNTER_NCCL_IB as _COUNTER_NCCL_IB,
    BF16_ATOL as _BF16_ATOL,
    BF16_TOLERANCE as _BF16_TOLERANCE,  # noqa: F401 (re-export for tests)
    BF16_RTOL as _BF16_RTOL,
    TOLERANCE_METRIC as _TOLERANCE_METRIC,
    CANONICAL_ARGV as _CANONICAL_ARGV,
    CONFIRMATION_PROMPT as _CONFIRMATION_PROMPT,
    CONFIRMATION_REQUIRED_RESPONSE as _CONFIRMATION_REQUIRED_RESPONSE,
    ENV_ALLOWLIST as _ENV_ALLOWLIST,
    EXIT_INVALID,
    EXIT_MALFORMED,
    EXIT_MISSING_SEAM,
    EXIT_VALID,
    NATIVE_PROBE_SCRIPT as _NATIVE_PROBE_SCRIPT,
    NCCL_IB_SELECTOR_ENVS as _NCCL_IB_SELECTOR_ENVS,
    NCCL_SOCKET_SELECTOR_ENVS as _NCCL_SOCKET_SELECTOR_ENVS,
    PINNED_BINARY_IDENTITY as _PINNED_BINARY_IDENTITY,
    NCCL_IB_ENV_VARS as _NCCL_IB_ENV_VARS,
    PINNED_ORDER as _PINNED_ORDER,
    NCCL_IB_REQUIRED_KEYS as _NCCL_IB_REQUIRED_KEYS,
    PINNED_PROBE_IDENTITY as _PINNED_PROBE_IDENTITY,
    PINNED_TOPOLOGY as _PINNED_TOPOLOGY,
    PINNED_WORKLOAD as _PINNED_WORKLOAD,
    SAFETY_MUTATES_HOST,
    SAFETY_OFFLINE,
    SELECTOR_NCCL_IB as SELECTOR_NCCL,  # noqa: F401 (re-export for tests)
    SELECTOR_SIRCL,
    SIRCL_SELECTOR_ENVS,
    _VALID_SELECTORS,
    TRANSPORT_NCCL_IB,
    TRANSPORT_NCCL_SOCKET,
    TRANSPORT_NCCL_SOCKET_DIAGNOSTIC,
    TRANSPORT_SIRCL,
    VALID_SAFETY_CLASSES as _VALID_SAFETY_CLASSES,
    build_env_projection,
)

# Backward-compat aliases: existing code and tests import underscore-prefixed
# names.  The canonical constants now live in spark_transport_contract.
# Goal 11: the NCCL control arm is nccl_ib, not nccl_socket.  The
# legacy _NCCL_SELECTOR_ENVS now points to the IB env vars; the Socket
# diagnostic arm uses _NCCL_SOCKET_SELECTOR_ENVS.
_build_env_projection = build_env_projection
_TRANSPORT_SIRCL = TRANSPORT_SIRCL
_TRANSPORT_NCCL_IB = TRANSPORT_NCCL_IB
_TRANSPORT_NCCL_SOCKET = TRANSPORT_NCCL_SOCKET  # legacy: Socket diagnostic
_TRANSPORT_NCCL_SOCKET_DIAGNOSTIC = TRANSPORT_NCCL_SOCKET_DIAGNOSTIC
_ARM_NAME_SIRCL = ARM_NAME_SIRCL
_ARM_NAME_NCCL = ARM_NAME_NCCL_IB  # Goal 11: NCCL control arm is nccl_ib
_ARM_NAME_NCCL_SOCKET_DIAGNOSTIC = ARM_NAME_NCCL_SOCKET_DIAGNOSTIC
_SIRCL_SELECTOR_ENVS = SIRCL_SELECTOR_ENVS
_NCCL_SELECTOR_ENVS = _NCCL_IB_SELECTOR_ENVS  # Goal 11: control arm = IB
_NCCL_SOCKET_SELECTOR_ENVS = _NCCL_SOCKET_SELECTOR_ENVS
_ARM_BINDING = ARM_BINDING
_COUNTER_CUSTOM = "custom_collectives"
_COUNTER_NATIVE = _COUNTER_NATIVE  # Goal 11: native_collectives
_COUNTER_NCCL_IB = _COUNTER_NCCL_IB  # Goal 11: nccl_ib_collectives
_COUNTER_UNSUPPORTED = "unsupported_bypassed_collectives"
_COUNTER_UNCLASSIFIED = "unclassified_collectives"
_COUNTER_FALLBACK = "fallback_collectives"
_VALID_COUNTERS = frozenset({
    _COUNTER_CUSTOM, _COUNTER_FALLBACK,
    _COUNTER_UNSUPPORTED, _COUNTER_UNCLASSIFIED,
    _COUNTER_NATIVE, _COUNTER_NCCL_IB,
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArmSpec:
    """Specification for one arm of the two-arm benchmark.

    The only functional delta between arms is ``transport`` and
    ``selector_env_vars``; everything else must be identical.
    """

    transport: str
    selector_env_vars: frozenset[str]
    world_size: int = 4
    iterations: int = 1000
    elements: int = 6144
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if not isinstance(self.transport, str) or not self.transport:
            raise ValueError("transport must be a non-empty string")
        if not isinstance(self.selector_env_vars, frozenset):
            raise ValueError("selector_env_vars must be a frozenset")
        if not self.selector_env_vars:
            raise ValueError("selector_env_vars must not be empty")
        if not isinstance(self.world_size, int) or isinstance(self.world_size, bool):
            raise ValueError("world_size must be an int")
        if self.world_size < 1:
            raise ValueError("world_size must be >= 1")
        if not isinstance(self.iterations, int) or isinstance(self.iterations, bool):
            raise ValueError("iterations must be an int")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if not isinstance(self.elements, int) or isinstance(self.elements, bool):
            raise ValueError("elements must be an int")
        if self.elements < 1:
            raise ValueError("elements must be >= 1")
        if not isinstance(self.timeout_seconds, int) or isinstance(self.timeout_seconds, bool):
            raise ValueError("timeout_seconds must be an int")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be >= 1")


@dataclass(frozen=True)
class RankLaunchEntry:
    """One rank's launch entry for a four-node deployment.

    Each rank runs on one host. The ``executor`` seam is responsible
    for launching the probe on each host and collecting receipts.
    """

    rank: int
    host: str
    command: list[str]
    env_vars: dict[str, str]

@dataclass(frozen=True)
class ArmPlan:
    """Rendered plan for one arm.

    ``rank_launches`` contains one ``RankLaunchEntry`` per rank,
    truthfully representing one rank per host.
    """

    arm_name: str
    transport: str
    safety_class: str
    selector: str
    rank_launches: tuple[RankLaunchEntry, ...]
    timeout_seconds: int

    def __post_init__(self) -> None:
        if not isinstance(self.arm_name, str) or not self.arm_name:
            raise ValueError("arm_name must be a non-empty string")
        if not isinstance(self.transport, str) or not self.transport:
            raise ValueError("transport must be a non-empty string")
        if self.safety_class not in _VALID_SAFETY_CLASSES:
            raise ValueError(
                f"safety_class must be one of {sorted(_VALID_SAFETY_CLASSES)}, "
                f"got '{self.safety_class}'"
            )
        if self.selector not in _VALID_SELECTORS:
            raise ValueError(
                f"selector must be one of {sorted(_VALID_SELECTORS)}, "
                f"got '{self.selector}'"
            )
        if not isinstance(self.rank_launches, tuple) or not self.rank_launches:
            raise ValueError("rank_launches must be a non-empty tuple")
        if not isinstance(self.timeout_seconds, int) or isinstance(self.timeout_seconds, bool):
            raise ValueError("timeout_seconds must be an int")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be >= 1")


@dataclass(frozen=True)
class TwoArmPlan:
    """Complete two-arm benchmark plan.

    In dry-run mode, no execution is performed. The
    ``confirmation_required`` flag is always True; the operator
    must type the exact confirmation string before any execution.
    """

    sircl_arm: ArmPlan
    nccl_arm: ArmPlan
    shared_identity: dict[str, str]
    dry_run: bool
    confirmation_required: bool
    executor_available: bool

    def __post_init__(self) -> None:
        if not isinstance(self.sircl_arm, ArmPlan):
            raise ValueError("sircl_arm must be an ArmPlan")
        if not isinstance(self.nccl_arm, ArmPlan):
            raise ValueError("nccl_arm must be an ArmPlan")
        if not isinstance(self.shared_identity, dict):
            raise ValueError("shared_identity must be a dict")
        if not isinstance(self.dry_run, bool):
            raise ValueError("dry_run must be a bool")
        if not isinstance(self.confirmation_required, bool):
            raise ValueError("confirmation_required must be a bool")
        if not self.confirmation_required:
            raise ValueError("confirmation_required must always be True")
        if not isinstance(self.executor_available, bool):
            raise ValueError("executor_available must be a bool")


@dataclass(frozen=True)
class RankReceipt:
    """Observed per-rank probe receipt from an executed arm.

    Counters are observed from the probe's stdout, not caller-authored.
    Each collective is classified exactly once.

    Execution fields (selector, iterations, elements, world_size) carry
    the identity of the actual run.  The validator binds these against
    the authoritative plan/manifest — a receipt claiming different
    values is rejected.

    Numerical evidence fields carry execution-derived proof:
    - ``expected_fp32_hash``: SHA-256 of the deterministic FP32 reference
    - ``actual_output_hash``: SHA-256 of the actual collective output
      (caller-supplied, tamper-evident — NOT non-forgeable)
    - ``all_finite``: whether all output elements are finite (not NaN/Inf)
    - ``max_abs_error``: maximum absolute error vs FP32 reference (diagnostic)
    - ``max_rel_error``: maximum relative error (zero-denominator policy:
      when FP32 reference is 0.0, absolute error is used) (diagnostic)
    - ``tolerance_result``: "pass" or "fail" under the elementwise criterion
    - ``tolerance_metric``: the criterion name (e.g. "elementwise_atol_rtol")
    - ``sample_count``: number of elements inspected
    - ``run_contract_hash``: SHA-256 binding plan+arm+rank+execution fields
    - ``rank_identity``: sanitized stable rank identity (rank-N-of-M)

    These fields are **optional** (default empty/zero) for receipts
    that are not "successful" (e.g. fallback or unclassified
    receipts).  The validator rejects successful receipts (all-native
    for SIRCL, all-fallback for NCCL) that carry empty/default
    numerical fields — missing required evidence is rejected, not
    silently accepted with benign defaults.
    """
    rank: int
    host: str
    transport: str
    selector: str
    iterations: int
    elements: int
    world_size: int
    custom_collectives: int
    fallback_collectives: int
    unsupported_bypassed_collectives: int
    unclassified_collectives: int
    total_collectives: int
    # Numerical evidence fields — NO defaults (Goal 10 requirement 5).
    # Omission at constructor must fail, not silently accept benign values.
    # Callers must explicitly pass values (empty string / 0.0 / False
    # for non-successful receipts; the validator rejects successful
    # receipts that carry empty/default numerical fields).
    expected_fp32_hash: str
    actual_output_hash: str
    actual_dtype: str
    actual_byte_order: str
    all_finite: bool
    max_abs_error: float
    max_rel_error: float
    tolerance_result: str
    tolerance_metric: str
    tolerance_atol: float
    tolerance_rtol: float
    sample_count: int
    run_contract_hash: str
    rank_identity: str
    # Goal 11 requirement 5: attribution fields.
    # These bind counter source identity and binary/library hashes.
    # Defaults to empty string for backward compat — the validator
    # rejects successful receipts that carry empty attribution.
    counter_source_hash: str = ""
    source_sha: str = ""
    sircl_so_sha: str = ""
    nccl_so_sha: str = ""
    image_receipt: str = ""
    # Goal 11 requirement 5: new externally attributable counters.
    native_collectives: int = 0
    nccl_ib_collectives: int = 0
    nccl_socket_collectives: int = 0
    fatal_after_native_collectives: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.rank, int) or isinstance(self.rank, bool):
            raise ValueError("rank must be an int")
        if self.rank < 0:
            raise ValueError("rank must be >= 0")
        if not isinstance(self.host, str) or not self.host:
            raise ValueError("host must be a non-empty string")
        if not isinstance(self.transport, str) or not self.transport:
            raise ValueError("transport must be a non-empty string")
        if not isinstance(self.selector, str) or not self.selector:
            raise ValueError("selector must be a non-empty string")
        if self.selector not in _VALID_SELECTORS:
            raise ValueError(
                f"selector must be one of {sorted(_VALID_SELECTORS)}, "
                f"got '{self.selector}'"
            )
        for name, val in [
            ("iterations", self.iterations),
            ("elements", self.elements),
            ("world_size", self.world_size),
            ("custom_collectives", self.custom_collectives),
            ("fallback_collectives", self.fallback_collectives),
            ("unsupported_bypassed_collectives", self.unsupported_bypassed_collectives),
            ("unclassified_collectives", self.unclassified_collectives),
            ("total_collectives", self.total_collectives),
        ]:
            if not isinstance(val, int) or isinstance(val, bool):
                raise ValueError(f"{name} must be an int")
            if val < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.elements < 1:
            raise ValueError("elements must be >= 1")
        if self.world_size < 1:
            raise ValueError("world_size must be >= 1")
        # Every collective classified exactly once: sum of categories == total.
        classified = (
            self.custom_collectives
            + self.fallback_collectives
            + self.unsupported_bypassed_collectives
            + self.unclassified_collectives
        )
        if classified != self.total_collectives:
            raise ValueError(
                f"rank {self.rank}: classified sum ({classified}) != "
                f"total_collectives ({self.total_collectives})"
            )
        # Numerical evidence field type validation.
        if not isinstance(self.expected_fp32_hash, str):
            raise ValueError("expected_fp32_hash must be a str")
        if not isinstance(self.actual_output_hash, str):
            raise ValueError("actual_output_hash must be a str")
        if not isinstance(self.all_finite, bool):
            raise ValueError("all_finite must be a bool")
        # Reject booleans-as-numbers for error metrics.
        if not isinstance(self.max_abs_error, (int, float)) or isinstance(self.max_abs_error, bool):
            raise ValueError("max_abs_error must be a number")
        if self.max_abs_error < 0:
            raise ValueError("max_abs_error must be >= 0")
        if not isinstance(self.max_rel_error, (int, float)) or isinstance(self.max_rel_error, bool):
            raise ValueError("max_rel_error must be a number")
        if self.max_rel_error < 0:
            raise ValueError("max_rel_error must be >= 0")
        # Reject NaN/Inf error metrics — they are never valid.
        import math
        if math.isnan(self.max_abs_error) or math.isinf(self.max_abs_error):
            raise ValueError(
                f"max_abs_error must be finite, got {self.max_abs_error}"
            )
        if math.isnan(self.max_rel_error) or math.isinf(self.max_rel_error):
            raise ValueError(
                f"max_rel_error must be finite, got {self.max_rel_error}"
            )
        if not isinstance(self.sample_count, int) or isinstance(self.sample_count, bool):
            raise ValueError("sample_count must be an int")
        if self.sample_count < 0:
            raise ValueError("sample_count must be >= 0")
        if not isinstance(self.run_contract_hash, str):
            raise ValueError("run_contract_hash must be a str")
        if not isinstance(self.rank_identity, str):
            raise ValueError("rank_identity must be a str")
        if not isinstance(self.tolerance_result, str):
            raise ValueError("tolerance_result must be a str")
        if not isinstance(self.tolerance_metric, str):
            raise ValueError("tolerance_metric must be a str")
        # Goal 10 requirement 5: validate new acceptance fields.
        if not isinstance(self.actual_dtype, str):
            raise ValueError("actual_dtype must be a str")
        if not isinstance(self.actual_byte_order, str):
            raise ValueError("actual_byte_order must be a str")
        if not isinstance(self.tolerance_atol, (int, float)) or isinstance(self.tolerance_atol, bool):
            raise ValueError("tolerance_atol must be a number")
        if self.tolerance_atol < 0:
            raise ValueError("tolerance_atol must be >= 0")
        if math.isnan(self.tolerance_atol) or math.isinf(self.tolerance_atol):
            raise ValueError(f"tolerance_atol must be finite, got {self.tolerance_atol}")
        if not isinstance(self.tolerance_rtol, (int, float)) or isinstance(self.tolerance_rtol, bool):
            raise ValueError("tolerance_rtol must be a number")
        if self.tolerance_rtol < 0:
            raise ValueError("tolerance_rtol must be >= 0")
        if math.isnan(self.tolerance_rtol) or math.isinf(self.tolerance_rtol):
            raise ValueError(f"tolerance_rtol must be finite, got {self.tolerance_rtol}")
        # Hash format validation: if provided, must be 64 lowercase hex.
        import re
        for name, val in [
            ("expected_fp32_hash", self.expected_fp32_hash),
            ("actual_output_hash", self.actual_output_hash),
            ("run_contract_hash", self.run_contract_hash),
            ("counter_source_hash", self.counter_source_hash),
            ("source_sha", self.source_sha),
            ("sircl_so_sha", self.sircl_so_sha),
            ("nccl_so_sha", self.nccl_so_sha),
        ]:
            if val and not re.match(r"^[0-9a-f]{64}$", val):
                raise ValueError(
                    f"{name} must be a 64-char lowercase hex SHA-256 "
                    f"or empty, got '{val[:20]}...'"
                )
        if self.image_receipt and not re.fullmatch(
            r"sha256:[0-9a-f]{64}", self.image_receipt
        ):
            raise ValueError(
                "image_receipt must be a canonical sha256:<64 lowercase "
                f"hex> digest or empty, got '{self.image_receipt[:20]}...'"
            )
        # Goal 11: validate new attribution counter fields.
        for name, val in [
            ("native_collectives", self.native_collectives),
            ("nccl_ib_collectives", self.nccl_ib_collectives),
            ("nccl_socket_collectives", self.nccl_socket_collectives),
            ("fatal_after_native_collectives", self.fatal_after_native_collectives),
        ]:
            if not isinstance(val, int) or isinstance(val, bool):
                raise ValueError(f"{name} must be an int")
            if val < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.custom_collectives != self.native_collectives:
            raise ValueError(
                "custom_collectives must equal native_collectives"
            )
        attributed_nccl = (
            self.nccl_ib_collectives + self.nccl_socket_collectives
        )
        if self.fallback_collectives != attributed_nccl:
            raise ValueError(
                "fallback_collectives must equal nccl_ib_collectives + "
                "nccl_socket_collectives"
            )

@dataclass(frozen=True)
class ArmResult:
    """Observed results for one arm from per-rank receipts."""

    arm_name: str
    transport: str
    receipts: tuple[RankReceipt, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.arm_name, str) or not self.arm_name:
            raise ValueError("arm_name must be a non-empty string")
        if not isinstance(self.transport, str) or not self.transport:
            raise ValueError("transport must be a non-empty string")
        if not isinstance(self.receipts, tuple) or not self.receipts:
            raise ValueError("receipts must be a non-empty tuple")


@dataclass
class TwoArmResult:
    """Observed results from an executed two-arm benchmark.

    Contains per-rank receipts for each arm and validation status.
    """

    sircl_arm: ArmResult
    nccl_arm: ArmResult
    valid: bool
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Executor seam protocol
# ---------------------------------------------------------------------------

class ExecutorSeal:
    """Marker base class for executor implementations.

    An executor is an injected callable that:
    1. Receives one ``ArmPlan`` and a confirmation string.
    2. Launches the probe on each rank's host.
    3. Collects per-rank receipts as ``RankReceipt`` objects.
    4. Returns an ``ArmResult``.

    The public checkout provides no concrete executor — it has no way
    to launch processes on remote hosts without CUDA/RDMA. This class
    exists so tests can inject a mock executor and verify the seam.
    """


# Type for the executor callable.
ExecutorCallable = type[ExecutorSeal] | object


def _executor_available(executor: ExecutorCallable | None) -> bool:
    """Check whether a real executor is available."""
    return executor is not None


# ---------------------------------------------------------------------------
# Plan rendering
# ---------------------------------------------------------------------------

def _build_rank_command(
    spec: ArmSpec, rank: int,
) -> list[str]:
    """Build the launch command for one rank on one host.

    Each rank runs the probe as a standalone process. The probe reads
    its selector from ``VLLM_SPARK_TP4_MODE`` and its element count
    from ``ELEMENTS``. An external launcher (executor seam) is
    responsible for coordinating the 4 ranks across 4 hosts.
    """
    return [
        "python",
        _NATIVE_PROBE_SCRIPT,
    ]


def _build_env_vars(spec: ArmSpec, rank: int) -> dict[str, str]:
    """Build the environment variables for one rank from selector env vars.

    Goal 11 requirement 2: for the NCCL-IB control arm, include the
    full patched switchless NCCL recipe from SETUP.md §8.3 — not just
    NCCL_NET and NCCL_IB_DISABLE.
    """
    from spark_transport_contract import (
        NCCL_IB_ENV_VARS as _FULL_NCCL_IB_ENV,
        NCCL_SOCKET_ENV_VARS as _FULL_NCCL_SOCKET_ENV,
        TRANSPORT_NCCL_IB as _T_IB,
        TRANSPORT_NCCL_SOCKET_DIAGNOSTIC as _T_SOCK_DIAG,
    )
    env: dict[str, str] = {}
    for pair in sorted(spec.selector_env_vars):
        if "=" not in pair:
            raise ValueError(f"invalid selector env var (no '='): {pair}")
        key, val = pair.split("=", 1)
        env[key] = val
    env["ITERATIONS"] = str(spec.iterations)
    env["WORLD_SIZE"] = str(spec.world_size)
    env["ELEMENTS"] = str(spec.elements)
    env["RANK"] = str(rank)
    # Goal 11 requirement 2: add full §8.3 env vars for NCCL-IB arm.
    if spec.transport == _T_IB:
        for k, v in _FULL_NCCL_IB_ENV.items():
            if k not in env:
                env[k] = v
    elif spec.transport == _T_SOCK_DIAG:
        for k, v in _FULL_NCCL_SOCKET_ENV.items():
            if k not in env:
                env[k] = v
    return env


def _extract_selector(spec: ArmSpec) -> str:
    """Extract the VLLM_SPARK_TP4_MODE value from the selector env vars."""
    for pair in spec.selector_env_vars:
        if pair.startswith("VLLM_SPARK_TP4_MODE="):
            return pair.split("=", 1)[1]
    raise ValueError(
        f"selector_env_vars must contain VLLM_SPARK_TP4_MODE: "
        f"got {sorted(spec.selector_env_vars)}"
    )


def _build_shared_identity(
    sircl_spec: ArmSpec, nccl_spec: ArmSpec,
) -> dict[str, str]:
    """Build the shared identity fields that must be identical across arms."""
    return {
        "binary": _NATIVE_PROBE_SCRIPT,
        "topology": "tp4_switchless_ring",
        "world_size": str(sircl_spec.world_size),
        "iterations": str(sircl_spec.iterations),
        "elements": str(sircl_spec.elements),
        "order": "identical",
        "workload": "tp4_numerical_audit",
        "launch_model": "one_rank_per_host",
    }


def render_plan(
    sircl_spec: ArmSpec,
    nccl_spec: ArmSpec,
    *,
    dry_run: bool = True,
    hosts: tuple[str, ...] | None = None,
    site_profile: dict | None = None,
    executor: ExecutorCallable | None = None,
) -> TwoArmPlan:
    """Render the exact two-arm plan.

    Each arm renders one ``RankLaunchEntry`` per rank, truthfully
    representing one rank per host. An explicit executor seam is
    required for execution; without it, ``--no-dry-run`` fails.

    Parameters
    ----------
    sircl_spec, nccl_spec
        Arm specifications. Must differ only in transport/selector.
    dry_run
        If True (default), the plan is rendered but nothing executed.
    hosts
        Optional tuple of hostnames for the 4 ranks. If None,
        placeholder hostnames ``spark-0`` through ``spark-3`` are used.
    """
    if sircl_spec.world_size != nccl_spec.world_size:
        raise ValueError(
            f"world_size mismatch: sircl={sircl_spec.world_size}, "
            f"nccl={nccl_spec.world_size}"
        )
    if sircl_spec.iterations != nccl_spec.iterations:
        raise ValueError(
            f"iterations mismatch: sircl={sircl_spec.iterations}, "
            f"nccl={nccl_spec.iterations}"
        )
    if sircl_spec.elements != nccl_spec.elements:
        raise ValueError(
            f"elements mismatch: sircl={sircl_spec.elements}, "
            f"nccl={nccl_spec.elements}"
        )

    if sircl_spec.selector_env_vars != _SIRCL_SELECTOR_ENVS:
        raise ValueError(
            f"SIRCL arm selector must be {sorted(_SIRCL_SELECTOR_ENVS)}, "
            f"got {sorted(sircl_spec.selector_env_vars)}"
        )
    if nccl_spec.selector_env_vars != _NCCL_SELECTOR_ENVS:
        raise ValueError(
            f"NCCL arm selector must be {sorted(_NCCL_SELECTOR_ENVS)}, "
            f"got {sorted(nccl_spec.selector_env_vars)}"
        )
    world_size = sircl_spec.world_size
    if hosts is None and site_profile is not None:
        hosts = tuple(site_profile.get("hosts", ()))
    if hosts is None:
        if not dry_run:
            raise ValueError(
                "non-dry-run plan requires a site_profile with hosts "
                "(placeholder spark-0..3 is not a runnable plan)"
            )
        hosts = tuple(f"spark-{i}" for i in range(world_size))
    if len(hosts) != world_size:
        raise ValueError(
            f"hosts must have {world_size} entries, got {len(hosts)}"
        )

    safety_class = SAFETY_OFFLINE if dry_run else SAFETY_MUTATES_HOST

    sircl_selector = _extract_selector(sircl_spec)
    nccl_selector = _extract_selector(nccl_spec)

    sircl_launches = tuple(
        RankLaunchEntry(
            rank=rank,
            host=hosts[rank],
            command=_build_rank_command(sircl_spec, rank),
            env_vars=_build_env_vars(sircl_spec, rank),
        )
        for rank in range(world_size)
    )
    nccl_launches = tuple(
        RankLaunchEntry(
            rank=rank,
            host=hosts[rank],
            command=_build_rank_command(nccl_spec, rank),
            env_vars=_build_env_vars(nccl_spec, rank),
        )
        for rank in range(world_size)
    )

    sircl_arm = ArmPlan(
        arm_name="sircl",
        transport=sircl_spec.transport,
        safety_class=safety_class,
        selector=sircl_selector,
        rank_launches=sircl_launches,
        timeout_seconds=sircl_spec.timeout_seconds,
    )

    nccl_arm = ArmPlan(
        arm_name=ARM_NAME_NCCL_IB,
        transport=nccl_spec.transport,
        safety_class=safety_class,
        selector=nccl_selector,
        rank_launches=nccl_launches,
        timeout_seconds=nccl_spec.timeout_seconds,
    )

    return TwoArmPlan(
        sircl_arm=sircl_arm,
        nccl_arm=nccl_arm,
        shared_identity=_build_shared_identity(sircl_spec, nccl_spec),
        dry_run=dry_run,
        confirmation_required=True,
        executor_available=_executor_available(executor),
    )


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------

def validate_plan(plan: TwoArmPlan) -> list[str]:
    """Validate a TwoArmPlan against the authoritative campaign contract.

    The campaign is exactly 4 ranks and exactly 2 arms.  The validator,
    not caller-controlled plan fields, owns these facts:
    - World size is exactly ``AUTHORITATIVE_WORLD_SIZE`` (4).
    - Ranks are exactly ``{0, 1, 2, 3}``.
    - SIRCL arm name binds to SIRCL transport and SIRCL selector.
    - NCCL arm name binds to NCCL transport and NCCL selector.

    Every rank's environment is validated independently, not only rank 0.
    """
    errors: list[str] = []

    # --- Four-rank authority ---
    shared = plan.shared_identity
    ws_str = shared.get("world_size")
    try:
        ws = int(ws_str) if ws_str is not None else 0
    except (ValueError, TypeError):
        errors.append(
            f"shared_identity world_size must be int, got '{ws_str}'"
        )
        ws = 0
    if ws != AUTHORITATIVE_WORLD_SIZE:
        errors.append(
            f"world_size must be exactly {AUTHORITATIVE_WORLD_SIZE}, "
            f"got {ws}"
        )

    # --- Non-swappable arms: bind arm name → transport → selector ---
    for arm_name, arm in [
        (_ARM_NAME_SIRCL, plan.sircl_arm),
        (_ARM_NAME_NCCL, plan.nccl_arm),
    ]:
        binding = _ARM_BINDING[arm_name]
        if arm.arm_name != arm_name:
            errors.append(
                f"plan {arm_name} arm name must be '{arm_name}', "
                f"got '{arm.arm_name}'"
            )
        if arm.transport != binding["transport"]:
            errors.append(
                f"plan {arm_name} arm transport must be "
                f"'{binding['transport']}', got '{arm.transport}'"
            )
        if arm.selector != binding["selector"]:
            errors.append(
                f"plan {arm_name} arm selector must be "
                f"'{binding['selector']}', got '{arm.selector}'"
            )

    # Different selectors enforced by the binding (already checked above).

    if plan.sircl_arm.timeout_seconds != plan.nccl_arm.timeout_seconds:
        errors.append(
            f"timeout mismatch: sircl={plan.sircl_arm.timeout_seconds}, "
            f"nccl={plan.nccl_arm.timeout_seconds}"
        )

    # --- Per-rank environment validation (every rank, not just rank 0) ---
    for arm_name, arm in [
        (_ARM_NAME_SIRCL, plan.sircl_arm),
        (_ARM_NAME_NCCL, plan.nccl_arm),
    ]:
        expected_env = _SIRCL_SELECTOR_ENVS if arm_name == _ARM_NAME_SIRCL else _NCCL_SELECTOR_ENVS
        for idx, rl in enumerate(arm.rank_launches):
            # RANK env must match the launch entry's rank.
            if rl.env_vars.get("RANK") != str(rl.rank):
                errors.append(
                    f"{arm_name} arm rank_launch[{idx}] RANK env "
                    f"({rl.env_vars.get('RANK')}) != rank ({rl.rank})"
                )
            # WORLD_SIZE env must be authoritative.
            if rl.env_vars.get("WORLD_SIZE") != str(AUTHORITATIVE_WORLD_SIZE):
                errors.append(
                    f"{arm_name} arm rank_launch[{idx}] WORLD_SIZE env "
                    f"({rl.env_vars.get('WORLD_SIZE')}) != "
                    f"{AUTHORITATIVE_WORLD_SIZE}"
                )
            # Selector env vars must match the arm's canonical set.
            env_selector_set = frozenset(
                f"{k}={v}" for k, v in rl.env_vars.items()
                if k in (
                    {"VLLM_SPARK_TP4_MODE"} if arm_name == _ARM_NAME_SIRCL
                    else {"VLLM_SPARK_TP4_MODE", "NCCL_NET", "NCCL_IB_DISABLE"}
                )
            )
            if env_selector_set != expected_env:
                errors.append(
                    f"{arm_name} arm rank_launch[{idx}] selector env vars "
                    f"must be {sorted(expected_env)}, got "
                    f"{sorted(env_selector_set)}"
                )
            # Goal 11: NCCL-IB arm must have ALL required §8.3 env vars
            # present and matching the pinned values.
            if arm_name == _ARM_NAME_NCCL:
                for key in _NCCL_IB_REQUIRED_KEYS:
                    pinned = _NCCL_IB_ENV_VARS[key]
                    actual = rl.env_vars.get(key)
                    if actual is None:
                        errors.append(
                            f"{arm_name} arm rank_launch[{idx}] missing "
                            f"required NCCL-IB env var '{key}'"
                        )
                    elif actual != pinned:
                        errors.append(
                            f"{arm_name} arm rank_launch[{idx}] env var "
                            f"'{key}' must be '{pinned}', got '{actual}'"
                        )
            # ITERATIONS and ELEMENTS must be present and positive.
            for field_name in ("ITERATIONS", "ELEMENTS"):
                val_str = rl.env_vars.get(field_name)
                if val_str is None:
                    errors.append(
                        f"{arm_name} arm rank_launch[{idx}] must set "
                        f"{field_name} env var"
                    )
                    continue
                try:
                    val = int(val_str)
                    if val < 1:
                        errors.append(
                            f"{arm_name} arm rank_launch[{idx}] "
                            f"{field_name} must be >= 1, got {val}"
                        )
                except (ValueError, TypeError):
                    errors.append(
                        f"{arm_name} arm rank_launch[{idx}] "
                        f"{field_name} must be int, got '{val_str}'"
                    )

    # --- Cross-arm consistency for ITERATIONS and ELEMENTS ---
    sircl_elements = plan.sircl_arm.rank_launches[0].env_vars.get("ELEMENTS")
    nccl_elements = plan.nccl_arm.rank_launches[0].env_vars.get("ELEMENTS")
    if sircl_elements is not None and nccl_elements is not None:
        if sircl_elements != nccl_elements:
            errors.append(
                f"elements mismatch: sircl={sircl_elements}, "
                f"nccl={nccl_elements}"
            )
    sircl_iters = plan.sircl_arm.rank_launches[0].env_vars.get("ITERATIONS")
    nccl_iters = plan.nccl_arm.rank_launches[0].env_vars.get("ITERATIONS")
    if sircl_iters is not None and nccl_iters is not None:
        if sircl_iters != nccl_iters:
            errors.append(
                f"iterations mismatch: sircl={sircl_iters}, "
                f"nccl={nccl_iters}"
            )

    # --- Rank coverage: exactly {0, 1, 2, 3} ---
    if ws == AUTHORITATIVE_WORLD_SIZE:
        for arm_name, arm in [
            (_ARM_NAME_SIRCL, plan.sircl_arm),
            (_ARM_NAME_NCCL, plan.nccl_arm),
        ]:
            if len(arm.rank_launches) != AUTHORITATIVE_WORLD_SIZE:
                errors.append(
                    f"{arm_name} arm must have {AUTHORITATIVE_WORLD_SIZE} "
                    f"rank launches, got {len(arm.rank_launches)}"
                )
                continue
            ranks = [rl.rank for rl in arm.rank_launches]
            # Reject booleans as rank IDs.
            if any(isinstance(r, bool) for r in ranks):
                errors.append(f"{arm_name} arm has boolean rank IDs")
            if len(set(ranks)) != len(ranks):
                errors.append(f"{arm_name} arm has duplicate rank IDs")
            if set(ranks) != AUTHORITATIVE_RANKS:
                errors.append(
                    f"{arm_name} arm ranks must be "
                    f"{sorted(AUTHORITATIVE_RANKS)}, got {sorted(ranks)}"
                )
            hosts = [rl.host for rl in arm.rank_launches]
            if len(set(hosts)) != len(hosts):
                errors.append(f"{arm_name} arm has duplicate hosts")

    # --- Shared identity consistency ---
    for arm_name, arm in [("sircl", plan.sircl_arm), ("nccl", plan.nccl_arm)]:
        env = arm.rank_launches[0].env_vars if arm.rank_launches else {}
        if env.get("ITERATIONS") is not None and env.get("ITERATIONS") != shared.get("iterations"):
            errors.append(
                f"{arm_name} arm ITERATIONS env ({env.get('ITERATIONS')}) != "
                f"shared_identity iterations ({shared.get('iterations')})"
            )
        if env.get("ELEMENTS") is not None and env.get("ELEMENTS") != shared.get("elements"):
            errors.append(
                f"{arm_name} arm ELEMENTS env ({env.get('ELEMENTS')}) != "
                f"shared_identity elements ({shared.get('elements')})"
            )
        if env.get("WORLD_SIZE") is not None and env.get("WORLD_SIZE") != shared.get("world_size"):
            errors.append(
                f"{arm_name} arm WORLD_SIZE env ({env.get('WORLD_SIZE')}) != "
                f"shared_identity world_size ({shared.get('world_size')})"
            )

    if shared.get("launch_model") != "one_rank_per_host":
        errors.append(
            "shared_identity launch_model must be 'one_rank_per_host', "
            f"got '{shared.get('launch_model')}'"
        )

    # Goal 9 requirement 4: bind shared identity fields to
    # validator-owned pinned values.  Caller-controlled mutually
    # consistent forgery is not authority.
    if shared.get("topology") != _PINNED_TOPOLOGY:
        errors.append(
            f"shared_identity topology must be '{_PINNED_TOPOLOGY}', "
            f"got '{shared.get('topology')}'"
        )
    if shared.get("workload") != _PINNED_WORKLOAD:
        errors.append(
            f"shared_identity workload must be '{_PINNED_WORKLOAD}', "
            f"got '{shared.get('workload')}'"
        )
    if shared.get("order") != _PINNED_ORDER:
        errors.append(
            f"shared_identity order must be '{_PINNED_ORDER}', "
            f"got '{shared.get('order')}'"
        )
    if shared.get("binary") != _PINNED_BINARY_IDENTITY:
        errors.append(
            f"shared_identity binary must be '{_PINNED_BINARY_IDENTITY}', "
            f"got '{shared.get('binary')}'"
        )

    if not plan.confirmation_required:
        errors.append("confirmation_required must be True")

    if plan.dry_run:
        for arm_name, arm in [
            (_ARM_NAME_SIRCL, plan.sircl_arm),
            (_ARM_NAME_NCCL, plan.nccl_arm),
        ]:
            if arm.safety_class != SAFETY_OFFLINE:
                errors.append(
                    f"dry-run {arm_name} arm safety_class must be "
                    f"{SAFETY_OFFLINE}, got {arm.safety_class}"
                )
    else:
        if not plan.executor_available:
            errors.append(
                "non-dry-run plan requires executor_available=True "
                "(no executor seam present)"
            )

    for arm_name, arm in [
        (_ARM_NAME_SIRCL, plan.sircl_arm),
        (_ARM_NAME_NCCL, plan.nccl_arm),
    ]:
        if arm.timeout_seconds < 1:
            errors.append(f"{arm_name} arm timeout must be >= 1")

    return errors


# ---------------------------------------------------------------------------
# Receipt validation (observed counters)
# ---------------------------------------------------------------------------

def _validate_receipts(
    transport: str,
    expected_selector: str,
    expected_iterations: int,
    expected_elements: int,
    expected_world_size: int,
    rank_host_map: dict[int, str],
    receipts: tuple[RankReceipt, ...],
) -> list[str]:
    """Validate per-rank receipts for one arm.

    Rules:
    - Exactly expected_world_size receipts, one per rank.
    - Each receipt's rank matches its position.
    - Each receipt's host matches the exact planned host for that rank
      (not just set membership).
    - Each receipt's selector matches the arm's expected selector.
    - Each receipt's iterations, elements, world_size match the plan.
    - Each collective classified exactly once (sum of categories == total).
    - SIRCL arm: any fallback/unclassified event invalidates the arm.
    - NCCL arm: any native/custom event invalidates the arm.
    - Totals reconcile: per-rank totals sum to global total.
    """
    errors: list[str] = []

    if len(receipts) != expected_world_size:
        errors.append(
            f"expected {expected_world_size} receipts, got {len(receipts)}"
        )
        return errors

    seen_ranks: set[int] = set()

    for idx, receipt in enumerate(receipts):
        if receipt.rank != idx:
            errors.append(
                f"receipt[{idx}].rank must be {idx}, got {receipt.rank}"
            )
        if receipt.rank in seen_ranks:
            errors.append(f"duplicate rank {receipt.rank} in receipts")
        seen_ranks.add(receipt.rank)
        # Bind each rank to its exact planned host — set membership
        # is insufficient (all ranks claiming the same allowed host
        # must be rejected).
        expected_host = rank_host_map.get(receipt.rank)
        if expected_host is None:
            errors.append(
                f"receipt[{idx}].rank {receipt.rank} has no planned host"
            )
        elif receipt.host != expected_host:
            errors.append(
                f"receipt[{idx}].host '{receipt.host}' != planned "
                f"host '{expected_host}' for rank {receipt.rank}"
            )
        if receipt.transport != transport:
            errors.append(
                f"receipt[{idx}].transport '{receipt.transport}' != "
                f"arm transport '{transport}'"
            )
        # Bind selector to the arm's expected selector.
        if receipt.selector != expected_selector:
            errors.append(
                f"receipt[{idx}].selector '{receipt.selector}' != "
                f"arm selector '{expected_selector}'"
            )
        # Bind execution fields to the plan.
        if receipt.iterations != expected_iterations:
            errors.append(
                f"receipt[{idx}].iterations {receipt.iterations} != "
                f"plan iterations {expected_iterations}"
            )
        if receipt.elements != expected_elements:
            errors.append(
                f"receipt[{idx}].elements {receipt.elements} != "
                f"plan elements {expected_elements}"
            )
        if receipt.world_size != expected_world_size:
            errors.append(
                f"receipt[{idx}].world_size {receipt.world_size} != "
                f"plan world_size {expected_world_size}"
            )
        # total_collectives must equal iterations.
        if receipt.total_collectives != expected_iterations:
            errors.append(
                f"receipt[{idx}].total_collectives "
                f"{receipt.total_collectives} != iterations "
                f"{expected_iterations}"
            )

    # Arm-specific invalidation.
    for receipt in receipts:
        if transport == _TRANSPORT_SIRCL:
            # SIRCL arm: any NCCL fallback invalidates.  This includes
            # both explicit fallback_collectives and
            # unsupported_bypassed_collectives (which is still NCCL
            # fallback execution, not a skipped collective).
            if receipt.fallback_collectives > 0:
                errors.append(
                    f"SIRCL arm invalidated: rank {receipt.rank} has "
                    f"{receipt.fallback_collectives} fallback events"
                )
            if receipt.unsupported_bypassed_collectives > 0:
                errors.append(
                    f"SIRCL arm invalidated: rank {receipt.rank} has "
                    f"{receipt.unsupported_bypassed_collectives} "
                    f"unsupported-bypassed (NCCL fallback) events"
                )
            if receipt.unclassified_collectives > 0:
                errors.append(
                    f"SIRCL arm invalidated: rank {receipt.rank} has "
                    f"{receipt.unclassified_collectives} unclassified events"
                )
            # Goal 11: fatal-after-native is a hard failure — a rank
            # that crashed after starting native execution must not pass.
            if receipt.fatal_after_native_collectives > 0:
                errors.append(
                    f"SIRCL arm invalidated: rank {receipt.rank} has "
                    f"{receipt.fatal_after_native_collectives} "
                    f"fatal-after-native events (crash after native start)"
                )
            # Goal 11: nccl_socket_collectives on SIRCL arm means
            # NCCL Socket transport was used — not native SIRCL.
            if receipt.nccl_socket_collectives > 0:
                errors.append(
                    f"SIRCL arm invalidated: rank {receipt.rank} has "
                    f"{receipt.nccl_socket_collectives} "
                    f"nccl_socket events (Socket transport used)"
                )
        elif transport in (_TRANSPORT_NCCL_SOCKET, _TRANSPORT_NCCL_IB, _TRANSPORT_NCCL_SOCKET_DIAGNOSTIC):
            # Goal 11: use native_collectives (not legacy custom_collectives)
            # to detect SIRCL transport leaking into the NCCL arm.
            if receipt.native_collectives > 0 or receipt.custom_collectives > 0:
                errors.append(
                    f"NCCL arm invalidated: rank {receipt.rank} has "
                    f"native/SIRCL events "
                    f"(native={receipt.native_collectives}, "
                    f"custom={receipt.custom_collectives})"
                )
            # Goal 11: fatal-after-native is a hard failure on NCCL arm too.
            if receipt.fatal_after_native_collectives > 0:
                errors.append(
                    f"NCCL arm invalidated: rank {receipt.rank} has "
                    f"{receipt.fatal_after_native_collectives} "
                    f"fatal-after-native events"
                )
            if receipt.unsupported_bypassed_collectives > 0:
                errors.append(
                    f"NCCL arm invalidated: rank {receipt.rank} has "
                    f"{receipt.unsupported_bypassed_collectives} "
                    "unsupported-bypassed events"
                )
            if receipt.unclassified_collectives > 0:
                errors.append(
                    f"NCCL arm invalidated: rank {receipt.rank} has "
                    f"{receipt.unclassified_collectives} unclassified events"
                )
            if transport == _TRANSPORT_NCCL_IB:
                if receipt.nccl_ib_collectives != receipt.total_collectives:
                    errors.append(
                        f"NCCL-IB arm invalidated: rank {receipt.rank} has "
                        f"nccl_ib_collectives={receipt.nccl_ib_collectives}, "
                        f"expected {receipt.total_collectives}"
                    )
                if receipt.nccl_socket_collectives != 0:
                    errors.append(
                        f"NCCL-IB arm invalidated: rank {receipt.rank} has "
                        f"{receipt.nccl_socket_collectives} nccl_socket events"
                    )
            else:
                if receipt.nccl_socket_collectives != receipt.total_collectives:
                    errors.append(
                        f"NCCL-Socket arm invalidated: rank {receipt.rank} has "
                        f"nccl_socket_collectives={receipt.nccl_socket_collectives}, "
                        f"expected {receipt.total_collectives}"
                    )
                if receipt.nccl_ib_collectives != 0:
                    errors.append(
                        f"NCCL-Socket arm invalidated: rank {receipt.rank} has "
                        f"{receipt.nccl_ib_collectives} nccl_ib events"
                    )

    # Numerical commitment validation for successful receipts.
    # A "successful" receipt is one where all collectives were
    # classified as native (SIRCL) or fallback (NCCL) — i.e. the
    # arm executed its intended path.  These must carry full numerical
    # evidence: expected FP32 hash, actual output hash, all-finite,
    # max abs/rel error, tolerance result/metric, sample count,
    # run-contract hash, and rank_identity.
    expected_sample_count = expected_iterations * expected_elements
    for idx, receipt in enumerate(receipts):
        is_sircl_success = (
            transport == _TRANSPORT_SIRCL
            and receipt.native_collectives == receipt.total_collectives
            and receipt.nccl_ib_collectives == 0
            and receipt.nccl_socket_collectives == 0
            and receipt.unsupported_bypassed_collectives == 0
            and receipt.unclassified_collectives == 0
        )
        is_nccl_success = (
            transport in (_TRANSPORT_NCCL_SOCKET, _TRANSPORT_NCCL_IB, _TRANSPORT_NCCL_SOCKET_DIAGNOSTIC)
            and (
                receipt.nccl_ib_collectives == receipt.total_collectives
                if transport == _TRANSPORT_NCCL_IB
                else receipt.nccl_socket_collectives == receipt.total_collectives
            )
            and receipt.native_collectives == 0
            and receipt.unsupported_bypassed_collectives == 0
            and receipt.unclassified_collectives == 0
        )
        if is_sircl_success or is_nccl_success:
            # All numerical fields must be present and non-default.
            # Goal 11: validate new attribution counters for successful
            # receipts.  SIRCL success requires native_collectives ==
            # total; NCCL-IB success requires nccl_ib_collectives ==
            # total (not just legacy fallback_collectives).
            if is_sircl_success and receipt.native_collectives != receipt.total_collectives:
                errors.append(
                    f"receipt[{idx}] rank {receipt.rank}: "
                    f"native_collectives ({receipt.native_collectives}) != "
                    f"total_collectives ({receipt.total_collectives}) "
                    f"for successful SIRCL receipt"
                )
            if is_nccl_success and transport == _TRANSPORT_NCCL_IB:
                if receipt.nccl_ib_collectives != receipt.total_collectives:
                    errors.append(
                        f"receipt[{idx}] rank {receipt.rank}: "
                        f"nccl_ib_collectives ({receipt.nccl_ib_collectives}) "
                        f"!= total_collectives ({receipt.total_collectives}) "
                        f"for successful NCCL-IB receipt"
                    )
            if not receipt.expected_fp32_hash:
                errors.append(
                    f"receipt[{idx}] rank {receipt.rank}: "
                    f"expected_fp32_hash required for successful receipt"
                )
            if not receipt.actual_output_hash:
                errors.append(
                    f"receipt[{idx}] rank {receipt.rank}: "
                    f"actual_output_hash required for successful receipt"
                )
            if not receipt.run_contract_hash:
                errors.append(
                    f"receipt[{idx}] rank {receipt.rank}: "
                    f"run_contract_hash required for successful receipt"
                )
            if not receipt.rank_identity:
                errors.append(
                    f"receipt[{idx}] rank {receipt.rank}: "
                    f"rank_identity required for successful receipt"
                )
            if not receipt.tolerance_result:
                errors.append(
                    f"receipt[{idx}] rank {receipt.rank}: "
                    f"tolerance_result required for successful receipt"
                )
            if not receipt.tolerance_metric:
                errors.append(
                    f"receipt[{idx}] rank {receipt.rank}: "
                    f"tolerance_metric required for successful receipt"
                )
            # all_finite must be True for successful receipts.
            if not receipt.all_finite:
                errors.append(
                    f"receipt[{idx}] rank {receipt.rank}: "
                    f"all_finite must be True for successful receipt"
                )
            # sample_count must equal iterations * elements.
            if receipt.sample_count != expected_sample_count:
                errors.append(
                    f"receipt[{idx}] rank {receipt.rank}: "
                    f"sample_count {receipt.sample_count} != "
                    f"iterations*elements ({expected_sample_count})"
                )
            # tolerance_result must be "pass" for successful receipts.
            if receipt.tolerance_result != "pass":
                errors.append(
                    f"receipt[{idx}] rank {receipt.rank}: "
                    f"tolerance_result must be 'pass' for successful "
                    f"receipt, got '{receipt.tolerance_result}'"
                )
            # tolerance_metric must match the validator-owned metric.
            if receipt.tolerance_metric != _TOLERANCE_METRIC:
                errors.append(
                    f"receipt[{idx}] rank {receipt.rank}: "
                    f"tolerance_metric must be '{_TOLERANCE_METRIC}', "
                    f"got '{receipt.tolerance_metric}'"
                )
            # Goal 10 requirement 5: validate new acceptance fields.
            if not receipt.actual_dtype:
                errors.append(
                    f"receipt[{idx}] rank {receipt.rank}: "
                    f"actual_dtype required for successful receipt"
                )
            if not receipt.actual_byte_order:
                errors.append(
                    f"receipt[{idx}] rank {receipt.rank}: "
                    f"actual_byte_order required for successful receipt"
                )
            # tolerance_atol/rtol must match validator-owned policy.
            if receipt.tolerance_atol != _BF16_ATOL:
                errors.append(
                    f"receipt[{idx}] rank {receipt.rank}: "
                    f"tolerance_atol {receipt.tolerance_atol} != "
                    f"validator-owned {_BF16_ATOL}"
                )
            if receipt.tolerance_rtol != _BF16_RTOL:
                errors.append(
                    f"receipt[{idx}] rank {receipt.rank}: "
                    f"tolerance_rtol {receipt.tolerance_rtol} != "
                    f"validator-owned {_BF16_RTOL}"
                )
            # rank_identity must match the validator-derived identity.
            expected_rank_identity = (
                f"rank-{receipt.rank}-of-{expected_world_size}"
            )
            if receipt.rank_identity != expected_rank_identity:
                errors.append(
                    f"receipt[{idx}] rank {receipt.rank}: "
                    f"rank_identity '{receipt.rank_identity}' != "
                    f"validator-derived '{expected_rank_identity}'"
                )
            # Error metrics are diagnostics — not the acceptance
            # criterion.  The elementwise criterion (tolerance_result)
            # governs acceptance.  We still reject NaN/Inf (done at
            # construction) but do not reject based on the old global
            # absolute threshold.
    return errors


def validate_arm_receipts(
    arm: ArmPlan,
    receipts: tuple[RankReceipt, ...],
    expected_iterations: int,
    expected_elements: int,
    expected_world_size: int,
) -> list[str]:
    """Validate per-rank receipts for one arm (ArmPlan-based).

    Binds each rank to its exact planned host from the arm's
    rank_launches, not just host-set membership.
    """
    rank_host_map = {rl.rank: rl.host for rl in arm.rank_launches}
    return _validate_receipts(
        arm.transport, arm.selector,
        expected_iterations, expected_elements, expected_world_size,
        rank_host_map, receipts,
    )


def validate_two_arm_results(
    result: TwoArmResult,
    plan: TwoArmPlan,
) -> list[str]:
    """Validate a complete two-arm result from observed receipts.

    **An authoritative plan is required.**  Public validation must
    not accept a self-sized result without an authoritative
    plan/manifest.  All identity fields (selector, iterations,
    elements, hosts, ranks, world_size, transport) are validated
    against the plan — not from self-asserted receipts.

    The plan itself is validated first via ``validate_plan`` — a
    caller-made plan that doesn't pass the complete plan validator
    is rejected before any receipts are inspected.

    Arm names are bound to constants: the SIRCL arm must be named
    ``"sircl"`` and the NCCL arm must be named ``"nccl_socket"``.
    Swapped arm names in either the plan or the result are rejected.

    Each arm is bound to its required transport/selector from the
    plan.  Each rank is bound to its exact planned host (set
    membership is insufficient — all ranks claiming the same allowed
    host must be rejected).

    Counter accounting is exhaustive: a NCCL arm with nonzero
    ``unclassified_collectives`` is rejected — NCCL fallback must
    classify every collective as fallback, not unclassified.
    """
    errors: list[str] = []

    # Step 1: validate the plan itself before inspecting receipts.
    plan_errors = validate_plan(plan)
    errors.extend(plan_errors)
    # If the plan is invalid, don't proceed with receipt validation
    # — the plan is the authority, and an invalid plan means the
    # bindings are unreliable.
    if plan_errors:
        result.errors = errors
        result.valid = False
        return errors

    # Step 2: bind arm names, transports, and selectors to the
    # authoritative _ARM_BINDING map.  This catches swapped arms even
    # if the plan and results are swapped consistently — the binding
    # is validator-owned, not caller-controlled.
    for arm_slot, expected_name, arm_result, plan_arm in [
        ("sircl", _ARM_NAME_SIRCL, result.sircl_arm, plan.sircl_arm),
        ("nccl", _ARM_NAME_NCCL, result.nccl_arm, plan.nccl_arm),
    ]:
        binding = _ARM_BINDING[expected_name]
        # Plan arm must match the authoritative binding.
        if plan_arm.arm_name != expected_name:
            errors.append(
                f"plan {arm_slot} arm name must be '{expected_name}', "
                f"got '{plan_arm.arm_name}'"
            )
        if plan_arm.transport != binding["transport"]:
            errors.append(
                f"plan {arm_slot} arm transport must be "
                f"'{binding['transport']}', got '{plan_arm.transport}'"
            )
        if plan_arm.selector != binding["selector"]:
            errors.append(
                f"plan {arm_slot} arm selector must be "
                f"'{binding['selector']}', got '{plan_arm.selector}'"
            )
        # Result arm must match the authoritative binding.
        if arm_result.arm_name != expected_name:
            errors.append(
                f"result {arm_slot} arm name must be '{expected_name}', "
                f"got '{arm_result.arm_name}'"
            )
        if arm_result.transport != binding["transport"]:
            errors.append(
                f"result {arm_slot} arm transport must be "
                f"'{binding['transport']}', got '{arm_result.transport}'"
            )

    # Extract plan values — these are authoritative.
    sircl_hosts = {rl.host for rl in plan.sircl_arm.rank_launches}
    nccl_hosts = {rl.host for rl in plan.nccl_arm.rank_launches}

    shared = plan.shared_identity
    try:
        plan_iters = int(shared.get("iterations", 0))
    except (ValueError, TypeError):
        plan_iters = -1
    try:
        plan_elements = int(shared.get("elements", 0))
    except (ValueError, TypeError):
        plan_elements = -1

    # World size is authoritative — always 4, regardless of what
    # the plan's rank_launches contain.
    ws = AUTHORITATIVE_WORLD_SIZE

    # Step 4: validate launch-env vs shared-identity consistency for ALL ranks.
    # Goal 10 requirement 3: each rank's argv and environment must be
    # validated independently against validator-owned arm/workload authority.
    # Previously only rank_launches[0] was checked — mutating rank 3's
    # ITERATIONS or ELEMENTS silently validated.
    for arm_label, arm in [("sircl", plan.sircl_arm), ("nccl", plan.nccl_arm)]:
        for idx, rl in enumerate(arm.rank_launches):
            env_iters = rl.env_vars.get("ITERATIONS")
            env_elements = rl.env_vars.get("ELEMENTS")
            env_ws = rl.env_vars.get("WORLD_SIZE")
            env_rank = rl.env_vars.get("RANK")
            env_selector = rl.env_vars.get("VLLM_SPARK_TP4_MODE")
            if env_iters is not None and env_iters != shared.get("iterations"):
                errors.append(
                    f"{arm_label} arm rank_launch[{idx}] ITERATIONS env "
                    f"({env_iters}) != shared_identity iterations "
                    f"({shared.get('iterations')})"
                )
            if env_elements is not None and env_elements != shared.get("elements"):
                errors.append(
                    f"{arm_label} arm rank_launch[{idx}] ELEMENTS env "
                    f"({env_elements}) != shared_identity elements "
                    f"({shared.get('elements')})"
                )
            if env_ws is not None and env_ws != shared.get("world_size"):
                errors.append(
                    f"{arm_label} arm rank_launch[{idx}] WORLD_SIZE env "
                    f"({env_ws}) != shared_identity world_size "
                    f"({shared.get('world_size')})"
                )
            # RANK env must match the launch entry's rank index.
            if env_rank is not None and env_rank != str(rl.rank):
                errors.append(
                    f"{arm_label} arm rank_launch[{idx}] RANK env "
                    f"({env_rank}) != rank ({rl.rank})"
                )
            # Selector must match the arm's expected selector.
            expected_selector = _ARM_BINDING[arm.arm_name]["selector"]
            if env_selector is not None and env_selector != expected_selector:
                errors.append(
                    f"{arm_label} arm rank_launch[{idx}] VLLM_SPARK_TP4_MODE "
                    f"({env_selector}) != expected selector "
                    f"({expected_selector})"
                )
            # NCCL-IB control arm must have NCCL_NET=IB and NCCL_IB_DISABLE=0.
            # Socket diagnostic arm (if present) must have NCCL_NET=Socket
            # and NCCL_IB_DISABLE=1, and is validated separately.
            if arm_label == "nccl":
                if rl.env_vars.get("NCCL_NET") != "IB":
                    errors.append(
                        f"{arm_label} arm rank_launch[{idx}] NCCL_NET "
                        f"must be 'IB', got '{rl.env_vars.get('NCCL_NET')}'"
                    )
                if rl.env_vars.get("NCCL_IB_DISABLE") != "0":
                    errors.append(
                        f"{arm_label} arm rank_launch[{idx}] NCCL_IB_DISABLE "
                        f"must be '0', got '{rl.env_vars.get('NCCL_IB_DISABLE')}'"
                    )
            # Argv must match canonical argv for every rank.
            if tuple(rl.command) != _CANONICAL_ARGV:
                errors.append(
                    f"{arm_label} arm rank_launch[{idx}] command "
                    f"{rl.command} != canonical argv {list(_CANONICAL_ARGV)}"
                )

    # Per-arm receipt validation: bind rank to exact host, selector,
    # iterations, elements, world_size against the authoritative plan.
    errors.extend(validate_arm_receipts(
        plan.sircl_arm, result.sircl_arm.receipts,
        plan_iters, plan_elements, ws,
    ))
    errors.extend(validate_arm_receipts(
        plan.nccl_arm, result.nccl_arm.receipts,
        plan_iters, plan_elements, ws,
    ))

    # Step 5: NCCL arm with nonzero unclassified_collectives must fail.
    for receipt in result.nccl_arm.receipts:
        if receipt.unclassified_collectives > 0:
            errors.append(
                f"NCCL arm invalidated: rank {receipt.rank} has "
                f"{receipt.unclassified_collectives} unclassified "
                f"collectives — NCCL must classify all as fallback"
            )

    # Cross-arm: hosts must match (same hardware).
    if sircl_hosts != nccl_hosts:
        errors.append(
            f"cross-arm host mismatch: sircl hosts "
            f"{sorted(sircl_hosts)} != nccl hosts "
            f"{sorted(nccl_hosts)}"
        )

    # Cross-arm: total collectives must match (same workload).
    sircl_total = sum(r.total_collectives for r in result.sircl_arm.receipts)
    nccl_total = sum(r.total_collectives for r in result.nccl_arm.receipts)
    if sircl_total != nccl_total:
        errors.append(
            f"cross-arm total mismatch: sircl={sircl_total}, "
            f"nccl={nccl_total}"
        )
    # Goal 11: cross-rank counter synchronization validation.
    # SIRCL arm: sum of native_collectives across all ranks must equal
    # sum of total_collectives (every collective attributed to native).
    # NCCL-IB arm: sum of nccl_ib_collectives across all ranks must
    # equal sum of total_collectives (every collective attributed to IB).
    sircl_native_sum = sum(r.native_collectives for r in result.sircl_arm.receipts)
    if sircl_native_sum != sircl_total:
        errors.append(
            f"SIRCL arm native_collectives sum ({sircl_native_sum}) != "
            f"total_collectives sum ({sircl_total}) — not all "
            f"collectives attributed to native transport"
        )
    nccl_ib_sum = sum(r.nccl_ib_collectives for r in result.nccl_arm.receipts)
    if nccl_ib_sum != nccl_total:
        errors.append(
            f"NCCL arm nccl_ib_collectives sum ({nccl_ib_sum}) != "
            f"total_collectives sum ({nccl_total}) — not all "
            f"collectives attributed to NCCL-IB transport"
        )
    # Goal 11: hidden fallback on SIRCL arm is a hard failure —
    # fallback_collectives > 0 means NCCL was secretly used.
    for receipt in result.sircl_arm.receipts:
        if receipt.fallback_collectives > 0:
            errors.append(
                f"SIRCL arm rank {receipt.rank}: hidden fallback — "
                f"fallback_collectives={receipt.fallback_collectives} "
                f"(NCCL transport was used instead of native)"
            )
    # Goal 11: mixed transport on same receipt is a hard failure —
    # both native and nccl_ib > 0 means the rank used both transports
    # in the same arm, which is not a valid single-transport result.
    for arm_label, arm_result in [
        ("sircl", result.sircl_arm),
        ("nccl", result.nccl_arm),
    ]:
        for receipt in arm_result.receipts:
            if receipt.native_collectives > 0 and receipt.nccl_ib_collectives > 0:
                errors.append(
                    f"{arm_label} arm rank {receipt.rank}: mixed transport — "
                    f"native_collectives={receipt.native_collectives} "
                    f"and nccl_ib_collectives={receipt.nccl_ib_collectives} "
                    f"both > 0 (single transport required per receipt)"
                )
    # Cross-rank equality (Goal 9 requirement 2): after all-reduce,
    # the actual output and expected FP32 reference must be identical
    # across all ranks within each arm.  A per-rank hash that differs
    # from its peers is rejected — the all-reduce produces the same
    # result on every rank.
    for arm_label, arm_result in [
        ("sircl", result.sircl_arm),
        ("nccl", result.nccl_arm),
    ]:
        fp32_hashes = {r.expected_fp32_hash for r in arm_result.receipts if r.expected_fp32_hash}
        output_hashes = {r.actual_output_hash for r in arm_result.receipts if r.actual_output_hash}
        if len(fp32_hashes) > 1:
            errors.append(
                f"cross-rank {arm_label} expected_fp32_hash mismatch: "
                f"{sorted(fp32_hashes)}"
            )
        if len(output_hashes) > 1:
            errors.append(
                f"cross-rank {arm_label} actual_output_hash mismatch: "
                f"{sorted(output_hashes)}"
            )
        # rank_identity must be unique per rank within an arm.
        rank_identities = [r.rank_identity for r in arm_result.receipts if r.rank_identity]
        if len(rank_identities) != len(set(rank_identities)):
            errors.append(
                f"cross-rank {arm_label} rank_identity duplicates: "
                f"{rank_identities}"
            )

    # Validator recomputation of expected_fp32_hash (Goal 9 requirement 3):
    # The validator recomputes the expected FP32 hash from the
    # authoritative deterministic workload contract (seed, elements,
    # iterations, world_size), not from the receipt's self-asserted
    # hash.  If the receipt's hash does not match the recomputed hash,
    # the receipt is rejected.
    try:
        from tp4_numerical_audit import make_rank_input
        import hashlib as _hashlib
        import torch as _torch
        recomputed_fp32_hasher = _hashlib.sha256()
        for seq in range(plan_iters):
            cpu_inputs = [
                make_rank_input(seq, r, plan_elements)
                for r in range(ws)
            ]
            fp32_sum = _torch.stack([t.float() for t in cpu_inputs]).sum(dim=0)
            t = fp32_sum.cpu().contiguous()
            recomputed_fp32_hasher.update(t.numpy().tobytes())
        recomputed_fp32_hash = recomputed_fp32_hasher.hexdigest()
    except Exception:
        # If recomputation fails (e.g. torch not available), skip
        # this check but do not silently accept — the error is noted.
        recomputed_fp32_hash = None
        errors.append(
            "validator could not recompute expected_fp32_hash "
            "from deterministic workload contract"
        )
    if recomputed_fp32_hash is not None:
        for arm_label, arm_result in [
            ("sircl", result.sircl_arm),
            ("nccl", result.nccl_arm),
        ]:
            for receipt in arm_result.receipts:
                if receipt.expected_fp32_hash and receipt.expected_fp32_hash != recomputed_fp32_hash:
                    errors.append(
                        f"{arm_label} rank {receipt.rank}: "
                        f"expected_fp32_hash '{receipt.expected_fp32_hash[:16]}...' "
                        f"!= validator-recomputed '{recomputed_fp32_hash[:16]}...'"
                    )

    # Validator recomputation of run_contract_hash (Goal 9 requirement 3):
    # The validator recomputes the run_contract_hash from validator-owned
    # arm binding plus the exact validated per-rank plan projection.
    try:
        import hashlib as _hashlib
        import json as _json
        for arm_label, arm_result, plan_arm in [
            ("sircl", result.sircl_arm, plan.sircl_arm),
            ("nccl", result.nccl_arm, plan.nccl_arm),
        ]:
            binding = _ARM_BINDING[arm_result.arm_name]
            for receipt in arm_result.receipts:
                rl = plan_arm.rank_launches[receipt.rank]
                env_proj = _build_env_projection(
                    binding["selector"], receipt.rank, ws,
                    plan_iters, plan_elements,
                    transport=binding["transport"],
                )
                contract = {
                    "arm": binding["transport"],
                    "selector": binding["selector"],
                    "transport": binding["transport"],
                    "rank": receipt.rank,
                    "rank_identity": f"rank-{receipt.rank}-of-{ws}",
                    "iterations": plan_iters,
                    "elements": plan_elements,
                    "world_size": ws,
                    "seed_identity": "0x5A17+seq*WORLD_SIZE+rank",
                    "argv_projection": list(_CANONICAL_ARGV),
                    "env_projection": env_proj,
                    "probe_identity": _PINNED_PROBE_IDENTITY,
                    "binary_identity": _PINNED_BINARY_IDENTITY,
                    "topology": _PINNED_TOPOLOGY,
                    "workload": _PINNED_WORKLOAD,
                    "order": _PINNED_ORDER,
                }
                recomputed_contract_hash = _hashlib.sha256(
                    _json.dumps(contract, sort_keys=True).encode()
                ).hexdigest()
                if receipt.run_contract_hash and receipt.run_contract_hash != recomputed_contract_hash:
                    errors.append(
                        f"{arm_label} rank {receipt.rank}: "
                        f"run_contract_hash '{receipt.run_contract_hash[:16]}...' "
                        f"!= validator-recomputed '{recomputed_contract_hash[:16]}...'"
                    )
    except Exception as exc:
        # Goal 10 requirement 6: no fail-open.  Any inability to
        # recompute is a validation failure, not a silently-passed check.
        errors.append(
            f"validator could not recompute run_contract_hash: {exc}"
        )

    # Validate env allowlist for all ranks (Goal 9 requirement 4):
    # Reject extra environment variables that can alter code loading/execution.
    # Argv validation is already done in Step 4 above for every rank.
    for arm_label, arm in [("sircl", plan.sircl_arm), ("nccl", plan.nccl_arm)]:
        for idx, rl in enumerate(arm.rank_launches):
            # Check for env vars outside the allowlist.
            extra_env = set(rl.env_vars.keys()) - _ENV_ALLOWLIST
            if extra_env:
                errors.append(
                    f"{arm_label} arm rank_launch[{idx}] has env vars "
                    f"outside allowlist: {sorted(extra_env)}"
                )

    result.errors = errors
    result.valid = len(errors) == 0
    return errors


# ---------------------------------------------------------------------------
# Execution (executor seam)
# ---------------------------------------------------------------------------

def execute_arm(
    arm: ArmPlan,
    executor: ExecutorCallable | None,
    confirmation: str,
) -> ArmResult | None:
    """Execute one arm via the injected executor seam.

    Returns ``ArmResult`` with per-rank receipts, or ``None`` if
    the executor is absent (missing seam).
    """
    if executor is None:
        return None
    if confirmation != _CONFIRMATION_REQUIRED_RESPONSE:
        return None
    # The executor is an injected callable that takes (arm, confirmation)
    # and returns ArmResult. The public checkout provides no such executor.
    if hasattr(executor, "__call__"):
        return executor(arm, confirmation)  # type: ignore[operator]
    return None


# ---------------------------------------------------------------------------
# Four-rank selector consensus (Goal 10 requirement 1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankCapability:
    """One rank's transport capability for selector consensus.

    A rank reports its selector choice, whether a native session
    is available, and its identity (rank, arm, workload, binary).
    The consensus preflight uses these to ensure all ranks agree on
    the same transport before any collective.

    Goal 11 requirement 1: the preflight must verify exact rank set,
    one record per rank, identical selector/arm/workload/binary
    identity, and native-session capability on all ranks for the SIRCL arm.
    """

    rank: int
    selector: str
    native_session_available: bool
    arm: str = ""
    workload: str = ""
    binary_identity: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.rank, int) or isinstance(self.rank, bool):
            raise ValueError("rank must be an int")
        if self.rank < 0:
            raise ValueError("rank must be >= 0")
        if self.selector not in _VALID_SELECTORS:
            raise ValueError(
                f"selector must be one of {sorted(_VALID_SELECTORS)}, "
                f"got '{self.selector}'"
            )
        if not isinstance(self.native_session_available, bool):
            raise ValueError("native_session_available must be a bool")


class SelectorConsensusError(RuntimeError):
    """Raised when ranks cannot agree on a transport before any collective.

    Goal 11 requirement 1: one rank's unsupported/failure state cannot
    let ranks choose different transports.  If consensus is not reached,
    the run fails before any SIRCL or NCCL data collective is invoked.
    """


class ControlPlanePreflight:
    """Synchronized four-process control-plane preflight.

    Goal 11 requirement 1: a real synchronized four-process control-plane
    preflight that runs inside the production execution path, BEFORE any
    SIRCL or NCCL data collective.

    The control channel is NOT the transport under test.  In production,
    this uses a dedicated Gloo/TCP control group (or an external
    coordinator) on the management network, never the RoCE fabric
    being measured.  The control group is initialized before the data
    path measurement begins.

    In tests, the control channel is modeled by a callable that collects
    RankCapability records from all ranks and returns the consensus
    decision.  This faithfully models the barrier semantics: no rank
    proceeds until all ranks have reported and the decision is identical
    on every rank.

    Rules enforced: exact rank set, identical identity, native session
    check for SIRCL, deterministic fallback or abort, and no NCCL
    after native SIRCL begins.
    """

    def __init__(
        self,
        control_channel: object | None = None,
    ) -> None:
        self._control_channel = control_channel
        self._collected: list[RankCapability] = []
        self._decision: str | None = None
        self._aborted = False

    def report_capability(self, cap: RankCapability) -> None:
        """One rank reports its capability to the control plane."""
        self._collected.append(cap)

    def run_preflight(
        self,
        fallback_arm: str | None = None,
    ) -> str:
        """Run the synchronized preflight and return the consensus decision.

        All ranks must have reported via report_capability() before this
        is called.  In production, this is an all-reduce barrier on the
        control group: no rank proceeds until all have reported.

        Returns the consensus selector (or fallback arm) if all ranks
        agree.  Raises SelectorConsensusError if ranks disagree, if the
        rank set is wrong, or if identity fields mismatch.

        If fallback_arm is provided and the SIRCL arm is selected but
        any rank lacks native session, all ranks select the fallback arm
        instead of aborting.  If no fallback is provided, the run aborts.
        """
        capabilities = tuple(self._collected)

        if not capabilities:
            raise SelectorConsensusError(
                "no capabilities reported to control plane"
            )
        if len(capabilities) != AUTHORITATIVE_WORLD_SIZE:
            raise SelectorConsensusError(
                f"expected {AUTHORITATIVE_WORLD_SIZE} rank capabilities, "
                f"got {len(capabilities)}"
            )

        ranks = [cap.rank for cap in capabilities]
        if any(isinstance(r, bool) for r in ranks):
            raise SelectorConsensusError(
                f"boolean rank IDs in capabilities: {ranks}"
            )
        if len(set(ranks)) != len(ranks):
            raise SelectorConsensusError(
                f"duplicate ranks in capabilities: {sorted(ranks)}"
            )
        if set(ranks) != AUTHORITATIVE_RANKS:
            raise SelectorConsensusError(
                f"rank set must be {sorted(AUTHORITATIVE_RANKS)}, "
                f"got {sorted(ranks)}"
            )

        sorted_caps = tuple(sorted(capabilities, key=lambda c: c.rank))

        selectors = {cap.selector for cap in sorted_caps}
        if len(selectors) > 1:
            raise SelectorConsensusError(
                f"selector consensus failed: ranks selected different "
                f"transports: {sorted(selectors)}"
            )

        consensus_selector = sorted_caps[0].selector

        for field_name in ("arm", "workload", "binary_identity"):
            values = {getattr(cap, field_name) for cap in sorted_caps}
            if len(values) > 1:
                raise SelectorConsensusError(
                    f"identity consensus failed: ranks have different "
                    f"{field_name} values: {sorted(values)}"
                )

        if consensus_selector == SELECTOR_SIRCL:
            lacking_native = [
                cap.rank for cap in sorted_caps
                if not cap.native_session_available
            ]
            if lacking_native:
                if fallback_arm is not None:
                    self._decision = fallback_arm
                    return fallback_arm
                raise SelectorConsensusError(
                    f"selector=custom but rank(s) {lacking_native} lack "
                    f"native session: aborting before any collective "
                    f"(Goal 11: no rank-local fallback; all abort or "
                    f"all select fallback arm)"
                )

        self._decision = consensus_selector
        return consensus_selector

    @property
    def decision(self) -> str | None:
        return self._decision

    @property
    def aborted(self) -> bool:
        return self._aborted


def check_selector_consensus(
    capabilities: tuple[RankCapability, ...],
) -> str:
    """Backward-compat wrapper: run a one-shot consensus check.

    Goal 11 requirement 1: this now delegates to ControlPlanePreflight
    which enforces exact rank set, identity matching, and deterministic
    fallback.  The old check_selector_consensus() accepted four duplicate
    rank-0 records; that illusion of coverage is deleted.
    """
    preflight = ControlPlanePreflight()
    for cap in capabilities:
        preflight.report_capability(cap)
    return preflight.run_preflight()


# ---------------------------------------------------------------------------
# Plan rendering (human-readable)
# ---------------------------------------------------------------------------

def print_plan(plan: TwoArmPlan) -> str:
    """Render a human-readable plan."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("TWO-ARM ORCHESTRATOR PLAN")
    lines.append("=" * 72)
    lines.append("")

    mode_label = "DRY-RUN (no execution)" if plan.dry_run else "EXECUTION MODE"
    lines.append(f"Mode: {mode_label}")
    lines.append(f"Confirmation required: {plan.confirmation_required}")
    lines.append(f"Executor available: {plan.executor_available}")
    lines.append("")

    lines.append("Missing execute seam:")
    lines.append(
        "  No executor seam in the public checkout. The probe requires"
    )
    lines.append(
        "  CUDA, RDMA, and a live 4-rank process group across 4 hosts."
    )
    lines.append(
        "  --no-dry-run without an injected executor fails before mutation."
    )
    lines.append("")

    lines.append("Shared identity (must be identical across arms):")
    for key in sorted(plan.shared_identity):
        lines.append(f"  {key}: {plan.shared_identity[key]}")
    lines.append("")

    for arm in (plan.sircl_arm, plan.nccl_arm):
        lines.append("-" * 72)
        lines.append(f"Arm: {arm.arm_name}")
        lines.append(f"  Transport: {arm.transport}")
        lines.append(f"  Selector: {arm.selector}")
        lines.append(f"  Safety class: {arm.safety_class}")
        lines.append(f"  Timeout: {arm.timeout_seconds}s")
        lines.append(f"  Rank launches ({len(arm.rank_launches)}):")
        for rl in arm.rank_launches:
            lines.append(f"    rank {rl.rank} @ {rl.host}:")
            lines.append(f"      command: {' '.join(rl.command)}")
            lines.append("      env:")
            for ek in sorted(rl.env_vars):
                lines.append(f"        {ek}={rl.env_vars[ek]}")
        lines.append("")

    lines.append("-" * 72)
    lines.append("All-rank results: must collect receipts from all ranks.")
    lines.append("Partial failure: if any rank fails, no evidence published.")
    lines.append("")

    lines.append("-" * 72)
    lines.append("Confirmation prompt:")
    lines.append(f"  Prompt: '{_CONFIRMATION_PROMPT}'")
    lines.append(f"  Required response: '{_CONFIRMATION_REQUIRED_RESPONSE}'")
    lines.append("  Wrong confirmation = no execution.")
    lines.append("")

    lines.append("Fail-closed: no evidence artifact published on incomplete execution.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tracer-bullet two-arm orchestrator for the SIRCL-vs-NCCL "
            "numerical probe. Renders exact commands; executes nothing "
            "in dry-run mode."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry-run mode: render plan but execute nothing (default: True).",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Disable dry-run mode (requires an injected executor seam).",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=4,
        help="World size (default: 4).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of iterations (default: 1000).",
    )
    parser.add_argument(
        "--elements",
        type=int,
        default=6144,
        help="Number of elements per tensor (default: 6144).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per arm in seconds (default: 300).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the plan and exit without printing the full plan.",
    )
    parser.add_argument(
        "--site-profile",
        type=str,
        default=None,
        help="Path to a sanitized site profile JSON (required for --no-dry-run).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Exit codes:
      0 — dry-run plan valid.
      1 — plan invalid.
      2 — malformed input.
      3 — non-dry-run with no executor seam (missing-seam result).
    """
    try:
        args = _parse_args(argv)

        # Load site profile if provided.
        site_profile = None
        if args.site_profile:
            import json as _json
            import pathlib as _pathlib
            profile_path = _pathlib.Path(args.site_profile)
            if not profile_path.exists():
                print(
                    f"ERROR: site profile not found: {args.site_profile}",
                    file=sys.stderr,
                )
                return EXIT_MALFORMED
            site_profile = _json.loads(profile_path.read_text())

        sircl_spec = ArmSpec(
            transport=_TRANSPORT_SIRCL,
            selector_env_vars=_SIRCL_SELECTOR_ENVS,
            world_size=args.world_size,
            iterations=args.iterations,
            elements=args.elements,
            timeout_seconds=args.timeout,
        )
        nccl_spec = ArmSpec(
            transport=TRANSPORT_NCCL_IB,
            selector_env_vars=_NCCL_IB_SELECTOR_ENVS,
            world_size=args.world_size,
            iterations=args.iterations,
            elements=args.elements,
            timeout_seconds=args.timeout,
        )
        # Non-dry-run without site_profile/executor: fail before
        # mutation with a precise missing-seam result.
        if not args.dry_run and not site_profile:
            print(
                "MISSING EXECUTOR SEAM: --no-dry-run requires a "
                "sanitized site profile (--site-profile) to launch "
                "the probe across 4 hosts and collect per-rank "
                "receipts. No site profile was provided.",
                file=sys.stderr,
            )
            return EXIT_MISSING_SEAM

        plan = render_plan(
            sircl_spec, nccl_spec,
            dry_run=args.dry_run,
            site_profile=site_profile,
        )


        # Non-dry-run without executor: fail before mutation with
        # a precise missing-seam result, not a generic validation error.
        if not args.dry_run and not plan.executor_available:
            print(
                "MISSING EXECUTOR SEAM: --no-dry-run requires an injected "
                "executor to launch the probe across 4 hosts and collect "
                "per-rank receipts. No executor is available in the public "
                "checkout.",
                file=sys.stderr,
            )
            return EXIT_MISSING_SEAM

        errors = validate_plan(plan)

        if errors:
            for err in errors:
                print(f"VALIDATION ERROR: {err}", file=sys.stderr)
            return EXIT_INVALID

        if not args.validate_only:
            print(print_plan(plan))

        print("PLAN VALID")
        return EXIT_VALID

    except (ValueError, TypeError) as e:
        print(f"ERROR: malformed input: {e}", file=sys.stderr)
        return EXIT_MALFORMED
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else EXIT_MALFORMED
        if code != 0:
            print(f"ERROR: argparse exited with code {code}", file=sys.stderr)
            return EXIT_MALFORMED
        return code
    except Exception as e:
        print(f"ERROR: unexpected: {e}", file=sys.stderr)
        return EXIT_MALFORMED


if __name__ == "__main__":
    sys.exit(main())
