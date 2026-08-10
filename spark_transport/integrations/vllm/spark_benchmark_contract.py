"""Same-stack SIRCL-versus-NCCL benchmark evidence contract.

This module defines a structured, fail-closed benchmark contract for
comparing the custom SIRCL transport against NCCL (Socket or IB) on
the *same workload, same runtime, same topology, same identity*.

**Key design principle: evidence is non-forgeable or explicitly
not judged.**  The comparator never returns ``correct=True`` from
structural consistency alone.  Since no compatible producer exists
that emits independently recomputable per-rank/per-iteration raw
observations (the existing ``tp4_numerical_audit.py`` prints
aggregate metrics on rank 0 only), correctness is always
``not_judged``.  The artifact is structurally validated against a
versioned tolerance policy, but structural validation is not
correctness.

The artifact is a raw-evidence JSON file containing per-rank
numerical observations.  The comparator loads the file from beneath a
declared evidence root, recomputes its SHA-256, validates the schema,
and checks structural consistency (tolerances, finiteness, bindings,
workload shape).  A free-form ``PASS`` string is never accepted,
and structural consistency never yields ``correct=True``.

The artifact is bound to its arm role (SIRCL or NCCL), transport,
allowlisted selector configuration, workload, topology, binary
identity, and runtime identity.

**Unverified declarations:** Repository/commit fields
(``source_commit``, ``runtime_commit``), clocks/power, topology/
device/GID/interface fields, and counter totals are caller-declared.
They are structurally validated (format, schema, keys, types) but
NOT verified against authoritative receipts or the actual checkout.
They are excluded from correctness/performance judgment.
Counter receipts are structurally validated but not used for
judgment — a real counter receipt needs per-rank before/after
snapshots, a synchronized bounded window, source identity/hash,
overflow/fatal counters, and transport-log binding.

**Five separate verdicts:**

1. ``valid`` — both records pass structural validation.
2. ``comparable`` — both records are individually valid AND have
   matching identity fields (including transport_library_hash) and
   matching workload/runtime fields, AND arm roles are enforced
   (arg1=SIRCL, arg2=NCCL, selector hashes differ).
3. ``performance`` — optional; if a threshold policy is declared,
   the SIRCL arm must meet it.  Without raw timing samples with a
   named timing boundary and clock source, performance is
   ``not_judged``.
4. ``correctness_verdict`` — ``structurally_consistent`` when both
   arms' raw evidence artifacts pass all structural derivations
   (tolerance policy, finiteness, bindings, shape);
   ``not_judged`` when artifacts are absent or not comparable;
   ``failed`` when derivation raises a contract error.  This is a
   structural label, NOT a correctness verdict.
5. ``correct`` — always ``False`` until a compatible producer
   emits independently recomputable evidence.  No free
   ``correct=True`` is ever returned.

The module does **not** invent results, fabricate hashes, or promote
a comparison to "passed" without an explicit threshold.  A free
``correct=true`` assertion is forbidden.  Structural consistency is
not correctness.

**CLI exit codes (review-8):**
0 — --validate-only mode: both records passed structural schema validation.
    This is the ONLY path to exit 0.  Compare/claim mode never exits 0.
1 — compare mode: indeterminate or incorrect benchmark evidence
    (not comparable, performance FAIL, correctness_verdict=failed or
    not_judged, or correct=false which is always the case since no
    compatible producer exists).
2 — malformed/contract-invalid input (BenchmarkContractError at
    parse/validation time, or unexpected exception).

**Live producer status:** The existing ``tp4_numerical_audit.py``
prints aggregate metrics on rank 0 only.  It does not currently emit
per-rank, per-iteration raw evidence in this schema.  The live
producer seam (modifying the audit script to emit raw evidence per
rank) is **goal-2 work**.  Until then, ``correctness_verdict`` is
``structurally_consistent`` at best (structural label only) and
``correct`` is always ``False``.  This module defines the offline
schema and comparator now so the producer has a contract to target.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import sys
from dataclasses import asdict, dataclass
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_LANES = frozenset({"public-functional"})
VALID_TRANSPORTS = frozenset({"sircl", "nccl_socket", "nccl_ib"})
NCCL_TRANSPORTS = frozenset({"nccl_socket", "nccl_ib"})

# TP4-only contract.
VALID_TOPOLOGIES = frozenset({"tp4_switchless_ring"})

# SHA-256: exactly 64 lowercase hexadecimal characters
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Full 40-hex git commit SHA-1
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

# Registry digest: "sha256:" followed by 64 hex
_REGISTRY_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

# Local image ID: "sha256:" + 64 hex (normalized) or "absent"
_LOCAL_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$|^absent$")

# Evidence label: nonempty stable string
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./-]{0,127}$")

# Clocks/power policy
_POLICY_VALUES = frozenset({"controlled", "uncontrolled", "not-claimed"})

# Maximum artifact file size: 64 MiB — raw per-rank/per-iteration
# evidence for 4 ranks × 1000 iterations is well under this.
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024

# Raw-evidence artifact schema
RAW_EVIDENCE_SCHEMA = "tp4_raw_evidence/v2"

# Tolerance policy: versioned, pinned thresholds.  These are
# structural validation limits — a caller-authored artifact with
# mismatch_count=6144000 and mae=1e30 must NOT pass these.  The
# tolerances are dtype-specific: BF16 thresholds cannot silently
# govern FP32/FP16/INT32.  ``mismatch_count`` is defined as the
# count of element-wise comparisons where the observed output differs
# from the FP32 reference truth **rounded once to the target dtype**
# (e.g. BF16).  It must be zero for any dtype — a nonzero count
# means the observed output is not the correctly-rounded result.
#
# ``outside_tolerance_count`` is governed by an explicit tolerance
# policy (TOLERANCE_POLICY_VERSION).  It counts elements whose
# absolute error vs the **unrounded** FP32 truth exceeds the
# dtype-specific threshold.  It may be nonzero but is bounded by
# total comparisons.  If no tolerance policy is active, it must be
# reported as ``NOT-JUDGED`` (the sentinel ``-1``).
#
# These do NOT prove correctness — they only filter forged artifacts.
TOLERANCE_POLICY_VERSION = "bf16_fixed_abs_v1"
_MAX_MISMATCH_COUNT = 0  # mismatches vs FP32 truth rounded to target dtype; must be zero

# Dtype-specific tolerance limits.  Each dtype has distinct
# precision characteristics; a single BF16 threshold cannot govern
# FP32 (which should be near-exact) or INT32 (integer-exact).
_DTYPE_TOLERANCES: dict[str, dict[str, float]] = {
    "torch.bfloat16": {
        "max_mae": 0.01,
        "max_rmse": 0.02,
        "max_abs_error": 0.1,
    },
    "torch.float16": {
        "max_mae": 0.005,
        "max_rmse": 0.01,
        "max_abs_error": 0.05,
    },
    "torch.float32": {
        "max_mae": 1e-6,
        "max_rmse": 1e-6,
        "max_abs_error": 1e-5,
    },
    "torch.int32": {
        "max_mae": 0.0,
        "max_rmse": 0.0,
        "max_abs_error": 0,
    },
}

# Dtype → element size in bytes for bytes_per_collective validation.
_DTYPE_BYTES: dict[str, int] = {
    "torch.bfloat16": 2,
    "torch.float16": 2,
    "torch.float32": 4,
    "torch.int32": 4,
}

# Maximum safe float for arithmetic without overflow.  Integers above
# this cannot be used in float arithmetic without precision loss.
_MAX_SAFE_FLOAT_INT = 2**53

# Selector allowlist: exact effective-environment mappings only.
# Each transport role has exactly one canonical set of environment
# variable bindings.  Arbitrary selector text is rejected.
#
# IMPORTANT: The effective selector variable for the TP4 adapter is
# ``VLLM_SPARK_TP4_MODE`` (read by ``spark_tp4_backend.py`` line 131
# and ``sitecustomize.py`` line 196).  The older ``SPARK_TP4_MODE``
# spelling is NOT read by the target process and is therefore
# ineffective — accepting it would let a record declare a selector
# that the runtime never consumed.
#
# NCCL arms must explicitly disable SIRCL by requiring
# ``VLLM_SPARK_TP4_MODE=disabled`` plus consistent NCCL controls:
#   - nccl_socket: NCCL_NET=Socket with NCCL_IB_DISABLE=1
#   - nccl_ib:     NCCL_NET=IB with NCCL_IB_DISABLE=0
# Inherited custom TP4 mode (VLLM_SPARK_TP4_MODE=custom) on an NCCL
# arm is rejected.  Contradictory or incomplete mappings (e.g. missing
# NCCL_IB_DISABLE, or NCCL_NET=IB with NCCL_IB_DISABLE=1) are
# rejected because the env_vars set must EXACTLY match the allowlist
# entry — no extra, no missing.
_SELECTOR_ALLOWLIST: dict[str, frozenset[str]] = {
    "sircl": frozenset({
        "VLLM_SPARK_TP4_MODE=custom",
    }),
    "nccl_socket": frozenset({
        "VLLM_SPARK_TP4_MODE=disabled",
        "NCCL_NET=Socket",
        "NCCL_IB_DISABLE=1",
    }),
    "nccl_ib": frozenset({
        "VLLM_SPARK_TP4_MODE=disabled",
        "NCCL_NET=IB",
        "NCCL_IB_DISABLE=0",
    }),
}

# Timing boundary labels for performance evidence
_VALID_TIMING_BOUNDARIES = frozenset({
    "host_submission",
    "gpu_elapsed",
    "transport_completion",
    "end_to_end",
})

# Clock source labels
_VALID_CLOCK_SOURCES = frozenset({
    "cuda_event",
    "chrono_steady",
    "chrono_system",
})

# Verifier implementation identity: tp4_numerical_audit.py is the
# existing audit script.  Its SHA-256 is recomputed at comparison
# time from the actual file on disk — not from a caller-declared
# string.  See TRUSTED_VERIFIER_HASH below.
#
# NOTE: The verifier hash proves only that the script bytes are
# unchanged — it is an implementation-byte check, NOT source-commit
# anchoring.  A copied fake repo with the same file hash does not
# establish provenance.  The artifact's provenance is bound to the
# record's identity (source_commit, runtime_commit, image, transport),
# not to the verifier hash alone.
VERIFIER_NAME = "tp4_numerical_audit.py"
# At comparison time, the module recomputes the hash of the actual
# verifier file on disk and compares it to this trusted hash.  This
# is an implementation-byte check: it proves the script is unchanged,
# NOT that it produced any given artifact or that the source commit
# matches the actual checkout.
#
# We use a checked-in trusted hash rather than runtime-lock.json
# because runtime-lock.json pins the container build toolchain
# (vLLM, Torch, NCCL patches, model config), not the numerical
# audit script.  The audit script lives in spark_transport/ and is
# part of the source tree, so its hash is an implementation-byte
# check, not a source-commit anchor.
TRUSTED_VERIFIER_HASH = "bc12ab0a987f3b950021dfb9ebeeae912d2cc3ab07943490731a73aea427300c"

# Path to the verifier implementation relative to the repository root.
_VERIFIER_REL_PATH = "spark_transport/integrations/vllm/tp4_numerical_audit.py"


class BenchmarkContractError(ValueError):
    """Raised when a benchmark contract record or artifact is invalid."""



def _is_int_not_bool(val: Any) -> bool:
    """True if val is an int but not a bool (bool is a subclass of int)."""
    return isinstance(val, int) and not isinstance(val, bool)
def _is_finite_number(val: Any) -> bool:
    """True if val is a finite int or float but not a bool."""
    if isinstance(val, bool):
        return False
    if isinstance(val, int):
        try:
            float(val)
            return True
        except OverflowError:
            return False
    if isinstance(val, float):
        try:
            return val == val and val != float("inf") and val != float("-inf")
        except OverflowError:
            return False
    return False


def _is_finite_float(val: Any) -> bool:
    """True if val is a finite float (or int), not bool, not NaN/Inf."""
    if isinstance(val, bool):
        return False
    if not isinstance(val, (int, float)):
        return False
    try:
        fval = float(val)
    except (OverflowError, ValueError):
        return False
    return not (math.isnan(fval) or math.isinf(fval))


def _validate_sha256(val: Any, name: str) -> str:
    if not isinstance(val, str):
        raise BenchmarkContractError(
            f"{name} must be a 64-hex SHA-256 string, got {type(val).__name__}"
        )
    if not _SHA256_RE.match(val):
        raise BenchmarkContractError(
            f"{name} must be exactly 64 lowercase hex chars (SHA-256), "
            f"got '{val[:20]}'"
        )
    return val


def _validate_commit(val: Any, name: str) -> str:
    """Require full 40-hex git commit SHA-1."""
    if not isinstance(val, str):
        raise BenchmarkContractError(
            f"{name} must be a git commit SHA string, got {type(val).__name__}"
        )
    if not _COMMIT_RE.match(val):
        raise BenchmarkContractError(
            f"{name} must be a full 40-hex git commit SHA-1, got '{val[:20]}'"
        )
    return val


def _validate_registry_digest(val: Any, name: str) -> str:
    """Validate a registry digest: 'sha256:<64hex>' or 'absent'."""
    if not isinstance(val, str):
        raise BenchmarkContractError(
            f"{name} must be a registry digest string, got {type(val).__name__}"
        )
    if val == "absent":
        return val
    if not _REGISTRY_DIGEST_RE.match(val):
        raise BenchmarkContractError(
            f"{name} must be 'sha256:<64hex>' (registry digest) or 'absent', "
            f"got '{val[:20]}'"
        )
    return val


def _validate_local_image_id(val: Any, name: str) -> str:
    """Validate a local image ID: 'sha256:<64hex>' (normalized) or 'absent'."""
    if not isinstance(val, str):
        raise BenchmarkContractError(
            f"{name} must be a local image ID string, got {type(val).__name__}"
        )
    if not _LOCAL_IMAGE_ID_RE.match(val):
        raise BenchmarkContractError(
            f"{name} must be 'sha256:<64hex>' (local image ID) or 'absent', "
            f"got '{val[:20]}'"
        )
    return val


def _validate_label(val: Any, name: str) -> str:
    if not isinstance(val, str):
        raise BenchmarkContractError(
            f"{name} must be a nonempty stable label string, "
            f"got {type(val).__name__}"
        )
    if not _LABEL_RE.match(val):
        raise BenchmarkContractError(
            f"{name} must be a nonempty stable label (alphanumeric, "
            f"'.', '/', '-', '_'), got '{val[:20]}'"
        )
    return val


def _validate_string_field(val: Any, name: str) -> str:
    if not isinstance(val, str) or not val:
        raise BenchmarkContractError(f"{name} must be a non-empty string")
    return val


def _validate_policy(val: Any, name: str) -> str:
    if not isinstance(val, str):
        raise BenchmarkContractError(
            f"{name} must be a string, got {type(val).__name__}"
        )
    if val not in _POLICY_VALUES:
        raise BenchmarkContractError(
            f"{name} must be one of {sorted(_POLICY_VALUES)}, got '{val}'"
        )
    return val


def _require_keys(
    data: Any, required: set[str], name: str,
) -> None:
    """Exact-key validation: reject missing and extra keys with contract errors."""
    if not isinstance(data, dict):
        raise BenchmarkContractError(
            f"{name} must be a dict, got {type(data).__name__}"
        )
    missing = required - set(data.keys())
    extra = set(data.keys()) - required
    if missing:
        raise BenchmarkContractError(
            f"missing {name} fields: {sorted(missing)}"
        )
    if extra:
        raise BenchmarkContractError(
            f"unexpected {name} fields: {sorted(extra)}"
        )


# ---------------------------------------------------------------------------
# Structured TP4 edge identity with exact topology enforcement
# ---------------------------------------------------------------------------

# TP4 schedule from SIRCL_BASELINE.md and tp4_schedule.cpp:
# round 0: peer = r XOR 1, device index 0  (0<->1, 2<->3)
# round 1: peer = r XOR 3, device index 1  (0<->3, 1<->2)
_TP4_EXPECTED_PEERS: dict[tuple[int, int], int] = {
    (0, 0): 1, (1, 0): 0, (2, 0): 3, (3, 0): 2,
    (0, 1): 3, (1, 1): 2, (2, 1): 1, (3, 1): 0,
}


@dataclass(frozen=True)
class Tp4EdgeIdentity:
    """Identity of one edge in one round for one rank.

    TP4 has exactly 2 rounds and 4 ranks.  Each (rank, round) pair
    has one edge with: the peer rank, the device name, the GID, and
    the peer interface.

    The device index (round 0 → device 0, round 1 → device 1) is
    enforced by the TP4 schedule but is NOT validated here against
    a hardcoded index — the ``device`` field carries the actual HCA
    device name (e.g. ``rocep1s0f0``).  The schedule binds round
    to device index in ``tp4_schedule.cpp``; this contract enforces
    that round 0 edges use one device and round 1 edges use a
    *different* device (two distinct direct-cable devices per rank).
    """

    rank: int
    round: int
    peer_rank: int
    device: str
    gid: int
    peer_interface: str

    # HCA device name pattern: roce/ib prefix + bus + function
    _DEVICE_RE = re.compile(r"^(roce?p|ib|mlx)\w+s\d+f\d+$")
    # Peer interface: IPv4 address
    _IPV4_RE = re.compile(
        r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    )

    def __post_init__(self) -> None:
        if not _is_int_not_bool(self.rank) or not 0 <= self.rank <= 3:
            raise BenchmarkContractError(
                f"edge.rank must be int in [0,3], got {self.rank}"
            )
        if not _is_int_not_bool(self.round) or not 0 <= self.round <= 1:
            raise BenchmarkContractError(
                f"edge.round must be int in [0,1], got {self.round}"
            )
        if not _is_int_not_bool(self.peer_rank) or not 0 <= self.peer_rank <= 3:
            raise BenchmarkContractError(
                f"edge.peer_rank must be int in [0,3], got {self.peer_rank}"
            )
        if self.peer_rank == self.rank:
            raise BenchmarkContractError(
                f"edge.peer_rank must differ from rank, got rank={self.rank}"
            )
        if not isinstance(self.device, str) or not self.device:
            raise BenchmarkContractError(
                f"edge.device must be a nonempty string, got {self.device!r}"
            )
        # Device must match HCA device name pattern (e.g. rocep1s0f0)
        if not self._DEVICE_RE.match(self.device):
            raise BenchmarkContractError(
                f"edge.device must match HCA device pattern "
                f"(roce?p|ib|mlx)\\w+s\\d+f\\d+, got '{self.device}'"
            )
        if not _is_int_not_bool(self.gid) or self.gid < 0:
            raise BenchmarkContractError(
                f"edge.gid must be a non-negative int, got {self.gid}"
            )
        if not isinstance(self.peer_interface, str) or not self.peer_interface:
            raise BenchmarkContractError(
                f"edge.peer_interface must be nonempty, got "
                f"{self.peer_interface!r}"
            )
        # Peer interface must be a valid IPv4 address
        if not self._IPV4_RE.match(self.peer_interface):
            raise BenchmarkContractError(
                f"edge.peer_interface must be an IPv4 address, "
                f"got '{self.peer_interface}'"
            )


def _validate_tp4_edges(edges: Any) -> tuple[Tp4EdgeIdentity, ...]:
    """Validate TP4 edge identity: exactly 8 edges (4 ranks × 2 rounds).

    Enforces exact TP4 topology — round 0 pairs 0<->1 and 2<->3;
    round 1 pairs 0<->3 and 1<->2.  Verifies reciprocal peer
    coverage, perfect matchings, and two distinct direct-cable
    devices per rank (round 0 device ≠ round 1 device for each rank).
    Rejects every inconsistent edge set.
    """
    if not isinstance(edges, (list, tuple)):
        raise BenchmarkContractError(
            "identity.tp4_edges must be a list or tuple, "
            f"got {type(edges).__name__}"
        )
    if len(edges) != 8:
        raise BenchmarkContractError(
            f"tp4_edges must have exactly 8 edges (4 ranks × 2 rounds), "
            f"got {len(edges)}"
        )
    validated: list[Tp4EdgeIdentity] = []
    seen_pairs: set[tuple[int, int]] = set()
    for i, e in enumerate(edges):
        if isinstance(e, Tp4EdgeIdentity):
            validated.append(e)
        elif isinstance(e, dict):
            _require_keys(
                e,
                {"rank", "round", "peer_rank", "device", "gid",
                 "peer_interface"},
                f"tp4_edges[{i}]",
            )
            try:
                validated.append(Tp4EdgeIdentity(
                    rank=e["rank"], round=e["round"],
                    peer_rank=e["peer_rank"], device=e["device"],
                    gid=e["gid"], peer_interface=e["peer_interface"],
                ))
            except BenchmarkContractError:
                raise
            except (TypeError, KeyError) as exc:
                raise BenchmarkContractError(
                    f"tp4_edges[{i}]: {exc}"
                ) from exc
        else:
            raise BenchmarkContractError(
                f"tp4_edges[{i}] must be a dict or Tp4EdgeIdentity, "
                f"got {type(e).__name__}"
            )
        pair = (validated[-1].rank, validated[-1].round)
        if pair in seen_pairs:
            raise BenchmarkContractError(
                f"duplicate (rank, round) pair: {pair}"
            )
        seen_pairs.add(pair)

    # Verify exact coverage: every (rank, round) in {0..3}×{0,1}
    expected = {(r, rnd) for r in range(4) for rnd in range(2)}
    if seen_pairs != expected:
        missing = expected - seen_pairs
        extra = seen_pairs - expected
        raise BenchmarkContractError(
            f"tp4_edges coverage mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    # Enforce exact TP4 topology — peer_rank must match the schedule
    for edge in validated:
        expected_peer = _TP4_EXPECTED_PEERS.get((edge.rank, edge.round))
        if expected_peer is None or edge.peer_rank != expected_peer:
            raise BenchmarkContractError(
                f"tp4_edges: rank={edge.rank} round={edge.round} has "
                f"peer_rank={edge.peer_rank}, expected {expected_peer} "
                f"(TP4 schedule: round 0 pairs r^1, round 1 pairs r^3)"
            )

    # Enforce reciprocal edges
    edge_map: dict[tuple[int, int], Tp4EdgeIdentity] = {
        (e.rank, e.round): e for e in validated
    }
    for edge in validated:
        reciprocal = edge_map.get((edge.peer_rank, edge.round))
        if reciprocal is None or reciprocal.peer_rank != edge.rank:
            raise BenchmarkContractError(
                f"tp4_edges: rank={edge.rank} round={edge.round} peers "
                f"{edge.peer_rank} but rank={edge.peer_rank} round="
                f"{edge.round} does not peer back to {edge.rank} "
                f"(non-reciprocal edge)"
            )

    # Enforce two distinct direct-cable devices per rank:
    # round 0 device ≠ round 1 device for each rank
    for rank in range(4):
        r0 = edge_map.get((rank, 0))
        r1 = edge_map.get((rank, 1))
        if r0 is not None and r1 is not None and r0.device == r1.device:
            raise BenchmarkContractError(
                f"tp4_edges: rank={rank} uses device '{r0.device}' "
                f"for both rounds — TP4 requires two distinct "
                f"direct-cable devices per rank"
            )

    return tuple(validated)


# ---------------------------------------------------------------------------
# Structured clocks/power settings (not free strings)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClocksPowerSettings:
    """Structured clocks/power identity.

    Replaces free-form ``clocks_details``/``power_details`` strings.
    When ``policy`` is ``controlled``, requires canonical settings
    fields plus a declaration SHA-256.  When ``uncontrolled`` or
    ``not-claimed``, requires explicit detail semantics (nonempty
    ``description``).  A one-character string cannot establish
    controlled identity.
    """
    policy: str
    # Controlled: canonical settings + caller declaration hash.
    # Renamed from receipt_sha256: this is a caller-declared hash,
    # not an authoritative receipt/attestation.
    gpu_clock_mhz: int | None = None
    power_limit_w: int | None = None
    declaration_sha256: str | None = None
    # Uncontrolled/not-claimed: explicit description
    description: str | None = None

    def __post_init__(self) -> None:
        _validate_policy(self.policy, "clocks_power.policy")
        if self.policy == "controlled":
            if not _is_int_not_bool(self.gpu_clock_mhz) or self.gpu_clock_mhz <= 0:
                raise BenchmarkContractError(
                    f"clocks_power.gpu_clock_mhz must be a positive int "
                    f"when controlled, got {self.gpu_clock_mhz}"
                )
            if not _is_int_not_bool(self.power_limit_w) or self.power_limit_w <= 0:
                raise BenchmarkContractError(
                    f"clocks_power.power_limit_w must be a positive int "
                    f"when controlled, got {self.power_limit_w}"
                )
            if not isinstance(self.declaration_sha256, str):
                raise BenchmarkContractError(
                    "clocks_power.declaration_sha256 must be a string "
                    "when controlled"
                )
            if not _SHA256_RE.match(self.declaration_sha256):
                raise BenchmarkContractError(
                    f"clocks_power.declaration_sha256 must be a 64-hex "
                    f"SHA-256 when controlled, got "
                    f"'{self.declaration_sha256[:20]}'"
                )
        else:
            # uncontrolled or not-claimed: require explicit description
            if not isinstance(self.description, str) or not self.description:
                raise BenchmarkContractError(
                    f"clocks_power.description must be a nonempty string "
                    f"when policy={self.policy}"
                )


@dataclass(frozen=True)
class SelectorConfig:
    """Canonical transport selector configuration for one arm.

    Replaces a bare ``transport_selector_hash`` string.  The
    comparator recomputes SHA-256 from the canonical JSON of this
    object and verifies it matches ``selector_hash``.  The
    ``transport_role`` must match the arm's SIRCL/NCCL role.

    ``env_vars`` is a frozenset of ``KEY=VALUE`` strings representing
    the exact effective environment for the target process.  It must
    exactly match the allowlisted set for the transport role — no
    extra, no missing variables.
    """

    transport_role: str   # "sircl" or "nccl_socket" / "nccl_ib"
    env_vars: frozenset[str]  # e.g. frozenset({"VLLM_SPARK_TP4_MODE=custom"})
    selector_hash: str  # SHA-256 of canonical JSON of transport_role + env_vars

    def __post_init__(self) -> None:
        if self.transport_role not in VALID_TRANSPORTS:
            raise BenchmarkContractError(
                f"selector.transport_role must be one of "
                f"{sorted(VALID_TRANSPORTS)}, got '{self.transport_role}'"
            )
        if not isinstance(self.env_vars, frozenset):
            # Accept any iterable of strings, coerce to frozenset
            if isinstance(self.env_vars, (set, list, tuple)):
                object.__setattr__(
                    self, "env_vars", frozenset(self.env_vars))
            else:
                raise BenchmarkContractError(
                    f"selector.env_vars must be a frozenset of "
                    f"'KEY=VALUE' strings, got {type(self.env_vars).__name__}"
                )
        for ev in self.env_vars:
            if not isinstance(ev, str) or "=" not in ev:
                raise BenchmarkContractError(
                    f"selector.env_vars entries must be 'KEY=VALUE' "
                    f"strings, got {ev!r}"
                )
        # Enforce allowlisted selector configuration — arbitrary
        # selector text is rejected.  The env_vars set must EXACTLY
        # match the allowlist entry: no extra, no missing.
        allowed = _SELECTOR_ALLOWLIST.get(self.transport_role, frozenset())
        if self.env_vars != allowed:
            raise BenchmarkContractError(
                f"selector.env_vars {sorted(self.env_vars)} "
                f"does not exactly match the allowlisted configuration "
                f"for transport_role '{self.transport_role}'.  "
                f"Required (exact, no more, no less): "
                f"{sorted(allowed)}"
            )
        _validate_sha256(self.selector_hash, "selector.selector_hash")
        # Verify hash matches canonical recomputation
        canonical = json.dumps({
            "transport_role": self.transport_role,
            "env_vars": sorted(self.env_vars),
        }, sort_keys=True)
        recomputed = hashlib.sha256(canonical.encode()).hexdigest()
        if recomputed != self.selector_hash:
            raise BenchmarkContractError(
                f"selector.selector_hash mismatch: declared "
                f"'{self.selector_hash[:16]}...' but recomputed "
                f"'{recomputed[:16]}...' from transport_role="
                f"'{self.transport_role}' env_vars="
                f"{sorted(self.env_vars)}"
            )

    @property
    def selector_config(self) -> str:
        """Backward-compatible single-string view for rendering."""
        return "; ".join(sorted(self.env_vars))


def _make_selector_config(
    transport_role: str,
    env_vars: frozenset[str] | set[str] | list[str] | tuple[str, ...] | None = None,
) -> SelectorConfig:
    """Build a SelectorConfig with a correctly computed hash."""
    if env_vars is None:
        env_vars = _SELECTOR_ALLOWLIST.get(transport_role, frozenset())
    env_vars = frozenset(env_vars)
    canonical = json.dumps({
        "transport_role": transport_role,
        "env_vars": sorted(env_vars),
    }, sort_keys=True)
    return SelectorConfig(
        transport_role=transport_role,
        env_vars=env_vars,
        selector_hash=hashlib.sha256(canonical.encode()).hexdigest(),
    )


# ---------------------------------------------------------------------------
# Identity spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IdentitySpec:
    """Fields that must be identical across both arms of a comparison.

    transport_library_hash must **match** because this is a same-stack
    comparison.  transport_selector_hash **must differ** — it encodes
    the different selector/config.

    The ``selector`` field carries structured selector configuration
    whose SHA-256 is recomputed and whose ``transport_role`` must
    match the arm's transport.

    The ``clocks_power`` field carries structured settings, not free
    strings.
    """

    schema_version: str
    source_commit: str
    runtime_commit: str
    registry_digest: str
    local_image_id: str
    transport_library_hash: str
    selector: SelectorConfig
    torch_version: str
    vllm_version: str
    cuda_version: str
    driver_version: str
    model_repository: str
    model_revision: str
    model_config_sha256: str
    tp4_edges: tuple[Tp4EdgeIdentity, ...]
    clocks_power: ClocksPowerSettings
    evidence_run_id: str

    def __post_init__(self) -> None:
        _validate_string_field(self.schema_version, "identity.schema_version")
        _validate_commit(self.source_commit, "identity.source_commit")
        _validate_commit(self.runtime_commit, "identity.runtime_commit")
        _validate_registry_digest(
            self.registry_digest, "identity.registry_digest")
        _validate_local_image_id(
            self.local_image_id, "identity.local_image_id")
        # At least one image identity must be non-absent
        if self.registry_digest == "absent" and self.local_image_id == "absent":
            raise BenchmarkContractError(
                "at least one of registry_digest or local_image_id "
                "must be non-absent"
            )
        _validate_sha256(self.transport_library_hash,
                         "identity.transport_library_hash")
        # selector is validated by SelectorConfig.__post_init__
        _validate_string_field(self.torch_version, "identity.torch_version")
        _validate_string_field(self.vllm_version, "identity.vllm_version")
        _validate_string_field(self.cuda_version, "identity.cuda_version")
        _validate_string_field(self.driver_version, "identity.driver_version")
        _validate_string_field(self.model_repository,
                               "identity.model_repository")
        _validate_commit(self.model_revision, "identity.model_revision")
        _validate_sha256(self.model_config_sha256,
                         "identity.model_config_sha256")
        object.__setattr__(
            self, "tp4_edges", _validate_tp4_edges(self.tp4_edges))
        _validate_label(self.evidence_run_id, "identity.evidence_run_id")

    @property
    def transport_selector_hash(self) -> str:
        """Convenience: the selector hash from the structured config."""
        return self.selector.selector_hash


# ---------------------------------------------------------------------------
# Workload and runtime specs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorkloadSpec:
    shape: tuple[int, ...]
    dtype: str
    bytes_per_collective: int
    collective_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.shape, (list, tuple)):
            raise BenchmarkContractError("workload.shape must be a tuple")
        shape = tuple(self.shape)
        if len(shape) == 0:
            raise BenchmarkContractError(
                "workload.shape must be non-empty"
            )
        for i, dim in enumerate(shape):
            if not _is_int_not_bool(dim) or dim <= 0:
                raise BenchmarkContractError(
                    f"workload.shape[{i}] must be a positive int, got {dim}"
                )
            if dim > _MAX_SAFE_FLOAT_INT:
                raise BenchmarkContractError(
                    f"workload.shape[{i}] ({dim}) exceeds max safe "
                    f"integer ({_MAX_SAFE_FLOAT_INT})"
                )
        object.__setattr__(self, "shape", shape)
        if self.dtype not in ("torch.bfloat16", "torch.float32",
                              "torch.int32", "torch.float16"):
            raise BenchmarkContractError(
                f"workload.dtype must be a known torch dtype string, "
                f"got '{self.dtype}'"
            )
        if not _is_int_not_bool(self.bytes_per_collective) or \
                self.bytes_per_collective <= 0:
            raise BenchmarkContractError(
                "workload.bytes_per_collective must be a positive int"
            )
        # Validate bytes_per_collective from dtype and shape:
        # element_count * dtype_bytes must equal bytes_per_collective.
        # Compute element_count safely — check for overflow before
        # byte calculation.
        element_count = 1
        for dim in shape:
            if element_count > _MAX_SAFE_FLOAT_INT // dim:
                raise BenchmarkContractError(
                    f"workload.shape product overflow: element_count "
                    f"exceeds {_MAX_SAFE_FLOAT_INT} — shape {shape} "
                    f"is too large"
                )
            element_count *= dim
        expected_bytes = element_count * _DTYPE_BYTES[self.dtype]
        if expected_bytes > _MAX_SAFE_FLOAT_INT:
            raise BenchmarkContractError(
                f"workload.bytes_per_collective overflow: "
                f"{element_count} * {_DTYPE_BYTES[self.dtype]} "
                f"exceeds {_MAX_SAFE_FLOAT_INT}"
            )
        if self.bytes_per_collective != expected_bytes:
            raise BenchmarkContractError(
                f"workload.bytes_per_collective ({self.bytes_per_collective}) "
                f"must equal shape_product * dtype_bytes "
                f"({element_count} * {_DTYPE_BYTES[self.dtype]} "
                f"= {expected_bytes}) for dtype='{self.dtype}'"
            )
        if not _is_int_not_bool(self.collective_count) or \
                self.collective_count <= 0:
            raise BenchmarkContractError(
                "workload.collective_count must be a positive int"
            )
        if self.collective_count > _MAX_SAFE_FLOAT_INT:
            raise BenchmarkContractError(
                f"workload.collective_count ({self.collective_count}) "
                f"exceeds max safe integer ({_MAX_SAFE_FLOAT_INT})"
            )


@dataclass(frozen=True)
class RuntimeSpec:
    lane: str
    transport: str
    topology: str
    world_size: int
    warmup_iterations: int
    sample_iterations: int

    def __post_init__(self) -> None:
        if self.lane not in VALID_LANES:
            raise BenchmarkContractError(
                f"lane must be {sorted(VALID_LANES)}, got '{self.lane}'"
            )
        if self.transport not in VALID_TRANSPORTS:
            raise BenchmarkContractError(
                f"transport must be {sorted(VALID_TRANSPORTS)}, got "
                f"'{self.transport}'"
            )
        if self.topology not in VALID_TOPOLOGIES:
            raise BenchmarkContractError(
                f"topology must be {sorted(VALID_TOPOLOGIES)}, got "
                f"'{self.topology}'"
            )
        if not _is_int_not_bool(self.world_size) or self.world_size != 4:
            raise BenchmarkContractError(
                f"world_size must be 4 for TP4, got {self.world_size}"
            )
        if not _is_int_not_bool(self.warmup_iterations) or \
                self.warmup_iterations < 0:
            raise BenchmarkContractError(
                "warmup_iterations must be a non-negative int"
            )
        if not _is_int_not_bool(self.sample_iterations) or \
                self.sample_iterations <= 0:
            raise BenchmarkContractError(
                "sample_iterations must be a positive int"
            )


@dataclass(frozen=True)
class LatencyStats:
    p50_us: float
    p95_us: float
    p99_us: float
    max_us: float
    sample_count: int
    timing_boundary: str = "host_submission"
    clock_source: str = "cuda_event"

    def __post_init__(self) -> None:
        for name, val in [("p50_us", self.p50_us), ("p95_us", self.p95_us),
                          ("p99_us", self.p99_us), ("max_us", self.max_us)]:
            if not _is_finite_number(val) or val <= 0:
                raise BenchmarkContractError(
                    f"latency.{name} must be a positive finite number, "
                    f"got {val}"
                )
        if not (self.p50_us <= self.p95_us <= self.p99_us <= self.max_us):
            raise BenchmarkContractError(
                f"percentile ordering violated: p50={self.p50_us} "
                f"p95={self.p95_us} p99={self.p99_us} "
                f"max={self.max_us} (must be p50<=p95<=p99<=max)"
            )
        if not _is_int_not_bool(self.sample_count) or self.sample_count <= 0:
            raise BenchmarkContractError(
                f"latency.sample_count must be a positive int, "
                f"got {self.sample_count}"
            )
        if self.timing_boundary not in _VALID_TIMING_BOUNDARIES:
            raise BenchmarkContractError(
                f"latency.timing_boundary must be one of "
                f"{sorted(_VALID_TIMING_BOUNDARIES)}, got "
                f"'{self.timing_boundary}'"
            )
        if self.clock_source not in _VALID_CLOCK_SOURCES:
            raise BenchmarkContractError(
                f"latency.clock_source must be one of "
                f"{sorted(_VALID_CLOCK_SOURCES)}, got "
                f"'{self.clock_source}'"
            )

    @classmethod
    def from_samples(
        cls,
        samples_us: Sequence[float],
        timing_boundary: str = "host_submission",
        clock_source: str = "cuda_event",
    ) -> "LatencyStats":
        if not samples_us:
            raise BenchmarkContractError(
                "cannot compute latency stats from empty samples"
            )
        for s in samples_us:
            if not isinstance(s, (int, float)) or isinstance(s, bool):
                raise BenchmarkContractError(
                    f"latency sample must be a number, got {type(s).__name__}"
                )
            if s <= 0 or s != s:  # NaN check
                raise BenchmarkContractError(
                    f"latency sample must be positive and finite, got {s}"
                )
        ordered = sorted(samples_us)
        n = len(ordered)

        def percentile(fraction: float) -> float:
            try:
                index = round((n - 1) * fraction)
            except OverflowError as exc:
                raise BenchmarkContractError(
                    f"latency percentile computation overflow: {exc}"
                ) from exc
            return ordered[index]

        return cls(
            p50_us=percentile(0.50),
            p95_us=percentile(0.95),
            p99_us=percentile(0.99),
            max_us=ordered[-1],
            sample_count=n,
            timing_boundary=timing_boundary,
            clock_source=clock_source,
        )


# ---------------------------------------------------------------------------
# Counter scope — restricted to one aggregate before/after pair
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CounterScope:
    """Counter scope and reset/window semantics for one arm.

    Restricted to one aggregate before/after pair.  Requires
    warmup_excluded=true, reset_source in {before-run, diff},
    nonnegative delta, and delta exactly equals custom+fallback.
    """

    counter_source: str
    warmup_excluded: bool
    reset_source: str
    before_snapshot: int
    after_snapshot: int

    def __post_init__(self) -> None:
        _validate_string_field(
            self.counter_source, "counter_scope.counter_source")
        if not isinstance(self.warmup_excluded, bool):
            raise BenchmarkContractError(
                "counter_scope.warmup_excluded must be a bool"
            )
        if not self.warmup_excluded:
            raise BenchmarkContractError(
                "counter_scope.warmup_excluded must be true for this contract"
            )
        if self.reset_source not in ("before-run", "diff"):
            raise BenchmarkContractError(
                f"counter_scope.reset_source must be 'before-run' or "
                f"'diff', got '{self.reset_source}'"
            )
        if not _is_int_not_bool(self.before_snapshot) or self.before_snapshot < 0:
            raise BenchmarkContractError(
                f"counter_scope.before_snapshot must be a non-negative int, "
                f"got {self.before_snapshot}"
            )
        if not _is_int_not_bool(self.after_snapshot) or self.after_snapshot < 0:
            raise BenchmarkContractError(
                f"counter_scope.after_snapshot must be a non-negative int, "
                f"got {self.after_snapshot}"
            )

    @property
    def delta(self) -> int:
        return self.after_snapshot - self.before_snapshot


# ---------------------------------------------------------------------------
# Raw-evidence artifact declaration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RawEvidenceArtifact:
    """Declaration of a raw-evidence artifact file on disk.

    At comparison time, the artifact file is loaded from beneath
    the evidence root, its SHA-256 is recomputed and compared to
    ``artifact_sha256`` (carried in this record, NOT inside the
    artifact file).  The schema is validated, the arm role / transport
    / selector / workload / topology / binary / runtime bindings are
    recomputed from the BenchmarkRecord.  No free-form verdict is
    accepted — correctness is *derived* from the raw per-rank/
    per-iteration observations inside the artifact.

    Required fields:
    - ``schema_name``: must equal RAW_EVIDENCE_SCHEMA.
    - ``artifact_path``: relative path beneath the evidence root.
    - ``artifact_sha256``: expected SHA-256 of the artifact file.
    - ``arm_transport``: the transport role this artifact belongs to
      (must match the record's transport).
    """

    schema_name: str
    artifact_path: str  # relative to evidence root
    artifact_sha256: str
    arm_transport: str  # "sircl" or "nccl_socket" / "nccl_ib"

    def __post_init__(self) -> None:
        if self.schema_name != RAW_EVIDENCE_SCHEMA:
            raise BenchmarkContractError(
                f"correctness.schema_name must be '{RAW_EVIDENCE_SCHEMA}', "
                f"got '{self.schema_name}'"
            )
        _validate_string_field(
            self.artifact_path, "correctness.artifact_path")
        if self.arm_transport not in VALID_TRANSPORTS:
            raise BenchmarkContractError(
                f"correctness.arm_transport must be one of "
                f"{sorted(VALID_TRANSPORTS)}, got '{self.arm_transport}'"
            )
        _validate_sha256(
            self.artifact_sha256, "correctness.artifact_sha256")


# ---------------------------------------------------------------------------
# Canonical binding computation
# ---------------------------------------------------------------------------

def _compute_workload_binding(workload: WorkloadSpec) -> str:
    """Compute canonical SHA-256 binding from workload spec."""
    canonical = json.dumps({
        "shape": list(workload.shape),
        "dtype": workload.dtype,
        "bytes_per_collective": workload.bytes_per_collective,
        "collective_count": workload.collective_count,
    }, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _compute_identity_binding(identity: IdentitySpec) -> str:
    """Compute canonical SHA-256 binding from identity spec.

    Uses all identity fields except transport_selector_hash (which differs
    by arm) — the binding proves both arms share the same binary,
    image, commit, and topology.
    """
    canonical = json.dumps({
        "schema_version": identity.schema_version,
        "source_commit": identity.source_commit,
        "runtime_commit": identity.runtime_commit,
        "registry_digest": identity.registry_digest,
        "local_image_id": identity.local_image_id,
        "transport_library_hash": identity.transport_library_hash,
        "torch_version": identity.torch_version,
        "vllm_version": identity.vllm_version,
        "cuda_version": identity.cuda_version,
        "driver_version": identity.driver_version,
        "model_repository": identity.model_repository,
        "model_revision": identity.model_revision,
        "model_config_sha256": identity.model_config_sha256,
        "tp4_edges": [
            {"rank": e.rank, "round": e.round, "peer_rank": e.peer_rank,
             "device": e.device, "gid": e.gid,
             "peer_interface": e.peer_interface}
            for e in identity.tp4_edges
        ],
        "clocks_power": {
            "policy": identity.clocks_power.policy,
            "gpu_clock_mhz": identity.clocks_power.gpu_clock_mhz,
            "power_limit_w": identity.clocks_power.power_limit_w,
            "declaration_sha256": identity.clocks_power.declaration_sha256,
            "description": identity.clocks_power.description,
        },
        "evidence_run_id": identity.evidence_run_id,
    }, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _compute_artifact_binding(record: BenchmarkRecord) -> str:
    """Compute canonical SHA-256 binding from the full record.

    Binds artifact to arm role, transport, canonical selector config,
    workload, topology, binary and runtime identities.  One artifact
    cannot be reused for both arms because the transport and selector
    differ.
    """
    canonical = json.dumps({
        "arm_transport": record.runtime.transport,
        "selector_transport_role": record.identity.selector.transport_role,
        "selector_env_vars": sorted(record.identity.selector.env_vars),
        "selector_hash": record.identity.selector.selector_hash,
        "transport_library_hash": record.identity.transport_library_hash,
        "workload_binding": _compute_workload_binding(record.workload),
        "topology": record.runtime.topology,
        "world_size": record.runtime.world_size,
        "source_commit": record.identity.source_commit,
        "runtime_commit": record.identity.runtime_commit,
        "evidence_run_id": record.identity.evidence_run_id,
    }, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Evidence root path resolution and safe file loading
# ---------------------------------------------------------------------------

def _resolve_evidence_path(
    evidence_root: str, relative_path: str,
) -> str:
    """Resolve ``relative_path`` beneath ``evidence_root``.

    Rejects absolute paths, ``..`` traversal, symlinks/reparse
    points, non-regular files, and files above the size cap.
    """
    if not isinstance(relative_path, str) or not relative_path:
        raise BenchmarkContractError(
            "artifact_path must be a nonempty relative path"
        )
    if os.path.isabs(relative_path):
        raise BenchmarkContractError(
            f"artifact_path must be relative to evidence root, "
            f"got absolute path '{relative_path}'"
        )
    # Normalize and check for traversal — reject any '..' component
    normalized = os.path.normpath(relative_path)
    parts = normalized.split(os.sep)
    if ".." in parts or normalized.startswith(".."):
        raise BenchmarkContractError(
            f"artifact_path escapes evidence root: '{relative_path}'"
        )
    # On Windows, also reject drive letters and UNC paths
    if re.match(r"^[A-Za-z]:", normalized):
        drive = os.path.splitdrive(normalized)[0]
        raise BenchmarkContractError(
            f"artifact_path must not contain drive letter: '{drive}'"
        )
    if normalized.startswith("\\\\"):
        raise BenchmarkContractError(
            f"artifact_path must not be a UNC path: '{normalized}'"
        )

    full_path = os.path.join(evidence_root, normalized)
    full_path = os.path.normpath(full_path)

    # Verify the resolved path is still beneath the evidence root
    root_abs = os.path.abspath(evidence_root)
    full_abs = os.path.abspath(full_path)
    if not full_abs.startswith(root_abs + os.sep) and full_abs != root_abs:
        raise BenchmarkContractError(
            f"artifact_path resolves outside evidence root: "
            f"'{relative_path}' -> '{full_abs}'"
        )

    # Check the evidence root itself for symlinks/junctions
    _check_path_components_no_symlink(root_abs)

    # Check every parent component for symlinks/reparse points
    _check_path_components_no_symlink(full_abs)

    return full_path


def _is_symlink_or_junction(path: str) -> bool:
    """True if ``path`` is a symlink, junction, or any reparse point.

    On Windows, ``os.path.islink`` returns ``False`` for junctions.
    ``os.path.isjunction`` (Python 3.12+) catches them; on older
    Python we fall back to the reparse-point file-attribute bit in
    ``os.lstat``.
    """
    if os.path.islink(path):
        return True
    # Windows junctions: os.path.isjunction is available on 3.12+
    isjunction = getattr(os.path, "isjunction", None)
    if isjunction is not None and isjunction(path):
        return True
    # Fallback: check reparse-point attribute via lstat for older Python
    try:
        st = os.lstat(path)
    except OSError:
        return False
    if hasattr(st, "st_file_attributes"):
        # FILE_ATTRIBUTE_REPARSE_POINT = 0x400
        if st.st_file_attributes & 0x400:
            return True
    if hasattr(st, "st_reparse_tag"):
        if st.st_reparse_tag != 0:
            return True
    return False


def _check_path_components_no_symlink(path: str) -> None:
    """Check every component of ``path`` (and all parents, and the
    evidence root itself) for symlinks, junctions, and reparse
    points.

    On Windows, ``os.path.islink`` returns ``False`` for junctions
    (``islink=False``, ``isjunction=True``).  A parent directory
    junction can redirect reads outside the evidence root.  This
    function rejects every reparse-point component, including the
    root and all parents, using Windows-aware inspection.
    """
    current = path
    visited: set[str] = set()
    while current and current not in visited:
        visited.add(current)
        try:
            if _is_symlink_or_junction(current):
                raise BenchmarkContractError(
                    f"path component is a symlink/junction/reparse "
                    f"point: '{current}'"
                )
        except OSError as exc:
            raise BenchmarkContractError(
                f"cannot stat path component '{current}': {exc}"
            ) from exc
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent


def _get_fd_final_path(fd: int) -> str | None:
    """Get the real filesystem path of an open file descriptor.

    On Windows, uses ``GetFinalPathNameByHandleW`` via ctypes.
    On Linux, reads ``/proc/self/fd/{fd}``.
    On macOS, uses ``fcntl.F_GETPATH``.
    Returns ``None`` if the platform cannot resolve the fd path.
    """
    if os.name == "nt":
        try:
            import ctypes
            import msvcrt
            from ctypes import wintypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            # Convert CRT fd → Windows HANDLE
            handle = msvcrt.get_osfhandle(fd)
            # GetFinalPathNameByHandleW(handle, buf, buf_len, flags)
            # flags=0 → VOLUME_NAME_DOS, FILE_NAME_NORMAL
            buf = ctypes.create_unicode_buffer(1024)
            # DWORD GetFinalPathNameByHandleW(HANDLE, LPWSTR, DWORD, DWORD)
            kernel32.GetFinalPathNameByHandleW.argtypes = [
                wintypes.HANDLE, wintypes.LPWSTR,
                wintypes.DWORD, wintypes.DWORD,
            ]
            kernel32.GetFinalPathNameByHandleW.restype = wintypes.DWORD
            ret = kernel32.GetFinalPathNameByHandleW(
                handle, buf, 1024, 0,
            )
            if ret == 0 or ret >= 1024:
                return None
            # Result is prefixed with \\?\ — normalize it
            path = buf.value
            if path.startswith("\\\\?\\"):
                path = path[4:]
            return os.path.normpath(path)
        except Exception:
            return None
    else:
        # POSIX: try /proc/self/fd (Linux) then fcntl F_GETPATH (macOS)
        try:
            link = f"/proc/self/fd/{fd}"
            if os.path.exists(link):
                return os.path.normpath(os.readlink(link))
        except OSError:
            pass
        try:
            import fcntl
            if hasattr(fcntl, "F_GETPATH"):
                return os.path.normpath(
                    fcntl.fcntl(fd, fcntl.F_GETPATH))
        except (OSError, AttributeError):
            pass
        return None


def _safe_open_and_read(
    full_path: str,
    evidence_root: str | None = None,
) -> tuple[bytes, str]:
    """Open the file once using a no-follow file descriptor, read all
    bytes (bounded to cap+1), verify it's a regular file, and
    return (bytes, sha256).

    Rejects symlinks/junctions/reparse points, non-regular files,
    and oversized files.

    **TOCTOU hardening (defect 2/3):** A parent directory can be
    swapped (e.g. replaced by a junction) after the component
    validation in ``_resolve_evidence_path`` but before the open.
    Leaf-level ``O_NOFOLLOW`` does not protect the parent chain.

    The defense is handle-confined: after opening the fd, we resolve
    the fd's real path via ``GetFinalPathNameByHandleW`` (Windows) or
    ``/proc/self/fd`` (Linux) and verify it is still confined within
    ``evidence_root``.  If the fd path cannot be resolved on this
    platform AND ``evidence_root`` is provided, we fail closed —
    the read is refused because confinement cannot be established.

    The read is bounded to ``_MAX_ARTIFACT_BYTES + 1`` so a file
    that grows past the cap between the stat and read is detected.
    """
    # Re-check all path components (including parents) for
    # symlinks/junctions immediately before opening.
    _check_path_components_no_symlink(full_path)

    # Open with O_NOFOLLOW on POSIX to prevent following symlinks
    # at the final component.  On Windows (no O_NOFOLLOW), we
    # rely on the comprehensive _check_path_components_no_symlink
    # call above plus the post-open fd confinement check below.
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    try:
        fd = os.open(full_path, flags)
    except OSError as exc:
        raise BenchmarkContractError(
            f"cannot open artifact file (no-follow): '{full_path}': {exc}"
        ) from exc

    try:
        # Post-open fd confinement check: resolve the fd's real path
        # and verify it is within evidence_root.  This catches a
        # parent-swap TOCTOU that happens between the pre-open
        # re-check and the open.  If evidence_root is provided and
        # the fd path cannot be resolved, fail closed.
        if evidence_root is not None:
            fd_path = _get_fd_final_path(fd)
            if fd_path is None:
                raise BenchmarkContractError(
                    f"cannot establish fd path confinement on this "
                    f"platform — failing closed for '{full_path}'"
                )
            root_abs = os.path.abspath(evidence_root)
            fd_abs = os.path.abspath(fd_path)
            if not (fd_abs == root_abs or
                    fd_abs.startswith(root_abs + os.sep)):
                raise BenchmarkContractError(
                    f"fd path confinement violated: fd resolves to "
                    f"'{fd_abs}' which is outside evidence root "
                    f"'{root_abs}' — parent-swap TOCTOU detected"
                )

        # fstat the opened fd to get file type and size
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise BenchmarkContractError(
                f"artifact path is not a regular file: '{full_path}'"
            )
        if st.st_size > _MAX_ARTIFACT_BYTES:
            raise BenchmarkContractError(
                f"artifact file exceeds size cap "
                f"({_MAX_ARTIFACT_BYTES} bytes): '{full_path}' is "
                f"{st.st_size} bytes"
            )
        if st.st_size == 0:
            raise BenchmarkContractError(
                f"artifact file is empty: '{full_path}'"
            )

        # Read in a bounded loop: reads up to cap+1 total bytes, so
        # a file that grows past the cap between fstat and read is
        # detected.  The loop cannot grow beyond the cap because
        # each chunk is counted and the total is checked every
        # iteration.
        max_read = _MAX_ARTIFACT_BYTES + 1
        chunks: list[bytes] = []
        total = 0
        while total < max_read:
            chunk = os.read(fd, min(max_read - total, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        artifact_bytes = b"".join(chunks)
        if len(artifact_bytes) > _MAX_ARTIFACT_BYTES:
            raise BenchmarkContractError(
                f"artifact file grew past size cap during read: "
                f"'{full_path}' exceeded {_MAX_ARTIFACT_BYTES} bytes"
            )
    finally:
        os.close(fd)

    recomputed_sha = hashlib.sha256(artifact_bytes).hexdigest()
    return artifact_bytes, recomputed_sha


# ---------------------------------------------------------------------------
# Verifier hash verification
# ---------------------------------------------------------------------------

def _verify_verifier_receipt(repo_root: str) -> str:
    """Verify the verifier implementation file on disk matches the
    trusted hash.

    The verifier is ``tp4_numerical_audit.py`` in the source tree.
    This is an implementation-byte check: it proves the script is
    unchanged, NOT that the source commit matches the actual checkout.
    If the file has been modified, the hash won't match and
    verification fails closed.

    IMPORTANT: This hash proves only that the script is unchanged.
    It does NOT prove that the script produced any given artifact.
    The artifact's provenance is bound to the record's identity
    (source_commit, runtime_commit, image, transport), not to the
    verifier hash alone.  A copied fake repo with the same file hash
    does not establish provenance.  The record's source_commit is
    a caller-declared field — this function does NOT verify it
    against the actual git checkout.  Provenance is labeled
    declared, not verified, unless an independent attestation binds
    the commit to the checkout.
    Returns the recomputed hash on success.
    """
    verifier_path = os.path.join(repo_root, _VERIFIER_REL_PATH)
    if not os.path.isfile(verifier_path):
        raise BenchmarkContractError(
            f"verifier implementation not found: {verifier_path}"
        )
    with open(verifier_path, "rb") as f:
        verifier_hash = hashlib.sha256(f.read()).hexdigest()
    if verifier_hash != TRUSTED_VERIFIER_HASH:
        raise BenchmarkContractError(
            f"verifier hash mismatch: tp4_numerical_audit.py has "
            f"SHA-256 '{verifier_hash[:16]}...' but trusted hash "
            f"is '{TRUSTED_VERIFIER_HASH[:16]}...' — verifier "
            f"implementation has been modified"
        )
    return verifier_hash


# ---------------------------------------------------------------------------
# Raw-evidence artifact loading and derivation
# ---------------------------------------------------------------------------

# Required keys in the v2 raw-evidence artifact file.
# The v2 schema stores per-rank/per-iteration raw observations
# (hashes, output samples, metrics) that the validator recomputes
# from deterministic inputs.  It does NOT carry caller-authored
# aggregate metrics or counter receipts — those are derived.
_RAW_EVIDENCE_REQUIRED_KEYS = {
    "schema", "evidence_type", "iterations", "elements", "ranks",
    "workload_pattern", "workload_rows", "tolerance_policy",
    "per_rank_raw", "artifact_sha256",
}

def _load_and_derive_correctness(
    record: BenchmarkRecord,
    evidence_root: str,
    repo_root: str,
) -> dict[str, Any]:
    """Load the v2 raw-evidence artifact, recompute all hashes and
    metrics from deterministic inputs, enforce the versioned tolerance
    policy, and derive structural consistency.

    The validator in ``spark_raw_evidence.validate_raw_evidence``
    regenerates all inputs, recomputes FP32 truth and the TP4 ring
    reduction, verifies all stored hashes, and recomputes all metrics
    (MAE, RMSE, max error, mismatch count, tolerance count).  It
    does NOT trust caller-supplied aggregates.

    Never accepts a free-form verdict.  Structural consistency is a
    structural label, NOT a correctness verdict.  Raises
    BenchmarkContractError on any failure.
    """
    ca = record.correctness

    # Verify arm_transport matches the record's transport
    if ca.arm_transport != record.runtime.transport:
        raise BenchmarkContractError(
            f"correctness arm_transport mismatch: artifact declares "
            f"'{ca.arm_transport}' but record transport is "
            f"'{record.runtime.transport}'"
        )

    # Resolve path beneath evidence root (rejects abs, .., symlinks,
    # parent junction escapes)
    full_path = _resolve_evidence_path(evidence_root, ca.artifact_path)

    # Safe open and read (rejects symlinks, non-regular, oversized,
    # TOCTOU-safe via O_NOFOLLOW and bounded read)
    artifact_bytes, recomputed_sha = _safe_open_and_read(
        full_path, evidence_root)

    # Verify SHA-256 matches the record's declared hash
    if ca.artifact_sha256 != recomputed_sha:
        raise BenchmarkContractError(
            f"correctness artifact SHA-256 mismatch: record declares "
            f"'{ca.artifact_sha256[:16]}...' but file recomputes "
            f"'{recomputed_sha[:16]}...' (tampered or stale)"
        )

    # Parse JSON
    try:
        artifact_data = json.loads(artifact_bytes)
    except json.JSONDecodeError as exc:
        raise BenchmarkContractError(
            f"correctness artifact is not valid JSON: {exc}"
        ) from exc

    if not isinstance(artifact_data, dict):
        raise BenchmarkContractError(
            f"correctness artifact must be a JSON object, got "
            f"{type(artifact_data).__name__}"
        )

    # Exact-key validation
    _require_keys(artifact_data, _RAW_EVIDENCE_REQUIRED_KEYS,
                   "correctness artifact")

    # Validate schema
    if artifact_data["schema"] != RAW_EVIDENCE_SCHEMA:
        raise BenchmarkContractError(
            f"correctness artifact schema must be "
            f"'{RAW_EVIDENCE_SCHEMA}', got '{artifact_data['schema']}'"
        )

    # Verify tolerance policy matches the current pinned policy
    if artifact_data["tolerance_policy"] != TOLERANCE_POLICY_VERSION:
        raise BenchmarkContractError(
            f"tolerance_policy mismatch: artifact declares "
            f"'{artifact_data['tolerance_policy']}' but "
            f"current policy is '{TOLERANCE_POLICY_VERSION}'"
        )

    # Reject observed evidence — no real offline runtime output seam
    # exists.  A modeled artifact must remain explicitly modeled and
    # cannot satisfy live numerical proof.  The validator also rejects
    # this, but we check early for a clearer error message.
    if artifact_data.get("evidence_type") == "observed":
        raise BenchmarkContractError(
            "correctness artifact evidence_type='observed' is disabled — "
            "no real offline runtime output seam exists; only "
            "evidence_type='modeled' is accepted"
        )
    # Delegate to the v2 validator which recomputes ALL hashes,
    # metrics, and output samples from deterministic inputs.
    # This is the core anti-forgery check.
    try:
        from spark_raw_evidence import validate_raw_evidence
        validation_result = validate_raw_evidence(artifact_data)
    except Exception as exc:
        raise BenchmarkContractError(
            f"raw evidence validation failed: {exc}"
        ) from exc

    # Extract recomputed per-rank reduced metrics.
    per_rank_reduced = validation_result["per_rank_reduced"]

    iterations = validation_result["iterations"]
    elements = validation_result["elements"]
    ranks = validation_result["ranks"]

    # Bind artifact iterations to declared sample iterations
    if iterations != record.runtime.sample_iterations:
        raise BenchmarkContractError(
            f"artifact iterations ({iterations}) must equal "
            f"sample_iterations ({record.runtime.sample_iterations})"
        )

    # Bind artifact elements to workload shape product
    expected_elements = 1
    for dim in record.workload.shape:
        expected_elements *= dim
    if elements != expected_elements:
        raise BenchmarkContractError(
            f"artifact elements ({elements}) must equal workload "
            f"shape product ({expected_elements})"
        )

    if ranks != 4:
        raise BenchmarkContractError(
            f"artifact ranks must be 4, got {ranks}"
        )

    # Numerical coverage:
    #   total_comparisons = iterations * elements * collective_count
    total_comparisons = iterations * elements * record.workload.collective_count
    if total_comparisons > _MAX_SAFE_FLOAT_INT:
        raise BenchmarkContractError(
            f"numerical coverage overflow: iterations*elements*"
            f"collective_count ({total_comparisons}) exceeds "
            f"_MAX_SAFE_FLOAT_INT ({_MAX_SAFE_FLOAT_INT})"
        )

    # Enforce dtype-specific tolerance policy on recomputed metrics.
    dtype_tols = _DTYPE_TOLERANCES.get(record.workload.dtype)
    if dtype_tols is None:
        raise BenchmarkContractError(
            f"no tolerance policy for dtype '{record.workload.dtype}'"
        )

    all_finite = True
    for rank_idx, rm in enumerate(per_rank_reduced):
        if rm["rank"] != rank_idx:
            raise BenchmarkContractError(
                f"per_rank_reduced[{rank_idx}].rank must be {rank_idx}, "
                f"got {rm['rank']}"
            )
        if rm["iterations"] != iterations:
            raise BenchmarkContractError(
                f"per_rank_reduced[{rank_idx}].iterations must be "
                f"{iterations}, got {rm['iterations']}"
            )
        # Coverage validation: counts cannot exceed total comparisons.
        for k in ("mismatch_count", "outside_tolerance_count"):
            if rm[k] > total_comparisons:
                raise BenchmarkContractError(
                    f"per_rank_reduced[{rank_idx}].{k} "
                    f"({rm[k]}) exceeds total element "
                    f"comparisons ({total_comparisons})"
                )
        # Tolerance policy limits.
        if rm["mismatch_count"] > _MAX_MISMATCH_COUNT:
            raise BenchmarkContractError(
                f"per_rank_reduced[{rank_idx}].mismatch_count "
                f"({rm['mismatch_count']}) exceeds tolerance "
                f"policy limit ({_MAX_MISMATCH_COUNT})"
            )
        if rm["mae"] > dtype_tols["max_mae"]:
            raise BenchmarkContractError(
                f"per_rank_reduced[{rank_idx}].mae ({rm['mae']}) "
                f"exceeds tolerance policy limit for "
                f"{record.workload.dtype} ({dtype_tols['max_mae']})"
            )
        if rm["rmse"] > dtype_tols["max_rmse"]:
            raise BenchmarkContractError(
                f"per_rank_reduced[{rank_idx}].rmse ({rm['rmse']}) "
                f"exceeds tolerance policy limit for "
                f"{record.workload.dtype} ({dtype_tols['max_rmse']})"
            )
        if rm["max_abs_error"] > dtype_tols["max_abs_error"]:
            raise BenchmarkContractError(
                f"per_rank_reduced[{rank_idx}].max_abs_error "
                f"({rm['max_abs_error']}) exceeds tolerance "
                f"policy limit for {record.workload.dtype} "
                f"({dtype_tols['max_abs_error']})"
            )

    # Recompute canonical bindings from the BenchmarkRecord.
    expected_workload_binding = _compute_workload_binding(record.workload)
    expected_identity_binding = _compute_identity_binding(record.identity)
    expected_artifact_binding = _compute_artifact_binding(record)

    return {
        "artifact_sha256": recomputed_sha,
        "workload_binding": expected_workload_binding,
        "identity_binding": expected_identity_binding,
        "artifact_binding": expected_artifact_binding,
        "schema": RAW_EVIDENCE_SCHEMA,
        "arm_transport": record.runtime.transport,
        "derived_consistent": True,
        "all_finite": all_finite,
        "iterations": iterations,
        "elements": elements,
        "ranks": ranks,
        "tolerance_policy_version": TOLERANCE_POLICY_VERSION,
        "per_rank_reduced": per_rank_reduced,
    }


# ---------------------------------------------------------------------------
# Benchmark record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkRecord:
    """One complete benchmark measurement for one transport on one workload."""

    identity: IdentitySpec
    workload: WorkloadSpec
    runtime: RuntimeSpec
    latency: LatencyStats
    custom_collectives: int
    fallback_collectives: int
    unsupported_bypassed_collectives: int
    unclassified_collectives: int
    counter_scope: CounterScope
    correctness: RawEvidenceArtifact
    evidence_label: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict serializes frozenset as-is; convert to sorted list
        # for JSON compatibility.
        ev = d.get("identity", {}).get("selector", {}).get("env_vars")
        if isinstance(ev, frozenset):
            d["identity"]["selector"]["env_vars"] = sorted(ev)
        return d

    def validate(self) -> None:
        errors: list[str] = []

        for fname in ("custom_collectives", "fallback_collectives",
                      "unsupported_bypassed_collectives",
                      "unclassified_collectives"):
            v = getattr(self, fname)
            if not _is_int_not_bool(v):
                errors.append(f"{fname} must be an integer")
            elif v < 0:
                errors.append(f"{fname} must be >= 0")

        if self.custom_collectives == 0 and self.fallback_collectives == 0:
            errors.append("indeterminate: zero custom and zero fallback")

        expected_total = self.runtime.sample_iterations * self.workload.collective_count
        # All classified calls must sum to expected total
        actual_total = (
            self.custom_collectives + self.fallback_collectives
            + self.unsupported_bypassed_collectives
            + self.unclassified_collectives
        )
        if actual_total != expected_total:
            errors.append(
                f"count mismatch: custom+fallback+unsupported+unclassified "
                f"({actual_total}) != samples*collectives ({expected_total})"
            )

        if self.runtime.transport == "sircl":
            if self.fallback_collectives != 0:
                errors.append(
                    f"SIRCL arm must have zero fallback_collectives, "
                    f"got {self.fallback_collectives}"
                )
            if self.custom_collectives == 0:
                errors.append("SIRCL arm must have > 0 custom_collectives")
        elif self.runtime.transport in NCCL_TRANSPORTS:
            if self.custom_collectives != 0:
                errors.append(
                    f"NCCL arm must have zero custom_collectives, "
                    f"got {self.custom_collectives}"
                )
            if self.fallback_collectives == 0:
                errors.append("NCCL arm must have > 0 fallback_collectives")

        # Counter delta must equal custom+fallback
        delta = self.counter_scope.delta
        if delta < 0:
            errors.append(f"counter delta ({delta}) is negative")
        if delta != actual_total:
            errors.append(
                f"counter delta ({delta}) != "
                f"custom+fallback+unsupported+unclassified "
                f"({actual_total})"
            )

        if self.latency.sample_count != self.runtime.sample_iterations:
            errors.append(
                f"latency sample_count ({self.latency.sample_count}) must equal "
                f"sample_iterations ({self.runtime.sample_iterations})"
            )

        if not isinstance(self.evidence_label, str) or \
                not _LABEL_RE.match(self.evidence_label):
            errors.append(
                f"evidence_label must be a nonempty stable label, "
                f"got '{self.evidence_label}'"
            )

        # Verify selector transport_role matches the record's transport
        if self.identity.selector.transport_role != self.runtime.transport:
            errors.append(
                f"selector transport_role '{self.identity.selector.transport_role}' "
                f"does not match runtime transport '{self.runtime.transport}'"
            )

        if errors:
            raise BenchmarkContractError(
                "benchmark record validation failed: " + "; ".join(errors)
            )


# ---------------------------------------------------------------------------
# Transport semantics enforcement
# ---------------------------------------------------------------------------

def _check_transport_semantics(record: BenchmarkRecord) -> list[str]:
    """Check that transport-specific count expectations are met.

    Fallback accounting is an acceptance condition:
    - SIRCL arm: must have > 0 custom, zero fallback, zero
      unsupported_bypassed, zero unclassified.  Any fallback means
      the SIRCL arm is invalid, not merely slower.
    - NCCL arm: must have zero custom (proving SIRCL was not invoked),
      > 0 fallback, zero unsupported_bypassed, zero unclassified.
    """
    errors: list[str] = []
    if record.runtime.transport == "sircl":
        if record.custom_collectives == 0:
            errors.append("SIRCL arm must have > 0 custom collectives")
        if record.fallback_collectives != 0:
            errors.append(
                f"SIRCL arm with unexpected fallback is invalid: "
                f"got {record.fallback_collectives} fallback collectives"
            )
        if record.unsupported_bypassed_collectives != 0:
            errors.append(
                f"SIRCL arm must have zero unsupported_bypassed, "
                f"got {record.unsupported_bypassed_collectives}"
            )
        if record.unclassified_collectives != 0:
            errors.append(
                f"SIRCL arm must have zero unclassified, "
                f"got {record.unclassified_collectives}"
            )
    elif record.runtime.transport in NCCL_TRANSPORTS:
        if record.custom_collectives != 0:
            errors.append(
                f"NCCL arm must prove SIRCL was not invoked: "
                f"got {record.custom_collectives} custom collectives"
            )
        if record.fallback_collectives == 0:
            errors.append("NCCL arm must have > 0 fallback collectives")
        if record.unsupported_bypassed_collectives != 0:
            errors.append(
                f"NCCL arm must have zero unsupported_bypassed, "
                f"got {record.unsupported_bypassed_collectives}"
            )
        if record.unclassified_collectives != 0:
            errors.append(
                f"NCCL arm must have zero unclassified, "
                f"got {record.unclassified_collectives}"
            )
    return errors


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """Result of comparing two benchmark records.

    Verdicts:
    - ``valid``: both records pass structural validation
    - ``comparable``: both records are valid AND identity/workload match
      AND arm roles are enforced
    - ``performance_verdict``: PASS/FAIL if threshold declared and
      raw timing samples with named boundary/clock source are provided;
      ``not_judged`` otherwise
    - ``correctness_verdict``: ``structurally_consistent`` if both
      arms' raw evidence artifacts pass all structural derivations
      (tolerance policy, finiteness, bindings, shape);
      ``not_judged`` if artifacts are absent or not comparable;
      ``failed`` if derivation raises a contract error.  This is a
      structural label, NOT a correctness verdict.
      Counter receipts are structurally validated (schema/keys/types)
      but NOT used for judgment — they are caller-authored aggregates,
      not authoritative receipts.
    - ``correct``: always ``False``.  Structural consistency is
      not correctness.  No compatible producer exists that emits
      independently recomputable per-rank/per-iteration raw
      observations.  Until one does, ``correct`` is ``False``.
      Repository/commit fields, clocks/power, topology/device/GID/
      interface fields, and counter totals are caller-declared
      unverified declarations — they are structurally validated
      but not verified against authoritative receipts or the actual
      checkout, and are excluded from correctness/performance
      judgment.
    """

    sircl: BenchmarkRecord
    reference: BenchmarkRecord
    p50_ratio: float
    p95_ratio: float
    p99_ratio: float
    valid: bool
    comparable: bool
    performance_verdict: str
    performance_threshold: float | None
    correctness_verdict: str
    correct: bool
    failure_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sircl": self.sircl.to_dict(),
            "reference": self.reference.to_dict(),
            "p50_ratio": self.p50_ratio,
            "p95_ratio": self.p95_ratio,
            "p99_ratio": self.p99_ratio,
            "valid": self.valid,
            "comparable": self.comparable,
            "performance_verdict": self.performance_verdict,
            "performance_threshold": self.performance_threshold,
            "correctness_verdict": self.correctness_verdict,
            "correct": self.correct,
            "failure_reasons": self.failure_reasons,
        }


def _identity_mismatches(
    sircl: BenchmarkRecord, reference: BenchmarkRecord,
) -> list[str]:
    """Check that all identity fields match except transport_selector_hash.

    transport_selector_hash must DIFFER.  All other identity fields
    must match exactly.
    """
    mismatches: list[str] = []
    for field_name in (
        "schema_version", "source_commit", "runtime_commit",
        "registry_digest", "local_image_id", "transport_library_hash",
        "torch_version", "vllm_version", "cuda_version", "driver_version",
        "model_repository", "model_revision", "model_config_sha256",
        "tp4_edges", "clocks_power", "evidence_run_id",
    ):
        s_val = getattr(sircl.identity, field_name)
        r_val = getattr(reference.identity, field_name)
        if s_val != r_val:
            mismatches.append(
                f"identity.{field_name}: SIRCL={s_val} vs NCCL={r_val}"
            )
    # Selector hashes must differ
    if sircl.identity.selector.selector_hash == reference.identity.selector.selector_hash:
        mismatches.append(
            "identity.transport_selector_hash must differ between arms "
            f"(both are {sircl.identity.selector.selector_hash[:16]}...)"
        )
    return mismatches


def _workload_mismatches(
    sircl: BenchmarkRecord, reference: BenchmarkRecord,
) -> list[str]:
    """Check that workload and runtime fields match."""
    mismatches: list[str] = []
    if sircl.workload != reference.workload:
        mismatches.append(
            f"workload mismatch: {sircl.workload} vs {reference.workload}"
        )
    if sircl.runtime.world_size != reference.runtime.world_size:
        mismatches.append("world_size mismatch")
    if sircl.runtime.topology != reference.runtime.topology:
        mismatches.append("topology mismatch")
    if sircl.runtime.warmup_iterations != reference.runtime.warmup_iterations:
        mismatches.append("warmup_iterations mismatch")
    if sircl.runtime.sample_iterations != reference.runtime.sample_iterations:
        mismatches.append("sample_iterations mismatch")
    if sircl.runtime.lane != reference.runtime.lane:
        mismatches.append(
            f"cross-lane prohibited: {sircl.runtime.lane} vs "
            f"{reference.runtime.lane}"
        )
    return mismatches


def _check_arm_roles(
    sircl: BenchmarkRecord, reference: BenchmarkRecord,
) -> list[str]:
    """Enforce arm roles — arg1 must be SIRCL, arg2 must be NCCL."""
    errors: list[str] = []
    if sircl.runtime.transport != "sircl":
        errors.append(
            f"first argument must be SIRCL transport, got "
            f"'{sircl.runtime.transport}'"
        )
    if reference.runtime.transport not in NCCL_TRANSPORTS:
        errors.append(
            f"second argument must be NCCL transport, got "
            f"'{reference.runtime.transport}'"
        )
    return errors


def compare(
    sircl: BenchmarkRecord,
    reference: BenchmarkRecord,
    performance_threshold: float | None = None,
    evidence_root: str | None = None,
    repo_root: str | None = None,
) -> ComparisonResult:
    """Compare a SIRCL record against an NCCL reference record.

    Arm roles are enforced — arg1 must be SIRCL, arg2 must be NCCL.
    Selector hashes must differ; all other identity fields must match.

    ``evidence_root`` is the directory beneath which raw-evidence
    artifact files are resolved.  If not provided, correctness
    cannot be derived and ``correctness_verdict`` is ``not_judged``.

    ``repo_root`` is the repository root for verifier receipt
    verification.  If not provided, defaults to the current
    working directory.

    **Key design:** ``correct`` is always ``False``.  Structural
    consistency (``correctness_verdict = structurally_consistent``)
    means the artifact passed all structural checks (tolerance policy,
    finiteness, bindings, shape, per-rank counters).  It does NOT
    mean the arm is numerically correct — no compatible producer
    exists that emits independently recomputable evidence.

    Performance is ``not_judged`` unless raw timing samples with a
    named timing boundary and clock source are provided.  The
    LatencyStats fields ``timing_boundary`` and ``clock_source``
    declare these, but until raw samples are independently bound to
    each arm's execution, performance remains ``not_judged``.
    """
    if performance_threshold is not None:
        if not _is_finite_number(performance_threshold) or \
                performance_threshold <= 0:
            raise BenchmarkContractError(
                f"performance_threshold must be a positive finite number, "
                f"got {performance_threshold}"
            )

    # Structural validation
    valid = True
    failures: list[str] = []
    try:
        sircl.validate()
    except BenchmarkContractError as e:
        valid = False
        failures.append(f"SIRCL validation: {e}")
    try:
        reference.validate()
    except BenchmarkContractError as e:
        valid = False
        failures.append(f"NCCL validation: {e}")

    if valid:
        failures.extend(_check_arm_roles(sircl, reference))
        failures.extend(_check_transport_semantics(sircl))
        failures.extend(_check_transport_semantics(reference))
        failures.extend(_identity_mismatches(sircl, reference))
        failures.extend(_workload_mismatches(sircl, reference))

    comparable = valid and len(failures) == 0

    try:
        p50_ratio = sircl.latency.p50_us / reference.latency.p50_us
        p95_ratio = sircl.latency.p95_us / reference.latency.p95_us
        p99_ratio = sircl.latency.p99_us / reference.latency.p99_us
    except (OverflowError, ZeroDivisionError) as exc:
        raise BenchmarkContractError(
            f"latency ratio computation overflow: {exc}"
        ) from exc

    # Performance verdict: NOT-JUDGED until raw timing samples with
    # named timing boundary and clock source are independently bound
    # to each arm's execution.  A threshold alone is not sufficient.
    verdict = "NOT-JUDGED"
    if performance_threshold is not None and not comparable:
        verdict = "FAIL"
    elif performance_threshold is not None and comparable:
        # Threshold declared and comparable, but performance requires
        # raw timing samples independently bound to each arm.  The
        # LatencyStats carries timing_boundary and clock_source, but
        # without raw samples in the evidence artifact, performance
        # is not_judged, not PASS/FAIL.
        verdict = "NOT-JUDGED"
        failures.append(
            "performance not judged: raw timing samples not "
            "independently bound to each arm's execution"
        )

    # Correctness derivation from raw evidence
    # correct is ALWAYS False.  Structural consistency is not
    # correctness — no compatible producer exists.
    correct = False
    correctness_verdict = "not_judged"
    if comparable and evidence_root is not None:
        try:
            _verify_verifier_receipt(
                repo_root or os.getcwd())
            _load_and_derive_correctness(
                sircl, evidence_root, repo_root or os.getcwd())
            _load_and_derive_correctness(
                reference, evidence_root, repo_root or os.getcwd())
            # NOTE: Independent-arm provenance remains NOT-JUDGED until
            # a compatible raw producer binds each execution.  The
            # previous rule rejecting identical per_rank_metrics across
            # arms has been removed: equal correct outputs are
            # legitimate (both transports can produce the same result),
            # and changing one self-declared MAE trivially bypasses
            # the unequal-metrics rule anyway.  Repository/commit
            # fields (source_commit, runtime_commit) are caller-declared
            # and not verified against the actual checkout in this
            # offline contract — they are structural identity fields,
            # not provenance proof.

            # Structural consistency only — NOT correctness
            correctness_verdict = "structurally_consistent"
            # correct remains False
        except BenchmarkContractError as e:
            failures.append(f"correctness derivation: {e}")
            correctness_verdict = "failed"
    elif comparable and evidence_root is None:
        # Artifacts not available — not_judged, never PASS
        correctness_verdict = "not_judged"
        failures.append("correctness not judged: evidence_root not provided")

    return ComparisonResult(
        sircl=sircl,
        reference=reference,
        p50_ratio=p50_ratio,
        p95_ratio=p95_ratio,
        p99_ratio=p99_ratio,
        valid=valid,
        comparable=comparable,
        performance_verdict=verdict,
        performance_threshold=performance_threshold,
        correctness_verdict=correctness_verdict,
        correct=correct,
        failure_reasons=failures,
    )


def render_comparison(comparison: ComparisonResult) -> str:
    """Render a human-readable comparison report."""
    s = comparison.sircl
    r = comparison.reference
    w = s.workload
    lines = [
        "SIRCL_VS_NCCL_BENCHMARK",
        f"valid={'yes' if comparison.valid else 'NO'}",
        f"comparable={'yes' if comparison.comparable else 'NO'}",
        f"performance_verdict={comparison.performance_verdict}",
        f"performance_threshold={comparison.performance_threshold}",
        f"correctness_verdict={comparison.correctness_verdict}",
        f"correct={'yes' if comparison.correct else 'NO'}",
        f"lane={s.runtime.lane} topology={s.runtime.topology} "
        f"world_size={s.runtime.world_size}",
        f"shape={w.shape} dtype={w.dtype} "
        f"bytes_per_collective={w.bytes_per_collective} "
        f"collective_count={w.collective_count}",
        f"warmup={s.runtime.warmup_iterations} "
        f"samples={s.runtime.sample_iterations}",
        "",
        "| transport | p50_us | p95_us | p99_us | max_us | "
        "custom | fallback | structural |",
        "|---|---|---|---|---|---|---|---|",
        f"| SIRCL | {s.latency.p50_us:.3f} | {s.latency.p95_us:.3f} | "
        f"{s.latency.p99_us:.3f} | {s.latency.max_us:.3f} | "
        f"{s.custom_collectives} | {s.fallback_collectives} | "
        f"{comparison.correctness_verdict} |",
        f"| NCCL | {r.latency.p50_us:.3f} | {r.latency.p95_us:.3f} | "
        f"{r.latency.p99_us:.3f} | {r.latency.max_us:.3f} | "
        f"{r.custom_collectives} | {r.fallback_collectives} | "
        f"{comparison.correctness_verdict} |",
        "",
        f"| ratio | {comparison.p50_ratio:.4f} | "
        f"{comparison.p95_ratio:.4f} | {comparison.p99_ratio:.4f} | "
        f"- | - | - | - |",
    ]
    if comparison.failure_reasons:
        lines.append("")
        lines.append("Failures:")
        for reason in comparison.failure_reasons:
            lines.append(f"  {reason}")
    return "\n".join(lines)


def emit_json(comparison: ComparisonResult) -> str:
    """Emit machine-readable JSON evidence."""
    return json.dumps(comparison.to_dict(), indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Fail-closed dict parsing (never raw KeyError)
# ---------------------------------------------------------------------------

_EDGE_KEYS = {
    "rank", "round", "peer_rank", "device", "gid", "peer_interface",
}


def _edge_from_dict(data: Any, index: int) -> Tp4EdgeIdentity:
    """Parse one edge from a dict, converting all errors to contract errors."""
    if not isinstance(data, dict):
        raise BenchmarkContractError(
            f"tp4_edges[{index}] must be a dict, got {type(data).__name__}"
        )
    _require_keys(data, _EDGE_KEYS, f"tp4_edges[{index}]")
    try:
        return Tp4EdgeIdentity(
            rank=data["rank"], round=data["round"],
            peer_rank=data["peer_rank"], device=data["device"],
            gid=data["gid"], peer_interface=data["peer_interface"],
        )
    except BenchmarkContractError:
        raise
    except (TypeError, KeyError) as exc:
        raise BenchmarkContractError(
            f"tp4_edges[{index}]: {exc}"
        ) from exc


def _selector_from_dict(data: Any) -> SelectorConfig:
    """Parse selector config from dict.

    Accepts ``env_vars`` as a list of ``KEY=VALUE`` strings.  Also
    accepts the legacy ``selector_config`` single-string format for
    backward compatibility — it is converted to a single-element
    frozenset.
    """
    if not isinstance(data, dict):
        raise BenchmarkContractError(
            f"selector must be a dict, got {type(data).__name__}"
        )
    # Accept either env_vars (new) or selector_config (legacy)
    has_env = "env_vars" in data
    has_legacy = "selector_config" in data
    if not has_env and not has_legacy:
        raise BenchmarkContractError(
            "selector must have 'env_vars' (list of KEY=VALUE strings) "
            "or 'selector_config' (single KEY=VALUE string)"
        )
    extra = set(data.keys()) - {"transport_role", "selector_hash",
                                "env_vars", "selector_config"}
    if extra:
        raise BenchmarkContractError(
            f"unexpected selector fields: {sorted(extra)}"
        )
    if "transport_role" not in data:
        raise BenchmarkContractError("selector missing 'transport_role'")
    if "selector_hash" not in data:
        raise BenchmarkContractError("selector missing 'selector_hash'")
    try:
        if has_env:
            ev = data["env_vars"]
            if not isinstance(ev, (list, tuple)):
                raise BenchmarkContractError(
                    f"selector.env_vars must be a list, got "
                    f"{type(ev).__name__}"
                )
            env_vars = frozenset(ev)
        else:
            # Legacy: single string → single-element frozenset
            sc = data["selector_config"]
            if not isinstance(sc, str):
                raise BenchmarkContractError(
                    f"selector.selector_config must be a string, got "
                    f"{type(sc).__name__}"
                )
            env_vars = frozenset({sc})
        return SelectorConfig(
            transport_role=data["transport_role"],
            env_vars=env_vars,
            selector_hash=data["selector_hash"],
        )
    except BenchmarkContractError:
        raise
    except (TypeError, KeyError) as exc:
        raise BenchmarkContractError(f"selector: {exc}") from exc


def _clocks_power_from_dict(data: Any) -> ClocksPowerSettings:
    """Parse clocks/power settings from dict."""
    if not isinstance(data, dict):
        raise BenchmarkContractError(
            f"clocks_power must be a dict, got {type(data).__name__}"
        )
    _require_keys(data, {"policy", "gpu_clock_mhz", "power_limit_w",
                         "declaration_sha256", "description"}, "clocks_power")
    try:
        return ClocksPowerSettings(
            policy=data["policy"],
            gpu_clock_mhz=data["gpu_clock_mhz"],
            power_limit_w=data["power_limit_w"],
            declaration_sha256=data["declaration_sha256"],
            description=data["description"],
        )
    except BenchmarkContractError:
        raise
    except (TypeError, KeyError) as exc:
        raise BenchmarkContractError(f"clocks_power: {exc}") from exc


def _identity_from_dict(data: Any) -> IdentitySpec:
    """Parse identity from dict with exact-key validation."""
    if not isinstance(data, dict):
        raise BenchmarkContractError(
            f"identity must be a dict, got {type(data).__name__}"
        )
    required = {
        "schema_version", "source_commit", "runtime_commit",
        "registry_digest", "local_image_id", "transport_library_hash",
        "selector", "torch_version", "vllm_version",
        "cuda_version", "driver_version", "model_repository",
        "model_revision", "model_config_sha256", "tp4_edges",
        "clocks_power", "evidence_run_id",
    }
    _require_keys(data, required, "identity")
    edges_raw = data["tp4_edges"]
    if not isinstance(edges_raw, (list, tuple)):
        raise BenchmarkContractError(
            "identity.tp4_edges must be a list or tuple, "
            f"got {type(edges_raw).__name__}"
        )
    edges = tuple(_edge_from_dict(e, i) for i, e in enumerate(edges_raw))
    selector = _selector_from_dict(data["selector"])
    clocks_power = _clocks_power_from_dict(data["clocks_power"])
    try:
        return IdentitySpec(
            schema_version=data["schema_version"],
            source_commit=data["source_commit"],
            runtime_commit=data["runtime_commit"],
            registry_digest=data["registry_digest"],
            local_image_id=data["local_image_id"],
            transport_library_hash=data["transport_library_hash"],
            selector=selector,
            torch_version=data["torch_version"],
            vllm_version=data["vllm_version"],
            cuda_version=data["cuda_version"],
            driver_version=data["driver_version"],
            model_repository=data["model_repository"],
            model_revision=data["model_revision"],
            model_config_sha256=data["model_config_sha256"],
            tp4_edges=edges,
            clocks_power=clocks_power,
            evidence_run_id=data["evidence_run_id"],
        )
    except BenchmarkContractError:
        raise
    except (TypeError, KeyError) as exc:
        raise BenchmarkContractError(f"identity: {exc}") from exc


def _workload_from_dict(data: Any) -> WorkloadSpec:
    """Parse workload from dict with exact-key validation."""
    if not isinstance(data, dict):
        raise BenchmarkContractError(
            f"workload must be a dict, got {type(data).__name__}"
        )
    _require_keys(data, {"shape", "dtype", "bytes_per_collective",
                         "collective_count"}, "workload")
    try:
        return WorkloadSpec(
            shape=tuple(data["shape"]),
            dtype=data["dtype"],
            bytes_per_collective=data["bytes_per_collective"],
            collective_count=data["collective_count"],
        )
    except BenchmarkContractError:
        raise
    except (TypeError, KeyError) as exc:
        raise BenchmarkContractError(f"workload: {exc}") from exc


def _runtime_from_dict(data: Any) -> RuntimeSpec:
    """Parse runtime from dict with exact-key validation."""
    if not isinstance(data, dict):
        raise BenchmarkContractError(
            f"runtime must be a dict, got {type(data).__name__}"
        )
    _require_keys(data, {"lane", "transport", "topology", "world_size",
                         "warmup_iterations", "sample_iterations"},
                   "runtime")
    try:
        return RuntimeSpec(
            lane=data["lane"],
            transport=data["transport"],
            topology=data["topology"],
            world_size=data["world_size"],
            warmup_iterations=data["warmup_iterations"],
            sample_iterations=data["sample_iterations"],
        )
    except BenchmarkContractError:
        raise
    except (TypeError, KeyError) as exc:
        raise BenchmarkContractError(f"runtime: {exc}") from exc


def _latency_from_dict(data: Any) -> LatencyStats:
    """Parse latency from dict with exact-key validation."""
    if not isinstance(data, dict):
        raise BenchmarkContractError(
            f"latency must be a dict, got {type(data).__name__}"
        )
    _require_keys(data, {"p50_us", "p95_us", "p99_us", "max_us",
                         "sample_count", "timing_boundary",
                         "clock_source"}, "latency")
    try:
        return LatencyStats(
            p50_us=data["p50_us"],
            p95_us=data["p95_us"],
            p99_us=data["p99_us"],
            max_us=data["max_us"],
            sample_count=data["sample_count"],
            timing_boundary=data["timing_boundary"],
            clock_source=data["clock_source"],
        )
    except BenchmarkContractError:
        raise
    except (TypeError, KeyError) as exc:
        raise BenchmarkContractError(f"latency: {exc}") from exc


def _counter_scope_from_dict(data: Any) -> CounterScope:
    """Parse counter_scope from dict with exact-key validation."""
    if not isinstance(data, dict):
        raise BenchmarkContractError(
            f"counter_scope must be a dict, got {type(data).__name__}"
        )
    _require_keys(data, {"counter_source", "warmup_excluded", "reset_source",
                         "before_snapshot", "after_snapshot"},
                   "counter_scope")
    try:
        return CounterScope(
            counter_source=data["counter_source"],
            warmup_excluded=data["warmup_excluded"],
            reset_source=data["reset_source"],
            before_snapshot=data["before_snapshot"],
            after_snapshot=data["after_snapshot"],
        )
    except BenchmarkContractError:
        raise
    except (TypeError, KeyError) as exc:
        raise BenchmarkContractError(f"counter_scope: {exc}") from exc


def _correctness_from_dict(data: Any) -> RawEvidenceArtifact:
    """Parse raw-evidence artifact declaration from dict."""
    if not isinstance(data, dict):
        raise BenchmarkContractError(
            f"correctness must be a dict, got {type(data).__name__}"
        )
    _require_keys(data, {"schema_name", "artifact_path", "artifact_sha256",
                         "arm_transport"}, "correctness")
    try:
        return RawEvidenceArtifact(
            schema_name=data["schema_name"],
            artifact_path=data["artifact_path"],
            artifact_sha256=data["artifact_sha256"],
            arm_transport=data["arm_transport"],
        )
    except BenchmarkContractError:
        raise
    except (TypeError, KeyError) as exc:
        raise BenchmarkContractError(f"correctness: {exc}") from exc


def _record_from_dict(data: dict[str, Any]) -> BenchmarkRecord:
    """Reconstruct a BenchmarkRecord from a JSON-like dict.

    Every nested schema is exact-key validated. Missing keys,
    extra keys, and type errors are all converted to stable
    BenchmarkContractError — never raw KeyError or TypeError.
    """
    if not isinstance(data, dict):
        raise BenchmarkContractError(
            f"record must be a dict, got {type(data).__name__}"
        )
    required_top = {"identity", "workload", "runtime", "latency",
                     "custom_collectives", "fallback_collectives",
                     "unsupported_bypassed_collectives",
                     "unclassified_collectives",
                     "counter_scope", "correctness", "evidence_label"}
    missing = required_top - set(data.keys())
    extra = set(data.keys()) - required_top - {"notes"}
    if missing:
        raise BenchmarkContractError(
            f"missing record fields: {sorted(missing)}"
        )
    if extra:
        raise BenchmarkContractError(
            f"unexpected record fields: {sorted(extra)}"
        )

    identity = _identity_from_dict(data["identity"])
    workload = _workload_from_dict(data["workload"])
    runtime = _runtime_from_dict(data["runtime"])
    latency = _latency_from_dict(data["latency"])
    counter_scope = _counter_scope_from_dict(data["counter_scope"])
    correctness = _correctness_from_dict(data["correctness"])

    return BenchmarkRecord(
        identity=identity,
        workload=workload,
        runtime=runtime,
        latency=latency,
        custom_collectives=data["custom_collectives"],
        fallback_collectives=data["fallback_collectives"],
        unsupported_bypassed_collectives=data["unsupported_bypassed_collectives"],
        unclassified_collectives=data["unclassified_collectives"],
        counter_scope=counter_scope,
        correctness=correctness,
        evidence_label=data["evidence_label"],
        notes=data.get("notes", ""),
    )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None) -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate and compare SIRCL vs NCCL benchmark records",
    )
    parser.add_argument("--sircl-json", required=True)
    parser.add_argument("--nccl-json", required=True)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Optional p50_ratio threshold for performance verdict")
    parser.add_argument("--evidence-root", default=None,
                        help="Root directory for raw-evidence artifact files")
    parser.add_argument("--repo-root", default=None,
                        help="Repository root for verifier hash check (default: cwd)")
    parser.add_argument("--emit-json", action="store_true")
    parser.add_argument("--validate-only", action="store_true",
                        help="Schema-validation mode: validate both records "
                             "structurally and exit 0 on success.  "
                             "No comparison or evidence loading.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    # Exit code contract (review-8):
    #   0 — --validate-only mode: both records passed structural
    #        schema validation.  This is the ONLY path to exit 0.
    #   1 — compare mode: not comparable, performance FAIL,
    #        correctness_verdict=failed, correctness_verdict=not_judged,
    #        or correct=false.  Any non-validate-only run that does
    #        not produce a fully judged, correct result exits nonzero.
    #   2 — malformed/contract-invalid input (BenchmarkContractError
    #        at parse or validation time, or unexpected exception).
    try:
        args = _parse_args(argv)
        with open(args.sircl_json) as f:
            sircl_data = json.load(f)
        with open(args.nccl_json) as f:
            nccl_data = json.load(f)
        sircl = _record_from_dict(sircl_data)
        nccl = _record_from_dict(nccl_data)

        if args.validate_only:
            # Schema-validation mode: validate both records structurally.
            # Exit 0 only if both pass.  No comparison or evidence.
            sircl.validate()
            nccl.validate()
            result = {
                "validate_only": True,
                "sircl_valid": True,
                "nccl_valid": True,
            }
            if args.emit_json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print("VALIDATION PASSED: both records structurally valid")
            return 0

        comparison = compare(
            sircl, nccl,
            performance_threshold=args.threshold,
            evidence_root=args.evidence_root,
            repo_root=args.repo_root,
        )
        if args.emit_json:
            print(emit_json(comparison))
        else:
            print(render_comparison(comparison))
        # Compare mode: exit 0 is NOT available.  correct is always
        # False (no compatible producer), and performance is NOT-JUDGED
        # without raw timing samples.  Any compare-mode run exits
        # nonzero because the benchmark evidence is indeterminate
        # or incorrect.
        #   1 — indeterminate or incorrect evidence:
        #        not comparable, performance FAIL, correctness_verdict
        #        = failed or not_judged, or correct=false (always).
        #   2 — malformed/contract-invalid (BenchmarkContractError).
        return 1
    except BenchmarkContractError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"ERROR: unexpected: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
