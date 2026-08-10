"""Shared canonical transport contract for SIRCL-vs-NCCL probes.

Goal 11 requirement 3: the canonical environment/argv/identity contract
must live in ONE importable module used by the probe, plan builder, JSON
parser, and validator.  No duplicate ``_build_env_projection()`` functions
that can drift.

This module is the SINGLE source of truth for:
- Arm names, transport labels, selector values, and their bindings.
- The environment projection used in ``run_contract_hash``.
- The canonical argv.
- Validator-owned pinned identity values.
- World size and required ranks.
- BF16 tolerance policy.
- Environment allowlist.
- NCCL-IB and NCCL-Socket arm environment variable sets.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# World size and ranks
# ---------------------------------------------------------------------------

AUTHORITATIVE_WORLD_SIZE = 4
AUTHORITATIVE_RANKS = frozenset({0, 1, 2, 3})

# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

SELECTOR_SIRCL = "custom"
SELECTOR_CUSTOM = SELECTOR_SIRCL  # alias used by the probe
SELECTOR_NCCL_IB = "disabled"  # NCCL-IB control arm uses selector=disabled
SELECTOR_DISABLED = SELECTOR_NCCL_IB  # alias used by the probe
SELECTOR_NCCL_SOCKET = "disabled"  # Socket diagnostic arm also uses disabled
# Both NCCL arms use selector=disabled; the transport is distinguished by
# the NCCL_NET / NCCL_IB_DISABLE env vars, not the selector.
_VALID_SELECTORS = frozenset({SELECTOR_SIRCL, SELECTOR_NCCL_IB})

# ---------------------------------------------------------------------------
# Transport labels
# ---------------------------------------------------------------------------

TRANSPORT_SIRCL = "sircl"
TRANSPORT_NCCL_IB = "nccl_ib"
TRANSPORT_NCCL_SOCKET = "nccl_socket"
TRANSPORT_NCCL_SOCKET_DIAGNOSTIC = "nccl_socket_diagnostic"

# ---------------------------------------------------------------------------
# Arm names (bound to roles, not caller labels)
# ---------------------------------------------------------------------------

ARM_NAME_SIRCL = "sircl"
ARM_NAME_NCCL_IB = "nccl_ib"
ARM_NAME_NCCL_SOCKET_DIAGNOSTIC = "nccl_socket_diagnostic"

# ---------------------------------------------------------------------------
# Authoritative arm-to-transport-to-selector binding.
# The validator owns this mapping — caller-controlled plan fields cannot
# reassign which transport or selector belongs to which arm.
# ---------------------------------------------------------------------------

ARM_BINDING: dict[str, dict[str, str]] = {
    ARM_NAME_SIRCL: {
        "transport": TRANSPORT_SIRCL,
        "selector": SELECTOR_SIRCL,
    },
    ARM_NAME_NCCL_IB: {
        "transport": TRANSPORT_NCCL_IB,
        "selector": SELECTOR_NCCL_IB,
    },
    ARM_NAME_NCCL_SOCKET_DIAGNOSTIC: {
        "transport": TRANSPORT_NCCL_SOCKET_DIAGNOSTIC,
        "selector": SELECTOR_NCCL_SOCKET,
    },
}

# ---------------------------------------------------------------------------
# Counter names — observed from per-rank receipts.
# ---------------------------------------------------------------------------

COUNTER_NATIVE = "native_collectives"
COUNTER_NCCL_IB = "nccl_ib_collectives"
COUNTER_NCCL_SOCKET = "nccl_socket_collectives"
COUNTER_UNSUPPORTED = "unsupported_bypassed_collectives"
COUNTER_UNCLASSIFIED = "unclassified_collectives"
COUNTER_FATAL_AFTER_NATIVE = "fatal_after_native_collectives"

VALID_COUNTERS = frozenset({
    COUNTER_NATIVE, COUNTER_NCCL_IB, COUNTER_NCCL_SOCKET,
    COUNTER_UNSUPPORTED, COUNTER_UNCLASSIFIED, COUNTER_FATAL_AFTER_NATIVE,
})

# Legacy counter names retained for backward compatibility with existing
# receipts/tests.  The probe emits the new names; the validator accepts
# both old and new names during the transition.
COUNTER_CUSTOM_LEGACY = "custom_collectives"
COUNTER_FALLBACK_LEGACY = "fallback_collectives"

# ---------------------------------------------------------------------------
# BF16 tolerance policy (validator-owned)
# ---------------------------------------------------------------------------

BF16_ATOL = 0.0078125  # 2^{-7}
BF16_RTOL = 0.0078125  # 2^{-7}
TOLERANCE_METRIC = "elementwise_atol_rtol"
# Deprecated alias retained for test compat.
BF16_TOLERANCE = 0.0078125

# ---------------------------------------------------------------------------
# Required output dtype and byte-order policy (validator-owned)
# ---------------------------------------------------------------------------

REQUIRED_OUTPUT_DTYPE = "bfloat16"
REQUIRED_BYTE_ORDER = "little"  # native on ARM64/x86_64

# ---------------------------------------------------------------------------
# JSON receipt schema (exact-key, closed schema)
# ---------------------------------------------------------------------------

RECEIPT_SCHEMA_VERSION = "tp4_receipt/v1"

# Exact required keys for a rank receipt JSON.  No extra keys allowed.
RECEIPT_REQUIRED_KEYS = frozenset({
    "schema_version", "rank", "transport", "selector",
    "iterations", "elements", "world_size",
    "native_collectives", "nccl_ib_collectives", "nccl_socket_collectives",
    "custom_collectives", "fallback_collectives",
    "unsupported_bypassed_collectives", "unclassified_collectives",
    "fatal_after_native_collectives", "total_collectives",
    "expected_fp32_hash", "actual_output_hash",
    "actual_dtype", "actual_byte_order",
    "all_finite", "max_abs_error", "max_rel_error",
    "tolerance_result", "tolerance_metric",
    "tolerance_atol", "tolerance_rtol",
    "sample_count", "run_contract_hash", "rank_identity",
    "counter_source_hash", "source_sha", "sircl_so_sha",
    "nccl_so_sha", "image_receipt",
})

# ---------------------------------------------------------------------------
# Canonical environment variable sets per arm
# ---------------------------------------------------------------------------

# SIRCL arm selector environment variables.
SIRCL_SELECTOR_ENVS = frozenset({"VLLM_SPARK_TP4_MODE=custom"})

# NCCL-IB control arm selector environment variables.
# This is the patched switchless NCCL recipe from SETUP.md §8.3.
NCCL_IB_SELECTOR_ENVS = frozenset({
    "VLLM_SPARK_TP4_MODE=disabled",
    "NCCL_NET=IB",
    "NCCL_IB_DISABLE=0",
})

# NCCL Socket diagnostic arm selector environment variables.
# This is the optional diagnostic-only arm, NOT a control arm.
NCCL_SOCKET_SELECTOR_ENVS = frozenset({
    "VLLM_SPARK_TP4_MODE=disabled",
    "NCCL_NET=Socket",
    "NCCL_IB_DISABLE=1",
})

# Legacy alias for backward compatibility.
NCCL_SELECTOR_ENVS = NCCL_SOCKET_SELECTOR_ENVS

# ---------------------------------------------------------------------------
# NCCL environment variable dictionaries (used in env_projection)
# ---------------------------------------------------------------------------

# NCCL-IB control arm: full patched switchless NCCL recipe from
# SETUP.md §8.3.  Every field here is a required pin for the
# NCCL-IB control arm — the validator rejects a plan or receipt
# that omits or contradicts any of these.
NCCL_IB_ENV_VARS = {
    "NCCL_NET": "IB",
    "NCCL_IB_DISABLE": "0",
    "NCCL_ALGO": "Ring",
    "NCCL_SKIP_TREE_CONNECT": "1",
    "NCCL_IB_HCA": "rocep1s0f0,rocep1s0f1",
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_IB_SUBNET_AWARE_ROUTING": "1",
    "NCCL_IB_SUBNET_PREFIX_LEN": "24",
    "NCCL_IB_MERGE_NICS": "0",
    "NCCL_CROSS_NIC": "1",
    "NCCL_MAX_NCHANNELS": "4",
    "NCCL_MIN_NCHANNELS": "4",
    "NCCL_CUMEM_ENABLE": "0",
    "NCCL_DEBUG": "INFO",
    "NCCL_DEBUG_SUBSYS": "INIT,NET",
    "NCCL_SOCKET_IFNAME": "<MGMT_IFNAME>",  # bootstrap/management only
    "VLLM_SPARK_NCCL_TRANSPORT_MODE": "switchless_ib",
    # VLLM_SPARK_SWITCHLESS_NCCL_SHA256 is per-build (hash differs);
    # it is allowlisted but not pinned to a fixed value here.
    # LD_PRELOAD / VLLM_NCCL_SO_PATH are also per-build; allowlisted
    # but not pinned to fixed values here.
}

# Required keys for NCCL-IB arm validation (the pinned values above
# that MUST be present and match exactly in every NCCL-IB plan/receipt).
NCCL_IB_REQUIRED_KEYS = frozenset({
    "NCCL_NET", "NCCL_IB_DISABLE", "NCCL_ALGO", "NCCL_SKIP_TREE_CONNECT",
    "NCCL_IB_HCA", "NCCL_IB_GID_INDEX", "NCCL_IB_SUBNET_AWARE_ROUTING",
    "NCCL_IB_SUBNET_PREFIX_LEN", "NCCL_IB_MERGE_NICS", "NCCL_CROSS_NIC",
    "NCCL_MAX_NCHANNELS", "NCCL_MIN_NCHANNELS", "NCCL_CUMEM_ENABLE",
    "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS",
    "VLLM_SPARK_NCCL_TRANSPORT_MODE",
})

# NCCL-IB observed log evidence requirements (SETUP.md §8.3 / T8).
# The validator must check observed NCCL logs for these markers.
NCCL_IB_REQUIRED_LOG_MARKERS = (
    "2.30.7",        # patched NCCL version
    "NET/IB",         # data path on IB, never Socket
    "Connected all rings",  # all rings connected
)
NCCL_IB_FORBIDDEN_LOG_MARKERS = (
    "NET/Socket",     # zero NET/Socket data-path lines
)

# NCCL Socket diagnostic arm env vars.
NCCL_SOCKET_ENV_VARS = {
    "NCCL_NET": "Socket",
    "NCCL_IB_DISABLE": "1",
    "NCCL_PROTO": "Simple",
    "NCCL_SOCKET_NTHREADS": "4",
}

# Legacy alias.
_NCCL_ENV_VARS = NCCL_SOCKET_ENV_VARS

# ---------------------------------------------------------------------------
# Runtime-lock pins for NCCL (from runtime/runtime-lock.json)
# ---------------------------------------------------------------------------

NCCL_COMMIT = "73cf112295c33aee2b895f329f592f2a9b4b0f97"
NCCL_PATCH_PATHS = (
    "spark_transport/experiments/nccl_switchless_ring/nccl-2.30.7-skip-tree-pat.patch",
    "spark_transport/experiments/nccl_switchless_ring/nccl-2.30.7-advertise-all-listener-gids.patch",
)
NCCL_PATCH_SHA256 = {
    "spark_transport/experiments/nccl_switchless_ring/nccl-2.30.7-skip-tree-pat.patch":
        "097656d07a5774919f0d51558b51ec05de8168c0097ed6cb7764c33230ba6eb2",
    "spark_transport/experiments/nccl_switchless_ring/nccl-2.30.7-advertise-all-listener-gids.patch":
        "dccfce86d14c15c39f0e0a742863960205a3d9823c464b31a7f7389354844178",
}
NCCL_VERSION = "2.30.7"

# ---------------------------------------------------------------------------
# Canonical environment allowlist
# ---------------------------------------------------------------------------

ENV_ALLOWLIST = frozenset({
    "VLLM_SPARK_TP4_MODE", "RANK", "WORLD_SIZE", "ITERATIONS",
    "ELEMENTS", "NCCL_NET", "NCCL_IB_DISABLE",
    # NCCL-IB control arm additional env vars (from SETUP.md §8.3).
    "NCCL_IB_HCA", "NCCL_IB_GID_INDEX",
    "NCCL_IB_SUBNET_AWARE_ROUTING", "NCCL_IB_SUBNET_PREFIX_LEN",
    "NCCL_IB_MERGE_NICS", "NCCL_CROSS_NIC",
    "NCCL_ALGO", "NCCL_SKIP_TREE_CONNECT",
    "NCCL_SOCKET_IFNAME", "NCCL_MAX_NCHANNELS",
    "NCCL_MIN_NCHANNELS", "NCCL_CUMEM_ENABLE",
    "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS",
    "LD_PRELOAD", "VLLM_NCCL_SO_PATH",
    "VLLM_SPARK_NCCL_TRANSPORT_MODE",
    "VLLM_SPARK_SWITCHLESS_NCCL_SHA256",
    # Socket diagnostic arm additional env vars.
    "NCCL_PROTO", "NCCL_SOCKET_NTHREADS",
    # SIRCL arm: native library path (read by _NativeSession).
    "SPARK_TP4_LIBRARY",
    # SIRCL arm: native transport .so path (alternate name).
    "VLLM_SPARK_TRANSPORT_SO_PATH",
    # SIRCL arm: peer/device/GID settings.
    "SPARK_TP4_PEER0", "SPARK_TP4_PEER1",
    "SPARK_TP4_DEVICE0", "SPARK_TP4_DEVICE1",
    "SPARK_TP4_GID0", "SPARK_TP4_GID1",
    # Image/runtime receipt.
    "VLLM_SPARK_IMAGE_RECEIPT",
})

# ---------------------------------------------------------------------------
# Canonical argv
# ---------------------------------------------------------------------------

NATIVE_PROBE_SCRIPT = "spark_transport/integrations/vllm/tp4_numerical_audit.py"
CANONICAL_ARGV = ("python", NATIVE_PROBE_SCRIPT)

# ---------------------------------------------------------------------------
# Validator-owned pinned identity values
# ---------------------------------------------------------------------------

PINNED_TOPOLOGY = "tp4_switchless_ring"
PINNED_WORKLOAD = "tp4_numerical_audit"
PINNED_ORDER = "identical"
PINNED_BINARY_IDENTITY = NATIVE_PROBE_SCRIPT
PINNED_PROBE_IDENTITY = NATIVE_PROBE_SCRIPT

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

EXIT_VALID = 0
EXIT_INVALID = 1
EXIT_MALFORMED = 2
EXIT_MISSING_SEAM = 3

# ---------------------------------------------------------------------------
# Safety class labels
# ---------------------------------------------------------------------------

SAFETY_OFFLINE = "OFFLINE"
SAFETY_READ_ONLY_REMOTE = "READ-ONLY REMOTE"
SAFETY_MUTATES_HOST = "MUTATES HOST"
SAFETY_STOPS_SERVING = "STOPS SERVING"

VALID_SAFETY_CLASSES = frozenset({
    SAFETY_OFFLINE,
    SAFETY_READ_ONLY_REMOTE,
    SAFETY_MUTATES_HOST,
    SAFETY_STOPS_SERVING,
})

# ---------------------------------------------------------------------------
# Confirmation prompt
# ---------------------------------------------------------------------------

CONFIRMATION_PROMPT = "CONFIRM EXECUTE TWO_ARM BENCHMARK"
CONFIRMATION_REQUIRED_RESPONSE = "I CONFIRM EXECUTE TWO_ARM BENCHMARK"


# ---------------------------------------------------------------------------
# Canonical environment projection — SINGLE source of truth
# ---------------------------------------------------------------------------

def build_env_projection(
    selector: str,
    rank: int,
    world_size: int,
    iterations: int,
    elements: int,
    *,
    transport: str | None = None,
) -> dict[str, str]:
    """Build the canonical environment projection for the run contract.

    This is the SINGLE source of truth used by the probe (receipt
    construction), the validator (run_contract_hash recomputation),
    the plan builder, and the JSON parser.  No duplicated lists.

    For the NCCL-IB arm (transport=nccl_ib), NCCL_NET=IB and
    NCCL_IB_DISABLE=0 are included.
    For the NCCL Socket diagnostic arm (transport=nccl_socket_diagnostic),
    NCCL_NET=Socket and NCCL_IB_DISABLE=1 are included.
    For the SIRCL arm, no NCCL env vars are included.
    """
    proj: dict[str, str] = {
        "VLLM_SPARK_TP4_MODE": selector,
        "RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "ITERATIONS": str(iterations),
        "ELEMENTS": str(elements),
    }
    if transport == TRANSPORT_NCCL_IB:
        proj.update(NCCL_IB_ENV_VARS)
    elif transport == TRANSPORT_NCCL_SOCKET_DIAGNOSTIC:
        proj.update(NCCL_SOCKET_ENV_VARS)
    elif transport == TRANSPORT_NCCL_SOCKET:
        # Legacy: nccl_socket transport → Socket env vars.
        proj.update(NCCL_SOCKET_ENV_VARS)
    elif selector == SELECTOR_NCCL_IB and transport is None:
        # Backward compat: if only selector is given and it's disabled,
        # default to Socket env vars (legacy behavior).
        proj.update(NCCL_SOCKET_ENV_VARS)
    return proj
