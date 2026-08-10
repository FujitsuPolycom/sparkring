"""Compare direct-cable TP4 and NCCL BF16 sums against FP32 ground truth.

This probe selects exactly one transport implementation per invocation:

- ``VLLM_SPARK_TP4_MODE=custom``   → native SIRCL via ``_NativeSession``
- ``VLLM_SPARK_TP4_MODE=disabled``  → fallback NCCL via ``dist.all_reduce``

The selector is read and validated **before** initializing any transport.
Unknown or missing values fail closed — the probe exits with an error
and no collective is executed.

Failure boundary (Goal 9):
Once the native transport is invoked, any exception is **process/run fatal**.
The probe does NOT fall back to NCCL after native work may have been
enqueued — doing so would risk silent transport switching and
double-enqueue.  Fallback, if needed, must be a pre-attempt capability
decision made before native submission.  The direct NCCL arm
(``selector=disabled``) remains the comparison control.

The element count is consumed from the ``ELEMENTS`` environment variable
(default 6144), not hardcoded.

Each collective is classified exactly once as:

- ``native``              — SIRCL native transport (custom mode)
- ``fallback``            — NCCL fallback (disabled mode)
- ``unsupported_bypassed``— signature ineligible, bypassed to stock NCCL
- ``unclassified``         — could not be classified (never in normal operation)

The core execution logic lives in ``run_probe()``, which accepts
injectable ``native_session`` and ``dist_backend`` parameters so tests
can exercise the same code path ``main()`` uses without CUDA/RDMA.

Per-iteration the probe computes a deterministic FP32 reference (sum all
rank inputs in FP32), inspects the actual collective output, rejects
NaN/Inf (raises, no receipt), and accumulates SHA-256 hashes of the
FP32 reference and actual output tensors.  The emitted receipt carries
numerical evidence:

- ``expected_fp32_hash``  — SHA-256 of unrounded FP32 reference tensor bytes
- ``actual_output_hash``   — SHA-256 of actual output tensor bytes in declared dtype
- ``actual_dtype``         — declared dtype of the actual output (e.g. "bfloat16")
- ``actual_byte_order``     — declared byte order of the actual output hash ("little")
- ``all_finite``           — whether all output elements are finite
- ``max_abs_error``         — max |actual − fp32_ref| over all elements (diagnostic)
- ``max_rel_error``         — max relative error (abs used when ref == 0) (diagnostic)
- ``tolerance_result``      — "pass" or "fail" under the elementwise criterion
- ``tolerance_metric``      — "elementwise_atol_rtol" (the criterion name)
- ``sample_count``          — iterations × elements
- ``run_contract_hash``     — SHA-256 of canonical JSON binding
- ``rank_identity``         — sanitized stable rank identity (rank-N-of-M)

Numerical acceptance uses an elementwise criterion:
``abs_error <= atol + rtol * abs(reference)`` with validator-owned
``BF16_ATOL`` and ``BF16_RTOL``.  The global absolute bound
(``2^{-7}``) is NOT used as a sole acceptance threshold because it
rejects correct BF16-rounded outputs whose reference values exceed 1.0.
Max absolute/relative error remain as diagnostics but do not govern
acceptance.

The ``actual_output_hash`` is a caller-supplied digest of the actual
output tensor.  It is tamper-evident (any change to the output
changes the hash) but is NOT non-forgeable — a caller can compute any
hash and present it.  The validator recomputes ``expected_fp32_hash``
from the deterministic workload contract; ``actual_output_hash`` cannot
be independently recomputed by the validator without the actual output
bytes.

Per-rank observed counters are emitted as JSON on stdout.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import torch
import torch.distributed as dist

from spark_transport_contract import (
    BF16_ATOL,
    BF16_RTOL,
    ENV_ALLOWLIST,
    NATIVE_PROBE_SCRIPT as _NATIVE_PROBE_IDENTITY,
    RECEIPT_REQUIRED_KEYS,
    RECEIPT_SCHEMA_VERSION,
    REQUIRED_BYTE_ORDER,
    REQUIRED_OUTPUT_DTYPE,
    SELECTOR_CUSTOM,
    SELECTOR_NCCL_IB as SELECTOR_DISABLED,
    TOLERANCE_METRIC,
    TRANSPORT_NCCL_IB,
    TRANSPORT_NCCL_SOCKET_DIAGNOSTIC,
    TRANSPORT_SIRCL,
    build_env_projection,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORLD_SIZE = 4
ELEMENTS = 6144  # default element count (overridable via ELEMENTS env at runtime)
# Supported transport selectors — imported from shared contract.
_VALID_SELECTORS = {SELECTOR_CUSTOM, SELECTOR_DISABLED}

# Counter classification labels.
CLASS_NATIVE = "native"
CLASS_NCCL_IB = "nccl_ib"
CLASS_NCCL_SOCKET = "nccl_socket"
CLASS_FALLBACK = "fallback"  # legacy alias
CLASS_UNSUPPORTED = "unsupported_bypassed"
CLASS_UNCLASSIFIED = "unclassified"
CLASS_FATAL_AFTER_NATIVE = "fatal_after_native"

# Retained for backward compatibility with tests that import it.
# NOT used as the acceptance criterion — use BF16_ATOL/BF16_RTOL instead.
TOLERANCE_THRESHOLD_BF16 = 0.0078125
# _NATIVE_PROBE_IDENTITY and build_env_projection are imported from
# spark_transport_contract (Goal 11 requirement 3).  The old local
# definitions are removed — no duplicate _build_env_projection functions.
# Backward-compat alias for tests that import _build_env_projection.
_build_env_projection = build_env_projection


def make_rank_input(sequence: int, rank: int, elements: int = ELEMENTS) -> torch.Tensor:
    """Generate broad-scale BF16 inputs, including cancellation-heavy cases."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0x5A17 + sequence * WORLD_SIZE + rank)
    independent = torch.randn(elements, generator=generator)

    indices = torch.arange(elements)
    exponents = ((indices + sequence) % 12) - 6
    scale = torch.pow(2.0, exponents)

    if sequence & 1:
        shared_generator = torch.Generator(device="cpu")
        shared_generator.manual_seed(0xC011A + sequence)
        shared = torch.randn(elements, generator=shared_generator) * scale
        coefficient = (1.0, -1.0, 0.5, -0.5)[rank]
        value = shared * coefficient + independent * scale * 0.001
    else:
        value = independent * scale
    return value.to(torch.bfloat16)


def _validate_selector() -> str:
    """Read and validate the transport selector before any transport init.

    Fail closed on unknown/missing values: exit with error, no collective.
    """
    selector = os.getenv("VLLM_SPARK_TP4_MODE", "").lower()
    if selector not in _VALID_SELECTORS:
        print(
            f"ERROR: VLLM_SPARK_TP4_MODE must be one of "
            f"{sorted(_VALID_SELECTORS)}, got '{selector}'",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
    return selector


def _validate_process_env(selector: str) -> None:
    """Read and validate the actual process environment before data path.

    Goal 11 requirement 3: the probe must read and validate the actual
    process environment and actual argv before initializing the data
    path.  It may not synthesize expected values and hash those as
    observations.

    For selector=disabled (NCCL arms), validates that NCCL_NET and
    NCCL_IB_DISABLE are present and consistent:
    + NCCL_NET=IB requires NCCL_IB_DISABLE=0 (NCCL-IB control arm).
    + NCCL_NET=Socket requires NCCL_IB_DISABLE=1 (Socket diagnostic arm).
    + NCCL_NET=IB with NCCL_IB_DISABLE=1 is a contradiction — rejected.
    + NCCL_NET=Socket with NCCL_IB_DISABLE=0 is a contradiction — rejected.

    Fails before any data collective on missing, extra, or mismatched
    behavior-affecting fields.
    """
    if selector == SELECTOR_CUSTOM:
        # SIRCL arm: NCCL_NET and NCCL_IB_DISABLE must NOT be set.
        # If they are set, the process environment is inconsistent with
        # the SIRCL selector.
        nccl_net = os.getenv("NCCL_NET")
        if nccl_net is not None:
            print(
                f"ERROR: selector=custom but NCCL_NET={nccl_net} is set "
                f"in the process environment — SIRCL arm must not have "
                f"NCCL transport env vars",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)
        return

    # selector=disabled: NCCL arm.  NCCL_NET and NCCL_IB_DISABLE must
    # be present and consistent.
    nccl_net = os.getenv("NCCL_NET", "")
    nccl_ib_disable = os.getenv("NCCL_IB_DISABLE", "")

    if not nccl_net:
        print(
            "ERROR: selector=disabled requires NCCL_NET to be set "
            "in the process environment",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
    if not nccl_ib_disable:
        print(
            "ERROR: selector=disabled requires NCCL_IB_DISABLE to be set "
            "in the process environment",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    nccl_net_upper = nccl_net.upper()
    if nccl_net_upper == "IB" and nccl_ib_disable != "0":
        print(
            f"ERROR: NCCL_NET=IB requires NCCL_IB_DISABLE=0, "
            f"got NCCL_IB_DISABLE={nccl_ib_disable}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
    if nccl_net_upper == "SOCKET" and nccl_ib_disable != "1":
        print(
            f"ERROR: NCCL_NET=Socket requires NCCL_IB_DISABLE=1, "
            f"got NCCL_IB_DISABLE={nccl_ib_disable}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
    if nccl_net_upper not in ("IB", "SOCKET"):
        print(
            f"ERROR: NCCL_NET must be 'IB' or 'Socket', "
            f"got '{nccl_net}'",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Return the raw bytes of a tensor for SHA-256 hashing.

    BF16 has no native numpy dtype, so the bits are reinterpreted as
    int16 before conversion.  The tensor is moved to CPU and made
    contiguous first.
    """
    t = tensor.cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes()
    return t.numpy().tobytes()


def _tensor_bytes_fp32(tensor: torch.Tensor) -> bytes:
    """Return the raw bytes of an FP32 tensor for SHA-256 hashing.

    The unrounded FP32 reference is hashed in its native float32
    representation — this is the true FP32 ground truth, not a
    BF16-rounded proxy.
    """
    t = tensor.cpu().contiguous()
    return t.numpy().tobytes()

def _hash_argv() -> str:
    """Hash the actual sys.argv for the run contract.

    Goal 11 requirement 3: the probe must read and validate the actual
    process argv before initializing the data path.  It may not
    synthesize expected values and hash those as observations.
    """
    return hashlib.sha256(
        json.dumps(list(sys.argv), sort_keys=True).encode()
    ).hexdigest()


def _hash_allowlisted_env() -> str:
    """Hash the actual allowlisted environment variables.

    Goal 11 requirement 3: hash and validate actual os.environ entries
    that are on the ENV_ALLOWLIST.  Unknown or missing behavior-affecting
    fields fail before data collectives.
    """
    env_pairs = {
        k: os.environ.get(k, "")
        for k in sorted(ENV_ALLOWLIST)
        if os.environ.get(k) is not None
    }
    return hashlib.sha256(
        json.dumps(env_pairs, sort_keys=True).encode()
    ).hexdigest()

# Env vars that legitimately differ per-rank and must be excluded
# from cross-rank shared identity validation.
#
# Review v4 requirement 1: exclude canonical per-rank topology
# values that legitimately differ across ranks.  These are set by
# the launcher/runtime on a per-rank basis (confirmed against
# scripts/sparkring_launcher.py _SITE_DERIVED_ENVIRONMENT and
# scripts/sparkring_runtime.py SITE_DERIVED_ENVIRONMENT).
# Including them in the shared hash would cause false consensus
# failures on legitimate per-rank topology.
_PER_RANK_ENV_VARS = frozenset({
    "RANK", "WORLD_SIZE",
    # Per-rank topology: peer addresses, device names, GIDs, and HCA.
    "SPARK_TP4_PEER0", "SPARK_TP4_PEER1",
    "SPARK_TP4_DEVICE0", "SPARK_TP4_DEVICE1",
    "SPARK_TP4_GID0", "SPARK_TP4_GID1",
    "NCCL_IB_HCA",
})


def _hash_shared_env() -> str:
    """Hash allowlisted env vars that must be identical across ranks.

    Goal 12 review v2 requirement 5: MASTER_ADDR, MASTER_PORT, and
    GLOO_SOCKET_IFNAME must participate in the allowlisted
    environment hash and cross-rank shared validation.  A mismatch
    in any one must abort pre-data.  RANK and WORLD_SIZE are per-rank
    fields (already validated separately); excluding them from the
    env hash lets all ranks agree on the shared environment.

    Review v4 requirement 1: canonical per-rank topology values
    (SPARK_TP4_PEER0/1, SPARK_TP4_DEVICE0/1, SPARK_TP4_GID0/1,
    NCCL_IB_HCA) are also excluded — they legitimately differ per
    rank.  Confirmed against scripts/sparkring_launcher.py
    _SITE_DERIVED_ENVIRONMENT and scripts/sparkring_runtime.py
    SITE_DERIVED_ENVIRONMENT, which mark these as site-derived
    (per-rank) environment keys.
    """
    env_pairs = {
        k: os.environ.get(k, "")
        for k in sorted(ENV_ALLOWLIST - _PER_RANK_ENV_VARS)
        if os.environ.get(k) is not None
    }
    # Closed environment: these are not in ENV_ALLOWLIST but must be
    # identical across ranks for a valid Gloo rendezvous.
    for closed_key in ("MASTER_ADDR", "MASTER_PORT", "GLOO_SOCKET_IFNAME"):
        env_pairs[closed_key] = os.environ.get(closed_key, "")
    return hashlib.sha256(
        json.dumps(env_pairs, sort_keys=True).encode()
    ).hexdigest()


def _compute_source_sha() -> str:
    """Compute SHA-256 of the probe source file for identity binding."""
    import pathlib
    p = pathlib.Path(__file__)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _get_sircl_so_sha() -> str:
    """Get SHA-256 of the SIRCL .so if present, else empty string.

    Goal 12 requirement 2: bind the loaded library bytes to an
    observed SHA-256.  Check SPARK_TP4_LIBRARY (the env var read by
    _NativeSession) as well as VLLM_SPARK_TRANSPORT_SO_PATH.
    """
    so_path = os.getenv("SPARK_TP4_LIBRARY", "")
    if not so_path:
        so_path = os.getenv("VLLM_SPARK_TRANSPORT_SO_PATH", "")
    if not so_path:
        return ""
    import pathlib
    try:
        p = pathlib.Path(so_path)
        if p.exists():
            return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        pass
    return ""


def _get_nccl_so_sha() -> str:
    """Get SHA-256 of the patched NCCL .so if present, else empty string."""
    so_path = os.getenv("VLLM_NCCL_SO_PATH", "")
    if not so_path:
        return ""
    import pathlib
    try:
        p = pathlib.Path(so_path)
        if p.exists():
            return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        pass
    return ""


def _get_image_receipt() -> str:
    """Get the container/image receipt identifier if present.

    Review v4 requirement 3: consume the canonical launcher
    identity ``SPARKRING_IMAGE_DIGEST``.  The repo has no producer
    for ``VLLM_SPARK_IMAGE_RECEIPT`` — the launcher
    (scripts/sparkring_launcher.py) and runtime
    (scripts/sparkring_runtime.py) inject ``SPARKRING_IMAGE_DIGEST``
    as the canonical image identity.  ``VLLM_SPARK_IMAGE_RECEIPT``
    is accepted only as a legacy fallback when the canonical env
    is absent.
    """
    digest = os.getenv("SPARKRING_IMAGE_DIGEST", "")
    if digest:
        return digest
    # Legacy fallback (no repo producer, but tolerate old envs).
    return os.getenv("VLLM_SPARK_IMAGE_RECEIPT", "")


def _get_nccl_library_path() -> str:
    """Return the actual NCCL library path from the environment.

    Goal 12 review requirement 2: include the actual NCCL library path
    for the transport required by the selected arm, not just its hash.
    """
    return os.getenv("VLLM_NCCL_SO_PATH", "")


def _get_runtime_identity() -> str:
    """Return a compact runtime identity string.

    Goal 12 review requirement 2: bind the runtime identity (Python
    version + Torch version + platform) into the closed record so all
    ranks share the same execution environment.

    Review v5 requirement 3: _get_runtime_identity must read and hash
    verified canonical runtime evidence bytes, not append an unchecked
    path.  For SPARKRING_RUNTIME_MANIFEST, require an existing regular
    file, read it safely, hash its bytes, and include a versioned field
    in the closed record.  If canonical runtime receipt/lock hashes
    are available (SPARKRING_IMAGE_DIGEST), consume validated values
    explicitly.  A nonexistent path, directory, unreadable file, or
    mismatched/empty evidence must fail closed.
    """
    import pathlib
    import platform
    parts = [
        "py=" + platform.python_version(),
        "torch=" + torch.__version__,
        "plat=" + platform.platform(),
    ]

    # Canonical SparkRing runtime manifest: read and hash the actual
    # file bytes — never append an unchecked path string.
    manifest_path = os.getenv("SPARKRING_RUNTIME_MANIFEST", "")
    if manifest_path:
        p = pathlib.Path(manifest_path)
        if not p.exists():
            raise RuntimeError(
                f"SPARKRING_RUNTIME_MANIFEST={manifest_path} does not "
                f"exist — cannot bind runtime identity to a nonexistent "
                f"manifest"
            )
        if not p.is_file():
            raise RuntimeError(
                f"SPARKRING_RUNTIME_MANIFEST={manifest_path} is not a "
                f"regular file — directories and special files are "
                f"rejected as runtime evidence"
            )
        try:
            manifest_bytes = p.read_bytes()
        except OSError as exc:
            raise RuntimeError(
                f"SPARKRING_RUNTIME_MANIFEST={manifest_path} is not "
                f"readable — cannot hash unreadable manifest: {exc}"
            ) from exc
        if not manifest_bytes:
            raise RuntimeError(
                f"SPARKRING_RUNTIME_MANIFEST={manifest_path} is empty "
                f"— empty evidence cannot bind runtime identity"
            )
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        parts.append("manifest_sha=sha256:" + manifest_sha)

    # Canonical image digest (validated by the launcher from the
    # actual container registry — a hash value, not a path).
    image_digest = os.getenv("SPARKRING_IMAGE_DIGEST", "")
    if image_digest:
        parts.append("image=" + image_digest)

    return "|".join(parts)

def _check_native_capable(library_path: str) -> bool:
    """Check whether a library path points to a loadable SIRCL native library.

    Review v4 requirement 2: a generic ``ctypes.CDLL``-loadable
    libc/kernel32 is NOT SIRCL capability.  The library must
    export the required SIRCL C API symbols used by
    ``spark_tp4_backend.py``: at minimum ``spark_tp4_create``,
    ``spark_tp4_all_reduce``, and ``spark_tp4_destroy``.

    We attempt to load the library with ``ctypes.CDLL`` and verify
    each required symbol is present as a callable attribute.
    A non-loadable file or a library missing any required symbol
    returns False.  No environment variable, filename substring,
    or nonexistent path may bypass this check.
    """
    # Required SIRCL C API symbols (confirmed against
    # spark_tp4_backend.py _NativeSession.__init__).
    _REQUIRED_SIRCL_SYMBOLS = (
        "spark_tp4_create",
        "spark_tp4_all_reduce",
        "spark_tp4_destroy",
    )
    if not library_path:
        return False
    try:
        import ctypes
        import pathlib
        p = pathlib.Path(library_path)
        if not p.is_file():
            return False
        # Attempt actual dynamic load.
        lib = ctypes.CDLL(str(p))
        # Verify each required SIRCL C API symbol is exported.
        for symbol in _REQUIRED_SIRCL_SYMBOLS:
            if not hasattr(lib, symbol):
                return False
        return True
    except Exception:
        return False


def _check_nccl_identity(library_path: str) -> bool:
    """Check whether a library path points to an actual NCCL library.

    Review v4 requirement 4: a generic ``ctypes.CDLL``-loadable
    libc/kernel32 is NOT an NCCL library.  We verify the library
    exports NCCL-specific symbols.  The NCCL C API exports
    ``ncclCommInitRank`` (or ``ncclCommInitRankAll``) as its
    core init entry point — a generic system library will not have
    these.

    No environment variable, filename substring, or nonexistent path
    may bypass this check.  A text file or libc/kernel32 renamed
    to contain ``nccl`` is forbidden.
    """
    if not library_path:
        return False
    try:
        import ctypes
        import pathlib
        p = pathlib.Path(library_path)
        if not p.is_file():
            return False
        lib = ctypes.CDLL(str(p))
        # NCCL C API core symbols — a generic libc/kernel32
        # will not export these.
        for symbol in ("ncclCommInitRank", "ncclCommInitRankAll"):
            if hasattr(lib, symbol):
                return True
        return False
    except Exception:
        return False


def _collect_rank_identity_record(
    selector: str,
    rank: int,
    world_size: int,
    iterations: int,
    elements: int,
) -> dict:
    """Build a closed per-rank pre-data identity record.

    Goal 12 requirement 2: this record is collected BEFORE any SIRCL
    or NCCL data transport initialization.  It contains exact rank/world,
    selector/arm, workload dimensions, native capability, actual
    argv/environment hashes, source hash, required SIRCL/NCCL library
    path+hash, and image/runtime identity.

    Review requirement 1: include exact workload dimensions (iterations,
    elements) and validate them across ranks.
    Review requirement 2: include both path and hash for the transport
    library required by the selected arm, plus image and runtime
    identities.
    """
    library_path = os.getenv("SPARK_TP4_LIBRARY", "")
    sircl_sha = _get_sircl_so_sha() if library_path else ""
    nccl_path = _get_nccl_library_path()
    nccl_sha = _get_nccl_so_sha()

    # Review v4 requirement 2: native capability must verify the
    # library exports the required SIRCL C API symbols, not just
    # that a generic library is CDLL-loadable.
    native_capable = _check_native_capable(library_path)
    # Review v4 requirement 4: NCCL identity must prove the library
    # is an actual NCCL library, not a generic system library.
    nccl_identity = _check_nccl_identity(nccl_path)

    return {
        "rank": rank,
        "world_size": world_size,
        "selector": selector,
        "arm": (
            TRANSPORT_SIRCL if selector == SELECTOR_CUSTOM
            else TRANSPORT_NCCL_IB
        ),
        "workload": "tp4_numerical_audit",
        "iterations": iterations,
        "elements": elements,
        "native_capable": native_capable,
        "argv_hash": _hash_argv(),
        "env_hash": _hash_allowlisted_env(),
        "shared_env_hash": _hash_shared_env(),
        "source_sha": _compute_source_sha(),
        "sircl_library_path": library_path,
        "sircl_library_sha": sircl_sha,
        "nccl_library_path": nccl_path,
        "nccl_library_sha": nccl_sha,
        "nccl_identity": nccl_identity,
        "image_receipt": _get_image_receipt(),
        "runtime_identity": _get_runtime_identity(),
        "master_addr": os.getenv("MASTER_ADDR", ""),
        "master_port": os.getenv("MASTER_PORT", ""),
        "gloo_socket_ifname": os.getenv("GLOO_SOCKET_IFNAME", ""),
    }


def _run_control_plane_consensus(
    selector: str,
    rank: int,
    world_size: int,
    *,
    control_backend=None,
    identity_record: dict | None = None,
) -> str:
    """Run synchronized four-rank consensus on the control plane.

    Goal 12 requirement 1: use module-level ``dist.all_reduce(...,
    group=control_group)`` and ``dist.barrier(group=control_group)``.
    A ``ProcessGroup`` object is not a module and must not be invoked
    as ``group.all_reduce``, ``group.ReduceOp``, or ``group.barrier``
    unless the exact installed Torch API proves that contract.

    Goal 12 requirement 2: validate exact ranks {0,1,2,3} and
    cross-rank shared identity via ``dist.all_gather_object`` before
    any SIRCL or NCCL data transport initialization.  One-rank
    missing library/capability/mismatch aborts every rank with zero
    native and zero NCCL data calls.

    For test mode, ``control_backend`` is a mock object with
    ``all_reduce``, ``all_gather``, ``all_gather_object``,
    ``barrier``, and ``ReduceOp.SUM`` attributes — the test mock is
    NOT a ProcessGroup, it is a test double.  In production,
    ``control_backend`` is a ``ProcessGroup`` from
    ``dist.new_group(backend="gloo")``, and we call
    ``dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=control_pg)``.

    Defect 1 repair: ``control_backend=None`` is a fail-closed
    condition — missing rendezvous/init failure must never reach a
    data call or success receipt.
    """
    if control_backend is None:
        # Defect 1: fail closed — a missing control backend means the
        # control plane is not initialized.  Production must never
        # silently proceed to data transport without consensus.
        raise RuntimeError(
            "control-plane consensus failed: control backend is None "
            "— control group not initialized, cannot proceed to data "
            "transport (rank {rank})".format(rank=rank)
        )

    import torch as _torch

    # Distinguish production ProcessGroup from test mock.
    is_production_pg = isinstance(control_backend, dist.ProcessGroup) \
        if hasattr(dist, "ProcessGroup") else False

    # Encode selector as an integer: custom=0, disabled=1
    selector_code = 0 if selector == SELECTOR_CUSTOM else 1
    selector_tensor = _torch.tensor(selector_code, device="cpu")

    if is_production_pg:
        dist.all_reduce(
            selector_tensor,
            op=dist.ReduceOp.SUM,
            group=control_backend,
        )
    else:
        control_backend.all_reduce(
            selector_tensor, op=control_backend.ReduceOp.SUM,
        )
    consensus_sum = int(selector_tensor.item())
    expected_sum = selector_code * world_size
    if consensus_sum != expected_sum:
        raise RuntimeError(
            f"control-plane consensus failed: selector sum {consensus_sum} "
            f"!= expected {expected_sum} (rank {rank}, world_size "
            f"{world_size}) — ranks disagree on transport"
        )

    # Goal 12 requirement 2: validate rank identity via all_gather.
    rank_tensor = _torch.tensor(rank, device="cpu")
    gathered = [_torch.tensor(0, device="cpu") for _ in range(world_size)]
    if is_production_pg:
        dist.all_gather(gathered, rank_tensor, group=control_backend)
    else:
        control_backend.all_gather(gathered, rank_tensor, group=None)
    gathered_ranks = sorted(int(t.item()) for t in gathered)
    expected_ranks = list(range(world_size))
    if gathered_ranks != expected_ranks:
        raise RuntimeError(
            f"control-plane consensus failed: rank set {gathered_ranks} "
            f"!= expected {expected_ranks} (duplicate, missing, or "
            f"out-of-range rank detected)"
        )

    # Goal 12 requirement 2: cross-rank identity validation via
    # all_gather_object on the control group.  Each rank contributes
    # its pre-data identity record; all records are gathered and
    # validated for exact rank coverage and shared identity.
    if identity_record is None:
        identity_record = _collect_rank_identity_record(
            selector, rank, world_size, 0, 0,
        )


    gathered_records: list[dict] = [{} for _ in range(world_size)]
    if is_production_pg:
        dist.all_gather_object(
            gathered_records, identity_record, group=control_backend,
        )
    else:
        control_backend.all_gather_object(
            gathered_records, identity_record, group=control_backend,
        )

    # Validate exact rank coverage {0, 1, ..., world_size-1}.
    record_ranks = sorted(r.get("rank", -1) for r in gathered_records)
    if record_ranks != expected_ranks:
        raise RuntimeError(
            f"control-plane consensus failed: identity record ranks "
            f"{record_ranks} != expected {expected_ranks} (missing, "
            f"duplicate, or out-of-range rank)"
        )

    # Validate cross-rank shared identity: all ranks must agree on
    # all closed identity fields (review requirement 3).
    shared_fields = (
        "selector", "arm",
        "workload", "iterations", "elements",
        "world_size",
        "argv_hash", "shared_env_hash", "source_sha",
        "sircl_library_path", "sircl_library_sha",
        "nccl_library_path", "nccl_library_sha",
        "nccl_identity",
        "image_receipt", "runtime_identity",
        # Review v2 requirement 5: closed environment fields must
        # be identical across ranks.
        "master_addr", "master_port", "gloo_socket_ifname",
    )
    for field in shared_fields:
        values = set(r.get(field) for r in gathered_records)
        if len(values) > 1:
            raise RuntimeError(
                f"control-plane consensus failed: cross-rank identity "
                f"field '{field}' mismatch — values {values} "
                f"(rank {rank})"
            )

    # Review v2 requirement 6: require the selected arm's library
    # path and hash to be non-empty and valid.  Equality of empty
    # strings is not attestation.
    if selector == SELECTOR_CUSTOM:
        # SIRCL arm: sircl_library_path and sircl_library_sha must
        # be non-empty on every rank.
        for rec in gathered_records:
            r = rec.get("rank", -1)
            if not rec.get("sircl_library_path"):
                raise RuntimeError(
                    f"control-plane consensus failed: rank {r} has "
                    f"empty sircl_library_path — SIRCL arm requires "
                    f"a non-empty library path"
                )
            if not rec.get("sircl_library_sha"):
                raise RuntimeError(
                    f"control-plane consensus failed: rank {r} has "
                    f"empty sircl_library_sha — SIRCL arm requires "
                    f"a non-empty library hash"
                )
    else:
        # NCCL arm: nccl_library_path and nccl_library_sha must be
        # non-empty on every rank.
        for rec in gathered_records:
            r = rec.get("rank", -1)
            if not rec.get("nccl_library_path"):
                raise RuntimeError(
                    f"control-plane consensus failed: rank {r} has "
                    f"empty nccl_library_path — NCCL arm requires "
                    f"a non-empty library path"
                )
            if not rec.get("nccl_library_sha"):
                raise RuntimeError(
                    f"control-plane consensus failed: rank {r} has "
                    f"empty nccl_library_sha — NCCL arm requires "
                    f"a non-empty library hash"
                )
        # Review v4 requirement 4: verify NCCL identity — the
        # library at nccl_library_path must be an actual NCCL
        # library, not a generic system library.  If any rank's
        # NCCL library fails the identity check, abort all ranks.
        for rec in gathered_records:
            r = rec.get("rank", -1)
            if not rec.get("nccl_identity", False):
                raise RuntimeError(
                    f"control-plane consensus failed: rank {r} "
                    f"nccl_library_path does not point to an actual "
                    f"NCCL library — NCCL identity not proven"
                )

    # Review v4 requirement 3: a canonical digest must produce
    # nonempty image identity.  When SPARKRING_IMAGE_DIGEST is set
    # in the environment, every rank's image_receipt must be
    # non-empty.  An empty or mismatched digest fails closed.
    if os.getenv("SPARKRING_IMAGE_DIGEST"):
        for rec in gathered_records:
            r = rec.get("rank", -1)
            if not rec.get("image_receipt"):
                raise RuntimeError(
                    f"control-plane consensus failed: rank {r} has "
                    f"empty image_receipt but SPARKRING_IMAGE_DIGEST "
                    f"is set — canonical image identity required"
                )

    # Validate native capability: if any rank is not native_capable and
    # the selector is custom (SIRCL), abort every rank.
    if selector == SELECTOR_CUSTOM:
        incapable = [
            r.get("rank") for r in gathered_records
            if not r.get("native_capable", False)
        ]
        if incapable:
            raise RuntimeError(
                f"control-plane consensus failed: rank(s) {incapable} "
                f"lack native SIRCL capability (missing SPARK_TP4_LIBRARY) "
                f"— aborting all ranks before any data call"
            )

    # Barrier on control group: all ranks proceed together.
    if is_production_pg:
        dist.barrier(group=control_backend)
    else:
        control_backend.barrier()
    return selector




def _init_control_group(
    rank: int,
    world_size: int,
    *,
    timeout: float = 30.0,
):
    """Initialize a dedicated Gloo/TCP control process group.

    Goal 12 requirement 1: Initialize a real default Gloo process
    group BEFORE creating any sub-group.  ``dist.new_group`` requires
    the default process group to already exist; calling it before
    ``dist.init_process_group`` is undefined.

    The control channel is NOT the transport under test.  This uses
    Gloo on the management network (TCP), never NCCL or the RoCE
    fabric.

    Defect 1 repair: fail closed on any initialization or subgroup
    failure.  Returns the Gloo control ProcessGroup, or raises
    RuntimeError — never returns None to allow silent success.

    Review requirement 5: explicit short PG timeout for deterministic
    cleanup/join in the real-process test.
    """
    import datetime as _dt

    if not dist.is_available():
        raise RuntimeError(
            "control-plane init failed: torch.distributed is not "
            "available — cannot initialize control group"
        )
    pg_timeout = _dt.timedelta(seconds=timeout)
    # Initialize the DEFAULT process group with Gloo first.
    # This must happen before any dist.new_group() call.
    # The rendezvous method is env:// by default, which uses
    # MASTER_ADDR/MASTER_PORT from the environment.
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            timeout=pg_timeout,
        )
    # Now create a dedicated sub-group for control-plane operations.
    # This sub-group uses Gloo (management network), never NCCL.
    try:
        control_pg = dist.new_group(
            ranks=list(range(world_size)),
            backend="gloo",
            timeout=pg_timeout,
        )
    except Exception as exc:
        raise RuntimeError(
            f"control-plane init failed: cannot create Gloo sub-group: {exc}"
        ) from exc
    if control_pg is None:
        raise RuntimeError(
            "control-plane init failed: new_group returned None"
        )
    return control_pg



def _destroy_control_group(control_pg) -> None:
    """Destroy the control process group."""
    if control_pg is not None:
        try:
            dist.destroy_process_group(control_pg)
        except Exception:
            pass
def _increment_recorder(recorder, key: str) -> None:
    """Atomically increment a counter in the shared data-call recorder.

    Review v3 requirement 7: shared recorder increments must be
    atomic across processes.  If the recorder provides a ``_lock``
    attribute (a multiprocessing Manager lock), use it; otherwise
    fall back to a non-atomic read-modify-write (sufficient for
    single-process tests).
    """
    if recorder is None:
        return
    lock = getattr(recorder, "_lock", None)
    if lock is not None:
        with lock:
            recorder[key] = recorder.get(key, 0) + 1
    else:
        recorder[key] = recorder.get(key, 0) + 1

def run_probe(
    selector: str,
    rank: int,
    iterations: int,
    elements: int,
    world_size: int,
    native_session=None,
    dist_backend=None,
    *,
    data_call_recorder=None,
) -> dict:
    """Execute the numerical audit probe and return a receipt dict.

    This function contains the exact collective execution + numerical
    validation logic.  ``main()`` calls it with defaults from env
    vars; tests call it with injected mocks.

    Parameters
    ----------
    selector : str
        Transport selector (``"custom"`` or ``"disabled"``).
    rank : int
        Local rank index.
    iterations : int
        Number of all-reduce iterations.
    elements : int
        Tensor element count per rank.
    world_size : int
        Number of ranks participating in the collective.
    native_session : object, optional
        Object with an ``all_reduce(tensor)`` method.  When ``None``
        and selector is ``"custom"``, the real ``_NativeSession`` is
        imported and instantiated (requires CUDA).
    dist_backend : object, optional
        Object with ``all_reduce``, ``barrier``, ``destroy_process_group``
        methods and a ``ReduceOp.SUM`` attribute.  When ``None``, the
        real ``torch.distributed`` is used (requires CUDA/NCCL).

    Returns
    -------
    dict
        Receipt with counter fields and numerical evidence fields.

    Raises
    ------
    ValueError
        If any output contains NaN/Inf, if the output element count
        does not match ``elements``, or if the elementwise tolerance
        criterion is not met.
    RuntimeError
        If the native transport raises after work may have been
        enqueued.  Once native is invoked, exceptions are fatal —
        the probe does not fall back to NCCL.
    """
    # Determine mode: production (real CUDA/dist) or test (injected mocks).
    injected = native_session is not None or dist_backend is not None

    control_pg = None
    # Goal 12: observed library SHA (computed in production mode).
    _production_library_sha = None
    # Defect 3 repair: local variable for the NCCL data process group.
    # Never stored on the global dist module.
    nccl_data_pg = None

    if not injected:
        control_pg = _init_control_group(rank, world_size)
        # Goal 12 requirement 2: collect per-rank identity record
        # BEFORE consensus and BEFORE any data transport.
        identity_record = _collect_rank_identity_record(
            selector, rank, world_size, iterations, elements,
        )
        # Run consensus on the control plane before any data transport.
        # Identity collection and consensus occur before DATA, never after.
        _run_control_plane_consensus(
            selector, rank, world_size,
            control_backend=control_pg,
            identity_record=identity_record,
        )
    if not injected:
        # Production mode: initialize CUDA.
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)

        # Goal 12 requirement 1: the default process group is already
        # initialized (Gloo) by _init_control_group above.  The NCCL
        # data arm must create a SUB-GROUP with backend=nccl, not call
        # dist.init_process_group again — that would re-init the
        # default group and destroy the Gloo control plane.
        # The SIRCL arm must NEVER create or use an NCCL data process
        # group.
        if selector == SELECTOR_DISABLED:
            # Defect 3 repair: never store a data process group on the
            # global torch.distributed module.  Keep groups local and
            # pass them explicitly.
            nccl_data_pg = dist.new_group(
                ranks=list(range(world_size)),
                backend="nccl",
            )
            # d remains dist (the module) but we pass nccl_data_pg
            # explicitly to collective calls — no attribute mutation.
            d = dist
        else:
            # SIRCL arm: no NCCL data group.  The native session
            # handles all inter-rank communication via the RoCE
            # fabric directly — NCCL is not involved.
            d = None  # SIRCL arm does not use dist for data
            nccl_data_pg = None  # no NCCL group for SIRCL

            # Goal 12 requirement 2: instantiate _NativeSession with its
            # real production signature: (rank, payload_bytes).
            # payload_bytes = elements * sizeof(bfloat16) = elements * 2
            payload_bytes = elements * 2  # BF16 = 2 bytes per element

            # Goal 12 requirement 2: set and validate SPARK_TP4_LIBRARY
            # before constructing _NativeSession.
            library_path = os.environ.get("SPARK_TP4_LIBRARY", "")
            if not library_path:
                raise RuntimeError(
                    "SPARK_TP4_LIBRARY is not set — cannot create "
                    "native SIRCL session (Goal 12 requirement 2)"
                )
            if not os.path.isfile(library_path):
                raise RuntimeError(
                    f"SPARK_TP4_LIBRARY={library_path} does not exist "
                    f"as a file — cannot create native SIRCL session"
                )
            with open(library_path, "rb") as f:
                library_bytes = f.read()
            library_sha = hashlib.sha256(library_bytes).hexdigest()
            _production_library_sha = library_sha  # for receipt

            from spark_tp4_backend import _NativeSession
            native_session = _NativeSession(rank, payload_bytes)
    else:
        device = torch.device("cpu")
        d = dist_backend if dist_backend is not None else dist


    # Observed counters (per-rank) — externally attributable.
    # Goal 11 requirement 5: per-rank counters distinguish
    # native SIRCL, patched NCCL-IB, TCP Socket diagnostic,
    # unsupported, and fatal-after-native.
    native_count = 0
    nccl_ib_count = 0
    nccl_socket_count = 0
    fallback_count = 0  # legacy alias for NCCL counts
    unsupported_count = 0
    unclassified_count = 0
    fatal_after_native_count = 0
    total_collectives = 0

    # Numerical evidence accumulators.
    all_finite = True
    max_abs_error = 0.0
    max_rel_error = 0.0
    sample_count = 0
    tolerance_result = "pass"

    # Running SHA-256 hashers across all iterations.
    fp32_hasher = hashlib.sha256()
    output_hasher = hashlib.sha256()

    for sequence in range(iterations):
        cpu_inputs = [
            make_rank_input(sequence, source_rank, elements)
            for source_rank in range(world_size)
        ]

        # FP32 reference: sum all rank inputs in FP32 (unrounded).
        # This is the true FP32 ground truth — we hash this, not a
        # BF16-rounded proxy.
        fp32_sum = torch.stack([t.float() for t in cpu_inputs]).sum(dim=0)

        # Local input — move to device only in production mode.
        local = cpu_inputs[rank] if injected else cpu_inputs[rank].to(device=device)

        # Execute the all-reduce via the selected transport.
        #
        # Failure boundary (Goal 9): once the native transport is
        # invoked, any exception is process/run fatal.  The probe
        # does NOT fall back to NCCL after native work may have been
        # enqueued.  Fallback must be a pre-attempt capability
        # decision made before native submission, not exception
        # recovery.
        candidate = None
        classification = CLASS_UNCLASSIFIED

        if selector == SELECTOR_CUSTOM:
            if native_session is not None:
                # Review v3 requirement 7: increment the shared
                # data-call recorder BEFORE the actual SIRCL call so
                # attempted calls are counted even if the call raises.
                _increment_recorder(data_call_recorder, "sircl_calls")
                # Native invocation — exceptions propagate as fatal.
                # No try/except, no NCCL fallback, no silent
                # transport switch.  The exception exits run_probe
                # and the caller (main or executor) handles it.
                candidate = native_session.all_reduce(local)
                classification = CLASS_NATIVE
            else:
                # selector=custom but no native_session available:
                # This is a pre-attempt capability miss.  The probe
                # MUST NOT independently execute NCCL while labeling
                # the result unclassified (Goal 10 requirement 1).
                # Either all ranks select the explicit control/fallback
                # arm before submission, or the run fails before any
                # collective.  We fail closed here — no NCCL call.
                raise RuntimeError(
                    "selector=custom requires a native session; "
                    "no native_session available — failing closed "
                    "before any collective (Goal 10: selector=custom "
                    "must not independently execute NCCL as unclassified)"
                )
        elif selector == SELECTOR_DISABLED:
            candidate = local.clone()
            # Review v3 requirement 7: increment the shared
            # data-call recorder BEFORE the actual NCCL call so
            # attempted calls are counted even if the call raises.
            _increment_recorder(data_call_recorder, "nccl_calls")
            if not injected and nccl_data_pg is not None:
                # Production: use module-level dist.all_reduce with
                # group=nccl_data_pg (the local NCCL sub-group).
                dist.all_reduce(candidate, op=dist.ReduceOp.SUM, group=nccl_data_pg)
            else:
                d.all_reduce(candidate, op=d.ReduceOp.SUM)
            # Goal 11 requirement 5: distinguish NCCL-IB from
            # NCCL-Socket by the actual NCCL_NET env var.
            nccl_net = os.getenv("NCCL_NET", "").upper()
            if nccl_net == "IB":
                classification = CLASS_NCCL_IB
            elif nccl_net == "SOCKET":
                classification = CLASS_NCCL_SOCKET
            else:
                classification = CLASS_FALLBACK

        total_collectives += 1
        if classification == CLASS_NATIVE:
            native_count += 1
        elif classification == CLASS_NCCL_IB:
            nccl_ib_count += 1
            fallback_count += 1  # legacy alias
        elif classification == CLASS_NCCL_SOCKET:
            nccl_socket_count += 1
            fallback_count += 1  # legacy alias
        elif classification == CLASS_FALLBACK:
            fallback_count += 1
        elif classification == CLASS_UNSUPPORTED:
            unsupported_count += 1
        else:
            unclassified_count += 1

        # Inspect actual output — move to CPU for inspection.
        actual = candidate.cpu()

        # Reject NaN/Inf — raise error, no receipt.
        if not bool(torch.isfinite(actual).all()):
            raise ValueError(
                f"Non-finite output detected at iteration {sequence} "
                f"(rank {rank})"
            )
        all_finite = all_finite and True  # already verified finite above

        # Verify candidate tensor dtype (Goal 10/11 requirement 4).
        # The receipt's actual_dtype must reflect the real output dtype,
        # not a hardcoded assumption.  Goal 11: reject float32 or any
        # other dtype before emitting a receipt — successful output
        # must be exactly torch.bfloat16.
        actual_dtype_str = str(actual.dtype).replace("torch.", "")
        if actual_dtype_str != REQUIRED_OUTPUT_DTYPE:
            raise ValueError(
                f"Output dtype must be {REQUIRED_OUTPUT_DTYPE}, "
                f"got '{actual_dtype_str}' at iteration {sequence} "
                f"(rank {rank}) — no receipt emitted"
            )
        # Compute errors in FP32 against the unrounded FP32 reference.
        actual_f32 = actual.float()
        abs_err = (actual_f32 - fp32_sum).abs()
        iter_max_abs = float(abs_err.max().item())
        if iter_max_abs > max_abs_error:
            max_abs_error = iter_max_abs

        # Relative error (zero-denominator: use abs when fp32_ref == 0).
        ref_abs = fp32_sum.abs()
        rel_err = torch.where(
            ref_abs > 0.0,
            abs_err / ref_abs,
            abs_err,
        )
        iter_max_rel = float(rel_err.max().item())
        if iter_max_rel > max_rel_error:
            max_rel_error = iter_max_rel

        # Elementwise tolerance criterion (Goal 9):
        # abs_error <= atol + rtol * abs(reference) for every element.
        # This correctly accepts BF16-rounded outputs whose reference
        # values are large (where absolute error can exceed 2^{-7}
        # but relative error is within bound).
        tolerance_bound = BF16_ATOL + BF16_RTOL * ref_abs
        iter_tolerance_fail = bool((abs_err > tolerance_bound).any().item())
        if iter_tolerance_fail:
            tolerance_result = "fail"

        sample_count += elements

        # Hash the unrounded FP32 reference (not the BF16 proxy).
        fp32_hasher.update(_tensor_bytes_fp32(fp32_sum))
        # Hash actual output in its declared dtype/byte order.
        output_hasher.update(_tensor_bytes(actual))

    # Elementwise tolerance criterion (Goal 9): if any element failed,
    # the run is numerically wrong — raise, do not emit a receipt.
    if tolerance_result == "fail":
        raise ValueError(
            f"Numerical tolerance violation: elementwise criterion "
            f"{TOLERANCE_METRIC} (atol={BF16_ATOL}, rtol={BF16_RTOL}) "
            f"not met — max_abs_error={max_abs_error}, "
            f"max_rel_error={max_rel_error} (rank {rank})"
        )

    # Build complete immutable run contract from canonical JSON binding.
    # Transport is determined from the actual process environment: the
    # selector alone is insufficient because both NCCL-IB and NCCL-Socket
    # use selector=disabled.  NCCL_NET distinguishes them.
    if selector == SELECTOR_CUSTOM:
        transport = TRANSPORT_SIRCL
    else:
        nccl_net = os.getenv("NCCL_NET", "")
        if nccl_net.upper() == "IB":
            transport = TRANSPORT_NCCL_IB
        else:
            transport = TRANSPORT_NCCL_SOCKET_DIAGNOSTIC
    # Sanitized stable rank identity — not raw host strings.
    rank_identity = f"rank-{rank}-of-{world_size}"
    # Deterministic seed/input identity.
    seed_identity = "0x5A17+seq*WORLD_SIZE+rank"
    # Canonical argv for this probe — the exact command used to launch.
    argv_projection = ["python", _NATIVE_PROBE_IDENTITY]
    env_projection = _build_env_projection(
        selector, rank, world_size, iterations, elements,
        transport=transport,
    )
    contract = {
        "arm": transport,
        "selector": selector,
        "transport": transport,
        "rank": rank,
        "rank_identity": rank_identity,
        "iterations": iterations,
        "elements": elements,
        "world_size": world_size,
        "seed_identity": seed_identity,
        "argv_projection": argv_projection,
        "env_projection": env_projection,
        "probe_identity": _NATIVE_PROBE_IDENTITY,
        "binary_identity": _NATIVE_PROBE_IDENTITY,
        "topology": "tp4_switchless_ring",
        "workload": "tp4_numerical_audit",
        "order": "identical",
    }
    run_contract_hash = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode()
    ).hexdigest()

    # Goal 11 requirement 3: hash actual argv and allowlisted env.
    argv_hash = _hash_argv()
    env_hash = _hash_allowlisted_env()
    source_sha = _compute_source_sha()
    sircl_so_sha = _production_library_sha if _production_library_sha else _get_sircl_so_sha()
    nccl_so_sha = _get_nccl_so_sha()
    image_receipt = _get_image_receipt()
    # Counter source hash: bind counter source identity.
    counter_source_hash = hashlib.sha256(
        json.dumps({
            "argv_hash": argv_hash,
            "source_sha": source_sha,
            "env_hash": env_hash,
        }, sort_keys=True).encode()
    ).hexdigest()

    # Goal 11 requirement 4: validate byte order.
    if sys.byteorder != REQUIRED_BYTE_ORDER:
        raise ValueError(
            f"Byte order must be '{REQUIRED_BYTE_ORDER}', "
            f"got '{sys.byteorder}' (rank {rank})"
        )

    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "rank": rank,
        "transport": transport,
        "selector": selector,
        "iterations": iterations,
        "elements": elements,
        "world_size": world_size,
        # Goal 11 requirement 5: externally attributable counters.
        "native_collectives": native_count,
        "nccl_ib_collectives": nccl_ib_count,
        "nccl_socket_collectives": nccl_socket_count,
        "unsupported_bypassed_collectives": unsupported_count,
        "unclassified_collectives": unclassified_count,
        "fatal_after_native_collectives": fatal_after_native_count,
        # Legacy counter aliases for backward compat.
        "custom_collectives": native_count,
        "fallback_collectives": fallback_count,
        "total_collectives": total_collectives,
        "expected_fp32_hash": fp32_hasher.hexdigest(),
        "actual_output_hash": output_hasher.hexdigest(),
        "actual_dtype": actual_dtype_str,
        "actual_byte_order": sys.byteorder,
        "all_finite": all_finite,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "tolerance_result": tolerance_result,
        "tolerance_metric": TOLERANCE_METRIC,
        "tolerance_atol": BF16_ATOL,
        "tolerance_rtol": BF16_RTOL,
        "sample_count": sample_count,
        "run_contract_hash": run_contract_hash,
        "rank_identity": rank_identity,
        # Goal 11 requirement 3/5: identity/hash binding.
        "counter_source_hash": counter_source_hash,
        "source_sha": source_sha,
        "sircl_so_sha": sircl_so_sha,
        "nccl_so_sha": nccl_so_sha,
        "image_receipt": image_receipt,
    }
    if not injected:
        # Defect 3 repair: use the local nccl_data_pg variable, not
        # an attribute on the global dist module.
        if nccl_data_pg is not None:
            dist.barrier(group=nccl_data_pg)
            dist.destroy_process_group(nccl_data_pg)
        # Destroy the control group (both arms).
        _destroy_control_group(control_pg)


    return receipt

def parse_receipt_json(data: dict) -> dict:
    """Parse a rank receipt JSON with exact-key validation.

    Goal 11 requirement 4: implement a real exact-key JSON parser.
    Reject missing/extra/null fields, bool-as-number, NaN/Inf,
    negative or inconsistent counts, wrong rank coverage, and
    wrong schema/version.
    """
    import math

    if not isinstance(data, dict):
        raise ValueError("receipt must be a dict")

    # Exact-key validation: no extra keys, no missing keys.
    data_keys = set(data.keys())
    missing = RECEIPT_REQUIRED_KEYS - data_keys
    extra = data_keys - RECEIPT_REQUIRED_KEYS
    if missing:
        raise ValueError(
            f"receipt missing required keys: {sorted(missing)}"
        )
    if extra:
        raise ValueError(
            f"receipt has extra keys: {sorted(extra)}"
        )

    # Schema version must match.
    if data["schema_version"] != RECEIPT_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be '{RECEIPT_SCHEMA_VERSION}', "
            f"got '{data['schema_version']}'"
        )

    # Reject null values for any key.
    for key in sorted(data_keys):
        if data[key] is None:
            raise ValueError(f"receipt field '{key}' is null")

    # Validate integer fields: reject bool-as-number, negative, NaN/Inf.
    int_fields = [
        "rank", "iterations", "elements", "world_size",
        "native_collectives", "nccl_ib_collectives",
        "nccl_socket_collectives", "custom_collectives",
        "fallback_collectives", "unsupported_bypassed_collectives",
        "unclassified_collectives", "fatal_after_native_collectives",
        "total_collectives", "sample_count",
    ]
    for field in int_fields:
        val = data[field]
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError(
                f"receipt field '{field}' must be int, "
                f"got {type(val).__name__}"
            )
        if field not in ("rank",) and val < 0:
            raise ValueError(
                f"receipt field '{field}' must be >= 0, got {val}"
            )

    # rank can be 0 but not negative.
    if data["rank"] < 0:
        raise ValueError(f"rank must be >= 0, got {data['rank']}")

    # Count consistency: sum of categories == total_collectives.
    classified = (
        data["native_collectives"]
        + data["nccl_ib_collectives"]
        + data["nccl_socket_collectives"]
        + data["unsupported_bypassed_collectives"]
        + data["unclassified_collectives"]
        + data["fatal_after_native_collectives"]
    )
    if classified != data["total_collectives"]:
        raise ValueError(
            f"classified sum ({classified}) != total_collectives "
            f"({data['total_collectives']})"
        )

    # The legacy fields remain on the v1 wire schema until the orchestrator
    # has a typed new-counter adapter.  They are aliases, not independent
    # evidence, so contradictory values must fail closed.
    if data["custom_collectives"] != data["native_collectives"]:
        raise ValueError(
            "custom_collectives must equal native_collectives"
        )
    attributed_nccl = (
        data["nccl_ib_collectives"] + data["nccl_socket_collectives"]
    )
    if data["fallback_collectives"] != attributed_nccl:
        raise ValueError(
            "fallback_collectives must equal nccl_ib_collectives + "
            "nccl_socket_collectives"
        )

    # Float fields: reject NaN/Inf, reject bool-as-number.
    float_fields = [
        "max_abs_error", "max_rel_error",
        "tolerance_atol", "tolerance_rtol",
    ]
    for field in float_fields:
        val = data[field]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(
                f"receipt field '{field}' must be a number, "
                f"got {type(val).__name__}"
            )
        fval = float(val)
        if math.isnan(fval) or math.isinf(fval):
            raise ValueError(
                f"receipt field '{field}' must be finite, got {fval}"
            )
        if fval < 0:
            raise ValueError(
                f"receipt field '{field}' must be >= 0, got {fval}"
            )

    # all_finite must be a bool.
    if not isinstance(data["all_finite"], bool):
        raise ValueError(
            f"all_finite must be a bool, got {type(data['all_finite']).__name__}"
        )

    # String fields must be non-empty strings.
    str_fields = [
        "transport", "selector", "actual_dtype", "actual_byte_order",
        "tolerance_result", "tolerance_metric", "rank_identity",
        "counter_source_hash", "source_sha",
    ]
    for field in str_fields:
        if not isinstance(data[field], str) or not data[field]:
            raise ValueError(
                f"receipt field '{field}' must be a non-empty string"
            )

    # Hash fields: 64-char lowercase hex or empty string.
    hash_fields = [
        "expected_fp32_hash", "actual_output_hash", "run_contract_hash",
        "sircl_so_sha", "nccl_so_sha",
    ]
    import re
    for field in hash_fields:
        val = data[field]
        if not isinstance(val, str):
            raise ValueError(f"receipt field '{field}' must be a string")
        if val and not re.match(r"^[0-9a-f]{64}$", val):
            raise ValueError(
                f"receipt field '{field}' must be 64-char hex or empty"
            )

    # Container identities use Docker's canonical digest spelling.  Keep
    # this distinct from bare content hashes so one representation is used
    # from launcher through receipt validation.
    image_receipt = data["image_receipt"]
    if not isinstance(image_receipt, str):
        raise ValueError("receipt field 'image_receipt' must be a string")
    if image_receipt and not re.fullmatch(r"sha256:[0-9a-f]{64}", image_receipt):
        raise ValueError(
            "receipt field 'image_receipt' must be a canonical "
            "sha256:<64 lowercase hex> digest or empty"
        )

    # Goal 11 requirement 4: reject wrong dtype.
    if data["actual_dtype"] != REQUIRED_OUTPUT_DTYPE:
        raise ValueError(
            f"actual_dtype must be '{REQUIRED_OUTPUT_DTYPE}', "
            f"got '{data['actual_dtype']}'"
        )

    # Goal 11 requirement 4: reject wrong byte order.
    if data["actual_byte_order"] != REQUIRED_BYTE_ORDER:
        raise ValueError(
            f"actual_byte_order must be '{REQUIRED_BYTE_ORDER}', "
            f"got '{data['actual_byte_order']}'"
        )

    return data


def main() -> None:
    selector = _validate_selector()

    # Goal 11 requirement 3: read and validate the actual process
    # environment before initializing the data path.  The probe must
    # not synthesize expected values and hash those as observations.
    _validate_process_env(selector)

    rank = int(os.environ["RANK"])
    iterations = int(os.getenv("ITERATIONS", "1000"))
    elements = int(os.getenv("ELEMENTS", "6144"))

    try:
        receipt = run_probe(
            selector, rank, iterations, elements, WORLD_SIZE,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    print(
        "TP4_RECEIPT " + json.dumps(receipt, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
