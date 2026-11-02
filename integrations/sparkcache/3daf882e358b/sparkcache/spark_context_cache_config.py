"""Configuration parsing and immutable cache-identity construction.

Both the scheduler and worker connector roles share one parsing path.
``parse_connector_config`` reads all SparkCache settings from a
``VllmConfig``-like object — extra-config keys, environment variables,
parallel degrees, model profile, and KV-cache group topology. It validates
the verified-or-recompute deployment contract and returns an immutable
:class:`ConnectorConfig` carrying every value the connector needs at
construction.

The immutable cache-identity base (``identity_base``) is built here so both
roles construct identical wire bytes for the same deployment. ``SHA256_RE``
is the shared lowercase digest syntax used by construction and streaming
commit validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from sparkcache.spark_context_cache_profiles import (
    ProfileError,
    resolve_profile,
)
from sparkcache.spark_context_cache_store import (
    CacheIdentity,
    CapacityPolicy,
    validate_clear_once_request,
)
from sparkcache.streaming.feature_gate import (
    ENVIRONMENT_KEY as _STREAMING_SNAPSHOTS_ENV,
    EXTRA_CONFIG_KEY as _STREAMING_SNAPSHOTS_CONFIG,
    is_enabled as _streaming_snapshots_enabled,
)

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CUDA_PLACEMENT_ARENA_BYTES = frozenset(
    {64 * 1024 * 1024, 128 * 1024 * 1024, 256 * 1024 * 1024}
)
_MISSING = object()
_ACCESS_MODES = {
    "read-write": (True, True),
    "restore-only": (False, True),
    "store-only": (True, False),
    "disabled": (False, False),
}
_LEGACY_CUDA_RESTORE_WARNING_LOCK = threading.Lock()
_LEGACY_CUDA_RESTORE_WARNING_EMITTED = False
_ASYNC_PAGE_CAPTURE_CONFIG = "spark_cache_async_page_capture"
_ASYNC_PAGE_CAPTURE_ENV = "SPARK_CONTEXT_CACHE_ASYNC_PAGE_CAPTURE"
_DEFAULT_SHARED_PREFIX_LEASE_TTL_SECONDS = 15.0
_MAX_SHARED_PREFIX_LEASE_TTL_SECONDS = 300.0


def _warn_legacy_cuda_restore_config() -> None:
    """Warn once when a process relies only on legacy configuration names."""

    global _LEGACY_CUDA_RESTORE_WARNING_EMITTED
    with _LEGACY_CUDA_RESTORE_WARNING_LOCK:
        if _LEGACY_CUDA_RESTORE_WARNING_EMITTED:
            return
        _LEGACY_CUDA_RESTORE_WARNING_EMITTED = True
    warnings.warn(
        "legacy SparkCache CUDA configuration names are deprecated; use the"
        " canonical SparkCache CUDA restore configuration names",
        FutureWarning,
        stacklevel=3,
    )


def _compat_config_value(
    extra: Callable[[str, Any], Any],
    *,
    canonical_key: str,
    legacy_key: str,
    canonical_env: str,
    legacy_env: str,
    default: Any,
    normalize: Callable[[Any], Any] = str,
) -> Any:
    """Resolve one canonical setting and its compatibility alias."""

    canonical_extra = extra(canonical_key, _MISSING)
    legacy_extra = extra(legacy_key, _MISSING)
    canonical_value = (
        canonical_extra
        if canonical_extra is not _MISSING
        else os.environ.get(canonical_env, _MISSING)
    )
    legacy_value = (
        legacy_extra
        if legacy_extra is not _MISSING
        else os.environ.get(legacy_env, _MISSING)
    )
    if canonical_value is not _MISSING and legacy_value is not _MISSING:
        try:
            disagree = normalize(canonical_value) != normalize(legacy_value)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "spark-context-cache: conflicting SparkCache CUDA restore"
                f" settings {canonical_key} and legacy alias {legacy_key}"
            ) from error
        if disagree:
            raise RuntimeError(
                "spark-context-cache: conflicting SparkCache CUDA restore"
                f" settings {canonical_key} and legacy alias {legacy_key}"
            )
        return canonical_value
    if canonical_value is not _MISSING:
        return canonical_value
    if legacy_value is not _MISSING:
        _warn_legacy_cuda_restore_config()
        return legacy_value
    return default


def _config_bool(value: Any) -> bool:
    if value in (1, "1", True, "true"):
        return True
    return False


def _access_controls(
    extra: Callable[[str, Any], Any],
) -> tuple[str, bool, bool]:
    """Resolve persistent-cache read and publication controls.

    ``spark_cache_access_mode`` provides an operator-readable baseline. The
    independent ``spark_cache_store`` and ``spark_cache_restore`` settings
    remain supported and override their respective baseline values. This
    preserves existing configurations while allowing a deployment to request
    restore-only operation explicitly.
    """

    mode_raw = extra(
        "spark_cache_access_mode",
        os.environ.get("SPARK_CONTEXT_CACHE_ACCESS_MODE", "read-write"),
    )
    mode = str(mode_raw).strip().lower()
    try:
        default_store, default_restore = _ACCESS_MODES[mode]
    except KeyError as error:
        choices = ", ".join(sorted(_ACCESS_MODES))
        raise RuntimeError(
            "spark-context-cache: spark_cache_access_mode must be one of "
            f"{choices}"
        ) from error

    store_raw = extra("spark_cache_store", _MISSING)
    if store_raw is _MISSING:
        store_raw = os.environ.get("SPARK_CONTEXT_CACHE_STORE", _MISSING)
    restore_raw = extra("spark_cache_restore", _MISSING)
    if restore_raw is _MISSING:
        restore_raw = os.environ.get("SPARK_CONTEXT_CACHE_RESTORE", _MISSING)
    store_enabled = (
        default_store if store_raw is _MISSING else _config_bool(store_raw)
    )
    restore_enabled = (
        default_restore if restore_raw is _MISSING else _config_bool(restore_raw)
    )
    effective_mode = next(
        name
        for name, controls in _ACCESS_MODES.items()
        if controls == (store_enabled, restore_enabled)
    )
    return effective_mode, store_enabled, restore_enabled


def _nonnegative_config_int(value: Any, label: str) -> int:
    try:
        if isinstance(value, (bool, float)):
            raise ValueError("Boolean and floating values are not integer limits")
        parsed = int(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"spark-context-cache: {label} must be an integer"
        ) from error
    if parsed < 0:
        raise RuntimeError(f"spark-context-cache: {label} must be non-negative")
    return parsed


def _bounded_positive_config_float(
    value: Any,
    label: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    """Parse a finite positive duration with an explicit upper bound."""

    try:
        if isinstance(value, bool):
            raise ValueError("Boolean values are not durations")
        parsed = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"spark-context-cache: {label} must be a finite number"
        ) from error
    if not math.isfinite(parsed):
        raise RuntimeError(
            f"spark-context-cache: {label} must be a finite number"
        )
    if not minimum <= parsed <= maximum:
        raise RuntimeError(
            f"spark-context-cache: {label} must be at least {minimum:g} and at most"
            f" {maximum:g} seconds"
        )
    return parsed


def _freeze_config_value(value: Any) -> Any:
    """Recursively freeze connector metadata shared across runtime roles."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_config_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config_value(item) for item in value)
    return value


def _cache_spec_reuse_policy(spec: Any) -> tuple[str, int | None]:
    """Classify one layer's supported persistent-cache reuse semantics."""

    class_names = {base.__name__ for base in type(spec).__mro__}
    if "SlidingWindowSpec" in class_names:
        window = int(getattr(spec, "sliding_window", 0) or 0)
        if window <= 1:
            raise RuntimeError(
                "spark-context-cache: sliding KV-cache window must exceed one token"
            )
        return "sliding", window
    if "FullAttentionSpec" in class_names:
        return "full", None
    if "MambaSpec" in class_names:
        mode = str(getattr(spec, "mamba_cache_mode", "none"))
        if mode != "align":
            raise RuntimeError(
                "spark-context-cache: recurrent block-page restore requires"
                f" mamba_cache_mode 'align' (configured: {mode!r})"
            )
        return "recurrent_align", None
    raise RuntimeError(
        "spark-context-cache: unsupported block-page KV-cache spec "
        f"{type(spec).__name__}"
    )


def _group_reuse_policy(
    spec: Any, layer_names: Sequence[str]
) -> tuple[str, int | None]:
    """Resolve one engine group's common supported reuse policy."""

    if not layer_names:
        raise RuntimeError("spark-context-cache: KV-cache group has no layers")
    per_layer = getattr(spec, "kv_cache_specs", None)
    policies = {
        _cache_spec_reuse_policy(
            per_layer[name]
            if isinstance(per_layer, dict) and name in per_layer
            else spec
        )
        for name in layer_names
    }
    if len(policies) != 1:
        rendered = sorted(
            policy if window is None else f"{policy}:{window}"
            for policy, window in policies
        )
        raise RuntimeError(
            "spark-context-cache: one KV-cache group mixes incompatible reuse"
            f" policies: {rendered}"
        )
    return policies.pop()


def _recurrent_state_identity(
    spec: Any, layer_names: Sequence[str]
) -> dict[str, Any] | None:
    """Return identity fields that affect Mamba-align checkpoint semantics."""

    per_layer = getattr(spec, "kv_cache_specs", None)
    layer_specs = tuple(
        per_layer[name] if isinstance(per_layer, dict) and name in per_layer else spec
        for name in layer_names
    )
    recurrent = tuple(
        layer_spec
        for layer_spec in layer_specs
        if "MambaSpec" in {base.__name__ for base in type(layer_spec).__mro__}
    )
    if not recurrent:
        return None
    fields = (
        "mamba_cache_mode",
        "tokens_per_state",
        "num_speculative_blocks",
        "num_prefill_checkpoint_blocks",
    )
    values = {
        field: {getattr(layer_spec, field, None) for layer_spec in recurrent}
        for field in fields
    }
    inconsistent = [field for field, choices in values.items() if len(choices) != 1]
    if inconsistent:
        raise RuntimeError(
            "spark-context-cache: recurrent group mixes incompatible state"
            " geometry: " + ", ".join(inconsistent)
        )
    identity = {field: next(iter(choices)) for field, choices in values.items()}
    # Omit vLLM defaults to preserve namespaces on earlier source lines.
    for field, default in (
        ("tp_replicated", False),
        ("prefill_checkpoint_alignment", None),
        ("boundary_capture", False),
    ):
        choices = {getattr(layer_spec, field, default) for layer_spec in recurrent}
        if len(choices) != 1:
            raise RuntimeError(
                "spark-context-cache: recurrent group mixes incompatible state"
                " geometry: " + field
            )
        value = choices.pop()
        if value != default:
            identity[field] = value
    return identity


def _group_identity_spec(spec: Any, layers: Sequence[str]) -> Any:
    """Use the per-layer representative retained by vLLM's scheduler.

    UniformTypeKVCacheSpecs exists only in the worker view: vLLM's
    generate_scheduler_kv_cache_config replaces it with the first value in
    kv_cache_specs, preserving insertion order. Its summed page size and
    inherited DCP default are therefore not shared identity facts. Physical
    per-layer sizes remain in the worker's registered PageLayout.

    Other spec types retain their wire identity, including aggregate specs
    used by qualified source lines.
    """
    if "UniformTypeKVCacheSpecs" not in {base.__name__ for base in type(spec).__mro__}:
        return spec
    members = getattr(spec, "kv_cache_specs", None)
    if not isinstance(members, dict) or not members or set(members) != set(layers):
        raise RuntimeError("spark-context-cache: uniform KV-cache group needs every layer spec")
    representative = next(iter(members.values()))
    # Byte sizes may differ (for example FP8 target and BF16 MTP); manager
    # token geometry and DCP ownership must agree across all members.
    def geometry(member: Any) -> tuple[Any, ...]:
        policy, _ = _cache_spec_reuse_policy(member)
        return (type(member).__name__, getattr(member, "block_size", 0),
                getattr(member, "storage_block_size", 0),
                bool(getattr(member, "dcp_replicated", policy == "recurrent_align")))

    if any(geometry(member) != geometry(representative) for member in members.values()):
        raise RuntimeError("spark-context-cache: uniform KV-cache group mixes incompatible manager geometry")
    return representative


def kv_group_topology(
    kv_cache_config: Any,
    *,
    dcp_degree: int = 1,
) -> tuple[dict[str, Any], ...]:
    """Describe each manager group, including its DCP page ownership."""

    if dcp_degree <= 0:
        raise RuntimeError("spark-context-cache: DCP degree must be positive")
    groups = tuple(getattr(kv_cache_config, "kv_cache_groups", ()) or ())
    topology = []
    for group_index, group in enumerate(groups):
        # Ephemeral replicas and host pools are not device pages we can restore.
        role = getattr(group, "role", "default")
        role = getattr(role, "value", role)
        if (not getattr(group, "enable_kv_transfer", True)
                or role != "default" or getattr(group, "host_resident", False)):
            raise RuntimeError(
                "spark-context-cache: unsupported non-transferable KV-cache group"
                f" {group_index} (role={role!r})"
            )
        spec = getattr(group, "kv_cache_spec", None)
        layers = tuple(sorted(getattr(group, "layer_names", ()) or ()))
        identity_spec = _group_identity_spec(spec, layers)
        reuse_policy, reuse_window_tokens = _group_reuse_policy(spec, layers)
        recurrent_state = _recurrent_state_identity(spec, layers)
        spec = identity_spec
        dcp_replicated = bool(
            getattr(spec, "dcp_replicated", reuse_policy == "recurrent_align")
        )
        dcp_shard_count = 1 if dcp_replicated else dcp_degree
        block_size = int(getattr(spec, "block_size", 0) or 0)
        if block_size <= 0:
            raise RuntimeError(
                "spark-context-cache: KV-cache group block size must be positive"
            )
        group_identity = {
            "group": group_index,
            "spec": type(spec).__name__,
            "block_size": block_size,
            "storage_block_size": int(getattr(spec, "storage_block_size", 0) or 0),
            "page_size_bytes": int(getattr(spec, "page_size_bytes", 0) or 0),
            "dcp_replicated": dcp_replicated,
            "dcp_shard_count": dcp_shard_count,
            "logical_tokens_per_block": block_size * dcp_shard_count,
            "reuse_policy": reuse_policy,
            "reuse_window_tokens": reuse_window_tokens,
            "eagle": bool(getattr(group, "is_eagle_group", False)),
            "layers": layers,
        }
        if recurrent_state is not None:
            group_identity["recurrent_state"] = recurrent_state
        topology.append(group_identity)
    return tuple(topology)


def kv_group_topology_digest(kv_cache_config: Any, *, dcp_degree: int = 1) -> str:
    encoded = json.dumps(
        kv_group_topology(kv_cache_config, dcp_degree=dcp_degree),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ConnectorConfig:
    """Immutable result of parsing all SparkCache connector settings.

    Every core connector field is determined solely by the vLLM config,
    extra-config keys, environment variables, and model profile supplied at
    construction. The optional streaming factory owns its C++/CUDA artifact,
    vLLM-root, lease-contract, and timing settings and validates them while the
    connector is still starting. Both scheduler and worker roles receive the
    same :class:`ConnectorConfig` for the same deployment, so cache-identity
    wire bytes are identical.
    """

    tp_degree: int
    dcp_degree: int
    cp_kv_cache_interleave_size: int
    block_size: int
    profile: Any
    storage_mode: str
    publication_schema: str
    group_topology: tuple[Mapping[str, Any], ...]
    chunk_tokens: int
    root: str
    clear_once_token: str
    capacity_policy: CapacityPolicy
    min_span: int
    max_span: int
    access_mode: str
    store_enabled: bool
    restore_enabled: bool
    streaming_snapshots_enabled: bool
    async_page_capture_enabled: bool
    page_snapshot_interval_tokens: int
    page_delta_stage_limit_count: int
    hot_page_snapshot_bytes: int
    payload_integrity: str
    cuda_restore_enabled: bool
    cuda_placement_library_path: str
    cuda_placement_library_sha256: str
    cuda_placement_arena_bytes: int
    cuda_restore_arena_budget_bytes: int
    cuda_restore_io_workers: int
    scheduler_probe: str
    identity_base: Mapping[str, Any]
    load_thread_limit: int
    max_pending_restores: int
    max_delayed_stores: int
    shared_prefix_lease_ttl_seconds: float

    @property
    def native_restore_enabled(self) -> bool:
        """Compatibility alias for :attr:`cuda_restore_enabled`."""

        return self.cuda_restore_enabled

    @property
    def native_library_path(self) -> str:
        """Compatibility alias for the CUDA placement library path."""

        return self.cuda_placement_library_path

    @property
    def native_library_sha256(self) -> str:
        """Compatibility alias for the CUDA placement library digest."""

        return self.cuda_placement_library_sha256

    @property
    def native_arena_bytes(self) -> int:
        """Compatibility alias for the CUDA placement arena size."""

        return self.cuda_placement_arena_bytes

    @property
    def native_io_workers(self) -> int:
        """Compatibility alias for the CUDA restore I/O worker count."""

        return self.cuda_restore_io_workers

    def build_identity(self, shard_rank: int, tp_shard_rank: int) -> CacheIdentity:
        """Construct a :class:`CacheIdentity` for a DCP shard rank.

        The connector resolves scheduler-versus-worker physical rank ownership
        before calling this interface. Both shard ranks are therefore explicit
        and no unowned storage namespace can be produced.
        """
        return CacheIdentity(
            dcp_shard_rank=shard_rank,
            tp_shard_rank=tp_shard_rank,
            **self.identity_base,
        )


def parse_connector_config(
    vllm_config: Any,
    kv_transfer_config: Any,
    kv_cache_config: Any,
) -> ConnectorConfig:
    """Parse and validate all SparkCache settings at construction time.

    Explicit validation failures raise :class:`RuntimeError` with the
    ``spark-context-cache:`` prefix. Direct integer conversion for span and
    load-thread settings retains Python's :class:`ValueError` behavior.
    """

    block_size = vllm_config.cache_config.block_size
    parallel = vllm_config.parallel_config
    tp_degree = max(1, getattr(parallel, "tensor_parallel_size", 1))
    dcp_degree = max(1, getattr(parallel, "decode_context_parallel_size", 1))
    cp_kv_cache_interleave_size = getattr(
        parallel,
        "cp_kv_cache_interleave_size",
        getattr(parallel, "dcp_kv_cache_interleave_size", 1),
    )
    if (
        type(cp_kv_cache_interleave_size) is not int
        or cp_kv_cache_interleave_size <= 0
    ):
        raise RuntimeError(
            "spark-context-cache: cp_kv_cache_interleave_size must be a"
            " positive integer"
        )
    if tp_degree % dcp_degree:
        raise RuntimeError(
            "spark-context-cache: decode context parallel size must divide"
            f" tensor parallel size (tp={tp_degree},"
            f" dcp={dcp_degree})"
        )
    if (
        block_size < cp_kv_cache_interleave_size
        or block_size % cp_kv_cache_interleave_size
    ):
        raise RuntimeError(
            "spark-context-cache: cache block size must be at least and"
            " divisible by cp_kv_cache_interleave_size"
        )
    pp_degree = max(1, getattr(parallel, "pipeline_parallel_size", 1))
    if pp_degree > 1:
        # CacheIdentity has no pipeline-stage field. Two stages would otherwise
        # publish different tensors into one manifest namespace.
        raise RuntimeError(
            "spark-context-cache: pipeline parallelism is unsupported;"
            f" pipeline_parallel_size={pp_degree}"
        )
    extra = kv_transfer_config.get_from_extra_config
    profile_value = extra(
        "spark_cache_model_profile",
        os.environ.get("SPARK_CONTEXT_CACHE_MODEL_PROFILE"),
    )
    if profile_value is None or not str(profile_value).strip():
        raise RuntimeError(
            "spark-context-cache: spark_cache_model_profile is required; "
            "select a registered deployment profile explicitly"
        )
    profile_name = str(profile_value).strip()
    try:
        profile = resolve_profile(profile_name)
    except ProfileError as error:
        raise RuntimeError(f"spark-context-cache: {error}") from error
    storage_mode = profile.storage_mode
    publication_schema_raw = str(
        extra(
            "spark_cache_publication_schema",
            os.environ.get(
                "SPARK_CONTEXT_CACHE_PUBLICATION_SCHEMA",
                "snapshot-v1",
            ),
        )
    )
    if publication_schema_raw not in (
        "snapshot-v1",
        "tail-cow-v1",
        "tail-cow-v2",
    ):
        raise RuntimeError(
            "spark-context-cache: spark_cache_publication_schema must be"
            " 'snapshot-v1', 'tail-cow-v1', or 'tail-cow-v2'"
        )
    publication_schema = ""
    if publication_schema_raw == "tail-cow-v1":
        publication_schema = (
            "page-tail-cow-v1" if storage_mode == "block_pages_v1" else "tail-cow-v1"
        )
    elif publication_schema_raw == "tail-cow-v2":
        if storage_mode != "block_pages_v1":
            raise RuntimeError(
                "spark-context-cache: tail-cow-v2 requires block-page storage"
            )
        publication_schema = "page-tail-cow-v2"
    # Payload integrity. ``fast`` (the default for tail-cow-v2 publication)
    # checks payload objects and layer tails with CRC32 and selects the
    # page-tail-cow-v3-fast schema; ``none`` skips payload checks; ``sha256``
    # keeps the content-addressed format, the default for every other schema.
    # The v3 schemas have their own cache identity, so readers of one format
    # miss entries of another cleanly.
    payload_integrity = str(
        extra(
            "spark_cache_payload_integrity",
            os.environ.get(
                "SPARK_CONTEXT_CACHE_PAYLOAD_INTEGRITY",
                "fast" if publication_schema == "page-tail-cow-v2" else "sha256",
            ),
        )
    )
    if payload_integrity not in ("sha256", "fast", "none"):
        raise RuntimeError(
            "spark-context-cache: spark_cache_payload_integrity must be"
            " 'sha256', 'fast', or 'none'"
        )
    if payload_integrity != "sha256":
        if publication_schema != "page-tail-cow-v2":
            raise RuntimeError(
                "spark-context-cache: spark_cache_payload_integrity fast or none"
                " requires spark_cache_publication_schema tail-cow-v2"
            )
        publication_schema = f"page-tail-cow-v3-{payload_integrity}"
    group_topology = kv_group_topology(kv_cache_config, dcp_degree=dcp_degree)
    if storage_mode == "block_pages_v1" and not group_topology:
        raise RuntimeError(
            "spark-context-cache: block-page storage requires KV-cache groups"
        )
    chunk_tokens = profile.chunk_tokens
    if storage_mode == "per_token_rows" and cp_kv_cache_interleave_size != 1:
        raise RuntimeError(
            "spark-context-cache: per-token row storage supports only"
            " cp_kv_cache_interleave_size=1"
        )
    load_policy = str(
        getattr(kv_transfer_config, "kv_load_failure_policy", "recompute")
    )
    if load_policy != "recompute":
        # Every restore failure must become a clean re-prefill. Any other vLLM
        # policy can surface a failed restore as a request error.
        raise RuntimeError(
            "spark-context-cache: kv_load_failure_policy must be"
            f" 'recompute' (configured: {load_policy!r})"
        )
    root = str(
        extra(
            "spark_cache_root",
            os.environ.get("SPARK_CONTEXT_CACHE_ROOT", "/cache/context"),
        )
    )
    clear_once_raw = extra("spark_cache_clear_once", "")
    if not isinstance(clear_once_raw, str):
        raise RuntimeError(
            "spark-context-cache: spark_cache_clear_once must be a string"
        )
    clear_once_token = clear_once_raw
    if clear_once_token:
        try:
            validated_root, _token_digest = validate_clear_once_request(
                root,
                clear_once_token,
            )
        except ValueError as error:
            raise RuntimeError(f"spark-context-cache: {error}") from error
        root = str(validated_root)
    max_bytes = _nonnegative_config_int(
        extra(
            "spark_cache_max_bytes",
            os.environ.get("SPARK_CONTEXT_CACHE_MAX_BYTES", "0"),
        ),
        "spark_cache_max_bytes",
    )
    default_low_bytes = max_bytes * 9 // 10 if max_bytes else 0
    low_watermark_bytes = _nonnegative_config_int(
        extra(
            "spark_cache_low_watermark_bytes",
            os.environ.get(
                "SPARK_CONTEXT_CACHE_LOW_WATERMARK_BYTES",
                str(default_low_bytes),
            ),
        ),
        "spark_cache_low_watermark_bytes",
    )
    ttl_seconds = _nonnegative_config_int(
        extra(
            "spark_cache_ttl_seconds",
            os.environ.get("SPARK_CONTEXT_CACHE_TTL_SECONDS", "0"),
        ),
        "spark_cache_ttl_seconds",
    )
    try:
        capacity_policy = CapacityPolicy(
            max_bytes=max_bytes,
            low_watermark_bytes=low_watermark_bytes,
            ttl_seconds=ttl_seconds,
            maintenance_max_deletions=_nonnegative_config_int(
                extra("spark_cache_maintenance_max_deletions", os.environ.get(
                    "SPARK_CONTEXT_CACHE_MAINTENANCE_MAX_DELETIONS", "0")),
                "spark_cache_maintenance_max_deletions",
            ),
            maintenance_interval_ms=_nonnegative_config_int(
                extra("spark_cache_maintenance_interval_ms", os.environ.get(
                    "SPARK_CONTEXT_CACHE_MAINTENANCE_INTERVAL_MS", "0")),
                "spark_cache_maintenance_interval_ms",
            ),
        )
    except ValueError as error:
        raise RuntimeError(f"spark-context-cache: {error}") from error
    min_span = int(
        extra(
            "spark_cache_min_span_tokens",
            os.environ.get("SPARK_CONTEXT_CACHE_MIN_SPAN", "1024"),
        )
    )
    page_snapshot_interval_tokens = _nonnegative_config_int(
        extra(
            "spark_cache_page_snapshot_interval_tokens",
            os.environ.get("SPARK_CONTEXT_CACHE_PAGE_SNAPSHOT_INTERVAL_TOKENS", "0"),
        ),
        "spark_cache_page_snapshot_interval_tokens",
    )
    page_delta_stage_limit_count = _nonnegative_config_int(
        extra(
            "spark_cache_page_delta_stage_limit_count",
            os.environ.get("SPARK_CONTEXT_CACHE_PAGE_DELTA_STAGE_LIMIT_COUNT", "0"),
        ),
        "spark_cache_page_delta_stage_limit_count",
    )
    # Process-local budget for complete page snapshots this worker has hashed
    # (research-only; zero disables). A hit places or republishes a boundary
    # without any object read. The memory is host memory of the worker
    # process, which on unified-memory hosts competes with the GPU.
    hot_page_snapshot_bytes = _nonnegative_config_int(
        extra(
            "spark_cache_hot_snapshot_bytes",
            os.environ.get("SPARK_CONTEXT_CACHE_HOT_SNAPSHOT_BYTES", "0"),
        ),
        "spark_cache_hot_snapshot_bytes",
    )
    model_max = int(getattr(vllm_config.model_config, "max_model_len", 0) or 0)
    default_max_span = str(model_max if model_max > 0 else 1 << 30)
    max_span = int(
        extra(
            "spark_cache_max_span_tokens",
            os.environ.get("SPARK_CONTEXT_CACHE_MAX_SPAN", default_max_span),
        )
    )
    access_mode, store_enabled, restore_enabled = _access_controls(extra)
    try:
        streaming_snapshots_requested = _streaming_snapshots_enabled(
            extra(
                _STREAMING_SNAPSHOTS_CONFIG,
                os.environ.get(_STREAMING_SNAPSHOTS_ENV, "0"),
            )
        )
    except ValueError as error:
        raise RuntimeError(f"spark-context-cache: {error}") from error
    streaming_snapshots_enabled = streaming_snapshots_requested and store_enabled
    try:
        async_page_capture_enabled = _streaming_snapshots_enabled(
            extra(
                _ASYNC_PAGE_CAPTURE_CONFIG,
                os.environ.get(_ASYNC_PAGE_CAPTURE_ENV, "0"),
            )
        )
    except ValueError as error:
        raise RuntimeError(
            "spark-context-cache: spark_cache_async_page_capture must be"
            " 0/1 or false/true"
        ) from error
    if async_page_capture_enabled and storage_mode != "block_pages_v1":
        raise RuntimeError(
            "spark-context-cache: asynchronous manager-page capture requires"
            " block-page storage"
        )
    if async_page_capture_enabled and not store_enabled:
        raise RuntimeError(
            "spark-context-cache: asynchronous manager-page capture requires"
            " cache publication"
        )
    if async_page_capture_enabled and streaming_snapshots_enabled:
        raise RuntimeError(
            "spark-context-cache: asynchronous manager-page capture and"
            " row streaming snapshots are mutually exclusive"
        )
    if storage_mode == "block_pages_v1" and streaming_snapshots_enabled:
        raise RuntimeError(
            "spark-context-cache: block-page storage does not support"
            " streaming snapshots"
        )
    if publication_schema and streaming_snapshots_enabled:
        raise RuntimeError(
            "spark-context-cache: tail-cow-v1 publication does not support"
            " streaming snapshots"
        )
    cuda_restore_requested = _config_bool(
        _compat_config_value(
            extra,
            canonical_key="spark_cache_cuda_restore",
            legacy_key="spark_cache_native_restore",
            canonical_env="SPARK_CONTEXT_CACHE_CUDA_RESTORE",
            legacy_env="SPARK_CONTEXT_CACHE_NATIVE_RESTORE",
            default="0",
            normalize=_config_bool,
        )
    )
    cuda_restore_enabled = cuda_restore_requested and restore_enabled
    cuda_placement_library_path = str(
        _compat_config_value(
            extra,
            canonical_key="spark_cache_cuda_placement_library",
            legacy_key="spark_cache_native_library",
            canonical_env="SPARK_CONTEXT_CACHE_CUDA_PLACEMENT_LIBRARY",
            legacy_env="SPARK_CONTEXT_CACHE_NATIVE_LIBRARY",
            default="",
        )
        or ""
    )
    cuda_placement_library_sha256 = str(
        _compat_config_value(
            extra,
            canonical_key="spark_cache_cuda_placement_library_sha256",
            legacy_key="spark_cache_native_library_sha256",
            canonical_env="SPARK_CONTEXT_CACHE_CUDA_PLACEMENT_LIBRARY_SHA256",
            legacy_env="SPARK_CONTEXT_CACHE_NATIVE_LIBRARY_SHA256",
            default="",
        )
        or ""
    )
    cuda_placement_arena_raw = str(
        _compat_config_value(
            extra,
            canonical_key="spark_cache_cuda_placement_arena_bytes",
            legacy_key="spark_cache_native_arena_bytes",
            canonical_env="SPARK_CONTEXT_CACHE_CUDA_PLACEMENT_ARENA_BYTES",
            legacy_env="SPARK_CONTEXT_CACHE_NATIVE_ARENA_BYTES",
            default="",
            normalize=int,
        )
        or ""
    )
    try:
        cuda_placement_arena_bytes = (
            int(cuda_placement_arena_raw) if cuda_placement_arena_raw else 0
        )
    except ValueError as error:
        if cuda_restore_enabled:
            raise RuntimeError(
                "spark-context-cache: SparkCache CUDA restore requires an integer"
                " placement arena size"
            ) from error
        cuda_placement_arena_bytes = 0
    cuda_restore_workers_raw = _compat_config_value(
        extra,
        canonical_key="spark_cache_cuda_restore_io_workers",
        legacy_key="spark_cache_native_io_workers",
        canonical_env="SPARK_CONTEXT_CACHE_CUDA_RESTORE_IO_WORKERS",
        legacy_env="SPARK_CONTEXT_CACHE_NATIVE_IO_WORKERS",
        default="8",
        normalize=int,
    )
    try:
        cuda_restore_io_workers = int(cuda_restore_workers_raw)
    except (TypeError, ValueError) as error:
        if cuda_restore_enabled:
            raise RuntimeError(
                "spark-context-cache: SparkCache CUDA restore requires an integer"
                " IO worker count"
            ) from error
        cuda_restore_io_workers = 8
    if cuda_restore_enabled:
        library_path = Path(cuda_placement_library_path)
        if (
            not cuda_placement_library_path
            or not library_path.is_absolute()
            or SHA256_RE.fullmatch(cuda_placement_library_sha256) is None
            or cuda_placement_arena_bytes not in _CUDA_PLACEMENT_ARENA_BYTES
        ):
            raise RuntimeError(
                "spark-context-cache: SparkCache CUDA restore requires an"
                " absolute CUDA placement library path, a 64-character"
                " lowercase SHA-256, and placement arena bytes equal to"
                " 64, 128, or 256 MiB"
            )
        if not 1 <= cuda_restore_io_workers <= 32:
            raise RuntimeError(
                "spark-context-cache: SparkCache CUDA restore IO workers"
                " must be in [1, 32]"
            )
    draft_policy = extra(
        "spark_cache_draft_policy",
        os.environ.get(
            "SPARK_CONTEXT_CACHE_DRAFT_POLICY",
            profile.default_draft_kv_policy,
        ),
    )
    target_id = str(
        extra(
            "spark_cache_target_checkpoint_sha256",
            os.environ.get(
                "SPARK_CONTEXT_CACHE_TARGET_CHECKPOINT_SHA256",
                "",
            ),
        )
        or ""
    )
    if SHA256_RE.fullmatch(target_id) is None:
        raise RuntimeError(
            "spark-context-cache: target checkpoint identity must be a"
            " 64-character lowercase SHA-256; set"
            " spark_cache_target_checkpoint_sha256"
        )
    draft_id = str(
        extra(
            "spark_cache_draft_checkpoint_sha256",
            os.environ.get(
                "SPARK_CONTEXT_CACHE_DRAFT_CHECKPOINT_SHA256",
                "",
            ),
        )
        or ""
    )
    if str(draft_policy) == "colocated_target":
        if draft_id and draft_id != target_id:
            raise RuntimeError(
                "spark-context-cache: colocated_target draft state must"
                " use the target checkpoint identity; omit"
                " spark_cache_draft_checkpoint_sha256"
            )
        draft_id = target_id
    elif SHA256_RE.fullmatch(draft_id) is None:
        raise RuntimeError(
            "spark-context-cache: separate draft checkpoint identity must"
            " be a 64-character lowercase SHA-256; set"
            " spark_cache_draft_checkpoint_sha256"
        )
    quantization_layout = profile.quantization_layout
    if storage_mode == "block_pages_v1":
        # A cache-manager block ID names one complete physical page, including
        # split kernel rows and opaque bytes beyond the logical tensor shape.
        quantization_layout += ":manager-pages-v2:" + kv_group_topology_digest(
            kv_cache_config,
            dcp_degree=dcp_degree,
        )
    record_schema = (
        ("target_ckv", "logical_positions") if storage_mode == "block_pages_v1" else ()
    )
    identity_base: dict[str, Any] = dict(
        target_checkpoint=target_id,
        draft_checkpoint=draft_id,
        quantization_layout=quantization_layout,
        rope_layout=profile.rope_layout,
        tp_degree=tp_degree,
        dcp_degree=dcp_degree,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        chunk_tokens=chunk_tokens,
        boundary_hidden_policy=profile.boundary_hidden_policy,
        draft_kv_policy=str(draft_policy),
    )
    if record_schema:
        identity_base["record_schema"] = record_schema
    if publication_schema:
        identity_base["publication_schema"] = publication_schema
    try:
        profile.validate_for_deployment(
            dcp_degree=dcp_degree,
            block_size=block_size,
            min_span_tokens=min_span,
            cuda_restore=cuda_restore_enabled,
        )
    except ProfileError as error:
        raise RuntimeError(f"spark-context-cache: {error}") from error
    # "tp0" probes rank 0's local manifest store and therefore requires the
    # scheduler to share that worker's cache root. "none" admits on worker
    # quorum alone when the scheduler has no rank-local cache filesystem.
    scheduler_probe = str(
        extra(
            "spark_cache_scheduler_probe",
            os.environ.get("SPARK_CONTEXT_CACHE_SCHEDULER_PROBE", "tp0"),
        )
    )
    if scheduler_probe not in ("tp0", "none"):
        raise RuntimeError(
            "spark-context-cache: spark_cache_scheduler_probe must be"
            f" 'tp0' or 'none' (configured: {scheduler_probe!r})"
        )
    # Each CUDA placement lane owns two arenas. Limit page concurrency by both
    # the requested threads and an optional rank-wide arena allocation budget.
    load_thread_limit = min(
        8,
        max(
            1,
            int(
                extra(
                    "spark_cache_load_threads",
                    os.environ.get("SPARK_CONTEXT_CACHE_LOAD_THREADS", "1"),
                )
            ),
        ),
    )
    if cuda_restore_enabled and storage_mode != "block_pages_v1":
        load_thread_limit = 1
    cuda_restore_arena_budget_bytes = _nonnegative_config_int(
        extra(
            "spark_cache_cuda_restore_arena_budget_bytes",
            os.environ.get("SPARK_CONTEXT_CACHE_CUDA_RESTORE_ARENA_BUDGET_BYTES", "0"),
        ),
        "spark_cache_cuda_restore_arena_budget_bytes",
    )
    if cuda_restore_enabled and cuda_restore_arena_budget_bytes:
        from sparkcache.spark_cache_cuda import ARENA_COUNT

        lane_bytes = ARENA_COUNT * cuda_placement_arena_bytes
        if cuda_restore_arena_budget_bytes < lane_bytes:
            raise RuntimeError(
                "spark-context-cache: spark_cache_cuda_restore_arena_budget_bytes"
                f" must be at least {lane_bytes} bytes for one lane's two arenas"
            )
        load_thread_limit = min(
            load_thread_limit, cuda_restore_arena_budget_bytes // lane_bytes
        )
    max_pending_restores_raw = extra(
        "spark_cache_max_pending_restores",
        os.environ.get("SPARK_CONTEXT_CACHE_MAX_PENDING_RESTORES", "64"),
    )
    try:
        if isinstance(max_pending_restores_raw, bool):
            raise ValueError("Boolean values are not integer limits")
        max_pending_restores = int(max_pending_restores_raw)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "spark-context-cache: spark_cache_max_pending_restores must be an integer"
        ) from error
    if max_pending_restores < 1:
        raise RuntimeError(
            "spark-context-cache: spark_cache_max_pending_restores must be at least 1"
        )
    max_delayed_stores = _nonnegative_config_int(
        extra(
            "spark_cache_max_delayed_stores",
            os.environ.get("SPARK_CONTEXT_CACHE_MAX_DELAYED_STORES", "16"),
        ),
        "spark_cache_max_delayed_stores",
    )
    if max_delayed_stores < 1:
        raise RuntimeError(
            "spark-context-cache: spark_cache_max_delayed_stores must be at least 1"
        )
    shared_prefix_lease_ttl_seconds = _bounded_positive_config_float(
        extra(
            "spark_cache_shared_prefix_lease_ttl_seconds",
            os.environ.get(
                "SPARK_CONTEXT_CACHE_SHARED_PREFIX_LEASE_TTL_SECONDS",
                str(_DEFAULT_SHARED_PREFIX_LEASE_TTL_SECONDS),
            ),
        ),
        "spark_cache_shared_prefix_lease_ttl_seconds",
        minimum=1.0,
        maximum=_MAX_SHARED_PREFIX_LEASE_TTL_SECONDS,
    )
    return ConnectorConfig(
        tp_degree=tp_degree,
        dcp_degree=dcp_degree,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        block_size=block_size,
        profile=profile,
        storage_mode=storage_mode,
        publication_schema=publication_schema,
        group_topology=tuple(_freeze_config_value(group) for group in group_topology),
        chunk_tokens=chunk_tokens,
        root=root,
        clear_once_token=clear_once_token,
        capacity_policy=capacity_policy,
        min_span=min_span,
        max_span=max_span,
        access_mode=access_mode,
        store_enabled=store_enabled,
        restore_enabled=restore_enabled,
        streaming_snapshots_enabled=streaming_snapshots_enabled,
        async_page_capture_enabled=async_page_capture_enabled,
        page_snapshot_interval_tokens=page_snapshot_interval_tokens,
        page_delta_stage_limit_count=page_delta_stage_limit_count,
        hot_page_snapshot_bytes=hot_page_snapshot_bytes,
        payload_integrity=payload_integrity,
        cuda_restore_enabled=cuda_restore_enabled,
        cuda_placement_library_path=cuda_placement_library_path,
        cuda_placement_library_sha256=cuda_placement_library_sha256,
        cuda_placement_arena_bytes=cuda_placement_arena_bytes,
        cuda_restore_arena_budget_bytes=cuda_restore_arena_budget_bytes,
        cuda_restore_io_workers=cuda_restore_io_workers,
        scheduler_probe=scheduler_probe,
        identity_base=_freeze_config_value(identity_base),
        load_thread_limit=load_thread_limit,
        max_pending_restores=max_pending_restores,
        max_delayed_stores=max_delayed_stores,
        shared_prefix_lease_ttl_seconds=shared_prefix_lease_ttl_seconds,
    )
