"""Fail-closed feature gate for the production snapshot-ring path.

The ordinary SparkCache store remains the default publication path. Keeping
the gate separate from the CUDA binding makes the default path safe to import
in CPU-only scheduler processes. Explicit opt-in is assembled by the builtin
production factory and fails closed unless its native library and pinned vLLM
lease contract attest exactly.
"""

from __future__ import annotations

from typing import Any


EXTRA_CONFIG_KEY = "spark_cache_streaming_snapshots"
ENVIRONMENT_KEY = "SPARK_CONTEXT_CACHE_STREAMING_SNAPSHOTS"

_TRUE = frozenset((True, 1, "1", "true", "yes", "on"))
_FALSE = frozenset((False, 0, "0", "false", "no", "off", ""))


class StreamingSnapshotsUnavailable(RuntimeError):
    """The feature was explicitly requested before its safety gates exist."""


def is_enabled(value: Any) -> bool:
    """Parse the narrow, explicit flag vocabulary used by connector config."""

    if value is None:
        return False
    if isinstance(value, str):
        value = value.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(
        f"{EXTRA_CONFIG_KEY} must be one of 0/1 or false/true, got {value!r}"
    )


def require_live_integration() -> None:
    """Legacy guard for callers that bypass the connector factory.

    The connector no longer calls this function: it lazy-installs the
    production role factory on explicit enable. An older embedding that calls
    this guard directly still fails closed instead of silently taking the
    synchronous store path.
    """

    raise StreamingSnapshotsUnavailable(
        "spark-context-cache: direct streaming enable is unsupported; "
        "install the production connector runtime factory or leave "
        f"{EXTRA_CONFIG_KEY}=0"
    )
