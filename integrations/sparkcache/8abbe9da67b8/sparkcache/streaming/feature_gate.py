"""Verified opt-in requirements for model-serving snapshot rings.

The ordinary SparkCache store remains the default publication path. Keeping
the opt-in check separate from the CUDA binding keeps CPU-only scheduler
imports free of CUDA side effects. Explicit opt-in is assembled by the builtin
model-serving factory and is rejected unless its C++/CUDA library and pinned
vLLM lease contract attest exactly.
"""

from __future__ import annotations

from typing import Any


EXTRA_CONFIG_KEY = "spark_cache_streaming_snapshots"
ENVIRONMENT_KEY = "SPARK_CONTEXT_CACHE_STREAMING_SNAPSHOTS"

_TRUE = frozenset((True, 1, "1", "true", "yes", "on"))
_FALSE = frozenset((False, 0, "0", "false", "no", "off", ""))


class StreamingSnapshotsUnavailable(RuntimeError):
    """The feature was requested before its safety requirements were met."""


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
    """Reject callers that bypass the connector factory.

    The connector installs the model-serving role factory when streaming is
    explicitly enabled. An embedding that calls this guard directly has not
    supplied that factory and therefore cannot enter the synchronous store
    path as a substitute.
    """

    raise StreamingSnapshotsUnavailable(
        "spark-context-cache: direct streaming enable is unsupported; "
        "install the model-serving connector runtime factory or leave "
        f"{EXTRA_CONFIG_KEY}=0"
    )
