"""Model-specific adapters for generic SparkCache mechanisms."""

from __future__ import annotations

from typing import Any


def resolve_streaming_profile_adapter(profile_name: str) -> Any:
    """Return the streaming adapter registered for one explicit profile."""

    if profile_name == "glm52-nvfp4":
        from .glm52_streaming import GLM52_STREAMING_ADAPTER

        return GLM52_STREAMING_ADAPTER
    raise RuntimeError(
        "streaming snapshots are not implemented for model profile "
        f"{profile_name!r}"
    )


__all__ = ["resolve_streaming_profile_adapter"]
