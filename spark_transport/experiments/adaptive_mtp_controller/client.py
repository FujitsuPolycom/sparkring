"""Frontend-local helpers for the existing EngineCore utility RPC."""

from __future__ import annotations

from typing import Any


_METHOD = "spark_adaptive_mtp_control"
_ACTIONS = frozenset(("status", "reset"))


def _validate_action(action: str) -> None:
    if action not in _ACTIONS:
        raise ValueError(f"adaptive-MTP action must be one of {sorted(_ACTIONS)}")


def control_sync(engine_or_client: Any, action: str) -> dict[str, Any]:
    """Call the scheduler-local control method through a sync core client."""
    _validate_action(action)
    client = getattr(engine_or_client, "engine_core", engine_or_client)
    call = getattr(client, "call_utility", None)
    if not callable(call):
        raise RuntimeError("object has no synchronous EngineCore utility RPC")
    result = call(_METHOD, action)
    if not isinstance(result, dict):
        raise RuntimeError("adaptive-MTP utility RPC returned a non-dict result")
    return result


async def control_async(engine_or_client: Any, action: str) -> dict[str, Any]:
    """Call the scheduler-local control method through an async core client."""
    _validate_action(action)
    client = getattr(engine_or_client, "engine_core", engine_or_client)
    call = getattr(client, "call_utility_async", None)
    if not callable(call):
        raise RuntimeError("object has no asynchronous EngineCore utility RPC")
    result = await call(_METHOD, action)
    if not isinstance(result, dict):
        raise RuntimeError("adaptive-MTP utility RPC returned a non-dict result")
    return result

