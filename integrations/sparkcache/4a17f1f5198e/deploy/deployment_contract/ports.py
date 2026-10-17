"""Port validation shared by deployment profile adapters."""

from __future__ import annotations

from .errors import DeploymentContractError


def validate_port(
    value: int | None,
    role: str,
    *,
    error_type: type[ValueError] = DeploymentContractError,
) -> int | None:
    """Return a TCP/UDP port after validating its numeric range."""

    if value is not None and not 1 <= value <= 65535:
        raise error_type(f"{role} must be in [1, 65535]")
    return value
