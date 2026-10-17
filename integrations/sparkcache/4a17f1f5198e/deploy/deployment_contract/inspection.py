"""Model-neutral parsing of Docker inspection fields."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .errors import DeploymentContractError


def read_single_inspection(
    path: Path,
    *,
    error_type: type[ValueError] = DeploymentContractError,
) -> dict[str, Any]:
    """Read one Docker inspection object, accepting optional UTF-8 BOM bytes."""

    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(document, list):
        if len(document) != 1:
            raise error_type("inspect file must contain one container")
        document = document[0]
    if not isinstance(document, dict):
        raise error_type("inspect file must contain one object")
    return document


def environment_map(
    values: Iterable[str],
    *,
    require_unique: bool = True,
    error_type: type[ValueError] = DeploymentContractError,
) -> dict[str, str]:
    """Parse Docker ``Config.Env`` entries into a mutable mapping.

    Profile adapters choose whether duplicate names are rejected or retain
    Docker's last-value behavior.
    """

    result: dict[str, str] = {}
    for value in values:
        name, separator, setting = value.partition("=")
        invalid = not separator or not name
        duplicate = name in result
        if invalid or (require_unique and duplicate):
            qualifier = "unique " if require_unique else ""
            raise error_type(
                f"docker inspect Config.Env must contain {qualifier}NAME=VALUE entries"
            )
        result[name] = setting
    return result
