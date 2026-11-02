"""Deterministic parsing and rewriting of vLLM command arguments."""

from __future__ import annotations

import json
from typing import Any

from .errors import DeploymentContractError


def compact_json(value: Any) -> str:
    """Encode a command-line JSON value with deterministic byte content."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def option_values(
    arguments: list[str],
    option: str,
    *,
    error_type: type[ValueError] = DeploymentContractError,
) -> list[str]:
    """Return values supplied through ``--name value`` or ``--name=value``."""

    values: list[str] = []
    for index, argument in enumerate(arguments):
        if argument == option:
            if index + 1 >= len(arguments):
                raise error_type(f"source command has no value after {option}")
            values.append(arguments[index + 1])
        elif argument.startswith(option + "="):
            values.append(argument.split("=", 1)[1])
    return values


def drop_option(
    arguments: list[str],
    option: str,
    *,
    error_type: type[ValueError] = DeploymentContractError,
) -> list[str]:
    """Return arguments without any occurrence of one value-bearing option."""

    result: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == option:
            if index + 1 >= len(arguments):
                raise error_type(f"source command has no value after {option}")
            index += 2
            continue
        if argument.startswith(option + "="):
            index += 1
            continue
        result.append(argument)
        index += 1
    return result


def one_option(
    arguments: list[str],
    option: str,
    *,
    error_type: type[ValueError] = DeploymentContractError,
) -> str:
    """Return the single required value for an option."""

    values = option_values(arguments, option, error_type=error_type)
    if len(values) != 1:
        raise error_type(f"source command requires exactly one {option}")
    return values[0]


def optional_one_option(
    arguments: list[str],
    option: str,
    *,
    error_type: type[ValueError] = DeploymentContractError,
) -> str | None:
    """Return one optional value and reject duplicate occurrences."""

    values = option_values(arguments, option, error_type=error_type)
    if len(values) > 1:
        raise error_type(f"source command has duplicate {option}")
    return values[0] if values else None


def integer_option(
    arguments: list[str],
    option: str,
    *,
    error_type: type[ValueError] = DeploymentContractError,
) -> int:
    """Return one required option as an integer."""

    try:
        return int(one_option(arguments, option, error_type=error_type))
    except ValueError as error:
        if isinstance(error, error_type):
            raise
        raise error_type(f"source {option} must be an integer") from error


def vllm_arguments(command: list[str]) -> list[str]:
    """Normalize a direct or underscore-wrapped Docker command to ``serve``."""

    try:
        marker = command.index("_")
    except ValueError:
        direct = list(command)
        if direct and (
            direct[0] == "vllm" or direct[0].rstrip("/").endswith("/vllm")
        ):
            direct = direct[1:]
        if direct[:1] != ["serve"]:
            raise RuntimeError(
                "saved command must contain the '_' shell marker or begin"
                " with a vLLM serve invocation"
            )
        return direct
    wrapped = command[marker + 1 :]
    if wrapped[:1] != ["serve"]:
        raise RuntimeError("saved wrapper marker is not followed by vLLM serve")
    return wrapped
