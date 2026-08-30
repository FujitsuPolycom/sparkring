"""Canonical SparkCache CUDA configuration names and compatibility aliases."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


CANONICAL_CONNECTOR_KEYS = {
    "spark_cache_cuda_restore": "spark_cache_native_restore",
    "spark_cache_cuda_placement_library": "spark_cache_native_library",
    "spark_cache_cuda_placement_library_sha256": (
        "spark_cache_native_library_sha256"
    ),
    "spark_cache_cuda_placement_arena_bytes": "spark_cache_native_arena_bytes",
    "spark_cache_cuda_restore_io_workers": "spark_cache_native_io_workers",
}


class SparkCacheTerminologyError(ValueError):
    """Canonical and compatibility names cannot produce one exact contract."""


def _same_json_value(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def canonicalize_connector_extra_config(extra: Mapping[str, Any]) -> dict[str, Any]:
    """Return canonical connector keys and reject contradictory aliases."""

    result = dict(extra)
    for canonical, legacy in CANONICAL_CONNECTOR_KEYS.items():
        canonical_present = canonical in result
        legacy_present = legacy in result
        if canonical_present and legacy_present and not _same_json_value(
            result[canonical], result[legacy]
        ):
            raise SparkCacheTerminologyError(
                f"SparkCache connector keys {canonical} and compatibility alias "
                f"{legacy} have conflicting values"
            )
        if legacy_present and not canonical_present:
            result[canonical] = result[legacy]
        result.pop(legacy, None)
    return result


def canonicalize_connector_arguments(arguments: Sequence[str]) -> tuple[str, ...]:
    """Normalize SparkCache keys inside one vLLM KV-transfer argument."""

    result = tuple(arguments)
    locations: list[tuple[int, str, str]] = []
    for index, argument in enumerate(result):
        if argument == "--kv-transfer-config":
            if index + 1 >= len(result):
                raise SparkCacheTerminologyError(
                    "--kv-transfer-config requires a JSON value"
                )
            locations.append((index + 1, result[index + 1], "separate"))
        elif argument.startswith("--kv-transfer-config="):
            locations.append(
                (
                    index,
                    argument.split("=", 1)[1],
                    "equals",
                )
            )
    if not locations:
        return result
    if len(locations) != 1:
        raise SparkCacheTerminologyError(
            "a runtime profile must contain at most one --kv-transfer-config"
        )

    index, encoded, form = locations[0]
    try:
        transfer = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise SparkCacheTerminologyError(
            "--kv-transfer-config must contain valid JSON"
        ) from error
    if not isinstance(transfer, dict):
        raise SparkCacheTerminologyError(
            "--kv-transfer-config must contain a JSON object"
        )
    extra = transfer.get("kv_connector_extra_config")
    if extra is None:
        return result
    if not isinstance(extra, dict):
        raise SparkCacheTerminologyError(
            "kv_connector_extra_config must contain a JSON object"
        )
    canonical = canonicalize_connector_extra_config(extra)
    if canonical == extra:
        return result

    transfer["kv_connector_extra_config"] = canonical
    replacement = json.dumps(transfer, separators=(",", ":"))
    mutable = list(result)
    mutable[index] = (
        replacement
        if form == "separate"
        else f"--kv-transfer-config={replacement}"
    )
    return tuple(mutable)


def canonicalize_profile_connector_arguments(
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a profile whose emitted vLLM arguments use canonical keys."""

    result = dict(profile)
    arguments = result.get("extra_vllm_args")
    if arguments is None:
        return result
    if not isinstance(arguments, list) or not all(
        isinstance(argument, str) for argument in arguments
    ):
        raise SparkCacheTerminologyError(
            "profile extra_vllm_args must contain a JSON string array"
        )
    normalized = canonicalize_connector_arguments(arguments)
    if normalized != tuple(arguments):
        result["extra_vllm_args"] = list(normalized)
    return result


def resolve_string_alias(
    canonical_value: str | None,
    legacy_value: str | None,
    *,
    canonical_name: str,
    legacy_name: str,
) -> str:
    """Resolve one CLI/Python alias pair without accepting ambiguity."""

    if (
        canonical_value is not None
        and legacy_value is not None
        and canonical_value != legacy_value
    ):
        raise SparkCacheTerminologyError(
            f"{canonical_name} and compatibility alias {legacy_name} "
            "have conflicting values"
        )
    selected = canonical_value if canonical_value is not None else legacy_value
    if selected is None:
        raise SparkCacheTerminologyError(f"{canonical_name} is required")
    return selected
