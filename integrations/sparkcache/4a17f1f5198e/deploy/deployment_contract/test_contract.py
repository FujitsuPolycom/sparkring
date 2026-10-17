from __future__ import annotations

import json
from pathlib import Path

import pytest

from deploy.deployment_contract import (
    DeploymentContractError,
    compact_json,
    drop_option,
    environment_map,
    integer_option,
    one_option,
    option_values,
    optional_one_option,
    read_single_inspection,
    validate_port,
    vllm_arguments,
)


class AdapterError(DeploymentContractError):
    pass


def test_command_options_preserve_order_and_support_both_spellings() -> None:
    arguments = ["serve", "/model", "--port", "8000", "--port=8100", "--flag"]

    assert option_values(arguments, "--port") == ["8000", "8100"]
    assert drop_option(arguments, "--port") == ["serve", "/model", "--flag"]


def test_option_cardinality_and_integer_errors_use_adapter_error() -> None:
    with pytest.raises(AdapterError, match="requires exactly one --port"):
        one_option([], "--port", error_type=AdapterError)
    with pytest.raises(AdapterError, match="duplicate --port"):
        optional_one_option(
            ["--port", "8000", "--port=8100"],
            "--port",
            error_type=AdapterError,
        )
    with pytest.raises(AdapterError, match="must be an integer"):
        integer_option(["--port", "invalid"], "--port", error_type=AdapterError)


def test_environment_map_rejects_duplicates_with_adapter_error() -> None:
    assert environment_map(["A=1", "B=2"]) == {"A": "1", "B": "2"}
    with pytest.raises(AdapterError, match="unique NAME=VALUE"):
        environment_map(["A=1", "A=2"], error_type=AdapterError)
    assert environment_map(["A=1", "A=2"], require_unique=False) == {"A": "2"}


def test_vllm_arguments_accept_direct_and_wrapped_commands() -> None:
    assert vllm_arguments(["vllm", "serve", "/model"]) == ["serve", "/model"]
    assert vllm_arguments(["bash", "-lc", "_", "serve", "/model"]) == [
        "serve",
        "/model",
    ]


def test_port_and_json_results_are_deterministic() -> None:
    assert validate_port(8000, "api_port") == 8000
    assert compact_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    with pytest.raises(AdapterError, match=r"\[1, 65535\]"):
        validate_port(0, "api_port", error_type=AdapterError)


def test_single_inspection_accepts_a_bom_and_rejects_multiple_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "inspect.json"
    record = {"Config": {"Cmd": ["serve", "/model"]}}
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps([record]).encode("utf-8"))
    assert read_single_inspection(path) == record

    path.write_text(json.dumps([record, record]), encoding="utf-8")
    with pytest.raises(AdapterError, match="one container"):
        read_single_inspection(path, error_type=AdapterError)
