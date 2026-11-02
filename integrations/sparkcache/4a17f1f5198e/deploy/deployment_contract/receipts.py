"""Verification of source-bound overlay receipts and generated files."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from .errors import DeploymentContractError
from .source import file_sha256


def validate_overlay_receipt(
    receipt_path: Path,
    *,
    role: str,
    schema: str,
    expected_files: Mapping[str, str],
    file_paths: Mapping[str, Path],
    source_sha256: str,
    error_type: type[ValueError] = DeploymentContractError,
) -> str:
    """Verify one receipt, its source identity, and every generated file."""

    if set(file_paths) != set(expected_files):
        raise ValueError("file_paths and expected_files must name the same outputs")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise error_type(f"cannot read {role} overlay receipt") from error
    if not isinstance(receipt, dict) or receipt.get("schema") != schema:
        raise error_type(f"{role} overlay receipt schema differs")
    records = {
        record.get("output"): record.get("sha256")
        for record in receipt.get("files", ())
        if isinstance(record, dict)
    }
    if set(records) != set(expected_files):
        raise error_type(f"{role} overlay receipt inventory differs")
    for name, digest in expected_files.items():
        if records[name] != digest:
            raise error_type(f"{role} overlay receipt hash differs for {name}")
    receipt_source_sha256 = receipt.get("sparkcache_source_sha256")
    if (
        not isinstance(receipt_source_sha256, str)
        or len(receipt_source_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in receipt_source_sha256
        )
    ):
        raise error_type(
            f"SparkCache source digest is absent from the {role} overlay receipt"
        )
    if receipt_source_sha256 != source_sha256:
        raise error_type(
            f"SparkCache source digest differs from the {role} deployment contract"
        )
    for name, digest in expected_files.items():
        path = file_paths[name]
        try:
            actual = file_sha256(path)
        except OSError as error:
            raise error_type(f"cannot read {role} overlay {path}") from error
        if actual != digest:
            raise error_type(f"{role} overlay hash differs for {name}")
    return source_sha256
