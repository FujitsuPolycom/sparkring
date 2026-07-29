"""Verify the exact vLLM files that provide delayed KV-block reclamation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class ContractError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_contract(vllm_root: Path, contract_path: Path) -> list[Path]:
    try:
        contract: dict[str, Any] = json.loads(contract_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read contract: {error}") from error
    if contract.get("schema") != "sparkring-vllm-kv-block-lease-contract/v1":
        raise ContractError("unsupported lease-contract schema")
    files = contract.get("files")
    if not isinstance(files, list) or not files:
        raise ContractError("lease contract has no files")

    root = vllm_root.resolve()
    verified: list[Path] = []
    for record in files:
        if not isinstance(record, dict):
            raise ContractError("invalid lease-contract file record")
        relative = Path(str(record.get("path", "")))
        expected_single = record.get("sha256")
        accepted_named = record.get("accepted_sha256")
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError(f"unsafe contract path: {relative}")
        if (expected_single is None) == (accepted_named is None):
            raise ContractError(
                f"{relative} must define exactly one of sha256 or accepted_sha256"
            )
        if expected_single is not None:
            expected = str(expected_single).lower()
            if len(expected) != 64 or any(
                character not in "0123456789abcdef" for character in expected
            ):
                raise ContractError(f"invalid SHA-256 for {relative}")
            accepted = {"pinned": expected}
        else:
            if not isinstance(accepted_named, dict) or not accepted_named:
                raise ContractError(f"invalid accepted_sha256 map for {relative}")
            accepted = {}
            for label, digest in accepted_named.items():
                if (
                    not isinstance(label, str)
                    or not label
                    or not isinstance(digest, str)
                ):
                    raise ContractError(
                        f"invalid accepted_sha256 entry for {relative}"
                    )
                normalized = digest.lower()
                if len(normalized) != 64 or any(
                    character not in "0123456789abcdef"
                    for character in normalized
                ):
                    raise ContractError(
                        f"invalid SHA-256 for {relative} state {label!r}"
                    )
                accepted[label] = normalized
            if len(set(accepted.values())) != len(accepted):
                raise ContractError(
                    f"duplicate accepted SHA-256 states for {relative}"
                )
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root):
            raise ContractError(f"contract path escapes vLLM root: {relative}")
        try:
            actual = _sha256(candidate)
        except OSError as error:
            raise ContractError(f"cannot hash {relative}: {error}") from error
        if actual not in accepted.values():
            states = ", ".join(
                f"{label}={digest}" for label, digest in accepted.items()
            )
            raise ContractError(
                f"vLLM lease-contract mismatch for {relative}: "
                f"expected one of [{states}], got {actual}"
            )
        verified.append(candidate)
    return verified


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", required=True, type=Path)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name("vllm-kv-block-lease-contract.json"),
    )
    args = parser.parse_args()
    verified = verify_contract(args.vllm_root, args.contract)
    print(f"verified SparkCache KV-block lease contract: {len(verified)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
