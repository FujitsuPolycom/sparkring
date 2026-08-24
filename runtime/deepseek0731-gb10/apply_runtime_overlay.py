#!/usr/bin/env python3
"""Apply the exact GB10 DeepSeek runtime hardening patch fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_CONTRACT = HERE / "runtime-contract.json"


class OverlayError(RuntimeError):
    """The installed GB10 tree or packaged patch differs from its contract."""


@dataclass(frozen=True)
class FileContract:
    path: str
    preimage_sha256: str
    result_sha256: str


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    old_lines: tuple[str, ...]
    new_lines: tuple[str, ...]


@dataclass(frozen=True)
class FilePatch:
    path: str
    hunks: tuple[Hunk, ...]


@dataclass(frozen=True)
class PreparedFile:
    path: Path
    result: bytes
    mode: int


_HUNK = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OverlayError(f"cannot read runtime contract {path}: {exc}") from exc
    if value.get("schema") != "sparkring-deepseek-v4-gb10-hardening/v1":
        raise OverlayError("unsupported GB10 runtime contract schema")
    return value


def _strip_patch_prefix(value: str) -> str:
    value = value.split("\t", 1)[0].strip()
    if value.startswith(("a/", "b/")):
        value = value[2:]
    if not value or value.startswith("/") or ".." in Path(value).parts:
        raise OverlayError(f"unsafe patch path: {value!r}")
    return value


def parse_unified_patch(text: str) -> dict[str, FilePatch]:
    """Parse the deliberately small text-only unified patch subset we ship."""

    lines = text.splitlines(keepends=True)
    result: dict[str, FilePatch] = {}
    index = 0
    while index < len(lines):
        if not lines[index].startswith("--- "):
            index += 1
            continue
        old_path = _strip_patch_prefix(lines[index][4:])
        index += 1
        if index >= len(lines) or not lines[index].startswith("+++ "):
            raise OverlayError(f"patch omits result path after {old_path}")
        new_path = _strip_patch_prefix(lines[index][4:])
        if old_path != new_path:
            raise OverlayError(f"patch renames are unsupported: {old_path} -> {new_path}")
        if new_path in result:
            raise OverlayError(f"patch repeats path: {new_path}")
        index += 1
        hunks: list[Hunk] = []
        while index < len(lines) and not lines[index].startswith(
            ("--- ", "diff --git ")
        ):
            if not lines[index].startswith("@@ "):
                index += 1
                continue
            match = _HUNK.match(lines[index])
            if match is None:
                raise OverlayError(f"invalid hunk header: {lines[index].rstrip()}")
            old_start = int(match.group(1))
            old_count = int(match.group(2) or "1")
            new_start = int(match.group(3))
            new_count = int(match.group(4) or "1")
            index += 1
            old_lines: list[str] = []
            new_lines: list[str] = []
            while index < len(lines):
                line = lines[index]
                if line.startswith(("@@ ", "--- ", "diff --git ")) or line == "-- \n":
                    break
                if line.startswith("\\ No newline at end of file"):
                    raise OverlayError("patches without a final newline are unsupported")
                if not line or line[0] not in " +-":
                    raise OverlayError(f"invalid unified-patch line: {line.rstrip()}")
                if line[0] in " -":
                    old_lines.append(line[1:])
                if line[0] in " +":
                    new_lines.append(line[1:])
                index += 1
            if len(old_lines) != old_count or len(new_lines) != new_count:
                raise OverlayError(
                    f"hunk count differs for {new_path}: "
                    f"header {old_count}/{new_count}, body "
                    f"{len(old_lines)}/{len(new_lines)}"
                )
            hunks.append(
                Hunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    old_lines=tuple(old_lines),
                    new_lines=tuple(new_lines),
                )
            )
        if not hunks:
            raise OverlayError(f"patch has no hunks for {new_path}")
        result[new_path] = FilePatch(new_path, tuple(hunks))
    if not result:
        raise OverlayError("patch contains no file diffs")
    return result


def apply_file_patch(source: bytes, patch: FilePatch) -> bytes:
    try:
        lines = source.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError as exc:
        raise OverlayError(f"patch input is not UTF-8: {patch.path}") from exc
    offset = 0
    for hunk in patch.hunks:
        position = hunk.old_start if hunk.old_count == 0 else hunk.old_start - 1
        position += offset
        observed = tuple(lines[position : position + hunk.old_count])
        if observed != hunk.old_lines:
            raise OverlayError(
                f"patch context differs for {patch.path} at old line {hunk.old_start}"
            )
        lines[position : position + hunk.old_count] = hunk.new_lines
        offset += hunk.new_count - hunk.old_count
    return "".join(lines).encode("utf-8")


def _target(root: Path, relative: str) -> Path:
    resolved_root = root.resolve(strict=True)
    target = (resolved_root / relative).resolve(strict=True)
    if resolved_root not in target.parents or not target.is_file():
        raise OverlayError(f"patch target escapes root or is not a file: {target}")
    return target


def _contracts(values: Sequence[dict[str, Any]]) -> tuple[FileContract, ...]:
    return tuple(
        FileContract(
            path=str(value["path"]),
            preimage_sha256=str(value["preimage_sha256"]),
            result_sha256=str(value["result_sha256"]),
        )
        for value in values
    )


def prepare_patch_set(
    root: Path,
    patch_path: Path,
    contracts: Sequence[FileContract],
    expected_patch_sha256: str,
) -> tuple[str, list[PreparedFile]]:
    observed_patch_sha256 = sha256_file(patch_path)
    if observed_patch_sha256 != expected_patch_sha256:
        raise OverlayError(
            f"patch SHA-256 differs: expected {expected_patch_sha256}, "
            f"got {observed_patch_sha256}"
        )
    parsed = parse_unified_patch(patch_path.read_text(encoding="utf-8"))
    expected_paths = {item.path for item in contracts}
    if set(parsed) != expected_paths:
        raise OverlayError(
            "patch path set differs: "
            f"expected {sorted(expected_paths)}, got {sorted(parsed)}"
        )

    states: list[str] = []
    originals: dict[str, tuple[Path, bytes, int]] = {}
    for item in contracts:
        path = _target(root, item.path)
        source = path.read_bytes()
        observed = sha256_bytes(source)
        if observed == item.preimage_sha256:
            states.append("preimage")
        elif observed == item.result_sha256:
            states.append("result")
        else:
            raise OverlayError(
                f"{path} differs: expected {item.preimage_sha256} or "
                f"{item.result_sha256}, got {observed}"
            )
        originals[item.path] = (path, source, path.stat().st_mode)

    if all(state == "result" for state in states):
        return "already-applied", []
    if not all(state == "preimage" for state in states):
        raise OverlayError("patch set is partially applied; refusing mixed state")

    prepared: list[PreparedFile] = []
    for item in contracts:
        path, source, mode = originals[item.path]
        result = apply_file_patch(source, parsed[item.path])
        observed = sha256_bytes(result)
        if observed != item.result_sha256:
            raise OverlayError(
                f"{path} result differs: expected {item.result_sha256}, got {observed}"
            )
        if path.suffix == ".py":
            try:
                compile(result.decode("utf-8"), str(path), "exec")
            except (SyntaxError, UnicodeDecodeError) as exc:
                raise OverlayError(f"patched Python does not compile: {path}: {exc}") from exc
        prepared.append(PreparedFile(path=path, result=result, mode=mode))
    return "applied", prepared


def commit_patch_set(prepared: Sequence[PreparedFile]) -> None:
    for item in prepared:
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=item.path.parent, delete=False) as stream:
                temporary_name = stream.name
                stream.write(item.result)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary_name, item.mode)
            os.replace(temporary_name, item.path)
            bytecode_dir = item.path.parent / "__pycache__"
            if bytecode_dir.is_dir():
                for bytecode in bytecode_dir.glob(f"{item.path.stem}.*.pyc"):
                    bytecode.unlink()
        finally:
            if temporary_name is not None and os.path.exists(temporary_name):
                os.unlink(temporary_name)


def apply_patch_set(
    root: Path,
    patch_path: Path,
    contracts: Sequence[FileContract],
    expected_patch_sha256: str,
) -> str:
    status, prepared = prepare_patch_set(
        root, patch_path, contracts, expected_patch_sha256
    )
    commit_patch_set(prepared)
    return status


def installed_site_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.submodule_search_locations is None:
        raise OverlayError("installed vLLM package is unavailable")
    locations = list(spec.submodule_search_locations)
    if len(locations) != 1:
        raise OverlayError(f"expected one installed vLLM package, found {locations}")
    return Path(locations[0]).resolve().parent


def attest_noop_files(site_root: Path, records: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for record in records:
        path = _target(site_root, str(record["path"]))
        observed = sha256_file(path)
        expected = str(record["sha256"])
        if observed != expected:
            raise OverlayError(
                f"no-op attestation differs for {path}: expected {expected}, got {observed}"
            )
        result.append({"path": str(path), "sha256": observed, "status": "attested"})

    sparse = (_target(site_root, "vllm/models/deepseek_v4/sparse_mla.py")).read_text(
        encoding="utf-8"
    )
    for required in (
        "self.c128a_max_compressed",
        "global_decode_buffer[:num_decode_tokens]",
        "global_decode_buffer.stride(0)",
        "prefill_buffer.stride(0)",
    ):
        if required not in sparse:
            raise OverlayError(f"GB10 C128A semantic attestation is missing {required!r}")

    attention = (_target(site_root, "vllm/models/deepseek_v4/attention.py")).read_text(
        encoding="utf-8"
    )
    shortcut = "indexer_metadata.max_seq_len // self.compress_ratio <= self.topk_tokens"
    if shortcut in attention:
        raise OverlayError("GB10 unexpectedly contains the captured all-candidates shortcut")
    if "return self.indexer_op(hidden_states, q_quant, k, weights)" not in attention:
        raise OverlayError("GB10 indexer dispatch semantic attestation is missing")
    return result


def build_receipt(
    site_root: Path,
    contract_path: Path,
    *,
    attest_noops: bool = True,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    noops = (
        attest_noop_files(site_root, contract["attested_noop"])
        if attest_noops
        else []
    )
    runtime_patch = contract["runtime_patch"]
    patch_path = (contract_path.parent / runtime_patch["path"]).resolve(strict=True)
    records = _contracts(runtime_patch["files"])
    status = apply_patch_set(
        site_root,
        patch_path,
        records,
        str(runtime_patch["sha256"]),
    )
    return {
        "schema": "sparkring-deepseek-v4-gb10-runtime-overlay/v1",
        "status": status,
        "target_kind": "installed" if attest_noops else "retained-source",
        "site_root": str(site_root.resolve()),
        "contract_sha256": sha256_file(contract_path),
        "patch_sha256": sha256_file(patch_path),
        "attested_noop": noops,
        "files": [
            {
                "path": str(_target(site_root, item.path)),
                "sha256": item.result_sha256,
            }
            for item in records
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-root", type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument(
        "--patch-only",
        action="store_true",
        help="patch the exact retained source tree without installed-only no-op attestations",
    )
    args = parser.parse_args(argv)
    try:
        contract_path = args.contract.resolve(strict=True)
        site_root = (
            args.site_root.resolve(strict=True)
            if args.site_root is not None
            else installed_site_root()
        )
        result = build_receipt(
            site_root,
            contract_path,
            attest_noops=not args.patch_only,
        )
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.receipt is None:
            print(payload, end="")
        else:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
    except (FileExistsError, OSError, OverlayError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
