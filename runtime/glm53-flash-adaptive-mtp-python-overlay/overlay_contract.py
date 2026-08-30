#!/usr/bin/env python3
"""Verify the composed vLLM Python and retained-native runtime contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable


PINS_SCHEMA = "sparkring-glm53-public-python-overlay/v1"
OVERLAY_SCHEMA = "sparkring-vllm-python-overlay/v1"
BASE_RECORD_SCHEMA = "sparkring-vllm-retained-native/v1"
SHA256_HEX_LENGTH = 64
ELF_MAGIC = b"\x7fELF"
DISPATCH_PREFIXES = ("vllm::", "_C::", "_C_cache_ops::")


class ContractError(RuntimeError):
    """The composed runtime differs from its byte-exact contract."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256_bytes(payload)


def load_json(path: Path, schema: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ContractError(f"{path} does not use schema {schema}")
    return value


def safe_relative_path(value: Any) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError(f"invalid overlay path: {value!r}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.parts[0] != "vllm":
        raise ContractError(f"overlay path escapes the vLLM package: {value!r}")
    return path


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return value


def validate_overlay_manifest(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 31:
        raise ContractError("the vLLM Python overlay must contain exactly 31 files")
    records: list[dict[str, Any]] = []
    paths: set[str] = set()
    additions = 0
    for index, raw in enumerate(files):
        if not isinstance(raw, dict):
            raise ContractError(f"overlay file record {index} is not an object")
        relative = safe_relative_path(raw.get("path"))
        normalized = relative.as_posix()
        if normalized in paths:
            raise ContractError(f"duplicate overlay path: {normalized}")
        paths.add(normalized)
        operation = raw.get("operation")
        if operation not in ("add", "replace"):
            raise ContractError(f"unsupported overlay operation for {normalized}")
        base_hash = raw.get("base_sha256")
        if operation == "add":
            additions += 1
            if base_hash is not None:
                raise ContractError(f"added overlay path has a base hash: {normalized}")
        else:
            _sha256(base_hash, f"{normalized} base hash")
        _sha256(raw.get("target_sha256"), f"{normalized} target hash")
        target_bytes = raw.get("target_bytes")
        if not isinstance(target_bytes, int) or target_bytes <= 0:
            raise ContractError(f"invalid target byte count for {normalized}")
        records.append(raw)
    if additions != 1:
        raise ContractError("the vLLM Python overlay must add exactly one file")
    return tuple(records)


def final_file_hashes(pins: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for patch in pins["sparkcache"]["patches"]:
        target = safe_relative_path(patch["target"]).as_posix()
        result[target] = _sha256(
            patch["postimage_sha256"], f"{target} patch postimage"
        )
    for patch in pins["vllm"].get("composed_runtime_patches", ()):
        for target_record in patch["targets"]:
            target = safe_relative_path(target_record["path"]).as_posix()
            result[target] = _sha256(
                target_record["postimage_sha256"],
                f"{target} composed patch postimage",
            )
    return result


def validate_dflash_loader_source(source: str) -> None:
    """Require DFlash to pass its optional draft LoadConfig to get_model."""

    tree = ast.parse(source)
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "load_dflash_model"
        ),
        None,
    )
    if function is None:
        raise ContractError("DFlash loader source omits load_dflash_model")
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "get_model"
    ]
    if len(calls) != 1:
        raise ContractError("DFlash loader must contain exactly one get_model call")
    keyword = next(
        (item for item in calls[0].keywords if item.arg == "load_config"),
        None,
    )
    value = None if keyword is None else keyword.value
    if not (
        isinstance(value, ast.Attribute)
        and value.attr == "draft_load_config"
        and isinstance(value.value, ast.Name)
        and value.value.id == "speculative_config"
    ):
        raise ContractError(
            "DFlash get_model must consume speculative_config.draft_load_config"
        )


def validate_optional_load_config_fallback(source: str) -> None:
    """Require get_model(None) to retain the enclosing vLLM LoadConfig."""

    tree = ast.parse(source)
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "get_model"
        ),
        None,
    )
    if function is None:
        raise ContractError("model-loader source omits get_model")
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "get_model_loader"
    ]
    if len(calls) != 1 or len(calls[0].args) != 1:
        raise ContractError("get_model must select exactly one model loader")
    selected = calls[0].args[0]
    if not (
        isinstance(selected, ast.BoolOp)
        and isinstance(selected.op, ast.Or)
        and len(selected.values) == 2
        and isinstance(selected.values[0], ast.Name)
        and selected.values[0].id == "load_config"
        and isinstance(selected.values[1], ast.Attribute)
        and selected.values[1].attr == "load_config"
        and isinstance(selected.values[1].value, ast.Name)
        and selected.values[1].value.id == "vllm_config"
    ):
        raise ContractError(
            "get_model must fall back from a missing draft LoadConfig to "
            "vllm_config.load_config"
        )


def validate_recurrent_boundary_sources(root: Path) -> None:
    """Require exact-boundary selection and the SchedulerOutput hand-off."""

    output_source = (root / "vllm/v1/core/sched/output.py").read_text(
        encoding="utf-8"
    )
    scheduler_source = (root / "vllm/v1/core/sched/scheduler.py").read_text(
        encoding="utf-8"
    )
    manager_source = (
        root / "vllm/v1/core/single_type_kv_cache_manager.py"
    ).read_text(encoding="utf-8")
    cache_source = (root / "vllm/v1/core/kv_cache_manager.py").read_text(
        encoding="utf-8"
    )
    required = {
        "SchedulerOutput field": (
            "recurrent_boundary_blocks: "
            "dict[str, list[tuple[int, int, int]]] | None = None",
            output_source,
        ),
        "scheduler hand-off": (
            "recurrent_boundary_blocks=pending_recurrent_boundary_blocks",
            scheduler_source,
        ),
        "connector capability gate": (
            '"supports_recurrent_boundary_blocks"',
            scheduler_source,
        ),
        "consumer-compatible free fence": (
            "kv_transfer_config.is_kv_consumer",
            scheduler_source,
        ),
        "overlapping-step free fence": (
            "multiple_inflight_batches and (",
            scheduler_source,
        ),
        "prompt-minus-one replay rule": (
            "(request.num_prompt_tokens - 1) // self.block_pool.hash_block_size",
            manager_source,
        ),
        "exact token-boundary proof": (
            "block.block_hash_num_tokens != replay_boundary",
            manager_source,
        ),
        "exact group proof": (
            "get_group_id(block.block_hash) != self.kv_cache_group_id",
            manager_source,
        ),
        "request-lifetime pin": ("self._pin_recurrent_boundary", cache_source),
    }
    for label, (fragment, source) in required.items():
        if fragment not in source:
            raise ContractError(f"recurrent-boundary source omits {label}")


def verify_vllm_runtime_patch_files(
    root: Path,
    pins: dict[str, Any],
) -> list[dict[str, str]]:
    """Verify installed runtime-patch postimages and loader semantics."""

    verified = []
    for record in pins["vllm"].get("runtime_patches", ()):
        relative = safe_relative_path(record["target"])
        path = root / relative
        observed = sha256_file(path)
        if observed != record["postimage_sha256"]:
            raise ContractError(
                f"vLLM runtime patch postimage mismatch for {relative}: "
                f"expected {record['postimage_sha256']}, got {observed}"
            )
        compile(path.read_bytes(), str(path), "exec")
        if relative.as_posix() == "vllm/v1/worker/gpu/spec_decode/dflash/utils.py":
            validate_dflash_loader_source(path.read_text(encoding="utf-8"))
        verified.append(
            {
                "path": relative.as_posix(),
                "sha256": observed,
            }
        )
    for record in pins["vllm"].get("composed_runtime_patches", ()):
        for target_record in record["targets"]:
            relative = safe_relative_path(target_record["path"])
            path = root / relative
            observed = sha256_file(path)
            if observed != target_record["postimage_sha256"]:
                raise ContractError(
                    f"vLLM composed patch postimage mismatch for {relative}: "
                    f"expected {target_record['postimage_sha256']}, got {observed}"
                )
            compile(path.read_bytes(), str(path), "exec")
            verified.append({"path": relative.as_posix(), "sha256": observed})
    if pins["vllm"].get("composed_runtime_patches"):
        validate_recurrent_boundary_sources(root)
    loader = root / "vllm/model_executor/model_loader/__init__.py"
    validate_optional_load_config_fallback(loader.read_text(encoding="utf-8"))
    return verified


def verify_overlay_files(
    root: Path,
    manifest: dict[str, Any],
    *,
    stage: str,
    pins: dict[str, Any] | None = None,
) -> list[Path]:
    records = validate_overlay_manifest(manifest)
    if stage not in ("base", "target", "final"):
        raise ContractError(f"unsupported overlay verification stage: {stage}")
    final_hashes = final_file_hashes(pins) if stage == "final" and pins else {}
    verified: list[Path] = []
    root = root.resolve()
    for record in records:
        relative = safe_relative_path(record["path"])
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ContractError(f"overlay path escapes root: {relative}") from exc
        if stage == "base" and record["operation"] == "add":
            if path.exists():
                raise ContractError(f"base image unexpectedly contains {relative}")
            continue
        if not path.is_file():
            raise ContractError(f"overlay file is absent: {relative}")
        expected = (
            record["base_sha256"]
            if stage == "base"
            else final_hashes.get(relative.as_posix(), record["target_sha256"])
        )
        observed = sha256_file(path)
        if observed != expected:
            raise ContractError(
                f"{stage} vLLM source mismatch for {relative}: "
                f"expected {expected}, got {observed}"
            )
        if stage != "base" and path.stat().st_size != record["target_bytes"]:
            if relative.as_posix() not in final_hashes:
                raise ContractError(f"target byte count mismatch for {relative}")
        verified.append(path)
    return verified


def elf_manifest(package_root: Path) -> list[dict[str, Any]]:
    package_root = package_root.resolve()
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in package_root.rglob("*") if item.is_file()):
        with path.open("rb") as handle:
            magic = handle.read(len(ELF_MAGIC))
        if magic != ELF_MAGIC:
            continue
        records.append(
            {
                "path": path.relative_to(package_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not records:
        raise ContractError(f"no ELF files found below {package_root}")
    return records


def distribution_metadata_manifest(site_root: Path, console_script: Path) -> dict[str, Any]:
    candidates = sorted(site_root.glob("vllm-*.dist-info"))
    if len(candidates) != 1:
        raise ContractError(
            f"expected one vLLM dist-info directory below {site_root}, got {len(candidates)}"
        )
    dist_info = candidates[0]
    files = []
    for path in sorted(item for item in dist_info.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(site_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    version_file = site_root / "vllm" / "_version.py"
    for path, label in ((version_file, "vLLM generated version"), (console_script, "vLLM console script")):
        if not path.is_file():
            raise ContractError(f"{label} is absent: {path}")
    return {
        "distribution_version": importlib.metadata.version("vllm"),
        "dist_info": dist_info.name,
        "files": files,
        "version_file_sha256": sha256_file(version_file),
        "console_script": str(console_script),
        "console_script_sha256": sha256_file(console_script),
    }


def dispatch_manifest() -> list[str]:
    try:
        import torch
        import vllm._custom_ops  # noqa: F401
    except Exception as exc:
        raise ContractError(f"cannot load retained vLLM native operators: {exc}") from exc
    names = sorted(
        name
        for name in torch._C._dispatch_get_all_op_names()
        if name.startswith(DISPATCH_PREFIXES)
    )
    if not names:
        raise ContractError("vLLM native dispatch operator set is empty")
    return names


def make_base_record(site_root: Path, console_script: Path) -> dict[str, Any]:
    native = elf_manifest(site_root / "vllm")
    dispatch = dispatch_manifest()
    metadata = distribution_metadata_manifest(site_root, console_script)
    return {
        "schema": BASE_RECORD_SCHEMA,
        "native_elf": native,
        "native_elf_manifest_sha256": canonical_sha256(native),
        "native_dispatch": dispatch,
        "native_dispatch_manifest_sha256": canonical_sha256(dispatch),
        "distribution_metadata": metadata,
        "distribution_metadata_manifest_sha256": canonical_sha256(metadata),
    }


def verify_retained_native(
    site_root: Path, console_script: Path, base_record: dict[str, Any]
) -> None:
    if base_record.get("schema") != BASE_RECORD_SCHEMA:
        raise ContractError("retained-native record uses an unsupported schema")
    current = make_base_record(site_root, console_script)
    for key in (
        "native_elf_manifest_sha256",
        "native_dispatch_manifest_sha256",
        "distribution_metadata_manifest_sha256",
    ):
        if current[key] != base_record.get(key):
            raise ContractError(
                f"retained vLLM {key} mismatch: "
                f"expected {base_record.get(key)}, got {current[key]}"
            )


def verify_b12x_contract(pins: dict[str, Any]) -> dict[str, Any]:
    try:
        import b12x
        from b12x.gemm import bf16_vocab_projection
        from b12x.sequence import gdn_decode
    except Exception as exc:
        raise ContractError(f"cannot import the pinned B12X interfaces: {exc}") from exc
    version = importlib.metadata.version("b12x")
    expected_version = pins["b12x"]["package_version"]
    if version != expected_version:
        raise ContractError(f"B12X version mismatch: expected {expected_version}, got {version}")
    field_names, bind_parameters = validate_b12x_surface(
        gdn_decode.Caps,
        gdn_decode.bind_kda,
        bf16_vocab_projection,
        required_field=pins["b12x"]["required_caps_field"],
    )
    caps = gdn_decode.Caps(
        device="cuda:0",
        max_tokens=1,
        max_seqs=1,
        max_state_slots=1,
        key_heads=1,
        value_heads=1,
        gate_activation="sigmoid",
        kda_metadata_validation=pins["b12x"]["required_caps_value"],
    )
    if caps.kda_metadata_validation != pins["b12x"]["required_caps_value"]:
        raise ContractError("B12X Caps did not retain trusted metadata validation")
    return {
        "version": version,
        "module": str(Path(b12x.__file__).resolve()),
        "caps_fields": sorted(field_names),
        "bind_kda_parameters": list(bind_parameters),
    }


def validate_b12x_surface(
    caps_type: Any,
    bind_kda: Any,
    bf16_vocab_projection: Any,
    *,
    required_field: str,
) -> tuple[set[str], tuple[str, ...]]:
    field_names = {item.name for item in fields(caps_type)}
    if required_field not in field_names:
        raise ContractError(f"B12X Caps omits required field {required_field}")
    bind_parameters = tuple(inspect.signature(bind_kda).parameters)
    required_bind_parameters = {
        "mixed_qkv",
        "raw_g",
        "raw_beta",
        "z",
        "query_start_loc",
        "num_accepted_tokens",
        "state_indices",
        "num_seqs",
        "num_tokens",
        "output",
    }
    missing = sorted(required_bind_parameters.difference(bind_parameters))
    if missing:
        raise ContractError(f"B12X bind_kda omits live tensor parameters: {missing}")
    if not hasattr(bf16_vocab_projection, "plan"):
        raise ContractError("B12X BF16 vocabulary projection omits plan")
    return field_names, bind_parameters


def verify_dependencies(pins: dict[str, Any]) -> dict[str, str]:
    expected = pins["dependencies"]
    observed = {
        "torch": importlib.metadata.version("torch"),
        "fastsafetensors": importlib.metadata.version("fastsafetensors"),
        "instanttensor": importlib.metadata.version("instanttensor"),
    }
    for name, version in observed.items():
        if version != expected[name]:
            raise ContractError(
                f"{name} version mismatch: expected {expected[name]}, got {version}"
            )
    nccl = Path(expected["nccl_library"])
    if not nccl.is_file():
        raise ContractError(f"pinned NCCL library is absent: {nccl}")
    observed_nccl = sha256_file(nccl)
    if observed_nccl != expected["nccl_library_sha256"]:
        raise ContractError(
            f"NCCL library mismatch: expected {expected['nccl_library_sha256']}, "
            f"got {observed_nccl}"
        )
    observed["nccl_sha256"] = observed_nccl
    return observed


def compile_overlay_files(root: Path, manifest: dict[str, Any]) -> None:
    for record in validate_overlay_manifest(manifest):
        path = root / safe_relative_path(record["path"])
        compile(path.read_bytes(), str(path), "exec")


def composed_report(
    *,
    root: Path,
    site_root: Path,
    console_script: Path,
    pins: dict[str, Any],
    manifest: dict[str, Any],
    base_record: dict[str, Any],
) -> dict[str, Any]:
    verified = verify_overlay_files(root, manifest, stage="final", pins=pins)
    compile_overlay_files(root, manifest)
    verify_retained_native(site_root, console_script, base_record)
    runtime_patches = verify_vllm_runtime_patch_files(root, pins)
    return {
        "schema": "sparkring-glm53-public-python-overlay-verification/v1",
        "status": "implemented",
        "vllm_python_files_verified": len(verified),
        "vllm_python_commit": pins["vllm"]["python_commit"],
        "vllm_native_commit": pins["vllm"]["native_commit"],
        "vllm_runtime_patches": runtime_patches,
        "b12x": verify_b12x_contract(pins),
        "dependencies": verify_dependencies(pins),
        "native_elf_manifest_sha256": base_record["native_elf_manifest_sha256"],
        "native_dispatch_manifest_sha256": base_record[
            "native_dispatch_manifest_sha256"
        ],
        "distribution_metadata_manifest_sha256": base_record[
            "distribution_metadata_manifest_sha256"
        ],
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_files = subparsers.add_parser("verify-files")
    verify_files.add_argument("--root", type=Path, required=True)
    verify_files.add_argument("--stage", choices=("base", "target", "final"), required=True)

    record_base = subparsers.add_parser("record-base")
    record_base.add_argument("--site-root", type=Path, required=True)
    record_base.add_argument("--console-script", type=Path, default=Path("/usr/local/bin/vllm"))
    record_base.add_argument("--output", type=Path, required=True)

    verify_composed = subparsers.add_parser("verify-composed")
    verify_composed.add_argument("--root", type=Path, required=True)
    verify_composed.add_argument("--site-root", type=Path, required=True)
    verify_composed.add_argument("--console-script", type=Path, default=Path("/usr/local/bin/vllm"))
    verify_composed.add_argument("--base-record", type=Path, required=True)
    verify_composed.add_argument("--output", type=Path)

    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        pins = load_json(args.pins, PINS_SCHEMA)
        manifest = load_json(args.manifest, OVERLAY_SCHEMA)
        if args.command == "verify-files":
            verified = verify_overlay_files(
                args.root,
                manifest,
                stage=args.stage,
                pins=pins,
            )
            report: dict[str, Any] = {
                "stage": args.stage,
                "verified": len(verified),
            }
        elif args.command == "record-base":
            report = make_base_record(args.site_root, args.console_script)
            _write_json(args.output, report)
        else:
            base_record = load_json(args.base_record, BASE_RECORD_SCHEMA)
            report = composed_report(
                root=args.root,
                site_root=args.site_root,
                console_script=args.console_script,
                pins=pins,
                manifest=manifest,
                base_record=base_record,
            )
            if args.output:
                _write_json(args.output, report)
    except (ContractError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
