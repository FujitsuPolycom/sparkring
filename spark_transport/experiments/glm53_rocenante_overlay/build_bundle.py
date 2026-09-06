#!/usr/bin/env python3
"""Compose a content-addressed SIRCL and B12X RoCEnante transport bundle.

Status: research-only. The builder reads local source trees and writes one
output directory after requiring that target to be absent. It does not contact a host, configure networking, or start a model.
Only ``b12x.comm.roce`` overlays the B12X package already present in the image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path, PurePosixPath


MANIFEST_SCHEMA = "sparkring.glm53-rocenante-private-bundle/v1"
HERE = Path(__file__).resolve().parent
REQUIRED_SIRCL_FILES = {
    "sitecustomize.py",
    "spark_collective_audit.py",
    "spark_graph_status_reporter.py",
    "spark_persistent_output_ring.py",
    "spark_tp4_backend.py",
    "spark_tp4_capability.py",
    "spark_tp4_health_gate.py",
    "spark_tp4_port_namespace.py",
    "spark_tp4_query_contract.py",
    "spark_tp4_query_row_provider.py",
    "libspark_transport_capi.so",
}
SUPPLIED_SIRCL_SUPPORT = {
    "spark_tp4_capability.py",
    "spark_tp4_health_gate.py",
}
# Native checkpoint manifests identify the library and backend by direct hashes.
NATIVE_CHECKPOINT_SCHEMA = "sparkring-private-sircl-bundle/v1"
NATIVE_CHECKPOINT_KEYS = {
    "schema",
    "date",
    "source_commit",
    "image_id",
    "library_sha256",
    "backend_sha256",
    "prefill_exposure",
    "prefill_rail_mode",
    "fused_min_query_rows",
    "fused_max_query_rows",
    "operation_slots",
    "elements_per_row",
    "public",
}


class BundleError(ValueError):
    """A source bundle is incomplete, ambiguous, or not content-addressable."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_tree(root: Path) -> tuple[str, list[dict[str, str]]]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(root).as_posix()
        records.append({"path": relative, "sha256": sha256_file(path)})
    if not records:
        raise BundleError(f"source tree contains no files: {root}")
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest(), records


def _git_state(repository: Path) -> dict[str, object]:
    provenance = repository / "provenance.json"
    if provenance.is_file():
        document = json.loads(provenance.read_text(encoding="utf-8"))
        actual, _ = _canonical_tree(repository / "b12x" / "comm" / "roce")
        if document.get("roce_tree_sha256") != actual:
            raise BundleError("vendored RoCEnante source differs from its provenance")
        state = document.get("git_state")
        if not isinstance(state, dict) or set(state) != {"commit", "roce_source_dirty", "roce_status"}:
            raise BundleError("vendored RoCEnante provenance has invalid git_state")
        return state

    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )
        if completed.returncode != 0:
            raise BundleError(
                f"git {' '.join(arguments)} failed in {repository}: "
                f"{(completed.stderr or completed.stdout).strip()}"
            )
        return completed.stdout.strip()

    status = run("status", "--porcelain", "--", "b12x/comm/roce")
    return {
        "commit": run("rev-parse", "HEAD"),
        "roce_source_dirty": bool(status),
        "roce_status": status.splitlines(),
    }


def _safe_relative(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise BundleError("base SIRCL manifest path must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise BundleError(f"base SIRCL manifest path escapes its root: {value!r}")
    return path


def _base_manifest(
    root: Path,
) -> tuple[Path, dict[str, object], list[tuple[PurePosixPath, str]]]:
    public_path = root / "sparkring-overlay-manifest.json"
    private_path = root / "manifest.json"
    present = [path for path in (public_path, private_path) if path.is_file()]
    if len(present) != 1:
        raise BundleError(
            "base SIRCL bundle must contain exactly one of "
            "sparkring-overlay-manifest.json or manifest.json"
        )
    manifest_path = present[0]
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleError(f"base SIRCL manifest cannot be read: {error}") from error
    if not isinstance(document, dict):
        raise BundleError("base SIRCL manifest root must be an object")
    if manifest_path.name == "manifest.json":
        if (
            set(document) != NATIVE_CHECKPOINT_KEYS
            or document.get("schema") != NATIVE_CHECKPOINT_SCHEMA
        ):
            raise BundleError(
                f"native checkpoint manifest must use exact {NATIVE_CHECKPOINT_SCHEMA}"
            )
        native = root / "libspark_transport_capi.so"
        backend = root / "spark_tp4_backend.py"
        if not native.is_file() or sha256_file(native) != document["library_sha256"]:
            raise BundleError("SIRCL native library differs from checkpoint manifest.json")
        if not backend.is_file() or sha256_file(backend) != document["backend_sha256"]:
            raise BundleError("SIRCL backend differs from checkpoint manifest.json")
        result = []
        for source in sorted(path for path in root.rglob("*") if path.is_file()):
            if source in (manifest_path, public_path):
                continue
            if "__pycache__" in source.parts or source.suffix in {".pyc", ".pyo"}:
                continue
            relative = PurePosixPath(source.relative_to(root).as_posix())
            result.append((relative, sha256_file(source)))
        return manifest_path, document, result

    values = document.get("files") if isinstance(document, dict) else None
    if not isinstance(values, list) or not values:
        raise BundleError("base SIRCL manifest must contain a nonempty files array")
    result = []
    for value in values:
        if not isinstance(value, dict):
            raise BundleError("base SIRCL manifest file entry must be an object")
        relative = _safe_relative(value.get("path"))
        digest = value.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise BundleError(f"base SIRCL file {relative} has no SHA-256")
        source = root.joinpath(*relative.parts)
        if not source.is_file() or sha256_file(source) != digest:
            raise BundleError(f"base SIRCL file differs from its manifest: {relative}")
        result.append((relative, digest))
    native = root / "libspark_transport_capi.so"
    if native.is_file() and all(str(path) != native.name for path, _ in result):
        result.append((PurePosixPath(native.name), sha256_file(native)))
    return manifest_path, document, result


def build(base_sircl: Path, b12x_repository: Path, output: Path, *,
          captured_sircl_rows: tuple[int, ...] | None = None) -> dict[str, object]:
    """Write one transport overlay directory from verified immutable inputs."""

    base_sircl = base_sircl.resolve()
    b12x_repository = b12x_repository.resolve()
    output = output.resolve()
    if output.exists():
        raise BundleError(f"output already exists: {output}")
    if captured_sircl_rows is not None and (
        list(captured_sircl_rows) != sorted(set(captured_sircl_rows))
        or any(type(row) is not int or not 1 <= row <= 32 for row in captured_sircl_rows)
    ):
        raise BundleError("captured SIRCL rows must be sorted unique integers in [1,32]")
    source_roce = b12x_repository / "b12x" / "comm" / "roce"
    if not (source_roce / "__init__.py").is_file():
        raise BundleError(f"B12X RoCEnante package is missing: {source_roce}")
    base_manifest_path, base_manifest, base_files = _base_manifest(base_sircl)
    base_manifest_sha256 = sha256_file(base_manifest_path)
    roce_tree_sha256, roce_files = _canonical_tree(source_roce)
    git_state = _git_state(b12x_repository)

    output.mkdir(parents=True)
    try:
        copied = []
        if base_manifest_path.name == "manifest.json":
            destination = output / "manifest.json"
            shutil.copyfile(base_manifest_path, destination)
            copied.append(
                {
                    "role": "base_sircl_manifest",
                    "path": "manifest.json",
                    "sha256": base_manifest_sha256,
                }
            )
        for relative, expected_sha256 in base_files:
            source = base_sircl.joinpath(*relative.parts)
            destination_relative = (
                PurePosixPath("sircl_sitecustomize.py")
                if relative == PurePosixPath("sitecustomize.py")
                else relative
            )
            destination = output.joinpath(*destination_relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied.append(
                {
                    "role": "base_sircl",
                    "path": destination_relative.as_posix(),
                    "sha256": expected_sha256,
                }
            )

        copied_base_names = {item["path"] for item in copied}
        integration_root = HERE.parents[2] / "spark_transport" / "integrations" / "vllm"
        for name in sorted(SUPPLIED_SIRCL_SUPPORT):
            if name in copied_base_names:
                continue
            source = integration_root / name
            if not source.is_file():
                raise BundleError(f"SparkRing SIRCL support file is missing: {source}")
            destination = output / name
            shutil.copyfile(source, destination)
            copied.append(
                {
                    "role": "spark_sircl_support",
                    "path": name,
                    "sha256": sha256_file(destination),
                }
            )
            copied_base_names.add(name)
        for name in (
            "sitecustomize.py",
            "rocenante_vllm_overlay.py",
            "rocenante_health_gate.py",
        ):
            source = HERE / name
            destination = output / name
            shutil.copyfile(source, destination)
            copied.append(
                {
                    "role": "glm53_rocenante_overlay",
                    "path": name,
                    "sha256": sha256_file(destination),
                }
            )
            copied_base_names.add(name)
        missing = sorted(REQUIRED_SIRCL_FILES - copied_base_names)
        if missing:
            raise BundleError("composed SIRCL bundle is missing: " + ",".join(missing))
        if "sircl_sitecustomize.py" not in copied_base_names:
            raise BundleError("base SIRCL bundle is missing sitecustomize.py")

        destination_roce = output / "b12x_overlay" / "b12x" / "comm" / "roce"
        shutil.copytree(
            source_roce,
            destination_roce,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        for record in roce_files:
            copied.append(
                {
                    "role": "b12x_comm_roce",
                    "path": "b12x_overlay/b12x/comm/roce/" + record["path"],
                    "sha256": record["sha256"],
                }
            )

        contract = json.loads(
            (HERE / "overlay_contract.json").read_text(encoding="utf-8")
        )
        if captured_sircl_rows is not None:
            contract["runtime"]["captured_sircl_query_rows"] = list(captured_sircl_rows)
            contract["runtime"]["execution_mode"] = "both"
        contract["artifacts"] = {
            "base_sircl_manifest_sha256": base_manifest_sha256,
            "base_sircl_manifest_name": base_manifest_path.name,
            "base_sircl_manifest": base_manifest,
            "b12x_roce_tree_sha256": roce_tree_sha256,
            "b12x_git": git_state,
        }
        config_path = output / "rocenante-overlay-config.json"
        config_path.write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        copied.append(
            {
                "role": "generated_contract",
                "path": config_path.name,
                "sha256": sha256_file(config_path),
            }
        )
        copied.sort(key=lambda item: item["path"])
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "status": "research-only",
            "base_sircl_manifest_sha256": base_manifest_sha256,
            "base_sircl_manifest_name": base_manifest_path.name,
            "b12x_roce_tree_sha256": roce_tree_sha256,
            "b12x_git": git_state,
            "files": copied,
        }
        (output / "sparkring-overlay-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return manifest
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-sircl-bundle", required=True, type=Path)
    parser.add_argument("--b12x-repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        manifest = build(
            arguments.base_sircl_bundle,
            arguments.b12x_repository,
            arguments.output,
        )
    except (
        BundleError,
        OSError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "files": len(manifest["files"]),
                "b12x_roce_tree_sha256": manifest["b12x_roce_tree_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
