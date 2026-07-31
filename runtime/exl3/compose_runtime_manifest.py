#!/usr/bin/env python3
"""Compose an EXL3 receipt from the exact inherited NF3 runtime receipt."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path


BASE_MANIFEST = Path("/opt/sparkring/runtime-manifest.json")
PREFIX = Path("/opt/sparkring-exl3")
SITE_PACKAGES = Path("/opt/venv/lib/python3.12/site-packages")
SPARK_RUNTIME_ROOT = Path("/opt/spark-vllm")
OUTPUT = PREFIX / "runtime-manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_manifest_bytes(manifest: dict) -> bytes:
    stripped = {key: value for key, value in manifest.items() if key != "self_hash"}
    return json.dumps(
        stripped, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def package_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(SITE_PACKAGES).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def compose_spark_runtime_overlay(manifest: dict, pins: dict) -> None:
    private_component_name = "public-exl3-runtime-overlay"
    if any(
        component.get("name") == private_component_name
        for component in manifest["components"]
    ):
        raise RuntimeError(
            f"inherited receipt already contains {private_component_name}"
        )
    components = [
        component
        for component in manifest["components"]
        if component["name"] == "public-overlay"
    ]
    if len(components) != 1:
        raise RuntimeError(
            "inherited receipt must contain exactly one public-overlay component"
        )
    verification = components[0].get("verification")
    if (
        not isinstance(verification, dict)
        or verification.get("mode") != "tree"
        or verification.get("root") != str(SPARK_RUNTIME_ROOT)
        or not isinstance(verification.get("files"), dict)
    ):
        raise RuntimeError("inherited public-overlay receipt contract drift")
    receipt_files = verification["files"]
    private_receipt_files: dict[str, dict[str, str]] = {}
    for relative, record in pins["spark_runtime_overlay_files"].items():
        inherited = receipt_files.get(relative)
        if inherited != record["preimage"]:
            raise RuntimeError(
                f"inherited Spark runtime receipt drift for {relative}: "
                f"{inherited} != {record['preimage']}"
            )
        installed = SPARK_RUNTIME_ROOT / relative
        if not installed.is_file():
            raise RuntimeError(
                f"Spark runtime overlay file is missing: {installed}"
            )
        observed = sha256(installed)
        if observed != record["postimage"]:
            raise RuntimeError(
                f"Spark runtime overlay hash mismatch for {relative}: "
                f"{observed} != {record['postimage']}"
            )
        receipt_files[relative] = record["postimage"]
        private_receipt_files[relative] = {
            "preimage": record["preimage"],
            "postimage": record["postimage"],
        }
    if not private_receipt_files:
        raise RuntimeError("public EXL3 Spark runtime overlay receipt is empty")
    manifest["components"].append(
        {
            "name": private_component_name,
            "verification": {
                "mode": "patched_files",
                "root": str(SPARK_RUNTIME_ROOT),
                "files": private_receipt_files,
            },
        }
    )


def main() -> int:
    manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    pins = json.loads((PREFIX / "pins.json").read_text(encoding="utf-8"))
    compose_spark_runtime_overlay(manifest, pins)
    patch_component = next(
        component
        for component in manifest["components"]
        if component["name"] == "patch-overlay"
    )
    patch_files = patch_component["verification"]["files"]
    for relative, postimage in pins["overlay_files"].items():
        existing = patch_files.get(relative)
        inherited = pins["installed_base_files"].get(relative)
        if existing is not None:
            if inherited is None:
                raise RuntimeError(
                    f"overlay changes receipt-owned file without inherited pin: {relative}"
                )
            if existing["postimage"] != inherited:
                raise RuntimeError(
                    f"inherited receipt drift for {relative}: "
                    f"{existing['postimage']} != {inherited}"
                )
            existing["preimage"] = inherited
            existing["postimage"] = postimage
        else:
            patch_files[relative] = {
                "preimage": inherited,
                "postimage": postimage,
            }
    for relative, record in pins["profile_patches"].items():
        existing = patch_files.get(relative)
        if existing is not None and existing["postimage"] != record["preimage"]:
            raise RuntimeError(
                f"profile patch preimage drift for {relative}: "
                f"{existing['postimage']} != {record['preimage']}"
            )
        patch_files[relative] = dict(record)

    package_map: dict[str, str] = {}
    for package in ("sparkinfer", "exllamav3"):
        root = SITE_PACKAGES / package
        if not root.is_dir():
            raise RuntimeError(f"installed package directory is missing: {root}")
        package_map.update(package_files(root))
    extensions = sorted(SITE_PACKAGES.glob("exllamav3_ext*.so"))
    if len(extensions) != 1:
        raise RuntimeError(
            f"expected one exllamav3_ext shared object, got {len(extensions)}"
        )
    extension = extensions[0]
    package_map[extension.relative_to(SITE_PACKAGES).as_posix()] = sha256(extension)
    manifest["components"].append(
        {
            "name": "public-exl3-packages",
            "verification": {
                "mode": "tree",
                "root": str(SITE_PACKAGES),
                "files": package_map,
            },
        }
    )
    manifest["native_libs"].append(
        {"path": str(extension), "sha256": sha256(extension)}
    )

    manifest["packages"].update(
        {
            "exllamav3": importlib.metadata.version("exllamav3"),
            "nvidia-cutlass-dsl": importlib.metadata.version(
                "nvidia-cutlass-dsl"
            ),
            "sparkinfer": importlib.metadata.version("sparkinfer"),
        }
    )
    manifest["runtime_id"] = pins["profile_id"]
    manifest["lock"]["model"] = {
        "repository": pins["model"]["repository"],
        "revision": pins["model"]["revision"],
        "config_sha256": pins["model"]["config_sha256"],
        "index_sha256": pins["model"]["index_sha256"],
        "tier_bitmap_sha256": pins["model"]["tier_bitmap_sha256"],
        "manifest_sha256": pins["model"]["manifest_sha256"],
    }
    manifest["lock"]["status"] = "public-exl3-bootstrap-candidate"
    manifest["lock"]["notes"].append(
        "Public EXL3 bootstrap candidate: inherited receipt-gated NF3/SparkRing "
        "bytes plus hash-pinned SM121 SparkInfer/ExLlamaV3 source composition."
    )
    manifest["self_hash"] = hashlib.sha256(
        canonical_manifest_bytes(manifest)
    ).hexdigest()
    OUTPUT.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest["self_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
