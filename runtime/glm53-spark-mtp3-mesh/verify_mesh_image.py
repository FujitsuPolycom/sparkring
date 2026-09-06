#!/usr/bin/env python3
"""Verify embedded mesh artifacts without starting a model or accessing devices."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

RECEIPTS = Path("/opt/sparkring/receipts/glm53-spark-mtp3-mesh")
BASE_RECEIPTS = Path("/opt/sparkring/receipts/jj-r8-sparkcache-arm64")
SITE = Path("/usr/local/lib/python3.12/dist-packages")
BUNDLE = Path("/opt/spark-sircl")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def check_file(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256(path) != expected:
        raise ValueError(f"Image file differs from its content pin: {path}")


def verify_file_map(root: Path, records: dict) -> int:
    if not isinstance(records, dict) or not records:
        raise ValueError("Source file manifest is empty")
    for relative, expected in records.items():
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
            raise ValueError(f"Unsafe source file path: {relative}")
        check_file(root.joinpath(*path.parts), expected)
    return len(records)


def verify_warmup(path: Path, expected: str, environment: dict) -> dict:
    """Check the readiness-only helper override and its explicit temperature."""
    check_file(path, expected)
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if environment.get("SPARKRING_WARMUP_TEMPERATURE") != "1":
        raise ValueError("Mesh image readiness warmup temperature must be one")
    return {"helper_sha256": expected, "temperature": 1.0,
            "environment": "SPARKRING_WARMUP_TEMPERATURE"}


def verify_inside_image() -> dict:
    """Check package source identities, overlay imports, and marker linkage."""
    profile = load(RECEIPTS / "profile-pins.json")
    base = load(RECEIPTS / "parent-pins.json")
    source = load(RECEIPTS / "source-receipt.json")
    check_file(BUNDLE / "sparkring-overlay-manifest.json", profile["canonical_bundle_manifest_sha256"])
    bundle = load(BUNDLE / "sparkring-overlay-manifest.json")
    bundle_files = {record["path"]: record["sha256"] for record in bundle["files"]}
    verified_bundle = verify_file_map(BUNDLE, bundle_files)
    python_files = 0
    for relative in bundle_files:
        if relative.endswith(".py"):
            ast.parse((BUNDLE / relative).read_text(encoding="utf-8"), filename=relative)
            python_files += 1
    for relative, expected in source["files"].items():
        if relative.startswith("receipts/"):
            check_file(RECEIPTS / relative.removeprefix("receipts/"), expected)
    check_file(Path("/opt/sparkring/bin/verify-mtp3-mesh-image.py"), source["files"]["verify_mesh_image.py"])
    marker_source = Path("/opt/sparkring/src/mtp3-mesh/mlx5_rdma_tx_rewrite_probe.c")
    check_file(marker_source, profile["marker"]["source_sha256"])
    warmup = verify_warmup(Path("/opt/sparkring/bin/warmup_dflash.py"),
                           source["files"]["warmup_dflash.py"], os.environ)
    package_counts = {}
    for package in ("vllm", "b12x", "sparkcache"):
        manifest = load(BASE_RECEIPTS / f"{package}-source-manifest.json")
        if manifest.get("commit") != base[package]["commit"]:
            raise ValueError(f"Parent source receipt has the wrong {package} revision")
        package_counts[package] = verify_file_map(SITE, manifest["files"])
    native = load(BASE_RECEIPTS / "native-extension-manifest.json")["files"]
    native_count = verify_file_map(SITE / "vllm", native)
    check_file(BUNDLE / "libspark_transport_capi.so", base["sircl"]["native_sha256"])
    check_file(Path("/opt/sparkring/nccl/libnccl.so.2.30.7"), base["transport"]["nccl_sha256"])
    for name in ("placement", "snapshot"):
        check_file(Path(f"/opt/sparkcache-src/sparkcache/native/build-cuda/libspark_cache_{name}.so"),
                   base["sparkcache"][f"cuda_{name}_sha256"])
    for package in ("vllm", "b12x", "fastsafetensors", "torch"):
        if importlib.util.find_spec(package) is None:
            raise ValueError(f"Required Python package cannot be resolved: {package}")
    # Import only the lazy communication API; do not instantiate a transport.
    torch = importlib.import_module("torch")
    comm = importlib.import_module("b12x.comm")
    comm.__path__.insert(0, str(BUNDLE / "b12x_overlay/b12x/comm"))
    roce = importlib.import_module("b12x.comm.roce")
    check_file(Path(roce.__file__), bundle_files["b12x_overlay/b12x/comm/roce/__init__.py"])
    if torch.cuda.is_initialized():
        raise ValueError("CPU-only image verification unexpectedly initialized CUDA")
    marker = Path("/opt/sparkring/bin/mlx5-rdma-tx-marker")
    linked = subprocess.run([str(marker), "--help"], capture_output=True, text=True, check=True)
    if "--device" not in linked.stdout + linked.stderr:
        raise ValueError("Marker helper did not report its device-scoped interface")
    return {
        "status": "research-only", "checks_passed": True,
        "bundle_manifest_sha256": profile["canonical_bundle_manifest_sha256"],
        "source_receipt_sha256": sha256(RECEIPTS / "source-receipt.json"),
        "bundle_files": verified_bundle, "python_syntax_files": python_files,
        "parent_package_files": package_counts, "vllm_native_extensions": native_count,
        "vllm_commit": base["vllm"]["commit"], "b12x_commit": base["b12x"]["commit"],
        "sparkcache_commit": base["sparkcache"]["commit"],
        "sircl_native_sha256": base["sircl"]["native_sha256"],
        "marker_source_sha256": sha256(marker_source), "marker_binary_sha256": sha256(marker),
        "rocenante_lazy_import": str(roce.__file__), "cuda_initialized": False,
        "device_access": False, "model_loaded": False,
        "readiness_warmup": warmup,
        "limitation": "Content and CPU checks do not qualify CUDA graphs, RDMA forwarding, native MTP, cache restoration, or model performance.",
    }


def inspect_image(engine: str, image: str) -> dict:
    result = subprocess.run([engine, "image", "inspect", image], check=True, capture_output=True, text=True)
    documents = json.loads(result.stdout)
    if len(documents) != 1:
        raise ValueError("Container engine returned an unexpected image inspection")
    return documents[0]


def verify_external(image: str, engine: str, parent_id: str, source_sha: str, bundle_sha: str) -> dict:
    document = inspect_image(engine, image)
    parent = inspect_image(engine, parent_id)
    if document.get("Architecture") != "arm64" or document.get("Os") != "linux":
        raise ValueError("Mesh image must use linux/arm64")
    if parent["Id"] != parent_id:
        raise ValueError("Parent image ID differs from the expected identity")
    parent_layers = parent["RootFS"]["Layers"]
    if document["RootFS"]["Layers"][:len(parent_layers)] != parent_layers:
        raise ValueError("Mesh image does not retain the complete parent layer prefix")
    labels = document["Config"].get("Labels") or {}
    environment = dict(item.split("=", 1) for item in document["Config"].get("Env", []))
    if environment.get("SPARKRING_WARMUP_TEMPERATURE") != "1":
        raise ValueError("Mesh image readiness warmup temperature must be one")
    expected_labels = {
        "org.sparkring.runtime.status": "research-only",
        "org.sparkring.mesh.parent-image": parent_id,
        "org.sparkring.mesh.bundle-manifest-sha256": bundle_sha,
        "org.sparkring.mesh.source-receipt-sha256": source_sha,
    }
    for name, expected in expected_labels.items():
        if labels.get(name) != expected:
            raise ValueError(f"Mesh image label differs from its expected value: {name}")
    argv = [engine, "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--cpus", "2", "--memory", "2g", "--pids-limit", "128",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=128m", "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--entrypoint", "python3", document["Id"], "-I", "/opt/sparkring/bin/verify-mtp3-mesh-image.py", "--inside-image"]
    completed = subprocess.run(argv, check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout)
    if result["source_receipt_sha256"] != source_sha or result["bundle_manifest_sha256"] != bundle_sha:
        raise ValueError("Embedded artifact identities differ from the expected source receipt")
    return {
        "schema": "sparkring-mtp3-mesh-image-receipt/v1", "status": "research-only",
        "checks_passed": True, "image": image, "image_id": document["Id"],
        "image_reference": document["Id"], "platform": "linux/arm64",
        "image_size_bytes": document["Size"], "parent_image_id": parent_id,
        "parent_layers_retained": len(parent_layers),
        "added_layers": len(document["RootFS"]["Layers"]) - len(parent_layers),
        "source_receipt_sha256": source_sha, "bundle_manifest_sha256": bundle_sha,
        "inside_image": result,
        "verification_command": argv,
        "limitation": "No full-model, GPU, fabric, or four-rank serving test was performed for this image.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside-image", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--engine", default="docker")
    parser.add_argument("--expected-parent")
    parser.add_argument("--expected-source")
    parser.add_argument("--expected-bundle")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.inside_image == bool(args.image):
        parser.error("Choose exactly one of --inside-image or --image")
    if args.inside_image:
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        sys.dont_write_bytecode = True
        result = verify_inside_image()
    else:
        if not all((args.expected_parent, args.expected_source, args.expected_bundle)):
            parser.error("External verification requires --expected-parent, --expected-source, and --expected-bundle")
        result = verify_external(args.image, args.engine, args.expected_parent, args.expected_source, args.expected_bundle)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
