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
COMPUTE = Path("/opt/sparkring-compute")
COMPUTE_RECEIPT = Path("/opt/sparkring/receipts/glm53-compute-installed.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_map_sha256(records: dict) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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
        candidate = root.joinpath(*path.parts)
        if not candidate.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Source file escapes manifest root: {relative}")
        check_file(candidate, expected)
    return len(records)


def verify_layered_file_map(root: Path, parent_records: dict,
                            overrides: dict) -> dict:
    """Verify every parent file, substituting only manifest-bound overrides."""
    if not isinstance(parent_records, dict) or not parent_records:
        raise ValueError("Parent source file manifest is empty")
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError("Compute override manifest is empty")
    expected = dict(parent_records)
    for relative, record in overrides.items():
        if relative not in parent_records:
            raise ValueError(f"Compute override is absent from the parent manifest: {relative}")
        if not isinstance(record, dict):
            raise ValueError(f"Compute override record is invalid: {relative}")
        if record.get("base_sha256") != parent_records[relative]:
            raise ValueError(f"Compute override base identity differs from the parent: {relative}")
        result = record.get("result_sha256")
        if not isinstance(result, str) or len(result) != 64:
            raise ValueError(f"Compute override result identity is invalid: {relative}")
        expected[relative] = result
    verify_file_map(root, expected)
    return {"parent_files": len(parent_records), "overrides": len(overrides)}


def verify_complete_package_file_map(root: Path, package: str,
                                     records: dict) -> int:
    """Verify all installed package files and reject inherited stale files."""
    prefix = f"{package}/"
    if not isinstance(records, dict) or not records:
        raise ValueError(f"{package} source manifest is empty")
    if any(not relative.startswith(prefix) for relative in records):
        raise ValueError(f"{package} source manifest contains an invalid path")
    verify_file_map(root, records)
    package_root = root / package
    if package_root.is_symlink() or any(path.is_symlink() for path in package_root.rglob("*")):
        raise ValueError(f"{package} contains an unmanifested symbolic link")
    observed = {
        path.relative_to(root).as_posix()
        for path in (root / package).rglob("*")
        if (path.is_file() and not path.is_symlink()
            and "__pycache__" not in path.parts and path.suffix != ".pyc")
    }
    if observed != set(records):
        missing = sorted(set(records) - observed)
        extra = sorted(observed - set(records))
        raise ValueError(
            f"{package} source set differs from its complete manifest; "
            f"missing={missing}, extra={extra}"
        )
    return len(records)


def verify_required_environment(environment: dict, expected: dict) -> dict:
    """Require exact construction-time values for compute-selection settings."""
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Required compute environment is empty")
    for name, value in expected.items():
        if environment.get(name) != value:
            raise ValueError(
                f"Mesh compute environment differs from its required value: {name}"
            )
    return dict(sorted(expected.items()))


def verify_compute(profile: dict, base: dict, source: dict, environment: dict,
                   *, compute_root: Path = COMPUTE,
                   receipt_path: Path = COMPUTE_RECEIPT,
                   site: Path = SITE,
                   base_receipts: Path = BASE_RECEIPTS,
                   cuda_root_override: Path | None = None,
                   ptxas_runner=subprocess.run) -> dict:
    """Verify the manifest-bound vLLM, B12X, and CUDA compute composition."""
    pin = profile.get("compute")
    required_pin_fields = {
        "source_lock", "source_lock_sha256", "vllm_base_revision",
        "b12x_revision", "b12x_tree", "cuda_version",
    }
    if not isinstance(pin, dict) or not required_pin_fields.issubset(pin):
        raise ValueError("Mesh profile does not bind the required compute source")
    if pin["source_lock"] != "compute/source-lock.json":
        raise ValueError("Mesh profile compute source-lock locator is unsupported")
    lock_path = compute_root / "source-lock.json"
    check_file(lock_path, pin["source_lock_sha256"])
    if source["files"].get("compute/source-lock.json") != pin["source_lock_sha256"]:
        raise ValueError("Image source receipt does not bind the compute source lock")
    lock = load(lock_path)
    if lock.get("schema") != "sparkring-glm53-compute-source/v1":
        raise ValueError("Compute source lock uses an unsupported schema")
    if lock["vllm"]["base_revision"] != pin["vllm_base_revision"]:
        raise ValueError("Compute vLLM base revision differs from the profile pin")
    if lock["b12x"]["revision"] != pin["b12x_revision"]:
        raise ValueError("Compute B12X revision differs from the profile pin")
    if lock["b12x"]["tree"] != pin["b12x_tree"]:
        raise ValueError("Compute B12X tree differs from the profile pin")
    if lock["cuda"]["version"] != pin["cuda_version"]:
        raise ValueError("Compute CUDA version differs from the profile pin")

    installed = load(receipt_path)
    if installed.get("schema") != "sparkring-glm53-compute-installed/v1":
        raise ValueError("Installed compute receipt uses an unsupported schema")
    if installed.get("source_lock_sha256") != pin["source_lock_sha256"]:
        raise ValueError("Installed compute receipt uses a different source lock")
    if installed.get("vllm_revision") != lock["vllm"]["base_revision"]:
        raise ValueError("Installed vLLM base revision differs from the source lock")
    if (installed.get("b12x_revision") != lock["b12x"]["revision"]
            or installed.get("b12x_tree") != lock["b12x"]["tree"]):
        raise ValueError("Installed B12X identity differs from the source lock")

    parent_vllm = load(base_receipts / "vllm-source-manifest.json")
    if parent_vllm.get("commit") != base["vllm"]["commit"]:
        raise ValueError("Parent source receipt has the wrong vllm revision")
    if base["vllm"]["commit"] != lock["vllm"]["base_revision"]:
        raise ValueError("Compute source does not extend the pinned parent vLLM")
    overrides = {
        relative: {"base_sha256": parent_hash, "result_sha256": result_hash}
        for relative, parent_hash, result_hash in lock["vllm"]["files"]
    }
    expected_results = {
        relative: record["result_sha256"] for relative, record in overrides.items()
    }
    if installed.get("vllm_overrides") != expected_results:
        raise ValueError("Installed vLLM override map differs from the source lock")
    vllm = verify_layered_file_map(site, parent_vllm["files"], overrides)

    b12x_prefix = "compute/b12x-source/"
    expected_b12x_files = {
        relative.removeprefix(b12x_prefix): expected
        for relative, expected in source["files"].items()
        if relative.startswith(b12x_prefix + "b12x/")
    }
    b12x_files = installed.get("b12x_files")
    if b12x_files != expected_b12x_files:
        raise ValueError(
            "Installed B12X file map differs from the source-receipt-bound source"
        )
    if file_map_sha256(b12x_files) != lock["b12x"]["package_files_sha256"]:
        raise ValueError("Installed B12X file-map identity differs from the source lock")
    b12x_count = verify_complete_package_file_map(site, "b12x", b12x_files)
    if installed.get("cuda_components") != lock["cuda"]["components"]:
        raise ValueError("Installed CUDA component map differs from the source lock")
    cuda_root = cuda_root_override or Path(f"/opt/cuda-{lock['cuda']['version']}")
    cuda_manifest = load(cuda_root / "sparkring-component-manifest.json")
    if cuda_manifest != lock["cuda"]["components"]:
        raise ValueError("CUDA component manifest differs from the source lock")
    ptxas = cuda_root / "bin/ptxas"
    if ptxas.is_symlink() or not ptxas.is_file():
        raise ValueError("Pinned CUDA toolkit does not contain ptxas")
    version_result = ptxas_runner(
        [str(ptxas), "--version"], capture_output=True, text=True, check=True
    )
    version_text = version_result.stdout + version_result.stderr
    if f"release {lock['cuda']['version']}" not in version_text:
        raise ValueError("Installed ptxas version differs from the CUDA source lock")
    expected_environment = {
        **lock["environment"],
        "CUDA_HOME": f"/opt/cuda-{lock['cuda']['version']}",
        "TRITON_PTXAS_PATH": f"/opt/cuda-{lock['cuda']['version']}/bin/ptxas",
    }
    if installed.get("environment") != lock["environment"]:
        raise ValueError("Installed compute environment differs from the source lock")
    verified_environment = verify_required_environment(
        environment, expected_environment
    )
    if installed.get("target_head_quantization") is not False:
        raise ValueError("Target LM head must remain unquantized")
    return {
        "source_lock_sha256": pin["source_lock_sha256"],
        "vllm_parent_files": vllm["parent_files"],
        "vllm_overrides": vllm["overrides"],
        "b12x_revision": lock["b12x"]["revision"],
        "b12x_tree": lock["b12x"]["tree"],
        "b12x_files": b12x_count,
        "cuda_version": lock["cuda"]["version"],
        "environment": verified_environment,
        # Source hashes bind the MTP constructor and verified environment sets
        # VLLM_MTP_NVFP4_LM_HEAD=1. This reports selected construction policy,
        # not inspection of loaded model tensors (no model is loaded here).
        "proposal_head_nvfp4": True,
        "target_head_quantization": False,
    }


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
    compute = verify_compute(profile, base, source, os.environ)
    sparkcache_manifest = load(BASE_RECEIPTS / "sparkcache-source-manifest.json")
    if sparkcache_manifest.get("commit") != base["sparkcache"]["commit"]:
        raise ValueError("Parent source receipt has the wrong sparkcache revision")
    package_counts = {
        "vllm": compute["vllm_parent_files"],
        "b12x": compute["b12x_files"],
        "sparkcache": verify_file_map(SITE, sparkcache_manifest["files"]),
    }
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
        "vllm_commit": base["vllm"]["commit"],
        "b12x_commit": compute["b12x_revision"],
        "sparkcache_commit": base["sparkcache"]["commit"],
        "sircl_native_sha256": base["sircl"]["native_sha256"],
        "marker_source_sha256": sha256(marker_source), "marker_binary_sha256": sha256(marker),
        "rocenante_lazy_import": str(roce.__file__), "cuda_initialized": False,
        "device_access": False, "model_loaded": False,
        "compute": compute,
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
