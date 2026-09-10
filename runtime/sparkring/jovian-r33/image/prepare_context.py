#!/usr/bin/env python3
"""Prepare a fail-closed R33 candidate context from verified local artifacts."""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from validate_receipts import validate as validate_receipts


HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def copy_checked(source: Path, destination: Path, expected: str) -> dict:
    if not source.is_file():
        raise RuntimeError(f"required input is missing: {source}")
    actual = digest(source)
    if actual != expected:
        raise RuntimeError(f"input hash mismatch: {source}: {actual} != {expected}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if digest(destination) != expected:
        raise RuntimeError(f"copied input changed: {destination}")
    return {"source": str(source), "destination": destination.as_posix(), "sha256": expected, "size": destination.stat().st_size}


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise RuntimeError(f"required asset tree is missing: {source}")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", type=Path, default=Path("/var/tmp/sparkring-r33-20260910"))
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_context = args.context.resolve()
    if requested_context.exists():
        raise RuntimeError(f"context must be a new path: {requested_context}")
    lock = json.loads((HERE / "artifact-lock.json").read_text())
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", lock["foundation"]["reference"]],
        text=True,
    ).strip()
    if image_id != lock["foundation"]["image_id"]:
        raise RuntimeError(f"foundation image mismatch: {image_id}")
    media_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", lock["media_runtime"]["reference"]],
        text=True,
    ).strip()
    if media_id != lock["media_runtime"]["image_id"]:
        raise RuntimeError(f"media runtime image mismatch: {media_id}")
    foundation_layers = json.loads(subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{json .RootFS.Layers}}", lock["foundation"]["reference"]],
        text=True,
    ))
    media_layers = json.loads(subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{json .RootFS.Layers}}", lock["media_runtime"]["reference"]],
        text=True,
    ))
    if media_layers[:len(foundation_layers)] != foundation_layers:
        raise RuntimeError("content-addressed media runtime is not based on the locked foundation")

    runtime = args.repository_root / "runtime"
    required_files = [
        *(HERE / name for name in ("Dockerfile.candidate", "entrypoint.py", "verify_candidate.py", "verify_context.py", "validate_receipts.py", "capture_installed.py", "finalize_lock.py", "download_report.py")),
        runtime / "sparkring/jovian-r33/contracts/vllm-connector-jobs-r33-prefill-a2ad36d.json",
        args.build_root / "artifacts/vllm-package/verification.json",
        args.build_root / "artifacts/vllm-package/import-smoke-v2.json",
        args.build_root / "artifacts/flashinfer/SHA256SUMS",
        args.build_root / "artifacts/flashinfer/source-receipt.txt",
        args.build_root / "artifacts/flashinfer/resume-source-receipt.json",
        args.build_root / "artifacts/b12x/source-receipt.txt",
        args.build_root / "artifacts/lmcache/source-receipt.txt",
        args.build_root / "artifacts/instanttensor/source-receipt.txt",
        args.build_root / "artifacts/xgrammar-r33/source-receipt.txt",
        args.build_root / "artifacts/xgrammar-r33/xgrammar-transformers5.patch",
        args.build_root / "artifacts/xgrammar-r33/wheel-verification.json",
        args.build_root / "artifacts/sparkcache/source-receipt.txt",
        args.build_root / "artifacts/torchvision-r33-v2/source-receipt.txt",
        args.build_root / "artifacts/torchaudio/source-receipt.txt",
        args.build_root / "artifacts/rust/source-receipt.txt",
        args.build_root / "artifacts/vllm-native/source-receipt.txt",
        args.build_root / "artifacts/vllm-native/all-source-receipt.txt",
        args.build_root / "artifacts/vllm-native/flashkda-source-identity.json",
        args.build_root / "artifacts/arm-wheels/SHA256SUMS",
        args.build_root / "artifacts/torch/SHA256SUMS",
        args.build_root / "artifacts/b12x/SHA256SUMS",
        args.build_root / "artifacts/lmcache/SHA256SUMS",
        args.build_root / "artifacts/instanttensor/SHA256SUMS",
        args.build_root / "artifacts/sparkcache/SHA256SUMS",
        args.build_root / "artifacts/torchaudio/SHA256SUMS",
        args.build_root / "artifacts/torchvision-r33-v2/SHA256SUMS",
        args.build_root / "artifacts/xgrammar-r33/SHA256SUMS",
        args.build_root / "artifacts/rust/SHA256SUMS",
        args.build_root / "artifacts/vllm-native/SHA256SUMS",
        args.build_root / "artifacts/vllm-native/ALL-SHA256SUMS",
        args.build_root / "artifacts/vllm-package/SHA256SUMS",
        args.build_root / "artifacts/post-build-source-identities.json",
        args.build_root / "nccl-port-a69a4376/arm-port-receipt.json",
        args.build_root / "build/sircl-cu133-sm121/build-receipt.json",
        args.build_root / "artifacts/foundation-image.json",
    ]
    required_directories = [
        runtime / "sparkring/jovian-r33/profiles",
        runtime / "transport_profiles",
        runtime / "glm53-spark-mtp3-mesh/performance/transport/bundle-source",
        runtime / "glm53-spark-mtp3-mesh",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    missing.extend(str(path) for path in required_directories if not path.is_dir())
    if missing:
        raise RuntimeError(f"required context assets are missing: {missing}")
    for item in lock["artifacts"]:
        source = args.build_root / item["source"]
        if not source.is_file() or digest(source) != item["sha256"]:
            raise RuntimeError(f"required input missing or changed: {source}")
    flashinfer_sums = {}
    for line in (args.build_root / "artifacts/flashinfer/SHA256SUMS").read_text().splitlines():
        expected, filename = line.split(maxsplit=1)
        flashinfer_sums[Path(filename).name] = expected
    pending_matches = {}
    for pending in lock["pending_artifacts"]:
        matches = sorted(args.build_root.glob(pending["source_glob"]))
        if len(matches) != 1:
            raise RuntimeError(f"required pending input must resolve exactly once: {pending['source_glob']}: {matches}")
        if flashinfer_sums.get(matches[0].name) != digest(matches[0]):
            raise RuntimeError(f"pending input is absent from the completed FlashInfer receipt: {matches[0]}")
        pending_matches[pending["name"]] = matches[0]
    pending_native = {}
    for item in lock["pending_native_artifacts"]:
        if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise RuntimeError(f"native artifact hash is pending: {item['name']}")
        source = args.build_root / item["source"]
        if not source.is_file() or digest(source) != item["sha256"]:
            raise RuntimeError(f"required native input missing or changed: {source}")
        pending_native[item["name"]] = source
    if not (args.build_root / "artifacts/sparkcache-native/build-receipt.json").is_file():
        raise RuntimeError("SparkCache native build receipt is pending")

    requested_context.parent.mkdir(parents=True, exist_ok=True)
    context = Path(tempfile.mkdtemp(prefix=requested_context.name + ".preparing-", dir=requested_context.parent))
    cleanup_state = {"promoted": False}

    def cleanup_staging() -> None:
        if not cleanup_state["promoted"] and context.is_dir():
            shutil.rmtree(context)

    atexit.register(cleanup_staging)

    copied = {}
    install_wheels = []
    for item in lock["artifacts"]:
        source = args.build_root / item["source"]
        destination = context / item["destination"]
        copied[item["name"]] = copy_checked(source, destination, item["sha256"])
        if destination.suffix == ".whl":
            install_wheels.append(destination.name)

    for pending in lock["pending_artifacts"]:
        source = pending_matches[pending["name"]]
        destination = context / pending["destination_directory"] / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied[pending["name"]] = {
            "source": str(source),
            "destination": destination.relative_to(context).as_posix(),
            "sha256": digest(destination),
            "size": destination.stat().st_size,
        }
        if pending["install"]:
            install_wheels.append(destination.name)
    for item in lock["pending_native_artifacts"]:
        destination = context / item["destination"]
        copied[item["name"]] = copy_checked(pending_native[item["name"]], destination, item["sha256"])

    for forbidden in lock["forbidden_inputs"]:
        forbidden_name = (args.build_root / forbidden).name
        if forbidden_name in install_wheels:
            raise RuntimeError(f"forbidden input selected: {forbidden_name}")
    if not any(item["destination"].split("/")[-1].startswith("torchaudio-2.11.0+cu133-") for item in lock["artifacts"]):
        raise RuntimeError("CUDA 13.3 TorchAudio artifact was not selected")

    for name in ("Dockerfile.candidate", "entrypoint.py", "verify_candidate.py", "verify_context.py", "validate_receipts.py", "capture_installed.py", "finalize_lock.py", "download_report.py"):
        shutil.copy2(HERE / name, context / name)
    copy_tree(runtime / "sparkring/jovian-r33/profiles", context / "profile-contract")
    copy_tree(runtime / "transport_profiles", context / "profile-assets/transport_profiles")
    (context / "profile-assets/glm53-spark-mtp3-mesh").mkdir(parents=True)
    shutil.copy2(runtime / "glm53-spark-mtp3-mesh/pins.json", context / "profile-assets/glm53-spark-mtp3-mesh/pins.json")

    copy_tree(runtime / "transport_profiles", context / "transports")
    copy_tree(
        runtime / "glm53-spark-mtp3-mesh/performance/transport/bundle-source",
        context / "sircl-python",
    )
    contract = runtime / "sparkring/jovian-r33/contracts/vllm-connector-jobs-r33-prefill-a2ad36d.json"
    (context / "contracts").mkdir()
    shutil.copy2(contract, context / "contracts" / contract.name)

    receipts = context / "receipts"
    receipts.mkdir()
    receipt_inputs = {
        "artifact-lock.json": HERE / "artifact-lock.json",
        "flashinfer-SHA256SUMS": args.build_root / "artifacts/flashinfer/SHA256SUMS",
        "flashinfer-source-receipt.txt": args.build_root / "artifacts/flashinfer/source-receipt.txt",
        "flashinfer-resume-source-receipt.json": args.build_root / "artifacts/flashinfer/resume-source-receipt.json",
        "b12x-source-receipt.txt": args.build_root / "artifacts/b12x/source-receipt.txt",
        "lmcache-source-receipt.txt": args.build_root / "artifacts/lmcache/source-receipt.txt",
        "instanttensor-source-receipt.txt": args.build_root / "artifacts/instanttensor/source-receipt.txt",
        "xgrammar-source-receipt.txt": args.build_root / "artifacts/xgrammar-r33/source-receipt.txt",
        "xgrammar-transformers5.patch": args.build_root / "artifacts/xgrammar-r33/xgrammar-transformers5.patch",
        "xgrammar-wheel-verification.json": args.build_root / "artifacts/xgrammar-r33/wheel-verification.json",
        "sparkcache-source-receipt.txt": args.build_root / "artifacts/sparkcache/source-receipt.txt",
        "sparkcache-native-build-receipt.json": args.build_root / "artifacts/sparkcache-native/build-receipt.json",
        "torchvision-source-receipt.txt": args.build_root / "artifacts/torchvision-r33-v2/source-receipt.txt",
        "torchaudio-source-receipt.txt": args.build_root / "artifacts/torchaudio/source-receipt.txt",
        "rust-source-receipt.txt": args.build_root / "artifacts/rust/source-receipt.txt",
        "vllm-native-source-receipt.txt": args.build_root / "artifacts/vllm-native/source-receipt.txt",
        "vllm-all-native-source-receipt.txt": args.build_root / "artifacts/vllm-native/all-source-receipt.txt",
        "flashkda-source-identity.json": args.build_root / "artifacts/vllm-native/flashkda-source-identity.json",
        "arm-wheels-SHA256SUMS": args.build_root / "artifacts/arm-wheels/SHA256SUMS",
        "torch-SHA256SUMS": args.build_root / "artifacts/torch/SHA256SUMS",
        "b12x-SHA256SUMS": args.build_root / "artifacts/b12x/SHA256SUMS",
        "lmcache-SHA256SUMS": args.build_root / "artifacts/lmcache/SHA256SUMS",
        "instanttensor-SHA256SUMS": args.build_root / "artifacts/instanttensor/SHA256SUMS",
        "sparkcache-SHA256SUMS": args.build_root / "artifacts/sparkcache/SHA256SUMS",
        "torchaudio-SHA256SUMS": args.build_root / "artifacts/torchaudio/SHA256SUMS",
        "torchvision-SHA256SUMS": args.build_root / "artifacts/torchvision-r33-v2/SHA256SUMS",
        "xgrammar-SHA256SUMS": args.build_root / "artifacts/xgrammar-r33/SHA256SUMS",
        "rust-SHA256SUMS": args.build_root / "artifacts/rust/SHA256SUMS",
        "vllm-native-SHA256SUMS": args.build_root / "artifacts/vllm-native/SHA256SUMS",
        "vllm-native-ALL-SHA256SUMS": args.build_root / "artifacts/vllm-native/ALL-SHA256SUMS",
        "vllm-package-SHA256SUMS": args.build_root / "artifacts/vllm-package/SHA256SUMS",
        "post-build-source-identities.json": args.build_root / "artifacts/post-build-source-identities.json",
        "vllm-package-verification.json": args.build_root / "artifacts/vllm-package/verification.json",
        "vllm-package-import-smoke.json": args.build_root / "artifacts/vllm-package/import-smoke-v2.json",
        "nccl-arm-port-receipt.json": args.build_root / "nccl-port-a69a4376/arm-port-receipt.json",
        "sircl-build-receipt.json": args.build_root / "build/sircl-cu133-sm121/build-receipt.json",
        "foundation-image.json": args.build_root / "artifacts/foundation-image.json",
    }
    for name, source in receipt_inputs.items():
        if not source.is_file():
            raise RuntimeError(f"required receipt is missing: {source}")
        shutil.copy2(source, receipts / name)

    (context / "install-wheels.txt").write_text("\n".join(install_wheels) + "\n")
    partial = {
        "schema": "sparkring-r33-candidate-source-lock-partial/v1",
        "status": "context-prepared-python-closure-pending",
        "foundation": lock["foundation"],
        "media_runtime": lock["media_runtime"],
        "source_identities": lock["source_identities"],
        "inputs": copied,
        "install_wheels": install_wheels,
        "forbidden_inputs": lock["forbidden_inputs"],
    }
    (context / "source-lock.partial.json").write_text(json.dumps(partial, indent=2, sort_keys=True) + "\n")
    semantic = validate_receipts(lock, copied, receipts)
    (receipts / "semantic-validation.json").write_text(json.dumps(semantic, indent=2, sort_keys=True) + "\n")
    context.replace(requested_context)
    cleanup_state["promoted"] = True
    print(json.dumps({"context": str(requested_context), "inputs": len(copied), "install_wheels": len(install_wheels), "status": partial["status"]}, indent=2))


if __name__ == "__main__":
    main()
