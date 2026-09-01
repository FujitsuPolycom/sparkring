#!/usr/bin/env python3
"""Verify the constructed JJ r8 and DCP-aware SparkCache image."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PINS = HERE / "pins.json"
SITE_ROOT = Path("/usr/local/lib/python3.12/dist-packages")
IMAGE_RECEIPTS = Path("/opt/sparkring/receipts/jj-r8-sparkcache-arm64")


class VerificationError(RuntimeError):
    """The image differs from its exact construction contract."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VerificationError(f"expected a JSON object: {path}")
    return value


def verify_manifest(root: Path, path: Path) -> None:
    files = load_json(path).get("files")
    if not isinstance(files, dict) or not files:
        raise VerificationError(f"source manifest is empty: {path}")
    for relative, expected in files.items():
        candidate = root / relative
        if not candidate.is_file() or file_sha256(candidate) != expected:
            raise VerificationError(f"source identity mismatch: {relative}")


def native_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*.so"))
    }


def verify_inside_image() -> dict[str, Any]:
    pins = load_json(IMAGE_RECEIPTS / "pins.json")
    verify_manifest(
        SITE_ROOT, IMAGE_RECEIPTS / "vllm-source-manifest.json"
    )
    verify_manifest(
        SITE_ROOT, IMAGE_RECEIPTS / "sparkcache-source-manifest.json"
    )
    for relative, expected in pins["vllm"]["runtime_file_sha256"].items():
        if file_sha256(SITE_ROOT / relative) != expected:
            raise VerificationError(f"r8 runtime file identity changed: {relative}")
    expected_native = load_json(
        IMAGE_RECEIPTS / "native-extension-manifest.json"
    )["files"]
    observed_native = native_manifest(SITE_ROOT / "vllm")
    if observed_native != expected_native:
        raise VerificationError("retained vLLM native extension identity changed")
    expected_placement = pins["sparkcache"]["cuda_placement_sha256"]
    placement = Path(
        "/opt/sparkcache-src/sparkcache/native/build-cuda/"
        "libspark_cache_placement.so"
    )
    if file_sha256(placement) != expected_placement:
        raise VerificationError("SparkCache CUDA placement identity changed")
    expected_snapshot = pins["sparkcache"]["cuda_snapshot_sha256"]
    snapshot = Path(
        "/opt/sparkcache-src/sparkcache/native/build-cuda/"
        "libspark_cache_snapshot.so"
    )
    if file_sha256(snapshot) != expected_snapshot:
        raise VerificationError("SparkCache CUDA snapshot identity changed")
    expected_nccl = pins["transport"]["nccl_sha256"]
    nccl = Path("/opt/sparkring/nccl/libnccl.so.2.30.7")
    if file_sha256(nccl) != expected_nccl:
        raise VerificationError("switchless NCCL identity changed")
    if importlib.util.find_spec("fastsafetensors") is None:
        raise VerificationError("fastsafetensors is unavailable")
    if tuple(SITE_ROOT.glob("deep_ep*")):
        raise VerificationError("unused DeepEP files are installed")
    return {
        "status": "implemented",
        "vllm_commit": pins["vllm"]["commit"],
        "sparkcache_commit": pins["sparkcache"]["commit"],
        "native_extensions": len(observed_native),
        "nccl_sha256": expected_nccl,
        "cuda_placement_sha256": expected_placement,
        "cuda_snapshot_sha256": expected_snapshot,
        "live_qualification": "not-established-by-image-verification",
    }


def expected_labels(pins: dict[str, Any]) -> dict[str, str]:
    return {
        "org.sparkring.runtime.status": "implemented-live-verification-required",
        "org.sparkring.parent.image": pins["parent"]["image_id"],
        "org.sparkring.vllm.proven-base": pins["vllm"]["proven_base_commit"],
        "org.sparkring.vllm.python-commit": pins["vllm"]["commit"],
        "org.sparkring.vllm.python-tree": pins["vllm"]["tree"],
        "org.sparkring.vllm.sparkcache-composition": pins["vllm"]["commit"],
        "org.sparkring.vllm.tree": pins["vllm"]["tree"],
        "org.sparkring.vllm.community-release": pins["vllm"][
            "community_release"
        ],
        "org.sparkring.vllm.community-parent": pins["vllm"][
            "community_parent_commit"
        ],
        "org.sparkring.vllm.sparse-pooled-index": pins["vllm"][
            "sparse_pooled_index_commit"
        ],
        "org.sparkring.vllm.fwht-scaling": pins["vllm"]["fwht_scaling_commit"],
        "org.sparkring.vllm.prefill-cadence-component": pins["vllm"][
            "scheduler_prefill_cadence_component_commit"
        ],
        "org.sparkring.vllm.prefill-cadence-pr-head": pins["vllm"][
            "scheduler_prefill_cadence_pull_request_head"
        ],
        "org.sparkring.vllm.delta-patch-id": pins["vllm"]["delta_patch_id"],
        "org.sparkring.b12x.composition": pins["b12x"]["commit"],
        "org.sparkring.b12x.tree": pins["b12x"]["tree"],
        "org.sparkring.b12x.package-tree": pins["b12x"]["package_tree"],
        "org.sparkring.nccl.sha256": pins["transport"]["nccl_sha256"],
        "org.sparkring.loader": "fastsafetensors",
        "org.sparkring.diagnostics": "compact-startup-no-deep-ep",
        "org.sparkcache.commit": pins["sparkcache"]["commit"],
        "org.sparkcache.tree": pins["sparkcache"]["tree"],
        "org.sparkcache.source-sha256": pins["sparkcache"][
            "source_tree_sha256"
        ],
        "org.sparkcache.cuda-placement-sha256": pins["sparkcache"][
            "cuda_placement_sha256"
        ],
        "org.sparkcache.cuda-snapshot-sha256": pins["sparkcache"][
            "cuda_snapshot_sha256"
        ],
        "org.sparkcache.startup-inventory": "connector-handshake-all-rank",
        "org.sparkcache.cache-geometry": "manager-pages-v2",
        "org.sparkcache.publication-schema": "snapshot-v1",
        "org.sparkcache.dcp-layouts": "1,2,4",
    }


def verify_external(image: str, engine: str) -> dict[str, Any]:
    inspected = subprocess.run(
        (engine, "image", "inspect", image),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    documents = json.loads(inspected.stdout)
    if len(documents) != 1:
        raise VerificationError("container engine returned an unexpected inspection")
    document = documents[0]
    if document.get("Architecture") != "arm64" or document.get("Os") != "linux":
        raise VerificationError("image platform is not linux/arm64")
    pins = load_json(PINS)
    labels = document.get("Config", {}).get("Labels") or {}
    for name, expected in expected_labels(pins).items():
        if labels.get(name) != expected:
            raise VerificationError(
                f"image label {name} drift: {labels.get(name)!r} != {expected!r}"
            )
    probe = subprocess.run(
        (
            engine,
            "run",
            "--rm",
            "--entrypoint",
            "python3",
            image,
            "/opt/sparkring/bin/verify-jj-r8-sparkcache-image.py",
            "--inside-image",
        ),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    result = json.loads(probe.stdout)
    return {
        "schema": "sparkring-glm53-jj-r8-gb10-image-receipt/v1",
        "status": "implemented",
        "image": image,
        "image_id": document["Id"],
        "platform": "linux/arm64",
        "labels": dict(sorted(labels.items())),
        "inside_image": result,
        "limitation": (
            "Construction verification does not establish live TP4/DCP1, DCP2, "
            "or DCP4 serving behavior."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside-image", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--engine", default="docker")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.inside_image == bool(args.image):
        parser.error("choose exactly one of --inside-image or --image")
    result = verify_inside_image() if args.inside_image else verify_external(
        args.image, args.engine
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
