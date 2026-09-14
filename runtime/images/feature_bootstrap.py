"""Activate explicitly selected, image-baked SparkRing feature bundles."""

import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parent


def description(root=ROOT):
    record = json.loads((root / "capabilities.json").read_text())
    if record.get("schema") != "sparkring-image-capabilities/v1":
        raise ValueError("Unsupported SparkRing capability inventory")
    return record


def verify_assets(root, feature):
    for relative, expected in feature["files"].items():
        name = PurePosixPath(relative)
        if name.is_absolute() or ".." in name.parts or "\\" in relative:
            raise ValueError("Invalid feature asset path")
        path = root / relative
        if (
            path.is_symlink()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Feature asset differs: " + relative)


def prefill_environment(environment, digest):
    """Derive an idempotent compiler namespace while preserving the site's cache root."""
    namespace = "qwen-prefill-" + digest[:12]
    for key, leaf in [
        ("VLLM_CACHE_ROOT", "vllm"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
    ]:
        current = environment.get(key, "/cache/" + leaf).rstrip("/")
        suffix = "/" + namespace + "/" + leaf
        if not current.endswith(suffix):
            environment[key] = str(PurePosixPath(current) / namespace / leaf)
    environment["SPARKRING_QWEN_PREFILL_MANIFEST_SHA256"] = digest


def install(root=ROOT, environment=None, importer=importlib.import_module):
    environment = os.environ if environment is None else environment
    selected = [
        part.strip()
        for part in environment.get("SPARKRING_FEATURES", "").split(",")
        if part.strip()
    ]
    if not selected:
        return
    record = description(root)
    if len(set(selected)) != len(selected) or any(
        name not in record["features"] for name in selected
    ):
        raise ValueError("Unknown or duplicate SparkRing feature selection")
    # Validate the complete selection before any hook changes imports or environment.
    for name in selected:
        verify_assets(root, record["features"][name])
    for name in selected:
        feature = record["features"][name]
        path = str(root / feature["directory"])
        if path not in sys.path:
            sys.path.insert(0, path)
        if name == "qwen-collectives":
            environment.setdefault("QWEN_DISPATCH_MODE", "both")
            environment.setdefault("QWEN_DISPATCH_AR_BYTES", "20480")
            environment.setdefault("QWEN_DISPATCH_TRACE", "0")
            importer("qwen38_collective_policy")
        elif name == "qwen-prefill":
            prefill_environment(environment, feature["manifest_sha256"])
            importer("prefill_bootstrap").install()
        else:
            raise ValueError("Feature has no activation implementation: " + name)


def install_or_exit():
    try:
        install()
    except Exception as error:
        # Ordinary .pth exceptions are ignored by Python; explicit selection must fail closed.
        raise SystemExit(
            "SparkRing feature activation failed: " + str(error)
        ) from error


if __name__ == "__main__":
    print(json.dumps(description(), indent=2))
