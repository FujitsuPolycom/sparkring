"""Verify an opt-in Qwen prefill bundle before installing its import hooks."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys

IMAGE_SOURCES = {
    "/opt/venv/lib/python3.12/site-packages/vllm/models/qwen3_8_flash_next/hyperconnection.py": "adb84c0979921579c12dc8ab6496d9c64d1a3770cbbccd3bf3368bcdcdb3302e",
    "/opt/venv/lib/python3.12/site-packages/b12x/sequence/mtp_feedback/_kernels.py": "be326f7e8767f03ca3e4a1514191c42790b1795e72033f061f196489ba384d47",
}
FILES = {
    "prefill_bootstrap.py",
    "qwen_hc_fusion.py",
    "qwen_mtp_gemm.py",
    "fused_gate_kernel.py",
    "qwen_prefill.pth",
}


def verify(root, expected, filesystem_root=Path("/")):
    if not re.fullmatch("[0-9a-f]{64}", expected):
        raise ValueError("A Qwen prefill manifest SHA-256 is required")
    payload = (root / "manifest.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Qwen prefill manifest identity mismatch")
    record = json.loads(payload)
    if (
        record.get("schema") != "sparkring-qwen-prefill/v1"
        or set(record.get("files", {})) != FILES
        or record.get("image_source_preimages") != IMAGE_SOURCES
    ):
        raise ValueError("Unsupported Qwen prefill bundle contract")
    for name, digest in record["files"].items():
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Qwen prefill bundle file mismatch: " + name)
    for name, digest in IMAGE_SOURCES.items():
        path = filesystem_root / name.lstrip("/")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Qwen prefill image source mismatch: " + name)
    return record


def verify_cache_namespace(expected, environment):
    namespace = "qwen-prefill-" + expected[:12]
    for key, suffix in [
        ("VLLM_CACHE_ROOT", "/vllm"),
        ("TORCHINDUCTOR_CACHE_DIR", "/inductor"),
    ]:
        if not environment.get(key, "").endswith("/" + namespace + suffix):
            raise ValueError(
                key + " must use the source-bound Qwen prefill cache namespace"
            )


def install():
    try:
        expected = os.environ.get("SPARKRING_QWEN_PREFILL_MANIFEST_SHA256", "")
        verify(Path(__file__).resolve().parent, expected)
        verify_cache_namespace(expected, os.environ)
        targets = [
            "vllm.models.qwen3_8_flash_next.hyperconnection",
            "b12x.sequence.mtp_feedback._kernels",
        ]
        if any(name in sys.modules for name in targets):
            raise ValueError("Qwen prefill hooks must load before their target modules")
        os.environ["QWEN_HC_FUSION"] = "1"
        os.environ["QWEN_HC_SOURCE_SHA256"] = IMAGE_SOURCES[
            next(p for p in IMAGE_SOURCES if p.endswith("hyperconnection.py"))
        ]
        os.environ["QWEN_MTP_TORCH_PREFILL"] = "1"
        os.environ["QWEN_MTP_SOURCE_SHA256"] = IMAGE_SOURCES[
            next(p for p in IMAGE_SOURCES if p.endswith("_kernels.py"))
        ]
        importlib.import_module("qwen_hc_fusion")
        importlib.import_module("qwen_mtp_gemm")
    except Exception as error:
        # Python ignores ordinary .pth exceptions; explicit selection must stop startup.
        raise SystemExit("Qwen prefill admission failed: " + str(error)) from error
