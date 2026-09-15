"""Inspect a pinned LIL foundation without GPU devices, networks or host mounts."""

import ctypes
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform

site = Path("/opt/venv/lib/python3.12/site-packages")
result = {
    "machine": platform.machine(),
    "python": platform.python_version(),
    "packages": {},
    "sources": {},
    "imports": {},
}
for name in ("torch", "vllm", "b12x", "flashinfer-python", "lmcache", "sparkcache"):
    try:
        result["packages"][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        result["packages"][name] = None
for relative in (
    "vllm/models/qwen3_8_flash_next/hyperconnection.py",
    "b12x/sequence/mtp_feedback/_kernels.py",
    "vllm/distributed/device_communicators/b12x_roce_all_reduce.py",
):
    path = site / relative
    result["sources"][relative] = (
        hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    )
for name in ("torch", "vllm", "b12x"):
    try:
        module = importlib.import_module(name)
        result["imports"][name] = {"ok": True}
        if name == "torch":
            result["torch_cuda"] = module.version.cuda
            result["visible_cuda_devices"] = module.cuda.device_count()
    except Exception as error:
        result["imports"][name] = {"ok": False, "error": str(error)}
result["nccl"] = []
for path in sorted({p.resolve() for p in Path("/opt/nccl/lib").glob("libnccl.so*")}):
    raw = path.read_bytes()
    record = {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "preserve_pci_domain_switch": b"NCCL_IB_PRESERVE_PCI_DOMAIN" in raw,
        "subnet_aware_routing_switch": b"NCCL_IB_SUBNET_AWARE_ROUTING" in raw,
    }
    try:
        version = ctypes.c_int()
        library = ctypes.CDLL(str(path))
        record["version_status"] = library.ncclGetVersion(ctypes.byref(version))
        record["version"] = version.value
    except Exception as error:
        record["load_error"] = str(error)
    result["nccl"].append(record)
result["sparkring_paths"] = {
    path: Path(path).exists()
    for path in (
        "/opt/sparkring/sircl/libspark_transport_capi.so",
        "/opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so",
        "/opt/sparkring/receipts/candidate-installed.json",
    )
}
print("SPARKRING_BASE_PROBE=" + json.dumps(result, sort_keys=True))
